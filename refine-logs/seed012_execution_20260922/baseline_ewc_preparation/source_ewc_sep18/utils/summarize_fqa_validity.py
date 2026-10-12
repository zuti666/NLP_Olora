#!/usr/bin/env python3
"""Build empirical EWC-surrogate validity metrics from FQA curves.

The FQA curve measures

    |L(theta + r u) - L(theta) - 0.5 r^2 u^T F u|

for sampled unit directions u.  This script turns those curves into the
empirical counterpart of the note's delta_epsilon radius: the largest sampled
radius whose approximation error remains below a chosen tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EPS = (1e-5, 5e-5, 1e-4, 2e-4, 5e-4)
EPS_DENOM = 1e-12


def _to_float(value: object, default: float = float("nan")) -> float:
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def _to_str(value: object) -> str:
    return "" if value is None else str(value)


def _eps_tag(value: float) -> str:
    text = f"{value:.0e}"
    text = text.replace("+", "").replace("-", "m").replace(".", "p")
    return text


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _setting_from_config(config: str) -> str:
    if not config:
        return ""
    path = Path(config)
    if path.suffix == ".json":
        return path.stem
    return ""


def _setting_from_log(log_path: str) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "imagenet_r_task1_fqa_adam_gam" and idx + 2 < len(parts):
            maybe_setting = parts[idx + 2]
            if maybe_setting in {"adam", "gam_rho_0p05"} or maybe_setting.startswith("gam_"):
                return maybe_setting
    name = path.name
    if name.startswith("adam_"):
        return "adam"
    if name.startswith("gam_rho_"):
        return name.split("_task1", 1)[0]
    return ""


def _setting_from_row(row: Dict[str, str]) -> str:
    return (
        _to_str(row.get("setting")).strip()
        or _setting_from_config(_to_str(row.get("config")))
        or _setting_from_log(_to_str(row.get("log_path")))
    )


def _dedupe_curve_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop duplicate parser rows from top-level and nested log copies."""
    chosen: Dict[Tuple[str, str, str, float], Dict[str, str]] = {}
    for row in rows:
        setting = _setting_from_row(row)
        radius = _to_float(row.get("radius"))
        if not setting or math.isnan(radius):
            continue
        seed = _to_str(row.get("seed")).strip()
        task = _to_str(row.get("task")).strip()
        key = (setting, seed, task, radius)
        row = dict(row)
        row["setting"] = setting

        prev = chosen.get(key)
        if prev is None:
            chosen[key] = row
            continue

        prev_score = _row_quality(prev)
        score = _row_quality(row)
        if score > prev_score:
            chosen[key] = row
    return sorted(
        chosen.values(),
        key=lambda r: (_setting_from_row(r), _to_str(r.get("seed")), _to_str(r.get("task")), _to_float(r.get("radius"))),
    )


def _row_quality(row: Dict[str, str]) -> int:
    score = 0
    if _to_str(row.get("linear_corrected_error_mean")).strip():
        score += 8
    if _to_str(row.get("linear_term_abs_mean")).strip():
        score += 4
    if _to_str(row.get("fqa_base_loss")).strip():
        score += 4
    if _to_str(row.get("zero_radius_bias")).strip():
        score += 2
    if "_seed" in _to_str(row.get("log_path")):
        score += 1
    return score


def _mechanism_setting(row: Dict[str, str]) -> str:
    exp_name = _to_str(row.get("exp_name")).strip()
    if exp_name:
        return exp_name.rstrip("/").split("/")[-1]
    return _setting_from_config(_to_str(row.get("config"))) or _setting_from_log(_to_str(row.get("log_path")))


def _load_mechanism_metrics(result_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    rows = _read_csv(result_dir / "mechanism_metrics.csv")
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        setting = _mechanism_setting(row)
        if not setting:
            continue
        seed = _to_str(row.get("seed")).strip()
        task = _to_str(row.get("mechanism_task", row.get("task", ""))).strip()
        if task in {"0.0"}:
            task = "0"
        key = (setting, seed, task)
        prev = out.get(key)
        if prev is None or _row_quality(row) > _row_quality(prev):
            out[key] = row
    return out


def _trapezoid_auc(points: Sequence[Tuple[float, float]]) -> float:
    valid = sorted((x, y) for x, y in points if not math.isnan(x) and not math.isnan(y))
    if len(valid) < 2:
        return float("nan")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(valid, valid[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    span = valid[-1][0] - valid[0][0]
    return area / span if span > 0 else float("nan")


def _random_first_order_abs(radius: float, grad_norm: float, dim: float) -> float:
    if (
        math.isnan(radius)
        or math.isnan(grad_norm)
        or math.isnan(dim)
        or radius < 0
        or grad_norm < 0
        or dim <= 0
    ):
        return float("nan")
    return radius * grad_norm * math.sqrt(2.0 / (math.pi * dim))


def _largest_radius_under(rows: Sequence[Dict[str, object]], metric: str, eps: float) -> str:
    radius = None
    for row in rows:
        value = _to_float(row.get(metric))
        r = _to_float(row.get("radius"))
        if not math.isnan(value) and not math.isnan(r) and value <= eps:
            radius = r if radius is None else max(radius, r)
    return "" if radius is None else f"{radius:g}"


def _max_abs_loss_delta(row: Dict[str, str]) -> float:
    lo = _to_float(row.get("loss_delta_min"))
    hi = _to_float(row.get("loss_delta_max"))
    vals = [abs(v) for v in (lo, hi) if not math.isnan(v)]
    return max(vals) if vals else float("nan")


def _group_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (_setting_from_row(row), _to_str(row.get("seed")).strip(), _to_str(row.get("task")).strip())
        grouped.setdefault(key, []).append(row)
    for key in list(grouped):
        grouped[key] = sorted(grouped[key], key=lambda r: _to_float(r.get("radius")))
    return grouped


def _diagnostic_rows(
    curve_rows: Sequence[Dict[str, str]],
    mechanism: Dict[Tuple[str, str, str], Dict[str, str]],
    result_name: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in curve_rows:
        setting = _setting_from_row(row)
        seed = _to_str(row.get("seed")).strip()
        task = _to_str(row.get("task")).strip()
        mech = mechanism.get((setting, seed, task), {})
        radius = _to_float(row.get("radius"))
        error_mean = _to_float(row.get("error_mean"))
        error_max = _to_float(row.get("error_max"))
        loss_abs = _to_float(row.get("loss_delta_abs_mean"))
        fisher_quad = _to_float(row.get("fisher_quad_mean"))
        max_abs_delta = _max_abs_loss_delta(row)
        denom = radius * radius if radius > 0 else float("nan")
        loss_den = loss_abs if not math.isnan(loss_abs) and abs(loss_abs) > EPS_DENOM else float("nan")
        endpoint_grad_norm = _to_float(mech.get("endpoint_grad_norm"))
        endpoint_param_count = _to_float(mech.get("endpoint_param_count"))
        endpoint_random_abs = _random_first_order_abs(radius, endpoint_grad_norm, endpoint_param_count)
        endpoint_stationarity = (
            endpoint_grad_norm * radius
            if not math.isnan(endpoint_grad_norm) and not math.isnan(radius)
            else float("nan")
        )
        out.append({
            "result_name": result_name,
            "setting": setting,
            "optimizer_family": row.get("optimizer_family", ""),
            "gam_train_rho": row.get("gam_train_rho", ""),
            "seed": seed,
            "task": task,
            "radius": row.get("radius", ""),
            "samples": row.get("samples", ""),
            "paired_base_loss": row.get("fqa_base_loss") or row.get("base_loss", ""),
            "loss_delta_abs_mean": row.get("loss_delta_abs_mean", ""),
            "loss_delta_max_abs": "" if math.isnan(max_abs_delta) else max_abs_delta,
            "fisher_quad_mean": row.get("fisher_quad_mean", ""),
            "fqa_error_mean": row.get("error_mean", ""),
            "fqa_error_max": row.get("error_max", ""),
            "relative_error_mean": row.get("relative_error_mean", ""),
            "fqa_grad_norm": row.get("fqa_grad_norm", ""),
            "linear_available": row.get("linear_available", ""),
            "linear_term_abs_mean": row.get("linear_term_abs_mean", ""),
            "linear_term_abs_max": row.get("linear_term_abs_max", ""),
            "linear_term_abs_over_r2_mean": row.get("linear_term_abs_over_r2_mean", ""),
            "linear_term_abs_to_loss_abs_ratio": row.get("linear_term_abs_to_loss_abs_ratio", ""),
            "stationarity_bound": row.get("stationarity_bound", ""),
            "stationarity_bound_over_r2": row.get("stationarity_bound_over_r2", ""),
            "random_linear_abs_theory": row.get("random_linear_abs_theory", ""),
            "random_linear_abs_theory_over_r2": row.get("random_linear_abs_theory_over_r2", ""),
            "endpoint_param_count": mech.get("endpoint_param_count", ""),
            "endpoint_stationarity_bound": "" if math.isnan(endpoint_stationarity) else endpoint_stationarity,
            "endpoint_stationarity_bound_over_r2": (
                "" if math.isnan(endpoint_stationarity) or math.isnan(denom) else endpoint_stationarity / denom
            ),
            "endpoint_random_linear_abs_theory": "" if math.isnan(endpoint_random_abs) else endpoint_random_abs,
            "endpoint_random_linear_abs_theory_over_r2": (
                "" if math.isnan(endpoint_random_abs) or math.isnan(denom) else endpoint_random_abs / denom
            ),
            "endpoint_random_linear_abs_theory_to_loss_abs_ratio": (
                "" if math.isnan(endpoint_random_abs) or math.isnan(loss_den) else endpoint_random_abs / loss_den
            ),
            "linear_corrected_delta_abs_mean": row.get("linear_corrected_delta_abs_mean", ""),
            "linear_corrected_abs_to_loss_abs_ratio": row.get("linear_corrected_abs_to_loss_abs_ratio", ""),
            "linear_corrected_error_mean": row.get("linear_corrected_error_mean", ""),
            "linear_corrected_error_max": row.get("linear_corrected_error_max", ""),
            "linear_corrected_relative_error_mean": row.get("linear_corrected_relative_error_mean", ""),
            "fisher_to_loss_abs_ratio": (
                "" if math.isnan(loss_den) or math.isnan(fisher_quad) else fisher_quad / loss_den
            ),
            "error_mean_over_r2": "" if math.isnan(denom) or math.isnan(error_mean) else error_mean / denom,
            "error_max_over_r2": "" if math.isnan(denom) or math.isnan(error_max) else error_max / denom,
            "loss_abs_mean_over_r2": "" if math.isnan(denom) or math.isnan(loss_abs) else loss_abs / denom,
            "fisher_quad_mean_over_r2": "" if math.isnan(denom) or math.isnan(fisher_quad) else fisher_quad / denom,
            "zero_radius_bias": row.get("zero_radius_bias", ""),
            "log_path": row.get("log_path", ""),
            "config": row.get("config", ""),
        })
    return out


def _summary_rows(
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]],
    mechanism: Dict[Tuple[str, str, str], Dict[str, str]],
    result_name: str,
    eps_values: Sequence[float],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        setting, seed, task = key
        mech = mechanism.get(key, {})
        first = rows[0]
        nonzero = [row for row in rows if _to_float(row.get("radius")) > 0]
        last = nonzero[-1] if nonzero else rows[-1]
        zero = next((row for row in rows if abs(_to_float(row.get("radius"))) < 1e-12), None)
        endpoint_grad_norm = _to_float(mech.get("endpoint_grad_norm"))
        endpoint_param_count = _to_float(mech.get("endpoint_param_count"))

        zero_bias = _to_float((zero or {}).get("zero_radius_bias"), default=float("nan"))
        if math.isnan(zero_bias) and zero is not None:
            zero_bias = _to_float(zero.get("loss_delta_mean"))
        zero_ok = (not math.isnan(zero_bias)) and abs(zero_bias) <= 1e-8

        radii = [_to_float(row.get("radius")) for row in rows]
        row_out: Dict[str, object] = {
            "result_name": result_name,
            "setting": setting,
            "optimizer_family": first.get("optimizer_family", ""),
            "gam_train_rho": first.get("gam_train_rho", ""),
            "seed": seed,
            "task": task,
            "r_min": min(r for r in radii if not math.isnan(r)) if radii else "",
            "r_max_observed": max(r for r in radii if not math.isnan(r)) if radii else "",
            "n_radii": len(rows),
            "samples": first.get("samples", ""),
            "base_loss": first.get("base_loss", ""),
            "fqa_base_loss": first.get("fqa_base_loss", ""),
            "fqa_grad_base_loss": first.get("fqa_grad_base_loss", ""),
            "fqa_grad_norm": first.get("fqa_grad_norm", ""),
            "paired_eval_ok": bool(zero_ok and _to_str(first.get("fqa_base_loss")).strip()),
            "zero_radius_bias": "" if math.isnan(zero_bias) else zero_bias,
            "zero_radius_abs_bias": "" if math.isnan(zero_bias) else abs(zero_bias),
            "fqa_error_mean_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("error_mean"))) for row in rows
            ]),
            "fqa_error_max_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("error_max"))) for row in rows
            ]),
            "loss_abs_mean_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("loss_delta_abs_mean"))) for row in rows
            ]),
            "fisher_quad_mean_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("fisher_quad_mean"))) for row in rows
            ]),
            "linear_term_abs_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("linear_term_abs_mean"))) for row in rows
            ]),
            "linear_corrected_delta_abs_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("linear_corrected_delta_abs_mean"))) for row in rows
            ]),
            "linear_corrected_error_mean_auc": _trapezoid_auc([
                (_to_float(row.get("radius")), _to_float(row.get("linear_corrected_error_mean"))) for row in rows
            ]),
            "endpoint_random_linear_abs_theory_auc": _trapezoid_auc([
                (
                    _to_float(row.get("radius")),
                    _random_first_order_abs(_to_float(row.get("radius")), endpoint_grad_norm, endpoint_param_count),
                )
                for row in rows
            ]),
            "fqa_error_mean_at_rmax": last.get("error_mean", ""),
            "fqa_error_max_at_rmax": last.get("error_max", ""),
            "loss_abs_mean_at_rmax": last.get("loss_delta_abs_mean", ""),
            "loss_delta_max_abs_at_rmax": _max_abs_loss_delta(last),
            "fisher_quad_mean_at_rmax": last.get("fisher_quad_mean", ""),
            "linear_term_abs_mean_at_rmax": last.get("linear_term_abs_mean", ""),
            "linear_term_abs_to_loss_abs_ratio_at_rmax": last.get("linear_term_abs_to_loss_abs_ratio", ""),
            "stationarity_bound_at_rmax": last.get("stationarity_bound", ""),
            "random_linear_abs_theory_at_rmax": last.get("random_linear_abs_theory", ""),
            "linear_corrected_delta_abs_mean_at_rmax": last.get("linear_corrected_delta_abs_mean", ""),
            "linear_corrected_abs_to_loss_abs_ratio_at_rmax": last.get("linear_corrected_abs_to_loss_abs_ratio", ""),
            "linear_corrected_error_mean_at_rmax": last.get("linear_corrected_error_mean", ""),
            "linear_corrected_error_max_at_rmax": last.get("linear_corrected_error_max", ""),
            "fisher_to_loss_abs_ratio_at_rmax": "",
            "error_mean_over_r2_at_rmax": "",
            "loss_abs_mean_over_r2_at_rmax": "",
            "fisher_quad_mean_over_r2_at_rmax": "",
            "endpoint_grad_norm": mech.get("endpoint_grad_norm", ""),
            "endpoint_first_order_sharpness": mech.get("endpoint_first_order_sharpness", ""),
            "endpoint_sh0_max": mech.get("endpoint_sh0_max", ""),
            "endpoint_param_count": mech.get("endpoint_param_count", ""),
            "endpoint_fisher_total_mass": mech.get("endpoint_fisher_total_mass", ""),
            "endpoint_fisher_mean": mech.get("endpoint_fisher_mean", ""),
            "endpoint_fisher_max": mech.get("endpoint_fisher_max", ""),
            "log_path": first.get("log_path", ""),
            "config": first.get("config", ""),
        }

        r_last = _to_float(last.get("radius"))
        r2_last = r_last * r_last if r_last > 0 else float("nan")
        loss_last = _to_float(last.get("loss_delta_abs_mean"))
        fisher_last = _to_float(last.get("fisher_quad_mean"))
        err_last = _to_float(last.get("error_mean"))
        endpoint_random_last = _random_first_order_abs(r_last, endpoint_grad_norm, endpoint_param_count)
        if not math.isnan(loss_last) and abs(loss_last) > EPS_DENOM and not math.isnan(fisher_last):
            row_out["fisher_to_loss_abs_ratio_at_rmax"] = fisher_last / loss_last
        if not math.isnan(endpoint_grad_norm) and not math.isnan(r_last):
            row_out["endpoint_stationarity_bound_at_rmax"] = endpoint_grad_norm * r_last
        if not math.isnan(endpoint_random_last):
            row_out["endpoint_random_linear_abs_theory_at_rmax"] = endpoint_random_last
            if not math.isnan(loss_last) and abs(loss_last) > EPS_DENOM:
                row_out["endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax"] = (
                    endpoint_random_last / loss_last
                )
        if not math.isnan(r2_last):
            if not math.isnan(err_last):
                row_out["error_mean_over_r2_at_rmax"] = err_last / r2_last
            if not math.isnan(loss_last):
                row_out["loss_abs_mean_over_r2_at_rmax"] = loss_last / r2_last
            if not math.isnan(fisher_last):
                row_out["fisher_quad_mean_over_r2_at_rmax"] = fisher_last / r2_last

        metric_rows: List[Dict[str, object]] = []
        for source in rows:
            item = dict(source)
            item["loss_delta_max_abs"] = _max_abs_loss_delta(source)
            metric_rows.append(item)

        for eps in eps_values:
            tag = _eps_tag(eps)
            row_out[f"valid_radius_mean_eps_{tag}"] = _largest_radius_under(metric_rows, "error_mean", eps)
            row_out[f"valid_radius_max_eps_{tag}"] = _largest_radius_under(metric_rows, "error_max", eps)
            row_out[f"linear_corrected_valid_radius_mean_eps_{tag}"] = _largest_radius_under(
                metric_rows, "linear_corrected_error_mean", eps
            )
            row_out[f"linear_corrected_valid_radius_max_eps_{tag}"] = _largest_radius_under(
                metric_rows, "linear_corrected_error_max", eps
            )
            row_out[f"loss_stability_radius_mean_eps_{tag}"] = _largest_radius_under(
                metric_rows, "loss_delta_abs_mean", eps
            )
            row_out[f"loss_stability_radius_max_eps_{tag}"] = _largest_radius_under(
                metric_rows, "loss_delta_max_abs", eps
            )
        out.append(row_out)
    return out


def _comparison_rows(summary_rows: Sequence[Dict[str, object]], eps_values: Sequence[float]) -> List[Dict[str, object]]:
    by_seed_task: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in summary_rows:
        key = (_to_str(row.get("result_name")), _to_str(row.get("seed")), _to_str(row.get("task")))
        by_seed_task.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (_result_name, seed, task), rows in sorted(by_seed_task.items()):
        baselines = [row for row in rows if _to_str(row.get("setting")) == "adam"]
        gams = [row for row in rows if _to_str(row.get("setting")).startswith("gam")]
        if not baselines:
            continue
        base = baselines[0]
        for gam in gams:
            item: Dict[str, object] = {
                "result_name": gam.get("result_name", ""),
                "baseline_setting": base.get("setting", ""),
                "gam_setting": gam.get("setting", ""),
                "seed": seed,
                "task": task,
                "delta_accuracy_hint": "",
                "fqa_error_mean_auc_ratio_gam_over_adam": _ratio(
                    gam.get("fqa_error_mean_auc"), base.get("fqa_error_mean_auc")
                ),
                "fqa_error_max_auc_ratio_gam_over_adam": _ratio(
                    gam.get("fqa_error_max_auc"), base.get("fqa_error_max_auc")
                ),
                "loss_abs_mean_auc_ratio_gam_over_adam": _ratio(
                    gam.get("loss_abs_mean_auc"), base.get("loss_abs_mean_auc")
                ),
                "fisher_quad_mean_auc_ratio_gam_over_adam": _ratio(
                    gam.get("fisher_quad_mean_auc"), base.get("fisher_quad_mean_auc")
                ),
                "linear_term_abs_auc_ratio_gam_over_adam": _ratio(
                    gam.get("linear_term_abs_auc"), base.get("linear_term_abs_auc")
                ),
                "linear_corrected_error_mean_auc_ratio_gam_over_adam": _ratio(
                    gam.get("linear_corrected_error_mean_auc"), base.get("linear_corrected_error_mean_auc")
                ),
                "endpoint_random_linear_abs_theory_auc_ratio_gam_over_adam": _ratio(
                    gam.get("endpoint_random_linear_abs_theory_auc"),
                    base.get("endpoint_random_linear_abs_theory_auc"),
                ),
                "endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax_ratio_gam_over_adam": _ratio(
                    gam.get("endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax"),
                    base.get("endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax"),
                ),
                "endpoint_fisher_total_mass_ratio_gam_over_adam": _ratio(
                    gam.get("endpoint_fisher_total_mass"), base.get("endpoint_fisher_total_mass")
                ),
                "endpoint_sh0_max_ratio_gam_over_adam": _ratio(
                    gam.get("endpoint_sh0_max"), base.get("endpoint_sh0_max")
                ),
                "endpoint_grad_norm_ratio_gam_over_adam": _ratio(
                    gam.get("endpoint_grad_norm"), base.get("endpoint_grad_norm")
                ),
            }
            for eps in eps_values:
                tag = _eps_tag(eps)
                for prefix in (
                    "valid_radius_mean",
                    "valid_radius_max",
                    "linear_corrected_valid_radius_mean",
                    "linear_corrected_valid_radius_max",
                    "loss_stability_radius_mean",
                    "loss_stability_radius_max",
                ):
                    name = f"{prefix}_eps_{tag}"
                    item[f"{name}_adam"] = base.get(name, "")
                    item[f"{name}_gam"] = gam.get(name, "")
                    item[f"{name}_delta_gam_minus_adam"] = _diff(gam.get(name), base.get(name))
            out.append(item)
    return out


def _ratio(num: object, den: object) -> str:
    n = _to_float(num)
    d = _to_float(den)
    if math.isnan(n) or math.isnan(d) or abs(d) <= EPS_DENOM:
        return ""
    return f"{n / d:.8g}"


def _diff(lhs: object, rhs: object) -> str:
    a = _to_float(lhs)
    b = _to_float(rhs)
    if math.isnan(a) or math.isnan(b):
        return ""
    return f"{a - b:.8g}"


def _summary_fieldnames(eps_values: Sequence[float]) -> List[str]:
    fields = [
        "result_name", "setting", "optimizer_family", "gam_train_rho", "seed", "task",
        "r_min", "r_max_observed", "n_radii", "samples", "base_loss", "fqa_base_loss",
        "fqa_grad_base_loss", "fqa_grad_norm",
        "paired_eval_ok", "zero_radius_bias", "zero_radius_abs_bias",
        "fqa_error_mean_auc", "fqa_error_max_auc", "loss_abs_mean_auc",
        "fisher_quad_mean_auc", "linear_term_abs_auc",
        "linear_corrected_delta_abs_auc", "linear_corrected_error_mean_auc",
        "endpoint_random_linear_abs_theory_auc",
        "fqa_error_mean_at_rmax", "fqa_error_max_at_rmax",
        "loss_abs_mean_at_rmax", "loss_delta_max_abs_at_rmax", "fisher_quad_mean_at_rmax",
        "linear_term_abs_mean_at_rmax", "linear_term_abs_to_loss_abs_ratio_at_rmax",
        "stationarity_bound_at_rmax", "random_linear_abs_theory_at_rmax",
        "endpoint_stationarity_bound_at_rmax",
        "endpoint_random_linear_abs_theory_at_rmax",
        "endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax",
        "linear_corrected_delta_abs_mean_at_rmax",
        "linear_corrected_abs_to_loss_abs_ratio_at_rmax",
        "linear_corrected_error_mean_at_rmax", "linear_corrected_error_max_at_rmax",
        "fisher_to_loss_abs_ratio_at_rmax", "error_mean_over_r2_at_rmax",
        "loss_abs_mean_over_r2_at_rmax", "fisher_quad_mean_over_r2_at_rmax",
        "endpoint_grad_norm", "endpoint_first_order_sharpness", "endpoint_sh0_max",
        "endpoint_param_count", "endpoint_fisher_total_mass", "endpoint_fisher_mean",
        "endpoint_fisher_max",
    ]
    for eps in eps_values:
        tag = _eps_tag(eps)
        fields.extend([
            f"valid_radius_mean_eps_{tag}",
            f"valid_radius_max_eps_{tag}",
            f"linear_corrected_valid_radius_mean_eps_{tag}",
            f"linear_corrected_valid_radius_max_eps_{tag}",
            f"loss_stability_radius_mean_eps_{tag}",
            f"loss_stability_radius_max_eps_{tag}",
        ])
    fields.extend(["log_path", "config"])
    return fields


def _diagnostic_fieldnames() -> List[str]:
    return [
        "result_name", "setting", "optimizer_family", "gam_train_rho", "seed", "task",
        "radius", "samples", "paired_base_loss", "loss_delta_abs_mean",
        "loss_delta_max_abs", "fisher_quad_mean", "fqa_error_mean", "fqa_error_max",
        "relative_error_mean", "fisher_to_loss_abs_ratio", "error_mean_over_r2",
        "error_max_over_r2", "loss_abs_mean_over_r2", "fisher_quad_mean_over_r2",
        "fqa_grad_norm", "linear_available", "linear_term_abs_mean",
        "linear_term_abs_max", "linear_term_abs_over_r2_mean",
        "linear_term_abs_to_loss_abs_ratio", "stationarity_bound",
        "stationarity_bound_over_r2", "random_linear_abs_theory",
        "random_linear_abs_theory_over_r2", "endpoint_param_count",
        "endpoint_stationarity_bound", "endpoint_stationarity_bound_over_r2",
        "endpoint_random_linear_abs_theory",
        "endpoint_random_linear_abs_theory_over_r2",
        "endpoint_random_linear_abs_theory_to_loss_abs_ratio",
        "linear_corrected_delta_abs_mean",
        "linear_corrected_abs_to_loss_abs_ratio", "linear_corrected_error_mean",
        "linear_corrected_error_max", "linear_corrected_relative_error_mean",
        "zero_radius_bias", "log_path", "config",
    ]


def _comparison_fieldnames(eps_values: Sequence[float]) -> List[str]:
    fields = [
        "result_name", "baseline_setting", "gam_setting", "seed", "task",
        "delta_accuracy_hint", "fqa_error_mean_auc_ratio_gam_over_adam",
        "fqa_error_max_auc_ratio_gam_over_adam",
        "loss_abs_mean_auc_ratio_gam_over_adam",
        "fisher_quad_mean_auc_ratio_gam_over_adam",
        "linear_term_abs_auc_ratio_gam_over_adam",
        "linear_corrected_error_mean_auc_ratio_gam_over_adam",
        "endpoint_random_linear_abs_theory_auc_ratio_gam_over_adam",
        "endpoint_random_linear_abs_theory_to_loss_abs_ratio_at_rmax_ratio_gam_over_adam",
        "endpoint_fisher_total_mass_ratio_gam_over_adam",
        "endpoint_sh0_max_ratio_gam_over_adam",
        "endpoint_grad_norm_ratio_gam_over_adam",
    ]
    for eps in eps_values:
        tag = _eps_tag(eps)
        for prefix in (
            "valid_radius_mean",
            "valid_radius_max",
            "linear_corrected_valid_radius_mean",
            "linear_corrected_valid_radius_max",
            "loss_stability_radius_mean",
            "loss_stability_radius_max",
        ):
            name = f"{prefix}_eps_{tag}"
            fields.extend([f"{name}_adam", f"{name}_gam", f"{name}_delta_gam_minus_adam"])
    return fields


def _parse_fqa_partial_stem(stem: str) -> Tuple[str, str, str]:
    """Parse '{setting}_eval_task{task}_{prefix}_fqa_partial' stem.

    Returns (setting, task, prefix); falls back to (stem, "", "") if unrecognized.
    """
    m = re.match(r"^(.+?)_eval_task(\w+)_(.+)_fqa_partial$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return stem, "", ""


def _parse_s1c_fqa_partial_stem(stem: str) -> Tuple[str, str, str]:
    """Parse 's1c_{optimizer}_..._endpoint_{split}_fqa_partial' stem.

    Returns (optimizer, split, experiment_tag); falls back to ("", "", stem).
    """
    m = re.match(r"^s1c_(sgd|gam)_(.+)_endpoint_(train|test)_fqa_partial$", stem)
    if m:
        return m.group(1), m.group(3), m.group(2)
    optimizer = ""
    if re.search(r"(?:^|_)sgd(?:_|$)", stem):
        optimizer = "sgd"
    elif re.search(r"(?:^|_)gam(?:_|$)", stem):
        optimizer = "gam"
    split = ""
    if "_endpoint_train" in stem:
        split = "train"
    elif "_endpoint_test" in stem:
        split = "test"
    return optimizer, split, stem


def _jsonl_mean_rows(
    jsonl_path: Path,
    setting: str,
    optimizer: str = "",
    split: str = "",
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Read one *_fqa_partial.json JSONL file; keep only the 'mean' aggregates.

    Returns (rows, mean_fieldnames).  Each row contains metadata fields plus all
    keys from the 'mean' dict.  The raw 'per_u' arrays are discarded.
    """
    rows: List[Dict[str, object]] = []
    mean_fields: List[str] = []
    mean_seen: set = set()

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mean = obj.get("mean") or {}
            row: Dict[str, object] = {
                "setting": setting,
                "optimizer": optimizer,
                "split": split,
                "radius": obj.get("radius", ""),
                "radius_idx": obj.get("radius_idx", ""),
                "radius_count": obj.get("radius_count", ""),
                "K": obj.get("K", ""),
                "main_prefix": obj.get("main_prefix", ""),
            }
            row.update(mean)
            # collect mean key order on first encounter
            for k in mean:
                if k not in mean_seen:
                    mean_fields.append(k)
                    mean_seen.add(k)
            rows.append(row)

    return rows, mean_fields


def convert_fqa_partial_dir(
    partial_dir: Path, out_csv: Optional[Path] = None
) -> int:
    """Convert all *_fqa_partial.json files in partial_dir to a single CSV.

    Supports both the legacy '{setting}_eval_task{task}_{prefix}_fqa_partial.json'
    format and the s1c format 's1c_{optimizer}_..._endpoint_{split}_fqa_partial.json'.
    Only the per-radius 'mean' aggregates are written; per_u raw arrays are dropped.

    Args:
        partial_dir: directory containing the JSONL files.
        out_csv:     output CSV path; defaults to partial_dir/fqa_partial_means.csv.

    Returns total number of data rows written.
    """
    partial_dir = partial_dir.resolve()
    json_files = sorted(partial_dir.glob("*_fqa_partial.json"))
    if not json_files:
        return 0

    all_rows: List[Dict[str, object]] = []
    extra_fields: List[str] = []
    extra_seen: set = set()

    for jf in json_files:
        stem = jf.stem
        optimizer, split, experiment = _parse_s1c_fqa_partial_stem(stem)
        if optimizer or split:
            setting = experiment
        else:
            setting = _parse_fqa_partial_stem(stem)[0]
        rows, mean_fields = _jsonl_mean_rows(jf, setting, optimizer=optimizer, split=split)
        all_rows.extend(rows)
        for f in mean_fields:
            if f not in extra_seen:
                extra_fields.append(f)
                extra_seen.add(f)

    if out_csv is None:
        out_csv = partial_dir / "fqa_partial_means.csv"

    meta = ["optimizer", "split", "setting", "radius", "radius_idx", "radius_count", "K", "main_prefix"]
    fieldnames = meta + extra_fields
    _write_csv(out_csv, all_rows, fieldnames)
    return len(all_rows)


def summarize_result_dir(result_dir: Path, eps_values: Sequence[float]) -> Tuple[int, int, int]:
    result_dir = result_dir.resolve()
    curve_path = result_dir / "fqa_curve.csv"
    curve_rows = _dedupe_curve_rows(_read_csv(curve_path))
    mechanism = _load_mechanism_metrics(result_dir)
    grouped = _group_rows(curve_rows)
    result_name = result_dir.name

    diagnostics = _diagnostic_rows(curve_rows, mechanism, result_name)
    summaries = _summary_rows(grouped, mechanism, result_name, eps_values)
    comparisons = _comparison_rows(summaries, eps_values)

    _write_csv(result_dir / "fqa_radius_diagnostics.csv", diagnostics, _diagnostic_fieldnames())
    _write_csv(result_dir / "fqa_validity_radius.csv", summaries, _summary_fieldnames(eps_values))
    _write_csv(result_dir / "fqa_validity_comparison.csv", comparisons, _comparison_fieldnames(eps_values))
    return len(diagnostics), len(summaries), len(comparisons)


def parse_eps(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values or list(DEFAULT_EPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        help="Result directories containing fqa_curve.csv.",
    )
    parser.add_argument(
        "--eps",
        default=",".join(f"{x:g}" for x in DEFAULT_EPS),
        help="Comma-separated error/loss tolerances for radius metrics.",
    )
    parser.add_argument(
        "--partial-dir",
        type=Path,
        default=None,
        help="Directory containing *_fqa_partial.json JSONL files to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (used with --partial-dir; defaults to <partial-dir>/fqa_partial_means.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.partial_dir is not None:
        n = convert_fqa_partial_dir(args.partial_dir, out_csv=args.output)
        out = args.output or (args.partial_dir.resolve() / "fqa_partial_means.csv")
        print(f"[OK] {args.partial_dir}: wrote {n} row(s) to {out}")
        return

    eps_values = parse_eps(args.eps)
    for result_dir in args.result_dirs:
        diagnostics, summaries, comparisons = summarize_result_dir(result_dir, eps_values)
        print(
            f"[OK] {result_dir}: diagnostics={diagnostics} "
            f"summary={summaries} comparison={comparisons}"
        )


if __name__ == "__main__":
    main()
