#!/usr/bin/env python3
"""Summarize ImageNet-A full-model EWC lambda sweep metrics.

The trainer writes one ``*_cl_metrics.json`` per run.  This utility collects
those JSON files and exports:

* ``lambda_sweep_runs.csv``: one row per run, with final accuracy/forgetting
  and compact mechanism/failure indicators.
* ``lambda_sweep_task_metrics.csv``: one row per task-level mechanism record.
* ``lambda_sweep_epoch_dynamics.csv``: one row per epoch-level dynamics record.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KNOWN_DATASETS = (
    "flowers",
    "pets",
    "imageneta",
    "imagenetr",
    "domainnet",
    "cifar100",
    "cifar224",
)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _nanmean(values: Iterable[Any]) -> float:
    xs = [_to_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _nanmax(values: Iterable[Any]) -> float:
    xs = [_to_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return float(max(xs)) if xs else float("nan")


def _last_finite(values: Iterable[Any]) -> float:
    out = float("nan")
    for value in values:
        x = _to_float(value)
        if math.isfinite(x):
            out = x
    return out


def _num_from_prefix(text: str) -> Optional[float]:
    patterns = (
        r"(?:^|[_-])lam(?:bda)?([0-9]+(?:p[0-9]+)?(?:e[+-]?[0-9]+)?)",
        r"(?:^|[_-])lambda(?:=|_)?([0-9]+(?:p[0-9]+)?(?:e[+-]?[0-9]+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return _to_float(m.group(1).replace("p", "."))
    return None


def discover_metrics(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        raw = str(Path(item).expanduser())
        matches = glob.glob(raw, recursive=True) if glob.has_magic(raw) else [raw]
        for match in matches:
            path = Path(match).expanduser()
            if path.is_dir():
                out.extend(path.rglob("*_cl_metrics.json"))
            elif path.is_file() and path.name.endswith("_cl_metrics.json"):
                out.append(path)
    seen = set()
    uniq = []
    for path in sorted(p.resolve() for p in out):
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def infer_path_meta(path: Path) -> Dict[str, Any]:
    parts = list(path.parts)
    dataset = ""
    seed = ""
    for name in KNOWN_DATASETS:
        if name in parts:
            idx = parts.index(name)
            dataset = name
            if idx + 1 < len(parts):
                seed = parts[idx + 1]
            break
    method = ""
    optimizer_tag = ""
    for root_name in ("logs_inc", "logs_inc_lora"):
        if root_name in parts:
            idx = parts.index(root_name)
            if idx + 1 < len(parts):
                method = parts[idx + 1]
            if idx + 2 < len(parts):
                optimizer_tag = parts[idx + 2]
            break
    run_prefix = path.name
    suffix = "_cl_metrics.json"
    if run_prefix.endswith(suffix):
        run_prefix = run_prefix[: -len(suffix)]
    if "_vit_" in run_prefix:
        run_prefix = run_prefix.split("_vit_", 1)[0]
    return {
        "metrics_path": str(path),
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "optimizer_tag": optimizer_tag,
        "run_prefix": run_prefix,
    }


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_step_items(section: Dict[str, Any]):
    steps = section.get("steps", {}) if isinstance(section, dict) else {}
    def key_fn(item):
        key, _ = item
        try:
            return int(key)
        except Exception:
            return 10**9
    return sorted(steps.items(), key=key_fn)


def collect_task_rows(payload: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for step, metrics in sorted_step_items(payload.get("mechanism", {})):
        if not isinstance(metrics, dict):
            continue
        row = dict(meta)
        row.update(metrics)
        row["step"] = int(step)
        row["mechanism_task"] = metrics.get("task", step)
        rows.append(row)
    return rows


def collect_epoch_rows(payload: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for step, metrics in sorted_step_items(payload.get("epoch_dynamics", {})):
        if not isinstance(metrics, dict):
            continue
        for epoch_row in metrics.get("epochs", []) or []:
            if not isinstance(epoch_row, dict):
                continue
            row = dict(meta)
            row.update(epoch_row)
            row["step"] = int(step)
            row.setdefault("mechanism_task", epoch_row.get("task", step))
            rows.append(row)
    return rows


def build_run_row(
    payload: Dict[str, Any],
    meta: Dict[str, Any],
    task_rows: List[Dict[str, Any]],
    epoch_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cnn = payload.get("cnn", {}) if isinstance(payload.get("cnn", {}), dict) else {}
    final = cnn.get("final", {}) if isinstance(cnn.get("final", {}), dict) else {}
    ca = final.get("CA", final.get("vectors", {}).get("CA", []))
    if not isinstance(ca, list):
        ca = []

    matrix = cnn.get("matrices", {}).get("final", [])
    final_task_accs = []
    if isinstance(matrix, list) and matrix:
        last = matrix[-1]
        if isinstance(last, list):
            final_task_accs = last

    lambda_candidates = [
        *(row.get("lambda_ewc") for row in epoch_rows),
        *(row.get("lambda_ewc") for row in task_rows),
    ]
    lambda_ewc = _last_finite(lambda_candidates)
    if not math.isfinite(lambda_ewc):
        parsed = _num_from_prefix(str(meta.get("run_prefix", "")))
        lambda_ewc = float(parsed) if parsed is not None else float("nan")

    losses = [_to_float(row.get("loss_total")) for row in epoch_rows]
    losses_finite = [x for x in losses if math.isfinite(x)]
    first_loss = losses_finite[0] if losses_finite else float("nan")
    max_loss = max(losses_finite) if losses_finite else float("nan")
    nonfinite_loss = any(
        row.get("loss_total_finite") is False
        or (row.get("loss_total") not in (None, "") and not _is_finite(row.get("loss_total")))
        for row in epoch_rows
    )

    row = dict(meta)
    row.update({
        "lambda_ewc": lambda_ewc,
        "final_avg_acc": _to_float(final.get("Acc", final.get("FAA"))),
        "final_forget": _to_float(final.get("Forget_avg")),
        "final_current_acc": _last_finite(ca),
        "mean_current_acc": _nanmean(ca),
        "final_task_acc_mean": _nanmean(final_task_accs),
        "num_tasks": len(ca) if ca else len(final_task_accs),
        "records_task_mechanism": int(bool(task_rows)),
        "records_epoch_dynamics": int(bool(epoch_rows)),
        "task_metric_count": len(task_rows),
        "epoch_metric_count": len(epoch_rows),
        "mean_DF_task": _nanmean(row.get("fisher_weighted_drift") for row in task_rows),
        "final_DF_task": _last_finite(row.get("fisher_weighted_drift") for row in task_rows),
        "max_DF_epoch": _nanmax(row.get("fisher_weighted_drift") for row in epoch_rows),
        "mean_DF_epoch": _nanmean(row.get("fisher_weighted_drift") for row in epoch_rows),
        "mean_D2_task": _nanmean(row.get("delta_norm") for row in task_rows),
        "final_D2_task": _last_finite(row.get("delta_norm") for row in task_rows),
        "max_D2_epoch": _nanmax(row.get("delta_norm_sq", row.get("delta_norm")) for row in epoch_rows),
        "max_r_EWC_task": _nanmax(row.get("train_ewc_ce_ratio_mean") for row in task_rows),
        "mean_r_EWC_task": _nanmean(row.get("train_ewc_ce_ratio_mean") for row in task_rows),
        "max_r_EWC_epoch": _nanmax(row.get("ewc_ce_ratio") for row in epoch_rows),
        "mean_r_EWC_epoch": _nanmean(row.get("ewc_ce_ratio") for row in epoch_rows),
        "first_loss_total": first_loss,
        "max_loss_total": max_loss,
        "nonfinite_loss": int(nonfinite_loss),
        "loss_explosion": 0,
        "ewc_dominated": 0,
        "underfit_vs_lambda0": 0,
        "failure": "no",
    })
    return row


def add_failure_labels(
    run_rows: List[Dict[str, Any]],
    *,
    loss_explosion_factor: float,
    ratio_threshold: float,
    underfit_fraction: float,
) -> None:
    baselines: Dict[str, float] = {}
    for row in run_rows:
        lam = _to_float(row.get("lambda_ewc"))
        if math.isfinite(lam) and abs(lam) < 1e-12:
            key = f"{row.get('dataset','')}::{row.get('seed','')}"
            baselines[key] = _to_float(row.get("mean_current_acc"))

    for row in run_rows:
        first_loss = _to_float(row.get("first_loss_total"))
        max_loss = _to_float(row.get("max_loss_total"))
        loss_explosion = (
            math.isfinite(first_loss)
            and first_loss > 0
            and math.isfinite(max_loss)
            and max_loss > float(loss_explosion_factor) * first_loss
        )
        ewc_dominated = _to_float(row.get("max_r_EWC_epoch")) > float(ratio_threshold)

        key = f"{row.get('dataset','')}::{row.get('seed','')}"
        baseline_current = baselines.get(key, float("nan"))
        mean_current = _to_float(row.get("mean_current_acc"))
        underfit = (
            math.isfinite(baseline_current)
            and baseline_current > 0
            and math.isfinite(mean_current)
            and mean_current < float(underfit_fraction) * baseline_current
        )

        row["loss_explosion"] = int(loss_explosion)
        row["ewc_dominated"] = int(ewc_dominated)
        row["underfit_vs_lambda0"] = int(underfit)
        if int(row.get("nonfinite_loss", 0)) or loss_explosion:
            row["failure"] = "optimization_collapse"
        elif underfit:
            row["failure"] = "underfit_vs_lambda0"
        elif ewc_dominated:
            row["failure"] = "ewc_dominated"
        else:
            row["failure"] = "no"


def union_fieldnames(rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> List[str]:
    seen = []
    used = set()
    for key in preferred or []:
        if key not in used:
            seen.append(key)
            used.add(key)
    for row in rows:
        for key in row.keys():
            if key not in used:
                seen.append(key)
                used.add(key)
    return seen


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = union_fieldnames(rows, preferred)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Metrics JSON files, globs, or directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSVs.")
    parser.add_argument("--loss-explosion-factor", type=float, default=1e3)
    parser.add_argument("--ratio-threshold", type=float, default=1e2)
    parser.add_argument("--underfit-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_paths = discover_metrics(args.inputs)
    out_dir = Path(args.output_dir).expanduser().resolve()

    run_rows: List[Dict[str, Any]] = []
    all_task_rows: List[Dict[str, Any]] = []
    all_epoch_rows: List[Dict[str, Any]] = []

    for path in metrics_paths:
        payload = read_json(path)
        meta = infer_path_meta(path)
        task_rows = collect_task_rows(payload, meta)
        epoch_rows = collect_epoch_rows(payload, meta)
        all_task_rows.extend(task_rows)
        all_epoch_rows.extend(epoch_rows)
        run_rows.append(build_run_row(payload, meta, task_rows, epoch_rows))

    add_failure_labels(
        run_rows,
        loss_explosion_factor=args.loss_explosion_factor,
        ratio_threshold=args.ratio_threshold,
        underfit_fraction=args.underfit_fraction,
    )
    run_rows.sort(key=lambda row: (_to_float(row.get("lambda_ewc")), str(row.get("seed", ""))))
    all_task_rows.sort(key=lambda row: (_to_float(row.get("lambda_ewc")), str(row.get("seed", "")), int(row.get("step", 0))))
    all_epoch_rows.sort(key=lambda row: (
        _to_float(row.get("lambda_ewc")),
        str(row.get("seed", "")),
        int(row.get("task", row.get("step", 0))),
        int(row.get("epoch", 0)),
    ))

    run_cols = [
        "dataset", "seed", "method", "optimizer_tag", "run_prefix", "lambda_ewc",
        "final_avg_acc", "final_forget", "final_current_acc", "mean_current_acc",
        "mean_DF_task", "final_DF_task", "max_DF_epoch", "mean_DF_epoch",
        "mean_D2_task", "final_D2_task", "max_D2_epoch",
        "max_r_EWC_task", "max_r_EWC_epoch", "mean_r_EWC_epoch",
        "nonfinite_loss", "loss_explosion", "ewc_dominated",
        "underfit_vs_lambda0", "failure", "records_task_mechanism",
        "records_epoch_dynamics", "metrics_path",
    ]
    task_cols = [
        "dataset", "seed", "run_prefix", "lambda_ewc", "step", "mechanism_task",
        "use_ewc", "compute_fisher", "fisher_mode", "fisher_weighted_drift",
        "normalized_fisher_drift", "delta_norm", "ewc_penalty_value",
        "train_ce_mean", "train_ewc_mean", "train_ewc_ce_ratio_mean",
        "train_fisher_weighted_drift_mean", "train_normalized_fisher_drift_mean",
        "metrics_path",
    ]
    epoch_cols = [
        "dataset", "seed", "run_prefix", "lambda_ewc", "step", "task", "epoch",
        "epochs", "stage", "use_ewc", "loss_total", "loss_ce", "loss_ewc",
        "ewc_ce_ratio", "train_acc_current", "test_acc_current", "test_acc_seen",
        "fisher_weighted_drift", "normalized_fisher_drift", "delta_norm_sq",
        "ewc_penalty_value", "loss_total_finite", "metrics_path",
    ]

    write_csv(out_dir / "lambda_sweep_runs.csv", run_rows, run_cols)
    write_csv(out_dir / "lambda_sweep_task_metrics.csv", all_task_rows, task_cols)
    write_csv(out_dir / "lambda_sweep_epoch_dynamics.csv", all_epoch_rows, epoch_cols)

    print(f"[OK] metrics files: {len(metrics_paths)}")
    print(f"[OK] wrote {out_dir / 'lambda_sweep_runs.csv'} rows={len(run_rows)}")
    print(f"[OK] wrote {out_dir / 'lambda_sweep_task_metrics.csv'} rows={len(all_task_rows)}")
    print(f"[OK] wrote {out_dir / 'lambda_sweep_epoch_dynamics.csv'} rows={len(all_epoch_rows)}")


if __name__ == "__main__":
    main()
