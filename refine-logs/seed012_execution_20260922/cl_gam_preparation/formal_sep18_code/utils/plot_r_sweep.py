#!/usr/bin/env python3
"""
Line-plot visualisation of FQA metrics vs radius r.

Supports two input formats:
  (A) fqa_partial_means.csv  -- produced by summarize_fqa_validity
  (B) *_fqa_partial.json     -- raw per-radius JSON files (one obj per line)

Six panels (2×3):
  (a) Accuracy change   acc_delta_mean
  (b) Loss |ΔL_A|       loss_delta_abs
  (c) Fisher quad ΔF    {prefix}_fisher_quad
  (d) Error |Δ − ΔF|   {prefix}_error
  (e) Relative error    {prefix}_relative_error
  (f) Linear term |g·Δ| linear_term_abs

Each panel shows SGD and GAM as coloured lines with markers.
When multiple splits (train / test) are present, each gets its own
line style (train=solid, test=dashed).

Usage:
  conda run -n low-rank-cl python utils/plot_r_sweep.py <fqa_partial_dir>

  # compare two directories side-by-side (same optimizer, different run)
  conda run -n low-rank-cl python utils/plot_r_sweep.py <dir1> <dir2>

Output: <fqa_partial_dir>/r_sweep.pdf  (and r_sweep_zoom.pdf for r ≤ 5)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator


# ── styling ───────────────────────────────────────────────────────────────────

def _setup_font() -> None:
    matplotlib.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "axes.unicode_minus": False,
    })


OPT_COLORS   = {"sgd": "#1f77b4", "gam": "#ff7f0e"}
OPT_MARKERS  = {"sgd": "o",       "gam": "s"}
OPT_LABELS   = {"sgd": "SGD",     "gam": "GAM"}

SPLIT_LS     = {"train": "-",     "test": "--"}
SPLIT_ALPHA  = {"train": 1.0,     "test": 0.70}


# ── data loading ──────────────────────────────────────────────────────────────

def _float(v: object) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return float("nan")


def _parse_multi_json(path: Path) -> List[dict]:
    """Parse a file containing one or more concatenated JSON objects."""
    raw = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    pos, results = 0, []
    while pos < len(raw):
        # skip whitespace
        while pos < len(raw) and raw[pos] in " \t\n\r":
            pos += 1
        if pos >= len(raw):
            break
        obj, end = decoder.raw_decode(raw, pos)
        results.append(obj)
        pos = end
    return results


def _prefix_from_filename(name: str) -> Optional[str]:
    """Extract e.g. 'endpoint_train' or 'endpoint_test' from filename."""
    m = re.search(r"(endpoint_(?:train|test))", name)
    if m:
        return m.group(1)
    # bare-'endpoint' FQA files, e.g. *_task00_endpoint_fqa_partial.json
    # (full-model multi-Fisher runs evaluate on current-task train data only)
    if re.search(r"(?:^|_)endpoint_fqa_partial", name):
        return "endpoint"
    return None


def _optimizer_from_filename(name: str) -> Optional[str]:
    """Extract 'sgd' or 'gam' from filename like s1c_sgd_... or fm_multifisher_fqa_sgd_..."""
    m = re.match(r"s1c_(sgd|gam)_", name)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|_)(sgd|gam)(?:_|$)", name)
    return m.group(1) if m else None


def _optimizer_from_path(path: Path) -> Optional[str]:
    """Extract 'sgd' or 'gam' from a filename or one of its parent names."""
    for part in reversed(path.parts):
        opt = _optimizer_from_filename(part)
        if opt:
            return opt
        if re.search(r"(^|_)sgd(_|$)", part):
            return "sgd"
        if re.search(r"(^|_)gam(_|$)", part):
            return "gam"
    return None


def _radius_token_to_float(token: str) -> float:
    """Convert radius tokens such as '0p25' or '3' to floats."""
    return float(token.replace("p", "."))


def _finite_float_list(values: object) -> List[float]:
    if not isinstance(values, list):
        return []
    out = []
    for v in values:
        fv = _float(v)
        if math.isfinite(fv):
            out.append(fv)
    return out


def load_from_csv(csv_path: Path) -> List[Dict]:
    """Load rows from fqa_partial_means.csv, returning normalised dicts."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_from_json_dir(fqa_dir: Path) -> List[Dict]:
    """Load from *_fqa_partial.json files, returning rows matching CSV schema."""
    rows = []
    for jpath in sorted(fqa_dir.glob("*_fqa_partial.json")):
        rows.extend(load_from_fqa_partial_json(jpath))
    return rows


def load_from_fqa_partial_json(jpath: Path) -> List[Dict]:
    """Load a single *_fqa_partial.json file."""
    prefix = _prefix_from_filename(jpath.name)
    optimizer = _optimizer_from_filename(jpath.name)
    if prefix is None or optimizer is None:
        return []
    split = "test" if "test" in prefix else "train"

    rows = []
    objs = _parse_multi_json(jpath)
    for obj in objs:
        r = obj.get("radius", 0.0)
        mean = obj.get("mean", {})
        if not mean:
            # older format: fields directly in obj (not nested under mean)
            mean = {k: v for k, v in obj.items()
                    if k not in ("radius", "radius_idx", "radius_count",
                                 "K", "main_prefix", "extra_prefixes",
                                 "per_u", "hessian_ref")}
        row: Dict = {
            "optimizer":   optimizer,
            "split":       split,
            "setting":     obj.get("main_prefix", prefix),
            "radius":      str(r),
            "radius_idx":  str(obj.get("radius_idx", "")),
            "radius_count": str(obj.get("radius_count", "")),
            "K":           str(obj.get("K", "")),
            "main_prefix": prefix,
        }
        row.update({k: str(v) for k, v in mean.items()})
        per_u = obj.get("per_u", {})
        if isinstance(per_u, dict):
            loss_vals = _finite_float_list(per_u.get("loss_delta"))
            fq_vals = _finite_float_list(per_u.get(f"{prefix}_fisher_quad"))
            lin_abs_vals = _finite_float_list(per_u.get("linear_term_abs"))
            if loss_vals:
                row["_per_u_loss_delta"] = loss_vals
                row["loss_delta_max"] = str(max(loss_vals))
                row["loss_delta_min"] = str(min(loss_vals))
            if fq_vals:
                row["_per_u_fisher_quad"] = fq_vals
                row["fisher_quad_max"] = str(max(fq_vals))
                row["fisher_quad_min"] = str(min(fq_vals))
            if lin_abs_vals:
                row["_per_u_linear_term_abs"] = lin_abs_vals
                row["linear_term_abs_max"] = str(max(lin_abs_vals))
            if loss_vals and fq_vals and len(loss_vals) == len(fq_vals):
                residuals = [dl - qf for dl, qf in zip(loss_vals, fq_vals)]
                worst = max(residuals, key=lambda x: abs(x))
                row["signed_error_worst"] = str(worst)
                row["error_worst"] = str(abs(worst))
        rows.append(row)
    return rows


def _iter_cl_metric_steps(obj: Dict) -> List[Dict]:
    steps = obj.get("mechanism", {}).get("steps", {})
    if isinstance(steps, dict):
        return [v for _, v in sorted(steps.items()) if isinstance(v, dict)]
    if isinstance(steps, list):
        return [v for v in steps if isinstance(v, dict)]
    return []


def load_from_cl_metrics_json(jpath: Path) -> List[Dict]:
    """Load endpoint train/test FQA rows from a *_cl_metrics.json file."""
    optimizer = _optimizer_from_path(jpath)
    if optimizer is None:
        return []

    obj = json.loads(jpath.read_text(encoding="utf-8"))
    suffix_map = {
        "perturbed_loss_mean": "perturbed_loss_mean",
        "perturbed_loss_std": "perturbed_loss_std",
        "perturbed_acc_mean": "perturbed_acc_mean",
        "perturbed_acc_std": "perturbed_acc_std",
        "acc_delta_mean": "acc_delta_mean",
        "acc_delta_std": "acc_delta_std",
        "loss_delta_mean": "loss_delta_signed",
        "loss_delta_std": "loss_delta_std",
        "loss_delta_abs_mean": "loss_delta_abs",
        "linear_term_mean": "linear_term_signed",
        "linear_term_abs_mean": "linear_term_abs",
        "fisher_quad_mean": "fisher_quad",
        "signed_error_mean": "signed_error",
        "error_mean": "error",
        "relative_error_mean": "relative_error",
    }
    pat = re.compile(r"^(endpoint_(train|test)_fqa)_r([0-9]+(?:p[0-9]+)?|0)(?:_(.+))$")

    rows: List[Dict] = []
    for step in _iter_cl_metric_steps(obj):
        grouped: Dict[Tuple[str, str], Dict] = {}
        for key, val in step.items():
            m = pat.match(key)
            if not m:
                continue
            fqa_prefix, split, r_token, suffix = m.groups()
            out_col = suffix_map.get(suffix)
            if out_col is None:
                continue
            main_prefix = fqa_prefix[:-4]  # endpoint_train_fqa -> endpoint_train
            row = grouped.setdefault((fqa_prefix, r_token), {
                "optimizer": optimizer,
                "split": split,
                "setting": main_prefix,
                "radius": str(_radius_token_to_float(r_token)),
                "main_prefix": main_prefix,
                "base_loss": str(step.get(f"{fqa_prefix}_base_loss", "")),
                "base_acc": str(step.get(f"{fqa_prefix}_base_acc", "")),
            })
            row[out_col] = str(val)
        rows.extend(grouped.values())
    return rows


def load_input(path: Path) -> List[Dict]:
    """Load rows from a fqa_partial dir, summary CSV, partial JSON, or cl_metrics JSON."""
    if path.is_file():
        if path.suffix.lower() == ".csv":
            return load_from_csv(path)
        if path.name.endswith("_cl_metrics.json"):
            return load_from_cl_metrics_json(path)
        if path.name.endswith("_fqa_partial.json"):
            return load_from_fqa_partial_json(path)
        raise ValueError(f"Unsupported input file: {path}")

    rows = load_dir(path)
    if rows:
        return rows

    cl_rows: List[Dict] = []
    for jpath in sorted(path.rglob("*_cl_metrics.json")):
        cl_rows.extend(load_from_cl_metrics_json(jpath))
    if cl_rows:
        return cl_rows
    raise ValueError(f"No FQA CSV/JSON inputs found under: {path}")


def _outputs_root_for_staged_input(path: Path) -> Optional[Path]:
    """Map logs/<group>/staged/<stage>/<run>/... to outputs_logs/<group>/<stage>/<run>."""
    parts = path.resolve().parts
    try:
        logs_idx = parts.index("logs")
        staged_idx = parts.index("staged")
    except ValueError:
        return None
    if staged_idx + 2 >= len(parts) or logs_idx + 1 >= len(parts):
        return None
    group = parts[logs_idx + 1]
    stage = parts[staged_idx + 1]
    run = parts[staged_idx + 2]
    return Path.cwd() / "outputs_logs" / group / stage / run


def _extract_base_losses_from_outputs(root: Path) -> Dict[Tuple[str, str], float]:
    """Extract paired base losses keyed by (optimizer, split) from logs or cl_metrics."""
    base: Dict[Tuple[str, str], float] = {}
    if not root.exists():
        return base

    for jpath in sorted(root.rglob("*_cl_metrics.json")):
        opt = _optimizer_from_path(jpath)
        if opt is None:
            continue
        try:
            obj = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for step in _iter_cl_metric_steps(obj):
            for split in ("train", "test"):
                val = step.get(f"endpoint_{split}_fqa_base_loss")
                fval = _float(val)
                if math.isfinite(fval):
                    base[(opt, split)] = fval

    pat = re.compile(
        r"\[FQA\]\[endpoint_(train|test)\]\s+paired\s+base_loss=([0-9.eE+-]+)"
    )
    for log_path in sorted(root.rglob("*.log")):
        opt = _optimizer_from_path(log_path)
        if opt is None:
            continue
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            m = pat.search(line)
            if not m:
                continue
            split, val = m.groups()
            fval = _float(val)
            if math.isfinite(fval):
                base[(opt, split)] = fval
    return base


def augment_loss_fields(rows: List[Dict], source_path: Path) -> None:
    """Add base_loss / perturbed_loss_mean when recoverable from outputs_logs."""
    roots = []
    if "outputs_logs" in source_path.resolve().parts:
        roots.append(source_path if source_path.is_dir() else source_path.parent)
    mapped = _outputs_root_for_staged_input(source_path)
    if mapped is not None:
        roots.append(mapped)

    base_losses: Dict[Tuple[str, str], float] = {}
    for root in roots:
        base_losses.update(_extract_base_losses_from_outputs(root))
    if not base_losses:
        return

    for row in rows:
        opt = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        base = base_losses.get((opt, split))
        if base is None:
            continue
        if not str(row.get("base_loss", "")).strip():
            row["base_loss"] = str(base)
        pert = _resolve_col(row, "perturbed_loss_mean")
        delta = _resolve_col(row, "loss_delta_signed")
        if not math.isfinite(pert) and math.isfinite(delta):
            row["perturbed_loss_mean"] = str(base + delta)


def augment_per_u_extrema(rows: List[Dict], source_path: Path) -> None:
    """Attach per-direction max/min stats from sibling *_fqa_partial.json files."""
    search_dir = source_path.parent if source_path.is_file() else source_path
    if not search_dir.exists():
        return
    json_paths = sorted(search_dir.glob("*_fqa_partial.json"))
    if not json_paths:
        return

    extrema: Dict[Tuple[str, str, float], Dict[str, str]] = {}
    for jpath in json_paths:
        for src in load_from_fqa_partial_json(jpath):
            opt = src.get("optimizer", "").lower()
            split = src.get("split", "train").lower()
            radius = round(_float(src.get("radius", "nan")), 12)
            if not (opt and split and math.isfinite(radius)):
                continue
            vals = {}
            for key in (
                "loss_delta_max", "loss_delta_min",
                "linear_term_abs_max",
                "fisher_quad_max", "fisher_quad_min",
                "signed_error_worst", "error_worst",
                "_per_u_loss_delta", "_per_u_fisher_quad", "_per_u_linear_term_abs",
            ):
                if str(src.get(key, "")).strip():
                    vals[key] = src[key]
            if vals:
                extrema[(opt, split, radius)] = vals

    if not extrema:
        return
    for row in rows:
        opt = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        radius = round(_float(row.get("radius", "nan")), 12)
        vals = extrema.get((opt, split, radius))
        if not vals:
            continue
        for key, val in vals.items():
            if not str(row.get(key, "")).strip():
                row[key] = val


def load_dir(fqa_dir: Path) -> List[Dict]:
    """Auto-detect CSV vs JSON and load rows."""
    csv_path = fqa_dir / "fqa_partial_means.csv"
    if csv_path.exists():
        return load_from_csv(csv_path)
    return load_from_json_dir(fqa_dir)


# ── metric resolution ─────────────────────────────────────────────────────────

def _resolve_col(row: Dict, base: str) -> float:
    """Return the value of base, or of {prefix}_{base} if base absent."""
    if base in row and row[base] != "":
        return _float(row[base])
    prefix = row.get("main_prefix", "")
    full   = f"{prefix}_{base}"
    if full in row and row[full] != "":
        return _float(row[full])
    # fallback: search for any key ending in _{base}
    for k, v in row.items():
        if k.endswith("_" + base) and v != "":
            return _float(v)
    return float("nan")


PANELS: List[Tuple[str, str, str]] = [
    ("acc_delta_mean",     r"$\Delta\mathrm{Acc}$ (%)",
     r"(a) Accuracy change  $\Delta\mathrm{Acc}$"),
    ("loss_delta_abs",     r"$|\Delta L_A|$",
     r"(b) Loss movement  $|\Delta L_A|$"),
    ("fisher_quad",        r"$Q_F$",
     r"(c) Fisher quad  $\Delta_F = Q_F$"),
    ("error",              r"$|\Delta L_A - Q_F|$",
     r"(d) Surrogate error  $|\Delta - \Delta_F|$"),
    ("relative_error",     r"relative error",
     r"(e) Relative error  $|\Delta - \Delta_F| / |\Delta|$"),
    ("linear_term_abs",    r"$|g_A^\top (r u)|$",
     r"(f) Linear term  $|g_A^\top(ru)|$"),
]


# ── drawing helpers ───────────────────────────────────────────────────────────

def _style_ax(ax, ylabel: str, title: str, r_ceil: float) -> None:
    ax.set_xlim(0.0, r_ceil * 1.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("radius $r$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(9)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.88,
              handletextpad=0.4, borderpad=0.45, labelspacing=0.3)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lam_scaled_error(row: Dict, lam: float) -> float:
    """Compute |ΔL_A − λ·Q_F| from stored fields.

    Stored fields:
      loss_delta_abs         = |ΔL_A|   (always ≥ 0)
      {prefix}_fisher_quad   = Q_F      (no lambda, always ≥ 0)

    Because both are already averaged over directions (means), we compute:
      |mean(|ΔL_A|) − λ · mean(Q_F)|
    This is an approximation; the true per-sample version is not stored.
    """
    loss = _float(row.get("loss_delta_abs", "nan"))
    qf   = _resolve_col(row, "fisher_quad")
    if not (math.isfinite(loss) and math.isfinite(qf)):
        return float("nan")
    return abs(loss - lam * qf)


def _signed_fisher_error(row: Dict, lam: Optional[float] = None) -> float:
    """Compute ΔL_A − Q_F, or ΔL_A − λQ_F when lam is provided."""
    if lam is None:
        stored = _resolve_col(row, "signed_error")
        if math.isfinite(stored):
            return stored
    loss = _resolve_col(row, "loss_delta_signed")
    qf = _resolve_col(row, "fisher_quad")
    if not (math.isfinite(loss) and math.isfinite(qf)):
        return float("nan")
    scale = lam if lam is not None else 1.0
    return loss - scale * qf


def _train_loss_value(row: Dict) -> float:
    """Return L_A(w+δ) when available, otherwise signed ΔL_A."""
    pert = _resolve_col(row, "perturbed_loss_mean")
    if math.isfinite(pert):
        return pert
    base = _resolve_col(row, "base_loss")
    delta = _resolve_col(row, "loss_delta_signed")
    if math.isfinite(base) and math.isfinite(delta):
        return base + delta
    return delta


def _train_loss_worst_value(row: Dict) -> float:
    """Return max_u L_A(w+ru), or max signed ΔL_A if base loss is unavailable."""
    delta_max = _resolve_col(row, "loss_delta_max")
    if not math.isfinite(delta_max):
        return _train_loss_value(row)
    base = _resolve_col(row, "base_loss")
    if math.isfinite(base):
        return base + delta_max
    return delta_max


def _signed_fisher_error_worst(row: Dict, lam: Optional[float] = None) -> float:
    """Return signed residual at argmax_u |ΔL_A - scale*Q_F|."""
    loss_vals = row.get("_per_u_loss_delta")
    fq_vals = row.get("_per_u_fisher_quad")
    if isinstance(loss_vals, list) and isinstance(fq_vals, list) and len(loss_vals) == len(fq_vals):
        scale = lam if lam is not None else 1.0
        residuals = [dl - scale * qf for dl, qf in zip(loss_vals, fq_vals)]
        finite = [v for v in residuals if math.isfinite(v)]
        if finite:
            return max(finite, key=lambda x: abs(x))

    if lam is None:
        worst = _resolve_col(row, "signed_error_worst")
        if math.isfinite(worst):
            return worst
    return _signed_fisher_error(row, lam=lam)


def _get_value(row: Dict, metric: str, lam: Optional[float] = None) -> float:
    """Return the numeric value for `metric`, handling the lam_error special case."""
    if metric == "lam_error":
        if lam is None:
            return float("nan")
        return _lam_scaled_error(row, lam)
    if metric == "signed_fisher_error":
        return _signed_fisher_error(row)
    if metric == "signed_lam_fisher_error":
        if lam is None:
            return float("nan")
        return _signed_fisher_error(row, lam=lam)
    if metric == "train_loss":
        return _train_loss_value(row)
    if metric == "train_loss_worst":
        return _train_loss_worst_value(row)
    if metric == "signed_fisher_error_worst":
        return _signed_fisher_error_worst(row)
    if metric == "signed_lam_fisher_error_worst":
        if lam is None:
            return float("nan")
        return _signed_fisher_error_worst(row, lam=lam)
    if metric == "linear_term_abs_worst":
        worst = _resolve_col(row, "linear_term_abs_max")
        if math.isfinite(worst):
            return worst
        return _resolve_col(row, "linear_term_abs")
    if metric == "fisher_quad_worst":
        worst = _resolve_col(row, "fisher_quad_max")
        if math.isfinite(worst):
            return worst
        return _resolve_col(row, "fisher_quad")
    return _resolve_col(row, metric)


def _draw_lines(
    ax,
    rows: List[Dict],
    metric: str,
    r_max: Optional[float],
    tag: str = "",
    split_filter: Optional[str] = None,
    lam: Optional[float] = None,
    include_zero: bool = False,
    ls_override: Optional[str] = None,
    label_prefix: str = "",
    linewidth: float = 1.6,
    markersize: float = 2.4,
) -> None:
    """Draw one line per (optimizer, split) combination.

    split_filter: if set (e.g. 'train'), only that split is drawn.
    lam: if metric=='lam_error', scale Q_F by this factor.
    ls_override: if set, override the split-based line style.
    label_prefix: prepended to every legend label.
    """
    groups: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in rows:
        opt   = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        if split_filter and split != split_filter:
            continue
        r = _float(row.get("radius", "nan"))
        y = _get_value(row, metric, lam=lam)
        if not (math.isfinite(r) and math.isfinite(y)):
            continue
        if r < 0 or (r == 0 and not include_zero):
            continue
        if r_max is not None and r > r_max + 1e-9:
            continue
        groups.setdefault((opt, split), []).append((r, y))

    for (opt, split), pts in sorted(groups.items()):
        if opt not in OPT_COLORS:
            continue
        pts.sort()
        rs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color  = OPT_COLORS[opt]
        marker = OPT_MARKERS[opt]
        ls     = ls_override if ls_override is not None else SPLIT_LS.get(split, "-")
        label_parts = [OPT_LABELS.get(opt, opt)]
        if split != "train":
            label_parts.append(split)
        if tag:
            label_parts.append(tag)
        label = label_prefix + "  ".join(label_parts)
        ax.plot(rs, ys, color=color, linestyle=ls, linewidth=linewidth,
                marker=marker, markersize=markersize, markevery=1,
                alpha=SPLIT_ALPHA.get(split, 1.0),
                label=label, zorder=3)


# ── main figure ───────────────────────────────────────────────────────────────

def draw_r_sweep(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: str = "FQA metrics vs perturbation radius $r$",
) -> Path:
    """
    dirs_rows: list of (tag, rows) pairs.  tag="" when only one directory.
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv > 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 3.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (metric, ylabel, title) in zip(axes.flatten(), PANELS):
        for tag, rows in dirs_rows:
            _draw_lines(ax, rows, metric, r_max=r_max, tag=tag)
        _style_ax(ax, ylabel, title, r_ceil)

    plt.tight_layout()
    if out_path is None:
        out_path = Path("r_sweep.pdf")
    fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── 1×3 core figure ──────────────────────────────────────────────────────────

def _extract_lam_from_path(path: Path, default: float = 1.0) -> float:
    """Try to extract lambda from any component of path, e.g. 'lam2000' → 2000."""
    for part in reversed(path.parts):
        m = re.search(r"lam(\d+)(?:p(\d+))?", part)
        if m:
            integer = int(m.group(1))
            frac    = int(m.group(2)) if m.group(2) else 0
            return float(f"{integer}.{frac}") if frac else float(integer)
    return default


def _extract_lam(dirs_rows: List[Tuple[str, List[Dict]]], default: float = 1.0) -> float:
    """Try to extract lambda from path tags like 'lam2000' or 'lam0p5'."""
    for tag, _ in dirs_rows:
        m = re.search(r"lam(\d+)(?:p(\d+))?", tag)
        if m:
            integer = int(m.group(1))
            frac    = int(m.group(2)) if m.group(2) else 0
            return float(f"{integer}.{frac}") if frac else float(integer)
    return default


def draw_1x3_core(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    lam: Optional[float] = None,
) -> Path:
    """
    1×3 figure with three panels:

      (a) Accuracy         perturbed_acc_mean  [%]       — true value, no abs
      (b) Loss |ΔL_A|      loss_delta_abs                — with abs, train only
      (c) EWC error        |ΔL_A − λ·Q_F|               — computed, with abs

    Panel (c) notes
    ───────────────
    The stored field `endpoint_*_error` = |ΔL_A − Q_F| uses raw Q_F without λ.
    Because λ is NOT stored in the JSON/CSV, we compute:
        |ΔL_A − λ·Q_F|  =  |loss_delta_abs − lam * fisher_quad|
    where lam is extracted from the path name (e.g. "lam2000" → λ=2000)
    or passed explicitly via the `lam` argument.

    If λ cannot be determined, panel (c) falls back to |ΔL_A − Q_F| (no λ).

    Field mapping
    ─────────────
    perturbed_acc_mean  : true accuracy after perturbation (%)
    loss_delta_abs      : |ΔL_A| (always ≥ 0)
    lam_error (virtual) : |loss_delta_abs − lam * fisher_quad| (computed)
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv > 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 3.0)

    # Resolve lambda
    eff_lam = lam if lam is not None else _extract_lam(dirs_rows, default=1.0)
    lam_str = f"$\\lambda$={eff_lam:.4g}"

    if suptitle is None:
        suptitle = f"FQA core metrics vs perturbation radius $r$  ({lam_str})"

    CORE_PANELS = [
        ("perturbed_acc_mean",
         r"Accuracy (%)",
         r"(a) Accuracy  $\overline{f}(r)$"),
        ("loss_delta_abs",
         r"$|\Delta L_A|$",
         r"(b) Loss movement  $|\Delta L_A|$  [train]"),
        ("lam_error",
         r"$|\Delta L_A - \lambda Q_F|$",
         rf"(c) EWC error  $|\Delta L_A - \lambda Q_F|$  ({lam_str})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (metric, ylabel, title) in zip(axes, CORE_PANELS):
        sf = "train" if metric in ("loss_delta_abs", "lam_error") else None
        for tag, rows in dirs_rows:
            _draw_lines(ax, rows, metric, r_max=r_max, tag=tag,
                        split_filter=sf, lam=eff_lam)

        ax.set_xlim(0.0, r_ceil * 1.05)
        ax.set_ylim(bottom=82.0 if metric == "perturbed_acc_mean" else 0.0)
        ax.set_xlabel("radius $r$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(9)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(loc="upper left" if metric != "perturbed_acc_mean" else "lower left",
                  fontsize=8.5, framealpha=0.88,
                  handletextpad=0.4, borderpad=0.45, labelspacing=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if out_path is None:
        out_path = Path("r_sweep_1x3.pdf")
    fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _has_absolute_train_loss(dirs_rows: List[Tuple[str, List[Dict]]]) -> bool:
    for _, rows in dirs_rows:
        for row in rows:
            if row.get("split", "train").lower() != "train":
                continue
            if math.isfinite(_resolve_col(row, "perturbed_loss_mean")):
                return True
            base = _resolve_col(row, "base_loss")
            delta = _resolve_col(row, "loss_delta_signed")
            if math.isfinite(base) and math.isfinite(delta):
                return True
    return False


def _style_triptych_ax(ax, ylabel: str, title: str, r_ceil: float,
                       signed: bool = False,
                       y_min: Optional[float] = None) -> None:
    ax.set_xlim(0.0, r_ceil * 1.03 if r_ceil > 0 else 1.0)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("radius $r$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    if signed:
        ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.75, zorder=1)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(9)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.88,
              handletextpad=0.4, borderpad=0.45, labelspacing=0.3)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_full_data_triptych(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    lam: Optional[float] = None,
    error_scale: str = "raw",
) -> Path:
    """
    Full-range 1×3 figure requested for r-sweep results:

      (a) accuracy vs r                         perturbed_acc_mean
      (b) training loss vs r                    perturbed_loss_mean, or base_loss+ΔL
      (c) signed Fisher residual vs r           ΔL_A − Q_F

    If only summary CSV fields are available and base_loss cannot be recovered,
    panel (b) falls back to signed ΔL_A.  Set error_scale="lambda" to draw
    ΔL_A − λQ_F instead of the stored/raw ΔL_A − Q_F.
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    if error_scale == "lambda":
        err_metric = "signed_lam_fisher_error"
        lam_str = f"$\\lambda$={lam:.4g}" if lam is not None else "$\\lambda$"
        err_ylabel = r"$\Delta L_A - \lambda Q_F$"
        err_title = rf"(c) Signed residual  $\Delta L_A-\lambda Q_F$  ({lam_str})"
    else:
        err_metric = "signed_fisher_error"
        err_ylabel = r"$\Delta L_A - Q_F$"
        err_title = r"(c) Signed residual  $\Delta L_A-Q_F$"

    loss_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$"
    loss_title = (r"(b) Training loss  $L_A(w+\delta)$  [train]"
                  if abs_loss else
                  r"(b) Training loss movement  $\Delta L_A$  [train]")

    if suptitle is None:
        suptitle = "Full r-sweep: accuracy, training loss, signed Fisher residual"

    panels = [
        ("perturbed_acc_mean",
         r"Accuracy (%)",
         r"(a) Accuracy vs $r$",
         None,
         False,
         82.0),
        ("train_loss",
         loss_ylabel,
         loss_title,
         "train",
         not abs_loss,
         None),
        (err_metric,
         err_ylabel,
         err_title,
         "train",
         True,
         None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (metric, ylabel, title, split_filter, signed, y_min) in zip(axes, panels):
        for tag, rows in dirs_rows:
            _draw_lines(ax, rows, metric, r_max=r_max, tag=tag,
                        split_filter=split_filter, lam=lam, include_zero=True)
        _style_triptych_ax(ax, ylabel, title, r_ceil, signed=signed, y_min=y_min)

    plt.tight_layout()
    if out_path is None:
        out_path = Path("r_sweep_full_data_1x3.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_full_data_1x5(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    lam: Optional[float] = None,
    error_scale: str = "raw",
) -> Path:
    """
    Full-range 3+2 figure:
      top:    accuracy | training loss | signed residual
      bottom: linear term | Fisher quad

    The last two panels follow the same metrics as r_sweep.pdf panels (f) and (c):
      linear_term_abs = |g_A^T(ru)|
      fisher_quad     = Q_F = 0.5 * delta^T F delta
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    if error_scale == "lambda":
        err_metric = "signed_lam_fisher_error"
        lam_str = f"$\\lambda$={lam:.4g}" if lam is not None else "$\\lambda$"
        err_ylabel = r"$\Delta L_A - \lambda Q_F$"
        err_title = rf"(c) Signed residual  $\Delta L_A-\lambda Q_F$  ({lam_str})"
    else:
        err_metric = "signed_fisher_error"
        err_ylabel = r"$\Delta L_A - Q_F$"
        err_title = r"(c) Signed residual  $\Delta L_A-Q_F$"

    loss_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$"
    loss_title = (r"(b) Training loss  $L_A(w+\delta)$  [train]"
                  if abs_loss else
                  r"(b) Training loss movement  $\Delta L_A$  [train]")

    if suptitle is None:
        suptitle = "Full r-sweep: accuracy, training loss, residual, linear term, Fisher quad"

    panels = [
        ("perturbed_acc_mean",
         r"Accuracy (%)",
         r"(a) Accuracy vs $r$",
         None,
         False,
         82.0),
        ("train_loss",
         loss_ylabel,
         loss_title,
         "train",
         not abs_loss,
         None),
        (err_metric,
         err_ylabel,
         err_title,
         "train",
         True,
         None),
        ("linear_term_abs",
         r"$|g_A^\top(ru)|$",
         r"(d) Linear term  $|g_A^\top(ru)|$",
         None,
         False,
         0.0),
        ("fisher_quad_worst",
         r"$Q_F$",
         r"(e) Worst Fisher quad  $\max_u Q_F(r,u)$  [train]",
         "train",
         False,
         0.0),
    ]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (metric, ylabel, title, split_filter, signed, y_min) in zip(axes, panels):
        for tag, rows in dirs_rows:
            _draw_lines(ax, rows, metric, r_max=r_max, tag=tag,
                        split_filter=split_filter, lam=lam, include_zero=True)
        _style_triptych_ax(ax, ylabel, title, r_ceil, signed=signed, y_min=y_min)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path is None:
        out_path = Path("r_sweep_full_data_1x5.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_full_data_worst_1x5(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    lam: Optional[float] = None,
    error_scale: str = "raw",
) -> Path:
    """
    Worst-case full-range 3+2 figure.

      (a) mean accuracy after perturbation
      (b) max_u L_A(w+ru) on train
      (c) signed residual at argmax_u |ΔL_A - Q_F| on train
      (d) max_u |g_A^T(ru)| for each split
      (e) max_u Q_F(r,u) on train
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    if error_scale == "lambda":
        err_metric = "signed_lam_fisher_error_worst"
        lam_str = f"$\\lambda$={lam:.4g}" if lam is not None else "$\\lambda$"
        err_ylabel = r"$\Delta L_A - \lambda Q_F$"
        err_title = rf"(c) Worst signed residual  argmax $|\Delta L_A-\lambda Q_F|$  ({lam_str}) [train]"
    else:
        err_metric = "signed_fisher_error_worst"
        err_ylabel = r"$\Delta L_A - Q_F$"
        err_title = r"(c) Worst signed residual  argmax $|\Delta L_A-Q_F|$  [train]"

    if suptitle is None:
        suptitle = "Worst-case r-sweep: accuracy, loss, residual, linear term, Fisher quad"

    panels = [
        ("perturbed_acc_mean",
         r"Accuracy (%)",
         r"(a) Accuracy vs $r$",
         None,
         False,
         82.0),
        ("train_loss_worst",
         r"$\max_u L_A(w+ru)$",
         r"(b) Worst training loss  $\max_u L_A(w+ru)$  [train]",
         "train",
         False,
         None),
        (err_metric,
         err_ylabel,
         err_title,
         "train",
         True,
         None),
        ("linear_term_abs_worst",
         r"$\max_u |g_A^\top(ru)|$",
         r"(d) Worst linear term  $\max_u |g_A^\top(ru)|$",
         None,
         False,
         0.0),
        ("fisher_quad_worst",
         r"$\max_u Q_F(r,u)$",
         r"(e) Worst Fisher quad  $\max_u Q_F(r,u)$  [train]",
         "train",
         False,
         0.0),
    ]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (metric, ylabel, title, split_filter, signed, y_min) in zip(axes, panels):
        for tag, rows in dirs_rows:
            _draw_lines(ax, rows, metric, r_max=r_max, tag=tag,
                        split_filter=split_filter, lam=lam, include_zero=True)
        _style_triptych_ax(ax, ylabel, title, r_ceil, signed=signed, y_min=y_min)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path is None:
        out_path = Path("r_sweep_full_data_worst_1x5.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── 1×6 decomposition figure ─────────────────────────────────────────────────
#
# Taylor decomposition colour code (matches the equation in the paper):
#   RED   = ΔL_A  = L_A(θ* + Δθ) − L_A(θ*)          [left-hand side]
#   BLUE  = ∇L_A^T Δθ                                  [linear/gradient term]
#   GREEN = ½ Δθ^T H_A Δθ  ≈ Q_F (diag Fisher)        [quadratic term]
#
# Panel titles use the same colour as the dominant term in that panel.

TERM_RED   = "#d62728"   # ΔL_A
TERM_BLUE  = "#1f77b4"   # linear / gradient term
TERM_GREEN = "#2ca02c"   # Fisher / Hessian quad
TERM_GRAY  = "#555555"   # residual / sum

# Line-style by optimizer (colour encodes term, style encodes optimizer)
OPT_LS_DECOMP  = {"sgd": "-",  "gam": "--"}
OPT_MK_DECOMP  = {"sgd": "o",  "gam": "s"}
OPT_ALPHA_DECOMP = {"sgd": 1.0, "gam": 0.80}


def _draw_term_lines(
    ax,
    rows: List[Dict],
    metric: str,
    r_max: Optional[float],
    term_color: str,
    split_filter: Optional[str] = None,
    lam: Optional[float] = None,
    include_zero: bool = True,
    linestyle_override: Optional[str] = None,
    alpha_override: Optional[float] = None,
    label_prefix: str = "",
    linewidth: float = 1.7,
) -> None:
    """Draw one line per optimizer in a fixed term colour; line-style = optimizer."""
    groups: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        opt   = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        if split_filter and split != split_filter:
            continue
        r = _float(row.get("radius", "nan"))
        y = _get_value(row, metric, lam=lam)
        if not (math.isfinite(r) and math.isfinite(y)):
            continue
        if r < 0 or (r == 0 and not include_zero):
            continue
        if r_max is not None and r > r_max + 1e-9:
            continue
        groups.setdefault(opt, []).append((r, y))

    for opt, pts in sorted(groups.items()):
        if opt not in OPT_LS_DECOMP:
            continue
        pts.sort()
        rs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ls    = linestyle_override if linestyle_override is not None else OPT_LS_DECOMP[opt]
        alpha = alpha_override if alpha_override is not None else OPT_ALPHA_DECOMP[opt]
        mk    = OPT_MK_DECOMP[opt]
        label = f"{label_prefix}{OPT_LABELS.get(opt, opt)}"
        ax.plot(rs, ys,
                color=term_color, linestyle=ls, linewidth=linewidth,
                marker=mk, markersize=2.4, markevery=1,
                alpha=alpha, label=label, zorder=3)


def _draw_decomp_panel(
    ax,
    rows: List[Dict],
    r_max: Optional[float],
    lam: Optional[float] = None,
    lw: float = 1.5,
    ms: float = 2.0,
) -> None:
    """Panel (f): overlay all three Taylor terms + their sum.

    Colour = optimizer  (blue=SGD, orange=GAM  — matches all other panels).
    Line style = term:
      solid   (-)   — ΔL_A
      dashed  (--)  — linear term
      dotted  (:)   — Q_F
      dash-dot(-.)  — linear + Q_F  (predicted sum)
    """
    # Term line-styles (same for every optimizer)
    TERM_LS = {
        "delta_L": "-",
        "linear":  "--",
        "fisher":  ":",
        "sum":     "-.",
    }

    groups: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for row in rows:
        opt   = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        if split != "train":
            continue
        r  = _float(row.get("radius", "nan"))
        dl = _get_value(row, "loss_delta_signed")
        lt = _get_value(row, "linear_term_signed")
        fq = _resolve_col(row, "fisher_quad")
        if not all(math.isfinite(v) for v in (r, dl, lt, fq)):
            continue
        if r < 0:
            continue
        if r_max is not None and r > r_max + 1e-9:
            continue
        groups.setdefault(opt, []).append((r, dl, lt, fq))

    for opt, pts in sorted(groups.items()):
        if opt not in OPT_COLORS:
            continue
        pts.sort()
        rs   = [p[0] for p in pts]
        dls  = [p[1] for p in pts]
        lts  = [p[2] for p in pts]
        fqs  = [p[3] for p in pts]
        sums = [lt + fq for lt, fq in zip(lts, fqs)]

        color = OPT_COLORS[opt]
        mk    = OPT_MARKERS[opt]
        name  = OPT_LABELS.get(opt, opt)
        alpha = SPLIT_ALPHA.get("train", 1.0)

        ax.plot(rs, dls,  color=color, linestyle=TERM_LS["delta_L"], linewidth=lw * 1.2,
                marker=mk, markersize=ms, markevery=1, alpha=alpha,
                label=f"$\\Delta L_A$  {name}", zorder=4)
        ax.plot(rs, sums, color=color, linestyle=TERM_LS["sum"],     linewidth=lw,
                marker=mk, markersize=ms, markevery=1, alpha=alpha * 0.75,
                label=f"linear$+Q_F$  {name}", zorder=2)


def _draw_lines_band(
    ax,
    rows: List[Dict],
    metric_mean: str,
    r_max: Optional[float],
    split_filter: Optional[str] = None,
    include_zero: bool = True,
    metric_lo: Optional[str] = None,
    metric_hi: Optional[str] = None,
    lw: float = 1.0,
    ms: float = 1.5,
    alpha_band: float = 0.15,
    label_prefix: str = "",
) -> None:
    """Draw mean line per optimizer + shaded band between metric_lo and metric_hi.

    If only metric_hi is given (one-sided), band is drawn from mean to hi.
    Colour = optimizer (blue=SGD, orange=GAM).
    """
    groups: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for row in rows:
        opt   = row.get("optimizer", "").lower()
        split = row.get("split", "train").lower()
        if split_filter and split != split_filter:
            continue
        r    = _float(row.get("radius", "nan"))
        ymean = _get_value(row, metric_mean)
        ylo   = _get_value(row, metric_lo)   if metric_lo else math.nan
        yhi   = _get_value(row, metric_hi)   if metric_hi else math.nan
        if not (math.isfinite(r) and math.isfinite(ymean)):
            continue
        if r < 0 or (r == 0 and not include_zero):
            continue
        if r_max is not None and r > r_max + 1e-9:
            continue
        groups.setdefault(opt, []).append((r, ymean, ylo, yhi))

    for opt, pts in sorted(groups.items()):
        if opt not in OPT_COLORS:
            continue
        pts.sort()
        rs    = [p[0] for p in pts]
        means = [p[1] for p in pts]
        los   = [p[2] for p in pts]
        his   = [p[3] for p in pts]
        color  = OPT_COLORS[opt]
        marker = OPT_MARKERS[opt]
        alpha  = SPLIT_ALPHA.get(split_filter or "train", 1.0)
        label  = label_prefix + OPT_LABELS.get(opt, opt)

        # shaded band
        have_hi = all(math.isfinite(v) for v in his)
        have_lo = all(math.isfinite(v) for v in los)
        if have_hi and have_lo:
            ax.fill_between(rs, los, his, color=color,
                            alpha=alpha_band, linewidth=0)
        elif have_hi:
            ax.fill_between(rs, means, his, color=color,
                            alpha=alpha_band, linewidth=0)

        # mean line (solid)
        ax.plot(rs, means, color=color, linestyle="-", linewidth=lw,
                marker=marker, markersize=ms, markevery=1, alpha=alpha,
                label=label, zorder=4)


def _style_triptych_ax_colored(
    ax,
    ylabel: str,
    title: str,
    r_ceil: float,
    title_color: str = "black",
    signed: bool = False,
    y_min: Optional[float] = None,
    legend_loc: str = "best",
    legend_ncol: int = 1,
) -> None:
    """Like _style_triptych_ax but with a coloured title."""
    ax.set_xlim(0.0, r_ceil * 1.03 if r_ceil > 0 else 1.0)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("radius $r$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    t = ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    t.set_color(title_color)
    if signed:
        ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.75, zorder=1)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_fontsize(9)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc=legend_loc, fontsize=7.5, framealpha=0.88,
              handletextpad=0.4, borderpad=0.4, labelspacing=0.25,
              ncol=legend_ncol)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_full_data_1x6(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    lam: Optional[float] = None,
) -> Path:
    """1×6 Taylor decomposition figure.

    Panels (single row):
      (a) Accuracy vs r                — train + test, all opts
      (b) Training loss L_A(w+δ)       — train only
      (c) ΔL_A [RED] + residual ΔL_A−Q_F  [title in red]
      (d) Linear term |g_A^T(ru)|      [title in blue]
      (e) Fisher quad Q_F              [title in green]
      (f) Decomposition: ΔL_A = linear + Q_F + residual  [all three terms]

    Colour convention (matches equation in paper):
      RED   = ΔL_A  (left-hand side)
      BLUE  = ∇L_A^T Δθ  (linear / gradient term)
      GREEN = ½ Δθ^T H_A Δθ ≈ Q_F  (Fisher / Hessian quad)
      GRAY dashed = linear + Q_F  (predicted sum)

    For panels (a)–(e) SGD=blue, GAM=orange (existing colour code).
    For panel (c), ΔL_A is overlaid in RED (solid SGD, dashed GAM).
    For panel (f), line colour = term; line style = optimizer.
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$  [signed]"
    loss_title  = (r"(b) Training loss  $L_A(w+\delta)$  [train]"
                   if abs_loss else
                   r"(b) Training loss movement  $\Delta L_A$  [train]")

    if suptitle is None:
        suptitle = (r"Taylor decomposition of old-task loss:  "
                    r"$\mathbf{\color{red}{\Delta L_A}}$ = "
                    r"$\mathbf{\color{blue}{\nabla L_A^\top \Delta\theta}}$ + "
                    r"$\mathbf{\color{green}{Q_F}}$ + residual")

    # ── build figure: 2 rows × 3 cols ────────────────────────────────────────
    #   Row 0: [c] ΔL_A+residual  [d] linear term   [e] Fisher quad
    #   Row 1: [a] accuracy        [b] training loss  [f] decomposition
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    fig.suptitle(
        r"Taylor decomposition: "
        r"$\mathbf{\Delta L_A}$ = "
        r"$\mathbf{\nabla L_A^\top\!\Delta\theta}$ + "
        r"$\mathbf{\frac{1}{2}\Delta\theta^\top H_A\Delta\theta}$ + "
        r"$o(\|\Delta\theta\|^2)$",
        fontsize=12, fontweight="bold",
    )

    # Row 0: delta_L, training_loss, accuracy
    ax_c = axes[0, 0]   # ΔL_A + residual
    ax_b = axes[0, 1]   # training loss
    ax_a = axes[0, 2]   # accuracy
    # Row 1: linear_term, fisher_quad, decomposition
    ax_d = axes[1, 0]   # linear term
    ax_e = axes[1, 1]   # Fisher quad
    ax_f = axes[1, 2]   # decomposition

    LW  = 1.0   # line width for all panels in this figure
    LWb = 1.2   # slightly thicker for the primary ΔL_A line
    MS  = 1.5   # marker size for all panels

    # ── [0,0] ΔL_A (solid) + residual ΔL_A − Q_F (dashed), colour = optimizer ─
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "loss_delta_signed", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style_triptych_ax_colored(
        ax_c, r"$\Delta L_A$",
        r"(1) $\Delta L_A = L_A(w+\delta) - L_A(w)$  [train]",
        r_ceil, title_color=TERM_RED, signed=True, legend_loc="upper left",
    )

    # ── [0,1] Training loss ───────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "train_loss", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_b, loss_ylabel, loss_title,
                                r_ceil, title_color="black",
                                legend_loc="upper left")

    # ── [0,2] Accuracy ────────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter=None, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_a, "Accuracy (%)", r"(3) Accuracy vs $r$",
                                r_ceil, title_color="black", y_min=82.0,
                                legend_loc="lower left")

    # ── [1,0] Linear term ─────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_d, rows, "linear_term_abs", r_max=r_max, tag=tag,
                    split_filter=None, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(
        ax_d, r"$|\nabla L_A^\top \Delta\theta|$",
        r"(4) Linear term  $|\nabla L_A^\top \Delta\theta|$",
        r_ceil, title_color=TERM_BLUE, y_min=0.0, legend_loc="upper left",
    )

    # ── [1,1] Fisher quad ─────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_e, rows, "fisher_quad_worst", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(
        ax_e, r"$Q_F$",
        r"(5) Fisher quad  $\max_u Q_F(r,u)$  [train]",
        r_ceil, title_color=TERM_GREEN, y_min=0.0, legend_loc="upper left",
    )

    # ── [1,2] (f) Decomposition panel ────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_decomp_panel(ax_f, rows, r_max=r_max, lam=lam, lw=LW, ms=MS)
    # Legend: 4 entries (2 terms × 2 optimisers) → 2 columns
    _style_triptych_ax_colored(
        ax_f, "value",
        r"(6) $\Delta L_A$  vs  linear $+ Q_F$",
        r_ceil, title_color=TERM_GRAY, signed=True,
        legend_loc="upper left", legend_ncol=2,
    )

    if out_path is None:
        out_path = Path("r_sweep_full_data_1x6.png")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_1x3_mean(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """1×3 figure — first three panels of draw_full_data_1x6, larger fonts, no title.

    (a) ΔL_A  [mean, train]
    (b) Training loss movement  [mean, train]
    (c) Accuracy  [mean, train]
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$"
    loss_title  = r"(b) Training loss  $L_A(w+\delta)$" if abs_loss else r"(b) $\Delta L_A$"

    LW   = 1.4
    LWb  = 1.6
    MS   = 2.0
    FSL  = 14    # axis label font size
    FST  = 13    # subplot title font size
    FSTK = 12    # tick label font size
    FSLEG = 11   # legend font size

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 3.8),
                                            gridspec_kw={"wspace": 0.38})

    def _style(ax, ylabel, title, title_color="black",
               signed=False, y_min=None, legend_loc="best"):
        ax.set_xlim(0.0, r_ceil * 1.03 if r_ceil > 0 else 1.0)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(r"radius $r$", fontsize=FSL)
        ax.set_ylabel(ylabel, fontsize=FSL)
        t = ax.set_title(title, fontsize=FST, fontweight="bold", pad=5)
        t.set_color(title_color)
        if signed:
            ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.75, zorder=1)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(FSTK - 1)
        ax.tick_params(axis="both", labelsize=FSTK)
        ax.legend(loc=legend_loc, fontsize=FSLEG, framealpha=0.88,
                  handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) ΔL_A ─────────────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "loss_delta_signed", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style(ax_a, r"$\Delta L_A = L_A(w\!+\!\delta) - L_A(w)$",
           r"(a) $\Delta L_A = L_A(w\!+\!\delta) - L_A(w)$",
           title_color="black", signed=True, legend_loc="upper left")

    # ── (b) Training loss ─────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "train_loss", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_b, loss_ylabel, loss_title,
           title_color="black", legend_loc="upper left")

    # ── (c) Accuracy [train] ──────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_c, "Accuracy (%)", r"(c) Accuracy",
           title_color="black", y_min=82.0, legend_loc="lower left")

    fig.tight_layout()

    if out_path is None:
        out_path = Path("r_sweep_1x3_mean.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_1x3_residual(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    lam: Optional[float] = None,
    opt_colors: Optional[Dict[str, str]] = None,
) -> Path:
    """1×3 figure derived from draw_1x3_core, larger fonts, no suptitle.

    (a) The approximation residual  R_t(Δ) = |ΔL_A − λQ_F|  — train only
    (b) |ΔL_A|                — train only
    (c) Accuracy              — train only

    opt_colors: optional dict to override OPT_COLORS, e.g. {"sgd": "red", "gam": "green"}
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv > 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 3.0)

    eff_lam = lam if lam is not None else _extract_lam(dirs_rows, default=1.0)

    # temporarily patch global colour map
    _saved_colors = OPT_COLORS.copy()
    if opt_colors:
        OPT_COLORS.update(opt_colors)

    LW   = 1.4
    LWb  = 1.6
    MS   = 2.0
    FSL  = 14
    FST  = 13
    FSTK = 12
    FSLEG = 11

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 3.8),
                                            gridspec_kw={"wspace": 0.38})

    def _style(ax, ylabel, title, y_min=None, legend_loc="best", signed=False,
               ylabel_fs=None):
        ax.set_xlim(0.0, r_ceil * 1.03 if r_ceil > 0 else 1.0)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(r"radius $r$", fontsize=FSL)
        ax.set_ylabel(ylabel, fontsize=ylabel_fs if ylabel_fs is not None else FSL)
        ax.set_title(title, fontsize=FST, fontweight="bold", pad=5)
        if signed:
            ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.75, zorder=1)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(FSTK - 1)
        ax.tick_params(axis="both", labelsize=FSTK)
        ax.legend(loc=legend_loc, fontsize=FSLEG, framealpha=0.88,
                  handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) The approximation residual R_t(Δ) = |ΔL_A − λQ_F| [train] ───────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "lam_error", r_max=r_max, tag=tag,
                    split_filter="train", lam=eff_lam, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_a,
           r"$\mathcal{R}_t(\boldsymbol{\Delta})$",
           "",
           y_min=0.0, legend_loc="upper left", ylabel_fs=18)

    # ── (b) ΔL_A [train] ─────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "loss_delta_abs", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style(ax_b, r"$\Delta L_A$",
           "",
           y_min=0.0, legend_loc="upper left", ylabel_fs=18)

    # ── (c) Accuracy [train only] ─────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_c, "Accuracy (%)", "",
           y_min=85.0, legend_loc="lower left", ylabel_fs=18)
    ax_c.yaxis.set_major_locator(MultipleLocator(1))

    fig.tight_layout()

    if out_path is None:
        out_path = Path("r_sweep_1x3_residual.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)

    # restore original colours
    OPT_COLORS.clear()
    OPT_COLORS.update(_saved_colors)

    return out_path


def draw_1x4_residual(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    lam: Optional[float] = None,
) -> Path:
    """1×4 figure: residual | |ΔL_A| | ΔL_A (signed) | Accuracy — no titles.

    (a) R_t(Δ) = |ΔL_A − λQ_F|   [train]
    (b) |ΔL_A|                     [train]
    (c) ΔL_A (signed mean)         [train]   ← panel-2 of draw_1x3_mean
    (d) Accuracy                   [train]
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv > 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 3.0)

    eff_lam = lam if lam is not None else _extract_lam(dirs_rows, default=1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_c_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$"

    LW   = 1.4
    LWb  = 1.6
    MS   = 2.0
    FSL  = 14
    FST  = 13
    FSTK = 12
    FSLEG = 11
    YFS  = 18   # y-axis label font size

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(1, 4, figsize=(19, 3.8),
                                                   gridspec_kw={"wspace": 0.40})

    def _style(ax, ylabel, y_min=None, legend_loc="best", signed=False,
               ylabel_fs=YFS):
        ax.set_xlim(0.0, r_ceil * 1.03 if r_ceil > 0 else 1.0)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(r"radius $r$", fontsize=FSL)
        ax.set_ylabel(ylabel, fontsize=ylabel_fs)
        if signed:
            ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.75, zorder=1)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(FSTK - 1)
        ax.tick_params(axis="both", labelsize=FSTK)
        ax.legend(loc=legend_loc, fontsize=FSLEG, framealpha=0.88,
                  handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) R_t(Δ) ───────────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "lam_error", r_max=r_max, tag=tag,
                    split_filter="train", lam=eff_lam, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_a, r"$\mathcal{R}_t(\boldsymbol{\Delta})$",
           y_min=0.0, legend_loc="upper left")

    # ── (b) |ΔL_A| ───────────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "loss_delta_abs", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style(ax_b, r"$\Delta L_A$",
           y_min=0.0, legend_loc="upper left")

    # ── (c) L_A(θ+Δ) signed (train_loss) ────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "train_loss", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_c, r"$L_A(\boldsymbol{\theta} + \boldsymbol{\Delta})$",
           y_min=0.0, legend_loc="upper left", signed=False)
    _top_c = max(ax_c.get_ylim()[1], 0.31)
    ax_c.set_ylim(0.3, _top_c)

    # ── (d) Accuracy [train] ──────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_d, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style(ax_d, "Accuracy (%)",
           y_min=85.0, legend_loc="lower left")
    ax_d.yaxis.set_major_locator(MultipleLocator(1))

    fig.tight_layout()

    if out_path is None:
        out_path = Path("r_sweep_1x4_residual.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_full_data_1x6_band(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    lam: Optional[float] = None,
) -> Path:
    """Same 2×3 layout as draw_full_data_1x6, but with mean ± per-direction band.

    Shaded region = [min, max] over K random unit directions.
    Panels that have no per-direction data (loss, accuracy) show mean only.

    Band coverage:
      (1) ΔL_A        — [loss_delta_min, loss_delta_max]
      (2) Training loss — mean only
      (3) Accuracy      — mean only
      (4) Linear term   — [mean, linear_term_abs_max]  (abs, one-sided)
      (5) Fisher quad   — [fisher_quad_min, fisher_quad_max]
      (6) Decomposition — ΔL_A with band; linear+Q_F mean only
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_ylabel = r"$L_A(w+\delta)$" if abs_loss else r"$\Delta L_A$  [signed]"
    loss_title  = (r"(2) Training loss  $L_A(w+\delta)$  [train]"
                   if abs_loss else
                   r"(2) Training loss movement  $\Delta L_A$  [train]")

    LW  = 1.0
    LWb = 1.2
    MS  = 1.5
    ALPHA_BAND = 0.18

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    fig.suptitle(
        r"Taylor decomposition (mean $\pm$ per-direction range): "
        r"$\Delta L_A$ = linear $+ Q_F +$ residual",
        fontsize=12, fontweight="bold",
    )

    ax_c = axes[0, 0]
    ax_b = axes[0, 1]
    ax_a = axes[0, 2]
    ax_d = axes[1, 0]
    ax_e = axes[1, 1]
    ax_f = axes[1, 2]

    # ── [0,0] ΔL_A with band [loss_delta_min, loss_delta_max] ────────────────
    for tag, rows in dirs_rows:
        _draw_lines_band(ax_c, rows, "loss_delta_signed", r_max,
                         split_filter="train", include_zero=True,
                         metric_lo="loss_delta_min", metric_hi="loss_delta_max",
                         lw=LWb, ms=MS, alpha_band=ALPHA_BAND)
    _style_triptych_ax_colored(
        ax_c, r"$\Delta L_A$",
        r"(1) $\Delta L_A = L_A(w+\delta) - L_A(w)$  [train]",
        r_ceil, title_color=TERM_RED, signed=True, legend_loc="upper left",
    )

    # ── [0,1] Training loss — mean only ──────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "train_loss", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_b, loss_ylabel, loss_title,
                                r_ceil, title_color="black",
                                legend_loc="upper left")

    # ── [0,2] Accuracy — mean only ────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter=None, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_a, "Accuracy (%)", r"(3) Accuracy vs $r$",
                                r_ceil, title_color="black", y_min=82.0,
                                legend_loc="lower left")

    # ── [1,0] Linear term — one-sided band [mean, max] ───────────────────────
    for tag, rows in dirs_rows:
        _draw_lines_band(ax_d, rows, "linear_term_abs", r_max,
                         split_filter=None, include_zero=True,
                         metric_hi="linear_term_abs_max",
                         lw=LW, ms=MS, alpha_band=ALPHA_BAND)
    _style_triptych_ax_colored(
        ax_d, r"$|\nabla L_A^\top \Delta\theta|$",
        r"(4) Linear term  $|\nabla L_A^\top \Delta\theta|$  [mean $\to$ max]",
        r_ceil, title_color=TERM_BLUE, y_min=0.0, legend_loc="upper left",
    )

    # ── [1,1] Fisher quad — full band [min, max] ──────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines_band(ax_e, rows, "fisher_quad", r_max,
                         split_filter="train", include_zero=True,
                         metric_lo="fisher_quad_min", metric_hi="fisher_quad_max",
                         lw=LW, ms=MS, alpha_band=ALPHA_BAND)
    _style_triptych_ax_colored(
        ax_e, r"$Q_F$",
        r"(5) Fisher quad  $Q_F(r,u)$  [train, mean $\pm$ range]",
        r_ceil, title_color=TERM_GREEN, y_min=0.0, legend_loc="upper left",
    )

    # ── [1,2] Decomposition — ΔL_A with band; sum (linear+Q_F) mean only ─────
    # collect per-optimizer: r → (mean_dl, lo_dl, hi_dl, mean_sum)
    decomp_groups: Dict[str, List] = {}
    for _, rows in dirs_rows:
        for row in rows:
            opt   = row.get("optimizer", "").lower()
            split = row.get("split", "train").lower()
            if split != "train" or opt not in OPT_COLORS:
                continue
            r   = _float(row.get("radius", "nan"))
            dl  = _get_value(row, "loss_delta_signed")
            dlo = _get_value(row, "loss_delta_min")
            dhi = _get_value(row, "loss_delta_max")
            lt  = _get_value(row, "linear_term_signed")
            fq  = _resolve_col(row, "fisher_quad")
            if not all(math.isfinite(v) for v in (r, dl, lt, fq)):
                continue
            if r < 0 or (r_max is not None and r > r_max + 1e-9):
                continue
            decomp_groups.setdefault(opt, []).append((r, dl, dlo, dhi, lt + fq))

    for opt, pts in sorted(decomp_groups.items()):
        pts.sort()
        rs    = [p[0] for p in pts]
        means = [p[1] for p in pts]
        los   = [p[2] for p in pts]
        his   = [p[3] for p in pts]
        sums  = [p[4] for p in pts]
        color  = OPT_COLORS[opt]
        marker = OPT_MARKERS[opt]
        name   = OPT_LABELS.get(opt, opt)
        # band for ΔL_A
        have_band = all(math.isfinite(v) for v in los + his)
        if have_band:
            ax_f.fill_between(rs, los, his, color=color,
                               alpha=ALPHA_BAND, linewidth=0)
        # mean ΔL_A solid
        ax_f.plot(rs, means, color=color, linestyle="-", linewidth=LWb,
                  marker=marker, markersize=MS, markevery=1,
                  label=f"$\\Delta L_A$  {name}", zorder=4)
        # sum dash-dot
        ax_f.plot(rs, sums, color=color, linestyle="-.", linewidth=LW,
                  marker=marker, markersize=MS, markevery=1, alpha=0.75,
                  label=f"linear$+Q_F$  {name}", zorder=3)

    _style_triptych_ax_colored(
        ax_f, "value",
        r"(6) $\Delta L_A$  vs  linear $+ Q_F$  [band=direction range]",
        r_ceil, title_color=TERM_GRAY, signed=True,
        legend_loc="upper left", legend_ncol=2,
    )

    if out_path is None:
        out_path = Path("r_sweep_full_data_1x6_band.png")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_full_data_1x6_worst(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    lam: Optional[float] = None,
) -> Path:
    """Same 2×3 layout, every line shows the **worst-case** (max) direction, no band.

    Panel mapping
    ─────────────
      (1) ΔL_A        — max over K directions (loss_delta_max)
      (2) Training loss — worst direction (base + loss_delta_max)
      (3) Accuracy      — mean only (no per-direction worst stored)
      (4) Linear term   — max (linear_term_abs_worst)
      (5) Fisher quad   — max (fisher_quad_worst)
      (6) Decomposition — worst ΔL_A vs worst sum (lt_max + fq_max)
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_ylabel = r"$L_A(w+\delta)^{\max}$" if abs_loss else r"$\Delta L_A^{\max}$  [signed]"
    loss_title  = (r"(2) Training loss — worst direction  [train]"
                   if abs_loss else
                   r"(2) Training loss movement — worst direction  [train]")

    LW  = 1.0
    LWb = 1.2
    MS  = 1.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    fig.suptitle(
        r"Taylor decomposition — worst direction (max over $K$ random $u$)",
        fontsize=12, fontweight="bold",
    )

    ax_c = axes[0, 0]
    ax_b = axes[0, 1]
    ax_a = axes[0, 2]
    ax_d = axes[1, 0]
    ax_e = axes[1, 1]
    ax_f = axes[1, 2]

    # ── (1) ΔL_A worst ───────────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "loss_delta_max", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style_triptych_ax_colored(
        ax_c, r"$\Delta L_A^{\max}$",
        r"(1) $\Delta L_A$ — worst direction  [train]",
        r_ceil, title_color=TERM_RED, signed=True, legend_loc="upper left",
    )

    # ── (2) Training loss — worst direction (base + loss_delta_max) ──────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "train_loss_worst", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_b, loss_ylabel, loss_title,
                                r_ceil, title_color="black",
                                legend_loc="upper left")

    # ── (3) Accuracy — mean only (no per-direction worst stored) ─────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter=None, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_a, "Accuracy (%)",
                                r"(3) Accuracy vs $r$  [mean only]",
                                r_ceil, title_color="black", y_min=82.0,
                                legend_loc="lower left")

    # ── (4) Linear term worst ─────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_d, rows, "linear_term_abs_worst", r_max=r_max, tag=tag,
                    split_filter=None, include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(
        ax_d, r"$|\nabla L_A^\top \Delta\theta|^{\max}$",
        r"(4) Linear term — worst direction",
        r_ceil, title_color=TERM_BLUE, y_min=0.0, legend_loc="upper left",
    )

    # ── (5) Fisher quad worst ─────────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_e, rows, "fisher_quad_worst", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(
        ax_e, r"$Q_F^{\max}$",
        r"(5) Fisher quad — worst direction  [train]",
        r_ceil, title_color=TERM_GREEN, y_min=0.0, legend_loc="upper left",
    )

    # ── (6) Decomposition — worst ΔL_A vs worst sum ───────────────────────────
    decomp_groups: Dict[str, list] = {}
    for _, rows in dirs_rows:
        for row in rows:
            opt   = row.get("optimizer", "").lower()
            split = row.get("split", "train").lower()
            if split != "train" or opt not in OPT_COLORS:
                continue
            r      = _float(row.get("radius", "nan"))
            dl_max = _get_value(row, "loss_delta_max")
            lt_max = _get_value(row, "linear_term_abs_worst")
            fq_max = _get_value(row, "fisher_quad_worst")
            if not all(math.isfinite(v) for v in (r, dl_max, lt_max, fq_max)):
                continue
            if r < 0 or (r_max is not None and r > r_max + 1e-9):
                continue
            decomp_groups.setdefault(opt, []).append((r, dl_max, lt_max + fq_max))

    for opt, pts in sorted(decomp_groups.items()):
        pts.sort()
        rs      = [p[0] for p in pts]
        dl_maxs = [p[1] for p in pts]
        sums    = [p[2] for p in pts]
        color  = OPT_COLORS[opt]
        marker = OPT_MARKERS[opt]
        name   = OPT_LABELS.get(opt, opt)
        ax_f.plot(rs, dl_maxs, color=color, linestyle="-", linewidth=LWb,
                  marker=marker, markersize=MS, markevery=1,
                  label=f"$\\Delta L_A^{{\\max}}$  {name}", zorder=4)
        ax_f.plot(rs, sums, color=color, linestyle="-.", linewidth=LW,
                  marker=marker, markersize=MS, markevery=1, alpha=0.75,
                  label=f"$(linear+Q_F)^{{\\max}}$  {name}", zorder=3)

    _style_triptych_ax_colored(
        ax_f, "value",
        r"(6) $\Delta L_A^{\max}$  vs  $(linear+Q_F)^{\max}$  [train]",
        r_ceil, title_color=TERM_GRAY, signed=True,
        legend_loc="upper left", legend_ncol=2,
    )

    if out_path is None:
        out_path = Path("r_sweep_full_data_1x6_worst.png")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_1x3_worst(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """1×3 worst-direction figure, no title.

    (a) ΔL_A          — max over K directions [train]
    (b) Training loss  — worst direction, base + loss_delta_max [train]
    (c) Accuracy       — mean [test only]
    """
    all_r: List[float] = []
    for _, rows in dirs_rows:
        for row in rows:
            rv = _float(row.get("radius", "nan"))
            if rv >= 0 and math.isfinite(rv):
                all_r.append(rv)
    r_ceil = r_max if r_max is not None else (max(all_r) if all_r else 1.0)

    abs_loss = _has_absolute_train_loss(dirs_rows)
    loss_ylabel = r"$L_A(w+\delta)^{\max}$" if abs_loss else r"$\Delta L_A^{\max}$"
    loss_title  = r"(b) Training loss — worst direction  [train]"

    LW  = 1.0
    LWb = 1.2
    MS  = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0),
                             gridspec_kw={"wspace": 0.35})

    ax_a, ax_b, ax_c = axes

    # ── (a) ΔL_A worst direction [train] ─────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_a, rows, "loss_delta_max", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LWb, markersize=MS)
    _style_triptych_ax_colored(
        ax_a, r"$\Delta L_A^{\max}$",
        r"(a) $\Delta L_A$ — worst direction  [train]",
        r_ceil, title_color=TERM_RED, signed=True, legend_loc="upper left",
    )

    # ── (b) Training loss worst direction [train] ─────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_b, rows, "train_loss_worst", r_max=r_max, tag=tag,
                    split_filter="train", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_b, loss_ylabel, loss_title,
                                r_ceil, title_color="black",
                                legend_loc="upper left")

    # ── (c) Accuracy — test only ──────────────────────────────────────────────
    for tag, rows in dirs_rows:
        _draw_lines(ax_c, rows, "perturbed_acc_mean", r_max=r_max, tag=tag,
                    split_filter="test", include_zero=True,
                    linewidth=LW, markersize=MS)
    _style_triptych_ax_colored(ax_c, "Accuracy (%)",
                                r"(c) Accuracy vs $r$  [test]",
                                r_ceil, title_color="black", y_min=82.0,
                                legend_loc="lower left")

    fig.tight_layout()

    if out_path is None:
        out_path = Path("r_sweep_1x3_worst.pdf")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CSV / Excel export ────────────────────────────────────────────────────────

# Ordered column definitions: (output_name, source_metric_or_virtual, description)
# "__lam_qf__" and "__lam_error__" are virtual: computed from stored fields + lambda.
EXPORT_COLS = [
    # ── metadata ──────────────────────────────────────────────────────────────
    ("optimizer",              None,              "sgd / gam"),
    ("split",                  None,              "train / test"),
    ("radius",                 None,              "perturbation radius r"),
    # ── accuracy ──────────────────────────────────────────────────────────────
    ("base_acc",               "base_acc",        "accuracy before perturbation (%)"),
    ("perturbed_acc_mean",     "perturbed_acc_mean", "mean accuracy after perturbation (%)"),
    ("perturbed_acc_std",      "perturbed_acc_std",  "std of accuracy over K directions (%)"),
    ("acc_delta_mean",         "acc_delta_mean",  "mean accuracy change = perturbed - base (%)"),
    ("acc_delta_std",          "acc_delta_std",   "std of accuracy change over K directions (%)"),
    # ── loss delta ────────────────────────────────────────────────────────────
    ("loss_delta_abs",         "loss_delta_abs",  "|ΔL_A| — mean absolute loss change"),
    ("loss_delta_signed",      "loss_delta_signed", "ΔL_A  — mean signed loss change"),
    # ── linear term ───────────────────────────────────────────────────────────
    ("linear_term_abs",        "linear_term_abs",  "|g_A^T(r·u)| — abs linear term"),
    ("linear_term_signed",     "linear_term_signed","g_A^T(r·u)  — signed linear term"),
    # ── Fisher quadratic ──────────────────────────────────────────────────────
    ("fisher_quad",            "fisher_quad",      "Q_F = 0.5·r²·u^T·F·u  (no lambda)"),
    ("lam_x_fisher_quad",      "__lam_qf__",       "λ·Q_F  (EWC penalty term, computed)"),
    # ── error: |ΔL_A − Q_F| ──────────────────────────────────────────────────
    ("stored_error",           "error",            "|ΔL_A − Q_F|   stored, no lambda"),
    ("stored_signed_error",    "signed_error",     "ΔL_A − Q_F     stored, signed, no lambda"),
    ("stored_relative_error",  "relative_error",   "relative error (stored, no lambda)"),
    # ── EWC error: |ΔL_A − λ·Q_F| ────────────────────────────────────────────
    ("ewc_error",              "__lam_error__",    "|ΔL_A − λ·Q_F|  EWC approx error (computed)"),
]


def export_summary(
    dirs_rows: List[Tuple[str, List[Dict]]],
    out_path: Path,
    lam: float = 1.0,
) -> Path:
    """Write a tidy CSV (and Excel if openpyxl is available) with all key metrics.

    Columns
    ───────
    optimizer, split, radius,
    base_acc, perturbed_acc_mean, acc_delta_mean,
    loss_delta_abs, loss_delta_signed,
    fisher_quad, lam_x_fisher_quad (=λ·Q_F),
    ewc_error (=|ΔL_A − λ·Q_F|), stored_error (=|ΔL_A − Q_F|),
    relative_error, linear_term_abs,
    [tag]   — only when multiple dirs are combined
    """
    use_tag = any(tag for tag, _ in dirs_rows)

    output_rows: List[Dict] = []
    for tag, rows in dirs_rows:
        for row in rows:
            r   = _float(row.get("radius", "nan"))
            opt = row.get("optimizer", "")
            spl = row.get("split", "train")
            if not math.isfinite(r):
                continue

            out: Dict = {}
            if use_tag:
                out["tag"] = tag
            out["optimizer"] = opt
            out["split"]     = spl
            out["radius"]    = r

            for col_name, src, _ in EXPORT_COLS:
                if col_name in ("optimizer", "split", "radius"):
                    continue
                if src == "__lam_qf__":
                    qf = _resolve_col(row, "fisher_quad")
                    val = lam * qf if math.isfinite(qf) else float("nan")
                elif src == "__lam_error__":
                    val = _lam_scaled_error(row, lam)
                elif src is None:
                    continue
                else:
                    val = _resolve_col(row, src)
                out[col_name] = round(val, 10) if math.isfinite(val) else ""

            output_rows.append(out)

    # Sort: tag → optimizer → split → radius
    output_rows.sort(key=lambda r: (
        r.get("tag", ""), r["optimizer"], r["split"], float(r["radius"])
    ))

    # ── write CSV ────────────────────────────────────────────────────────────
    fieldnames = (["tag"] if use_tag else []) + \
                 ["optimizer", "split", "radius"] + \
                 [c for c, _, _ in EXPORT_COLS
                  if c not in ("optimizer", "split", "radius")]
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(output_rows)
    print(f"[OK] wrote {csv_path}  ({len(output_rows)} rows)")

    # ── write Excel (optional) ───────────────────────────────────────────────
    xlsx_path = out_path.with_suffix(".xlsx")
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "r_sweep_summary"

        # Header style
        hdr_font  = Font(bold=True, color="FFFFFF")
        hdr_fill  = PatternFill("solid", fgColor="2F5496")
        hdr_align = Alignment(horizontal="center", wrap_text=True)
        thin = Side(style="thin", color="BBBBBB")
        border = Border(left=thin, right=thin, bottom=thin, top=thin)

        # Row colour by optimizer
        fill_sgd = PatternFill("solid", fgColor="DDEEFF")
        fill_gam = PatternFill("solid", fgColor="FFF3CD")

        ws.append(fieldnames)
        for cell in ws[1]:
            cell.font    = hdr_font
            cell.fill    = hdr_fill
            cell.alignment = hdr_align
            cell.border  = border

        for rdict in output_rows:
            row_vals = [rdict.get(f, "") for f in fieldnames]
            ws.append(row_vals)
            row_fill = fill_gam if rdict.get("optimizer") == "gam" else fill_sgd
            for cell in ws[ws.max_row]:
                cell.fill   = row_fill
                cell.border = border
                cell.alignment = Alignment(horizontal="right")

        # Auto-width
        for col_idx, col_name in enumerate(fieldnames, 1):
            max_len = max(
                len(str(col_name)),
                max((len(str(r.get(col_name, ""))) for r in output_rows), default=0),
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 22)

        # Freeze header row
        ws.freeze_panes = "A2"

        wb.save(xlsx_path)
        print(f"[OK] wrote {xlsx_path}")
    except ImportError:
        print("[skip] openpyxl not available — CSV only")

    return csv_path


# ── multi-Fisher variants figure ─────────────────────────────────────────────

MF_VARIANT_RE = re.compile(r"^(mf_[a-z0-9_]+)_fisher_quad$")


def _variant_prefixes(rows: List[Dict]) -> List[str]:
    """Collect multi-Fisher variant prefixes (e.g. 'mf_batch_empirical') from row keys."""
    out = set()
    for row in rows:
        for key in row:
            m = MF_VARIANT_RE.match(key)
            if m:
                out.add(m.group(1))
    return sorted(out)


def draw_multifisher_variants(
    dirs_rows: List[Tuple[str, List[Dict]]],
    r_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
) -> Optional[Path]:
    """1×3 comparison of all co-computed Fisher variants vs radius r.

      (a) Q_F per variant (log y), with |ΔL_A| in black as reference
      (b) FQA error |ΔL_A − Q_F| per variant (log y)
      (c) relative error |ΔL_A − Q_F| / |ΔL_A| per variant (log y)

    Colour = Fisher variant, line style = optimizer (SGD solid, GAM dashed).
    Returns None when no mf_* variant columns are present in the rows.
    """
    variants: List[str] = sorted({
        v for _, rows in dirs_rows for v in _variant_prefixes(rows)
    })
    if not variants:
        return None

    cmap = plt.get_cmap("tab10")
    var_colors = {v: cmap(i % 10) for i, v in enumerate(variants)}
    var_ls = {"sgd": "-", "gam": "--"}

    panels = [
        ("fisher_quad",    r"$Q_F$",
         r"(a) Fisher quad $Q_F$ per variant  [log]"),
        ("error",          r"$|\Delta L_A - Q_F|$",
         r"(b) FQA error $|\Delta L_A - Q_F|$  [log]"),
        ("relative_error", r"$|\Delta L_A - Q_F| / |\Delta L_A|$",
         r"(c) Relative error  [log]"),
    ]

    if suptitle is None:
        suptitle = "Multi-Fisher FQA: variant comparison vs perturbation radius $r$"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, (base, ylabel, title) in zip(axes, panels):
        for _, rows in dirs_rows:
            groups: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
            loss_ref: Dict[str, List[Tuple[float, float]]] = {}
            for row in rows:
                opt = row.get("optimizer", "").lower()
                if row.get("split", "train").lower() != "train":
                    continue
                r = _float(row.get("radius", "nan"))
                if not math.isfinite(r) or r <= 0:
                    continue
                if r_max is not None and r > r_max + 1e-9:
                    continue
                for v in variants:
                    y = _float(row.get(f"{v}_{base}", "nan"))
                    if math.isfinite(y) and y > 0:
                        groups.setdefault((v, opt), []).append((r, y))
                if base == "fisher_quad":
                    ld = _float(row.get("loss_delta_abs", "nan"))
                    if math.isfinite(ld) and ld > 0:
                        loss_ref.setdefault(opt, []).append((r, ld))

            for (v, opt), pts in sorted(groups.items()):
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=var_colors[v], linestyle=var_ls.get(opt, "-"),
                        linewidth=1.4, alpha=0.9,
                        label=f"{v[3:]} ({opt.upper()})", zorder=3)
            for opt, pts in sorted(loss_ref.items()):
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color="black", linestyle=var_ls.get(opt, "-"),
                        linewidth=1.8, alpha=0.85,
                        label=rf"$|\Delta L_A|$ ({opt.upper()})", zorder=4)

        ax.set_yscale("log")
        ax.set_xlabel("radius $r$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend on the right (panel (a) has the loss reference too).
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h_dedup, l_dedup = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h_dedup.append(h)
            l_dedup.append(l)
    fig.legend(h_dedup, l_dedup, loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=0.9,
               handletextpad=0.4, borderpad=0.5, labelspacing=0.35)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if out_path is None:
        out_path = Path("r_sweep_multifisher.png")
    fig.savefig(str(out_path), format=out_path.suffix.lstrip(".") or "png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path,
                   help="One or more fqa_partial dirs, summary CSVs, or cl_metrics JSONs")
    p.add_argument("--r-max", type=float, default=None,
                   help="Clip x-axis to this radius")
    p.add_argument("--zoom", type=float, default=5.0,
                   help="Also produce a zoomed plot up to this radius (default 5.0)")
    p.add_argument("--no-zoom", action="store_true",
                   help="Do not produce zoomed r<=zoom plots")
    p.add_argument("--lam", type=float, default=None,
                   help="EWC lambda for panel (c); auto-extracted from path if omitted")
    p.add_argument("--full-triptych", action="store_true",
                   help="Write full-range 1x3: accuracy / train loss / signed residual")
    p.add_argument("--full-five", action="store_true",
                   help="Write full-range 1x5: 1x3 plus linear term and Fisher quad")
    p.add_argument("--full-worst-five", action="store_true",
                   help="Write worst-case full-range 1x5 using per-direction maxima")
    p.add_argument("--triptych-name", default="r_sweep_full_data_1x3.pdf",
                   help="Filename for --full-triptych output")
    p.add_argument("--five-name", default="r_sweep_full_data_1x5.pdf",
                   help="Filename for --full-five output")
    p.add_argument("--worst-five-name", default="r_sweep_full_data_worst_1x5.pdf",
                   help="Filename for --full-worst-five output")
    p.add_argument("--error-scale", choices=("raw", "lambda"), default="raw",
                   help="For full-range panel (c): raw ΔL-Q_F or ΔL-λQ_F")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_font()

    # Load data from each directory
    use_tags = len(args.dirs) > 1
    dirs_rows: List[Tuple[str, List[Dict]]] = []
    for d in args.dirs:
        src = d.resolve()
        rows = load_input(src)
        augment_loss_fields(rows, src)
        augment_per_u_extrema(rows, src)
        tag = (d.stem if d.is_file() else d.name) if use_tags else ""
        dirs_rows.append((tag, rows))
        splits = sorted(set(r.get("split", "?") for r in rows))
        opts   = sorted(set(r.get("optimizer", "?") for r in rows))
        rvals  = sorted(set(_float(r.get("radius", "nan")) for r in rows if _float(r.get("radius","nan")) > 0))
        if rvals:
            r_desc = f"r∈[{min(rvals):.4g}, {max(rvals):.4g}]"
        else:
            r_desc = "r∈[n/a]"
        print(f"[{d.name}]  optimizers={opts}  splits={splits}  "
              f"n_radii={len(rvals)}  {r_desc}")

    # Primary output directory = first dir
    first = args.dirs[0].resolve()
    out_dir = first.parent if first.is_file() else first

    # Full-range plot
    out1 = draw_r_sweep(
        dirs_rows, r_max=args.r_max,
        out_path=out_dir / "r_sweep.pdf",
        suptitle="FQA metrics vs perturbation radius $r$",
    )
    print(f"[OK] wrote {out1}")

    # Zoomed full 2×3 plot
    if args.zoom is not None and not args.no_zoom:
        out2 = draw_r_sweep(
            dirs_rows, r_max=args.zoom,
            out_path=out_dir / f"r_sweep_r0to{int(args.zoom)}.pdf",
            suptitle=f"FQA metrics vs $r$  (zoom $r \\leq {args.zoom:.0f}$)",
        )
        print(f"[OK] wrote {out2}")

    # Resolve lambda: CLI wins, otherwise auto-extract from full path
    eff_lam = args.lam if args.lam is not None else _extract_lam_from_path(
        args.dirs[0].resolve())
    print(f"[lambda] using λ={eff_lam:.4g} for panel (c)")

    if args.full_triptych:
        out_full = draw_full_data_triptych(
            dirs_rows, r_max=args.r_max, lam=eff_lam,
            out_path=out_dir / args.triptych_name,
            error_scale=args.error_scale,
        )
        print(f"[OK] wrote {out_full}")

    if args.full_five:
        out_full5 = draw_full_data_1x5(
            dirs_rows, r_max=args.r_max, lam=eff_lam,
            out_path=out_dir / args.five_name,
            error_scale=args.error_scale,
        )
        print(f"[OK] wrote {out_full5}")

    if args.full_worst_five:
        out_worst5 = draw_full_data_worst_1x5(
            dirs_rows, r_max=args.r_max, lam=eff_lam,
            out_path=out_dir / args.worst_five_name,
            error_scale=args.error_scale,
        )
        print(f"[OK] wrote {out_worst5}")

    # 1×3 core figure: accuracy | loss | |ΔL − λ·Q_F|
    out3 = draw_1x3_core(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_1x3.pdf",
    )
    print(f"[OK] wrote {out3}")

    out3r = draw_1x3_residual(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_1x3_residual.pdf",
    )
    print(f"[OK] wrote {out3r}")

    out3rc = draw_1x3_residual(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_1x3_residual_color.pdf",
        opt_colors={"sgd": "#d62728", "gam": "#2ca02c"},
    )
    print(f"[OK] wrote {out3rc}")

    out4r = draw_1x4_residual(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_1x4_residual.pdf",
    )
    print(f"[OK] wrote {out4r}")

    if args.zoom is not None and not args.no_zoom:
        out4 = draw_1x3_core(
            dirs_rows, r_max=args.zoom, lam=eff_lam,
            out_path=out_dir / f"r_sweep_1x3_r0to{int(args.zoom)}.pdf",
        )
        print(f"[OK] wrote {out4}")

    # 1×6 decomposition figure (always produced)
    out6 = draw_full_data_1x6(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_full_data_1x6.png",
    )
    print(f"[OK] wrote {out6}")

    out3m = draw_1x3_mean(
        dirs_rows, r_max=args.r_max,
        out_path=out_dir / "r_sweep_1x3_mean.pdf",
    )
    print(f"[OK] wrote {out3m}")

    out6b = draw_full_data_1x6_band(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_full_data_1x6_band.png",
    )
    print(f"[OK] wrote {out6b}")

    out6w = draw_full_data_1x6_worst(
        dirs_rows, r_max=args.r_max, lam=eff_lam,
        out_path=out_dir / "r_sweep_full_data_1x6_worst.png",
    )
    print(f"[OK] wrote {out6w}")

    out3w = draw_1x3_worst(
        dirs_rows, r_max=args.r_max,
        out_path=out_dir / "r_sweep_1x3_worst.pdf",
    )
    print(f"[OK] wrote {out3w}")

    # Multi-Fisher variant comparison (only when mf_* columns are present)
    out_mf = draw_multifisher_variants(
        dirs_rows, r_max=args.r_max,
        out_path=out_dir / "r_sweep_multifisher.png",
    )
    if out_mf is not None:
        print(f"[OK] wrote {out_mf}")
        if args.zoom is not None and not args.no_zoom:
            out_mfz = draw_multifisher_variants(
                dirs_rows, r_max=args.zoom,
                out_path=out_dir / f"r_sweep_multifisher_r0to{int(args.zoom)}.png",
            )
            print(f"[OK] wrote {out_mfz}")

    # Export CSV + Excel
    export_summary(dirs_rows, out_path=out_dir / "r_sweep_summary", lam=eff_lam)


if __name__ == "__main__":
    main()
