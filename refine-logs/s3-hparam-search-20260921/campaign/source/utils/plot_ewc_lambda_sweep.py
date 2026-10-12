"""
EWC lambda sweep diagnostic figures.

Figure 1 (--out):  raw accuracy three-panel
  (a) Final Acc vs λ  — uniform x-spacing, region colours
  (b) A_t trajectory  — rainbow by λ, legend
  (c) EWC/CE ratio    — rainbow by λ, legend

Figure 2 (--out-forgetting):  forgetting diagnosis three-panel
  (a) Final avg forgetting vs λ
  (b) Normalised retention trajectory  Ret_t^(λ)
  (c) Current-task acquisition ratio   C̃_t^(λ)

Usage:
    python utils/plot_ewc_lambda_sweep.py \\
        --nonorm-dir  logs/.../nonorm_10t/shell \\
        --nonorm-dir  logs/.../nonorm_10/shell  \\
        --baseline    logs/.../A0_noewc_seed0.log \\
        --out         figures/ewc_raw.pdf \\
        --out-forgetting figures/ewc_forgetting.pdf \\
        --select-all
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── colour helpers ───────────────────────────────────────────────────────────

REGION_COLOURS = {
    "weak":     "#9ecae1",
    "feasible": "#2ca02c",
    "risky":    "#ff7f0e",
    "collapse": "#d62728",
    "baseline": "#7f7f7f",
}


def _rainbow(lam, all_lams_nz):
    """Map λ → rainbow colour (log-normalised across the full λ range)."""
    logs = np.log10(np.array(all_lams_nz, dtype=float))
    vmin, vmax = logs.min(), logs.max()
    t = (np.log10(float(lam)) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return plt.get_cmap("rainbow")(float(np.clip(t, 0, 1)))


# ── log parsing ──────────────────────────────────────────────────────────────

def parse_cnn_totals(path):
    vals = []
    with open(path) as f:
        for line in f:
            if "=> CNN: {'total'" not in line:
                continue
            m = re.search(r"'total': np\.float64\(([0-9.]+)\)", line)
            if m:
                vals.append(float(m.group(1)))
    return vals[:10]


def parse_task_matrix(path):
    """mat[t][j] = acc on task j after training task t."""
    rows = []
    with open(path) as f:
        for line in f:
            if "=> CNN: {'total'" not in line:
                continue
            row = {}
            for m in re.finditer(
                r"'(\d{2}-\d{2})': np\.float64\(([0-9.]+|nan)\)", line
            ):
                row[m.group(1)] = (
                    float(m.group(2)) if m.group(2) != "nan" else float("nan")
                )
            rows.append(row)
    if not rows:
        return np.full((1, 1), float("nan"))
    T = len(rows)
    all_keys = sorted({k for r in rows for k in r})[:T]
    mat = np.full((T, T), float("nan"))
    for t, row in enumerate(rows):
        for j, key in enumerate(all_keys):
            mat[t, j] = row.get(key, float("nan"))
    return mat


def parse_ewc_ce_ratio(path):
    task_ratios: dict[int, list[float]] = {}
    with open(path) as f:
        for line in f:
            if "[EWConProbe][EpochDynamics]" not in line:
                continue
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group())
                t = int(d.get("task", 0))
                r = d.get("ewc_ce_ratio", 0)
                if t >= 1 and isinstance(r, (int, float)) and np.isfinite(r) and r < 1e8:
                    task_ratios.setdefault(t, []).append(float(r))
            except Exception:
                pass
    return task_ratios


# ── metric helpers ───────────────────────────────────────────────────────────

def _final_acc(totals):
    arr = [v for v in totals if not np.isnan(v)]
    return float(arr[-1]) if arr else float("nan")


def task_wise_avg_acc(mat):
    T = mat.shape[0]
    At = np.full(T, float("nan"))
    for t in range(T):
        row = mat[t, : t + 1]
        valid = row[~np.isnan(row)]
        if valid.size:
            At[t] = float(np.mean(valid))
    return At


def max_ewc_ce_per_task(task_ratios, n_tasks=10):
    out = np.zeros(n_tasks)
    for t, ratios in task_ratios.items():
        if 0 < t < n_tasks and ratios:
            out[t] = max(ratios)
    return out


def collapse_onset(totals, threshold=5.0, min_from=2):
    for i in range(min_from, len(totals)):
        if not np.isnan(totals[i]) and totals[i] < threshold:
            return i
    return None


# ── forgetting metrics ───────────────────────────────────────────────────────

def forgetting_matrix(mat):
    """F[t,j] = max_{k in [j,t-1]} R[k,j] - R[t,j]  (t > j)."""
    T = mat.shape[0]
    F = np.full((T, T), float("nan"))
    for t in range(1, T):
        for j in range(t):
            hist = np.nanmax(mat[j:t, j])
            cur  = mat[t, j]
            if np.isfinite(hist) and np.isfinite(cur):
                F[t, j] = hist - cur
    return F


def avg_forgetting_per_task(F_mat):
    """Ft = mean F[t, 0:t] for t = 1..T-1."""
    T = F_mat.shape[0]
    Ft = np.full(T, float("nan"))
    for t in range(1, T):
        row   = F_mat[t, :t]
        valid = row[np.isfinite(row)]
        if valid.size:
            Ft[t] = float(np.nanmean(valid))
    return Ft


def compute_final_forgetting(mat):
    """Mean forgetting across all old tasks at the very last step."""
    F = forgetting_matrix(mat)
    T = mat.shape[0]
    row   = F[T - 1, : T - 1]
    valid = row[np.isfinite(row)]
    return float(np.nanmean(valid)) if valid.size else float("nan")


def normalized_retention(mat, B_diag, eps=1e-6):
    """Ret_t = mean_j<t  R[t,j] / B[j]  (SGD-normalised)."""
    T = mat.shape[0]
    Ret = np.full(T, float("nan"))
    for t in range(1, T):
        vals = []
        for j in range(t):
            if np.isfinite(mat[t, j]) and np.isfinite(B_diag[j]) and B_diag[j] > 0:
                vals.append(mat[t, j] / (B_diag[j] + eps))
        if vals:
            Ret[t] = float(np.mean(vals))
    return Ret


def current_task_acq_ratio(mat, B_diag, eps=1e-6):
    """C̃_t = R[t,t] / B[t]  (SGD-normalised)."""
    T = mat.shape[0]
    Ct = np.full(T, float("nan"))
    for t in range(T):
        if np.isfinite(mat[t, t]) and np.isfinite(B_diag[t]) and B_diag[t] > 0:
            Ct[t] = mat[t, t] / (B_diag[t] + eps)
    return Ct


# ── load all runs ─────────────────────────────────────────────────────────────

def load_runs(log_dirs, baseline_path=None):
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]
    runs      = []
    seen_lams = set()

    # ── baseline (λ=0) ─────────────────────────────────────────────────────
    if baseline_path and os.path.exists(baseline_path):
        totals = parse_cnn_totals(baseline_path)
        mat    = parse_task_matrix(baseline_path)
        At     = task_wise_avg_acc(mat)
        runs.append(dict(
            lam=0, name="SGD (λ=0)",
            totals=totals, mat=mat, At=At,
            final_acc=_final_acc(totals),
            aac=float(np.nanmean(At)),   # AAA = (1/T) Σ A_t
            ratio_per_task={}, max_r_per_task=np.zeros(10),
            collapse_task=None, path=baseline_path,
        ))

    # ── EWC runs ───────────────────────────────────────────────────────────
    for log_dir in log_dirs:
        if not log_dir or not os.path.isdir(log_dir):
            continue
        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith("_seed0.log"):
                continue
            m = re.search(r"ewc_lam(\d+)_seed0\.log", fname)
            if not m:
                continue
            lam = int(m.group(1))
            if lam in seen_lams:
                continue
            seen_lams.add(lam)
            path   = os.path.join(log_dir, fname)
            totals = parse_cnn_totals(path)
            if not totals:
                continue
            mat    = parse_task_matrix(path)
            At     = task_wise_avg_acc(mat)
            ratios = parse_ewc_ce_ratio(path)
            max_r  = max_ewc_ce_per_task(ratios, n_tasks=10)
            runs.append(dict(
                lam=lam, name=f"λ={lam}",
                totals=totals, mat=mat, At=At,
                final_acc=_final_acc(totals),
                aac=float(np.nanmean(At)),   # AAA = (1/T) Σ A_t
                ratio_per_task=ratios, max_r_per_task=max_r,
                collapse_task=collapse_onset(totals),
                path=path,
            ))

    runs.sort(key=lambda r: r["lam"])
    return runs


def enrich_forgetting(runs):
    """Add forgetting/retention/acquisition fields to each run (in-place).
    Requires baseline (lam=0) to be present for SGD-normalised metrics."""
    baseline = next((r for r in runs if r["lam"] == 0), None)
    B_diag   = (np.diag(baseline["mat"])
                 if baseline is not None
                 else np.full(10, float("nan")))

    for r in runs:
        F_mat          = forgetting_matrix(r["mat"])
        r["F_mat"]     = F_mat
        r["Ft"]        = avg_forgetting_per_task(F_mat)
        r["F_final"]   = compute_final_forgetting(r["mat"])
        r["Ret_t"]     = normalized_retention(r["mat"], B_diag)
        r["Ct_ratio"]  = current_task_acq_ratio(r["mat"], B_diag)


# ── region classifier ─────────────────────────────────────────────────────────

def classify_region(run):
    lam   = run["lam"]
    ct    = run["collapse_task"]
    max_r = float(np.max(run["max_r_per_task"]))
    if lam  == 0:            return "baseline"
    if ct is not None:       return "collapse"
    if max_r < 0.04:         return "weak"
    if max_r <= 0.20:        return "feasible"
    return "risky"


# ── shared axis helpers ───────────────────────────────────────────────────────

def _uniform_lam_axis(ax, all_lams_nz, x_positions, rotate=45):
    """Replace linear/log x-axis with uniform spacing; labels = actual λ values."""
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(l) for l in all_lams_nz],
                       rotation=rotate, ha="right", fontsize=7)
    ax.set_xlim(-0.6, len(all_lams_nz) - 0.4)


def _legend_2col(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = max(1, len(handles) // 8)
    ax.legend(handles, labels, fontsize=7, ncol=ncol,
              loc="best", framealpha=0.7, **kw)


# ── Figure 1: raw accuracy ────────────────────────────────────────────────────

def plot_lambda_sensitivity(ax, runs):
    """(a) AAA = (1/T)Σ A_t vs λ — uniform x-spacing, region colours."""
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)
    x_pos       = list(range(len(all_lams_nz)))
    lam2x       = {l: i for i, l in enumerate(all_lams_nz)}

    # SGD baseline horizontal line (AAA)
    for r in runs:
        if r["lam"] == 0:
            ax.axhline(r["aac"], color=REGION_COLOURS["baseline"],
                       linestyle="--", linewidth=1.2,
                       label=f"SGD ({r['aac']:.1f}%)", zorder=2)
            break

    # grey line through non-collapsed points
    stable_x = [lam2x[r["lam"]] for r in runs
                 if r["lam"] > 0 and classify_region(r) != "collapse"]
    stable_y = [r["aac"] for r in runs
                if r["lam"] > 0 and classify_region(r) != "collapse"]
    if stable_x:
        ax.plot(stable_x, stable_y, color="#cccccc", linewidth=1.0, zorder=1)

    region_plotted = set()
    label_map = {
        "weak":     "Weak (EWC ≈ 0)",
        "feasible": "Non-collapse",
        "risky":    "Risky",
        "collapse": "Collapse",
    }
    for r in runs:
        if r["lam"] == 0:
            continue
        region      = classify_region(r)
        is_collapse = region == "collapse"
        label = None
        if region not in region_plotted:
            label = label_map.get(region)
            region_plotted.add(region)
        ax.scatter([lam2x[r["lam"]]], [r["aac"]],
                   c=[REGION_COLOURS.get(region, "#333")],
                   marker="x" if is_collapse else "o",
                   s=140 if is_collapse else 70,
                   linewidths=2.5 if is_collapse else 0.5,
                   zorder=4, label=label,
                   edgecolors="k" if not is_collapse else "none")

    _uniform_lam_axis(ax, all_lams_nz, x_pos)
    ax.set_xlabel("λ", fontsize=10)
    ax.set_ylabel(r"AAA = $\frac{1}{T}\sum_t A_t$ (%)", fontsize=10)
    ax.set_title("(a) λ Sensitivity (AAA)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_task_wise_acc(ax, runs, select_lams):
    """(b) A_t vs t — rainbow, legend, collapsed runs stop at ×."""
    task_ids    = np.arange(10)
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)

    for r in runs:
        if r["lam"] not in select_lams:
            continue
        if r["lam"] == 0:
            ax.plot(task_ids[:len(r["At"])], r["At"],
                    marker="o", markersize=3, linewidth=1.4,
                    color=REGION_COLOURS["baseline"], linestyle="--",
                    label="SGD (λ=0)", zorder=3)
            continue
        color = _rainbow(r["lam"], all_lams_nz)
        At, n, ct = r["At"], len(r["At"]), r["collapse_task"]
        if ct is not None and ct < n:
            ax.plot(task_ids[:ct + 1], At[:ct + 1],
                    marker="o", markersize=3, linewidth=1.5,
                    color=color, label=r["name"], zorder=3)
            ax.scatter([ct], [At[ct]], marker="X", s=120, color=color,
                       zorder=5, edgecolors="k", linewidths=0.8)
        else:
            ax.plot(task_ids[:n], At, marker="o", markersize=3,
                    linewidth=1.5, color=color, label=r["name"], zorder=3)

    ax.set_xlabel("Task t", fontsize=10)
    ax.set_ylabel("A_t (%)", fontsize=10)
    ax.set_title("(b) Task-wise Avg Accuracy", fontsize=11, fontweight="bold")
    ax.set_xticks(task_ids)
    _legend_2col(ax)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_ewc_ce_ratio(ax, runs, select_lams):
    """(c) max EWC/CE ratio per task — rainbow, legend, log scale."""
    task_ids    = np.arange(1, 10)
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)

    for r in runs:
        if r["lam"] not in select_lams or r["lam"] == 0:
            continue
        color  = _rainbow(r["lam"], all_lams_nz)
        max_r  = r["max_r_per_task"][1:].copy()
        ct     = r["collapse_task"]
        if ct is not None:
            max_r[ct - 1:] = np.nan
        valid = np.where(max_r > 0, max_r, np.nan)
        ax.plot(task_ids, valid, marker="o", markersize=3,
                linewidth=1.5, color=color, label=r["name"], zorder=3)

    ax.axhline(1.0, color="red",    linestyle="--", linewidth=1.0,
               label="r=1 (EWC=CE)", zorder=1)
    ax.axhline(0.1, color="orange", linestyle=":",  linewidth=0.8,
               label="r=0.1", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Task t", fontsize=10)
    ax.set_ylabel("Max EWC/CE ratio (log)", fontsize=10)
    ax.set_title("(c) EWC-to-CE Ratio", fontsize=11, fontweight="bold")
    ax.set_xticks(task_ids)
    _legend_2col(ax)
    ax.grid(True, alpha=0.3, which="both")
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# ── Figure 2: forgetting diagnosis ───────────────────────────────────────────

def plot_final_forgetting_vs_lambda(ax, runs):
    """(a) Final average forgetting F_final vs λ.
    Collapsed runs shown in grey and excluded from the trend line —
    their low forgetting is artefactual (plasticity destroyed, not retention).
    """
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)
    x_pos       = list(range(len(all_lams_nz)))
    lam2x       = {l: i for i, l in enumerate(all_lams_nz)}

    # SGD baseline
    for r in runs:
        if r["lam"] == 0:
            v = r.get("F_final", float("nan"))
            if np.isfinite(v):
                ax.axhline(v, color=REGION_COLOURS["baseline"],
                           linestyle="--", linewidth=1.2,
                           label=f"SGD ({v:.1f}%)", zorder=2)
            break

    # trend line through non-collapsed points only
    stable_x = [lam2x[r["lam"]] for r in runs
                 if r["lam"] > 0 and classify_region(r) != "collapse"]
    stable_y = [r.get("F_final", float("nan")) for r in runs
                if r["lam"] > 0 and classify_region(r) != "collapse"]
    if stable_x:
        ax.plot(stable_x, stable_y, color="#cccccc", linewidth=1.0, zorder=1)

    label_map = {
        "weak":     "Weak (EWC ≈ 0)",
        "feasible": "Non-collapse",
        "risky":    "Risky",
        "collapse": "Collapse (invalid — plasticity destroyed)",
    }
    region_plotted = set()
    for r in runs:
        if r["lam"] == 0:
            continue
        region      = classify_region(r)
        is_collapse = region == "collapse"
        label = None
        if region not in region_plotted:
            label = label_map.get(region)
            region_plotted.add(region)
        fval = r.get("F_final", float("nan"))
        # collapsed runs: grey hollow ×, excluded from reading
        if is_collapse:
            ax.scatter([lam2x[r["lam"]]], [fval],
                       c=["#bbbbbb"], marker="x", s=100,
                       linewidths=1.5, zorder=3, label=label,
                       alpha=0.6)
        else:
            ax.scatter([lam2x[r["lam"]]], [fval],
                       c=[REGION_COLOURS.get(region, "#333")],
                       marker="o", s=70, linewidths=0.5,
                       zorder=4, label=label, edgecolors="k")

    _uniform_lam_axis(ax, all_lams_nz, x_pos)
    ax.set_xlabel("λ", fontsize=10)
    ax.set_ylabel("Final Avg Forgetting (%)", fontsize=10)
    ax.set_title("(a) Final Forgetting vs λ", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_normalized_retention(ax, runs, select_lams):
    """(b) Ret_t^(λ) vs t — normalised retention, rainbow, legend."""
    task_ids    = np.arange(1, 10)
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)

    for r in runs:
        if r["lam"] not in select_lams:
            continue
        Ret = r.get("Ret_t", np.full(10, float("nan")))
        ct  = r["collapse_task"]
        if r["lam"] == 0:
            ax.plot(task_ids, Ret[1:], marker="o", markersize=3,
                    linewidth=1.4, color=REGION_COLOURS["baseline"],
                    linestyle="--", label="SGD (λ=0)", zorder=3)
            continue
        color = _rainbow(r["lam"], all_lams_nz)
        vals  = Ret[1:].copy()
        if ct is not None:
            vals[ct - 1:] = np.nan
        ax.plot(task_ids, vals, marker="o", markersize=3,
                linewidth=1.5, color=color, label=r["name"], zorder=3)
        if ct is not None and ct >= 1 and ct < 10:
            ax.scatter([ct], [Ret[ct]], marker="X", s=120, color=color,
                       zorder=5, edgecolors="k", linewidths=0.8)

    ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.9,
               label="perfect retention", zorder=1)
    ax.set_xlabel("Task t", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Ret}_t = \frac{1}{t}\sum_{j<t} R_{t,j}/R_{j,j}^{\mathrm{SGD}}$",
                  fontsize=9)
    ax.set_title("(b) Normalised Retention", fontsize=11, fontweight="bold")
    ax.set_xticks(task_ids)
    _legend_2col(ax)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_current_task_acq(ax, runs, select_lams):
    """(c) C̃_t^(λ) = R[t,t]^(λ) / R[t,t]^(SGD) vs t — rainbow, legend."""
    task_ids    = np.arange(10)
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)

    for r in runs:
        if r["lam"] not in select_lams:
            continue
        Ct = r.get("Ct_ratio", np.full(10, float("nan")))
        ct = r["collapse_task"]
        if r["lam"] == 0:
            ax.plot(task_ids, Ct, marker="o", markersize=3,
                    linewidth=1.4, color=REGION_COLOURS["baseline"],
                    linestyle="--", label="SGD (λ=0)", zorder=3)
            continue
        color = _rainbow(r["lam"], all_lams_nz)
        vals  = Ct.copy()
        if ct is not None:
            vals[ct:] = np.nan
        ax.plot(task_ids, vals, marker="o", markersize=3,
                linewidth=1.5, color=color, label=r["name"], zorder=3)
        if ct is not None and ct < 10:
            ax.scatter([ct], [Ct[ct]], marker="X", s=120, color=color,
                       zorder=5, edgecolors="k", linewidths=0.8)

    ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.9,
               label="SGD parity", zorder=1)
    ax.set_xlabel("Task t", fontsize=10)
    ax.set_ylabel(r"$\tilde{C}_t = R_{t,t}^{(\lambda)}/R_{t,t}^{\mathrm{SGD}}$",
                  fontsize=9)
    ax.set_title("(c) Current-task Acquisition Ratio", fontsize=11,
                 fontweight="bold")
    ax.set_xticks(task_ids)
    _legend_2col(ax)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonorm-dir", action="append", dest="nonorm_dirs",
                    metavar="DIR", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default="figures/ewc_lambda_sweep.pdf")
    ap.add_argument("--out-forgetting", default=None,
                    help="if set, also save forgetting-diagnosis figure here")
    ap.add_argument("--select-task-curves", default=None)
    ap.add_argument("--select-all", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = load_runs(args.nonorm_dirs, baseline_path=args.baseline)
    if not runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(runs)} runs: " + ", ".join(f"λ={r['lam']}" for r in runs))

    enrich_forgetting(runs)

    if args.select_all or args.select_task_curves is None:
        select_lams = set(r["lam"] for r in runs)
    else:
        select_lams = {int(x) for x in args.select_task_curves.split(",")}
    print(f"Panel b/c λ: {sorted(select_lams)}")

    # ── Figure 1: raw accuracy ───────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))
    fig1.suptitle(
        "Standard EWC on ImageNet-A  (10 tasks, ViT-B/16, SGD full fine-tuning)\n"
        "Narrow non-collapse λ window: weak → EWC ≈ 0  |  strong → collapse",
        fontsize=10,
    )
    plot_lambda_sensitivity(axes1[0], runs)
    plot_task_wise_acc(axes1[1], runs, select_lams)
    plot_ewc_ce_ratio(axes1[2], runs, select_lams)
    fig1.tight_layout()
    out1 = Path(args.out)
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved Fig1 → {out1}")

    # ── Figure 2: forgetting diagnosis ──────────────────────────────────────
    if args.out_forgetting:
        fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
        fig2.suptitle(
            "EWC forgetting diagnosis  (ImageNet-A, normalised by SGD task-diagonal)\n"
            "Grey × = collapsed runs (low forgetting artefactual — plasticity destroyed, not retention)  |  "
            "Acquisition < 1 = EWC suppresses current-task learning",
            fontsize=9,
        )
        plot_final_forgetting_vs_lambda(axes2[0], runs)
        plot_normalized_retention(axes2[1], runs, select_lams)
        plot_current_task_acq(axes2[2], runs, select_lams)
        fig2.tight_layout()
        out2 = Path(args.out_forgetting)
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved Fig2 → {out2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
