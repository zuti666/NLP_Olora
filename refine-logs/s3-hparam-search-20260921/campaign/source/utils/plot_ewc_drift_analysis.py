"""
Plot EWC drift decomposition per task:
  (a) ||δ||   — parameter drift magnitude
  (b) δᵀFδ   — Fisher-weighted drift (accumulation detector)
  (c) λδᵀFδ  — EWC penalty (= loss_ewc * 2)

Shows the cancellation: F↑ as tasks accumulate, δ↓, product stays flat.
Collapsed runs show the balance breaking.

Usage:
  python utils/plot_ewc_drift_analysis.py \
      --log-dir logs/.../shell \
      [--log-dir logs/.../shell ...] \
      --out figures/ewc_drift.pdf
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _rainbow(lam, all_lams_nz):
    if not all_lams_nz:
        return "#333"
    lo, hi = np.log10(all_lams_nz[0]), np.log10(all_lams_nz[-1])
    if hi == lo:
        t = 0.5
    else:
        t = (np.log10(lam) - lo) / (hi - lo)
    return cm.rainbow(t)


def parse_epoch_dynamics(path, n_tasks=10):
    """
    Return dict:
      task_t -> {
        'delta_norm':           last-epoch ||δ||,
        'fisher_weighted_drift': last-epoch δᵀFδ,
        'loss_ewc':             last-epoch L_EWC,
        'loss_ce':              last-epoch L_CE,
        'ewc_ce_ratio':         last-epoch ratio,
        'train_acc_current':    last-epoch train acc on current task,
      }
    """
    epoch_data = {}   # task -> list of dicts, sorted by epoch
    with open(path) as f:
        for line in f:
            if "[EWConProbe][EpochDynamics]" not in line:
                continue
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group())
                t  = int(d.get("task", -1))
                ep = int(d.get("epoch", 0))
                if t < 0:
                    continue
                epoch_data.setdefault(t, []).append({
                    "ep":    ep,
                    "dn":    float(d.get("delta_norm", 0)),         # ||δ||  (or ||δ||²)
                    "fwd":   float(d.get("fisher_weighted_drift", 0)),  # δᵀFδ
                    "ewc":   float(d.get("loss_ewc", 0)),            # 0.5λδᵀFδ
                    "ce":    float(d.get("loss_ce", 1)),
                    "ratio": float(d.get("ewc_ce_ratio", 0)),
                    "acc":   float(d.get("train_acc_current", float("nan"))),
                })
            except Exception:
                pass

    result = {}
    for t, rows in epoch_data.items():
        rows_sorted = sorted(rows, key=lambda r: r["ep"])
        last = rows_sorted[-1]
        result[t] = last
    return result


def detect_collapse(task_data, n_tasks=10, ratio_thresh=5.0):
    """Task index where ratio first exceeds thresh, or None."""
    for t in range(1, n_tasks):
        r = task_data.get(t, {}).get("ratio", 0)
        if r > ratio_thresh or not np.isfinite(r):
            return t
    return None


def load_runs(log_dirs, lam_pattern=None):
    """
    Scan log_dirs for ewc_lam*.log and A0_noewc*.log files.
    Returns list of run dicts.
    """
    if lam_pattern is None:
        lam_pattern = re.compile(r"ewc_lam(\d+(?:\.\d+)?)_seed(\d+)\.log$")
    base_pattern = re.compile(r"A0_noewc_seed(\d+)\.log$")

    runs = []
    seen_lam = {}   # lam -> index in runs (dedup by lam, keep first)

    for log_dir in log_dirs:
        for fname in sorted(os.listdir(log_dir)):
            fpath = os.path.join(log_dir, fname)

            # baseline
            m = base_pattern.match(fname)
            if m:
                seed = int(m.group(1))
                if 0 not in seen_lam:
                    td = parse_epoch_dynamics(fpath)
                    runs.append(dict(lam=0, seed=seed, name="SGD (λ=0)",
                                     task_data=td, path=fpath))
                    seen_lam[0] = len(runs) - 1
                continue

            # EWC
            m = lam_pattern.match(fname)
            if m:
                lam  = float(m.group(1))
                seed = int(m.group(2))
                if lam not in seen_lam:
                    td = parse_epoch_dynamics(fpath)
                    ct = detect_collapse(td)
                    runs.append(dict(lam=lam, seed=seed, name=f"λ={int(lam):,}",
                                     task_data=td, collapse_task=ct, path=fpath))
                    seen_lam[lam] = len(runs) - 1

    runs.sort(key=lambda r: r["lam"])
    return runs


# ── plotting ─────────────────────────────────────────────────────────────────

TASK_IDS = list(range(1, 10))   # T0 has no EWC (no prior Fisher)


def _get_series(run, key, tasks=TASK_IDS):
    vals = []
    for t in tasks:
        v = run["task_data"].get(t, {}).get(key, float("nan"))
        if not np.isfinite(v) or abs(v) > 1e15:
            v = float("nan")
        vals.append(v)
    return np.array(vals, dtype=float)


def _truncate_at_collapse(ys, collapse_task):
    """Set values at and after collapse_task to NaN."""
    if collapse_task is None:
        return ys
    out = ys.copy()
    ct_local = collapse_task - 1   # TASK_IDS starts at T1, index = t-1
    out[ct_local:] = float("nan")
    return out


def plot_drift_analysis(runs, out_path, n_tasks=10):
    all_lams_nz = sorted(r["lam"] for r in runs if r["lam"] > 0)
    tasks = TASK_IDS   # 1..9

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax_dn, ax_fwd, ax_ewc = axes

    # legend handles collected
    handles, labels = [], []

    for run in runs:
        lam = run["lam"]
        is_baseline = lam == 0
        ct = run.get("collapse_task")

        if is_baseline:
            color = "#555"
            ls    = "--"
            lw    = 1.4
            zorder = 5
        else:
            color  = _rainbow(lam, all_lams_nz)
            ls     = "-"
            lw     = 1.1
            zorder = 3

        dn  = _get_series(run, "dn")
        fwd = _get_series(run, "fwd")
        ewc = _get_series(run, "ewc")

        # for collapsed runs: truncate valid region, then show × at collapse point
        dn_plot  = _truncate_at_collapse(dn,  ct)
        fwd_plot = _truncate_at_collapse(fwd, ct)
        ewc_plot = _truncate_at_collapse(ewc, ct)

        h_list = []
        for ax, ys in [(ax_dn, dn_plot), (ax_fwd, fwd_plot), (ax_ewc, ewc_plot)]:
            h, = ax.plot(tasks, ys, color=color, ls=ls, lw=lw, zorder=zorder,
                         label=run["name"])
            h_list.append(h)

            # mark collapse onset
            if ct is not None and ct in tasks:
                ci = ct - 1   # index in tasks list
                for series in [dn, fwd, ewc]:
                    v = series[ci] if ci < len(series) else float("nan")
                if ax is ax_dn:
                    v = dn[ci]  if ci < len(dn)  else float("nan")
                elif ax is ax_fwd:
                    v = fwd[ci] if ci < len(fwd) else float("nan")
                else:
                    v = ewc[ci] if ci < len(ewc) else float("nan")
                if np.isfinite(v):
                    ax.scatter([ct], [v], marker="x", s=80, color=color,
                               linewidths=1.8, zorder=6)

        handles.append(h_list[0])
        labels.append(run["name"])

    # ── axis labels ──
    ax_dn.set_title("(a)  ||δ||  (parameter drift)", fontsize=10, fontweight="bold")
    ax_dn.set_ylabel("||δ||", fontsize=10)

    ax_fwd.set_title("(b)  δᵀFδ  (Fisher-weighted drift)", fontsize=10, fontweight="bold")
    ax_fwd.set_ylabel("δᵀFδ", fontsize=10)

    ax_ewc.set_title("(c)  L_EWC = 0.5λδᵀFδ  (EWC penalty)", fontsize=10, fontweight="bold")
    ax_ewc.set_ylabel("L_EWC", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Task index t", fontsize=9)
        ax.set_xticks(tasks)
        ax.set_xticklabels([f"T{t}" for t in tasks], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # shared legend — two columns
    ncol = max(1, len(labels) // 7 + 1)
    fig.legend(handles, labels, loc="lower center",
               ncol=ncol, fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "EWC drift decomposition per task  (ImageNet-A, 10 tasks, ViT-B/16, SGD)\n"
        "F accumulates → δ constrained → L_EWC ≈ flat  (balance breaks → collapse)",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # ── print table summary ──
    print("\nPer-task summary (last epoch, selected λ):")
    print(f"{'λ':>8} | {'T':>3} | {'||δ||':>10} | {'δᵀFδ':>12} | {'L_EWC':>10} | {'ratio':>7} | collapse")
    print("-" * 72)
    select = [r for r in runs if r["lam"] in {0, 200, 500, 1000, 1500, 2000}]
    for run in select:
        ct = run.get("collapse_task")
        for t in range(1, 10):
            td = run["task_data"].get(t, {})
            dn  = td.get("dn",    float("nan"))
            fwd = td.get("fwd",   float("nan"))
            ewc = td.get("ewc",   float("nan"))
            rat = td.get("ratio", float("nan"))
            col_mark = f"← COLLAPSE" if (ct is not None and t >= ct) else ""
            if abs(dn) > 1e15 or abs(fwd) > 1e15:
                dn = fwd = ewc = float("inf")
            print(f"{int(run['lam']):>8} | T{t} | {dn:>10.4f} | {fwd:>12.4e} | {ewc:>10.4f} | {rat:>7.4f} {col_mark}")
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", dest="log_dirs", action="append", required=True,
                    metavar="DIR", help="Shell log directory (repeatable)")
    ap.add_argument("--out", default="figures/ewc_drift.pdf")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    runs = load_runs(args.log_dirs)
    if not runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(runs)} runs: " +
          ", ".join(f"λ={int(r['lam'])}" for r in runs))

    plot_drift_analysis(runs, args.out)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
