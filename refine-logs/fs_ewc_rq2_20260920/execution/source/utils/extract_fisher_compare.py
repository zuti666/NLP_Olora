#!/usr/bin/env python3
"""
Extract Fisher / AAA / Task-0 accuracy from cl_metrics JSON files,
save to CSV, and plot a 3-row combined figure.

Usage:
    python utils/extract_fisher_compare.py \
        --sgd-json <path/to/sgd_cl_metrics.json> \
        --gam-json <path/to/gam_cl_metrics.json> \
        --out-dir  result_summary/fisher_compare/<TAG>

Auto-discovery shortcut (pass trainer_outputs root):
    python utils/extract_fisher_compare.py \
        --trainer-outputs outputs_logs/sgd_gam_s1c/stage2_T_compare/<TAG>/trainer_outputs \
        --out-dir result_summary/fisher_compare/<TAG>
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── JSON loading ───────────────────────────────────────────────────────────────

def find_json(trainer_outputs_root: str, optimizer: str) -> str:
    """Find cl_metrics.json for a given optimizer under trainer_outputs."""
    pattern = os.path.join(
        trainer_outputs_root, "**",
        f"*{optimizer}*", "**", "*cl_metrics.json"
    )
    hits = glob.glob(pattern, recursive=True)
    # prefer the path that contains the optimizer name in its directory
    hits = [h for h in hits if optimizer in h]
    if not hits:
        raise FileNotFoundError(
            f"No cl_metrics.json found for optimizer={optimizer!r} under {trainer_outputs_root}"
        )
    return sorted(hits)[0]


def load_metrics(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ── data extraction ────────────────────────────────────────────────────────────

FISHER_KEYS = [
    "train_clean_loss_mean",
    "train_fisher_loss_mean",
    "train_gam_mean",
    "train_ewc_gam_ratio_mean",
    "delta_norm",
    "fisher_weighted_drift",
    "normalized_fisher_drift",
    "ewc_penalty_value",
    "high_fisher_delta_energy_ratio",
    "high_fisher_weighted_energy_ratio",
    "full_fisher_fro",
    "accum_fisher_fro",
    "full_vs_accum_cosine_sim",
    "full_vs_accum_rel_fro",
    "full_vs_accum_rel_fro_wrt_full",
    "full_vs_accum_max_abs_diff",
    "full_fisher_trace",
    "accum_fisher_trace",
    "full_fisher_top1pct_trace",
    "accum_fisher_top1pct_by_full",
    "full_vs_accum_top1pct_accum_coverage",
    "full_fisher_num_samples",
]


def extract_fisher(d: dict) -> dict:
    """Return {task_int: {metric: value}} from mechanism.steps."""
    out = {}
    steps = d.get("mechanism", {}).get("steps", {})
    for step_str, v in steps.items():
        task = int(step_str)
        out[task] = {k: v.get(k) for k in FISHER_KEYS}
    return out


def extract_aaa_per_step(d: dict) -> list:
    """Return list[float] of average-accuracy-so-far at each step (from nme)."""
    return d.get("nme", {}).get("final", {}).get("vectors", {}).get("prefix_mean_t", [])


TASK0_ACC_FIELDS = [
    "test_local_acc",
    "train_local_acc",
    "test_full_acc",
    "train_full_acc",
    "delta_norm_sq",
    "delta_norm",
    "fisher_raw_quadratic",
    "fisher_quad",
    "lambda_fisher_quad",
    "normalized_fisher_drift",
    "step_delta_sq",
    "cum_delta0_sq",
    "step_delta_F0_loaded_anchor_delta",
    "step_delta_F0_anchor_delta",
    "step_delta_F0_dynamic_prev_delta",
    "step_delta_F0_dynamic_current_delta",
    "step_delta_Facc_delta",
    "cum_delta0_F0_loaded_anchor_delta0",
    "cum_delta0_F0_anchor_delta0",
    "cum_delta0_F0_dynamic_delta0",
    "cum_delta0_Facc_delta0",
    "norm_cum_delta0_F0_loaded_anchor_delta0",
    "norm_cum_delta0_F0_anchor_delta0",
    "norm_cum_delta0_F0_dynamic_delta0",
    "norm_cum_delta0_Facc_delta0",
    "task0_fisher_eval",
    "task0_fisher_current_num_samples",
    "task0_fisher_anchor_num_samples",
    "task0_fisher_current_trace",
    "task0_fisher_current_fro",
    "task0_fisher_current_l1",
    "task0_fisher_anchor_cosine_sim",
    "task0_fisher_anchor_mismatch",
    "task0_fisher_anchor_rel_fro_wrt_ref",
    "task0_fisher_anchor_top1pct_ref_coverage",
    "task0_fisher_loaded_anchor_cosine_sim",
    "task0_fisher_loaded_anchor_mismatch",
    "task0_fisher_accum_cosine_sim",
    "task0_fisher_accum_mismatch",
    "task0_fisher_accum_rel_fro_wrt_ref",
    "task0_fisher_accum_top1pct_ref_coverage",
    "test_local_loss_delta",
    "test_full_loss_delta",
]

def extract_task0_acc(d: dict) -> dict:
    """Return {step_int: {field: value}} from task0_trajectory.steps."""
    out = {}
    steps = d.get("task0_trajectory", {}).get("steps", {})
    for step_str, v in steps.items():
        out[int(step_str)] = {f: v.get(f) for f in TASK0_ACC_FIELDS}
    return out


def extract_final_metrics(d: dict) -> dict:
    nme_final = d.get("nme", {}).get("final", {})
    return {
        "AAA":          nme_final.get("AAA"),
        "Acc":          nme_final.get("Acc"),
        "Forget_avg":   nme_final.get("Forget_avg"),
        "ABWT":         nme_final.get("ABWT"),
    }


# ── CSV building ───────────────────────────────────────────────────────────────

def build_rows(optimizer, fisher, aaa_list, task0_acc, final, dynamics=None):
    dyn_tasks = set(dynamics.get("tasks", [])) if dynamics else set()
    all_tasks = sorted(set(fisher) | set(task0_acc) | dyn_tasks)
    rows = []
    for t in all_tasks:
        row = {"optimizer": optimizer, "task": t}
        fd = fisher.get(t, {})
        for k in FISHER_KEYS:
            row[k] = fd.get(k, "")
        delta_norm = fd.get("delta_norm")
        cosine = fd.get("full_vs_accum_cosine_sim")
        # mechanism.steps[].delta_norm is logged as the squared step drift
        # for the current LoRA Delta-W learner.
        row["delta_norm_sq"] = delta_norm if _is_num(delta_norm) else ""
        row["delta_norm_l2"] = math.sqrt(delta_norm) if _is_num(delta_norm) and delta_norm >= 0 else ""
        row["fisher_directional_mismatch"] = 1.0 - cosine if _is_num(cosine) else ""
        # AAA: index 0 = task-0 result (after training task 0)
        row["aaa_at_step"] = aaa_list[t] if t < len(aaa_list) else ""
        t0v = task0_acc.get(t, {})
        for f in TASK0_ACC_FIELDS:
            row[f"task0_{f}"] = t0v.get(f, "") if isinstance(t0v, dict) else ""
        if dynamics:
            for k in ["A_t", "O_t", "F_t", "B_t"]:
                vals = dynamics.get(k, [])
                row[k] = vals[t] if t < len(vals) else ""
        for k, v in final.items():
            row[f"final_{k}"] = v
        rows.append(row)
    return rows


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] saved → {path}")


# ── plotting ───────────────────────────────────────────────────────────────────

OPT_COLORS = {
    "sgd": "#E05C5C",
    "gam": "#4A90D9",
}
FISHER_STYLES = {
    "full_fisher_fro":  {"linestyle": "-",  "marker": "o", "label": "Full Fisher"},
    "accum_fisher_fro": {"linestyle": "--", "marker": "s", "label": "Accum Fisher"},
}


def _xticks(tasks):
    return sorted(set(tasks))


def plot_combined(all_data: dict, out_dir: str):
    """
    3-row figure:
      Row 1 – Fisher Frobenius norm (full vs accum, SGD vs GAM)
      Row 2 – AAA (average accuracy on all seen tasks) per step
      Row 3 – Task-0 test_local_acc per step
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

    # ── row 1: Fisher Fro ────────────────────────────────────────────────────
    ax = axes[0]
    all_tasks = []
    for opt, data in all_data.items():
        tasks = sorted(data["fisher"])
        all_tasks.extend(tasks)
        color = OPT_COLORS.get(opt, "gray")
        for metric, style in FISHER_STYLES.items():
            vals = [data["fisher"][t].get(metric) for t in tasks]
            ax.plot(
                tasks, vals,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8, markersize=5,
                label=f"{opt.upper()} – {style['label']}",
            )
    ax.set_ylabel("Fisher Frobenius norm", fontsize=11)
    ax.set_title("Full Fisher vs Accumulated Fisher (Frobenius norm)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(_xticks(all_tasks))

    # ── row 2: AAA per step ───────────────────────────────────────────────────
    ax = axes[1]
    for opt, data in all_data.items():
        aaa = data["aaa"]           # list indexed by task
        tasks = list(range(len(aaa)))
        final_aaa = data["final"].get("AAA")
        label = f"{opt.upper()} (final AAA={final_aaa:.1f}%)" if final_aaa else opt.upper()
        ax.plot(
            tasks, aaa,
            color=OPT_COLORS.get(opt, "gray"),
            linestyle="-", marker="o",
            linewidth=1.8, markersize=5,
            label=label,
        )
    ax.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax.set_title("AAA – Average Accuracy on All Seen Tasks", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(len(aaa))))

    # ── row 3: Task-0 local accuracy (test + train) ───────────────────────────
    ax = axes[2]
    _plot_task0_local(ax, all_data)
    ax.set_xlabel("Task index (T)", fontsize=11)

    path = os.path.join(out_dir, "fisher_aaa_task0_combined.png")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


def plot_fisher_only(all_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for opt, data in all_data.items():
        tasks = sorted(data["fisher"])
        color = OPT_COLORS.get(opt, "gray")
        for metric, style in FISHER_STYLES.items():
            vals = [data["fisher"][t].get(metric) for t in tasks]
            ax.plot(tasks, vals, color=color,
                    linestyle=style["linestyle"], marker=style["marker"],
                    linewidth=1.8, markersize=5,
                    label=f"{opt.upper()} – {style['label']}")
    ax.set_xlabel("Task index (T)", fontsize=12)
    ax.set_ylabel("Fisher Frobenius norm", fontsize=12)
    ax.set_title("Full Fisher vs Accumulated Fisher (Frobenius norm)\nSGD vs GAM", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fisher_fro_compare.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


def plot_aaa(all_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for opt, data in all_data.items():
        aaa = data["aaa"]
        tasks = list(range(len(aaa)))
        final_aaa = data["final"].get("AAA")
        label = f"{opt.upper()} (final={final_aaa:.1f}%)" if final_aaa else opt.upper()
        ax.plot(tasks, aaa, color=OPT_COLORS.get(opt, "gray"),
                linestyle="-", marker="o", linewidth=1.8, markersize=5, label=label)
    ax.set_xlabel("Task index (T)", fontsize=12)
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.set_title("AAA – Average Accuracy on All Seen Tasks (SGD vs GAM)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "aaa_compare.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


# line-style convention:  test=solid/circle  train=dashed/square
SPLIT_STYLES = {
    "test":  {"linestyle": "-",  "marker": "o"},
    "train": {"linestyle": "--", "marker": "s"},
}


def _add_baseline(ax, all_data, field):
    for data in all_data.values():
        v = data["task0"].get(0, {}).get(field)
        if v is not None:
            ax.axhline(v, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
                       label=f"Baseline = {v:.1f}%")
            break


def _plot_task0_local(ax, all_data):
    """Plot test_local_acc (solid) and train_local_acc (dashed) for each optimizer."""
    for opt, data in all_data.items():
        color = OPT_COLORS.get(opt, "gray")
        t0 = data["task0"]
        tasks = sorted(t0)
        for split, field in [("test", "test_local_acc"), ("train", "train_local_acc")]:
            vals = [t0[t].get(field) for t in tasks]
            st = SPLIT_STYLES[split]
            ax.plot(tasks, vals, color=color,
                    linestyle=st["linestyle"], marker=st["marker"],
                    linewidth=1.8, markersize=5,
                    label=f"{opt.upper()} – {split}")
    _add_baseline(ax, all_data, "test_local_acc")
    ax.set_ylabel("Task-0 accuracy (%)", fontsize=11)
    ax.set_title("Task-0 Local Head Accuracy (test=solid / train=dashed)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    all_tasks = sorted({t for data in all_data.values() for t in data["task0"]})
    ax.set_xticks(all_tasks)


def plot_task0(all_data: dict, out_dir: str):
    """Fig: test_local_acc + train_local_acc for SGD and GAM."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _plot_task0_local(ax, all_data)
    ax.set_xlabel("Task index (T)", fontsize=12)
    path = os.path.join(out_dir, "task0_acc_trajectory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


def plot_fisher_divergence(all_data: dict, out_dir: str):
    """
    3-row figure showing divergence between Full Fisher and Accumulated Fisher:
      Row 1 – relative Frobenius distance  (||full-accum||/||accum||)
      Row 2 – cosine similarity            (1 = same direction)
      Row 3 – top-1% coverage             (how much of full's top params accum covers)
    """
    DIVG_METRICS = [
        ("full_vs_accum_rel_fro",               "Relative L2 distance\n‖full−accum‖/‖accum‖",  False),
        ("full_vs_accum_cosine_sim",             "Cosine similarity\n(direction alignment)",      True),
        ("full_vs_accum_top1pct_accum_coverage", "Top-1% coverage\n(accum covers full's top 1%)", True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    for ax, (metric, ylabel, draw_ref) in zip(axes, DIVG_METRICS):
        all_tasks_seen = []
        for opt, data in all_data.items():
            tasks = sorted(data["fisher"])
            vals  = [data["fisher"][t].get(metric) for t in tasks]
            ax.plot(tasks, vals,
                    color=OPT_COLORS.get(opt, "gray"),
                    linestyle="-", marker="o",
                    linewidth=1.8, markersize=5,
                    label=opt.upper())
            all_tasks_seen.extend(tasks)
        if draw_ref:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(set(all_tasks_seen)))

    axes[0].set_title("Full Fisher vs Accumulated Fisher – Divergence Metrics (SGD vs GAM)", fontsize=12)
    axes[2].set_xlabel("Task index (T)", fontsize=11)

    path = os.path.join(out_dir, "fisher_divergence.png")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


def plot_task0_full(all_data: dict, out_dir: str):
    """Fig: test_full_acc + train_full_acc (joint head) for SGD and GAM."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for opt, data in all_data.items():
        color = OPT_COLORS.get(opt, "gray")
        t0 = data["task0"]
        tasks = sorted(t0)
        for split, field in [("test", "test_full_acc"), ("train", "train_full_acc")]:
            vals = [t0[t].get(field) for t in tasks]
            st = SPLIT_STYLES[split]
            ax.plot(tasks, vals, color=color,
                    linestyle=st["linestyle"], marker=st["marker"],
                    linewidth=1.8, markersize=5,
                    label=f"{opt.upper()} – {split}")
    _add_baseline(ax, all_data, "test_full_acc")
    ax.set_xlabel("Task index (T)", fontsize=12)
    ax.set_ylabel("Task-0 accuracy – joint head (%)", fontsize=12)
    ax.set_title("Task-0 Joint Head Accuracy (test=solid / train=dashed)\n"
                 "evaluated against all seen classes", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    all_tasks = sorted({t for data in all_data.values() for t in data["task0"]})
    ax.set_xticks(all_tasks)
    path = os.path.join(out_dir, "task0_full_acc_trajectory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {path}")


# ── CL dynamics (A_t / O_t / F_t / B_t) ──────────────────────────────────────

def extract_r_matrix(d: dict) -> list:
    """Return R[t][i] = test acc on task i after training step t. nan if not seen."""
    matrices = d.get("nme", {}).get("matrices", {})
    raw = matrices.get("final", [])
    if not raw:
        step_keys = [
            k for k in matrices
            if isinstance(k, str) and k.startswith("t") and k[1:].isdigit()
        ]
        if step_keys:
            raw = matrices[sorted(step_keys, key=lambda x: int(x[1:]))[-1]]
    return [[v if v is not None else float("nan") for v in row] for row in raw]


def compute_cl_dynamics(mat: list) -> dict:
    """Compute A_t, O_t, F_t, B_t for each t from the accuracy matrix R[t][i]."""
    import math
    T = len(mat)
    A_t, O_t, F_t, B_t = [], [], [], []

    for t in range(T):
        row = mat[t]

        # A_t: mean accuracy on all seen tasks (i = 0..t)
        seen = [row[i] for i in range(t + 1) if not math.isnan(row[i])]
        A_t.append(sum(seen) / len(seen) if seen else float("nan"))

        if t == 0:
            O_t.append(float("nan"))
            F_t.append(float("nan"))
            B_t.append(float("nan"))
            continue

        # O_t: mean accuracy on OLD tasks (i = 0..t-1)
        old = [row[i] for i in range(t) if not math.isnan(row[i])]
        O_t.append(sum(old) / len(old) if old else float("nan"))

        # F_t: running forgetting = mean over old tasks of (peak - current)
        forget_sum, cnt = 0.0, 0
        for i in range(t):
            peak = max(
                (mat[k][i] for k in range(i, t + 1) if not math.isnan(mat[k][i])),
                default=float("nan"),
            )
            cur = row[i]
            if not math.isnan(peak) and not math.isnan(cur):
                forget_sum += peak - cur
                cnt += 1
        F_t.append(forget_sum / cnt if cnt else float("nan"))

        # B_t: running BWT = mean over old tasks of (R_{t,i} - R_{i,i})
        bwt_sum, cnt = 0.0, 0
        for i in range(t):
            diag, cur = mat[i][i], row[i]
            if not math.isnan(diag) and not math.isnan(cur):
                bwt_sum += cur - diag
                cnt += 1
        B_t.append(bwt_sum / cnt if cnt else float("nan"))

    return {"tasks": list(range(T)), "A_t": A_t, "O_t": O_t, "F_t": F_t, "B_t": B_t}


def plot_cl_dynamics(all_data: dict, out_dir: str):
    """
    2×2 figure:
      (a) Seen-task mean accuracy A_t
      (b) Old-task mean accuracy O_t
      (c) Running forgetting F_t
      (d) Running BWT B_t
    x-axis: training task t
    """
    panels = [
        ("A_t", "Accuracy (%)",   "Seen-task mean accuracy $A_t$",  True),
        ("O_t", "Accuracy (%)",   "Old-task mean accuracy $O_t$",    True),
        ("F_t", "Forgetting (%)", "Running forgetting $F_t$",        False),
        ("B_t", "BWT (%)",        "Running BWT $B_t$",               False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, ylabel, title, _) in zip(axes, panels):
        for opt, data in all_data.items():
            dyn = data.get("dynamics")
            if dyn is None:
                continue
            tasks = dyn["tasks"]
            vals  = dyn[key]
            ax.plot(
                tasks, vals,
                color=OPT_COLORS.get(opt, "gray"),
                linestyle="-", marker="o",
                linewidth=1.8, markersize=5,
                label=opt.upper(),
            )
        if key == "B_t":
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Training task $t$", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        all_tasks = sorted({t for data in all_data.values()
                            if data.get("dynamics") for t in data["dynamics"]["tasks"]})
        ax.set_xticks(all_tasks)

    fig.suptitle("Continual Learning Dynamics: SGD vs GAM", fontsize=13)
    path = os.path.join(out_dir, "cl_dynamics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {path}")


# ── mechanism diagnosis figure + focused table ────────────────────────────────

FOCUS_TASKS = (5, 6, 7)


def _is_num(x):
    return isinstance(x, (int, float)) and not math.isnan(float(x))


def _series_from_mapping(mapping: dict, tasks: list, key: str) -> list:
    vals = []
    for t in tasks:
        v = mapping.get(t, {}).get(key)
        vals.append(v if _is_num(v) else float("nan"))
    return vals


def plot_mechanism_diagnosis(all_data: dict, out_dir: str):
    """
    5-panel mechanism figure:
      (a) Accuracy degradation over tasks: old-task mean + seen-task mean
      (b) Step-wise Fisher interference: delta^T F_acc delta
      (c) Squared Euclidean step drift: ||delta||_2^2
      (d) Task-0 normalized cumulative Fisher drift
      (e) Fisher directional mismatch: 1 - cos(full, accum)
    """
    fig, axes = plt.subplots(5, 1, figsize=(11, 15.5), sharex=True)

    all_tasks = sorted({
        t for data in all_data.values()
        for t in (
            set(data.get("dynamics", {}).get("tasks", []))
            | set(data.get("task0", {}))
            | set(data.get("fisher", {}))
        )
    })

    # (a) Accuracy degradation.
    ax = axes[0]
    for opt, data in all_data.items():
        dyn = data.get("dynamics", {})
        tasks = dyn.get("tasks", [])
        color = OPT_COLORS.get(opt, "gray")
        if tasks:
            ax.plot(tasks, dyn.get("O_t", []), color=color, linestyle="-",
                    marker="o", linewidth=1.9, markersize=4,
                    label=f"{opt.upper()} old-task mean")
            ax.plot(tasks, dyn.get("A_t", []), color=color, linestyle="--",
                    marker="s", linewidth=1.5, markersize=3.5,
                    label=f"{opt.upper()} seen-task mean")
    ax.set_title("(a) Accuracy degradation", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # (b) Step-wise Fisher interference.
    ax = axes[1]
    for opt, data in all_data.items():
        tasks = sorted(data["fisher"])
        vals = _series_from_mapping(data["fisher"], tasks, "fisher_weighted_drift")
        ax.plot(tasks, vals, color=OPT_COLORS.get(opt, "gray"), linestyle="-",
                marker="o", linewidth=1.9, markersize=4, label=opt.upper())
    ax.set_title("(b) Step-wise Fisher interference", fontsize=12)
    ax.set_ylabel("$I_F(t)=\\delta_t^T F^{acc}_{1:t-1}\\delta_t$", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (c) Squared Euclidean step drift.
    ax = axes[2]
    for opt, data in all_data.items():
        tasks = sorted(data["fisher"])
        delta_norm = _series_from_mapping(data["fisher"], tasks, "delta_norm")
        vals = [v * v if _is_num(v) else float("nan") for v in delta_norm]
        ax.plot(tasks, vals, color=OPT_COLORS.get(opt, "gray"), linestyle="-",
                marker="o", linewidth=1.9, markersize=4, label=opt.upper())
    ax.set_title("(c) Squared Euclidean step drift", fontsize=12)
    ax.set_ylabel("$S_\\delta(t)=||\\delta_t||_2^2$", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (d) Cumulative task-0 Fisher damage.
    ax = axes[3]
    for opt, data in all_data.items():
        tasks = sorted(data["task0"])
        vals = _series_from_mapping(data["task0"], tasks, "normalized_fisher_drift")
        ax.plot(tasks, vals, color=OPT_COLORS.get(opt, "gray"), linestyle="-",
                marker="o", linewidth=1.9, markersize=4, label=opt.upper())
    ax.set_title("(d) Task-0 normalized cumulative Fisher drift", fontsize=12)
    ax.set_ylabel("Task-0 $\\widetilde{D}_{F_0}(t)$", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (e) Fisher directional mismatch.
    ax = axes[4]
    for opt, data in all_data.items():
        tasks = sorted(data["fisher"])
        cosine = _series_from_mapping(data["fisher"], tasks, "full_vs_accum_cosine_sim")
        vals = [1.0 - v if _is_num(v) else float("nan") for v in cosine]
        ax.plot(tasks, vals, color=OPT_COLORS.get(opt, "gray"), linestyle="-",
                marker="o", linewidth=1.9, markersize=4, label=opt.upper())
    ax.set_title("(e) Fisher directional mismatch", fontsize=12)
    ax.set_ylabel("$M_F(t)=1-\\cos(F_t^{full},F_{1:t-1}^{acc})$", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if all_tasks:
        xmin, xmax = min(all_tasks), max(all_tasks)
        for ax in axes:
            ax.set_xlim(xmin - 0.2, xmax + 0.2)
            ax.set_xticks(all_tasks)
    axes[-1].set_xlabel("Training task $t$", fontsize=11)
    fig.suptitle("Mechanistic diagnosis of forgetting across the task sequence",
                 fontsize=14)

    path = os.path.join(out_dir, "mechanism_diagnosis_5panel.png")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    legacy_path = os.path.join(out_dir, "mechanism_diagnosis_4panel.png")
    fig.savefig(legacy_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {path}")
    print(f"[Plot] saved → {legacy_path} (updated 5-panel layout)")


def build_key_task_rows(all_data: dict, focus_tasks=FOCUS_TASKS) -> list:
    rows = []
    for opt in ["sgd", "gam"]:
        data = all_data.get(opt)
        if not data:
            continue
        dyn = data.get("dynamics", {})
        other = all_data.get("gam" if opt == "sgd" else "sgd", {})
        for t in focus_tasks:
            mech = data.get("fisher", {}).get(t, {})
            task0 = data.get("task0", {}).get(t, {})
            delta_norm = mech.get("delta_norm")
            cosine = mech.get("full_vs_accum_cosine_sim")
            row = {
                "optimizer": opt.upper(),
                "task": t,
                "old_task_mean_acc": _dyn_value(dyn, "O_t", t),
                "seen_task_mean_acc": _dyn_value(dyn, "A_t", t),
                "BWT": _dyn_value(dyn, "B_t", t),
                "delta_norm": delta_norm,
                "delta_norm_sq": delta_norm * delta_norm if _is_num(delta_norm) else None,
                "per_task_fisher_interference": mech.get("fisher_weighted_drift"),
                "task0_normalized_DF": task0.get("normalized_fisher_drift"),
                "task0_test_full_acc": task0.get("test_full_acc"),
                "fisher_cos_full_accum": cosine,
                "fisher_directional_mismatch": 1.0 - cosine if _is_num(cosine) else None,
                "full_fisher_fro": mech.get("full_fisher_fro"),
            }
            row["interpretation"] = _interpret_focus_row(opt, t, row, other)
            rows.append(row)
    return rows


def _dyn_value(dyn: dict, key: str, task: int):
    vals = dyn.get(key, [])
    return vals[task] if task < len(vals) else None


def _interpret_focus_row(opt: str, task: int, row: dict, other_data: dict) -> str:
    if not other_data:
        return ""
    other_mech = other_data.get("fisher", {}).get(task, {})
    other_dyn = other_data.get("dynamics", {})
    other_old = _dyn_value(other_dyn, "O_t", task)

    bits = []
    old_acc = row.get("old_task_mean_acc")
    if _is_num(old_acc) and _is_num(other_old):
        if opt == "gam" and old_acc > other_old:
            bits.append("better old-task retention")
        elif opt == "sgd" and old_acc < other_old:
            bits.append("lower old-task retention")

    cur_inter = row.get("per_task_fisher_interference")
    other_inter = other_mech.get("fisher_weighted_drift")
    if _is_num(cur_inter) and _is_num(other_inter):
        if cur_inter < other_inter:
            bits.append("lower Fisher interference")
        elif cur_inter > other_inter:
            bits.append("higher Fisher interference")

    cur_cos = row.get("fisher_cos_full_accum")
    other_cos = other_mech.get("full_vs_accum_cosine_sim")
    if _is_num(cur_cos) and _is_num(other_cos):
        if cur_cos > other_cos:
            bits.append("better Fisher alignment")
        elif cur_cos < other_cos:
            bits.append("worse Fisher alignment")

    return "; ".join(bits)


def save_key_task_tables(rows: list, out_dir: str):
    if not rows:
        return
    csv_path = os.path.join(out_dir, "mechanism_key_tasks.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] saved → {csv_path}")

    md_path = os.path.join(out_dir, "mechanism_key_tasks.md")
    with open(md_path, "w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt_table_value(row.get(h)) for h in headers) + " |\n")
    print(f"[Table] saved → {md_path}")


def _fmt_table_value(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3e}"
        return f"{v:.3f}"
    return str(v)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgd-json", default=None)
    parser.add_argument("--gam-json", default=None)
    parser.add_argument("--trainer-outputs", default=None,
                        help="Root trainer_outputs dir; auto-finds JSONs")
    parser.add_argument("--out-dir", default="result_summary/fisher_compare/default")
    args = parser.parse_args()

    if args.trainer_outputs:
        sgd_path = find_json(args.trainer_outputs, "sgd")
        gam_path = find_json(args.trainer_outputs, "gam")
    elif args.sgd_json and args.gam_json:
        sgd_path, gam_path = args.sgd_json, args.gam_json
    else:
        parser.error("Provide --trainer-outputs OR both --sgd-json and --gam-json")

    print(f"[SGD] {sgd_path}")
    print(f"[GAM] {gam_path}")

    sgd_d = load_metrics(sgd_path)
    gam_d = load_metrics(gam_path)

    os.makedirs(args.out_dir, exist_ok=True)

    all_data = {}
    all_rows = []
    for opt, d in [("sgd", sgd_d), ("gam", gam_d)]:
        fisher   = extract_fisher(d)
        aaa      = extract_aaa_per_step(d)
        task0    = extract_task0_acc(d)
        final    = extract_final_metrics(d)
        rmat     = extract_r_matrix(d)
        dynamics = compute_cl_dynamics(rmat)
        all_data[opt] = {"fisher": fisher, "aaa": aaa, "task0": task0, "final": final,
                         "dynamics": dynamics}
        rows = build_rows(opt, fisher, aaa, task0, final, dynamics)
        all_rows.extend(rows)
        t0_final = task0.get(max(task0), {}).get("test_local_acc", float("nan"))
        print(f"[{opt.upper()}] AAA={final.get('AAA'):.2f}%  "
              f"Forget={final.get('Forget_avg'):.2f}%  "
              f"task0_test_local final={t0_final:.2f}%")

    save_csv(all_rows, os.path.join(args.out_dir, "fisher_compare.csv"))
    save_key_task_tables(build_key_task_rows(all_data), args.out_dir)

    # ── console table ─────────────────────────────────────────────────────────
    print("\n── Fisher Fro + AAA + Task-0 ──────────────────────────────────────")
    print(f"{'opt':4s} {'T':>2s}  {'full_fro':>10s}  {'accum_fro':>10s}  "
          f"{'cosine':>7s}  {'AAA@T':>7s}  {'task0_acc':>9s}")
    for r in all_rows:
        if r["full_fisher_fro"] == "":
            continue
        print(
            f"{r['optimizer']:4s} {r['task']:>2d}  "
            f"{float(r['full_fisher_fro']):>10.5f}  {float(r['accum_fisher_fro']):>10.5f}  "
            f"{float(r['full_vs_accum_cosine_sim']):>7.3f}  "
            f"{float(r['aaa_at_step']) if r['aaa_at_step'] != '' else float('nan'):>7.2f}  "
            f"{float(r['task0_test_local_acc']) if r['task0_test_local_acc'] != '' else float('nan'):>9.2f}"
        )

    if not HAS_MPL:
        print("[Plot] matplotlib not found, skipping.")
        return

    plot_combined(all_data, args.out_dir)
    plot_fisher_only(all_data, args.out_dir)
    plot_aaa(all_data, args.out_dir)
    plot_task0(all_data, args.out_dir)
    plot_task0_full(all_data, args.out_dir)
    plot_fisher_divergence(all_data, args.out_dir)
    plot_cl_dynamics(all_data, args.out_dir)
    plot_mechanism_diagnosis(all_data, args.out_dir)


if __name__ == "__main__":
    main()
