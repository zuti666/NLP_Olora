#!/usr/bin/env python3
"""
Plot task-0 trajectory metrics comparing two methods (e.g. GAM vs SGD).

Usage:
    python3 utils/plot_task0_trajectory.py \
        --gam  logs/.../gam_T10_seed0.log \
        --sgd  logs/.../sgd_T10_seed0.log \
        --output plots/task0_traj_compare.png \
        --title "LoRA EWC – ImageNet-A (λ=2000, lr=0.02, 50ep)"

Metrics plotted (5 rows × 2 cols):
  Row 0 – Accuracy
    [0,0] Task-0 global test acc   (test_full_acc,  all seen classes)
    [0,1] Task-0 local head acc    (test_local_acc, task-0 head only)

  Row 1 – Task-0 raw loss
    [1,0] Task-0 full-head raw loss
          (test_full_loss solid, train_full_loss dashed)
    [1,1] Task-0 local-head raw loss
          (train_local_loss solid, test_local_loss dashed)

  Row 2 – Parameter damage / drift
    [2,0] Fisher-weighted task-0 damage
          (cum_delta0_F0_anchor_delta0, fallback to loaded anchor)
    [2,1] ||θ_t – θ_0||²           (cum_delta0_sq)

  Row 3 – Task-0 Fisher cosine structure
    [3,0] Fisher cosine(current, anchor)  (task0_fisher_anchor_cosine_sim)
    [3,1] Fisher cosine(current task-0, accum EWC)
          (task0_fisher_accum_cosine_sim)

  Row 4 – Oracle vs Accum Fisher quality
    [4,0] Oracle vs Accum cosine          (full_vs_accum_cosine_sim)
    [4,1] Oracle vs Accum relative Fro    (full_vs_accum_rel_fro)

Task0Trajectory panels include step 0, the task-0 anchor/baseline point.
Mechanism panels start at task 1 because full-vs-accum mechanism metrics are
logged after each subsequent task update.
"""

import argparse
import ast
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


def _setup_font() -> None:
    matplotlib.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "axes.unicode_minus": False,
    })


# ──────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────

def parse_trajectory_metrics(log_path):
    """Return {step: dict} from [Task0Trajectory] step=N metrics={...} lines."""
    steps = {}
    pat = re.compile(r'\[Task0Trajectory\] step=(\d+) metrics=(.+)$')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line.strip())
            if m:
                step = int(m.group(1))
                try:
                    steps[step] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
    return steps


def parse_trajectory_from_cl_metrics_json(json_path, head='nme'):
    """Return {step: dict} using the task-0 column of the standard CL accuracy
    matrix (cnn/nme) in a *_cl_metrics.json file, as a fallback when the
    [Task0Trajectory] probe never logged (e.g. it crashed at init).

    Only 'test_local_acc' is populated (matrix column 0 = task-0 test acc,
    evaluated jointly over all seen classes -- NOT the local-head-only value
    the live probe would have produced). All other trajectory fields
    (Fisher-weighted damage, param drift, task-0 train loss) have no
    equivalent in this JSON and are left absent (-> NaN downstream).
    """
    import json
    data = json.loads(Path(json_path).read_text())
    mats = data.get(head, {}).get('matrices', {})
    keys = [k for k in mats if re.fullmatch(r't\d+', k)]
    if not keys:
        return {}
    last_key = max(keys, key=lambda k: int(k[1:]))
    mat = mats[last_key]
    out = {}
    for step, row in enumerate(mat):
        val = row[0]
        if val is not None and val == val:  # skip NaN
            out[step] = {'test_local_acc': float(val)}
    return out


def parse_mechanism_from_cl_metrics_json(json_path):
    """Return {task: dict} from mechanism.steps in a *_cl_metrics.json file."""
    import json
    data = json.loads(Path(json_path).read_text())
    steps = data.get('mechanism', {}).get('steps', {})
    out = {}
    for key, row in steps.items():
        if not isinstance(row, dict):
            continue
        task = row.get('task', key)
        try:
            task = int(task)
        except Exception:
            continue
        if 'full_vs_accum_cosine_sim' in row:
            out[task] = row
    return out


def parse_mechanism_metrics(log_path, prefix='[EWCLoRA-GAM][Mechanism]'):
    """Return {task: dict} from mechanism log lines."""
    tasks = {}
    pat = re.compile(re.escape(prefix) + r' (.+)$')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line.strip())
            if m:
                try:
                    d = ast.literal_eval(m.group(1))
                    t = d.get('task')
                    if t is not None and 'full_vs_accum_cosine_sim' in d:
                        tasks[t] = d
                except Exception:
                    pass
    return tasks


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

nan = float('nan')

def seq(d, key, xs):
    return [d.get(x, {}).get(key, nan) for x in xs]


def seq_first(d, keys, xs):
    """Return the first available metric among keys for each x."""
    vals = []
    for x in xs:
        row = d.get(x, {})
        vals.append(next((row[k] for k in keys if k in row), nan))
    return vals


def plot_line(ax, xs, gam, sgd, title, ylabel, ylim=None, hline=None,
              xlabel='After task T', gam_c='#e74c3c', sgd_c='#3498db',
              xlabel_fontsize=8, ylabel_fontsize=8, title_fontsize=10):
    ax.plot(xs, gam, color=gam_c, marker='o', label='GAM', lw=2, ms=5)
    ax.plot(xs, sgd, color=sgd_c, marker='s', label='SGD', lw=2, ms=5)
    if hline is not None:
        ax.axhline(hline, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if xs:
        ax.set_xticks(xs)


def plot_raw_loss(ax, xs, gam_solid, sgd_solid, gam_dash, sgd_dash,
                  title, ylabel, solid_label, dash_label):
    """Plot raw task-0 losses with method colors and split-specific line styles."""
    gam_c, sgd_c = '#e74c3c', '#3498db'
    ax.plot(xs, gam_solid, color=gam_c, marker='o', label=f'GAM {solid_label}', lw=2, ms=5)
    ax.plot(xs, sgd_solid, color=sgd_c, marker='s', label=f'SGD {solid_label}', lw=2, ms=5)
    ax.plot(xs, gam_dash, color=gam_c, marker='o', ls='--', label=f'GAM {dash_label}', lw=2, ms=5)
    ax.plot(xs, sgd_dash, color=sgd_c, marker='s', ls='--', label=f'SGD {dash_label}', lw=2, ms=5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('After task T', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    if xs:
        ax.set_xticks(xs)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gam',    required=True, help='GAM run log path')
    parser.add_argument('--sgd',    required=True, help='SGD run log path')
    parser.add_argument('--output', default='task0_trajectory_compare.png')
    parser.add_argument('--title',  default='')
    parser.add_argument('--mech-prefix', default='[EWCLoRA-GAM][Mechanism]',
                        help='Mechanism log prefix (same for LoRA GAM and SGD files)')
    parser.add_argument('--row3', action='store_true',
                        help='Draw a single row with panels E (Fisher-weighted '
                             'cumulative damage), F (Euclidean cumulative drift), '
                             'D (task-0 local-head TRAIN loss only), and B (task-0 '
                             'local-head test accuracy). Legend: EWC=blue, Rel-EWC=orange')
    parser.add_argument('--row4', action='store_true',
                        help='Draw a single row with panels I/J: Oracle-vs-Accum '
                             'Fisher cosine and relative Frobenius gap. Supports '
                             'mechanism.steps from *_cl_metrics.json.')
    parser.add_argument('--acc-head', default='nme', choices=['nme', 'cnn'],
                        help='When --gam/--sgd point at *_cl_metrics.json (no '
                             '[Task0Trajectory] log available), which accuracy '
                             'matrix head to read the task-0 column from.')
    args = parser.parse_args()

    # ── Parse ──────────────────────────────────
    is_json = str(args.gam).endswith('.json') or str(args.sgd).endswith('.json')
    if is_json:
        gam_traj = parse_trajectory_from_cl_metrics_json(args.gam, head=args.acc_head)
        sgd_traj = parse_trajectory_from_cl_metrics_json(args.sgd, head=args.acc_head)
        gam_mech = parse_mechanism_from_cl_metrics_json(args.gam)
        sgd_mech = parse_mechanism_from_cl_metrics_json(args.sgd)
        print(f"[Note] Loaded from *_cl_metrics.json ({args.acc_head} head): "
              f"panel B uses the task-0 accuracy column; mechanism panels use "
              f"mechanism.steps when present. E/F/D have no equivalent in this "
              f"JSON and will be empty.")
    else:
        gam_traj = parse_trajectory_metrics(args.gam)
        sgd_traj = parse_trajectory_metrics(args.sgd)
        gam_mech = parse_mechanism_metrics(args.gam, args.mech_prefix)
        sgd_mech = parse_mechanism_metrics(args.sgd, args.mech_prefix)

    steps = sorted(set(gam_traj) | set(sgd_traj))
    tasks = sorted(set(gam_mech) | set(sgd_mech))

    # ── Metrics ────────────────────────────────
    g_full   = seq(gam_traj, 'test_full_acc',  steps)
    s_full   = seq(sgd_traj, 'test_full_acc',  steps)
    g_local  = seq(gam_traj, 'test_local_acc', steps)
    s_local  = seq(sgd_traj, 'test_local_acc', steps)
    g_test_full_loss   = seq(gam_traj, 'test_full_loss', steps)
    s_test_full_loss   = seq(sgd_traj, 'test_full_loss', steps)
    g_train_full_loss  = seq(gam_traj, 'train_full_loss', steps)
    s_train_full_loss  = seq(sgd_traj, 'train_full_loss', steps)
    g_train_local_loss = seq(gam_traj, 'train_local_loss', steps)
    s_train_local_loss = seq(sgd_traj, 'train_local_loss', steps)
    g_test_local_loss  = seq(gam_traj, 'test_local_loss', steps)
    s_test_local_loss  = seq(sgd_traj, 'test_local_loss', steps)

    g_cumsq  = seq(gam_traj, 'cum_delta0_sq', steps)
    s_cumsq  = seq(sgd_traj, 'cum_delta0_sq', steps)
    # Cumulative Fisher-weighted damage to task-0-important directions.
    # Prefer the recomputed task-0 local-head anchor Fisher. Older logs may only
    # have the loaded anchor or raw quadratic fields, so keep a conservative
    # fallback chain.
    fisher_damage_keys = [
        'cum_delta0_F0_anchor_delta0',
        'cum_delta0_F0_loaded_anchor_delta0',
        'fisher_raw_quadratic',
    ]
    g_fdamage = seq_first(gam_traj, fisher_damage_keys, steps)
    s_fdamage = seq_first(sgd_traj, fisher_damage_keys, steps)

    g_cosA   = seq(gam_traj, 'task0_fisher_anchor_cosine_sim', steps)
    s_cosA   = seq(sgd_traj, 'task0_fisher_anchor_cosine_sim', steps)
    g_cosAcc = seq(gam_traj, 'task0_fisher_accum_cosine_sim', steps)
    s_cosAcc = seq(sgd_traj, 'task0_fisher_accum_cosine_sim', steps)
    g_fvc    = seq(gam_mech, 'full_vs_accum_cosine_sim', tasks)
    s_fvc    = seq(sgd_mech, 'full_vs_accum_cosine_sim', tasks)

    g_rfro  = seq(gam_mech, 'full_vs_accum_rel_fro', tasks)
    s_rfro  = seq(sgd_mech, 'full_vs_accum_rel_fro', tasks)

    if args.row4:
        if not tasks:
            raise SystemExit(
                "No mechanism steps found. For JSON input, expected "
                "mechanism.steps with full_vs_accum_cosine_sim."
            )
        _setup_font()
        gam_c, sgd_c = '#e74c3c', '#3498db'
        LW, MS = 1.8, 4.5
        FSL, FSTK, FSLEG = 14, 12, 11

        fig, (ax_i, ax_j) = plt.subplots(
            1, 2, figsize=(10.2, 3.8), gridspec_kw={"wspace": 0.34})

        def _style(ax, gam, sgd, ylabel, ylim=None, hline=None):
            ax.plot(tasks, gam, color=gam_c, marker='o', label='GAM', lw=LW, ms=MS)
            ax.plot(tasks, sgd, color=sgd_c, marker='s', label='SGD', lw=LW, ms=MS)
            if hline is not None:
                ax.axhline(hline, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set_xlabel('Task', fontsize=FSL)
            ax.set_ylabel(ylabel, fontsize=FSL)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.tick_params(axis='both', labelsize=FSTK)
            ax.legend(loc='best', fontsize=FSLEG, framealpha=0.88,
                      handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if ylim:
                ax.set_ylim(ylim)

        _style(ax_i, g_fvc, s_fvc,
               r'$\cos(F_{1:t}^{oracle},\Omega_t)$',
               ylim=(0.0, 1.05), hline=1.0)
        _style(ax_j, g_rfro, s_rfro,
               r'$\|F_{1:t}^{oracle}-\Omega_t\|_F/\|\Omega_t\|_F$')

        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.96,
                            wspace=0.34)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), format=out.suffix.lstrip('.') or 'pdf',
                    bbox_inches='tight')
        print(f"[Saved] {out}")
        plt.close(fig)
        return

    if args.row3:
        # Style ported from utils/plot_r_sweep.py::draw_1x3_residual
        # (same 1x3 triptych used for r_sweep_1x3_residual.pdf): no title,
        # symbol-only y-labels, larger fonts, Times New Roman.
        _setup_font()
        gam_c, sgd_c = '#e67e22', '#3498db'  # GAM=orange, SGD=blue
        LW, MS = 1.6, 4
        FSL, FSTK, FSLEG = 14, 12, 11

        fig, (ax_e, ax_f, ax_d, ax_b) = plt.subplots(
            1, 4, figsize=(19.5, 3.8), gridspec_kw={"wspace": 0.38})

        def _style(ax, xs, gam, sgd, ylabel):
            ax.plot(xs, gam, color=gam_c, marker='o', label='Rel-EWC', lw=LW, ms=MS)
            ax.plot(xs, sgd, color=sgd_c, marker='s', label='EWC', lw=LW, ms=MS)
            ax.set_xlabel('Task', fontsize=FSL)
            ax.set_ylabel(ylabel, fontsize=FSL)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.tick_params(axis='both', labelsize=FSTK)
            ax.legend(loc='best', fontsize=FSLEG, framealpha=0.88,
                      handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        _style(ax_e, steps, g_fdamage, s_fdamage,
               r'$(\Delta_{0\to t})^{T}F_0\Delta_{0\to t}$')
        _style(ax_f, steps, g_cumsq, s_cumsq,
               r'$\|\Delta_{0\to t}\|_F^2=\|\theta_t-\theta_0\|_F^2$')
        _style(ax_d, steps, g_train_local_loss, s_train_local_loss,
               r'$L_{S_0}(\theta_t)$')
        _style(ax_b, steps, g_local, s_local, r'$A_0(\theta_t)$')

        fig.tight_layout()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), format=out.suffix.lstrip('.') or 'pdf',
                    bbox_inches='tight')
        print(f"[Saved] {out}")
        plt.close(fig)
        return

    # ── Main figure: 5×2 ───────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(13, 20))
    axes = axes.flatten()
    title = args.title or 'Task-0 Trajectory Analysis: GAM vs SGD'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    # Row 0 – Accuracy
    plot_line(axes[0], steps, g_full,  s_full,
              '(A) $A_0^{\\mathrm{joint}}(t)$: Task-0 Joint Test Accuracy\n'
              '(all seen classes / joint NME head)',
              'Test Acc (%)', ylim=(55, 90))
    plot_line(axes[1], steps, g_local, s_local,
              '(B) $A_0^{\\mathrm{local}}(t)$: Task-0 Local-Head Test Accuracy\n'
              '(task-0 classes only)',
              'Test Acc (%)', ylim=(70, 100))
    plot_raw_loss(
        axes[2],
        steps,
        g_test_full_loss,
        s_test_full_loss,
        g_train_full_loss,
        s_train_full_loss,
        '(C) $L_0^{\\mathrm{joint}}(t)$: Task-0 Full-Head Loss\n'
        '(test=solid, train=dashed; all seen classes)',
        'Full Loss',
        'test',
        'train',
    )
    plot_raw_loss(
        axes[3],
        steps,
        g_train_local_loss,
        s_train_local_loss,
        g_test_local_loss,
        s_test_local_loss,
        '(D) $L_0^{\\mathrm{local}}(t)$: Task-0 Local-Head Loss\n'
        '(train=solid, test=dashed; task-0 classes only)',
        'Local Loss',
        'train',
        'test',
    )
    plot_line(axes[4], steps, g_fdamage, s_fdamage,
              '(E) $Q_{F_0}(t)=\\Delta_{0\\to t}^{T}F_0^0\\Delta_{0\\to t}$\n'
              'Task-0 Fisher-Weighted Cumulative Damage',
              '$\\Delta_{0\\to t}^T F_0^0 \\Delta_{0\\to t}$')

    # Row 1 – Drift / Fisher structure
    plot_line(axes[5], steps, g_cumsq, s_cumsq,
              '(F) $D_\\theta^{(2)}(t)=\\|\\theta_t-\\theta_0\\|_2^2$\n'
              'Euclidean Cumulative Parameter Drift',
              '$\\|\\Delta\\theta\\|_2^2$')
    plot_line(axes[6], steps, g_cosA, s_cosA,
              '(G) $C_{F_0}^{\\mathrm{anchor}}(t)=\\cos(F_0(t),F_0^0)$\n'
              'Current Task-0 Fisher vs Anchor',
              'Cosine Similarity', ylim=(0.4, 1.05), hline=1.0)
    plot_line(axes[7], steps, g_cosAcc, s_cosAcc,
              '(H) $C_{F_0}^{\\mathrm{acc}}(t)=\\cos(F_0(t),\\Omega_t)$\n'
              'Current Task-0 Fisher vs Accumulated EWC',
              'Cosine Similarity', ylim=(0.4, 1.05), hline=1.0)
    plot_line(axes[8], tasks, g_fvc, s_fvc,
              '(I) $C_{\\mathrm{oracle}}^{\\mathrm{acc}}(t)=\\cos(F_{1:t}^{\\mathrm{oracle}},\\Omega_t)$\n'
              'Oracle Fisher vs Accumulated EWC',
              'Cosine Similarity', ylim=(0.4, 1.05), hline=1.0)
    plot_line(axes[9], tasks, g_rfro, s_rfro,
              '(J) $\\rho_F(t)=\\|F_{1:t}^{\\mathrm{oracle}}-\\Omega_t\\|_F/\\|\\Omega_t\\|_F$\n'
              'Oracle-Accumulated Relative Frobenius Gap',
              'Rel. Frobenius')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out}")
    plt.close()


if __name__ == '__main__':
    main()
