"""
Extract CNN Accuracy Matrices from EWC lambda-sweep log files,
save to JSON, and generate heatmap visualizations.

Usage:
    python utils/extract_cnn_accuracy_matrices.py \
        --log-dirs logs/imageneta_ewc_fullmodel_lambda_sweep/imageneta_ewc_fullmodel_lambda_sweep_lr0p01_e20_nonorm_10t/shell \
                   logs/imageneta_ewc_fullmodel_lambda_sweep/imageneta_ewc_fullmodel_lambda_sweep_lr0p01_e20_nonorm_10/shell \
        --output-dir result_summary/cnn_accuracy_matrices_nonorm
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ── parsing ────────────────────────────────────────────────────────────────────

def _parse_lambda(stem: str) -> float:
    """Extract lambda value from filename like 'ewc_lam500_seed0'."""
    m = re.search(r"lam(\d+)", stem)
    if m:
        return float(m.group(1))
    if "noewc" in stem or "A0" in stem:
        return 0.0
    raise ValueError(f"Cannot parse lambda from: {stem}")


def _extract_cnn_total_curve(log_path: Path) -> list[float]:
    """
    Extract the 'total' average accuracy after each task from lines like:
        CNN: {'total': np.float64(73.08), '00-19': ..., ...}
    Returns a list of total values (one per completed task), in order.
    """
    text = log_path.read_text(errors="replace")
    pattern = re.compile(r"CNN: \{[^}]*'total': np\.float64\(([0-9.]+)\)")
    return [float(m.group(1)) for m in pattern.finditer(text)]


def _extract_last_matrix(log_path: Path) -> np.ndarray | None:
    """
    Find the LAST occurrence of 'Accuracy Matrix (CNN Evaluation | time_by_task)'
    in the log file and parse the numpy-style matrix that follows.
    Returns a float ndarray or None if not found.
    """
    text = log_path.read_text(errors="replace")
    # Find all matrix blocks
    pattern = re.compile(
        r"Accuracy Matrix \(CNN Evaluation \| time_by_task\).*?"
        r"(\[\[.*?\]\])",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1)  # last occurrence = final state

    # Parse rows: each row looks like [86.97   nan   nan ...]
    rows = re.findall(r"\[([^\[\]]+)\]", raw)
    matrix = []
    for row in rows:
        vals = []
        for tok in row.split():
            vals.append(np.nan if tok == "nan" else float(tok))
        if vals:
            matrix.append(vals)
    if not matrix:
        return None
    return np.array(matrix, dtype=float)


# ── heatmap ───────────────────────────────────────────────────────────────────

def _draw_heatmap(matrix: np.ndarray, title: str, out_path: Path) -> None:
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(n * 1.1 + 1.8, n * 1.0 + 1.8))

    masked = np.ma.array(matrix, mask=np.isnan(matrix))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#e8e8e8")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "black" if 20 < val < 80 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{j}" for j in range(n)], fontsize=11)
    ax.set_yticklabels([f"T{i}" for i in range(n)], fontsize=11)
    ax.set_xlabel("Eval task j", fontsize=12)
    ax.set_ylabel("After training task i", fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Accuracy (%)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_overview(all_entries: list[dict], out_path: Path) -> None:
    """Multi-panel figure: one heatmap per lambda, sorted by lambda."""
    entries = sorted(all_entries, key=lambda e: e["lambda"])
    n_plots = len(entries)
    if n_plots == 0:
        return
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    n = entries[0]["matrix"].shape[0]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * (n * 0.72 + 1.2),
                                      nrows * (n * 0.65 + 1.4)),
                             squeeze=False)

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#e8e8e8")

    for idx, entry in enumerate(entries):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        mat = entry["matrix"]
        masked = np.ma.array(mat, mask=np.isnan(mat))
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "black" if 20 < v < 80 else "white"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")
        lam_str = entry.get("label") or f"λ={int(entry['lambda']) if entry['lambda']==int(entry['lambda']) else entry['lambda']}"
        avg = entry["mean_current_acc"]
        ax.set_title(f"{lam_str}\nmean_cur={avg:.1f}%", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"T{j}" for j in range(n)], fontsize=6)
        ax.set_yticklabels([f"T{i}" for i in range(n)], fontsize=6)

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("CNN Accuracy Matrices — batch_empirical (bs=128)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_avg_acc_heatmap(all_entries: list[dict], title: str, out_path: Path) -> None:
    """
    Reference-style heatmap: rows = lambda, cols = after each task,
    values = average accuracy on all seen tasks (CNN total).
    """
    entries = sorted(all_entries, key=lambda e: e["lambda"])
    n_tasks = max(len(e["cnn_total_curve"]) for e in entries)
    if n_tasks == 0:
        print("[WARN] No CNN total curve data found, skipping avg-acc heatmap.", file=sys.stderr)
        return

    # Build data matrix (n_lambdas × n_tasks); NaN for missing
    data = np.full((len(entries), n_tasks), np.nan)
    y_labels = []
    for i, e in enumerate(entries):
        curve = e["cnn_total_curve"][:n_tasks]
        data[i, :len(curve)] = curve
        y_labels.append(e.get("label", f"λ={e['lambda']:g}"))

    classes_per_task = 20
    x_labels = [f"After T{j}\n({j*classes_per_task}-{j*classes_per_task+classes_per_task-1})"
                for j in range(n_tasks)]

    vmin = max(0, np.nanmin(data) - 2)
    vmax = min(100, np.nanmax(data) + 2)

    fig_w = max(12, n_tasks * 1.4 + 2)
    fig_h = max(3, len(entries) * 0.65 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#e8e8e8")
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(len(entries)):
        for j in range(n_tasks):
            v = data[i, j]
            if not np.isnan(v):
                color = "black" if vmin + (vmax - vmin) * 0.2 < v < vmin + (vmax - vmin) * 0.85 else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(len(entries)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("After training task T", fontsize=11)
    ax.set_ylabel("Lambda", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Avg Accuracy (%)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved avg-acc heatmap -> {out_path}")


# ── summary table ─────────────────────────────────────────────────────────────

def _build_summary_df(all_entries: list[dict]) -> pd.DataFrame:
    """One row per lambda: diagonal acc per task + mean_current_acc + final_avg_acc."""
    entries = sorted(all_entries, key=lambda e: e["lambda"])
    rows = []
    for e in entries:
        mat = e["matrix"]
        n = mat.shape[0]
        row = {"lambda": int(e["lambda"]) if e["lambda"] == int(e["lambda"]) else e["lambda"],
               "label": e.get("label", f"λ={e['lambda']:g}")}
        # diagonal = accuracy right after learning each task
        for i in range(n):
            row[f"T{i}_current"] = round(float(mat[i, i]), 2) if not np.isnan(mat[i, i]) else None
        row["mean_current_acc"] = round(e["mean_current_acc"], 2)
        row["final_avg_acc"] = round(e["final_avg_acc"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def _load_log_file(log_file: Path, seen_lambdas: set, all_entries: list,
                   label: str | None = None) -> None:
    """Parse one log file and append to all_entries if successful."""
    try:
        lam = _parse_lambda(log_file.stem)
    except ValueError as e:
        print(f"[SKIP] {log_file.name}: {e}", file=sys.stderr)
        return

    if lam in seen_lambdas:
        print(f"[SKIP] λ={lam} already loaded, skipping {log_file}", file=sys.stderr)
        return

    mat = _extract_last_matrix(log_file)
    if mat is None:
        print(f"[SKIP] no CNN matrix found in {log_file.name}", file=sys.stderr)
        return

    n = mat.shape[0]
    if n < 10:
        print(f"[SKIP] λ={lam}: matrix is {n}×{n} (incomplete run, <10 tasks) in {log_file.name}", file=sys.stderr)
        return
    diag = [mat[i, i] for i in range(n) if not np.isnan(mat[i, i])]
    if len(diag) < n:
        print(f"[SKIP] λ={lam}: only {len(diag)}/{n} diagonal entries non-NaN in {log_file.name}", file=sys.stderr)
        return
    mean_cur = float(np.mean(diag)) if diag else float("nan")
    last_row = [mat[n - 1, j] for j in range(n) if not np.isnan(mat[n - 1, j])]
    final_avg = float(np.mean(last_row)) if last_row else float("nan")

    cnn_total_curve = _extract_cnn_total_curve(log_file)

    entry = {
        "lambda": lam,
        "label": label or f"λ={lam:g}",
        "source_file": str(log_file),
        "matrix": mat,
        "mean_current_acc": mean_cur,
        "final_avg_acc": final_avg,
        "cnn_total_curve": cnn_total_curve,
    }
    all_entries.append(entry)
    seen_lambdas.add(lam)
    tag = f" ({label})" if label else ""
    print(f"[OK] λ={lam:8g}{tag}  mean_cur={mean_cur:.2f}%  final_avg={final_avg:.2f}%  shape={mat.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract CNN accuracy matrices from EWC sweep logs.")
    parser.add_argument("--log-dirs", nargs="+", default=[],
                        help="Directories containing *.log files.")
    parser.add_argument("--extra-logs", nargs="+", default=[],
                        help="Individual log files to include (e.g. A0 baseline from outputs_logs).")
    parser.add_argument("--output-dir", default="result_summary/cnn_accuracy_matrices_nonorm",
                        help="Where to write JSON, CSV, and PNG outputs.")
    args = parser.parse_args()

    if not args.log_dirs and not args.extra_logs:
        parser.error("Provide at least one of --log-dirs or --extra-logs.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    all_entries = []
    seen_lambdas = set()

    # Extra individual files first (A0 baseline, etc.)
    for log_path in args.extra_logs:
        log_file = Path(log_path)
        if not log_file.exists():
            print(f"[WARN] extra log not found: {log_file}", file=sys.stderr)
            continue
        label = "A0_noewc" if ("noewc" in log_file.stem or "A0" in log_file.stem) else None
        _load_log_file(log_file, seen_lambdas, all_entries, label=label)

    # Directories
    for log_dir in args.log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            print(f"[WARN] log dir not found: {log_dir}", file=sys.stderr)
            continue
        for log_file in sorted(log_dir.glob("*.log")):
            _load_log_file(log_file, seen_lambdas, all_entries)

    if not all_entries:
        print("No matrices found — check --log-dirs.", file=sys.stderr)
        sys.exit(1)

    # ── save JSON ──
    json_data = []
    for e in sorted(all_entries, key=lambda x: x["lambda"]):
        mat_list = [[None if np.isnan(v) else round(float(v), 2) for v in row]
                    for row in e["matrix"]]
        json_data.append({
            "lambda": e["lambda"],
            "label": e.get("label", f"λ={e['lambda']:g}"),
            "source_file": e["source_file"],
            "mean_current_acc": round(e["mean_current_acc"], 4),
            "final_avg_acc": round(e["final_avg_acc"], 4),
            "cnn_total_curve": [round(v, 2) for v in e.get("cnn_total_curve", [])],
            "cnn_accuracy_matrix": mat_list,
        })
    json_path = out_dir / "cnn_accuracy_matrices.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON  -> {json_path}")

    # ── save summary CSV ──
    df = _build_summary_df(all_entries)
    csv_path = out_dir / "cnn_accuracy_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV   -> {csv_path}")

    # ── per-lambda heatmaps ──
    for e in all_entries:
        lam = int(e["lambda"]) if e["lambda"] == int(e["lambda"]) else e["lambda"]
        title = (f"CNN Accuracy Matrix  λ={lam}  "
                 f"(mean_cur={e['mean_current_acc']:.1f}%  "
                 f"final_avg={e['final_avg_acc']:.1f}%)")
        png = heatmap_dir / f"cnn_matrix_lam{lam}.png"
        _draw_heatmap(e["matrix"], title, png)
        print(f"Saved heatmap -> {png}")

    # ── overview figure (all lambdas) ──
    overview_path = out_dir / "cnn_matrices_overview.png"
    _draw_overview(all_entries, overview_path)
    print(f"Saved overview -> {overview_path}")

    # ── avg-acc-per-task heatmap (reference style) ──
    exp_tag = Path(args.output_dir).name.replace("cnn_accuracy_matrices_", "").upper()
    avg_acc_title = (
        f"Average Accuracy on All Seen Tasks (T0..Ti) After Training Task Ti\n"
        f"EWC Lambda Sweep — ImageNet-A (batch_empirical Fisher, {exp_tag})"
    )
    avg_acc_path = out_dir / "cnn_avg_acc_per_task.png"
    _draw_avg_acc_heatmap(all_entries, avg_acc_title, avg_acc_path)

    # ── print summary table ──
    print("\n" + "=" * 70)
    print("Summary (sorted by λ):")
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
