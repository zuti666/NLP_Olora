#!/usr/bin/env python3
"""Read pinned raw training logs and plot LoRA task-wise accuracy / running AAA.

Usage: python plot_lora_taskwise.py --metric both
The manifest selects runs; no plotted accuracy is hardcoded in this script.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = {"cifar100": ("CIFAR-100", 10), "domainnet": ("DomainNet", 5),
            "imagenetr": ("ImageNet-R", 10), "imageneta": ("ImageNet-A", 10)}
METHODS = ["Vanilla LoRA", "InfLoRA", "SD-LoRA", "CL-LoRA", "BiLoRA",
           "EWCLoRA", "GAM-EWC+Merge"]
COLORS = ["#777777", "#0072B2", "#D55E00", "#CCAA00", "#8B64AD", "#009E73", "#CC3377"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]
LINESTYLES = ["--", "-", "--", "-.", ":", "-", "-"]
CURVE_PATTERNS = [r"CNN top1 curve:\s*(\[[^\n]*?\])",
                  r"\(curve\) top1 Acc:\s*(\[[^\n]*?\])"]


def read_task_curve(path: str | Path, n_tasks: int, expected_sha256: str | None = None):
    """Return sample-weighted seen-class accuracy at each completed task.

    Only the classifier's class-incremental curve is read: neither NME /
    prototype nor task-oracle accuracy. Never use cl_metrics.json['AAA'],
    which can average the triangular matrix with unequal stage weights.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Raw source changed since manifest creation: {path}")
    text = raw.decode("utf-8", errors="replace")
    candidates = []
    for pattern in CURVE_PATTERNS:
        for match in re.finditer(pattern, text):
            value = re.sub(r"(?:np\.|numpy\.)?float(?:16|32|64)?\(([^()]*)\)", r"\1", match[1])
            try:
                curve = np.asarray(ast.literal_eval(value), dtype=float)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Malformed/non-finite curve in {path}: {value}") from exc
            candidates.append((match.start(), curve, text.count("\n", 0, match.start()) + 1))
    if not candidates:
        raise ValueError(f"No supported classifier curve found: {path}")
    _, curve, line = max(candidates, key=lambda x: x[0])
    if curve.shape != (n_tasks,) or not np.isfinite(curve).all():
        raise ValueError(f"Expected {n_tasks} finite task points, got {curve}: {path}")
    if np.any((curve < 0) | (curve > 100)):
        raise ValueError(f"Accuracy outside [0,100]: {path}")
    if re.search(r"\b(?:loss\w*|grad_norm)\s*[:= ]\s*(?:nan|[-+]?inf)\b", text, re.I):
        raise ValueError(f"Training contains non-finite loss/gradient; explicitly exclude this run: {path}")
    seed_match = re.search(r"=> seed:\s*(\d+)\s*$", text, re.M)
    return curve, {"sha256": digest, "curve_line": line,
                   "logged_seed": int(seed_match[1]) if seed_match else None}


def cumulative_aaa(curves: np.ndarray) -> np.ndarray:
    """Compute AAA_t=(A_1+...+A_t)/t independently for every seed (last axis)."""
    curves = np.asarray(curves, dtype=float)
    return np.cumsum(curves, axis=-1) / np.arange(1, curves.shape[-1] + 1)


def summarize_curves(curves: np.ndarray, metric="aaa", ddof=1):
    """Mean, SD and variance across seeds; single-seed SD is undefined, not zero."""
    values = cumulative_aaa(curves) if metric == "aaa" else np.asarray(curves, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=ddof) if len(values) > max(ddof, 1) else np.full_like(mean, np.nan)
    return mean, std, std**2


def load_raw_runs(manifest_path=HERE / "sources.json"):
    """Load every included run directly from its pinned original log."""
    manifest = json.loads(Path(manifest_path).read_text())
    groups = defaultdict(list)
    provenance, seen_keys, seen_hashes = [], set(), set()
    for run in manifest["runs"]:
        if not run.get("include", True):
            continue
        dataset, method, seed = run["dataset"], run["method"], int(run["seed"])
        key = (dataset, method, seed)
        if key in seen_keys:
            raise ValueError(f"Duplicate seed within dataset/method: {key}")
        curve, info = read_task_curve(run["log"], DATASETS[dataset][1], run.get("sha256"))
        if info["sha256"] in seen_hashes:
            raise ValueError(f"Same raw log counted twice: {run['log']}")
        if info["logged_seed"] is not None and info["logged_seed"] != seed:
            raise ValueError(f"Manifest/log seed mismatch: {run['log']}")
        seen_keys.add(key)
        seen_hashes.add(info["sha256"])
        record = dict(run, **info, accuracy=curve.tolist(), aaa=cumulative_aaa(curve).tolist())
        provenance.append(record)
        groups[(dataset, method)].append(record)
    for records in groups.values():
        records.sort(key=lambda r: r["seed"])
    return manifest, groups, provenance


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})


def plot_dataset(ax, groups, dataset, metric="aaa", ddof=1, label_map=None):
    """Draw one dataset, sharing method colors across all figures.

    Shade mean +/- SD across seeds after computing the chosen metric for
    each seed. This is neither variance in percent units nor a confidence interval.
    """
    label_map = label_map or {}
    title, n_tasks = DATASETS[dataset]
    x = np.arange(1, n_tasks + 1)
    for i, method in enumerate(METHODS):
        records = groups.get((dataset, method), [])
        if not records:
            continue
        curves = np.asarray([r["accuracy"] for r in records])
        mean, std, _ = summarize_curves(curves, metric, ddof)
        label = label_map.get(method, method)
        width = 2.2 if method == "GAM-EWC+Merge" else 1.5
        ax.plot(x, mean, color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                linewidth=width, markersize=4.8 if i == 6 else 3.7, label=label, zorder=4+i/10)
        if len(records) > max(ddof, 1):
            ax.fill_between(x, mean - std, mean + std, color=COLORS[i], alpha=0.14,
                            linewidth=0, zorder=1)
    ax.set_title(title, pad=8)  # Dataset labels follow the user's reference layout.
    ax.set_xlabel("Number of learned tasks")
    ax.set_ylabel("Cumulative AAA (%)" if metric == "aaa" else "Seen-class accuracy (%)")
    ax.set_xticks(x)
    ax.set_xlim(0.8, n_tasks + 0.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(y=0.09)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0, lo), min(100, hi))
    return ax


def save_figure(fig, stem):
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output, manifest, groups, provenance, ddof):
    raw, summary, final, comparisons, coverage, duplicates = [], [], [], [], [], []
    reference = {(r["dataset"], r["method"]): r for r in manifest.get("reported_table", [])}
    for dataset in DATASETS:
        for method in METHODS:
            records = groups.get((dataset, method), [])
            excluded = [r for r in manifest["runs"] if r["dataset"] == dataset and
                        r["method"] == method and not r.get("include", True)]
            coverage.append({"dataset": dataset, "method": method, "n": len(records),
                             "seeds": ",".join(str(r["seed"]) for r in records),
                             "excluded_seeds": ",".join(str(r["seed"]) for r in sorted(excluded, key=lambda r:r["seed"]))})
            if not records:
                continue
            curves = np.asarray([r["accuracy"] for r in records])
            for i, a in enumerate(records):
                for b in records[i+1:]:
                    if a["accuracy"] == b["accuracy"]:
                        duplicates.append({"dataset": dataset, "method": method,
                                           "seeds": [a["seed"], b["seed"]],
                                           "note": "Identical rounded curves in distinct seed-labelled logs; independent randomness not established."})
                for t, (acc, aaa) in enumerate(zip(a["accuracy"], a["aaa"]), 1):
                    raw.append({"dataset": dataset, "method": method, "seed": a["seed"],
                                "stage": t, "accuracy": acc, "aaa": aaa, "log": a["log"]})
            final_row = {"dataset": dataset, "method": method, "n": len(records)}
            for metric in ["accuracy", "aaa"]:
                mean, std, variance = summarize_curves(curves, metric, ddof)
                for t, (m, s, v) in enumerate(zip(mean, std, variance), 1):
                    summary.append({"dataset": dataset, "method": method, "metric": metric,
                                    "stage": t, "n": len(records), "mean": m,
                                    "std": s if np.isfinite(s) else "", "variance": v if np.isfinite(v) else ""})
                final_row[f"{metric}_mean"] = mean[-1]
                final_row[f"{metric}_std"] = std[-1] if np.isfinite(std[-1]) else ""
                ref = reference.get((dataset, method), {})
                if ref:
                    reported_mean, reported_std = ref[metric + "_mean"], ref[metric + "_std"]
                    comparisons.append({"dataset": dataset, "method": method, "metric": metric,
                                        "n": len(records), "raw_mean": mean[-1], "raw_std": std[-1] if np.isfinite(std[-1]) else "",
                                        "paper_mean": reported_mean, "paper_std": reported_std,
                                        "mean_difference_pp": mean[-1] - reported_mean,
                                        "matches_2dp": round(float(mean[-1]),2)==reported_mean and round(float(std[-1]),2)==reported_std})
            final.append(final_row)
    baseline = {r["dataset"]: r for r in final if r["method"] == "Vanilla LoRA"}
    for row in final:
        base = baseline.get(row["dataset"])
        for metric in ["accuracy", "aaa"]:
            delta = row[metric+"_mean"] - base[metric+"_mean"] if base else None
            row[metric+"_delta_vs_vanilla_pp"] = delta if delta is not None else ""
            row[metric+"_relative_gain_pct"] = 100 * delta / base[metric+"_mean"] if base and base[metric+"_mean"] else ""
    for name, rows in [("raw_per_seed", raw), ("stage_statistics", summary),
                       ("final_summary", final), ("table_comparison", comparisons), ("coverage", coverage)]:
        write_csv(output / f"{name}.csv", rows)
    for dataset in DATASETS:
        rows = [r for r in summary if r["dataset"] == dataset]
        write_csv(output / f"{dataset}_stage_statistics.csv", rows)
        sd_kind = "sample" if ddof else "population"
        md = [f"# {DATASETS[dataset][0]} — mean ± {sd_kind} SD across seeds", "",
              "Stage accuracy: sample-weighted seen-class classifier accuracy. AAA: cumulative mean of stage accuracy.", ""]
        tex = []
        for metric in ["aaa", "accuracy"]:
            md += [f"## {metric}", "", "| Method | n | " + " | ".join(str(i) for i in range(1,DATASETS[dataset][1]+1)) + " |",
                   "|---|---:" + "|---:" * DATASETS[dataset][1] + "|"]
            tex += [r"\begin{tabular}{l" + "c" * (DATASETS[dataset][1]+1) + "}", r"\toprule",
                    "Method & $n$ & " + " & ".join(str(i) for i in range(1, DATASETS[dataset][1]+1)) + r" \\", r"\midrule"]
            for method in METHODS:
                selected = [r for r in rows if r["metric"] == metric and r["method"] == method]
                if not selected:
                    continue
                cells = [f"{r['mean']:.2f} ± {r['std']:.2f}" if r["std"] != "" else f"{r['mean']:.2f}" for r in selected]
                md.append(f"| {method} | {selected[0]['n']} | " + " | ".join(cells) + " |")
                tex.append(method.replace("+", r"$+$") + f" & {selected[0]['n']} & " + " & ".join("$" + s.replace(" ± ", r"\pm ") + "$" for s in cells) + r" \\")
            md.append("")
            tex += [r"\bottomrule", r"\end{tabular}", ""]
        (output / f"{dataset}_table.md").write_text("\n".join(md))
        (output / f"{dataset}_table.tex").write_text("\n".join(tex))
    (output / "parsed_raw_runs.json").write_text(json.dumps(provenance, indent=2))
    (output / "duplicate_curve_warnings.json").write_text(json.dumps(duplicates, indent=2))
    (output / "computation.json").write_text(json.dumps({"ddof":ddof,
        "band":"mean +/- standard deviation across included seed-labelled runs",
        "aaa":"Per seed: cumsum(stage sample-weighted classifier accuracy) / stage number",
        "source_precision":"Accuracy is read at the decimal precision retained in original logs.",
        "all_excluded_runs":[r for r in manifest["runs"] if not r.get("include",True)]}, indent=2))
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE/"sources.json")
    parser.add_argument("--output", type=Path, default=HERE/"output")
    parser.add_argument("--metric", choices=["aaa", "accuracy", "both"], default="both")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--ddof", type=int, choices=[0,1], default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, groups, provenance = load_raw_runs(args.manifest)
    coverage = export_tables(args.output, manifest, groups, provenance, args.ddof)
    setup_style()
    metrics = ["aaa", "accuracy"] if args.metric == "both" else [args.metric]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    label_map = manifest.get("display_labels", {})
    captions = []
    for metric in metrics:
        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(5.6, 4.35))
            plot_dataset(ax, groups, dataset, metric, args.ddof, label_map)
            handles, labels = ax.get_legend_handles_labels()
            nmap = {label_map.get(m,m): len(groups.get((dataset,m),[])) for m in METHODS}
            labels = [f"{label} (n={nmap[label]})" for label in labels]
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52,0.055), ncol=2,
                       frameon=False, fontsize=8.7, handlelength=2.1, columnspacing=1.2)
            sd_kind = "sample" if args.ddof else "population"
            single_note = f"Mean ± {sd_kind} SD across seeds; method-specific training recipes."
            if any("†" in label for label in labels):
                single_note += "\n† Unbound single-seed references; no band."
            fig.text(0.52,0.012,single_note,ha="center",fontsize=7.3)
            legend_rows = (len(labels)+1)//2
            fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.25+0.045*(legend_rows-1))
            save_figure(fig, args.output / f"{dataset}_{metric}")
        if args.dataset == "all":
            fig, axes = plt.subplots(1,4,figsize=(13.4,3.75))
            handles_by_label = {}
            for ax, dataset in zip(axes, DATASETS):
                plot_dataset(ax, groups, dataset, metric, args.ddof, label_map)
                handles, labels = ax.get_legend_handles_labels()
                handles_by_label.update(zip(labels, handles))
            fig.legend(handles_by_label.values(), handles_by_label.keys(), loc="lower center",
                       bbox_to_anchor=(0.51,0.095), ncol=7, frameon=False, fontsize=9.1,
                       columnspacing=1.4, handlelength=2.1)
            footnote = manifest.get("figure_footnote", "Bands: mean ± SD across seeds; see coverage.csv for n.")
            if not args.ddof:
                footnote = footnote.replace("sample SD", "population SD")
            fig.text(0.51,0.015,footnote,
                     ha="center", fontsize=8.5)
            fig.subplots_adjust(left=0.049,right=0.995,top=0.9,bottom=0.30,wspace=0.38)
            save_figure(fig, args.output / f"lora_taskwise_{metric}")
        meaning = (r"Running $\mathrm{AAA}_t=t^{-1}\sum_{s=1}^t A_s$" if metric=="aaa"
                   else r"Seen-class classifier accuracy $A_t$ after learning task $t$")
        counts = "; ".join(DATASETS[d][0] + ": " + ", ".join(f"{r['method']} n={r['n']}" for r in coverage if r["dataset"]==d and r["n"]>0) for d in datasets)
        caption = meaning + ". Curves are means and bands are " + ("sample" if args.ddof else "population") + " standard deviations across seed-labelled runs, computed after the per-seed metric transform. "
        caption += manifest.get("caption_note", "") + " Replication: " + counts + "."
        figure_name = "lora_taskwise_" + metric if args.dataset=="all" else args.dataset+"_"+metric
        captions += [r"\begin{figure*}[t]", r"\centering",
                     r"\includegraphics[width=\textwidth]{" + args.output.name + "/" + figure_name + ".pdf}",
                     r"\caption{" + caption.replace("%",r"\%") + "}",
                     r"\label{fig:lora_taskwise_" + metric + "}", r"\end{figure*}", ""]
    (args.output/"latex_includes.tex").write_text("\n".join(captions))
    print(f"Read {len(provenance)} raw logs; {len(groups)} dataset/method groups. Saved figures and tables to {args.output}")


if __name__ == "__main__":
    main()
