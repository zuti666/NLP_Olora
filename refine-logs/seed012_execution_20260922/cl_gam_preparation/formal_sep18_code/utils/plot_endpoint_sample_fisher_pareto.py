#!/usr/bin/env python3
"""Plot raw sample-Fisher trace versus matched endpoint accuracy from run logs."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


MARKER = "[EWConProbe][EndpointSampleFisher]"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parse fixed-subset endpoint sample-Fisher metrics, filter methods "
            "against an endpoint-fit reference, and draw the trace/accuracy Pareto plot."
        )
    )
    parser.add_argument("logs", nargs="+", help="Log paths (shell globs are supported by the shell).")
    parser.add_argument("--output", default="logs_our_joost/imageneta_fishershape/endpoint_sample_fisher_pareto.png")
    parser.add_argument("--reference", default="ewc_sgd")
    parser.add_argument("--accuracy-tolerance", type=float, default=2.0, help="Percentage points.")
    parser.add_argument("--ce-tolerance", type=float, default=0.10)
    parser.add_argument("--min-matched-fraction", type=float, default=0.80)
    parser.add_argument("--min-coverage-fraction", type=float, default=1.00)
    return parser.parse_args()


def method_and_seed(path: Path):
    stem = path.stem
    method = stem.split("_imageneta_", 1)[0] if "_imageneta_" in stem else stem
    seed_match = re.search(r"_s(\d+)_gpu\d+$", stem)
    seed = int(seed_match.group(1)) if seed_match else 0
    return method, seed


def parse_log(path: Path):
    method, seed = method_and_seed(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER not in line:
                continue
            payload = line.split(MARKER, 1)[1].strip()
            try:
                metrics = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue
            required = {
                "task",
                "endpoint_sample_fisher_raw_trace",
                "endpoint_sample_fisher_ce",
                "endpoint_sample_fisher_accuracy",
            }
            if not required.issubset(metrics):
                continue
            rows.append(
                {
                    "file": str(path),
                    "method": method,
                    "seed": seed,
                    "task": int(metrics["task"]),
                    "raw_trace": float(metrics["endpoint_sample_fisher_raw_trace"]),
                    "endpoint_ce": float(metrics["endpoint_sample_fisher_ce"]),
                    "endpoint_accuracy": float(metrics["endpoint_sample_fisher_accuracy"]),
                    "num_samples": int(metrics.get("endpoint_sample_fisher_num_samples", 0)),
                    "subset_sha1": str(
                        metrics.get("endpoint_sample_fisher_subset_sha1", "")
                    ),
                }
            )
    return rows


def add_reference_matches(rows, reference, accuracy_tolerance, ce_tolerance):
    reference_rows = [row for row in rows if row["method"] == reference]
    if not reference_rows:
        raise SystemExit(f"No endpoint sample-Fisher rows found for reference method {reference!r}")
    lookup = {(row["seed"], row["task"]): row for row in reference_rows}
    for row in rows:
        ref = lookup.get((row["seed"], row["task"]))
        row["reference_available"] = ref is not None
        row["subset_compatible"] = bool(
            ref
            and row["num_samples"] == ref["num_samples"]
            and row["subset_sha1"]
            and row["subset_sha1"] == ref["subset_sha1"]
        )
        row["accuracy_delta_vs_reference"] = (
            row["endpoint_accuracy"] - ref["endpoint_accuracy"] if ref else float("nan")
        )
        row["ce_delta_vs_reference"] = (
            row["endpoint_ce"] - ref["endpoint_ce"] if ref else float("nan")
        )
        row["endpoint_matched"] = bool(
            ref
            and row["subset_compatible"]
            and abs(row["accuracy_delta_vs_reference"]) <= accuracy_tolerance
            and abs(row["ce_delta_vs_reference"]) <= ce_tolerance
        )
    return lookup


def summarize(rows, reference_lookup, min_matched_fraction, min_coverage_fraction):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    expected_pairs = set(reference_lookup)
    summaries = []
    for method, method_rows in sorted(groups.items()):
        comparable = [row for row in method_rows if row["reference_available"]]
        matched = [row for row in comparable if row["endpoint_matched"]]
        observed_pairs = {(row["seed"], row["task"]) for row in method_rows}
        coverage = len(observed_pairs & expected_pairs) / max(1, len(expected_pairs))
        matched_fraction = len(matched) / max(1, len(comparable))
        # Eligible method means must not be improved by endpoint-damaged rows.
        # For an ineligible method with zero matched rows, retain an all-row
        # location only so the hollow diagnostic point can still be displayed.
        mean_rows = matched if matched else method_rows
        summaries.append(
            {
                "method": method,
                "num_points": len(method_rows),
                "reference_comparable_points": len(comparable),
                "matched_points": len(matched),
                "mean_points": len(mean_rows),
                "matched_fraction": matched_fraction,
                "coverage_fraction": coverage,
                "mean_raw_trace": mean(row["raw_trace"] for row in mean_rows),
                "mean_endpoint_ce": mean(row["endpoint_ce"] for row in mean_rows),
                "mean_endpoint_accuracy": mean(row["endpoint_accuracy"] for row in mean_rows),
                "pareto_eligible": bool(
                    matched_fraction >= min_matched_fraction
                    and coverage >= min_coverage_fraction
                ),
            }
        )
    return summaries


def mark_pareto_front(summaries):
    eligible = [row for row in summaries if row["pareto_eligible"]]
    for row in summaries:
        row["pareto_front"] = False
    for candidate in eligible:
        dominated = any(
            other is not candidate
            and other["mean_raw_trace"] <= candidate["mean_raw_trace"]
            and other["mean_endpoint_accuracy"] >= candidate["mean_endpoint_accuracy"]
            and (
                other["mean_raw_trace"] < candidate["mean_raw_trace"]
                or other["mean_endpoint_accuracy"] > candidate["mean_endpoint_accuracy"]
            )
            for other in eligible
        )
        candidate["pareto_front"] = not dominated


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, summaries, output: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to create the Pareto plot") from exc

    methods = sorted({row["method"] for row in rows})
    cmap = plt.get_cmap("tab20")
    colors = {method: cmap(idx % 20) for idx, method in enumerate(methods)}
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        matched = [row for row in method_rows if row["endpoint_matched"]]
        unmatched = [row for row in method_rows if not row["endpoint_matched"]]
        if matched:
            axes[0].scatter(
                [row["raw_trace"] for row in matched],
                [row["endpoint_accuracy"] for row in matched],
                s=32,
                color=colors[method],
                alpha=0.80,
                label=method,
            )
        if unmatched:
            axes[0].scatter(
                [row["raw_trace"] for row in unmatched],
                [row["endpoint_accuracy"] for row in unmatched],
                s=32,
                color=colors[method],
                alpha=0.45,
                marker="x",
                label=method if not matched else None,
            )

    for summary in summaries:
        marker = "o" if summary["pareto_eligible"] else "x"
        edge = "black" if summary["pareto_front"] else colors[summary["method"]]
        axes[1].scatter(
            summary["mean_raw_trace"],
            summary["mean_endpoint_accuracy"],
            s=110,
            marker=marker,
            color=colors[summary["method"]],
            edgecolors=edge if marker == "o" else None,
            linewidths=1.8,
        )
        axes[1].annotate(
            summary["method"],
            (summary["mean_raw_trace"], summary["mean_endpoint_accuracy"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    front = sorted(
        (row for row in summaries if row["pareto_front"]),
        key=lambda row: row["mean_raw_trace"],
    )
    if len(front) > 1:
        axes[1].plot(
            [row["mean_raw_trace"] for row in front],
            [row["mean_endpoint_accuracy"] for row in front],
            linestyle="--",
            color="black",
            linewidth=1.0,
        )

    for axis, title in zip(
        axes,
        ("Per-task endpoints (x = endpoint mismatch)", "Method means (black edge = Pareto front)"),
    ):
        axis.set_xscale("log")
        axis.set_xlabel("Raw sample-empirical Fisher trace (lower is better)")
        axis.set_ylabel("Endpoint accuracy on the same fixed subset (%)")
        axis.set_title(title)
        axis.grid(True, which="both", linestyle=":", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    paths = [Path(item) for item in args.logs]
    rows = [row for path in paths for row in parse_log(path)]
    if not rows:
        raise SystemExit("No endpoint sample-Fisher records were found in the supplied logs")
    reference_lookup = add_reference_matches(
        rows, args.reference, args.accuracy_tolerance, args.ce_tolerance
    )
    summaries = summarize(
        rows,
        reference_lookup,
        args.min_matched_fraction,
        args.min_coverage_fraction,
    )
    mark_pareto_front(summaries)

    output = Path(args.output)
    write_csv(output.with_suffix(".points.csv"), rows)
    write_csv(output.with_suffix(".summary.csv"), summaries)
    plot(rows, summaries, output)

    print("method\ttrace\tendpoint_acc\tendpoint_ce\tmatched\tcoverage\tpareto")
    for row in summaries:
        print(
            f"{row['method']}\t{row['mean_raw_trace']:.6g}\t"
            f"{row['mean_endpoint_accuracy']:.3f}\t{row['mean_endpoint_ce']:.5f}\t"
            f"{row['matched_fraction']:.2f}\t{row['coverage_fraction']:.2f}\t"
            f"{int(row['pareto_front'])}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
