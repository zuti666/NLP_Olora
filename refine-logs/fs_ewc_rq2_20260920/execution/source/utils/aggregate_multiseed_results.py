#!/usr/bin/env python3
"""Aggregate Stage-2 (Block B5/B6/B7) results across seeds.

Scans outputs_logs/step_ablation for *_cl_metrics.json, classifies each run
into one of the known Stage-2 method families by its optimizer_tag_override
directory component, groups by seed, and reports mean +/- std for:

  - Overall_Acc: the pooled-over-all-samples accuracy at the final task,
    parsed from the training log's "=> CNN: {'total': X, ...}" line (NOT
    stored in the metrics JSON). This is the headline metric per the 2026-07
    decision to report the sample-weighted pooled accuracy rather than FAA.
    Note this is NOT the same as the log's separately-printed "Average
    Accuracy (CNN): Y" line -- that Y is the *cumulative running mean* of
    this per-task 'total' value across all tasks so far, a different number
    (e.g. Y=70.64 vs this task's raw total=57.8 in one seed=1 run). Always
    read 'total' directly from the CNN dict line; do not reconstruct it by
    inverting the running mean.
  - FAA/AAA/BWT_final_avg/Forget_avg from the JSON, kept for reference and
    because Forget_avg (and every Block B6/B7 mechanism metric) is built on
    the *unweighted per-task* accounting that FAA also uses -- switching the
    headline number to the sample-weighted Overall_Acc does not change that
    basis, the two answer different questions (see EXPERIMENT_PLAN.md).

For the two mechanism runs (mechanism_ours / mechanism_baseline) it also
aggregates the per-task mechanism fields used in Block B6/B7
(future_*_abs_residual, fisher_*_effective_dim): first averaged over tasks
within a seed, then mean +/- std across seeds.

Usage:
    python3 utils/aggregate_multiseed_results.py
    python3 utils/aggregate_multiseed_results.py --outputs-root outputs_logs/step_ablation
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

SEED_RE = re.compile(r"_sgd_s(\d+)_vit_base_patch16_224_cl_metrics\.json$")
# Matches "=> CNN: {'total': 57.8, ...}" or "=> CNN: {'total': np.float64(57.8), ...}"
LOG_ACC_RE = re.compile(r"=>\s*CNN:\s*\{'total':\s*(?:np\.float64\()?([0-9.]+)\)?")

# Ordered most-specific-first: substrings matched against the metrics json path.
METHOD_MARKERS = [
    ("mechanism_ours", "mechanism_ours_gam_sgd"),
    ("mechanism_baseline", "mechanism_baseline_gam_sgd"),
    ("component_ewc", "component_ewc_sgd"),
    ("component_gam", "component_gam_sgd"),
    ("component_merge", "component_merge_sgd"),
    ("component_ours", "gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30"),
]

# Glob patterns (relative to repo root) to find each method+seed's training
# log, used only to recover the "Average Accuracy (CNN)" headline number,
# which the metrics JSON does not persist. {seed} is substituted in.
LOG_GLOB_PATTERNS: Dict[str, List[str]] = {
    "component_ewc": [
        "logs_try/logs_s3_multiseed_verification/component_ewc_s{seed}_smigpu*.log",
        "logs_try/logs_s3_component_ablation_seed2/component_ewc_gpu*.log",
    ],
    "component_gam": [
        "logs_try/logs_s3_multiseed_verification/component_gam_s{seed}_smigpu*.log",
        "logs_try/logs_s3_component_ablation_seed2/component_gam_gpu*.log",
    ],
    "component_merge": [
        "logs_try/logs_s3_multiseed_verification/component_merge_s{seed}_smigpu*.log",
        "logs_try/logs_s3_component_ablation_seed2/component_merge_gpu*.log",
    ],
    "component_ours": [
        "logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s{seed}_gpu*.log",
    ],
    "mechanism_ours": [
        "logs_try/logs_s3_mechanism/mechanism_ours_s{seed}_v2_smigpu*.log",
    ],
    "mechanism_baseline": [
        "logs_try/logs_s3_mechanism/mechanism_baseline_s{seed}_v2_smigpu*.log",
    ],
}

CL_METRIC_KEYS = ["FAA", "AAA", "BWT_final_avg", "Forget_avg"]
MECHANISM_TASK_KEYS = [
    "anchor_endpoint_acc",
    "anchor_merge_acc",
    "future_endpoint_abs_residual",
    "future_merge_only_abs_residual",
    "future_merge_then_fisher_abs_residual",
    "fisher_endpoint_effective_dim",
    "fisher_merge_effective_dim",
]


def classify(path: str) -> Optional[str]:
    for name, marker in METHOD_MARKERS:
        if marker in path:
            return name
    return None


def mean_std(values: List[float]) -> str:
    values = [v for v in values if v is not None and v == v]  # drop NaN
    if not values:
        return "n=0"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1)"
    return f"{statistics.mean(values):.3f} +/- {statistics.stdev(values):.3f} (n={len(values)})"


def per_seed_task_average(steps: Dict, key: str) -> Optional[float]:
    vals = [s[key] for s in steps.values() if key in s]
    if not vals:
        return None
    return statistics.mean(vals)


def find_log_final_acc(method: str, seed: int, repo_root: Path) -> Optional[float]:
    """Parse the last "=> CNN: {'total': X, ...}" line from that run's log.

    ``'total'`` is the raw, single-checkpoint pooled accuracy at that task --
    NOT the log's separately-printed "Average Accuracy (CNN): Y" line, which
    is the cumulative running mean of this same 'total' value across all
    tasks so far and is a different, larger number. Always take 'total'
    directly; do not reconstruct it from the running mean.

    Patterns without a literal '{seed}' placeholder are legacy seed=2-only
    logs (filenames predate the multi-seed sweep) and are only tried for
    seed 2.
    """
    for pattern in LOG_GLOB_PATTERNS.get(method, []):
        if "{seed}" not in pattern and seed != 2:
            continue
        glob_pattern = pattern.format(seed=seed)
        matches = sorted(repo_root.glob(glob_pattern))
        if not matches:
            continue
        text = matches[0].read_text(errors="ignore")
        found = LOG_ACC_RE.findall(text)
        if found:
            return float(found[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs_logs/step_ablation"))
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Used to resolve log glob patterns.")
    args = parser.parse_args()

    # method -> seed -> parsed cl_metrics.json dict
    grouped: Dict[str, Dict[int, dict]] = {}
    for p in args.outputs_root.rglob("*_cl_metrics.json"):
        path_str = str(p)
        m = SEED_RE.search(path_str)
        if not m:
            continue
        seed = int(m.group(1))
        method = classify(path_str)
        if method is None:
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        grouped.setdefault(method, {})[seed] = data

    print("=" * 88)
    print("Stage-2 multi-seed summary")
    print("Overall_Acc = headline metric (log 'Average Accuracy (CNN)', sample-weighted, final task)")
    print("FAA/AAA/BWT_final_avg/Forget_avg = reference metrics (unweighted per-task; JSON)")
    print("=" * 88)
    for method, _ in METHOD_MARKERS:
        seeds = grouped.get(method, {})
        if not seeds:
            print(f"\n{method}: no runs found")
            continue
        print(f"\n{method}  (seeds: {sorted(seeds.keys())})")
        overall_acc_vals = [find_log_final_acc(method, seed, args.repo_root) for seed in seeds]
        print(f"  {'Overall_Acc':<15} {mean_std(overall_acc_vals)}")
        for key in CL_METRIC_KEYS:
            vals = [d["cnn"]["final"].get(key) for d in seeds.values()]
            print(f"  {key:<15} {mean_std(vals)}")

    print()
    print("=" * 88)
    print("Block B6/B7 mechanism metrics (per-seed: averaged over tasks; then mean +/- std over seeds)")
    print("=" * 88)
    for method in ("mechanism_ours", "mechanism_baseline"):
        seeds = grouped.get(method, {})
        if not seeds:
            print(f"\n{method}: no runs found")
            continue
        print(f"\n{method}  (seeds: {sorted(seeds.keys())})")
        for key in MECHANISM_TASK_KEYS:
            per_seed_vals = []
            for seed, d in seeds.items():
                steps = d.get("mechanism", {}).get("steps", {})
                v = per_seed_task_average(steps, key)
                if v is not None:
                    per_seed_vals.append(v)
            print(f"  {key:<35} {mean_std(per_seed_vals)}")


if __name__ == "__main__":
    main()
