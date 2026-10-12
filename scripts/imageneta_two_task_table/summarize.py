#!/usr/bin/env python3
"""Validate six paired ImageNet-A variants and export the two-task tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


VARIANTS = ("vanilla", "shaping_only", "ewc", "gam_ewc", "task0_shaping", "task1_shaping")
RULES = {
    "vanilla": ("sgd", "sgd", False, "Vanilla LoRA", "SGD", "SGD"),
    "shaping_only": ("gam", "gam", False, "Shaping-only LoRA", "GAM", "GAM"),
    "ewc": ("sgd", "sgd", True, "EWCLoRA", "SGD", "SGD+EWC"),
    "gam_ewc": ("gam", "gam", True, "Shaping + EWCLoRA", "GAM", "GAM+EWC"),
    "task0_shaping": ("gam", "sgd", True, "EWCLoRA + Task-1 shaping", "GAM", "SGD+EWC"),
    "task1_shaping": ("sgd", "gam", True, "EWCLoRA + Task-2 shaping", "SGD", "GAM+EWC"),
}
FIELDS = ("a11", "a21", "a22", "bwt", "B1", "B2", "old_train_loss_delta",
          "old_test_loss_delta", "r_eff", "old_trace_task0", "old_trace_task1", "new_trace_task1")
# EWC strength and optimizer/shaping switches vary by design; the rest must match.
COMMON_CONFIG = (
    "dataset", "class_shuffle", "init_cls", "increment", "lora_rank",
    "comparison_base_optimizer", "init_lr", "lrate", "init_epoch", "epochs",
    "batch_size", "init_momentum", "momentum", "init_weight_decay", "weight_decay",
    "ewc_gamma", "fisher_mode", "ewc_max_batches", "ewc_fisher_source",
    "ewc_normalize_fisher", "ewc_normalize_accumulated_fisher", "ewc_fisher_damping_beta",
    "frozen_backbone_checkpoint", "snapshot_epochs", "in_task_path_weighted",
    "in_task_path_beta", "in_task_path_start_ratio", "in_task_weight_merge_omega", "trajectory_merge_policy",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_signature(run):
    rows = [json.loads(line) for line in (run / "rng_receipts.jsonl").read_text().splitlines() if line.strip()]
    keys = {
        "head_initialized": ("task", "old_classes", "total_classes", "new_head_sha256"),
        "loader_iteration": ("task", "loader_serial", "iteration", "generator_seed", "samples",
                             "indices_sha256", "first_batch_pixels_targets_sha256"),
    }
    result = [{"event": row["event"], **{key: row[key] for key in keys[row["event"]]}}
              for row in rows if row["event"] in keys]
    for task in (0, 1):
        if sum(row["event"] == "head_initialized" and row["task"] == task for row in result) != 1:
            raise ValueError(f"Need exactly one replayed/current head receipt per task: {run}")
        if not any(row["event"] == "loader_iteration" and row["task"] == task for row in result):
            raise ValueError(f"Missing Task{task} training-data receipts: {run}")
    return result


def validate_variant(variant, config):
    expected = RULES[variant]
    if config["stage0_optimizer"] != expected[0] or config["stage1_optimizer"] != expected[1]:
        raise ValueError(f"Wrong update-rule identity: {variant}")
    strength = float(config["ewc_lambda"])
    if (expected[2] and strength <= 0) or (not expected[2] and strength != 0):
        raise ValueError(f"Wrong EWC switch: {variant}")
    if config["dataset"] != "imageneta" or config["comparison_base_optimizer"] != "sgd":
        raise ValueError("Only the ImageNet-A SGD-base protocol is supported")
    if config["init_cls"] != 20 or config["increment"] != 20 or config["lora_rank"] != 10:
        raise ValueError("Expected rank10 continuous Q/V LoRA, 20+20 classes")


def collect(root, seeds, allow_smoke=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one distinct seed is required")
    records, checks, by_seed = [], [], {}
    reference_across_seeds = None
    for seed in seeds:
        group = {}
        for variant in VARIANTS:
            run = root / variant / f"seed{seed}"
            marker = read_json(run / "COMPLETE.json")
            if marker.get("status") != "complete" or marker.get("completed_tasks") != 2:
                raise ValueError(f"Training incomplete: {run}")
            path = run / "evaluation/summary.json"
            evaluation = read_json(path)
            receipt = read_json(path.parent / "COMPLETE.json")
            if (evaluation.get("status") != "complete" or receipt.get("status") != "complete"
                    or receipt.get("summary_sha256") != sha_file(path)):
                raise ValueError(f"Evaluation incomplete or changed: {path}")
            if evaluation.get("variant", evaluation.get("method")) != variant or evaluation["seed"] != seed:
                raise ValueError(f"Variant/seed mismatch: {path}")
            if evaluation.get("smoke", False) and not allow_smoke:
                raise ValueError("Smoke results cannot produce paper tables; use --allow-smoke for sanity only")
            config = evaluation["inputs"]["config"]
            validate_variant(variant, config)
            for field in FIELDS:
                if not isinstance(evaluation[field], (int, float)) or not math.isfinite(evaluation[field]):
                    raise ValueError(f"Invalid {field}: {path}")
            if not math.isclose(evaluation["bwt"], evaluation["a21"] - evaluation["a11"], abs_tol=1e-10):
                raise ValueError(f"BWT mismatch: {path}")
            for task in (0, 1):
                proxy = evaluation["endpoints"][task]["current_task_batch_fisher_proxy"]
                if proxy["logical_batch_size"] != 128 or proxy["normalized"] or proxy["floor"] is not None:
                    raise ValueError(f"Wrong batch Fisher proxy protocol: {path}")
            records.append({"variant": variant, "seed": seed,
                            **{field: evaluation[field] for field in FIELDS}, "source": str(path)})
            group[variant] = (run, evaluation)
        base = group["vanilla"][1]
        base_config = base["inputs"]["config"]
        base_rng = rng_signature(group["vanilla"][0])
        ewc_strengths = set()
        for variant, (run, evaluation) in group.items():
            if bool(evaluation.get("smoke", False)) != bool(base.get("smoke", False)):
                raise ValueError("Cannot combine smoke and formal runs")
            config = evaluation["inputs"]["config"]
            for field in COMMON_CONFIG:
                if config[field] != base_config[field]:
                    raise ValueError(f"Unmatched common config {field}: seed{seed}/{variant}")
            if RULES[variant][2]:
                ewc_strengths.add(float(config["ewc_lambda"]))
            for field in ("code_sha256", "source_sha256", "data_manifest_sha256", "precision"):
                if evaluation["inputs"][field] != base["inputs"][field]:
                    raise ValueError(f"Unmatched evaluation {field}: seed{seed}/{variant}")
            if evaluation["checkpoint_audit"]["frozen_state_sha256"] != base["checkpoint_audit"]["frozen_state_sha256"]:
                raise ValueError(f"Different frozen backbone: seed{seed}/{variant}")
            for task, scope in ((0, "old_local20"), (1, "old_local20"), (1, "new_local20")):
                left = base["endpoints"][task]["fisher"][scope]
                right = evaluation["endpoints"][task]["fisher"][scope]
                if left["input_tensors_sha256"] != right["input_tensors_sha256"]:
                    raise ValueError(f"EF inputs differ: seed{seed}/{variant}/task{task}/{scope}")
            for task in (0, 1):
                left = base["endpoints"][task]["current_task_batch_fisher_proxy"]
                right = evaluation["endpoints"][task]["current_task_batch_fisher_proxy"]
                if left["input_tensors_sha256"] != right["input_tensors_sha256"]:
                    raise ValueError(f"Batch-proxy input/order differs: seed{seed}/{variant}/task{task}")
            if rng_signature(run) != base_rng:
                raise ValueError(f"Head/data RNG receipts differ: seed{seed}/{variant}")
        if len(ewc_strengths) != 1:
            raise ValueError(f"Different EWC strengths among protected variants: seed{seed}")
        for optimizer in ("sgd", "gam"):
            hashes = {group[v][1]["checkpoint_audit"]["checkpoints"][0]["sha256"]
                      for v in VARIANTS if RULES[v][0] == optimizer}
            if len(hashes) != 1:
                raise ValueError(f"Task0 {optimizer} anchor not shared: seed{seed}")
        # Class order and fixed Fisher samples must also be common across seeds.
        cross_signature = {
            "config": {key: base_config[key] for key in COMMON_CONFIG},
            "ewc_lambda": next(iter(ewc_strengths)),
            "code": base["inputs"]["code_sha256"], "source": base["inputs"]["source_sha256"],
            "data": base["inputs"]["data_manifest_sha256"],
            "frozen": base["checkpoint_audit"]["frozen_state_sha256"],
            "fisher_samples": {"old": base["endpoints"][0]["fisher"]["old_local20"]["samples"],
                               "new": base["endpoints"][1]["fisher"]["new_local20"]["samples"]},
            "smoke": bool(base.get("smoke", False)),
        }
        if reference_across_seeds is not None and reference_across_seeds != cross_signature:
            raise ValueError(f"Common protocol/data/backbone differs across seeds: seed{seed}")
        reference_across_seeds = cross_signature
        checks.append({"seed": seed, "six_variants_complete": True, "common_protocol": True,
                       "shared_task0_within_optimizer_groups": True, "same_frozen_backbone": True,
                       "same_Fisher_inputs_and_proxy_batch_order": True, "same_head_data_RNG": True})
        by_seed[seed] = group
    aggregate = []
    for variant in VARIANTS:
        for field in FIELDS:
            values = [row[field] for row in records if row["variant"] == variant]
            aggregate.append({"variant": variant, "metric": field, "n": len(values),
                              "mean": statistics.mean(values),
                              "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    comparisons = (("shaping_only", "vanilla"), ("ewc", "vanilla"), ("gam_ewc", "ewc"),
                   ("task0_shaping", "ewc"), ("task1_shaping", "ewc"))
    differences = {}
    for method, reference in comparisons:
        key = f"{method}_minus_{reference}"
        differences[key] = {}
        for field in FIELDS:
            values = [by_seed[seed][method][1][field] - by_seed[seed][reference][1][field] for seed in seeds]
            differences[key][field] = {"values": values, "mean": statistics.mean(values),
                                        "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
    return records, aggregate, checks, differences, reference_across_seeds


def latex_tables(aggregate, seeds, config, fisher_samples):
    def value(variant, metric, show_sd):
        cell = next(row for row in aggregate if row["variant"] == variant and row["metric"] == metric)
        mean, sd = cell["mean"], cell["sample_sd"]
        return f"${mean:.3f}\\pm{sd:.3f}$" if show_sd and sd is not None else f"${mean:.3f}$"

    policy = str(config["trajectory_merge_policy"]).replace("_", r"\_")
    common = (
        r"ImageNet-A with two tasks of 20 classes, ViT-B/16 and continuous rank-10 Q/V LoRA. "
        r"All GAM updates use SGD as the base optimizer; update rules are listed explicitly. "
        r"Endpoints are the saved outputs after the configured task-trajectory factor mixing "
        r"(policy \texttt{" + policy + r"}); only A/B factors are mixed, not classifier heads. "
        r"Accuracies use the seen-class head and are percentages; BWT is in percentage points. "
        r"$r_{QV}=\left[\sum_{\ell,c\in\{Q,V\}}\|B^{(2)}_{\ell c}A^{(2)}_{\ell c}"
        r"-B^{(1)}_{\ell c}A^{(1)}_{\ell c}\|_F^2\right]^{1/2}$ excludes classifier parameters. "
        + f"Means over {len(seeds)} training seed(s). "
    )
    proxy_caption = (
        r"$\mathcal B_j=N_j^{-1}\sum_b n_b\|\nabla_{W^{\mathrm{eff}}_{QV}}\bar\ell_b\|_2^2$ "
        r"is the current-task batch-mean-gradient proxy at endpoint $j$, using its local 20-class head. "
        r"It uses all current-task training images with deterministic test preprocessing and fixed "
        r"contiguous batches of 128 (the final short batch is weighted by its sample count). "
        r"No trace rescaling or positive floor is applied. It is not per-example empirical Fisher, "
        r"and $\mathcal B_2$ does not measure old-task sensitivity at the second endpoint. "
    )
    output = {}
    for show_sd, filename in ((False, "legacy_layout.tex"), (True, "comparison.tex")):
        lines = [r"% Requires \usepackage{booktabs,amsmath}.", r"\begin{table*}[t]",
                 r"\centering\scriptsize", r"\caption{" + common + proxy_caption +
                 (r"Sample standard deviations are shown when available." if show_sd else "") + "}",
                 r"\setlength{\tabcolsep}{2.5pt}",
                 r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
                 r"\toprule", r"& \multicolumn{2}{c}{Update rule} & \multicolumn{2}{c}{Task 1 endpoint} "
                 r"& \multicolumn{4}{c}{Task 2 endpoint} & Drift \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}",
                 r"Variant & Task 1 & Task 2 & $a_{11}\uparrow$ & $\mathcal B_1$ & $a_{21}\uparrow$ "
                 r"& $a_{22}\uparrow$ & BWT$\uparrow$ & $\mathcal B_2$ & $r_{QV}$ \\", r"\midrule"]
        for variant in VARIANTS:
            if variant == "task0_shaping":
                lines.append(r"\midrule")
            rule = RULES[variant]
            lines.append(" & ".join([rule[3], rule[4], rule[5]] +
                                    [value(variant, field, show_sd) for field in
                                     ("a11", "B1", "a21", "a22", "bwt", "B2", "r_eff")]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
        output[filename] = "\n".join(lines) + "\n"
    enhanced_caption = (
        r"$\tau_{k|t}=N_k^{-1}\sum_{i\in S_k}\|\nabla_{W^{\mathrm{eff}}_{QV}}\ell_i(\theta_t)\|_2^2$ "
        r"is raw per-example empirical Fisher trace with task $k$'s local 20-class head, "
        r"excluding head parameters. $S_k$ is a fixed proportional stratified training subset "
        + f"({int(fisher_samples['old'])} old-task and {int(fisher_samples['new'])} new-task images), "
        r"shared across variants and seeds, with deterministic test preprocessing. "
        r"No trace rescaling or positive floor is applied. $D_{\rm old}$ is signed old-task "
        r"local-20 CE at the second endpoint minus that at the first; negative values are retained. "
        r"Sample standard deviations are shown when available."
    )
    lines = [r"% Requires \usepackage{booktabs,amsmath}.", r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\caption{" + common + enhanced_caption + "}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Variant & $D_{\rm old}^{\rm train}$ & $D_{\rm old}^{\rm test}$ & $r_{QV}$ "
             r"& $\tau_{1|1}$ & $\tau_{1|2}$ & $\tau_{2|2}$ \\", r"\midrule"]
    for variant in VARIANTS:
        lines.append(RULES[variant][3] + " & " + " & ".join(value(variant, field, True) for field in
                     ("old_train_loss_delta", "old_test_loss_delta", "r_eff", "old_trace_task0",
                      "old_trace_task1", "new_trace_task1")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    output["enhanced.tex"] = "\n".join(lines) + "\n"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Permit smoke runs for sanity JSON/CSV only; never generate paper LaTeX")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    root = args.output_root.resolve()
    rows, aggregate, checks, differences, protocol = collect(root, seeds, args.allow_smoke)
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    smoke = protocol["smoke"]
    prefix = "sanity_" if smoke else ""
    (output / "COMPLETE.json").unlink(missing_ok=True)
    for name, values in ((prefix + "per_seed.csv", rows), (prefix + "mean_sd.csv", aggregate)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
               "dataset": "imageneta", "seeds": seeds, "variants": list(VARIANTS),
               "smoke": smoke, "paper_evidence_allowed": not smoke,
               "pair_checks": checks, "per_seed": rows, "statistics": aggregate,
               "paired_differences": differences, "common_protocol": protocol,
               "statistical_scope": ("SMOKE SANITY ONLY; not paper evidence" if smoke else
                                     "single-seed case study" if len(seeds) == 1 else "training-seed statistics")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not smoke:
        for name, contents in latex_tables(aggregate, seeds, protocol["config"], protocol["fisher_samples"]).items():
            (output / name).write_text(contents)
    receipt = {"status": "complete", "smoke": smoke, "paper_evidence_allowed": not smoke,
               "summary": str(output / "summary.json"), "summary_sha256": sha_file(output / "summary.json"),
               "seeds": seeds, "variants": list(VARIANTS), "runs": len(rows)}
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Validated six-variant results: {output}")


if __name__ == "__main__":
    main()
