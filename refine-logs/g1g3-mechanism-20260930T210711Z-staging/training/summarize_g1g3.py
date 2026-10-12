"""Five new runs plus four hash-pinned, read-only completed references."""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import io
import math
from pathlib import Path
import re

from common import (HERE, atomic_json, atomic_text, load_suite, now, read_json,
                    resolve, sha256)
from collect_results import collect


def audit_pure_gam(log_path, expected_tasks=10):
    text = Path(log_path).read_text(errors="replace")
    skips = [int(x) for x in re.findall(
        r"ewc_skip_task_fisher=true: skipped task-end delta-W Fisher after task (\d+)\.", text)]
    starts = re.findall(r"\[TaskStart\] task=(\d+)[^\r\n]*?has_fisher=(True|False)", text)
    if skips != list(range(expected_tasks)) or starts != [(str(t), "False") for t in range(expected_tasks)]:
        raise ValueError("Missing task Fisher skips or nonempty Fisher history")
    for marker in ("Computing raw Fisher snapshot", "Updated past Fisher", "[WeightMerge]",
                   "[WeightPath]", "[Checkpoint]", "source=post_weight_merge checkpoints="):
        if marker in text:
            raise ValueError(f"Unexpected Fisher/merge execution: {marker}")
    ewcs = [float(x) for x in re.findall(r"EWC\(λ·F·Δθ²\) ([\d.eE+-]+)", text)]
    if not ewcs or any(not math.isfinite(x) or x != 0 for x in ewcs):
        raise ValueError("Missing or nonzero EWC losses")
    mechanisms = []
    for line in text.splitlines():
        if "[AS1-NormFisher-Damp-WMergeThenFisher][Mechanism]" in line:
            mechanisms.append(ast.literal_eval(line[line.index("{"):]))
    if [row["task"] for row in mechanisms] != list(range(expected_tasks)):
        raise ValueError("Missing completed task mechanism metrics")
    zero_fields = ("train_fisher_loss_mean", "grad_fisher_norm_mean",
                   "grad_delta_w_penalty_norm_mean", "fisher_past_trace_mass",
                   "in_task_weight_merge_snapshots", "weight_merge_theta_end_to_merge_l2")
    if any(any(row.get(k) != 0 for k in zero_fields) or row.get("weight_merge_applied") is not False
           for row in mechanisms):
        raise ValueError("Nonzero Fisher/extra penalty or an applied merge")
    return {"task_fisher_skips": len(skips), "fisher_history_empty": True,
            "ewc_zero": True, "snapshot_count": 0, "weight_merge_applied": False}


def decorate(row, config, origin):
    row = dict(row, origin=origin, outer_radius=config["gam_grad_norm_rho"],
               beta1=config["gam_grad_beta_1"], beta2=config["gam_grad_beta_2"],
               beta3=config["gam_grad_beta_3"], gamma=config["gam_grad_gamma"])
    if row["status"] in {"completed", "reused"}:
        matrix = row["cnn_matrix"]
        diagonal = [matrix[t][t] for t in range(len(matrix))]
        row["acquisition_per_task"] = diagonal
        row["acquisition_task_macro"] = sum(diagonal) / len(diagonal)
    return row


def load_reused():
    reuse = read_json(HERE / "REUSED_RESULTS.json")
    for key in ("suite", "common", "source_manifest"):
        if sha256(reuse[key]) != reuse[key + "_sha256"]:
            raise ValueError(f"Reused {key} changed")
    if reuse["source_manifest_sha256"] != sha256(HERE / "source_manifest.json"):
        raise ValueError("Reused source differs from new training source")
    spec = importlib.util.spec_from_file_location("g1g3_reused_common", reuse["common"])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_suite = old.load_suite(Path(reuse["suite"]))
    old.verify_source(old_suite)
    old_jobs = {j["id"]: j for j in old_suite["jobs"]}
    rows = []
    for ref in reuse["rows"]:
        for key in ("config", "runtime_config", "receipt", "metrics_path"):
            hash_key = "metrics_sha256" if key == "metrics_path" else key + "_sha256"
            if sha256(ref[key]) != ref[hash_key]:
                raise ValueError(f"Reused {ref['id']} {key} changed")
        receipt = read_json(ref["receipt"])
        job = old_jobs[ref["id"]]
        if (receipt["status"] != "completed" or receipt["config_sha256"] != ref["config_sha256"]
                or receipt["source_manifest_sha256"] != reuse["source_manifest_sha256"]
                or job["config_sha256"] != ref["config_sha256"]):
            raise ValueError(f"Invalid reused completion: {ref['id']}")
        result = old.inspect_metrics(old_suite, job)
        if result["metrics_sha256"] != ref["metrics_sha256"] or result["metrics_sha256"] != receipt["result"]["metrics_sha256"]:
            raise ValueError(f"Reused metrics changed: {ref['id']}")
        run = Path(ref["receipt"]).parent
        if (run / "failed.json").exists() or (run / "queue_failed.json").exists():
            raise ValueError(f"Reused job has a failure receipt: {ref['id']}")
        row = {"id": ref["id"], "value": ref["value"], "seed": job["seed"],
               "status": "reused", "config_sha256": ref["config_sha256"], **result,
               "pure_gam_audit": audit_pure_gam(run / "console.log"),
               "completed_receipt": ref["receipt"]}
        rows.append(decorate(row, read_json(ref["config"]), "previous_completed"))
    return rows


def build_report(new_rows, configs):
    rows = [decorate(row, configs[row["id"]], "new_training") for row in new_rows]
    rows.extend(load_reused())
    baselines = {"sum_r1": next(row for row in rows if row["origin"] == "previous_completed"
                               and row["beta1"] == row["beta3"] == 1 and row["gamma"] == 0),
                 "g3_r1": next(row for row in rows if row["origin"] == "previous_completed"
                               and row["beta1"] == 0 and row["beta3"] == 1)}
    score_keys = ("aaa_task_macro", "faa_task_macro", "acquisition_task_macro",
                  "bwt", "forgetting", "aaa_sample_weighted", "faa_sample_weighted")
    for row in rows:
        if row["status"] not in {"completed", "reused"}:
            continue
        for label, baseline in baselines.items():
            row["delta_vs_" + label] = {key: row[key] - baseline[key] for key in score_keys}
    return {"updated_at": now(), "expected_new": 5,
            "completed_new": sum(row["status"] == "completed" for row in new_rows),
            "reused": sum(row["status"] == "reused" for row in rows), "rows": rows}


def write_report(document):
    keys = ["id", "value", "origin", "status", "outer_radius", "beta1", "beta2", "beta3", "gamma",
            "aaa_task_macro", "faa_task_macro", "acquisition_task_macro", "bwt", "forgetting",
            "aaa_sample_weighted", "faa_sample_weighted", "metrics_path", "metrics_sha256"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(document["rows"])
    atomic_text(HERE / "G1G3_COMPARISON.csv", output.getvalue())
    atomic_json(HERE / "G1G3_COMPARISON.json", document)
    lines = ["# GAM direction and outer-radius comparison", "",
             f"New completed: {document['completed_new']}/5; reused completed: {document['reused']}.",
             "Task AAA = mean of stage-wise task means; FAA = final task mean; acquisition = mean diagonal.",
             "All scores are percentages. BWT and forgetting exclude the final newly learned task.", "",
             "| Arm | Origin | Status | r | Task AAA | Task FAA | Acquisition | BWT | Forgetting | Sample AAA | Sample FAA |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in document["rows"]:
        vals = [f"{row[k]:.3f}" if row.get(k) is not None else "-" for k in keys[9:16]]
        lines.append("| " + " | ".join([row["value"], row["origin"], row["status"],
                                       f"{row['outer_radius']:g}", *vals]) + " |")
    lines += ["", "Seed 2 only; no uncertainty estimate across seeds.",
              "Frozen GAM retains all four probes and its gamma=0 decomposition arithmetic.",
              "No Fisher/EWC/weight fusion; lambda_flat remains 1. Mean direction uses beta1=beta3=0.5."]
    atomic_text(HERE / "G1G3_COMPARISON.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    suite = load_suite()
    state_path = HERE / "runtime/state.json"
    states = read_json(state_path).get("jobs", {}) if state_path.exists() else {}
    rows = collect(suite, states)["jobs"]
    for row in rows:
        if row["status"] == "completed":
            row["pure_gam_audit"] = audit_pure_gam(HERE / "runtime/jobs" / row["id"] / "console.log")
    configs = {job["id"]: read_json(resolve(job["config"])) for job in suite["jobs"]}
    document = build_report(rows, configs)
    write_report(document)
    print(f"G1G3 comparison: {document['completed_new']}/5 new, {document['reused']} reused.")
    if any(row["status"] in {"failed", "invalid"} for row in rows):
        return 1
    return 0 if args.allow_incomplete or document["completed_new"] == 5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
