"""Copied scientific training receipt checks, with the actual seed parameterized."""
import json
import math
from pathlib import Path
from contract import sha256

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise AssertionError("Nonfinite scientific evidence")
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)

def validate_job(job):
    out=Path(job["output_dir"])
    path=out/"final_summary.json"
    summary=json.loads(path.read_text())
    finite_tree(summary)
    if summary["status"]!="completed":
        raise AssertionError("Training did not complete")
    expected=(job["expected_tasks"],job["expected_epochs_per_task"],job["expected_steps"])
    if (summary["completed_tasks"],summary["epochs_per_task"],summary["completed_steps"])!=expected:
        raise AssertionError("Training budget mismatch")
    if summary["completed_epochs"]!=expected[0]*expected[1]:
        raise AssertionError("Missing epochs")
    if summary["attention_backend"]!=job["backend"] or summary["probe_branch"]!=job["branch"]:
        raise AssertionError("Backend/branch mismatch")
    if summary["update_direction"]!=job["direction"] or summary["step_mode"]!=job["mode"] or summary["seed"]!=job["seed"] or summary["r"]!=job["outer_radius"] or summary["rho"]!=job["inner_radius"]:
        raise AssertionError("Wrong scientific condition")
    c=summary["counters"]
    if c["training_ce_gradient_passes"]!=4*expected[2]:
        raise AssertionError("Ordinary four-probe budget mismatch")
    if any(c[k] for k in ("fisher_estimation_passes","weight_merge_updates","forbidden_work_attempts")):
        raise AssertionError("Forbidden Fisher/weight work")
    checks=summary["checks"]
    required={"finite","all_steps_self_spec","exact_snapshot_restore","four_base_probes",
              "fisher_empty_all_tasks","actual_weight_merges_zero"}
    if set(checks)!=required or not all(checks.values()):
        raise AssertionError("Scientific checks failed")
    hp=int(job["direction"] in ("hstart","hmid"))
    ref=int(job["direction"]=="hmid")
    if c["training_hvp_passes"]!=hp*expected[2] or c["training_create_graph_passes"]!=hp*expected[2] or c["training_hvp_reference_gradient_passes"]!=ref*expected[2]:
        raise AssertionError("HVP compute accounting differs")
    if summary["step_trace_sha256"]!=sha256(out/"step_trace.jsonl") or summary["task_matrix_sha256"]!=sha256(out/"task_matrix.json"):
        raise AssertionError("Result receipt hashes differ")
    rows=[json.loads(x) for x in (out/"step_trace.jsonl").read_text().splitlines()]
    if len(rows)!=expected[2]:
        raise AssertionError("Trace count differs")
    batches=[1]*expected[0] if job["smoke"] else [10,8,8,5,4,3,3,3,4,3]
    coordinates=[(t,e,b) for t in range(expected[0]) for e in range(1,expected[1]+1) for b in range(batches[t])]
    for i,(row,coord) in enumerate(zip(rows,coordinates)):
        finite_tree(row)
        if row["radius"] != job["outer_radius"] or row["rho"] != job["inner_radius"]:
            raise AssertionError("Actual per-step probe radii differ")
        if any(row[k] for k in ("fisher_calls", "weight_merge_calls", "gradient_projection_calls")):
            raise AssertionError("Forbidden per-step work")
        if row["step_index"]!=i or row["sgd_updates"]!=1 or not row["base_parameters_exact"] or not row["base_buffers_exact"]:
            raise AssertionError("Invalid actual step")
        if (row["task"],row["epoch"],row["batch"])!=coord or row["seed"]!=job["seed"]:
            raise AssertionError("Training task/epoch/batch coverage differs")
        if (row["probe_branch"],row["update_direction"],row["step_mode"],row["attention_backend"])!=(job["branch"],job["direction"],job["mode"],job["backend"]):
            raise AssertionError("Actual step scientific condition differs")
        if row["training_ce_gradient_passes"]!=4 or row["training_hvp_passes"]!=hp or row["training_create_graph_passes"]!=hp or row["training_hvp_reference_gradient_passes"]!=ref:
            raise AssertionError("Step derivative budget differs")
        if row["class_slice"]!=[20*coord[0],20*(coord[0]+1)] or row["seen_classes"]!=20*(coord[0]+1):
            raise AssertionError("Step class slice differs")
        if not row["state_restoration"] or not all(row["state_restoration"].values()):
            raise AssertionError("Incomplete step state restoration")
        if not isinstance(row["train_ce"],(int,float)) or not math.isfinite(row["train_ce"]):
            raise AssertionError("Nonfinite actual CE")
        if hp:
            observer=row["observer"]
            if observer["status"]!="passed" or observer["backend"]!=job["backend"] or observer["same_point_gradient"]["relative_l2"]>2e-5:
                raise AssertionError("Unverified HVP actual update")
            if observer["second_derivative_passes"]!=1 or not observer["direction_detached"]:
                raise AssertionError("HVP direction/compute mismatch")
        if job["mode"]=="matched" and (not row["matched"] or row["actual_match_relative_error"]>1e-3):
            raise AssertionError("Actual FP32 step was not matched")
    if sum(row["batch_samples"] for row in rows) != job["expected_samples"]:
        raise AssertionError("Sample exposure budget differs")
    if summary["input_config_sha256"] != job["config_sha256"]:
        raise AssertionError("Actual input configuration changed")
    matrix=json.loads((out/"task_matrix.json").read_text())["matrix"]
    if len(matrix)!=expected[0] or any(len(row)!=i+1 for i,row in enumerate(matrix)):
        raise AssertionError("Observed task matrix incomplete")
    if not all(math.isfinite(v) and 0<=v<=100 for row in matrix for v in row):
        raise AssertionError("Invalid observed accuracy")
    return dict(summary_sha256=sha256(path),trace_sha256=summary["step_trace_sha256"],
                matrix_sha256=summary["task_matrix_sha256"],updates=len(rows),tasks=len(matrix))
