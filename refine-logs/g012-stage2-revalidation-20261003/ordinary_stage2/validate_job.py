"""Completion is based on actual verified updates, not process exit alone."""
import json
import math
from pathlib import Path
from common import sha256

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise AssertionError("Nonfinite scientific evidence")
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)

def validate_hvp_gate(report,job):
    finite_tree(report)
    if report["status"]!="passed" or report.get("full_stage2_allowed") is not True:
        raise AssertionError("Full real-ViT HVP gate did not pass")
    if report["backend"]!=job["backend"] or report["seed"]!=2 or report["initial_feasibility_only"]:
        raise AssertionError("HVP gate backend/seed/reference mismatch")
    if report["optimizer_updates"]!=0 or set(report["tasks"])!={"0","4","9"}:
        raise AssertionError("Wrong full-gate tasks or an update occurred")
    grid=[.1,.03,.01,.003,.001]
    if report["h_grid"]!=grid or report["same_point_gradient_relative_max"]!=2e-5 or report["fd_relative_max"]!=5e-3 or report["fd_cosine_min"]!=.999:
        raise AssertionError("Frozen numerical criteria changed")
    for key,task in report["tasks"].items():
        if task["class_range"]!=[20*int(key),20*(int(key)+1)] or len(task["batches"])!=2 or not task["checkpoint_sha256"]:
            raise AssertionError("Missing predetermined checkpoint/batch evidence")
        for batch in task["batches"]:
            if batch["status"]!="passed" or batch["backend"]!=job["backend"] or batch["samples"]!=128:
                raise AssertionError("Unverified real HVP batch")
            if batch["optimizer_updates"] or batch["fisher_calls"] or batch["weight_merge_calls"]:
                raise AssertionError("HVP diagnostics performed forbidden work")
            if not batch["entry_state_restore"] or not all(batch["entry_state_restore"].values()):
                raise AssertionError("HVP gate did not restore all state")
            if set(batch["responses"])!={"hstart","hmid"}:
                raise AssertionError("Incomplete response candidates")
            for response in batch["responses"].values():
                if response["status"]!="passed":
                    raise AssertionError("A local response did not pass")
                if response["hvp"]["backend"]!=job["backend"] or response["hvp"]["same_point_gradient"]["relative_l2"]>2e-5:
                    raise AssertionError("Same-point derivative disagreement")
                if not all(response["hvp"]["entry_state_restore"].values()):
                    raise AssertionError("HVP response restoration failed")
                scan=response["scan"]
                if [s["h"] for s in scan]!=grid:
                    raise AssertionError("Center-difference grid changed")
                accepted=[]
                for row in scan:
                    stats=row["central_vs_hvp"]
                    accepted.append(stats["relative_l2"] is not None and stats["relative_l2"]<=5e-3
                                    and stats["cosine"] is not None and stats["cosine"]>=.999)
                if not any(a and b for a,b in zip(accepted,accepted[1:])):
                    raise AssertionError("No two adjacent accepted finite-difference points")

def validate_job(job):
    out=Path(job["output_dir"])
    if job["kind"]=="diagnostic":
        path=out/job["report_file"]
        report=json.loads(path.read_text())
        finite_tree(report)
        if report["status"]!="passed" or report["diagnostic_backend"]!=job["backend"] or report["seed"]!=2:
            raise AssertionError("Mechanism diagnostic condition/status mismatch")
        if set(report["tasks"])!={0,1,4,9} or set(report["points"])!={"entry","epoch1","end"}:
            raise AssertionError("Mechanism observation coverage changed")
        if report["state_batch_count"]!=24 or report["expected_state_batch_count"]!=24 or len(report["state_results"])!=12:
            raise AssertionError("Incomplete mechanism observations")
        if any(report[k] for k in ("optimizer_updates","fisher_calls","weight_merge_calls")):
            raise AssertionError("Mechanism diagnostic performed forbidden work")
        if not all(report[k] for k in ("source_unchanged","reference_assets_unchanged","diagnostic_code_unchanged","native_origin_restored")):
            raise AssertionError("Mechanism assets/restoration check failed")
        if sha256(out/"batch_registry.json")!=report["batch_registry_sha256"]:
            raise AssertionError("Mechanism registry changed")
        seen=set()
        for state in report["state_results"]:
            key=(state["task"],state["point"])
            if key in seen or len(state["batches"])!=2:
                raise AssertionError("Repeated/missing mechanism observation")
            seen.add(key)
            for row in state["batches"]:
                if row["status"]!="passed" or not all(row["entry_state_restore"].values()) or not row["trainable_values_unchanged"]:
                    raise AssertionError("Mechanism batch failed restoration")
                if row["backend"]!=job["backend"] or row["identity"]["samples"]!=128 or any(row[k] for k in ("optimizer_updates","fisher_calls","weight_merge_calls")):
                    raise AssertionError("Mechanism actual condition differs")
                if job["response"]:
                    if row["hvp_calls"]!=2 or set(row["responses"])!={"Sec","Hstart","Hmid"}:
                        raise AssertionError("Missing mechanism response candidates")
                elif set(row["branches"])!={"P","D","C","R"} or not row["exact_same_parameter_batch_rng"]:
                    raise AssertionError("Incomplete four-cell comparison")
        if seen!={(t,p) for t in (0,1,4,9) for p in ("entry","epoch1","end")}:
            raise AssertionError("Wrong mechanism coordinates")
        return dict(summary_sha256=sha256(path),gate="passed",state_batches=24)
    if job["kind"]=="hvp_gate":
        path=out/"hvp_gate_report.json"
        report=json.loads(path.read_text())
        validate_hvp_gate(report,job)
        return dict(summary_sha256=sha256(path),gate="passed")
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
    if summary["update_direction"]!=job["direction"] or summary["step_mode"]!=job["mode"] or summary["seed"]!=2 or summary["r"]!=.25 or summary["rho"]!=.05:
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
        if row["step_index"]!=i or row["sgd_updates"]!=1 or not row["base_parameters_exact"] or not row["base_buffers_exact"]:
            raise AssertionError("Invalid actual step")
        if (row["task"],row["epoch"],row["batch"])!=coord or row["seed"]!=2:
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
    matrix=json.loads((out/"task_matrix.json").read_text())["matrix"]
    if len(matrix)!=expected[0] or any(len(row)!=i+1 for i,row in enumerate(matrix)):
        raise AssertionError("Observed task matrix incomplete")
    if not all(math.isfinite(v) and 0<=v<=100 for row in matrix for v in row):
        raise AssertionError("Invalid observed accuracy")
    return dict(summary_sha256=sha256(path),trace_sha256=summary["step_trace_sha256"],
                matrix_sha256=summary["task_matrix_sha256"],updates=len(rows),tasks=len(matrix))
