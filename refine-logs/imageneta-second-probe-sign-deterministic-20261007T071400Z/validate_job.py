"""Accept only complete, pinned, correctly signed four-probe training evidence."""
import json
import math
import hashlib
from pathlib import Path
import subprocess

from common import HERE, PROTOCOL, bad_loss_lines, read_json, sha256, snapshot_rng_fingerprint, verify_config


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("Nonfinite observed evidence")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child)


def metrics(matrix):
    n = len(matrix)
    result = dict(AAA_task_macro=sum(sum(row)/len(row) for row in matrix)/n,
                  FAA_task_macro=sum(matrix[-1])/n,
                  LA=sum(matrix[index][index] for index in range(n))/n)
    result["BWT"] = sum(matrix[-1][index]-matrix[index][index] for index in range(n-1))/(n-1) if n > 1 else 0.
    result["forgetting"] = sum(max(matrix[stage][index] for stage in range(index, n))-matrix[-1][index]
                               for index in range(n-1))/(n-1) if n > 1 else 0.
    return result


def validate_second_probe(row, requested_sign):
    """Check the actual rho*N(g2) length, including the epsilon/zero-gradient case."""
    probe = row["second_probe_parameter_audit"]
    if probe["passed"] is not True or probe["requested_sign"] != requested_sign or probe["target_radius"] != row["rho"]:
        raise AssertionError("Second probe geometry/sign audit failed")
    norm2, rho, eps = row["norms"]["g2"], row["rho"], row["eps"]
    if norm2 < 0 or eps <= 0:
        raise AssertionError("Invalid norm or normalization epsilon")
    expected = rho*norm2/(norm2+eps)
    tolerance = max(1e-8, rho*1e-4)
    degenerate = expected <= 1e-24
    if abs(probe["expected_parameter_length"]-expected) > 1e-15 or probe["absolute_tolerance"] != tolerance or probe["cosine_tolerance"] != 1e-4:
        raise AssertionError("Second-probe audit normalization/tolerance differs")
    if probe["degenerate"] is not degenerate or abs(probe["parameter_length"]-expected) > tolerance:
        raise AssertionError("Second probe is not the requested normalized radius")
    if abs(probe["length_error"]-abs(probe["parameter_length"]-expected)) > 1e-15:
        raise AssertionError("Second-probe length-error evidence differs")
    if degenerate:
        if probe["parameter_length"] > 1e-24 or probe["alignment_cosine"] is not None:
            raise AssertionError("Zero-gradient probe must have zero displacement/undefined alignment")
    elif probe["alignment_cosine"] is None or abs(probe["alignment_cosine"]-requested_sign) > 1e-4:
        raise AssertionError("Second probe has the wrong signed alignment")


def validate_reuse(suite, item):
    raise AssertionError("Historical endpoints cannot enter deterministic main results")


def validate_execution_flags(record):
    if record.get("deterministic_enabled") is not True or record.get("deterministic_warn_only") is not False or record.get("cublas_workspace_config") != ":4096:8":
        raise AssertionError("Actual deterministic execution flags differ")


def validate_same_state_snapshot(job, summary, trace):
    """Verify pinned CPU tensors against the initial realized probe evidence."""
    reference = summary["same_state_snapshot"]
    if not job["smoke"]:
        if reference is not None or any(row.get("same_state_snapshot") is not None for row in trace):
            raise AssertionError("Formal training must not save smoke-only tensors")
        return None
    out = Path(job["output_dir"]).resolve()
    if not isinstance(reference, dict) or reference != trace[0].get("same_state_snapshot"):
        raise AssertionError("Smoke snapshot metadata differs from its first training step")
    path = Path(reference["path"])
    if path.resolve() != out / "smoke_g012_snapshot.pt" or reference.get("schema_version") != 1 or (reference.get("task"), reference.get("step_index")) != (0, 0):
        raise AssertionError("Unexpected smoke snapshot source/coordinates")
    if sha256(path) != reference["sha256"]:
        raise AssertionError("Smoke tensor snapshot changed")
    # This is our hash-pinned, locally generated artifact. RNG objects require
    # full loading; CUDA tensors are always mapped to CPU for this gate.
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["schema_version"], payload["task"], payload["step_index"]) != (1, 0, 0):
        raise AssertionError("Tensor snapshot payload coordinates differ")
    if payload["protocol"] != PROTOCOL:
        raise AssertionError("Tensor snapshot execution protocol differs")
    fields = ("first_probe_sign", "second_probe_sign", "second_difference_sign", "update_base", "second_difference_weight")
    if payload["identity"] != trace[0]["identity"] or payload["factors"] != {key:summary[key] for key in fields}:
        raise AssertionError("Tensor snapshot inputs/factors differ from training")
    names = payload["parameter_names"]
    if not names or len(names) != len(set(names)) or not all(isinstance(name, str) for name in names):
        raise AssertionError("Tensor snapshot parameter order is incomplete")
    for group, fingerprint_key, labels in (("grads", "g012_fingerprints", ("g0", "g1", "g2")),
                                           ("points", "point012_fingerprints", ("phi0", "phi1", "phi2"))):
        if set(payload[group]) != set(labels):
            raise AssertionError("Incomplete tensor snapshot probes")
        fingerprints = {}
        for label in labels:
            values = payload[group][label]
            if len(values) != len(names) or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.isfinite(value).all().item() for value in values):
                raise AssertionError("Invalid tensor snapshot vector")
            fingerprints[label] = [hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest() for value in values]
        if fingerprints != payload[fingerprint_key] or fingerprints != trace[0][fingerprint_key]:
            raise AssertionError("Tensor snapshot differs from exact training fingerprints")
    if any(payload["grads"][label][index].shape != payload["points"]["phi0"][index].shape or
           payload["grads"][label][index].dtype != payload["points"]["phi0"][index].dtype
           for label in ("g0", "g1", "g2") for index in range(len(names))):
        raise AssertionError("Tensor snapshot gradient/parameter shapes or dtypes differ")
    for key in ("rng0", "rng1"):
        state = payload[key]
        if not isinstance(state, tuple) or len(state) != 4 or not isinstance(state[0], tuple) or not isinstance(state[1], tuple) or len(state[1]) != 5 or not isinstance(state[2], torch.Tensor) or state[2].dtype != torch.uint8 or state[2].device.type != "cpu" or not isinstance(state[3], list):
            raise AssertionError("Smoke snapshot lacks original probe RNG states")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 for value in state[3]):
            raise AssertionError("Invalid stored CUDA RNG state")
    layout = [(name, str(value.dtype), list(value.shape)) for name,value in zip(names,payload["points"]["phi0"])]
    return dict(path=str(path), sha256=reference["sha256"], schema_version=1, verified=True,
                rng0_sha256=snapshot_rng_fingerprint(payload["rng0"]), rng1_sha256=snapshot_rng_fingerprint(payload["rng1"]),
                parameter_layout_sha256=hashlib.sha256(json.dumps(layout).encode()).hexdigest())


def validate_side_diagnostic(row, requested_sign):
    """Selected side is reused; only the opposite side is independently measured."""
    selected, opposite = ("plus", "minus") if requested_sign == 1 else ("minus", "plus")
    expected_counts = dict(diagnostic_gradient_calls=1, diagnostic_forward_count=1, diagnostic_sgd_updates=0,
                           diagnostic_g3_positive_gradient_calls=int(requested_sign == -1),
                           diagnostic_g3_negative_gradient_calls=int(requested_sign == 1),
                           training_selected_probe_sign=requested_sign, diagnostic_opposite_probe_sign=-requested_sign,
                           selected_side_new_gradient_calls=0, opposite_side_new_gradient_calls=1,
                           diagnostic_requested_side_gradient_calls=0, diagnostic_opposite_side_gradient_calls=1)
    if any(row[key] != value for key, value in expected_counts.items()):
        raise AssertionError("Diagnostic side work was not separately accounted")
    if row["selected_side_reused_from_training"] is not True or row["selected_side_reference_exact"] is not True or row["selected_side_gradient_source"] != "training_context_clone" or row["selected_side_exact_kind"] != "training_reference_clone_equality_not_independent_replay":
        raise AssertionError("Diagnostic selected-side provenance differs")
    if row["measurement_sources"] != {selected:"training_g3", opposite:"isolated_opposite_probe"}:
        raise AssertionError("Diagnostic measurement sources are mislabeled")
    if not row["state_restoration"] or not all(value is True for value in row["state_restoration"].values()):
        raise AssertionError("Diagnostics changed training state")
    if any(row[key] for key in ("fisher_calls", "weight_merge_calls", "gradient_projection_calls", "hvp_calls", "create_graph_calls")):
        raise AssertionError("Forbidden diagnostic operation")


def validate_job(job):
    out = Path(job["output_dir"])
    if job["kind"] == "pairing":
        path = out / job["report_file"]
        report = read_json(path)
        finite_tree(report)
        if report.get("status") != "passed" or report.get("smoke_count") != 16 or report.get("formal_count") != 16 or report.get("protocol") != PROTOCOL:
            raise AssertionError("Incomplete actual-GPU smoke pairing gate")
        required = {"all_smokes_accepted", "initial_parameters_equal", "initial_head_equal",
                    "paired_realized_batches", "class_slices_equal", "four_training_ce_per_step",
                    "same_radius_initial_g012", "same_radius_initial_points", "radius_factor_grid",
                    "same_state_tensor_snapshots_verified", "one_opposite_diagnostic_ce_per_task",
                    "initial_frozen_parameters_equal", "initial_rng_all_streams_equal", "initial_parameter_layout_equal"}
        if not required <= set(report["checks"]) or not all(value is True for value in report["checks"].values()):
            raise AssertionError("Smoke pairing gate failed")
        return dict(summary_sha256=sha256(path), gate="passed")
    config = verify_config(job)
    path = out / "final_summary.json"
    summary = read_json(path)
    finite_tree(summary)
    validate_execution_flags(summary)
    actual_flags = {key:summary[key] for key in ("deterministic_enabled", "deterministic_warn_only", "cublas_workspace_config")}
    if summary.get("protocol") != PROTOCOL or summary.get("execution_flags") != actual_flags:
        raise AssertionError("Summary deterministic protocol/flag receipt differs")
    expected = (job["expected_tasks"], job["expected_epochs_per_task"], job["expected_steps"])
    if summary["status"] != "completed" or (summary["completed_tasks"], summary["epochs_per_task"], summary["completed_steps"]) != expected:
        raise AssertionError("Incomplete training budget")
    if summary["completed_epochs"] != expected[0]*expected[1]:
        raise AssertionError("Incomplete epochs")
    identity = (summary["seed"], summary["r"], summary["rho"], summary["probe_branch"],
                summary["update_direction"], summary["step_mode"], summary["attention_backend"])
    if identity != (2, job["radius"], .05, "P", job["direction"], "natural", "native"):
        raise AssertionError("Wrong scientific condition")
    factors = (1, job["second_probe_sign"], job["second_difference_sign"], job["update_base"], 1.)
    fields = ("first_probe_sign", "second_probe_sign", "second_difference_sign", "update_base", "second_difference_weight")
    if tuple(summary[key] for key in fields) != factors:
        raise AssertionError("Summary sign/base mismatch")
    expected_source = job.get("source_root", job["command"][job["command"].index("--source")+1])
    if summary["source"] != expected_source:
        raise AssertionError("Source root differs from the frozen suite")
    counters = summary["counters"]
    steps = expected[2]
    count_expectations = dict(training_batches=steps, training_ce_gradient_passes=4*steps,
                              training_forward_count=4*steps, training_sgd_updates=steps,
                              training_g3_gradient_passes=steps,
                              training_g3_positive_gradient_passes=steps if job["second_probe_sign"] == 1 else 0,
                              training_g3_negative_gradient_passes=steps if job["second_probe_sign"] == -1 else 0,
                              diagnostic_snapshots=expected[0], diagnostic_gradient_calls=expected[0],
                              diagnostic_forward_count=expected[0], diagnostic_sgd_updates=0,
                              diagnostic_g3_positive_gradient_calls=expected[0] if job["second_probe_sign"] == -1 else 0,
                              diagnostic_g3_negative_gradient_calls=expected[0] if job["second_probe_sign"] == 1 else 0)
    for key, value in count_expectations.items():
        if counters[key] != value:
            raise AssertionError("Derivative/update accounting mismatch: " + key)
    forbidden = ("fisher_estimation_passes", "weight_merge_updates", "forbidden_work_attempts",
                 "training_hvp_passes", "training_create_graph_passes", "training_hvp_reference_gradient_passes",
                 "training_global_projection_calls", "training_opposite_g1_gradient_passes")
    if any(counters[key] for key in forbidden):
        raise AssertionError("Forbidden Fisher/merge/projection/HVP work")
    required = {"finite", "all_steps_self_spec", "exact_snapshot_restore", "four_probes", "g3_outer_probe",
                "opposite_probe_omitted", "natural_sgd_exact", "fisher_empty_all_tasks", "actual_weight_merges_zero",
                "second_probe_geometry", "second_probe_sign", "second_residual_formula", "isolated_diagnostics_restored"}
    if not required <= set(summary["checks"]) or not all(value is True for value in summary["checks"].values()):
        raise AssertionError("Failed scientific checks")
    if summary["input_config_sha256"] != job["config_sha256"] or summary["trainer_sha256"] != sha256(HERE / "trainer/four_probe_train.py"):
        raise AssertionError("Input config or trainer changed")
    for name, key in (("step_trace.jsonl", "step_trace_sha256"), ("task_matrix.json", "task_matrix_sha256"),
                      ("side_diagnostics.jsonl", "side_diagnostics_sha256")):
        if sha256(out / name) != summary[key]:
            raise AssertionError("Output evidence changed: " + name)
    effective = read_json(out / "effective_config.json")
    validate_execution_flags(effective)
    if effective.get("protocol") != PROTOCOL or effective.get("cuda_deterministic_algorithms") is not True or effective.get("cuda_deterministic_warn_only") is not False:
        raise AssertionError("Effective deterministic configuration differs")
    if (effective["gam_grad_rho"], effective["gam_grad_norm_rho"], effective["lora_rank"], effective["seed"]) != (.05, job["radius"], 10, 2):
        raise AssertionError("Effective radius/rank/seed differs")
    if tuple(effective[key] for key in fields) != factors:
        raise AssertionError("Effective sign/base differs")
    if effective["ewc_lambda"] != 0 or effective["ewc_skip_task_fisher"] is not True or any(effective[key] for key in
            ("approximate_fisher_in_task", "in_task_weight_merge", "task_endpoint_weight_merge", "ewc_normalize_fisher",
             "ewc_normalize_accumulated_fisher", "ewc_fisher_damping_alpha", "gam_use_sgv")):
        raise AssertionError("A disabled module is enabled")
    if bad_loss_lines(HERE / "runtime/jobs" / job["id"] / "console.log"):
        raise AssertionError("Nonfinite training loss in console")
    trace = [json.loads(line) for line in (out / "step_trace.jsonl").read_text().splitlines()]
    if len(trace) != steps:
        raise AssertionError("Trace count mismatch")
    batches = [1]*expected[0] if job["smoke"] else [10, 8, 8, 5, 4, 3, 3, 3, 4, 3]
    coords = [(task, epoch, batch) for task in range(expected[0]) for epoch in range(1, expected[1]+1)
              for batch in range(batches[task])]
    trace_forbidden = ("fisher_calls", "weight_merge_calls", "gradient_projection_calls", "opposite_g1_gradient_calls",
                       "hvp_calls", "create_graph_calls", "training_hvp_passes", "training_create_graph_passes",
                       "training_hvp_reference_gradient_passes")
    for index, (row, coord) in enumerate(zip(trace, coords)):
        finite_tree(row)
        validate_execution_flags(row)
        if row["step_index"] != index or (row["task"], row["epoch"], row["batch"]) != coord or row["seed"] != 2:
            raise AssertionError("Trace coverage/seed mismatch")
        if tuple(row[key] for key in fields) != factors or row["update_direction"] != job["direction"]:
            raise AssertionError("Step sign/base/direction mismatch")
        if row["ordinary_gradient_calls"] != 4 or row["training_ce_gradient_passes"] != 4 or row["training_forward_count"] != 4 or row["sgd_updates"] != 1:
            raise AssertionError("Step gradient/forward/update budget mismatch")
        if row["isolated_diagnostic_gradient_calls"] != int(coord[1:] == (1, 0)):
            raise AssertionError("Isolated diagnostic gradient budget differs")
        if row["radius"] != job["radius"] or row["rho"] != .05 or row["natural_sgd_exact"] is not True:
            raise AssertionError("Actual radius or natural update differs")
        if any(row[key] for key in trace_forbidden) or row["g3_gradient_calls"] != 1 or row["projection"] is not None:
            raise AssertionError("Forbidden step work or missing selected g3")
        if (row["g3_positive_gradient_calls"], row["g3_negative_gradient_calls"]) != (int(job["second_probe_sign"] == 1), int(job["second_probe_sign"] == -1)):
            raise AssertionError("The selected g3 side is wrong")
        validate_second_probe(row, job["second_probe_sign"])
        if row["second_residual_formula_exact"] is not True:
            raise AssertionError("Residual formula did not match")
        if row["base_parameters_exact"] is not True or row["base_buffers_exact"] is not True or not all(value is True for value in row["state_restoration"].values()):
            raise AssertionError("Complete state restoration failed")
        if row["class_slice"] != [20*coord[0], 20*(coord[0]+1)] or row["seen_classes"] != 20*(coord[0]+1):
            raise AssertionError("Wrong class/task slice")
        if row["scale"] != row["nominal_lr"] or row.get("matched", False) or row["forward_rng_rule"] != "R0 replay for every probe; final RNG=R1 from g0":
            raise AssertionError("Natural step or RNG rule changed")
        expected_lr = (.04 if coord[0] == 0 else .02)*(.1+.9*(1+math.cos(math.pi*(coord[1]-1)/50))/2)
        if abs(row["nominal_lr"]-expected_lr) > 1e-12:
            raise AssertionError("Learning-rate schedule mismatch")
    diagnostics = [json.loads(line) for line in (out / "side_diagnostics.jsonl").read_text().splitlines()]
    if len(diagnostics) != expected[0] or [row["task"] for row in diagnostics] != list(range(expected[0])):
        raise AssertionError("Missing isolated per-task diagnostics")
    for row in diagnostics:
        finite_tree(row)
        validate_side_diagnostic(row, job["second_probe_sign"])
        task_trace = next(item for item in trace if item["task"] == row["task"])
        if row["identity"] != task_trace["identity"] or row["step_index"] != task_trace["step_index"] or tuple(row[key] for key in fields) != factors:
            raise AssertionError("Diagnostic inputs/signs differ from its pending training step")
    snapshot = validate_same_state_snapshot(job, summary, trace)
    matrix_record = read_json(out / "task_matrix.json")
    matrix = matrix_record["matrix"]
    if len(matrix) != expected[0] or any(len(row) != index+1 for index, row in enumerate(matrix)):
        raise AssertionError("Incomplete task matrix")
    if any(not math.isfinite(value) or not 0 <= value <= 100 for row in matrix for value in row):
        raise AssertionError("Invalid observed accuracy")
    for key, value in metrics(matrix).items():
        if abs(matrix_record[key]-value) > 1e-9 or abs(summary["final_metrics"][key]-value) > 1e-9:
            raise AssertionError("Task-macro metric recomputation failed: " + key)
    return dict(summary_sha256=sha256(path), trace_sha256=summary["step_trace_sha256"],
                matrix_sha256=summary["task_matrix_sha256"], diagnostics_sha256=summary["side_diagnostics_sha256"],
                updates=len(trace), tasks=len(matrix), metric_recomputation=True, same_state_snapshot=snapshot,
                first_probe_sign=1, second_probe_sign=job["second_probe_sign"],
                second_difference_sign=job["second_difference_sign"], update_base=job["update_base"])
