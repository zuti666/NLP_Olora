#!/usr/bin/env python3
"""Read-only CPU review of all S3 mechanism smokes and observer neutrality.

Run after the smoke queue completes. Writes one new smoke_review.json, refusing
overwrite. Native A/B checkpoint files serialize local Linear modules; their
tensor state is compared exactly, rather than comparing pickle file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
SHORT_FIELDS = {"prefix", "outputs_root", "optimizer_tag_override", "nb_tasks",
                "total_sessions", "init_epoch", "epochs", "in_task_fisher_interval", "max_train_tasks"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, rel=1e-5, absolute=1e-8):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absolute)


def matrix_banks_equal(a, b):
    """Exact observed cells; explicit NaN equivalence only above the diagonal."""
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
                and all(matrix_banks_equal(a[key], b[key]) for key in a))
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    n = len(a)
    for i, (left, right) in enumerate(zip(a, b)):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != n or len(right) != n:
            return False
        for j, (x, y) in enumerate(zip(left, right)):
            if j > i and x is None and y is None:
                continue
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            if j > i and math.isnan(x) and math.isnan(y):
                continue
            if not math.isfinite(x) or not math.isfinite(y) or x != y:
                return False
    return True


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu()}
    if isinstance(value, torch.nn.Module):
        return flatten_tensors(value.state_dict(), prefix)
    if isinstance(value, dict):
        result = {}
        for key in sorted(value, key=str):
            result.update(flatten_tensors(value[key], f"{prefix}/{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for i, child in enumerate(value):
            result.update(flatten_tensors(child, f"{prefix}/{i}"))
        return result
    raise TypeError(f"Unexpected checkpoint object at {prefix}: {type(value).__name__}")


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare_checkpoint_tensors(a, b):
    # Only read known files generated in the explicitly selected local campaign.
    ta = flatten_tensors(torch.load(a, map_location="cpu", weights_only=False))
    tb = flatten_tensors(torch.load(b, map_location="cpu", weights_only=False))
    require(set(ta) == set(tb) and bool(ta), f"Checkpoint tensor keys differ: {a.name}")
    mismatches = []
    for key in ta:
        x, y = ta[key], tb[key]
        require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), f"Nonfinite checkpoint: {key}")
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            mismatches.append(key)
    return {"file_on": str(a), "file_off": str(b), "tensor_count": len(ta),
            "equal": not mismatches, "mismatch_keys": mismatches,
            "tensor_state_sha256_on": tensor_digest(ta), "tensor_state_sha256_off": tensor_digest(tb)}


def check_effects(cfg, data, output, expected, validator):
    steps = data.get("mechanism", {}).get("steps", {})
    require(set(steps) == set(map(str, range(expected))), "Missing per-task mechanisms")
    capture = bool(cfg.get("s3_boundary_capture", False))
    location = cfg.get("s3_fisher_location", "retained")
    result = {"capture": capture, "fisher_location": location, "tasks": []}
    if capture:
        result["boundary_validation"] = validator.validate_boundaries(output, expected, location)
    else:
        require(not list(output.rglob("mechanism_states/task_*.pt")), "Observer-off created a boundary archive")
    native_logs = list(output.rglob("*vit_base_patch16_224.log"))
    require(len(native_logs) == 1, "Expected one native training log")
    log = native_logs[0].read_text()
    damped = {int(t): (float(alpha), int(n), float(mx), float(mean), float(total))
              for t, alpha, n, mx, mean, total in re.findall(
                  r"\[AS1-NormFisher-Damp\] task=(\d+) alpha=([\d.eE+-]+) damped_fisher "
                  r"tensors=\d+ numel=(\d+) max=([\d.eE+-]+) mean=([\d.eE+-]+) sum=([\d.eE+-]+)", log)}
    gamma = float(cfg.get("ewc_gamma", .9))
    previous_mass = 0.0
    for task in range(expected):
        m = steps[str(task)]
        for key in ("train_clean_loss_mean", "train_ce_clean_mean", "train_ewc_mean", "train_gam_mean"):
            require(isinstance(m.get(key), (int, float)) and math.isfinite(m[key]), f"Task{task}: missing/nonfinite {key}")
        for key, value in m.items():
            if key.startswith(("train_", "grad_")) and isinstance(value, (int, float)):
                require(math.isfinite(value), f"Task{task}: nonfinite {key}")
        require(close(m["lambda_ewc"], cfg["ewc_lambda"]), f"Task{task}: EWC lambda not effective")
        require(close(m["lambda_flat"], cfg["lambda_flat"]), f"Task{task}: GAM strength not effective")
        require(m["fisher_mode"] == cfg.get("fisher_mode", "batch_empirical"), f"Task{task}: wrong Fisher estimator")
        require(close(m["ewc_fisher_damping_alpha"], cfg["ewc_fisher_damping_alpha"]), f"Task{task}: damping not effective")
        expected_source = "pre_weight_merge_endpoint" if location == "endpoint" else "post_weight_merge"
        require(m["current_fisher_source"] == expected_source, f"Task{task}: Fisher source mislabeled")
        require(bool(m["weight_merge_then_fisher_current_fisher_after_weight_merge"]) == (location != "endpoint"),
                f"Task{task}: Fisher after-merge flag wrong")
        if location == "endpoint":
            require(not m["weight_merge_then_fisher_fisher_source_post_merge"] and
                    not m["posterior_center_smoothing_before_fisher"], f"Task{task}: stale endpoint-source labels")
        expected_trace = 1.0 if cfg.get("ewc_normalize_fisher", True) else m["current_fisher_raw_trace"]
        require(close(m["current_fisher_trace"], expected_trace), f"Task{task}: task normalization differs")
        mass = m["fisher_past_trace_mass"]
        require(close(mass, gamma * previous_mass + m["current_fisher_trace"]), f"Task{task}: gamma/history recurrence differs")
        previous_mass = mass
        alpha = float(cfg["ewc_fisher_damping_alpha"])
        if alpha:
            require(task in damped, f"Task{task}: no damping execution record")
            logged_alpha, numel, mx, mean, total = damped[task]
            require(close(logged_alpha, alpha) and close(total, 1.0), f"Task{task}: damping/trace evidence differs")
            if alpha == 1:
                require(close(mx, 1 / numel, rel=1e-3, absolute=1e-12) and close(mean, 1 / numel, rel=1e-3, absolute=1e-12),
                        f"Task{task}: alpha1 not isotropic at printed precision")
        else:
            require(task not in damped, f"Task{task}: alpha0 unexpectedly emits damping")
        enabled = bool(cfg.get("in_task_weight_merge", True))
        start = int(cfg.get("in_task_weight_merge_start_task", 0))
        omega = float(cfg.get("in_task_weight_merge_omega", cfg.get("in_task_fisher_omega", .3)))
        applied = enabled and task >= start and omega > 0
        require(bool(m["in_task_weight_merge_applied"]) == applied, f"Task{task}: wrong merge execution")
        item = {"task": task, "merge_applied": applied, "fisher_source": expected_source,
                "fisher_trace": m["current_fisher_trace"], "fisher_mass": mass,
                "train_ce": m["train_ce_clean_mean"], "train_ewc": m["train_ewc_mean"]}
        if capture:
            payload = torch.load(output / "mechanism_states" / f"task_{task:02d}.pt", map_location="cpu", weights_only=True)
            epochs = int(cfg["init_epoch"] if task == 0 else cfg["epochs"])
            interval = int(cfg.get("in_task_weight_checkpoint_interval", cfg["in_task_fisher_interval"]))
            expected_epochs = list(range(interval, epochs + 1, interval)) if enabled and interval > 0 else []
            snaps = payload["snapshots"]
            require([s["epoch"] for s in snaps] == expected_epochs, f"Task{task}: checkpoint budget differs")
            quality_count = 0
            if enabled and cfg.get("in_task_path_weighted", False):
                for s in snaps:
                    q = s["metrics"]
                    require(math.isfinite(q["snapshot_train_ce"]) and q["snapshot_train_samples"] > 0 and q["snapshot_train_batches"] > 0,
                            f"Task{task}: quality-CE pass missing")
                    quality_count += 1
            selected = payload["selected_path_weights"]
            if applied:
                min_epoch = max(1, min(epochs, math.ceil(float(cfg.get("in_task_path_start_ratio", 0)) * epochs)))
                selected_epochs = [e for e in expected_epochs if e >= min_epoch] or expected_epochs[-1:]
                require([w["epoch"] for w in selected] == selected_epochs, f"Task{task}: wrong selected window")
                if float(cfg.get("in_task_path_beta", 3)) == 0 or not cfg.get("in_task_path_weighted", False):
                    require(all(close(x["weight"], 1 / len(selected), rel=0, absolute=1e-12) for x in selected),
                            f"Task{task}: uniform arm not uniform")
                if len(selected) == 1:
                    require(close(selected[0]["weight"], 1, rel=0, absolute=1e-12), f"Task{task}: latest-only weight not1")
            else:
                require(not selected and not payload["path_weights_computed"], f"Task{task}: no-merge unexpectedly computes a path")
            item.update(snapshot_epochs=expected_epochs, quality_evaluations=quality_count,
                        selected_weights=selected, reconstruction_max_abs_error=result["boundary_validation"]["per_task"][task]["reconstruction_max_abs_error"])
        # The native SGD branch does not collect GAM gradients. Do not call the
        # absent GAM counter a measured number of forwards.
        if cfg.get("optimizer_type") == "gam":
            require(m.get("grad_gam_norm_mean", 0) > 0, f"Task{task}: GAM gradient missing")
        else:
            require(m.get("grad_gam_norm_mean", 0) == 0, f"Task{task}: SGD has a GAM gradient statistic")
        result["tasks"].append(item)
    result["native_log_path"] = str(native_logs[0])
    result["native_log_sha256"] = sha(native_logs[0])
    return result


def review(campaign):
    smoke = campaign / read(campaign / "suite.json").get("integration_campaign", "preflight/smoke_campaign")
    common = module_from(smoke / "common.py", "smoke_review_common")
    validator = module_from(campaign / "preflight/verify_boundaries.py", "smoke_review_boundaries")
    suite, formal = read(smoke / "suite.json"), read(campaign / "suite.json")
    common.verify_source(suite)
    require(len(suite["jobs"]) == len(formal["jobs"]) + 2, "Expected every formal arm plus observer on/off")
    require(int(suite["expected_tasks"]) == 2, "Smoke must cover two tasks")
    expected_ids = {"smoke_" + j["id"] for j in formal["jobs"]} | {"smoke_observer_on_s2", "smoke_observer_off_s2"}
    require({j["id"] for j in suite["jobs"]} == expected_ids, "Smoke arm inventory differs")
    formal_by_id = {j["id"]: read(j["config"]) for j in formal["jobs"]}
    records, valid, errors = [], {}, []
    for job in suite["jobs"]:
        row = {"id": job["id"], "passed": False}
        try:
            run = smoke / "runtime/jobs" / job["id"]
            require(not (run / "failed.json").exists() and not (run / "queue_failed.json").exists(), "Failed receipt exists")
            receipt = read(run / "completed.json")
            cfg = read(job["config"])
            require(receipt["status"] == "completed", "No completed receipt")
            require(receipt["gpu"] in suite["allowed_gpus"], "Receipt GPU not authorized by this suite")
            require(receipt["host"] == suite["execution_hostname"], "Receipt host differs from execution suite")
            require(receipt["suite_sha256"] == sha(smoke / "suite.json"), "Smoke suite changed after job start")
            require(sha(job["config"]) == job["config_sha256"] == receipt["config_sha256"], "Config hashes differ")
            require(sha(run / "config.json") == job["config_sha256"], "Resolved run config differs")
            require(sha(suite["source_manifest"]) == receipt["source_manifest_sha256"], "Source manifest receipt differs")
            for name, digest in receipt["runtime_sha256"].items():
                require(sha(smoke / name) == digest, "Runtime changed after smoke: " + name)
            formal_id = job["id"].removeprefix("smoke_")
            if formal_id in formal_by_id:
                parent = formal_by_id[formal_id]
                differences = {k for k in set(parent) | set(cfg) if parent.get(k) != cfg.get(k)}
                require(differences <= SHORT_FIELDS, "Smoke alters extra scientific fields: " + str(differences - SHORT_FIELDS))
            require(cfg.get("max_train_tasks") == cfg["nb_tasks"] == cfg["total_sessions"] == cfg["init_epoch"] == cfg["epochs"] == 2,
                    "Wrong short-protocol dimensions")
            result = common.inspect_metrics(suite, job)
            require(result["metrics_sha256"] == receipt["result"]["metrics_sha256"], "Metrics changed after completion")
            data = read(result["metrics_path"])
            require("Restrict training to 2 tasks (of 10 total)" in (run / "console.log").read_text(),
                    "Native trainer did not confirm the two-task limit")
            output = Path(cfg["outputs_root"])
            effects = check_effects(cfg, data, output, 2, validator)
            if cfg.get("s3_boundary_capture"):
                require(effects["boundary_validation"]["files"] == receipt["result"]["boundary_validation"]["files"],
                        "Boundary changed after completion")
            row.update(passed=True, gpu=receipt["gpu"], host=receipt["host"],
                       config_path=job["config"], config_sha256=job["config_sha256"],
                       metrics_path=result["metrics_path"], metrics_sha256=result["metrics_sha256"],
                       effects=effects)
            valid[job["id"]] = {"config": cfg, "data": data, "result": result,
                                 "checkpoints": Path(result["metrics_path"]).parent.parent / "checkpoints"}
        except Exception as exc:
            row["error"] = repr(exc)
            errors.append(job["id"] + ": " + str(exc))
        records.append(row)
    observer = {"passed": False}
    try:
        on, off = valid["smoke_observer_on_s2"], valid["smoke_observer_off_s2"]
        differences = {k for k in set(on["config"]) | set(off["config"]) if on["config"].get(k) != off["config"].get(k)}
        require(differences == {"prefix", "outputs_root", "s3_boundary_capture"}, "Observer pair changes extra config fields")
        matrices_equal = matrix_banks_equal(on["data"]["cnn"]["matrices"], off["data"]["cnn"]["matrices"])
        curves_equal = on["result"]["cnn_top1_curve"] == off["result"]["cnn_top1_curve"]
        comparisons = []
        for task in range(2):
            for stem in ("lora_w_a", "lora_w_b", "fc_state"):
                filename = f"{stem}_{task}.pt"
                comparison = compare_checkpoint_tensors(on["checkpoints"] / filename, off["checkpoints"] / filename)
                comparisons.append(comparison)
        witness_path = campaign / "preflight/NATIVE_BOUNDARY_WITNESS.json"
        witness = read(witness_path)
        require(witness.get("passed") is True, "Native same-state observer witness failed")
        require(witness.get("source_manifest_sha256") == sha(campaign / "source_manifest.json"),
                "Native boundary witness is for different source")
        require(witness.get("witness_source_sha256") == sha(campaign / "preflight/test_native_boundary.py"),
                "Native boundary witness script changed")
        for path, digest in witness.get("source_files", {}).items():
            require(sha(path) == digest, "Native witness source changed: " + path)
        observer = {"passed": True,
                    "verification_basis": "actual native same-state equivalence plus per-boundary RNG and factor invariants",
                    "native_boundary_witness": str(witness_path), "native_boundary_witness_sha256": sha(witness_path),
                    "cross_run_trajectory_exact": matrices_equal and curves_equal and all(x["equal"] for x in comparisons),
                    "cnn_all_matrices_exact": matrices_equal, "sample_curve_exact": curves_equal,
                    "task_checkpoint_groups": 2, "tensor_comparisons": comparisons,
                    "tensor_count": sum(x["tensor_count"] for x in comparisons),
                    "fisher_tensor_equivalence": "not checked: Fisher tensors not archived"}
    except Exception as exc:
        observer["error"] = repr(exc)
        errors.append("observer neutrality: " + str(exc))
    return {"campaign": str(campaign), "passed": not errors, "expected_smokes": len(suite["jobs"]),
            "smoke_jobs": len(suite["jobs"]),
            "formal_suite_sha256": sha(campaign / "suite.json"),
            "smoke_suite_sha256": sha(smoke / "suite.json"),
            "verified_smokes": sum(row["passed"] for row in records), "jobs": records,
            "observer_neutrality": observer, "errors": errors,
            "scope_limits": ["Two-task/two-epoch integration only; no performance conclusion.",
                             "Alpha1 isotropy checked only at printed max/mean precision; no saved Fisher min/max tensors.",
                             "No per-update forward/probe counter archived; native SGD/GAM branch and gradient statistics checked.",
                             "No same-batch half-direction algebra gate is supplied by this script; require the dedicated optimizer check.",
                             "Fisher source verified through hook implementation, boundary location and final mechanism labels; no paired Fisher cosine computed.",
                             "Separate same-seed GPU runs differed before the observer executed in attempt1; cross-run tensor equality is reported as a reproducibility diagnostic, not an observer causality test."],
            "reviewer_source_sha256": sha(__file__), "source_manifest_sha256": sha(campaign / "source_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    campaign = args.campaign.resolve()
    output = args.output or campaign / "preflight/smoke_review.json"
    require(not output.exists(), "Refusing to overwrite review")
    report = review(campaign)
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"passed": report["passed"], "verified": report["verified_smokes"],
                      "expected": report["expected_smokes"], "errors": report["errors"], "report": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
