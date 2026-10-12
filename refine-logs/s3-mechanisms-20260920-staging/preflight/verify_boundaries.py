"""Validate immutable S3 factor sidecars on CPU, without model inference.

These archives contain no dual Fisher estimates. Their validity cannot support
an endpoint/retained Fisher-cosine claim; that requires a separate evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

EXPECTED_TENSORS = 48
FACTOR_SUFFIXES = (
    ".linear_a_q.weight", ".linear_b_q.weight",
    ".linear_a_v.weight", ".linear_b_v.weight",
)


def _require(condition, message):
    if not condition:
        raise ValueError(f"Boundary validation failed: {message}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _factors(value, label, reference=None):
    _require(isinstance(value, dict), f"{label}: factor dictionary required")
    _require(len(value) == EXPECTED_TENSORS, f"{label}: expected 48 factors, got {len(value)}")
    modules = {}
    for name, tensor in value.items():
        _require(isinstance(name, str) and name.endswith(FACTOR_SUFFIXES), f"{label}: invalid factor name {name!r}")
        suffix = next(suffix for suffix in FACTOR_SUFFIXES if name.endswith(suffix))
        modules.setdefault(name[:-len(suffix)], set()).add(suffix)
        _require(isinstance(tensor, torch.Tensor), f"{label}/{name}: tensor required")
        _require(tensor.device.type == "cpu", f"{label}/{name}: expected CPU tensor")
        _require(tensor.ndim == 2 and tensor.numel() > 0, f"{label}/{name}: nonempty matrix required")
        _require(tensor.is_floating_point() and bool(torch.isfinite(tensor).all()), f"{label}/{name}: nonfinite or nonfloating tensor")
        if reference is not None:
            _require(name in reference, f"{label}/{name}: factor absent in endpoint")
            _require(tensor.shape == reference[name].shape and tensor.dtype == reference[name].dtype, f"{label}/{name}: shape or dtype differs from endpoint")
    _require(len(modules) == 12 and all(suffixes == set(FACTOR_SUFFIXES) for suffixes in modules.values()), f"{label}: expected complete Q/V A/B factors in 12 modules")
    for prefix in modules:
        for projection in ("q", "v"):
            a = value[f"{prefix}.linear_a_{projection}.weight"]
            b = value[f"{prefix}.linear_b_{projection}.weight"]
            _require(b.shape[1] == a.shape[0], f"{label}/{prefix}/{projection}: incompatible B@A shapes")
    return value


def _validate_one(payload, task, expected_location):
    _require(isinstance(payload, dict), f"task {task}: archive dictionary required")
    _require(payload.get("format_version") == "s3_boundary_controls_v1", f"task {task}: unknown format")
    _require(payload.get("task") == task, f"task {task}: stale or misnamed task {payload.get('task')!r}")
    _require(payload.get("s3_fisher_location") == expected_location, f"task {task}: Fisher location differs from requested configuration")
    _require(payload.get("factor_space") == "lora_a_b", f"task {task}: wrong factor space")
    known = payload.get("known_classes_before_after_task")
    total = payload.get("total_classes")
    _require(isinstance(known, int) and isinstance(total, int) and 0 <= known < total, f"task {task}: invalid class boundary")
    before = payload.get("rng_before_boundary_sha256", "")
    after = payload.get("rng_after_boundary_sha256", "")
    _require(isinstance(before, str) and re.fullmatch(r"[0-9a-f]{64}", before), f"task {task}: invalid RNG digest")
    _require(before == after, f"task {task}: observer/merge consumed RNG")
    for key in ("capture_extra_forward_count", "capture_extra_fisher_count"):
        _require(payload.get(key) == 0, f"task {task}: unexpected {key}")
    endpoint = _factors(payload.get("endpoint_factors"), f"task {task}/endpoint")
    retained = _factors(payload.get("retained_factors"), f"task {task}/retained", endpoint)
    snapshots = payload.get("snapshots")
    _require(isinstance(snapshots, list), f"task {task}: snapshots list required")
    previous_epoch = 0
    for index, snapshot in enumerate(snapshots):
        _require(isinstance(snapshot, dict) and snapshot.get("index") == index, f"task {task}: stale snapshot index {index}")
        epoch = snapshot.get("epoch")
        _require(isinstance(epoch, int) and epoch > previous_epoch, f"task {task}: snapshot epochs must increase")
        previous_epoch = epoch
        _factors(snapshot.get("factors"), f"task {task}/snapshot {index}", endpoint)
        _require(isinstance(snapshot.get("metrics"), dict), f"task {task}/snapshot {index}: metrics missing")
    merge_metrics = payload.get("merge_metrics")
    _require(isinstance(merge_metrics, dict), f"task {task}: merge metrics missing")
    applied = merge_metrics.get("in_task_weight_merge_applied")
    _require(applied in (0, 1, False, True), f"task {task}: merge-applied flag missing")
    weights = payload.get("selected_path_weights")
    _require(isinstance(weights, list), f"task {task}: actual selected weights missing")
    max_abs_error = 0.0
    if not applied:
        for key in endpoint:
            _require(torch.equal(endpoint[key], retained[key]), f"task {task}/{key}: no-merge changed a factor")
        reconstruction = "exact_no_merge"
    else:
        _require(payload.get("path_weights_computed") is True and len(weights) > 0, f"task {task}: applied merge has no observed path weights")
        seen = set()
        for selected in weights:
            _require(isinstance(selected, dict), f"task {task}: malformed selected weight")
            index = selected.get("index")
            _require(isinstance(index, int) and 0 <= index < len(snapshots) and index not in seen, f"task {task}: invalid/duplicate selected snapshot")
            seen.add(index)
            _require(selected.get("epoch") == snapshots[index]["epoch"], f"task {task}: selected epoch/index mismatch")
            weight = selected.get("weight")
            _require(isinstance(weight, (int, float)) and math.isfinite(weight) and weight >= 0, f"task {task}: invalid selection weight")
        weight_sum = sum(float(item["weight"]) for item in weights)
        _require(math.isclose(weight_sum, 1.0, rel_tol=0, abs_tol=1e-12), f"task {task}: selected weights do not sum to one")
        omega = merge_metrics.get("in_task_weight_merge_omega")
        if omega is None:
            config = payload.get("config", {})
            omega = config.get("in_task_weight_merge_omega")
            if omega is None:
                omega = config.get("in_task_fisher_omega")
        _require(isinstance(omega, (int, float)) and math.isfinite(omega) and 0 < omega <= 1, f"task {task}: invalid or missing merge omega")
        for key in endpoint:
            path_value = None
            for selected in weights:
                term = snapshots[selected["index"]]["factors"][key] * float(selected["weight"])
                path_value = term if path_value is None else path_value + term
            path_value = path_value / weight_sum
            expected = endpoint[key] * (1.0 - float(omega)) + path_value * float(omega)
            error = float((retained[key].double() - expected.double()).abs().max())
            max_abs_error = max(max_abs_error, error)
            _require(torch.allclose(retained[key], expected, rtol=1e-6, atol=1e-7), f"task {task}/{key}: retained factors do not reconstruct (max error {error:.6g})")
        reconstruction = "observed_factor_merge"
    return {
        "task": task, "passed": True, "factor_tensors": len(endpoint),
        "snapshot_epochs": [item["epoch"] for item in snapshots],
        "selected_epochs": [item["epoch"] for item in weights],
        "merge_applied": bool(applied), "reconstruction": reconstruction,
        "reconstruction_max_abs_error": max_abs_error,
        "rng_unchanged": True, "known_classes": known, "total_classes": total,
    }


def validate_boundaries(output, expected_tasks, expected_location):
    """Return validated archive hashes or raise; this routine uses CPU only."""
    expected_tasks = int(expected_tasks)
    _require(expected_tasks > 0, "expected_tasks must be positive")
    _require(expected_location in {"retained", "endpoint"}, "unsupported expected Fisher location")
    directory = Path(output).expanduser().resolve() / "mechanism_states"
    _require(directory.is_dir(), f"missing mechanism_states directory: {directory}")
    files = sorted(directory.glob("*.pt"))
    expected_names = {f"task_{task:02d}.pt" for task in range(expected_tasks)}
    _require({path.name for path in files} == expected_names, f"expected task archives {sorted(expected_names)}, got {[path.name for path in files]}")
    hashes = {}
    per_task = []
    for task in range(expected_tasks):
        path = directory / f"task_{task:02d}.pt"
        digest = _sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        per_task.append(_validate_one(payload, task, expected_location))
        _require(_sha256(path) == digest, f"archive changed during validation: {path}")
        hashes[str(path)] = digest
    for previous, current in zip(per_task, per_task[1:]):
        _require(previous["total_classes"] == current["known_classes"], f"noncontiguous class boundaries at task {current['task']}")
    return {
        "passed": True, "tasks": len(per_task), "expected_tasks": expected_tasks,
        "fisher_location": expected_location, "files": hashes, "per_task": per_task,
        "dual_fisher_cosine": {
            "collected": False,
            "reason": "Sidecars store A/B factors and merge metadata, not paired endpoint/retained Fisher estimates. No Fisher cosine was measured.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-location", choices=["retained", "endpoint"], default="retained")
    args = parser.parse_args()
    print(json.dumps(validate_boundaries(args.output, args.expected_tasks, args.expected_location), indent=2))
