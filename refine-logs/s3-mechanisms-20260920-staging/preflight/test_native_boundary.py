#!/usr/bin/env python3
"""CPU witness: actual frozen native merge versus its observer wrapper.

Uses actual task-0 endpoint/snapshots/head from a completed short run. The
learner is allocated without constructing a ViT; all native factor selection,
loss weighting, path averaging and merge methods execute unchanged. A module
tree holds all 48 real factors plus the actual classifier and sentinel buffers.
Any forward or Fisher estimation is an error. This is a boundary equivalence
test, not a full training reproducibility or Fisher-estimator equivalence test.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch import nn


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


CALLS = {"forward": 0, "fisher": 0}


class NoForwardModule(nn.Module):
    def forward(self, *args, **kwargs):
        CALLS["forward"] += 1
        raise AssertionError("Boundary observer must never run a forward")


def model_from(endpoint, fc):
    model = NoForwardModule()
    for name, value in {**endpoint, **{f"fc.{k}": v for k, v in fc.items()}}.items():
        node = model
        pieces = name.split(".")
        for piece in pieces[:-1]:
            if piece not in node._modules:
                node.add_module(piece, NoForwardModule())
            node = node._modules[piece]
        node.register_parameter(pieces[-1], nn.Parameter(value.detach().clone()))
    model.register_buffer("observer_sentinel", torch.tensor([3, 5, 7], dtype=torch.int64))
    for index, module in enumerate(model.modules()):
        module.training = bool(index % 2)
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad_(bool(index % 3))
        parameter.grad = None if index % 4 == 0 else torch.full_like(parameter, (index + 1) / 1024)
    return model


def state(model):
    return {
        "parameters": {k: v.detach().clone() for k, v in model.named_parameters()},
        "gradients": {k: None if v.grad is None else v.grad.detach().clone() for k, v in model.named_parameters()},
        "buffers": {k: v.detach().clone() for k, v in model.named_buffers()},
        "modes": {k: v.training for k, v in model.named_modules()},
        "requires_grad": {k: v.requires_grad for k, v in model.named_parameters()},
        "parameter_versions": {k: v._version for k, v in model.named_parameters()},
    }


def identity(model):
    return {k: (id(v), None if v.grad is None else id(v.grad)) for k, v in model.named_parameters()}


def reject_fisher(*args, **kwargs):
    CALLS["fisher"] += 1
    raise AssertionError("Boundary observer must never estimate Fisher")


def learner_from(cls, cfg, payload, output, capture, changes):
    learner = cls.__new__(cls)
    learner.args = {**copy.deepcopy(cfg), **changes, "outputs_root": str(output), "s3_boundary_capture": capture}
    attrs = {
        "_cur_task": 0, "_known_classes": 0, "_total_classes": 20,
        "_is_main_process": False, "_posterior_state_save": False,
        "_mechanism_anchor_eval": False, "_mechanism_future_surrogate_eval": False,
        "_in_task_total_epochs": int(cfg["epochs"]),
        "_in_task_weight_checkpoint_interval": int(cfg.get("in_task_weight_checkpoint_interval", cfg["in_task_fisher_interval"])),
        "_in_task_path_loss_eps": float(cfg.get("in_task_path_loss_eps", 1e-12)),
    }
    for field, default in (("in_task_weight_merge", True), ("in_task_weight_merge_start_task", 0),
                           ("in_task_path_weighted", True), ("in_task_path_start_ratio", 0), ("in_task_path_beta", 3)):
        attrs["_" + field] = learner.args.get(field, default)
    omega = float(learner.args.get("in_task_weight_merge_omega", learner.args["in_task_fisher_omega"]))
    attrs["_in_task_weight_merge_omega"] = attrs["_in_task_fisher_omega"] = omega
    for name, value in attrs.items():
        setattr(learner, name, value)
    learner._in_task_weight_snapshots = [(s["epoch"], copy.deepcopy(s["factors"])) for s in payload["snapshots"]]
    learner._in_task_snapshot_metrics = [copy.deepcopy(s["metrics"]) for s in payload["snapshots"]]
    learner._in_task_weight_snapshot_tensors = sum(len(values) for _, values in learner._in_task_weight_snapshots)
    for field in ("_in_task_path_weight_last_metrics", "_in_task_weight_merge_last_metrics", "_weight_merge_then_fisher_last_metrics"):
        setattr(learner, field, {})
    learner._compute_raw_delta_fisher = reject_fisher
    return learner


def run(campaign):
    source = campaign / "source"
    manifest_path = campaign / "source_manifest.json"
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for relative, expected_hash in manifest["files"].items():
        require(sha(source / relative) == expected_hash, "Source differs from frozen manifest: " + relative)
    sys.path.insert(0, str(source))
    from models_LoRAbasedCL.as1_normfisher_gam_intask_damp_weightmergethenfisher import Learner
    from utils import s3_mechanism_controls as controls
    native_apply = Learner._apply_in_task_weight_merge
    require(Path(inspect.getfile(native_apply)).resolve().is_relative_to(source.resolve()), "Native learner imported outside frozen source")
    require(not hasattr(native_apply, "__wrapped__"), "Expected original unpatched native learner")

    class ObservedLearner(Learner):
        pass

    controls.install(ObservedLearner)
    require(ObservedLearner._apply_in_task_weight_merge.__wrapped__ is native_apply, "Wrong native wrapper target")
    spec = importlib.util.spec_from_file_location("native_boundary_validator", campaign / "preflight/verify_boundaries.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    smoke = campaign / "preflight/smoke_campaign"
    cfg_path = smoke / "configs/smoke_observer_on_s2.json"
    cfg = json.loads(cfg_path.read_text())
    original_output = Path(cfg["outputs_root"])
    archive = original_output / "mechanism_states/task_00.pt"
    fc_path = next(original_output.rglob("fc_state_0.pt"))
    payload = torch.load(archive, map_location="cpu", weights_only=True)
    fc = torch.load(fc_path, map_location="cpu", weights_only=True)
    require(len(payload["endpoint_factors"]) == 48, "Expected 48 real factors")
    cases = {
        "weighted_center": {}, "uniform_same_snapshots": {"in_task_path_beta": 0},
        "latest_same_snapshots": {"in_task_path_start_ratio": .6},
        "matched_no_merge": {"in_task_weight_merge_start_task": 10},
        "module_off": {"in_task_weight_merge": False},
        "endpoint_fisher_mode_boundary_only": {"s3_fisher_location": "endpoint"},
    }
    results = []
    for name, changes in cases.items():
        with tempfile.TemporaryDirectory(prefix="s3_native_boundary_") as tmp:
            directory = Path(tmp)
            baseline = learner_from(Learner, cfg, payload, directory / "native", False, changes)
            observed = learner_from(ObservedLearner, cfg, payload, directory / "observed", True, changes)
            a = model_from(payload["endpoint_factors"], fc)
            b = model_from(payload["endpoint_factors"], fc)
            before_a, before_b, ids_a, ids_b = state(a), state(b), identity(a), identity(b)
            require(equal(before_a, before_b), f"{name}: inputs differ")
            snapshots_a = copy.deepcopy(baseline._in_task_weight_snapshots)
            snapshots_b = copy.deepcopy(observed._in_task_weight_snapshots)
            random.seed(173); np.random.seed(173); torch.manual_seed(173)
            rng = Learner._capture_global_rng_state()
            before_rng = controls._rng_digest(baseline)
            native_metrics = native_apply(baseline, a)
            native_rng = controls._rng_digest(baseline)
            Learner._restore_global_rng_state(rng)
            observer_metrics = observed._apply_in_task_weight_merge(b)
            observer_rng = controls._rng_digest(observed)
            after_a, after_b = state(a), state(b)
            require(equal(after_a, after_b), f"{name}: observer changes native parameter/gradient/mode/buffer result")
            require(equal(native_metrics, observer_metrics), f"{name}: native merge metrics changed")
            require(before_rng == native_rng == observer_rng, f"{name}: RNG changed")
            require(ids_a == identity(a) and ids_b == identity(b), f"{name}: parameter/gradient identities changed")
            for key in ("gradients", "buffers", "modes", "requires_grad", "parameter_versions"):
                require(equal(before_a[key], after_a[key]) and equal(before_b[key], after_b[key]), f"{name}: {key} mutated")
            for key in fc:
                require(torch.equal(before_b["parameters"]["fc." + key], after_b["parameters"]["fc." + key]), f"{name}: classifier changed")
            require(equal(snapshots_a, baseline._in_task_weight_snapshots) and equal(snapshots_b, observed._in_task_weight_snapshots), f"{name}: snapshot inputs mutated")
            require(equal(baseline._in_task_path_weight_last_metrics, observed._in_task_path_weight_last_metrics), f"{name}: path metrics differ")
            require(equal(baseline._weight_merge_then_fisher_last_metrics, observed._weight_merge_then_fisher_last_metrics), f"{name}: learner merge metrics differ")
            recorded = torch.load(directory / "observed/mechanism_states/task_00.pt", map_location="cpu", weights_only=True)
            require(equal(recorded["endpoint_factors"], payload["endpoint_factors"]), f"{name}: archive endpoint changed")
            require(equal(recorded["retained_factors"], observed._snapshot_lora_factor_weights(b)), f"{name}: archive/live retained mismatch")
            boundary = validator.validate_boundaries(directory / "observed", 1, changes.get("s3_fisher_location", "retained"))
            require(not (directory / "native").exists(), f"{name}: native baseline unexpectedly writes archive")
            results.append({"name": name, "passed": True, "all_50_parameter_tensors_bitwise_equal": True,
                            "gradients_buffers_modes_requires_grad_and_identities_unchanged": True,
                            "native_metrics_exact": True, "rng_sha256": before_rng,
                            "rng_unchanged": True, "merge_applied": bool(native_metrics["in_task_weight_merge_applied"]),
                            "boundary_reconstruction": boundary["per_task"][0],
                            "selected_weights": recorded["selected_path_weights"]})
    require(CALLS == {"forward": 0, "fisher": 0}, "Unexpected forward/Fisher call")
    source_paths = [source / "utils/s3_mechanism_controls.py", source / "models_LoRAbasedCL/as1_normfisher_gam_intask.py",
                    source / "models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py",
                    source / "models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py",
                    source / "models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py"]
    source_paths.append(Path(inspect.getfile(Learner._match_shape)))
    require(sha(manifest_path) == manifest_sha, "Source manifest changed during witness")
    for relative, expected_hash in manifest["files"].items():
        require(sha(source / relative) == expected_hash, "Source changed during witness: " + relative)
    return {"passed": True, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "execution": "CPU only; actual frozen native methods; real saved factors/snapshots/classifier",
            "cuda_available": torch.cuda.is_available(), "cases": results, "call_counts": CALLS,
            "source_manifest_sha256": manifest_sha, "source_files": {str(p): sha(p) for p in source_paths},
            "no_gpu": True, "extra_forward_count": CALLS["forward"], "extra_fisher_count": CALLS["fisher"], "errors": [],
            "inputs": {str(p): sha(p) for p in (archive, fc_path, cfg_path)}, "witness_source_sha256": sha(__file__),
            "scope_limits": ["Boundary state equivalence only; not full training reproducibility.",
                             "Module tree holds the actual factors and classifier without constructing the frozen ViT.",
                             "No Fisher pass is run; endpoint Fisher restoration needs the separate control and GPU smoke checks.",
                             "Uses saved 2-epoch task-0 snapshots; 50-epoch training outputs remain pending."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920'))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("NATIVE_BOUNDARY_WITNESS.json"))
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "Explicit CPU-only execution required")
    torch.set_num_threads(2)
    require(not args.output.exists(), "Refusing to overwrite existing witness")
    report = run(args.campaign.resolve())
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]), "calls": CALLS, "report": str(args.output)}))


if __name__ == "__main__":
    main()
