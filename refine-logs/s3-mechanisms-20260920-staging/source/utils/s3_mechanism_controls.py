"""Opt-in task-boundary controls for the frozen S3 WMergeThenFisher learner.

Call ``install()`` before constructing a learner. No existing source is changed.
``s3_fisher_location`` is ``retained`` (default) or ``endpoint``. The latter
changes only the factors used by the existing, single task-end Fisher pass.
``s3_boundary_capture`` saves CPU factors and actual merge metadata, without
additional model forwards, Fisher estimates, or random draws.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
import os
import pickle
from contextlib import contextmanager
from pathlib import Path

import torch

LOG = logging.getLogger(__name__)
VERSION = "s3_boundary_controls_v1"


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _options(learner):
    args = learner.args
    location = str(args.get("s3_fisher_location", "retained")).strip().lower()
    if location not in {"retained", "endpoint"}:
        raise ValueError(f"Invalid s3_fisher_location: {location!r}")
    capture = _bool(args.get("s3_boundary_capture", False))
    if location == "endpoint" or capture:
        incompatible = [
            key for key in (
                "posterior_state_save", "mechanism_anchor_eval",
                "mechanism_future_surrogate_eval", "task_endpoint_weight_merge",
                "ewc_skip_task_fisher",
            ) if _bool(args.get(key, False))
        ]
        if incompatible:
            raise ValueError(f"S3 controls do not support these active flags: {incompatible}")
    if capture and not args.get("outputs_root"):
        raise ValueError("s3_boundary_capture requires a nonempty outputs_root")
    return location, capture


def _rng_digest(learner):
    """Hash the existing learner's RNG snapshot without serializing storages."""
    state = learner._capture_global_rng_state()
    digest = hashlib.sha256()
    for key in ("python", "numpy"):
        digest.update(pickle.dumps(state[key], protocol=4))
    for value in [state["torch_cpu"], *(state.get("torch_cuda") or [])]:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_rng(learner, expected, where):
    actual = _rng_digest(learner)
    if actual != expected:
        raise RuntimeError(f"S3 {where} unexpectedly consumed RNG: {expected} != {actual}")
    return actual


def _snapshot(learner, model):
    result = learner._snapshot_lora_factor_weights(model)
    if not result:
        raise RuntimeError("S3 controls require nonempty LoRA factor snapshots")
    return {name: value.detach().cpu().clone() for name, value in result.items()}


@contextmanager
def _endpoint_loaded(learner, model, endpoint):
    """Load only the complete LoRA factor set; always restore exact live values."""
    named = dict(model.named_parameters())
    expected_names = {name for name, _ in learner._iter_lora_factor_weights(model)}
    if not endpoint or set(endpoint) != expected_names:
        raise RuntimeError("Endpoint snapshot does not match the live LoRA factor set")
    for name, value in endpoint.items():
        if name not in named or named[name].shape != value.shape:
            raise RuntimeError(f"Endpoint parameter missing or wrong shape: {name}")
    before_load_rng = _rng_digest(learner)
    saved = {name: named[name].detach().clone() for name in endpoint}
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.no_grad():
            for name, value in endpoint.items():
                named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
        _assert_rng(learner, before_load_rng, "endpoint loading")
        yield
    finally:
        before_restore_rng = _rng_digest(learner)
        with torch.no_grad():
            for name, value in saved.items():
                named[name].copy_(value)
        for module, mode in modes:
            module.training = mode
        _assert_rng(learner, before_restore_rng, "retained restoration")


def _endpoint_source_metrics(learner):
    updates = {
        "s3_controls_version": VERSION,
        "s3_fisher_location": "endpoint",
        "current_fisher_source": "pre_weight_merge_endpoint",
        "weight_merge_then_fisher_current_fisher_source": "pre_weight_merge_endpoint",
        "weight_merge_then_fisher_current_fisher_after_weight_merge": False,
        "weight_merge_then_fisher_fisher_source_post_merge": False,
        "posterior_center_smoothing_before_fisher": False,
    }
    for name in (
        "_in_task_fisher_last_metrics", "_weight_merge_then_fisher_last_metrics",
        "_in_task_weight_merge_last_metrics", "_last_mechanism_metrics",
    ):
        metrics = getattr(learner, name, None)
        if isinstance(metrics, dict):
            metrics.update(updates)


def _save_boundary(learner, endpoint, retained, before_rng, after_rng):
    task = int(learner._cur_task)
    directory = Path(os.path.expanduser(str(learner.args["outputs_root"]))).resolve() / "mechanism_states"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"task_{task:02d}.pt"
    snapshots = []
    snapshot_metrics = getattr(learner, "_in_task_snapshot_metrics", [])
    for index, (epoch, factors) in enumerate(learner._in_task_weight_snapshots):
        snapshots.append({
            "index": index,
            "epoch": int(epoch),
            "factors": {name: value.detach().cpu().clone() for name, value in factors.items()},
            "metrics": copy.deepcopy(snapshot_metrics[index] if index < len(snapshot_metrics) else {}),
        })
    payload = {
        "format_version": VERSION,
        "task": task,
        "known_classes_before_after_task": int(learner._known_classes),
        "total_classes": int(learner._total_classes),
        "s3_fisher_location": _options(learner)[0],
        "factor_space": "lora_a_b",
        "classifier_head": "unchanged_endpoint; not stored in this sidecar",
        "endpoint_factors": endpoint,
        "retained_factors": retained,
        "snapshots": snapshots,
        "path_weights_computed": bool(getattr(learner, "_s3_path_weights_called", False)),
        "selected_path_weights": copy.deepcopy(getattr(learner, "_s3_selected_path_weights", [])),
        "merge_metrics": copy.deepcopy(learner._in_task_weight_merge_last_metrics),
        "path_metrics": copy.deepcopy(getattr(learner, "_in_task_path_weight_last_metrics", {})),
        "config": {key: copy.deepcopy(learner.args.get(key)) for key in (
            "seed", "prefix", "s3_fisher_location", "s3_boundary_capture",
            "in_task_weight_merge", "in_task_weight_merge_start_task",
            "in_task_weight_checkpoint_interval", "in_task_fisher_interval",
            "in_task_weight_merge_omega", "in_task_fisher_omega",
            "in_task_path_start_ratio", "in_task_path_weighted", "in_task_path_beta",
            "fisher_mode", "ewc_preserve_fisher_rng", "ewc_lambda", "ewc_gamma",
            "ewc_fisher_damping_alpha",
        )},
        "rng_before_boundary_sha256": before_rng,
        "rng_after_boundary_sha256": after_rng,
        "capture_extra_forward_count": 0,
        "capture_extra_fisher_count": 0,
    }
    # Exclusive create deliberately fails instead of replacing earlier evidence.
    with path.open("xb") as stream:
        torch.save(payload, stream)
    _assert_rng(learner, after_rng, "boundary serialization")
    LOG.info("[S3BoundaryCapture] task=%d path=%s rng_unchanged=True", task, path)


def install(learner_cls=None):
    """Install once; an injected class allows dependency-light CPU preflight."""
    if learner_cls is None:
        from models_LoRAbasedCL.as1_normfisher_gam_intask_damp_weightmergethenfisher import Learner
        learner_cls = Learner
    if learner_cls.__dict__.get("_s3_controls_installed", False):
        return learner_cls
    original_apply = learner_cls._apply_in_task_weight_merge
    original_compute = learner_cls._compute_delta_fisher
    original_weights = learner_cls._path_checkpoint_weights
    original_after = learner_cls.after_task

    @functools.wraps(original_weights)
    def checkpoint_weights(self, selected):
        weights = original_weights(self, selected)
        if _options(self)[1]:
            if len(weights) != len(selected):
                raise RuntimeError("S3 checkpoint weights/selection length mismatch")
            self._s3_path_weights_called = True
            self._s3_selected_path_weights = [
                {"index": int(entry[0]), "epoch": int(entry[1]), "weight": float(weight)}
                for entry, weight in zip(selected, weights)
            ]
        return weights

    @functools.wraps(original_apply)
    def apply(self, model):
        location, capture = _options(self)
        if location == "retained" and not capture:
            return original_apply(self, model)
        before_rng = _rng_digest(self)
        endpoint = _snapshot(self, model)
        self._s3_endpoint_factors = endpoint
        self._s3_endpoint_task = int(self._cur_task)
        self._s3_path_weights_called = False
        self._s3_selected_path_weights = []
        result = original_apply(self, model)
        retained = _snapshot(self, model) if capture else None
        after_rng = _assert_rng(self, before_rng, "boundary snapshot/merge")
        if capture:
            _save_boundary(self, endpoint, retained, before_rng, after_rng)
        return result

    @functools.wraps(original_compute)
    def compute(self, loader, model, class_offset):
        location, _ = _options(self)
        if location == "retained":
            return original_compute(self, loader, model, class_offset)
        if getattr(self, "_s3_endpoint_task", None) != int(self._cur_task):
            raise RuntimeError("Missing current-task pre-merge endpoint; cannot estimate endpoint Fisher")
        endpoint = getattr(self, "_s3_endpoint_factors", {})
        with _endpoint_loaded(self, model, endpoint):
            result = original_compute(self, loader, model, class_offset)
        _endpoint_source_metrics(self)
        LOG.info(
            "[S3FisherSourceOverride] task=%d actual_source=pre_weight_merge_endpoint "
            "retained_model_restored=True existing_fisher_calls=1",
            int(self._cur_task),
        )
        return result

    @functools.wraps(original_after)
    def after(self):
        result = original_after(self)
        if _options(self)[0] == "endpoint":
            # Native after_task sets this label after copying Fisher metrics.
            _endpoint_source_metrics(self)
        return result

    learner_cls._apply_in_task_weight_merge = apply
    learner_cls._compute_delta_fisher = compute
    learner_cls._path_checkpoint_weights = checkpoint_weights
    learner_cls.after_task = after
    learner_cls._s3_controls_installed = VERSION
    return learner_cls
