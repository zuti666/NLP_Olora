"""Shared native-SGD/FS training with an effective-Delta-W KFAC EWC prior.

The current-task update is inherited without preconditioning. History is an
exact sum of Kronecker terms, evaluated as tr(D.T @ G @ D @ C), never as a
product of separately summed factors. A diagonal mirror supports legacy
bookkeeping; it is not the training penalty. Task-boundary checkpoints include
the full history and retained Delta-W anchor in the hashed Fisher sidecar.
"""
from __future__ import annotations

import math
import os
import time

import torch

from models_LoRAbasedCL.as1_normfisher_fishershape import Learner as _FSLearner
from utils.kfac_ewc import KFACEWCState
from utils.mechanism_eval import as_bool


class Learner(_FSLearner):
    _KFAC_MODES = {"lora_kfac_empirical", "lr_lora_kfac_empirical"}

    def __init__(self, args):
        args = dict(args)
        args.setdefault("fisher_mode", "lora_kfac_empirical")
        # Explicitly opt into the existing Fisher RNG isolation for this learner.
        args.setdefault("ewc_preserve_fisher_rng", True)
        args["resume_require_fisher_state"] = True
        args["checkpoint_require_nonempty_fisher"] = True
        super().__init__(args)
        if self._fisher_mode not in self._KFAC_MODES:
            raise ValueError(f"KFAC EWC requires fisher_mode in {sorted(self._KFAC_MODES)}")
        self._kfac_grad_scale = str(args.get("kfac_grad_scale", "mean_x_bs"))
        self._kfac_seq_agg = str(args.get("kfac_seq_agg", "token"))
        if self._kfac_grad_scale not in {"mean", "mean_x_bs", "sum"}:
            raise ValueError("Invalid kfac_grad_scale")
        if self._kfac_seq_agg not in {"token", "sample"}:
            raise ValueError("Invalid kfac_seq_agg")
        if len(self._multiple_gpus) > 1 or torch.distributed.is_initialized():
            raise ValueError("KFAC EWC currently supports one process on one device")
        if not math.isfinite(self._ewc_gamma) or self._ewc_gamma < 0:
            raise ValueError("ewc_gamma must be finite and nonnegative")
        if not math.isfinite(self._ewc_lambda) or self._ewc_lambda < 0:
            raise ValueError("ewc_lambda must be finite and nonnegative")
        if self._ewc_eta != 0:
            raise ValueError("Use ewc_eta=0; KFAC damping uses ewc_fisher_damping_beta")
        if self._ewc_fisher_damping_beta > 0 and not self._normalize_fisher:
            raise ValueError("KFAC isotropic damping requires per-task trace normalization")
        if self._dual_ascent:
            raise ValueError("This shared KFAC comparison uses a fixed ewc_lambda")
        if as_bool(args.get("learn_alpha", False)):
            raise ValueError("KFAC Delta-W EWC currently requires fixed LoRA scaling")
        if as_bool(args.get("endpoint_eval", False)) or self._compute_full_fisher:
            raise ValueError("Legacy diagonal endpoint/Fisher diagnostics are not KFAC diagnostics")
        self._kfac_history = KFACEWCState()
        self._pending_kfac = None
        self._kfac_last_metrics = {}

    def _kfac_protocol(self):
        return {
            "fisher_mode": self._fisher_mode,
            "parameter_space": "unscaled_delta_w_ba",
            "kfac_grad_scale": self._kfac_grad_scale,
            "kfac_seq_agg": self._kfac_seq_agg,
            "normalize_task": bool(self._normalize_fisher),
            "normalize_accumulated": bool(self._normalize_accumulated_fisher),
            "normalization_eps": float(self._fisher_norm_eps),
            "damping_beta": float(self._ewc_fisher_damping_beta),
            "gamma": float(self._ewc_gamma),
            "accumulation": "exact_sum_of_kronecker_terms",
            "anchor": "latest_retained_task_endpoint",
        }

    def _compute_delta_fisher(self, loader, model, class_offset):
        if torch.distributed.is_initialized() or isinstance(self._network, torch.nn.DataParallel):
            raise ValueError("KFAC factor estimation currently supports a single process/device")
        from scripts.compute_all_fisher_stage1 import compute_lora_kfac_fisher

        was_training = model.training
        rng = self._capture_global_rng_state() if self._ewc_preserve_fisher_rng else None
        started = time.perf_counter()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            started = time.perf_counter()
        try:
            factors, _, _, _, count = compute_lora_kfac_fisher(
                mode=self._fisher_mode, net=model, loader=loader, device=self._device,
                task_start=int(class_offset), task_end=int(self._total_classes),
                max_batches=self._ewc_max_batches,
                kfac_grad_scale=self._kfac_grad_scale, kfac_seq_agg=self._kfac_seq_agg,
            )
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            if rng is not None:
                self._restore_global_rng_state(rng)
        translated = {
            f"{entry['module_name']}.{entry['branch']}": {"G": entry["G"], "C": entry["C"]}
            for entry in factors.values()
        }
        expected = {key: tuple(value.shape) for key, value in self._iter_delta_terms(model)}
        if set(translated) != set(expected):
            raise ValueError("KFAC estimator did not cover all effective Delta-W branches")
        for key, shape in expected.items():
            if translated[key]["G"].shape != (shape[0], shape[0]) or translated[key]["C"].shape != (shape[1], shape[1]):
                raise ValueError(f"KFAC factor shape mismatch: {key}")
        raw_trace = sum(float(v["G"].double().trace() * v["C"].double().trace()) for v in translated.values())
        self._pending_kfac = KFACEWCState.from_estimate(
            translated, normalize=self._normalize_fisher,
            beta=self._ewc_fisher_damping_beta, eps=self._fisher_norm_eps,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._kfac_last_metrics = {
            "kfac_samples": int(count), "kfac_blocks": len(translated),
            "kfac_raw_trace": raw_trace, "kfac_task_trace": self._pending_kfac.trace(),
            "kfac_seconds": time.perf_counter() - started,
        }
        self._last_fisher_estimation_metrics = {
            "fisher_mode": self._fisher_mode, "fisher_estimator_family": "kronecker_factored_empirical",
            "fisher_num_samples": int(count), "fisher_is_per_sample_empirical": False,
            **self._kfac_last_metrics,
        }
        self._last_fisher_norm_metrics = {
            "fisher_raw_trace": raw_trace,
            "fisher_normalized_trace": self._pending_kfac.trace(),
        }
        self._log(f"[KFAC-EWC][Estimate] {self._kfac_last_metrics}")
        return self._pending_kfac.diagonal()

    def _accumulate_fisher(self, old, new):
        del old, new  # The exact factors, not their diagonal mirror, are authoritative.
        if self._pending_kfac is None:
            raise RuntimeError("KFAC accumulation requires a freshly estimated task state")
        effective_gamma = self._ewc_gamma
        new_trace = self._pending_kfac.trace()
        if self._normalize_accumulated_fisher and self._kfac_history.blocks:
            old_mass = self._fisher_past_trace_mass or self._kfac_history.trace()
            effective_gamma *= old_mass / self._kfac_history.trace()
        else:
            old_mass = self._kfac_history.trace()
        self._kfac_history = self._kfac_history.accumulate(
            self._pending_kfac, gamma=effective_gamma,
            normalize=self._normalize_accumulated_fisher, eps=self._fisher_norm_eps,
        )
        self._pending_kfac = None
        self._fisher_past_trace_mass = self._ewc_gamma * old_mass + new_trace
        self._kfac_last_metrics.update({
            "kfac_history_tasks": self._kfac_history.tasks,
            "kfac_history_trace": self._kfac_history.trace(),
            "kfac_history_tensor_bytes": self._kfac_history.tensor_bytes(),
        })
        return self._kfac_history.diagonal()

    def _kfac_displacements(self, model):
        terms = dict(self._iter_delta_terms(model))
        keys = set(self._kfac_history.blocks)
        if set(terms) != keys or set(self._delta_reference) != keys:
            raise ValueError("KFAC history, Delta-W parameters and anchors must have identical keys")
        return {key: value - self._delta_reference[key].to(value) for key, value in terms.items()}

    def _past_fisher_penalty(self, model):
        if not self._kfac_history.blocks or self._ewc_lambda == 0:
            return torch.zeros((), device=self._device)
        return 0.5 * self._ewc_lambda * self._kfac_history.penalty(self._kfac_displacements(model))

    def _compute_delta_drift_metrics(self):
        if not self._kfac_history.blocks or not self._delta_reference:
            return {"delta_drift_available": 0}
        with torch.no_grad():
            deltas = self._kfac_displacements(self._unwrap_network())
            norm = sum(float(d.square().sum()) for d in deltas.values())
            drift = float(self._kfac_history.penalty(deltas))
        return {
            "delta_drift_available": 1, "delta_tensors": len(deltas),
            "delta_norm": norm, "fisher_weighted_drift": drift,
            "fisher_eta_weighted_drift": drift,
            "normalized_fisher_drift": drift / (norm + self._mechanism_eps),
            "normalized_fisher_eta_drift": drift / (norm + self._mechanism_eps),
            "ewc_penalty_value": 0.5 * self._ewc_lambda * drift,
            "fisher_geometry": "sum_of_kronecker_terms",
        }

    def _extra_fisher_checkpoint_state(self):
        if not self._kfac_history.blocks:
            raise ValueError("Cannot save a KFAC task checkpoint without historical factors")
        if self._kfac_history.tasks != self._cur_task + 1:
            raise ValueError("KFAC task checkpoints require consolidation of every completed task")
        anchor = {key: value.detach().cpu().clone() for key, value in self._iter_delta_terms(self._unwrap_network())}
        return {"kfac_ewc_state": {
            "format": "delta_w_kfac_ewc_v1", "protocol": self._kfac_protocol(),
            "history": self._kfac_history.state_dict(), "reference_delta": anchor,
        }}

    def _restore_extra_fisher_checkpoint_state(self, state):
        extra = state.get("kfac_ewc_state")
        if not isinstance(extra, dict) or extra.get("format") != "delta_w_kfac_ewc_v1":
            raise ValueError("Required KFAC factor checkpoint is missing or has an unsupported format")
        if extra.get("protocol") != self._kfac_protocol():
            raise ValueError("KFAC checkpoint protocol differs from the requested Fisher protocol")
        current = dict(self._iter_delta_terms(self._unwrap_network()))
        shapes = {key: tuple(value.shape) for key, value in current.items()}
        history = KFACEWCState.from_state_dict(extra["history"], shapes=shapes)
        if not history.blocks:
            raise ValueError("KFAC checkpoint history must not be empty")
        if history.tasks != self._cur_task + 1:
            raise ValueError("KFAC history task count does not match the restored task")
        anchor = extra.get("reference_delta", {})
        if set(anchor) != set(current):
            raise ValueError("KFAC checkpoint anchor key mismatch")
        for key, value in anchor.items():
            if not torch.is_tensor(value) or tuple(value.shape) != shapes[key] or not torch.isfinite(value).all():
                raise ValueError(f"Invalid KFAC anchor: {key}")
            scale = float(value.detach().abs().max())
            if not torch.allclose(value.float().cpu(), current[key].detach().float().cpu(), rtol=1e-5, atol=scale * 1e-7):
                raise ValueError(f"KFAC anchor does not match the restored task-end model: {key}")
        diagonal = history.diagonal()
        if set(diagonal) != set(self._fisher_past_delta):
            raise ValueError("KFAC checkpoint diagonal mirror key mismatch")
        for key, value in diagonal.items():
            if not torch.equal(value, self._fisher_past_delta[key]):
                raise ValueError(f"KFAC checkpoint diagonal mirror mismatch: {key}")
        self._kfac_history = history
        self._delta_reference = {key: value.detach().cpu().clone() for key, value in anchor.items()}
        self._pending_kfac = None
        self._log(f"[KFAC-EWC][Resume] tasks={history.tasks} blocks={len(history.blocks)} trace={history.trace():.6g}")

    def _load_auxiliary_task_state(self, checkpoint_dir, task_idx):
        # KFAC factors are mandatory even if a legacy caller omits the optional flag.
        paths = [os.path.join(checkpoint_dir, f"fisher_state_{int(task_idx)}.pt"),
                 os.path.join(checkpoint_dir, f"fisher_state_{int(task_idx):02d}.pt")]
        if not any(os.path.isfile(path) for path in paths):
            raise FileNotFoundError("KFAC resume requires a complete factor sidecar")
        self._kfac_history = KFACEWCState()
        super()._load_auxiliary_task_state(checkpoint_dir, task_idx)
        if not self._kfac_history.blocks:
            raise ValueError("KFAC resume requires a complete factor sidecar")

    def after_task(self):
        if self._ewc_skip_task_fisher:
            self._kfac_last_metrics = {"kfac_skipped": True, "kfac_samples": 0, "kfac_seconds": 0.0}
        super().after_task()
        self._last_mechanism_metrics.update({
            "method_type": "as1_normfisher_fishershape_kfac",
            "fisher_geometry": "sum_of_kronecker_terms", **self._kfac_last_metrics,
        })
        self._log(f"[KFAC-EWC][FinalMechanism] {self._last_mechanism_metrics}")
