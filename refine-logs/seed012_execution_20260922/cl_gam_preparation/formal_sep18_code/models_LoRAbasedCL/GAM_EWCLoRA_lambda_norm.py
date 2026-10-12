"""Fixed-lambda GAM EWCLoRA with trace-normalized delta-W Fisher.

This is a compact norm-Fisher variant of ``GAM_EWCLoRA_lambda.py``.  It keeps
the base learner's per-batch GAM + explicit delta-W EWC update:

    g_final = g_gam + g_ewc

but removes the lambda-control and drift-control layers from that implementation:

    - no adaptive/dual lambda update;
    - no safe lambda clipping;
    - no Euclidean trust-region scaling;
    - no soft delta-W penalty;
    - no in-task/path Fisher averaging.

The only Fisher extension here is task Fisher trace normalization before
cross-task accumulation:

    F_tilde = F / (sum(F) + eps)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models_LoRAbasedCL.GAM_EWCLoRA_lambda import Learner as _GAMLambdaLearner
from optimer_PerturabtionType.gam import GAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


_FISHER_MODE_ALIASES = {
    "full": "batch_empirical",
    "batch": "batch_empirical",
    "empirical": "batch_empirical",
    "max_pred": "max_pred_batch_empirical",
    "lr": "lr_batch_empirical",
    "sample": "sample_empirical",
    "true": "true_sample",
    "ncm": "ncm_empirical",
    "ncm_batch": "ncm_empirical",
}

_FISHER_ESTIMATOR_INFO = {
    "batch_empirical": {
        "display_name": "batch-gradient delta-W Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "max_pred_batch_empirical": {
        "display_name": "max-pred batch-gradient delta-W Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "lr_batch_empirical": {
        "display_name": "logit-reversed batch-gradient delta-W Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "sample_empirical": {
        "display_name": "per-sample empirical delta-W Fisher",
        "estimator_family": "per_sample_empirical_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_delta_w_gradients",
        "is_per_sample_empirical_fisher": True,
    },
    "lr_sample_empirical": {
        "display_name": "logit-reversed per-sample empirical delta-W Fisher",
        "estimator_family": "per_sample_empirical_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_delta_w_gradients",
        "is_per_sample_empirical_fisher": True,
    },
    "true_sample": {
        "display_name": "model-distribution delta-W Fisher diagonal",
        "estimator_family": "model_distribution_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "ncm_empirical": {
        "display_name": "NCM batch-gradient delta-W Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "ncm_batch_empirical": {
        "display_name": "NCM batch-gradient delta-W Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_delta_w_gradients",
        "is_per_sample_empirical_fisher": False,
    },
}


def _canonical_fisher_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    return _FISHER_MODE_ALIASES.get(mode, mode)


def _fisher_estimator_info(mode: str) -> Dict:
    info = dict(_FISHER_ESTIMATOR_INFO.get(str(mode), {}))
    info.setdefault("display_name", str(mode))
    info.setdefault("estimator_family", "unknown")
    info.setdefault("gradient_aggregation", "unknown")
    info.setdefault("is_per_sample_empirical_fisher", False)
    return info


class Learner(_GAMLambdaLearner):
    """GAM + delta-W EWC with fixed lambda and trace-normalized Fisher."""

    @staticmethod
    def _fixed_control_args(args):
        try:
            fixed = dict(args)
        except (TypeError, ValueError):
            return args

        fixed.update(
            {
                "ewc_dual_ascent": False,
                "gam_ewc_dual_ascent": False,
                "ewc_dual_use_joint_constraints": False,
                "ewc_dual_euclidean_delta": None,
                "ewc_dual_l2_delta": None,
                "ewc_dual_r_e_sq": None,
                "ewc_dual_spectral_delta": None,
                "ewc_dual_fmax_l2_delta": None,
                "ewc_dual_r_m_sq": None,
                "ewc_safe_lambda_cap": False,
                "gam_ewc_euclidean_trust": False,
                "ewc_euclidean_trust": False,
                "euclidean_trust": False,
                "gam_ewc_delta_penalty": False,
                "ewc_delta_penalty": False,
                "delta_w_penalty": False,
                "approximte_fisher_inTask": False,
                "approximate_fisher_inTask": False,
                "approximate_fisher_in_task": False,
            }
        )
        fixed.setdefault("ewc_dual_delta", args.get("ewc_dual_delta", 1e-3))
        return fixed

    def __init__(self, args):
        super().__init__(self._fixed_control_args(args))

        self._normalize_fisher = as_bool(args.get("ewc_normalize_fisher", True))
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))
        self._normalize_accumulated_fisher = as_bool(
            args.get(
                "ewc_normalize_accumulated_fisher",
                args.get("ewc_normalize_past_fisher", False),
            )
        )
        self._fisher_past_trace_mass = 0.0
        fisher_type = str(args.get("fisher_type", "")).strip().lower()
        default_fisher_mode = "ncm_empirical" if fisher_type == "ncm" else "batch_empirical"
        self._fisher_mode = _canonical_fisher_mode(args.get("fisher_mode", default_fisher_mode))
        self._ncm_fisher_tau = float(
            args.get("ncm_fisher_tau", args.get("fisher_ncm_tau", args.get("ncm_temperature", 16.0)))
        )
        self._last_fisher_estimation_metrics: Dict[str, float] = {}

        # Force fixed-lambda behavior even if the config contains inherited keys.
        self._dual_ascent = False
        self._dual_update_by_epoch = True
        self._dual_step_count = 0
        self._dual_use_joint_constraints = False
        self._dual_euclidean_budget = None
        self._dual_spectral_budget = None
        self._safe_lambda_cap = False

        self._gam_ewc_euclidean_trust = False
        self._gam_ewc_delta_penalty = False
        self._gam_ewc_delta_penalty_lambda = 0.0

        self._approximate_fisher_in_task = False
        self._in_task_fisher_interval = 0
        self._in_task_fisher_omega = 0.0
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}
        self._last_fisher_norm_metrics: Dict[str, float] = {}

        self._sync_lambda(float(self._ewc_lambda))

        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-lambda-norm] fixed_lambda=%.6g "
                "normalize_fisher=%s normalize_accumulated=%s norm_eps=%.3g fisher_mode=%s",
                float(self._ewc_lambda),
                self._normalize_fisher,
                self._normalize_accumulated_fisher,
                self._fisher_norm_eps,
                self._fisher_mode,
            )

    def _trace_normalize_fisher(
        self,
        fisher: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not fisher or not self._normalize_fisher:
            self._last_fisher_norm_metrics = {
                "ewc_normalize_fisher": bool(self._normalize_fisher),
                "fisher_raw_trace": self._fisher_trace(fisher),
                "fisher_normalized_trace": self._fisher_trace(fisher),
            }
            return fisher

        raw_trace = self._fisher_trace(fisher)
        denom = float(raw_trace) + float(self._fisher_norm_eps)
        normalized = {key: value / denom for key, value in fisher.items()}
        normalized_trace = self._fisher_trace(normalized)
        self._last_fisher_norm_metrics = {
            "ewc_normalize_fisher": True,
            "ewc_fisher_norm_eps": float(self._fisher_norm_eps),
            "fisher_raw_trace": float(raw_trace),
            "fisher_normalized_trace": float(normalized_trace),
        }
        return normalized

    @staticmethod
    def _match_fisher_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value_cpu = value.detach().cpu().to(dtype=reference.dtype)
        ref_cpu = reference.detach().cpu()
        if tuple(value_cpu.shape) == tuple(ref_cpu.shape):
            return value_cpu

        if value_cpu.ndim != ref_cpu.ndim:
            flat_value = value_cpu.reshape(-1)
            matched_flat = torch.zeros(ref_cpu.numel(), dtype=ref_cpu.dtype)
            n = min(int(flat_value.numel()), int(matched_flat.numel()))
            if n > 0:
                matched_flat[:n] = flat_value[:n]
            return matched_flat.view_as(ref_cpu)

        matched = torch.zeros_like(ref_cpu)
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref_cpu.shape[dim])))
            for dim in range(value_cpu.ndim)
        )
        matched[slices] = value_cpu[slices]
        return matched

    def _accumulate_fisher(
        self,
        old: Dict[str, torch.Tensor],
        new: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self._normalize_accumulated_fisher:
            return super()._accumulate_fisher(old, new)

        if not old:
            self._fisher_past_trace_mass = max(self._fisher_trace(new), 0.0)
            return self._trace_normalize_fisher(new)

        old_trace = self._fisher_trace(old)
        old_mass = (
            float(self._fisher_past_trace_mass)
            if self._fisher_past_trace_mass > 0.0
            else old_trace
        )
        old_scale = old_mass / (old_trace + float(self._fisher_norm_eps))
        new_trace = self._fisher_trace(new)

        combined: Dict[str, torch.Tensor] = {}
        for key in set(old.keys()) | set(new.keys()):
            if key in old and key in new:
                old_raw = old[key] * float(old_scale)
                new_t = self._match_fisher_shape(new[key], old_raw)
                combined[key] = old_raw * float(self._ewc_gamma) + new_t
            elif key in old:
                combined[key] = old[key] * float(old_scale) * float(self._ewc_gamma)
            else:
                combined[key] = new[key].detach().cpu().clone()

        self._fisher_past_trace_mass = float(self._ewc_gamma) * old_mass + new_trace
        normalized = self._trace_normalize_fisher(combined)
        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-lambda-norm] Normalized accumulated Fisher "
                "task=%d old_mass=%.6g new_trace=%.6g mass=%.6g trace=%.6g",
                self._cur_task,
                old_mass,
                new_trace,
                self._fisher_past_trace_mass,
                self._fisher_trace(normalized),
            )
        return normalized

    def _compute_delta_fisher(self, loader, model, class_offset: int) -> Dict[str, torch.Tensor]:
        raw_fisher = self._compute_delta_fisher_by_mode(
            mode=self._fisher_mode,
            loader=loader,
            model=model,
            class_offset=class_offset,
        )
        return self._trace_normalize_fisher(raw_fisher)

    def _compute_delta_fisher_by_mode(
        self,
        mode: str,
        loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, torch.Tensor]:
        mode = _canonical_fisher_mode(mode)
        if mode in {"batch_empirical", "max_pred_batch_empirical", "lr_batch_empirical"}:
            return self._compute_delta_batch_fisher_mode(mode, loader, model, class_offset)
        if mode in {"sample_empirical", "lr_sample_empirical", "true_sample"}:
            return self._compute_delta_sample_fisher_mode(mode, loader, model, class_offset)
        if mode in {"ncm_empirical", "ncm_batch_empirical"}:
            return self._compute_delta_ncm_fisher_mode(mode, loader, model, class_offset)
        raise ValueError(
            f"Unsupported delta-W fisher_mode={mode!r}. Use one of: "
            "batch_empirical, max_pred_batch_empirical, lr_batch_empirical, "
            "sample_empirical, lr_sample_empirical, true_sample, ncm_empirical. "
            "KFAC Fisher modes are intentionally not supported here because this "
            "learner stores diagonal Fisher in effective LoRA delta-W space."
        )

    def _current_task_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        class_end = int(getattr(self, "_total_classes", 0))
        if class_end <= int(class_offset):
            class_end = int(targets.max().item()) + 1 if targets.numel() else int(class_offset)
        mask = ((targets >= int(class_offset)) & (targets < class_end)).nonzero().view(-1)
        if mask.numel() == 0:
            return None, None
        return torch.index_select(inputs, 0, mask), torch.index_select(targets, 0, mask) - int(class_offset)

    def _current_task_logits(
        self,
        outputs,
        class_offset: int,
    ) -> torch.Tensor:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        class_end = int(getattr(self, "_total_classes", 0))
        if class_end > int(class_offset):
            return logits[:, int(class_offset):class_end]
        return logits[:, int(class_offset):]

    @staticmethod
    def _accumulate_hook_grads(
        fisher: Dict[str, torch.Tensor],
        hook_grads,
        weight: int = 1,
    ) -> None:
        for key, grad in hook_grads:
            grad_sq = grad.detach().cpu().pow(2) * int(weight)
            fisher[key] = grad_sq if key not in fisher else fisher[key] + grad_sq

    def _finalize_delta_fisher(
        self,
        fisher: Dict[str, torch.Tensor],
        num_samples: int,
        mode: str,
        loss_reduction: str,
    ) -> Dict[str, torch.Tensor]:
        if num_samples > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(num_samples)
        info = _fisher_estimator_info(mode)
        self._last_fisher_estimation_metrics = {
            "fisher_space": "delta_w",
            "fisher_mode": str(mode),
            "fisher_estimator": str(mode),
            "fisher_estimator_display": info["display_name"],
            "fisher_estimator_family": info["estimator_family"],
            "fisher_gradient_aggregation": info["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(info["is_per_sample_empirical_fisher"]),
            "fisher_num_samples": int(num_samples),
            "fisher_loss_reduction": str(loss_reduction),
        }
        return fisher

    def _compute_delta_batch_fisher_mode(
        self,
        mode: str,
        loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        num_samples = 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(tqdm(
                loader,
                desc=f"Computing delta-W Fisher [{mode}]",
                disable=not self._is_main_process,
            )):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_batch(inputs, targets, class_offset)
                if inputs is None:
                    continue

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)
                with torch.enable_grad():
                    logits = self._current_task_logits(model(inputs), class_offset)
                    if mode == "max_pred_batch_empirical":
                        fisher_logits = logits
                        fisher_targets = logits.argmax(dim=1).detach()
                    else:
                        fisher_logits = -logits if mode == "lr_batch_empirical" else logits
                        fisher_targets = local_targets
                    F.cross_entropy(fisher_logits, fisher_targets).backward()

                batch_size = int(inputs.size(0))
                self._accumulate_hook_grads(fisher, self._iter_hook_grads(model), batch_size)
                num_samples += batch_size
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        return self._finalize_delta_fisher(fisher, num_samples, mode, "cross_entropy_mean")

    def _compute_delta_sample_fisher_mode(
        self,
        mode: str,
        loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        num_samples = 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(tqdm(
                loader,
                desc=f"Computing delta-W Fisher [{mode}]",
                disable=not self._is_main_process,
            )):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_batch(inputs, targets, class_offset)
                if inputs is None:
                    continue

                for idx in range(int(inputs.size(0))):
                    x_i = inputs[idx:idx + 1]
                    y_i = local_targets[idx:idx + 1]
                    model.zero_grad(set_to_none=True)
                    self._clear_delta_hook_grads(model)
                    with torch.enable_grad():
                        logits = self._current_task_logits(model(x_i), class_offset)
                        if mode == "true_sample":
                            probs = torch.softmax(logits.detach(), dim=-1)
                            fisher_targets = torch.multinomial(probs, num_samples=1).squeeze(1)
                            fisher_logits = logits
                        else:
                            fisher_targets = y_i
                            fisher_logits = -logits if mode == "lr_sample_empirical" else logits
                        F.cross_entropy(fisher_logits, fisher_targets).backward()
                    self._accumulate_hook_grads(fisher, self._iter_hook_grads(model), 1)
                    num_samples += 1
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        return self._finalize_delta_fisher(fisher, num_samples, mode, "per_sample_cross_entropy")

    def _ncm_prototypes(
        self,
        class_offset: int,
        class_end: int,
        device,
        dtype,
    ) -> torch.Tensor:
        class_means = getattr(self, "_class_means", None)
        if class_means is None:
            raise ValueError(
                "ncm_empirical delta-W Fisher requires class means. "
                "Run with compute_nme_class_means enabled or use another fisher_mode."
            )
        if int(class_means.shape[0]) < int(class_end):
            raise ValueError(
                f"NCM Fisher class means cover {class_means.shape[0]} classes, "
                f"but class_end={class_end}."
            )
        prototypes = torch.as_tensor(
            class_means[int(class_offset):int(class_end)],
            device=device,
            dtype=dtype,
        )
        return F.normalize(prototypes, p=2, dim=1, eps=1e-12)

    def _ncm_logits(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(inputs)
        if isinstance(outputs, dict) and "features" in outputs:
            features = outputs["features"]
        elif hasattr(model, "extract_vector"):
            features = model.extract_vector(inputs)
        else:
            raise ValueError("ncm_empirical delta-W Fisher requires model features or extract_vector().")
        if features.dim() > 2:
            features = features.reshape(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
        return float(self._ncm_fisher_tau) * features.matmul(prototypes.t())

    def _compute_delta_ncm_fisher_mode(
        self,
        mode: str,
        loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        num_samples = 0
        class_end = int(getattr(self, "_total_classes", class_offset))
        first_param = next(model.parameters())
        prototypes = self._ncm_prototypes(
            class_offset,
            class_end,
            device=self._device,
            dtype=first_param.dtype,
        )

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(tqdm(
                loader,
                desc=f"Computing delta-W Fisher [{mode}]",
                disable=not self._is_main_process,
            )):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_batch(inputs, targets, class_offset)
                if inputs is None:
                    continue

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)
                with torch.enable_grad():
                    logits = self._ncm_logits(model, inputs, prototypes)
                    F.cross_entropy(logits, local_targets).backward()

                batch_size = int(inputs.size(0))
                self._accumulate_hook_grads(fisher, self._iter_hook_grads(model), batch_size)
                num_samples += batch_size
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        fisher = self._finalize_delta_fisher(fisher, num_samples, mode, "ncm_cross_entropy_mean")
        self._last_fisher_estimation_metrics.update(
            {
                "ncm_fisher_tau": float(self._ncm_fisher_tau),
                "ncm_fisher_num_classes": int(class_end - int(class_offset)),
            }
        )
        return fisher

    def _mix_in_task_path_fisher(
        self,
        task_fisher: Dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        del model
        self._in_task_fisher_last_metrics = {
            "approximte_fisher_inTask": False,
            "in_task_fisher_used_path": False,
            "in_task_fisher_interval": 0,
            "in_task_fisher_omega": 0.0,
            "in_task_fisher_snapshots": 0,
            "in_task_fisher_final_trace": self._fisher_trace(task_fisher),
            "in_task_fisher_path_trace": 0.0,
        }
        return self._trace_normalize_fisher(task_fisher)

    # ------------------------------------------------------------------
    # Fixed-lambda no-op controller hooks.  The base training loop calls these
    # hooks unconditionally, so this class makes them explicit no-ops.
    # ------------------------------------------------------------------
    def _disabled_safe_stats(self, lambda_value: Optional[float] = None) -> Dict[str, float]:
        value = float(self._ewc_lambda if lambda_value is None else lambda_value)
        return {
            "safe_lambda_enabled": 0.0,
            "safe_lambda_min_eff": 0.0,
            "safe_lambda_max_eff": value,
            "safe_lambda_raw_max": value,
            "safe_lambda_fmax": 0.0,
            "safe_lambda_lr": self._get_current_base_lr(),
            "safe_lambda_c": 0.0,
            "safe_lambda_min_overridden": 0.0,
            "safe_lambda_before_clip": value,
            "safe_lambda_after_clip": value,
            "safe_lambda_clipped": 0.0,
            "lambda_before_joint_clip": value,
            "lambda_after_joint_clip": value,
        }

    def _compute_safe_lambda_bounds(self, optimizer=None) -> Tuple[float, float, Dict[str, float]]:
        del optimizer
        return 0.0, float(self._ewc_lambda), self._disabled_safe_stats()

    def _clip_lambda_by_joint_bounds(
        self,
        lambda_value: float,
        optimizer=None,
        reason: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        del optimizer, reason
        return float(lambda_value), self._disabled_safe_stats(lambda_value)

    def _apply_joint_lambda_clip(self, optimizer=None, reason: str = "") -> Dict[str, float]:
        del optimizer, reason
        self._sync_lambda(float(self._ewc_lambda))
        return self._disabled_safe_stats()

    def _default_dual_stats(self, ewc_ratio: float = 0.0, optimizer=None) -> Dict[str, float]:
        del ewc_ratio, optimizer
        value = float(self._ewc_lambda)
        return {
            "dual_drift": 0.0,
            "dual_budget": 0.0,
            "dual_rho_budget": 0.0,
            "dual_violation": 0.0,
            "lambda_before": value,
            "lambda_after": value,
            "dual_updated": 0.0,
            "dual_mode": 0.0,
            "lambda_controller_enabled": 0.0,
            **self._default_dual_constraint_stats(),
            **self._disabled_safe_stats(value),
        }

    def _maybe_update_dual_lambda(
        self,
        model: torch.nn.Module,
        current_loss: float,
        current_ewc_ratio: float = 0.0,
        optimizer: Optional[GAM] = None,
    ) -> Dict[str, float]:
        del model, current_loss, current_ewc_ratio, optimizer
        return self._default_dual_stats()

    def _update_dual_lambda(
        self,
        model: torch.nn.Module,
        optimizer: Optional[GAM] = None,
    ) -> Dict[str, float]:
        del model, optimizer
        return self._default_dual_stats()

    def after_task(self):
        drift_before = self._compute_delta_drift_metrics() if self._mechanism_eval else {}
        base_model = self._unwrap_network()
        new_fisher = self._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        self._fisher_past_delta = self._accumulate_fisher(self._fisher_past_delta, new_fisher)

        full_fisher_diff_metrics = {}
        self._last_full_fisher_diff = {}
        if self._compute_full_fisher:
            full_fisher_diff_metrics = self._compute_full_fisher_diff_metrics(base_model)
            self._last_full_fisher_diff = dict(full_fisher_diff_metrics)

        if self._is_main_process:
            f_vals = list(new_fisher.values())
            if f_vals:
                nf_max = float(max(v.max().item() for v in f_vals))
                nf_sum = float(sum(v.sum().item() for v in f_vals))
                nf_numel = sum(int(v.numel()) for v in f_vals)
                nf_mean = nf_sum / max(nf_numel, 1)
                self._log(
                    f"[GAM-EWCLoRA-lambda-norm][FisherStats] task={self._cur_task} "
                    f"new_fisher tensors={len(f_vals)} numel={nf_numel} "
                    f"max={nf_max:.3e} mean={nf_mean:.3e} sum={nf_sum:.3e}"
                )
            self._log(
                "[GAM-EWCLoRA-lambda-norm] Updated trace-normalized delta-W Fisher "
                f"for {len(new_fisher)} delta tensors after task {self._cur_task}."
            )

        self._delta_reference = {}
        self._param_reference = {}
        self._param_name_by_id = {}
        self._param_reference_skipped_classifier = 0
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._in_task_fisher_snapshots = []

        if self._mechanism_eval or full_fisher_diff_metrics:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "gam_ewclora_lambda_norm",
                "lambda_ewc": float(self._ewc_lambda),
                "fixed_lambda": True,
                "gam_base_optimizer": self._gam_base_optimizer,
                "gam_grad_rho": float(self._gam_grad_rho),
                "gam_grad_norm_rho": float(self._gam_grad_norm_rho),
                "ewc_normalize_accumulated_fisher": bool(self._normalize_accumulated_fisher),
                "fisher_past_trace_mass": float(self._fisher_past_trace_mass),
                **self._last_fisher_estimation_metrics,
                **self._last_fisher_norm_metrics,
                **self._summarize_mechanism_stats(),
                **drift_before,
                **full_fisher_diff_metrics,
            }
            if self._is_main_process:
                self._log(
                    f"[GAM-EWCLoRA-lambda-norm][Mechanism] {self._last_mechanism_metrics}"
                )


GAMEWCLoRALambdaNorm = Learner
GAMEWCLoRALambda = Learner
EWCLoRA = Learner
