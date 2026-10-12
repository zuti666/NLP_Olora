"""AS1 trace-normalized Fisher with optional in-task Fisher averaging.

This variant intentionally does not implement projected gradients.  It keeps the
AS1/GAM update from ``as1_normfisher_gam.py`` and only changes Fisher estimation
and optional cross-task Fisher normalization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from models_LoRAbasedCL.as1_normfisher_gam import Learner as _AS1NormFisherLearner
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_AS1NormFisherLearner):
    """AS1 + normalized Fisher with optional raw path-Fisher mixing.

    Default behavior is identical to ``as1_normfisher_gam``:

    - final Fisher is trace-normalized by the parent AS1 setting;
    - in-task/path Fisher is disabled;
    - accumulated past Fisher is not re-normalized after cross-task accumulation.
    """

    @staticmethod
    def _optional_positive_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "false", "off"}:
                return None
            value = stripped
        parsed = float(value)
        return parsed if parsed > 0.0 else None

    def __init__(self, args):
        super().__init__(args)

        self._approximate_fisher_in_task = as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get(
                    "approximate_fisher_inTask",
                    args.get("approximate_fisher_in_task", False),
                ),
            )
        )
        self._in_task_fisher_interval = int(
            args.get(
                "in_task_fisher_interval",
                args.get(
                    "approximate_fisher_interval",
                    args.get("approximte_fisher_interval", 10),
                ),
            )
        )
        self._in_task_fisher_omega = min(
            1.0,
            max(
                0.0,
                float(
                    args.get(
                        "in_task_fisher_omega",
                        args.get(
                            "path_fisher_omega",
                            args.get(
                                "approximate_fisher_omega",
                                args.get("approximte_fisher_omega", 0.5),
                            ),
                        ),
                    )
                ),
            ),
        )
        self._normalize_accumulated_fisher = as_bool(
            args.get(
                "ewc_normalize_accumulated_fisher",
                args.get("ewc_normalize_past_fisher", False),
            )
        )
        self._in_task_weight_merge = as_bool(
            args.get(
                "in_task_weight_merge",
                args.get("weight_merge_with_in_task_fisher", False),
            )
        )
        self._in_task_weight_merge_start_task = int(
            args.get("in_task_weight_merge_start_task", 0)
        )
        # Task-endpoint weight merge: save θ_start at task begin, merge
        # θ = (1-ω)*θ_end + ω*θ_start after training (no in-task snapshots needed).
        self._task_endpoint_weight_merge = as_bool(
            args.get("task_endpoint_weight_merge", False)
        )
        self._task_endpoint_weight_merge_omega = min(
            1.0, max(0.0, float(args.get(
                "task_endpoint_weight_merge_omega",
                args.get("in_task_fisher_omega", 0.3),
            )))
        )
        self._task_start_weights: Dict[str, torch.Tensor] = {}
        self._fisher_past_trace_mass = 0.0
        self._in_task_fisher_snapshots: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_weight_snapshots: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}
        self._in_task_weight_merge_last_metrics: Dict[str, float] = {}
        self._in_task_lora_snapshot_tensors = 0
        self._in_task_weight_snapshot_tensors = 0
        self._last_task_new_fisher_stats: Dict[str, float] = {}
        self._train_log_sums: Dict[str, float] = {}
        self._train_log_count = 0
        self._delta_w_penalty = as_bool(
            args.get(
                "as1_delta_w_penalty",
                args.get("ewc_delta_w_penalty", args.get("ewc_delta_penalty", False)),
            )
        )
        self._delta_w_penalty_lambda = float(
            args.get(
                "as1_delta_w_penalty_lambda",
                args.get("ewc_delta_w_penalty_lambda", args.get("ewc_delta_penalty_lambda", 0.0)),
            )
        )
        self._delta_w_penalty_radius = self._optional_positive_float(
            args.get(
                "as1_delta_w_penalty_radius",
                args.get("ewc_delta_w_penalty_radius", args.get("ewc_delta_penalty_radius", None)),
            )
        )
        self._delta_w_penalty_radius_from_gam = as_bool(
            args.get(
                "as1_delta_w_penalty_radius_from_gam",
                args.get("ewc_delta_w_penalty_radius_from_gam", args.get("ewc_delta_penalty_radius_from_gam", False)),
            )
        )
        self._delta_w_penalty_radius_source = str(
            args.get(
                "as1_delta_w_penalty_radius_source",
                args.get("ewc_delta_w_penalty_radius_source", args.get("ewc_delta_penalty_radius_source", "gam_grad_rho")),
            )
        ).lower()
        self._delta_w_penalty_radius_scale = float(
            args.get(
                "as1_delta_w_penalty_radius_scale",
                args.get("ewc_delta_w_penalty_radius_scale", args.get("ewc_delta_penalty_radius_scale", 1.0)),
            )
        )
        self._delta_w_penalty_eps = float(
            args.get(
                "as1_delta_w_penalty_eps",
                args.get("ewc_delta_w_penalty_eps", args.get("ewc_delta_penalty_eps", 1e-12)),
            )
        )
        raw_delta_w_penalty_warn_kappa = float(
            args.get(
                "as1_delta_w_penalty_warn_kappa",
                args.get(
                    "ewc_delta_w_penalty_warn_kappa",
                    args.get(
                        "ewc_delta_penalty_warn_kappa",
                        args.get("delta_w_penalty_warn_kappa", 1.0),
                    ),
                ),
            )
        )
        if raw_delta_w_penalty_warn_kappa < 0.0:
            logger.warning(
                "[AS1-NormFisher-InTask][DeltaWPenalty] warn_kappa %.4g is negative; clamping to 0.",
                raw_delta_w_penalty_warn_kappa,
            )
        # kappa >= 1 is valid: warn_radius = kappa * rho can exceed rho, creating a
        # wider tolerance zone before the penalty activates (e.g. kappa=2 -> 2x GAM radius).
        self._delta_w_penalty_warn_kappa = max(0.0, raw_delta_w_penalty_warn_kappa)
        if self._delta_w_penalty_lambda < 0.0:
            logger.warning(
                "[AS1-NormFisher-InTask][DeltaWPenalty] negative lambda %.4g; clamping to 0.",
                self._delta_w_penalty_lambda,
            )
            self._delta_w_penalty_lambda = 0.0
        if self._delta_w_penalty and (
            self._delta_w_penalty_lambda <= 0.0
            or self._resolve_delta_w_penalty_radius() is None
        ):
            logger.warning(
                "[AS1-NormFisher-InTask][DeltaWPenalty] enabled but lambda/radius is invalid; "
                "penalty will stay inactive until both are positive."
            )

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask] in_task=%s interval=%d omega=%.4g "
                "normalize_task=%s normalize_accumulated=%s lambda=%.6g "
                "lambda_flat=%.6g gam_rho=%.4g gam_norm_rho=%.4g "
                "weight_merge=%s weight_merge_start_task=%d "
                "delta_w_penalty=%s delta_w_penalty_lambda=%.4g "
                "delta_w_penalty_radius=%s delta_w_penalty_from_gam=%s "
                "delta_w_penalty_source=%s delta_w_penalty_radius_scale=%.4g "
                "delta_w_penalty_warn_kappa=%.4g",
                self._approximate_fisher_in_task,
                self._in_task_fisher_interval,
                self._in_task_fisher_omega,
                self._normalize_fisher,
                self._normalize_accumulated_fisher,
                float(self._lambda_ewc),
                float(self._lambda_flat),
                float(getattr(self, "_gam_grad_rho", 0.0)),
                float(getattr(self, "_gam_grad_norm_rho", 0.0)),
                self._in_task_weight_merge,
                self._in_task_weight_merge_start_task,
                self._delta_w_penalty,
                self._delta_w_penalty_lambda,
                self._resolve_delta_w_penalty_radius(),
                self._delta_w_penalty_radius_from_gam,
                self._delta_w_penalty_radius_source,
                self._delta_w_penalty_radius_scale,
                self._delta_w_penalty_warn_kappa,
            )

    @staticmethod
    def _fisher_trace(fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        total = 0.0
        for value in fisher.values():
            total += float(value.detach().float().sum().item())
        return float(total)

    def _trace_normalize_fisher(
        self, fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not fisher:
            return fisher
        denom = self._fisher_trace(fisher) + float(self._fisher_norm_eps)
        return {key: value / denom for key, value in fisher.items()}

    @staticmethod
    def _summarize_tensor_dict(tensors: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if not tensors:
            return {
                "tensors": 0,
                "numel": 0,
                "max": 0.0,
                "mean": 0.0,
                "sum": 0.0,
            }

        total_sum = 0.0
        total_numel = 0
        max_value = 0.0
        for value in tensors.values():
            value_f = value.detach().float()
            if value_f.numel() == 0:
                continue
            total_sum += float(value_f.sum().item())
            total_numel += int(value_f.numel())
            max_value = max(max_value, float(value_f.max().item()))

        return {
            "tensors": int(len(tensors)),
            "numel": int(total_numel),
            "max": float(max_value),
            "mean": float(total_sum / max(total_numel, 1)),
            "sum": float(total_sum),
        }

    @staticmethod
    def _is_lora_like_name(name: str) -> bool:
        lower_name = name.lower()
        markers = (
            "linear_a",
            "linear_b",
            "lora_a",
            "lora_b",
            "w_as",
            "w_bs",
            "scaling_factor",
            "wrapped_param",
        )
        return any(marker in lower_name for marker in markers)

    def _count_lora_like_tensors(self, model: torch.nn.Module) -> int:
        return sum(1 for name, _ in model.named_parameters() if self._is_lora_like_name(name))

    def _iter_lora_factor_weights(self, model: torch.nn.Module):
        for module_name, module in self._iter_delta_modules(model):
            for attr in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"):
                factor = getattr(module, attr, None)
                weight = getattr(factor, "weight", None)
                if weight is None:
                    continue
                yield f"{module_name}.{attr}.weight", weight

    def _snapshot_lora_factor_weights(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        return {
            key: weight.detach().cpu().clone()
            for key, weight in self._iter_lora_factor_weights(model)
        }

    @staticmethod
    def _summarize_weight_snapshot(
        tensors: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        if not tensors:
            return {
                "tensors": 0,
                "numel": 0,
                "l2": 0.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
            }

        norm_sq = 0.0
        abs_sum = 0.0
        max_abs = 0.0
        total_numel = 0
        for value in tensors.values():
            value_f = value.detach().float()
            if value_f.numel() == 0:
                continue
            norm_sq += float(value_f.pow(2).sum().item())
            abs_sum += float(value_f.abs().sum().item())
            max_abs = max(max_abs, float(value_f.abs().max().item()))
            total_numel += int(value_f.numel())

        return {
            "tensors": int(len(tensors)),
            "numel": int(total_numel),
            "l2": float(norm_sq ** 0.5),
            "mean_abs": float(abs_sum / max(total_numel, 1)),
            "max_abs": float(max_abs),
        }

    def _reset_train_log_stats(self) -> None:
        self._train_log_sums = {}
        self._train_log_count = 0

    def _record_train_log_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                self._train_log_sums[key] = self._train_log_sums.get(key, 0.0) + float(value)
        self._train_log_count += 1

    def _summarize_train_log_stats(self) -> Dict[str, float]:
        if self._train_log_count <= 0:
            return {}
        denom = float(self._train_log_count)
        return {
            f"train_{key}_mean": value / denom
            for key, value in sorted(self._train_log_sums.items())
        }

    def _resolve_delta_w_penalty_radius(self) -> Optional[float]:
        if self._delta_w_penalty_radius is not None:
            return float(self._delta_w_penalty_radius)
        if not self._delta_w_penalty_radius_from_gam:
            return None

        src = self._delta_w_penalty_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(getattr(self, "_gam_grad_rho", getattr(self, "_as1_rho", 0.0)))
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(getattr(self, "_gam_grad_norm_rho", getattr(self, "_as1_norm_rho", 0.0)))
        else:
            return None
        return float(self._delta_w_penalty_radius_scale * rho)

    def _default_delta_w_penalty_stats(self) -> Dict[str, float]:
        radius = self._resolve_delta_w_penalty_radius()
        boundary = float(radius or 0.0)
        warn_radius = float(self._delta_w_penalty_warn_kappa * boundary)
        return {
            "delta_w_penalty_enabled": float(self._delta_w_penalty),
            "delta_w_penalty_configured": 0.0,
            "delta_w_penalty_active": 0.0,
            "delta_w_penalty_loss": 0.0,
            "delta_w_penalty_lambda": float(self._delta_w_penalty_lambda),
            "delta_w_penalty_budget": boundary,
            "delta_w_penalty_boundary": boundary,
            "delta_w_penalty_warn_kappa": float(self._delta_w_penalty_warn_kappa),
            "delta_w_penalty_warn_radius": warn_radius,
            "delta_w_penalty_drift_norm": 0.0,
            "delta_w_penalty_drift_sq": 0.0,
            "delta_w_penalty_ratio": 0.0,
            "delta_w_penalty_warn_ratio": 0.0,
            "delta_w_penalty_excess": 0.0,
            "delta_w_penalty_grad_norm": 0.0,
            "delta_w_penalty_tensors": 0.0,
        }

    def _delta_w_penalty_loss(self, model: torch.nn.Module):
        stats = self._default_delta_w_penalty_stats()
        zero = torch.tensor(0.0, device=self._device)
        if not self._delta_w_penalty:
            return zero, stats
        if int(self._cur_task) <= 0 or not self._delta_reference:
            return zero, stats

        rho = self._resolve_delta_w_penalty_radius()
        lam = float(self._delta_w_penalty_lambda)
        if rho is None or lam <= 0.0:
            return zero, stats

        eps = float(self._delta_w_penalty_eps)
        warn_kappa = float(self._delta_w_penalty_warn_kappa)
        warn_radius = float(warn_kappa * float(rho))
        delta_norm_sq = None
        tensor_count = 0
        for key, delta_now in self._iter_delta_terms(model):
            delta_ref = self._delta_reference.get(key)
            if delta_ref is None:
                continue
            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=torch.float32)
            update = delta_now.float() - delta_ref_t
            term = update.pow(2).sum()
            delta_norm_sq = term if delta_norm_sq is None else delta_norm_sq + term
            tensor_count += 1

        if delta_norm_sq is None:
            stats["delta_w_penalty_configured"] = 1.0
            return zero, stats

        delta_norm = torch.sqrt(delta_norm_sq + eps)
        if warn_radius <= 0.0:
            excess = delta_norm
            penalty = lam * delta_norm_sq
        else:
            excess = torch.clamp(delta_norm - warn_radius, min=0.0)
            penalty = lam * excess.pow(2)

        drift_norm = float(delta_norm.detach().item())
        drift_sq = float(delta_norm_sq.detach().item())
        excess_value = float(excess.detach().item())
        stats.update(
            {
                "delta_w_penalty_configured": 1.0,
                "delta_w_penalty_active": float(
                    tensor_count > 0 and (warn_radius <= 0.0 or excess_value > 0.0)
                ),
                "delta_w_penalty_loss": float(penalty.detach().item()),
                "delta_w_penalty_budget": float(rho),
                "delta_w_penalty_boundary": float(rho),
                "delta_w_penalty_warn_kappa": float(warn_kappa),
                "delta_w_penalty_warn_radius": float(warn_radius),
                "delta_w_penalty_drift_norm": drift_norm,
                "delta_w_penalty_drift_sq": drift_sq,
                "delta_w_penalty_ratio": float(drift_norm / (float(rho) + eps)),
                "delta_w_penalty_warn_ratio": float(drift_norm / (warn_radius + eps)),
                "delta_w_penalty_excess": excess_value,
                "delta_w_penalty_tensors": float(tensor_count),
            }
        )
        return penalty, stats

    def _compute_delta_w_penalty_grad_cache(self, params: List[torch.nn.Parameter]):
        base_model = self._unwrap_network()
        with torch.enable_grad():
            penalty, stats = self._delta_w_penalty_loss(base_model)

        penalty_value = float(penalty.detach().item())
        if stats.get("delta_w_penalty_active", 0.0) <= 0.0 or not penalty.requires_grad:
            return penalty_value, {}, stats

        grads = torch.autograd.grad(penalty, params, allow_unused=True)
        grad_cache: Dict[int, torch.Tensor] = {}
        grad_norm_sq = 0.0
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_detached = grad.detach().clone()
            grad_cache[id(param)] = grad_detached
            grad_norm_sq += float(grad_detached.pow(2).sum().item())

        stats["delta_w_penalty_grad_norm"] = float(grad_norm_sq ** 0.5)
        return penalty_value, grad_cache, stats

    def _current_fisher_drift_value(self) -> float:
        if not self._fisher_past_delta or not self._delta_reference:
            return 0.0
        metrics = self._compute_delta_drift_metrics()
        return float(metrics.get("fisher_weighted_drift", 0.0))

    def _log_task_start_state(self) -> None:
        if not self._is_main_process:
            return

        fisher_stats = self._summarize_tensor_dict(self._fisher_past_delta)
        has_fisher = bool(self._fisher_past_delta)
        has_ref = bool(self._delta_reference)
        ref_info = (
            f"ref_tensors={len(self._delta_reference)}"
            if has_ref
            else "ref=EMPTY"
        )
        fisher_info = ""
        if has_fisher:
            fisher_info = (
                f"F_tensors={int(fisher_stats['tensors'])} "
                f"numel={int(fisher_stats['numel'])} "
                f"F_max={fisher_stats['max']:.3e} "
                f"F_mean={fisher_stats['mean']:.3e} "
                f"F_sum={fisher_stats['sum']:.3e}"
            )

        self._log(
            f"[AS1-NormFisher-InTask][TaskStart] task={self._cur_task} "
            f"lambda={float(self._lambda_ewc):.3e} "
            f"lambda_flat={float(self._lambda_flat):.3g} "
            f"delta_w_penalty={self._delta_w_penalty} "
            f"delta_w_penalty_lambda={self._delta_w_penalty_lambda:.3e} "
            f"delta_w_penalty_radius={self._resolve_delta_w_penalty_radius()} "
            f"delta_w_penalty_warn_kappa={self._delta_w_penalty_warn_kappa:.3g} "
            f"has_fisher={has_fisher} {fisher_info} | "
            f"has_ref={has_ref} {ref_info}"
        )

    def _compute_raw_delta_fisher(
        self, loader, model, class_offset: int
    ) -> Dict[str, torch.Tensor]:
        return self._compute_delta_fisher_by_mode(
            mode=self._fisher_mode,
            loader=loader,
            model=model,
            class_offset=class_offset,
        )

    def _compute_delta_fisher(
        self, loader, model, class_offset: int
    ) -> Dict[str, torch.Tensor]:
        final_raw = self._compute_raw_delta_fisher(loader, model, class_offset)
        mixed = self._mix_in_task_fisher(final_raw)
        self._last_task_new_fisher_stats = self._summarize_tensor_dict(mixed)
        return mixed

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        if not self._approximate_fisher_in_task:
            return False
        if self._in_task_fisher_interval <= 0:
            return False
        if self._in_task_fisher_omega <= 0.0:
            return False
        return int(epoch_idx) % int(self._in_task_fisher_interval) == 0

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: torch.nn.Module,
        optimizer,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        was_training = bool(model.training)
        fisher_loader = self._get_current_fisher_loader(train_loader)
        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask] Computing raw Fisher snapshot "
                "task=%d epoch=%d",
                self._cur_task,
                epoch_idx,
            )
        try:
            fisher = self._compute_raw_delta_fisher(fisher_loader, model, class_offset)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        snapshot = {
            key: value.detach().cpu().clone()
            for key, value in fisher.items()
        }
        self._in_task_fisher_snapshots.append((int(epoch_idx), snapshot))
        weight_snapshot: Dict[str, torch.Tensor] = {}
        if self._in_task_weight_merge:
            weight_snapshot = self._snapshot_lora_factor_weights(model)
            if weight_snapshot:
                self._in_task_weight_snapshots.append((int(epoch_idx), weight_snapshot))
                self._in_task_weight_snapshot_tensors += int(len(weight_snapshot))
        fisher_trace = self._fisher_trace(snapshot)
        lora_tensors = self._count_lora_like_tensors(model)
        self._in_task_lora_snapshot_tensors += int(lora_tensors)

        if self._is_main_process:
            weight_stats = self._summarize_weight_snapshot(weight_snapshot)
            logger.info(
                "[AS1-NormFisher-InTask] Saved raw Fisher snapshot "
                "task=%d epoch=%d count=%d fisher_trace=%.6g "
                "delta_tensors=%d lora_tensors=%d weight_tensors=%d "
                "weight_l2=%.6g",
                self._cur_task,
                epoch_idx,
                len(self._in_task_fisher_snapshots),
                fisher_trace,
                len(snapshot),
                lora_tensors,
                int(weight_stats["tensors"]),
                float(weight_stats["l2"]),
            )
            if fisher_trace <= float(self._fisher_norm_eps):
                logger.warning(
                    "[AS1-NormFisher-InTask] Fisher snapshot has near-zero trace "
                    "task=%d epoch=%d. Check that model_name triggers a LoRA backbone.",
                    self._cur_task,
                    epoch_idx,
                )
        return len(self._in_task_fisher_snapshots)

    def _mean_in_task_raw_fisher(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None

        accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _, fisher_snapshot in self._in_task_fisher_snapshots:
            for key, value in fisher_snapshot.items():
                value_cpu = value.detach().cpu()
                if key not in accum:
                    accum[key] = value_cpu.clone()
                    counts[key] = 1
                else:
                    accum[key] = accum[key] + value_cpu
                    counts[key] += 1

        return {
            key: value / float(max(counts.get(key, 1), 1))
            for key, value in accum.items()
        }

    def _mean_in_task_weight_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_weight_snapshots:
            return None

        accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _, weight_snapshot in self._in_task_weight_snapshots:
            for key, value in weight_snapshot.items():
                value_cpu = value.detach().cpu()
                if key not in accum:
                    accum[key] = value_cpu.clone()
                    counts[key] = 1
                else:
                    accum[key] = accum[key] + value_cpu
                    counts[key] += 1

        return {
            key: value / float(max(counts.get(key, 1), 1))
            for key, value in accum.items()
        }

    @staticmethod
    def _match_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value_cpu = value.detach().cpu().to(dtype=reference.dtype)
        ref_cpu = reference.detach().cpu()
        if tuple(value_cpu.shape) == tuple(ref_cpu.shape):
            return value_cpu

        matched = torch.zeros_like(ref_cpu)
        if value_cpu.ndim != ref_cpu.ndim:
            return matched
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref_cpu.shape[dim])))
            for dim in range(value_cpu.ndim)
        )
        matched[slices] = value_cpu[slices]
        return matched

    def _mix_in_task_fisher(
        self, final_raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        path_raw = self._mean_in_task_raw_fisher()
        final_trace = self._fisher_trace(final_raw)
        # Persist raw trace so baseLoRA can write it to fisher_state checkpoint.
        self._fisher_new_task_raw_trace = float(final_trace)
        path_trace = self._fisher_trace(path_raw)
        snapshot_epochs = [epoch for epoch, _ in self._in_task_fisher_snapshots]
        lora_snapshot_tensors = int(self._in_task_lora_snapshot_tensors)

        use_path = (
            self._approximate_fisher_in_task
            and self._in_task_fisher_omega > 0.0
            and path_raw is not None
        )
        if not use_path:
            self._in_task_fisher_last_metrics = {
                "in_task_fisher_used_path": False,
                "in_task_fisher_interval": int(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(self._in_task_fisher_omega),
                "in_task_fisher_snapshots": int(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
                "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_fisher_final_raw_trace": float(final_trace),
                "in_task_fisher_path_raw_trace": float(path_trace),
                "in_task_fisher_mixed_raw_trace": float(final_trace),
                "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
                "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
            }
            if self._is_main_process:
                logger.info(
                    "[AS1-NormFisher-InTask] Using endpoint Fisher only "
                    "task=%d snapshots=%d final_trace=%.6g path_trace=%.6g "
                    "normalize_task=%s",
                    self._cur_task,
                    len(snapshot_epochs),
                    final_trace,
                    path_trace,
                    self._normalize_fisher,
                )
            if not self._normalize_fisher or not final_raw:
                return final_raw
            return self._trace_normalize_fisher(final_raw)

        omega = float(self._in_task_fisher_omega)
        if not self._normalize_fisher:
            mixed_raw: Dict[str, torch.Tensor] = {}
            for key in set(final_raw.keys()) | set(path_raw.keys()):
                final_value = final_raw.get(key)
                path_value = path_raw.get(key)
                if final_value is None and path_value is None:
                    continue
                if final_value is None:
                    mixed_raw[key] = path_value.detach().cpu().clone()
                elif path_value is None:
                    mixed_raw[key] = final_value.detach().cpu().clone()
                else:
                    final_t = final_value.detach().cpu()
                    path_t = self._match_shape(path_value, final_t)
                    mixed_raw[key] = (1.0 - omega) * final_t + omega * path_t

            self._in_task_fisher_last_metrics = {
                "in_task_fisher_used_path": True,
                "in_task_fisher_interval": int(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(omega),
                "in_task_fisher_snapshots": int(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
                "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(self._fisher_trace(mixed_raw)),
                "in_task_fisher_final_raw_trace": float(final_trace),
                "in_task_fisher_path_raw_trace": float(path_trace),
                "in_task_fisher_mixed_raw_trace": float(self._fisher_trace(mixed_raw)),
                "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
                "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
            }
            if self._is_main_process:
                logger.info(
                    "[AS1-NormFisher-InTask] Mixed raw final/path Fisher "
                    "task=%d snapshots=%d omega=%.4g final_trace=%.6g "
                    "path_trace=%.6g mixed_trace=%.6g lora_snapshot_tensors=%d",
                    self._cur_task,
                    len(snapshot_epochs),
                    omega,
                    final_trace,
                    path_trace,
                    self._fisher_trace(mixed_raw),
                    lora_snapshot_tensors,
                )
            return mixed_raw

        final_norm = self._trace_normalize_fisher(final_raw)
        path_norm = self._trace_normalize_fisher(path_raw)
        mixed: Dict[str, torch.Tensor] = {}
        for key in set(final_norm.keys()) | set(path_norm.keys()):
            final_value = final_norm.get(key)
            path_value = path_norm.get(key)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[key] = path_value.detach().cpu().clone()
            elif path_value is None:
                mixed[key] = final_value.detach().cpu().clone()
            else:
                final_t = final_value.detach().cpu()
                path_t = self._match_shape(path_value, final_t)
                mixed[key] = (1.0 - omega) * final_t + omega * path_t

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "in_task_fisher_used_path": True,
            "in_task_fisher_interval": int(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": int(len(snapshot_epochs)),
            "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
            "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_fisher_final_raw_trace": float(final_trace),
            "in_task_fisher_path_raw_trace": float(path_trace),
            "in_task_fisher_mixed_raw_trace": float(mixed_trace),
            "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
            "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
        }
        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask] Mixed normalized final/path Fisher task=%d "
                "snapshots=%d omega=%.4g final_trace=%.6g path_trace=%.6g "
                "mixed_trace=%.6g lora_snapshot_tensors=%d",
                self._cur_task,
                len(snapshot_epochs),
                omega,
                final_trace,
                path_trace,
                mixed_trace,
                lora_snapshot_tensors,
            )
        return mixed

    def _accumulate_fisher(
        self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]
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
                new_t = self._match_shape(new[key], old_raw)
                combined[key] = old_raw * float(self._ewc_gamma) + new_t
            elif key in old:
                combined[key] = old[key] * float(old_scale) * float(self._ewc_gamma)
            else:
                combined[key] = new[key].detach().cpu().clone()

        self._fisher_past_trace_mass = float(self._ewc_gamma) * old_mass + new_trace
        normalized = self._trace_normalize_fisher(combined)

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask] Normalized accumulated Fisher "
                "task=%d old_mass=%.6g new_trace=%.6g mass=%.6g trace=%.6g",
                self._cur_task,
                old_mass,
                new_trace,
                self._fisher_past_trace_mass,
                self._fisher_trace(normalized),
            )
        return normalized

    def _apply_in_task_weight_merge(self, model: torch.nn.Module) -> Dict[str, float]:
        snapshot_epochs = [epoch for epoch, _ in self._in_task_weight_snapshots]
        omega = float(self._in_task_fisher_omega)
        metrics: Dict[str, float] = {
            "in_task_weight_merge_enabled": float(bool(self._in_task_weight_merge)),
            "in_task_weight_merge_applied": 0.0,
            "in_task_weight_merge_omega": float(omega),
            "in_task_weight_merge_snapshots": float(len(snapshot_epochs)),
            "in_task_weight_merge_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
            "in_task_weight_merge_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
            "in_task_weight_snapshot_tensors": float(self._in_task_weight_snapshot_tensors),
        }

        if not self._in_task_weight_merge:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics
        if int(self._cur_task) < int(self._in_task_weight_merge_start_task):
            self._in_task_weight_merge_last_metrics = metrics
            return metrics
        if omega <= 0.0:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics

        path_mean = self._mean_in_task_weight_snapshot()
        if not path_mean:
            self._in_task_weight_merge_last_metrics = metrics
            if self._is_main_process:
                logger.warning(
                    "[AS1-NormFisher-InTask][WeightMerge] requested but no "
                    "weight snapshots are available for task=%d",
                    self._cur_task,
                )
            return metrics

        final_norm_sq = 0.0
        path_norm_sq = 0.0
        merged_norm_sq = 0.0
        shift_norm_sq = 0.0
        path_delta_norm_sq = 0.0
        tensor_count = 0
        numel = 0

        with torch.no_grad():
            for key, weight in self._iter_lora_factor_weights(model):
                path_value = path_mean.get(key)
                if path_value is None:
                    continue

                final_cpu = weight.detach().cpu()
                path_t = self._match_shape(path_value, final_cpu)

                # Convex, normalized weighted average.  When omega=0.5 this is
                # (final + path_mean) / 2, not an unnormalized sum.
                merged = final_cpu * (1.0 - omega) + path_t * omega

                final_f = final_cpu.float()
                path_f = path_t.float()
                merged_f = merged.float()
                final_norm_sq += float(final_f.pow(2).sum().item())
                path_norm_sq += float(path_f.pow(2).sum().item())
                merged_norm_sq += float(merged_f.pow(2).sum().item())
                shift_norm_sq += float((merged_f - final_f).pow(2).sum().item())
                path_delta_norm_sq += float((path_f - final_f).pow(2).sum().item())
                tensor_count += 1
                numel += int(final_cpu.numel())

                weight.data.copy_(merged.to(device=weight.device, dtype=weight.dtype))

        metrics.update(
            {
                "in_task_weight_merge_applied": float(tensor_count > 0),
                "in_task_weight_merge_tensors": float(tensor_count),
                "in_task_weight_merge_numel": float(numel),
                "in_task_weight_final_l2": float(final_norm_sq ** 0.5),
                "in_task_weight_path_l2": float(path_norm_sq ** 0.5),
                "in_task_weight_merged_l2": float(merged_norm_sq ** 0.5),
                "in_task_weight_merge_shift_l2": float(shift_norm_sq ** 0.5),
                "in_task_weight_path_delta_l2": float(path_delta_norm_sq ** 0.5),
            }
        )
        self._in_task_weight_merge_last_metrics = metrics

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask][WeightMerge] task=%d applied=%s "
                "omega=%.4g snapshots=%d tensors=%d final_l2=%.6g "
                "path_l2=%.6g merged_l2=%.6g shift_l2=%.6g",
                self._cur_task,
                bool(tensor_count > 0),
                omega,
                len(snapshot_epochs),
                tensor_count,
                metrics["in_task_weight_final_l2"],
                metrics["in_task_weight_path_l2"],
                metrics["in_task_weight_merged_l2"],
                metrics["in_task_weight_merge_shift_l2"],
            )
        return metrics

    def _apply_endpoint_weight_merge(self, model: torch.nn.Module) -> Dict[str, float]:
        """Merge task-start weights with task-end weights: θ = (1-ω)·θ_end + ω·θ_start.

        Simpler alternative to in-task path merge — no snapshot scheduling needed,
        just one save at task start and one merge at task end.
        """
        omega = float(self._task_endpoint_weight_merge_omega)
        if not self._task_endpoint_weight_merge or not self._task_start_weights or omega <= 0.0:
            return {}

        final_norm_sq = start_norm_sq = merged_norm_sq = shift_norm_sq = 0.0
        tensor_count = numel = 0

        with torch.no_grad():
            for key, weight in self._iter_lora_factor_weights(model):
                start_value = self._task_start_weights.get(key)
                if start_value is None:
                    continue
                end_cpu = weight.detach().cpu()
                start_t = self._match_shape(start_value, end_cpu)
                merged = end_cpu * (1.0 - omega) + start_t * omega
                end_f, start_f, merged_f = end_cpu.float(), start_t.float(), merged.float()
                final_norm_sq += float(end_f.pow(2).sum())
                start_norm_sq += float(start_f.pow(2).sum())
                merged_norm_sq += float(merged_f.pow(2).sum())
                shift_norm_sq += float((merged_f - end_f).pow(2).sum())
                tensor_count += 1
                numel += end_cpu.numel()
                weight.data.copy_(merged.to(device=weight.device, dtype=weight.dtype))

        metrics = {
            "endpoint_weight_merge_applied": float(tensor_count > 0),
            "endpoint_weight_merge_omega": omega,
            "endpoint_weight_merge_tensors": float(tensor_count),
            "endpoint_weight_merge_numel": float(numel),
            "endpoint_weight_final_l2": float(final_norm_sq ** 0.5),
            "endpoint_weight_start_l2": float(start_norm_sq ** 0.5),
            "endpoint_weight_merged_l2": float(merged_norm_sq ** 0.5),
            "endpoint_weight_shift_l2": float(shift_norm_sq ** 0.5),
        }
        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask][EndpointMerge] task=%d omega=%.4g "
                "tensors=%d final_l2=%.6g start_l2=%.6g merged_l2=%.6g shift_l2=%.6g",
                self._cur_task, omega, tensor_count,
                metrics["endpoint_weight_final_l2"],
                metrics["endpoint_weight_start_l2"],
                metrics["endpoint_weight_merged_l2"],
                metrics["endpoint_weight_shift_l2"],
            )
        return metrics

    def _build_direction_projector(
        self,
        clean_grad_cache: Dict[int, torch.Tensor],
        fisher_grad_cache: Dict[int, torch.Tensor],
        delta_w_penalty_grad_cache: Optional[Dict[int, torch.Tensor]] = None,
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self._mechanism_eps)
        delta_w_penalty_grad_cache = delta_w_penalty_grad_cache or {}

        def projector(param_groups):
            clean_norm_sq = 0.0
            gam_norm_sq = 0.0
            flat_norm_sq = 0.0
            fisher_norm_sq = 0.0
            delta_w_penalty_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = 0.0
            clean_flat_dot = 0.0
            clean_fisher_dot = 0.0
            clean_delta_w_penalty_dot = 0.0
            flat_fisher_dot = 0.0
            flat_delta_w_penalty_dot = 0.0
            fisher_delta_w_penalty_dot = 0.0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    fisher_grad = fisher_grad_cache.get(id(param))
                    delta_w_penalty_grad = delta_w_penalty_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()

                    if clean_grad is None:
                        combined_grad = gam_grad
                    elif gam_grad is None:
                        combined_grad = clean_grad.clone()
                    else:
                        combined_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if fisher_grad is not None:
                        combined_grad = (
                            fisher_grad.clone()
                            if combined_grad is None
                            else combined_grad + fisher_grad
                        )

                    if delta_w_penalty_grad is not None:
                        combined_grad = (
                            delta_w_penalty_grad.clone()
                            if combined_grad is None
                            else combined_grad + delta_w_penalty_grad
                        )

                    if combined_grad is None:
                        continue

                    if self._mechanism_grad_stats:
                        if clean_grad is not None:
                            clean_norm_sq += float(clean_grad.pow(2).sum().item())
                        if gam_grad is not None:
                            gam_norm_sq += float(gam_grad.pow(2).sum().item())
                        if fisher_grad is not None:
                            fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                        if delta_w_penalty_grad is not None:
                            delta_w_penalty_norm_sq += float(
                                delta_w_penalty_grad.pow(2).sum().item()
                            )
                        combined_norm_sq += float(combined_grad.pow(2).sum().item())
                        if clean_grad is not None and gam_grad is not None:
                            flat_grad = gam_grad - clean_grad
                            flat_norm_sq += float(flat_grad.pow(2).sum().item())
                            clean_gam_dot += float((clean_grad * gam_grad).sum().item())
                            clean_flat_dot += float((clean_grad * flat_grad).sum().item())
                            if fisher_grad is not None:
                                flat_fisher_dot += float(
                                    (flat_grad * fisher_grad).sum().item()
                                )
                            if delta_w_penalty_grad is not None:
                                flat_delta_w_penalty_dot += float(
                                    (flat_grad * delta_w_penalty_grad).sum().item()
                                )
                        if clean_grad is not None and fisher_grad is not None:
                            clean_fisher_dot += float((clean_grad * fisher_grad).sum().item())
                        if clean_grad is not None and delta_w_penalty_grad is not None:
                            clean_delta_w_penalty_dot += float(
                                (clean_grad * delta_w_penalty_grad).sum().item()
                            )
                        if fisher_grad is not None and delta_w_penalty_grad is not None:
                            fisher_delta_w_penalty_dot += float(
                                (fisher_grad * delta_w_penalty_grad).sum().item()
                            )

                    if param.grad is None:
                        param.grad = combined_grad.clone()
                    else:
                        param.grad.data.copy_(combined_grad)

            if self._mechanism_grad_stats:
                clean_norm = clean_norm_sq ** 0.5
                gam_norm = gam_norm_sq ** 0.5
                flat_norm = flat_norm_sq ** 0.5
                fisher_norm = fisher_norm_sq ** 0.5
                delta_w_penalty_norm = delta_w_penalty_norm_sq ** 0.5
                combined_norm = combined_norm_sq ** 0.5
                self._accumulate_mechanism_grad_stats(
                    {
                        "clean_norm": clean_norm,
                        "gam_norm": gam_norm,
                        "flat_component_norm": flat_norm,
                        "fisher_norm": fisher_norm,
                        "delta_w_penalty_norm": delta_w_penalty_norm,
                        "combined_norm": combined_norm,
                        "flat_to_clean_norm_ratio": flat_norm / (clean_norm + eps),
                        "fisher_to_clean_norm_ratio": fisher_norm / (clean_norm + eps),
                        "delta_w_penalty_to_clean_norm_ratio": (
                            delta_w_penalty_norm / (clean_norm + eps)
                        ),
                        "cos_clean_gam": self._cosine_from_sums(
                            clean_gam_dot, clean_norm_sq, gam_norm_sq, eps
                        ),
                        "cos_clean_flat": self._cosine_from_sums(
                            clean_flat_dot, clean_norm_sq, flat_norm_sq, eps
                        ),
                        "cos_clean_fisher": self._cosine_from_sums(
                            clean_fisher_dot, clean_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_clean_delta_w_penalty": self._cosine_from_sums(
                            clean_delta_w_penalty_dot,
                            clean_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "cos_flat_fisher": self._cosine_from_sums(
                            flat_fisher_dot, flat_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_flat_delta_w_penalty": self._cosine_from_sums(
                            flat_delta_w_penalty_dot,
                            flat_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "cos_fisher_delta_w_penalty": self._cosine_from_sums(
                            fisher_delta_w_penalty_dot,
                            fisher_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                    }
                )

        return projector

    def _step_batch_with_log(
        self,
        optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ):
        if self._optimizer_type != "gam":
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            fisher_loss = self._past_fisher_penalty(self._unwrap_network())
            delta_w_penalty_loss, delta_w_penalty_stats = self._delta_w_penalty_loss(
                self._unwrap_network()
            )
            total_loss = task_loss + fisher_loss + delta_w_penalty_loss
            total_loss.backward()
            optimizer.step()

            ce_value = float(task_loss.detach().item())
            fisher_value = float(fisher_loss.detach().item())
            delta_w_penalty_value = float(delta_w_penalty_loss.detach().item())
            self._record_mechanism_loss_stats(ce_value, fisher_value)
            log_cache: Dict[str, float] = {
                "ce_clean": ce_value,
                "gam": ce_value,
                "ewc": fisher_value,
                "delta_w_penalty": delta_w_penalty_value,
                "lambda_ewc": float(self._lambda_ewc),
                "ewc_gam_ratio": float(fisher_value / (ce_value + float(self._mechanism_eps))),
                "delta_w_penalty_gam_ratio": float(
                    delta_w_penalty_value / (ce_value + float(self._mechanism_eps))
                ),
            }
            log_cache.update(delta_w_penalty_stats)
            return outputs["logits"].detach(), float(total_loss.detach().item()), log_cache

        params = self._flatten_optimizer_params(optimizer)
        clean_loss_value, clean_grad_cache = self._compute_clean_task_grad_cache(
            params=params,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        (
            delta_w_penalty_value,
            delta_w_penalty_grad_cache,
            delta_w_penalty_stats,
        ) = self._compute_delta_w_penalty_grad_cache(params)
        self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)

        log_cache: Dict[str, float] = {
            "ce_clean": float(clean_loss_value),
            "gam": float(clean_loss_value),
            "ewc": float(fisher_loss_value),
            "delta_w_penalty": float(delta_w_penalty_value),
            "lambda_ewc": float(self._lambda_ewc),
        }
        log_cache.update(delta_w_penalty_stats)

        optimizer.set_direction_projector(
            self._build_direction_projector(
                clean_grad_cache,
                fisher_grad_cache,
                delta_w_penalty_grad_cache,
            )
        )

        call_count = [0]

        def closure():
            call_count[0] += 1
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            if call_count[0] == 1:
                log_cache["ce_clean"] = float(task_loss.detach().item())
            else:
                log_cache["gam"] = float(task_loss.detach().item())
            loss_value = task_loss.detach()
            task_loss.backward()
            return outputs, loss_value

        try:
            outputs, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ce_value = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        log_cache["ce_clean"] = ce_value
        log_cache["ewc_gam_ratio"] = float(
            fisher_loss_value / (log_cache.get("gam", ce_value) + float(self._mechanism_eps))
        )
        log_cache["delta_w_penalty_gam_ratio"] = float(
            delta_w_penalty_value
            / (log_cache.get("gam", ce_value) + float(self._mechanism_eps))
        )

        total_loss_value = ce_value + float(fisher_loss_value) + float(delta_w_penalty_value)
        return outputs["logits"].detach(), float(total_loss_value), log_cache

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        logits, loss_value, _ = self._step_batch_with_log(
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        return logits, loss_value

    def _run_regularized_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        *,
        epochs: int,
        class_offset: int,
    ) -> None:
        self._snapshot_task_reference()
        if self._task_endpoint_weight_merge:
            self._task_start_weights = self._snapshot_lora_factor_weights(
                self._unwrap_network()
            )
        else:
            self._task_start_weights = {}
        self._in_task_fisher_snapshots = []
        self._in_task_weight_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._in_task_weight_merge_last_metrics = {}
        self._in_task_lora_snapshot_tensors = 0
        self._in_task_weight_snapshot_tensors = 0
        self._last_task_new_fisher_stats = {}
        self._reset_train_log_stats()
        self._log_task_start_state()
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            total_losses = 0.0
            ce_losses = 0.0
            gam_losses = 0.0
            ewc_losses = 0.0
            delta_w_penalty_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value, log_dict = self._step_batch_with_log(
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )
                total_losses += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", 0.0))
                gam_losses += float(log_dict.get("gam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                delta_w_penalty_losses += float(log_dict.get("delta_w_penalty", 0.0))
                loss_count += 1
                if self._mechanism_eval:
                    self._record_train_log_stats(log_dict)

                with torch.no_grad():
                    if class_offset > 0:
                        eval_logits = logits[:, class_offset:]
                        eval_targets = targets - class_offset
                    else:
                        eval_logits = logits
                        eval_targets = targets
                    preds = torch.argmax(eval_logits, dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += int(eval_targets.numel())

            if scheduler is not None:
                scheduler.step()

            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            denom = max(1, loss_count)
            gam_mean = gam_losses / denom
            ewc_mean = ewc_losses / denom
            dwpen_mean = delta_w_penalty_losses / denom
            epoch_ewc_ratio = float(ewc_mean / (gam_mean + float(self._mechanism_eps)))
            drift_display = self._current_fisher_drift_value()
            lam_ewc = float(self._lambda_ewc)
            lam_dw = float(self._delta_w_penalty_lambda)
            lam_df = lam_ewc * drift_display
            lam_df_gam = float(lam_df / (gam_mean + float(self._mechanism_eps)))
            # EWC and DWpen already include their lambdas (loss contributions)
            # D_F = Σ(F·Δθ²) is raw (no lambda); λ*D_F is the epoch-end weighted drift
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_losses / denom:.3f}, "
                f"Train_accy {train_acc:.2f}, "
                f"CE {ce_losses / denom:.3f}, GAM {gam_mean:.3f}, "
                f"EWC(λ·F·Δθ²) {ewc_mean:.3e}, "
                f"DWpen(λ·exc²) {dwpen_mean:.3e}, "
                f"D_F {drift_display:.3e}, "
                f"λ*D_F {lam_df:.3e}, "
                f"EWC/GAM {epoch_ewc_ratio:.3e}, "
                f"λ*D_F/GAM {lam_df_gam:.3e}, "
                f"λ_EWC {lam_ewc:.3g}, λ_DW {lam_dw:.3g}"
            )
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                cur_task_test_acc = self._compute_task_local_accuracy(
                    self._network, test_loader, class_offset, self._total_classes
                )
                info += f", Test_accy {test_acc:.2f}, CurTask_Test_accy {cur_task_test_acc:.2f}"
                val_loader = getattr(self, "val_loader", None)
                if val_loader is not None:
                    cur_task_val_acc = self._compute_task_local_accuracy(
                        self._network, val_loader, class_offset, self._total_classes
                    )
                    info += f", CurTask_Val_accy {cur_task_val_acc:.2f}"

            if self._should_capture_in_task_fisher(epoch + 1):
                snapshot_count = self._capture_in_task_fisher_snapshot(
                    train_loader=train_loader,
                    model=self._unwrap_network(),
                    optimizer=optimizer,
                    epoch_idx=epoch + 1,
                    class_offset=class_offset,
                )
                if snapshot_count > 0:
                    info += f", Fisher_snap {snapshot_count}"

            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process and info:
            self._log(info)
        self._apply_in_task_weight_merge(self._unwrap_network())
        self._apply_endpoint_weight_merge(self._unwrap_network())

    def after_task(self):
        super().after_task()
        if self._is_main_process:
            new_stats = dict(self._last_task_new_fisher_stats)
            if new_stats:
                self._log(
                    f"[AS1-NormFisher-InTask][FisherStats] task={self._cur_task} "
                    f"new_fisher tensors={int(new_stats['tensors'])} "
                    f"numel={int(new_stats['numel'])} max={new_stats['max']:.3e} "
                    f"mean={new_stats['mean']:.3e} sum={new_stats['sum']:.3e}"
                )
            accum_stats = self._summarize_tensor_dict(self._fisher_past_delta)
            if accum_stats["tensors"] > 0:
                self._log(
                    f"[AS1-NormFisher-InTask][FisherStats] task={self._cur_task} "
                    f"accum_fisher tensors={int(accum_stats['tensors'])} "
                    f"numel={int(accum_stats['numel'])} max={accum_stats['max']:.3e} "
                    f"mean={accum_stats['mean']:.3e} sum={accum_stats['sum']:.3e} "
                    f"trace_mass={float(self._fisher_past_trace_mass):.3e}"
                )
        if (
            (self._mechanism_eval or bool(getattr(self, "_compute_full_fisher", False)))
            and self._last_mechanism_metrics
        ):
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_gam_intask",
                    "approximate_fisher_in_task": bool(self._approximate_fisher_in_task),
                    "in_task_fisher_interval": int(self._in_task_fisher_interval),
                    "in_task_fisher_omega": float(self._in_task_fisher_omega),
                    "ewc_normalize_accumulated_fisher": bool(
                        self._normalize_accumulated_fisher
                    ),
                    "fisher_past_trace_mass": float(self._fisher_past_trace_mass),
                    "delta_w_penalty": bool(self._delta_w_penalty),
                    "delta_w_penalty_lambda": float(self._delta_w_penalty_lambda),
                    "delta_w_penalty_radius": float(
                        self._resolve_delta_w_penalty_radius() or 0.0
                    ),
                    "delta_w_penalty_warn_kappa": float(
                        self._delta_w_penalty_warn_kappa
                    ),
                    "delta_w_penalty_warn_radius": float(
                        self._delta_w_penalty_warn_kappa
                        * (self._resolve_delta_w_penalty_radius() or 0.0)
                    ),
                    "delta_w_penalty_radius_from_gam": bool(
                        self._delta_w_penalty_radius_from_gam
                    ),
                    "delta_w_penalty_radius_source": self._delta_w_penalty_radius_source,
                    "delta_w_penalty_radius_scale": float(
                        self._delta_w_penalty_radius_scale
                    ),
                    **self._in_task_fisher_last_metrics,
                    **self._in_task_weight_merge_last_metrics,
                    **self._summarize_train_log_stats(),
                }
            )
            if self._is_main_process:
                self._log(
                    f"[AS1-NormFisher-InTask][Mechanism] {self._last_mechanism_metrics}"
                )


EWCLoRA = Learner
