"""Full-model EWC + GAM with in-task Fisher and Euclidean L2 controls.

This learner is intentionally not a projected-gradient variant.  GAM still
builds its final direction, but before the optimizer step we replace the
gradient with a direct merge:

    g_final = g_clean + lambda_flat * (g_gam - g_clean) + g_ewc + g_l2_penalty

Optionally, hard Euclidean trust-region scaling is applied to this final merged
gradient so the cumulative task displacement remains inside an L2 budget.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from models_CL.ewc_gam_projected import (
    EWCGAMProjected,
    _BATCH_FISHER_MODES,
    _as_bool,
    _flatten_optimizer_params,
)
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class EWCGAMMergedInTaskEuc(EWCGAMProjected):
    """Full-model in-task Fisher + L2-constrained gradient merge.

    Unlike :class:`EWCGAMProjected`, this class does not project ``g_gam`` away
    from ``g_ewc``.  It only merges final gradient components.
    """

    def __init__(self, args):
        super().__init__(args)

        self._lambda_flat = float(args.get("lambda_flat", 1.0))
        self.ewc_weight = float(args.get("ewc_lambda", args.get("lambda", self.ewc_weight)))
        self.gamma = float(args.get("ewc_gamma", args.get("gamma", self.gamma)))
        self._normalize_fisher = _as_bool(args.get("ewc_normalize_fisher", False))
        self._normalize_accumulated_fisher = _as_bool(
            args.get("ewc_normalize_accumulated_fisher", args.get("ewc_normalize_past_fisher", False))
        )
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))
        self._fisher_past_trace_mass = 0.0

        self._approximate_fisher_in_task = _as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get("approximate_fisher_inTask", args.get("approximate_fisher_in_task", False)),
            )
        )
        self._in_task_fisher_interval = int(
            args.get(
                "in_task_fisher_interval",
                args.get("approximate_fisher_interval", args.get("approximte_fisher_interval", 10)),
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
                            args.get("approximate_fisher_omega", args.get("approximte_fisher_omega", 0.5)),
                        ),
                    )
                ),
            ),
        )

        self._in_task_weight_merge = _as_bool(
            args.get("in_task_weight_merge", args.get("weight_merge_with_in_task_fisher", False))
        )
        self._in_task_weight_merge_start_task = int(args.get("in_task_weight_merge_start_task", 0))
        self._in_task_weight_merge_exclude_classifier = _as_bool(
            args.get("in_task_weight_merge_exclude_classifier", True)
        )

        self._euc_trust = _as_bool(
            args.get("ewc_euclidean_trust", args.get("euclidean_trust", False))
        )
        self._euc_radius = self._parse_pos_float(
            args.get("ewc_euclidean_radius", args.get("euclidean_radius", None))
        )
        self._euc_radius_from_gam = _as_bool(args.get("ewc_euclidean_radius_from_gam", False))
        self._euc_radius_source = str(args.get("ewc_euclidean_radius_source", "gam_grad_rho")).lower()
        self._euc_radius_scale = float(args.get("ewc_euclidean_radius_scale", 1.0))
        self._euc_outside_action = str(args.get("ewc_euclidean_outside_action", "zero")).lower()
        self._euc_exclude_classifier = _as_bool(args.get("ewc_euclidean_exclude_classifier", True))
        self._euc_eps = float(args.get("ewc_euclidean_eps", 1e-12))

        self._dpen_active = _as_bool(
            args.get("ewc_delta_penalty", args.get("delta_w_penalty", False))
        )
        self._dpen_lambda = float(
            args.get("ewc_delta_penalty_lambda", args.get("delta_w_penalty_lambda", 0.0))
        )
        self._dpen_radius = self._parse_pos_float(
            args.get("ewc_delta_penalty_radius", args.get("delta_w_penalty_radius", None))
        )
        self._dpen_radius_from_gam = _as_bool(args.get("ewc_delta_penalty_radius_from_gam", False))
        self._dpen_radius_source = str(args.get("ewc_delta_penalty_radius_source", "gam_grad_rho")).lower()
        self._dpen_radius_scale = float(args.get("ewc_delta_penalty_radius_scale", 1.0))
        self._dpen_exclude_classifier = _as_bool(args.get("ewc_delta_penalty_exclude_classifier", True))
        self._dpen_eps = float(args.get("ewc_delta_penalty_eps", 1e-12))

        if self._dpen_active and self._euc_trust:
            logger.warning(
                "[EWCGAMMergedInTaskEuc] ewc_delta_penalty=True disables hard "
                "ewc_euclidean_trust."
            )
            self._euc_trust = False

        self._param_reference: Dict[int, torch.Tensor] = {}
        self._in_task_fisher_snapshots: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_weight_snapshots: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}
        self._in_task_weight_merge_last_metrics: Dict[str, float] = {}
        self._last_task_new_fisher_stats: Dict[str, float] = {}
        self._train_log_sums: Dict[str, float] = {}
        self._train_log_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}

        logger.info(
            "[EWCGAMMergedInTaskEuc] lambda=%.6g lambda_flat=%.6g gamma=%.4g "
            "normalize_fisher=%s normalize_accumulated=%s in_task=%s interval=%d "
            "omega=%.4g euc_trust=%s euc_radius=%s dpen=%s dpen_lambda=%.4g "
            "dpen_radius=%s",
            self.ewc_weight,
            self._lambda_flat,
            self.gamma,
            self._normalize_fisher,
            self._normalize_accumulated_fisher,
            self._approximate_fisher_in_task,
            self._in_task_fisher_interval,
            self._in_task_fisher_omega,
            self._euc_trust,
            self._resolve_euc_radius(),
            self._dpen_active,
            self._dpen_lambda,
            self._resolve_dpen_radius(),
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pos_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false", "off"}:
            return None
        parsed = float(value)
        return parsed if parsed > 0.0 else None

    @staticmethod
    def _is_classifier_param_name(name: str) -> bool:
        if name.startswith(("fc.", "head.", "classifier.")):
            return True
        return name.endswith(
            (
                ".fc.weight",
                ".fc.bias",
                ".head.weight",
                ".head.bias",
                ".classifier.weight",
                ".classifier.bias",
            )
        ) or ".classifiers." in name

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

    @staticmethod
    def _fisher_trace(fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        return float(sum(float(v.detach().float().sum().item()) for v in fisher.values()))

    def _trace_normalize_fisher(self, fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not fisher:
            return fisher
        denom = self._fisher_trace(fisher) + float(self._fisher_norm_eps)
        return {name: value / denom for name, value in fisher.items()}

    @staticmethod
    def _summarize_tensor_dict(tensors: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if not tensors:
            return {"tensors": 0, "numel": 0, "max": 0.0, "mean": 0.0, "sum": 0.0}

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

    def _named_trainable_params(self):
        net = self._net()
        yield from ((name, p) for name, p in net.named_parameters() if p.requires_grad)

    def _snapshot_param_reference(self, param_groups) -> None:
        name_map = {id(p): name for name, p in self._named_trainable_params()}
        refs: Dict[int, torch.Tensor] = {}
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pname = name_map.get(id(p), "")
                if self._euc_exclude_classifier and self._is_classifier_param_name(pname):
                    continue
                refs[id(p)] = p.detach().cpu().clone()
        self._param_reference = refs

    def _snapshot_model_weights(self) -> Dict[str, torch.Tensor]:
        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in self._named_trainable_params():
            if self._in_task_weight_merge_exclude_classifier and self._is_classifier_param_name(name):
                continue
            snapshot[name] = param.detach().cpu().clone()
        return snapshot

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
        return {f"train_{key}_mean": value / denom for key, value in sorted(self._train_log_sums.items())}

    # ------------------------------------------------------------------
    # Radius and L2 controls
    # ------------------------------------------------------------------

    def _resolve_euc_radius(self) -> Optional[float]:
        if self._euc_radius is not None:
            return float(self._euc_radius)
        if not self._euc_radius_from_gam:
            return None
        src = self._euc_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self.args.get("gam_grad_rho", 0.0))
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self.args.get("gam_grad_norm_rho", 0.0))
        else:
            return None
        return float(self._euc_radius_scale * rho) if rho > 0.0 else None

    def _resolve_dpen_radius(self) -> Optional[float]:
        if self._dpen_radius is not None:
            return float(self._dpen_radius)
        if not self._dpen_radius_from_gam:
            return None
        src = self._dpen_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self.args.get("gam_grad_rho", 0.0))
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self.args.get("gam_grad_norm_rho", 0.0))
        else:
            return None
        return float(self._dpen_radius_scale * rho) if rho > 0.0 else None

    def _default_dpen_stats(self) -> Dict[str, float]:
        radius = self._resolve_dpen_radius()
        return {
            "l2_penalty_enabled": float(self._dpen_active),
            "l2_penalty_configured": 0.0,
            "l2_penalty_active": 0.0,
            "l2_penalty_loss": 0.0,
            "l2_penalty_lambda": float(self._dpen_lambda),
            "l2_penalty_budget": float(radius or 0.0),
            "l2_penalty_drift_norm": 0.0,
            "l2_penalty_drift_sq": 0.0,
            "l2_penalty_ratio": 0.0,
            "l2_penalty_excess": 0.0,
            "l2_penalty_grad_norm": 0.0,
            "l2_penalty_tensors": 0.0,
        }

    def _delta_penalty_loss(self):
        stats = self._default_dpen_stats()
        zero = torch.zeros((), device=self._device)
        if not self._dpen_active or int(self._cur_task) <= 0 or not self._param_reference:
            return zero, stats

        rho = self._resolve_dpen_radius()
        lam = float(self._dpen_lambda)
        if rho is None or lam <= 0.0:
            return zero, stats

        eps = float(self._dpen_eps)
        delta_norm_sq = None
        tensor_count = 0
        for name, param in self._named_trainable_params():
            if self._dpen_exclude_classifier and self._is_classifier_param_name(name):
                continue
            ref = self._param_reference.get(id(param))
            if ref is None:
                continue
            delta = param.float() - ref.to(device=param.device, dtype=torch.float32)
            term = delta.pow(2).sum()
            delta_norm_sq = term if delta_norm_sq is None else delta_norm_sq + term
            tensor_count += 1

        if delta_norm_sq is None:
            stats["l2_penalty_configured"] = 1.0
            return zero, stats

        delta_norm = torch.sqrt(delta_norm_sq + eps)
        excess = torch.clamp(delta_norm - float(rho), min=0.0)
        penalty = lam * excess.pow(2)

        drift_norm = float(delta_norm.detach().item())
        drift_sq = float(delta_norm_sq.detach().item())
        excess_value = float(excess.detach().item())
        stats.update(
            {
                "l2_penalty_configured": 1.0,
                "l2_penalty_active": float(tensor_count > 0 and excess_value > 0.0),
                "l2_penalty_loss": float(penalty.detach().item()),
                "l2_penalty_budget": float(rho),
                "l2_penalty_drift_norm": drift_norm,
                "l2_penalty_drift_sq": drift_sq,
                "l2_penalty_ratio": float(drift_norm / (float(rho) + eps)),
                "l2_penalty_excess": excess_value,
                "l2_penalty_tensors": float(tensor_count),
            }
        )
        return penalty, stats

    def _compute_delta_penalty_grad_cache(self, params):
        with torch.enable_grad():
            penalty, stats = self._delta_penalty_loss()

        value = float(penalty.detach().item())
        if stats.get("l2_penalty_active", 0.0) <= 0.0 or not penalty.requires_grad:
            return value, {}, stats

        grads = torch.autograd.grad(penalty, params, allow_unused=True)
        grad_cache: Dict[int, torch.Tensor] = {}
        grad_norm_sq = 0.0
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_detached = grad.detach().clone()
            grad_cache[id(param)] = grad_detached
            grad_norm_sq += float(grad_detached.pow(2).sum().item())
        stats["l2_penalty_grad_norm"] = float(grad_norm_sq ** 0.5)
        return value, grad_cache, stats

    def _compute_euclidean_cumulative_scale(
        self,
        param_groups,
        param_to_v: Dict[int, torch.Tensor],
        rho: float,
    ) -> Tuple[float, Dict[str, float]]:
        eps = self._euc_eps
        delta_norm_sq = step_norm_sq = delta_dot_step = 0.0
        matched = total = 0

        for group in param_groups:
            lr = float(group.get("lr", 0.0))
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                total += 1
                ref = self._param_reference.get(id(param))
                if ref is None:
                    continue
                delta = param.detach().float() - ref.to(device=param.device, dtype=torch.float32)
                direction = param_to_v.get(id(param))
                step = (
                    float(lr) * direction.to(device=param.device, dtype=torch.float32)
                    if direction is not None
                    else torch.zeros_like(delta)
                )
                delta_norm_sq += float(delta.pow(2).sum().item())
                step_norm_sq += float(step.pow(2).sum().item())
                delta_dot_step += float((delta * step).sum().item())
                matched += 1

        A = step_norm_sq
        B = -2.0 * delta_dot_step
        C = delta_norm_sq - float(rho) ** 2
        current_feasible = C <= 0.0
        scale = 1.0
        found = False

        if A <= eps:
            found = current_feasible
            scale = 1.0 if current_feasible else 0.0
        else:
            disc = B * B - 4.0 * A * C
            if disc >= 0.0:
                sq = disc ** 0.5
                roots = [(-B - sq) / (2.0 * A + eps), (-B + sq) / (2.0 * A + eps)]
                lower = min(roots)
                upper = max(roots)
                lo = max(0.0, lower)
                hi = min(1.0, upper)
                if lo <= hi:
                    scale = max(0.0, min(1.0, hi))
                    found = True

        if not found:
            scale = 0.0 if self._euc_outside_action == "zero" else 1.0

        after_sq = max(delta_norm_sq - 2.0 * scale * delta_dot_step + scale * scale * step_norm_sq, 0.0)
        rho_eps = float(rho) + eps
        return float(scale), {
            "euc_scale": float(scale),
            "euc_before_norm": float(delta_norm_sq ** 0.5),
            "euc_after_norm": float(after_sq ** 0.5),
            "euc_budget": float(rho),
            "euc_ratio_before": float(delta_norm_sq ** 0.5 / rho_eps),
            "euc_ratio_after": float(after_sq ** 0.5 / rho_eps),
            "euc_current_feasible": float(current_feasible),
            "euc_matched": float(matched),
            "euc_total": float(total),
        }

    # ------------------------------------------------------------------
    # Gradient merge step
    # ------------------------------------------------------------------

    def _compute_clean_task_grad_cache(self, params, inputs, local_targets):
        self._network.zero_grad()
        with torch.enable_grad():
            _, ce_loss = self._compute_task_loss(inputs, local_targets)
        grads = torch.autograd.grad(ce_loss, params, allow_unused=True)
        cache = {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }
        self._network.zero_grad()
        return float(ce_loss.detach().item()), cache

    @staticmethod
    def _cosine_from_sums(dot: float, norm_a_sq: float, norm_b_sq: float, eps: float) -> float:
        return float(dot / ((norm_a_sq ** 0.5) * (norm_b_sq ** 0.5) + eps))

    def _build_merged_gradient_projector(
        self,
        clean_grad_cache,
        ewc_grad_cache,
        delta_penalty_grad_cache,
        step_stats: Dict[str, float],
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self.args.get("mechanism_eps", 1e-12))
        euc_active = self._euc_trust and bool(self._param_reference)
        euc_radius = self._resolve_euc_radius() if euc_active else None

        def projector(param_groups):
            clean_norm_sq = gam_norm_sq = flat_norm_sq = ewc_norm_sq = dpen_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = clean_flat_dot = clean_ewc_dot = clean_dpen_dot = 0.0
            flat_ewc_dot = flat_dpen_dot = ewc_dpen_dot = 0.0
            param_to_v: Dict[int, torch.Tensor] = {}

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()
                    ewc_grad = ewc_grad_cache.get(id(param))
                    dpen_grad = delta_penalty_grad_cache.get(id(param))

                    if clean_grad is None:
                        combined_grad = None if gam_grad is None else gam_grad.clone()
                    elif gam_grad is None:
                        combined_grad = clean_grad.clone()
                    else:
                        combined_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if ewc_grad is not None:
                        combined_grad = ewc_grad.clone() if combined_grad is None else combined_grad + ewc_grad
                    if dpen_grad is not None:
                        combined_grad = dpen_grad.clone() if combined_grad is None else combined_grad + dpen_grad
                    if combined_grad is None:
                        continue

                    if clean_grad is not None:
                        clean_norm_sq += float(clean_grad.pow(2).sum().item())
                    if gam_grad is not None:
                        gam_norm_sq += float(gam_grad.pow(2).sum().item())
                    if ewc_grad is not None:
                        ewc_norm_sq += float(ewc_grad.pow(2).sum().item())
                    if dpen_grad is not None:
                        dpen_norm_sq += float(dpen_grad.pow(2).sum().item())
                    combined_norm_sq += float(combined_grad.pow(2).sum().item())

                    if clean_grad is not None and gam_grad is not None:
                        flat_grad = gam_grad - clean_grad
                        flat_norm_sq += float(flat_grad.pow(2).sum().item())
                        clean_gam_dot += float((clean_grad * gam_grad).sum().item())
                        clean_flat_dot += float((clean_grad * flat_grad).sum().item())
                        if ewc_grad is not None:
                            flat_ewc_dot += float((flat_grad * ewc_grad).sum().item())
                        if dpen_grad is not None:
                            flat_dpen_dot += float((flat_grad * dpen_grad).sum().item())
                    if clean_grad is not None and ewc_grad is not None:
                        clean_ewc_dot += float((clean_grad * ewc_grad).sum().item())
                    if clean_grad is not None and dpen_grad is not None:
                        clean_dpen_dot += float((clean_grad * dpen_grad).sum().item())
                    if ewc_grad is not None and dpen_grad is not None:
                        ewc_dpen_dot += float((ewc_grad * dpen_grad).sum().item())

                    param_to_v[id(param)] = combined_grad.detach().clone()
                    if param.grad is None:
                        param.grad = combined_grad.clone()
                    else:
                        param.grad.data.copy_(combined_grad)

            euc_stats = {
                "euc_scale": 1.0,
                "euc_before_norm": 0.0,
                "euc_after_norm": 0.0,
                "euc_budget": float(euc_radius or 0.0),
                "euc_ratio_before": 0.0,
                "euc_ratio_after": 0.0,
                "euc_current_feasible": 1.0,
                "euc_matched": 0.0,
                "euc_total": 0.0,
            }
            if euc_active and euc_radius is not None:
                scale, euc_stats = self._compute_euclidean_cumulative_scale(
                    param_groups, param_to_v, float(euc_radius)
                )
                if scale != 1.0:
                    for group in param_groups:
                        for param in group["params"]:
                            if param.requires_grad and param.grad is not None:
                                param.grad.data.mul_(scale)

            clean_norm = clean_norm_sq ** 0.5
            gam_norm = gam_norm_sq ** 0.5
            flat_norm = flat_norm_sq ** 0.5
            ewc_norm = ewc_norm_sq ** 0.5
            dpen_norm = dpen_norm_sq ** 0.5
            combined_norm = combined_norm_sq ** 0.5
            step_stats.update(
                {
                    "grad_clean_norm": float(clean_norm),
                    "grad_gam_norm": float(gam_norm),
                    "grad_flat_component_norm": float(flat_norm),
                    "grad_fisher_norm": float(ewc_norm),
                    "grad_l2_penalty_norm": float(dpen_norm),
                    "grad_combined_norm": float(combined_norm),
                    "grad_flat_to_clean_norm_ratio": float(flat_norm / (clean_norm + eps)),
                    "grad_fisher_to_clean_norm_ratio": float(ewc_norm / (clean_norm + eps)),
                    "grad_l2_penalty_to_clean_norm_ratio": float(dpen_norm / (clean_norm + eps)),
                    "grad_cos_clean_gam": self._cosine_from_sums(clean_gam_dot, clean_norm_sq, gam_norm_sq, eps),
                    "grad_cos_clean_flat": self._cosine_from_sums(clean_flat_dot, clean_norm_sq, flat_norm_sq, eps),
                    "grad_cos_clean_fisher": self._cosine_from_sums(clean_ewc_dot, clean_norm_sq, ewc_norm_sq, eps),
                    "grad_cos_clean_l2_penalty": self._cosine_from_sums(clean_dpen_dot, clean_norm_sq, dpen_norm_sq, eps),
                    "grad_cos_flat_fisher": self._cosine_from_sums(flat_ewc_dot, flat_norm_sq, ewc_norm_sq, eps),
                    "grad_cos_flat_l2_penalty": self._cosine_from_sums(flat_dpen_dot, flat_norm_sq, dpen_norm_sq, eps),
                    "grad_cos_fisher_l2_penalty": self._cosine_from_sums(ewc_dpen_dot, ewc_norm_sq, dpen_norm_sq, eps),
                    **euc_stats,
                }
            )

        return projector

    def _step_projected(self, optimizer, inputs, local_targets):
        params = _flatten_optimizer_params(optimizer)
        clean_loss_value, clean_grad_cache = self._compute_clean_task_grad_cache(
            params, inputs, local_targets
        )
        ewc_loss_value, ewc_grad_cache = self._compute_ewc_grad_cache_with_value(params)
        dpen_value, dpen_grad_cache, dpen_stats = self._compute_delta_penalty_grad_cache(params)

        log_cache: Dict[str, float] = {
            "ce_clean": float(clean_loss_value),
            "ce": float(clean_loss_value),
            "gam": float(clean_loss_value),
            "ewc": float(ewc_loss_value),
            "l2_penalty": float(dpen_value),
            "lambda_ewc": float(self.ewc_weight),
            "lambda_flat": float(self._lambda_flat),
        }
        log_cache.update(dpen_stats)

        step_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_merged_gradient_projector(
                clean_grad_cache, ewc_grad_cache, dpen_grad_cache, step_stats
            )
        )

        call_count = [0]

        def closure():
            call_count[0] += 1
            optimizer.zero_grad()
            logits, task_loss = self._compute_task_loss(inputs, local_targets)
            if call_count[0] == 1:
                log_cache["ce_clean"] = float(task_loss.detach().item())
                log_cache["ce"] = float(task_loss.detach().item())
            else:
                log_cache["gam"] = float(task_loss.detach().item())
            task_loss.backward()
            return logits, task_loss.detach()

        try:
            logits, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ce_value = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        gam_value = float(log_cache.get("gam", ce_value))
        log_cache.update(step_stats)
        log_cache["ewc_gam_ratio"] = float(ewc_loss_value / (gam_value + float(self.args.get("mechanism_eps", 1e-12))))
        log_cache["l2_penalty_gam_ratio"] = float(dpen_value / (gam_value + float(self.args.get("mechanism_eps", 1e-12))))

        total_loss_value = ce_value + float(ewc_loss_value) + float(dpen_value)
        return logits.detach(), float(total_loss_value), log_cache

    def _compute_ewc_grad_cache_with_value(self, params):
        if self.count_updates == 0 or not self.omega:
            return 0.0, {}

        with torch.enable_grad():
            ewc_loss = self._compute_ewc_loss()

        value = float(ewc_loss.detach().item())
        if not ewc_loss.requires_grad:
            return value, {}
        grads = torch.autograd.grad(ewc_loss, params, allow_unused=True, retain_graph=False)
        return value, {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }

    # ------------------------------------------------------------------
    # Training loop with in-task snapshots
    # ------------------------------------------------------------------

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task > 0 and (self._euc_trust or self._dpen_active):
            self._snapshot_param_reference(optimizer.param_groups)
        else:
            self._param_reference = {}

        self._in_task_fisher_snapshots = []
        self._in_task_weight_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._in_task_weight_merge_last_metrics = {}
        self._last_task_new_fisher_stats = {}
        self._reset_train_log_stats()
        self._log_task_start_state()

        prog_bar = tqdm(range(self._run_epochs))
        info = ""
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            ce_losses = 0.0
            gam_losses = 0.0
            ewc_losses = 0.0
            l2_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                local_targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits, loss_value, log_dict = self._step_projected(optimizer, inputs, local_targets)
                losses += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", log_dict.get("ce", 0.0)))
                gam_losses += float(log_dict.get("gam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                l2_losses += float(log_dict.get("l2_penalty", 0.0))
                loss_count += 1
                self._record_train_log_stats(log_dict)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                total += len(local_targets)

            if scheduler is not None:
                scheduler.step()

            if self._should_capture_in_task_fisher(epoch + 1):
                self._capture_in_task_fisher_snapshot(
                    train_loader=train_loader,
                    model=self._net(),
                    optimizer=optimizer,
                    epoch_idx=epoch + 1,
                )

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            denom = max(1, loss_count)
            avg_loss = losses / denom
            avg_ce = ce_losses / denom
            avg_gam = gam_losses / denom
            avg_ewc = ewc_losses / denom
            avg_l2 = l2_losses / denom
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, "
                "CE {:.3f}, GAM {:.3f}, EWC {:.3e}, L2 {:.3e}, EWC/GAM {:.3e}"
            ).format(
                self._cur_task,
                epoch + 1,
                self._run_epochs,
                avg_loss,
                train_acc,
                avg_ce,
                avg_gam,
                avg_ewc,
                avg_l2,
                avg_ewc / (avg_gam + float(self.args.get("mechanism_eps", 1e-12))),
            )
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ", Test_accy {:.2f}".format(test_acc)
            prog_bar.set_description(info)

        logging.info(info)
        self._apply_in_task_weight_merge(self._net())

    # ------------------------------------------------------------------
    # In-task Fisher and accumulation
    # ------------------------------------------------------------------

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        return (
            self._approximate_fisher_in_task
            and self._in_task_fisher_interval > 0
            and self._in_task_fisher_omega > 0.0
            and int(epoch_idx) % int(self._in_task_fisher_interval) == 0
            and self.fisher_mode in _BATCH_FISHER_MODES
        )

    def _compute_endpoint_fisher(self, net) -> Dict[str, torch.Tensor]:
        task_start, task_end = self._task_range()
        if self.fisher_mode in _BATCH_FISHER_MODES:
            return super()._compute_fisher(net, task_start, task_end, mode=self.fisher_mode)
        if self.fisher_mode == "prototype":
            self._build_prototype_set(net, task_start, task_end)
            return self._compute_fisher_from_loader(
                net,
                self._make_prototype_loader(range(task_start, task_end)),
                all_classes=self._total_classes,
            )
        if self.fisher_mode == "prototype_all":
            self._build_prototype_set(net, task_start, task_end)
            return self._compute_fisher_from_loader(
                net,
                self._make_prototype_loader(range(0, self._total_classes)),
                all_classes=self._total_classes,
            )
        return {}

    def _capture_in_task_fisher_snapshot(self, train_loader, model, optimizer, epoch_idx: int) -> int:
        was_training = bool(model.training)
        logger.info(
            "[EWCGAMMergedInTaskEuc] Computing in-task Fisher snapshot task=%d epoch=%d",
            self._cur_task,
            epoch_idx,
        )
        try:
            task_start, task_end = self._task_range()
            fisher = super()._compute_fisher(model, task_start, task_end, mode=self.fisher_mode)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        snapshot = {name: value.detach().cpu().clone() for name, value in fisher.items()}
        self._in_task_fisher_snapshots.append((int(epoch_idx), snapshot))
        if self._in_task_weight_merge:
            weights = self._snapshot_model_weights()
            if weights:
                self._in_task_weight_snapshots.append((int(epoch_idx), weights))

        logger.info(
            "[EWCGAMMergedInTaskEuc] Saved in-task Fisher snapshot task=%d epoch=%d "
            "count=%d trace=%.6g tensors=%d",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(snapshot),
            len(snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    def _mean_in_task_raw_fisher(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None
        accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _, fisher_snapshot in self._in_task_fisher_snapshots:
            for name, value in fisher_snapshot.items():
                value_cpu = value.detach().cpu()
                if name not in accum:
                    accum[name] = value_cpu.clone()
                    counts[name] = 1
                else:
                    accum[name] = accum[name] + value_cpu
                    counts[name] += 1
        return {name: value / float(max(counts.get(name, 1), 1)) for name, value in accum.items()}

    def _mix_in_task_fisher(self, final_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        path_raw = self._mean_in_task_raw_fisher()
        final_trace = self._fisher_trace(final_raw)
        path_trace = self._fisher_trace(path_raw)
        snapshot_epochs = [epoch for epoch, _ in self._in_task_fisher_snapshots]

        use_path = (
            self._approximate_fisher_in_task
            and self._in_task_fisher_omega > 0.0
            and path_raw is not None
        )
        if not use_path:
            self._in_task_fisher_last_metrics = {
                "in_task_fisher_used_path": 0.0,
                "in_task_fisher_interval": float(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(self._in_task_fisher_omega),
                "in_task_fisher_snapshots": float(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
                "in_task_fisher_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_fisher_final_raw_trace": float(final_trace),
                "in_task_fisher_path_raw_trace": float(path_trace),
                "in_task_fisher_mixed_raw_trace": float(final_trace),
            }
            return self._trace_normalize_fisher(final_raw) if self._normalize_fisher and final_raw else final_raw

        omega = float(self._in_task_fisher_omega)
        if self._normalize_fisher:
            final_src = self._trace_normalize_fisher(final_raw)
            path_src = self._trace_normalize_fisher(path_raw)
        else:
            final_src = final_raw
            path_src = path_raw

        mixed: Dict[str, torch.Tensor] = {}
        for name in set(final_src.keys()) | set(path_src.keys()):
            final_value = final_src.get(name)
            path_value = path_src.get(name)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[name] = path_value.detach().cpu().clone()
            elif path_value is None:
                mixed[name] = final_value.detach().cpu().clone()
            else:
                final_t = final_value.detach().cpu()
                path_t = self._match_shape(path_value, final_t)
                mixed[name] = (1.0 - omega) * final_t + omega * path_t

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "in_task_fisher_used_path": 1.0,
            "in_task_fisher_interval": float(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": float(len(snapshot_epochs)),
            "in_task_fisher_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
            "in_task_fisher_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_fisher_final_raw_trace": float(final_trace),
            "in_task_fisher_path_raw_trace": float(path_trace),
            "in_task_fisher_mixed_raw_trace": float(mixed_trace),
        }
        logger.info(
            "[EWCGAMMergedInTaskEuc] Mixed endpoint/path Fisher task=%d snapshots=%d "
            "omega=%.4g final_trace=%.6g path_trace=%.6g mixed_trace=%.6g "
            "normalize_task=%s",
            self._cur_task,
            len(snapshot_epochs),
            omega,
            final_trace,
            path_trace,
            mixed_trace,
            self._normalize_fisher,
        )
        return mixed

    def _accumulate_fisher(self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not old:
            self._fisher_past_trace_mass = max(self._fisher_trace(new), 0.0)
            return self._trace_normalize_fisher(new) if self._normalize_accumulated_fisher else {
                name: value.detach().cpu().clone() for name, value in new.items()
            }

        if not self._normalize_accumulated_fisher:
            combined: Dict[str, torch.Tensor] = {}
            for name in set(old.keys()) | set(new.keys()):
                if name in old and name in new:
                    new_t = self._match_shape(new[name], old[name])
                    combined[name] = old[name] * float(self.gamma) + new_t
                elif name in old:
                    combined[name] = old[name] * float(self.gamma)
                else:
                    combined[name] = new[name].detach().cpu().clone()
            self._fisher_past_trace_mass = self._fisher_trace(combined)
            return combined

        old_trace = self._fisher_trace(old)
        old_mass = float(self._fisher_past_trace_mass) if self._fisher_past_trace_mass > 0.0 else old_trace
        old_scale = old_mass / (old_trace + float(self._fisher_norm_eps))
        new_trace = self._fisher_trace(new)

        combined: Dict[str, torch.Tensor] = {}
        for name in set(old.keys()) | set(new.keys()):
            if name in old and name in new:
                old_raw = old[name] * float(old_scale)
                new_t = self._match_shape(new[name], old_raw)
                combined[name] = old_raw * float(self.gamma) + new_t
            elif name in old:
                combined[name] = old[name] * float(old_scale) * float(self.gamma)
            else:
                combined[name] = new[name].detach().cpu().clone()

        self._fisher_past_trace_mass = float(self.gamma) * old_mass + new_trace
        normalized = self._trace_normalize_fisher(combined)
        logger.info(
            "[EWCGAMMergedInTaskEuc] Normalized accumulated Fisher task=%d "
            "old_mass=%.6g new_trace=%.6g mass=%.6g trace=%.6g",
            self._cur_task,
            old_mass,
            new_trace,
            self._fisher_past_trace_mass,
            self._fisher_trace(normalized),
        )
        return normalized

    def _mean_in_task_weight_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_weight_snapshots:
            return None
        accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _, weight_snapshot in self._in_task_weight_snapshots:
            for name, value in weight_snapshot.items():
                value_cpu = value.detach().cpu()
                if name not in accum:
                    accum[name] = value_cpu.clone()
                    counts[name] = 1
                else:
                    accum[name] = accum[name] + value_cpu
                    counts[name] += 1
        return {name: value / float(max(counts.get(name, 1), 1)) for name, value in accum.items()}

    def _apply_in_task_weight_merge(self, model) -> Dict[str, float]:
        snapshot_epochs = [epoch for epoch, _ in self._in_task_weight_snapshots]
        omega = float(self._in_task_fisher_omega)
        metrics: Dict[str, float] = {
            "in_task_weight_merge_enabled": float(bool(self._in_task_weight_merge)),
            "in_task_weight_merge_applied": 0.0,
            "in_task_weight_merge_omega": float(omega),
            "in_task_weight_merge_snapshots": float(len(snapshot_epochs)),
            "in_task_weight_merge_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
            "in_task_weight_merge_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
        }
        if not self._in_task_weight_merge or int(self._cur_task) < int(self._in_task_weight_merge_start_task) or omega <= 0.0:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics

        path_mean = self._mean_in_task_weight_snapshot()
        if not path_mean:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics

        final_norm_sq = path_norm_sq = merged_norm_sq = shift_norm_sq = 0.0
        tensor_count = numel = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if self._in_task_weight_merge_exclude_classifier and self._is_classifier_param_name(name):
                    continue
                path_value = path_mean.get(name)
                if path_value is None:
                    continue
                final_cpu = param.detach().cpu()
                path_t = self._match_shape(path_value, final_cpu)
                merged = final_cpu * (1.0 - omega) + path_t * omega
                final_f = final_cpu.float()
                path_f = path_t.float()
                merged_f = merged.float()
                final_norm_sq += float(final_f.pow(2).sum().item())
                path_norm_sq += float(path_f.pow(2).sum().item())
                merged_norm_sq += float(merged_f.pow(2).sum().item())
                shift_norm_sq += float((merged_f - final_f).pow(2).sum().item())
                tensor_count += 1
                numel += int(final_cpu.numel())
                param.data.copy_(merged.to(device=param.device, dtype=param.dtype))

        metrics.update(
            {
                "in_task_weight_merge_applied": float(tensor_count > 0),
                "in_task_weight_merge_tensors": float(tensor_count),
                "in_task_weight_merge_numel": float(numel),
                "in_task_weight_final_l2": float(final_norm_sq ** 0.5),
                "in_task_weight_path_l2": float(path_norm_sq ** 0.5),
                "in_task_weight_merged_l2": float(merged_norm_sq ** 0.5),
                "in_task_weight_merge_shift_l2": float(shift_norm_sq ** 0.5),
            }
        )
        self._in_task_weight_merge_last_metrics = metrics
        logger.info("[EWCGAMMergedInTaskEuc][WeightMerge] task=%d %s", self._cur_task, metrics)
        return metrics

    def _log_task_start_state(self) -> None:
        fisher_stats = self._summarize_tensor_dict(self.omega)
        ref_count = len(self.theta_star) if self.theta_star else 0
        logger.info(
            "[EWCGAMMergedInTaskEuc][TaskStart] task=%d lambda=%.3e lambda_flat=%.3g "
            "has_fisher=%s F_tensors=%d F_numel=%d F_max=%.3e F_mean=%.3e F_sum=%.3e "
            "has_ref=%s ref_tensors=%d euc_trust=%s euc_radius=%s dpen=%s dpen_radius=%s",
            self._cur_task,
            float(self.ewc_weight),
            float(self._lambda_flat),
            bool(self.omega),
            int(fisher_stats["tensors"]),
            int(fisher_stats["numel"]),
            fisher_stats["max"],
            fisher_stats["mean"],
            fisher_stats["sum"],
            bool(self.theta_star),
            ref_count,
            self._euc_trust,
            self._resolve_euc_radius(),
            self._dpen_active,
            self._resolve_dpen_radius(),
        )

    def after_task(self):
        print("=== Computing Fisher Information ===")
        net = self._net()
        final_raw = self._compute_endpoint_fisher(net)
        fisher = self._mix_in_task_fisher(final_raw)
        self._last_task_new_fisher_stats = self._summarize_tensor_dict(fisher)

        if self.fisher_mode == "prototype_all":
            self.omega = {name: value.detach().cpu().clone() for name, value in fisher.items()}
            self._fisher_past_trace_mass = self._fisher_trace(self.omega)
        else:
            self.omega = self._accumulate_fisher(self.omega, fisher)

        self.count_updates += 1
        self.theta_star = {
            name: param.detach().clone()
            for name, param in net.named_parameters()
            if param.requires_grad and not name.startswith("fc.")
        }
        self._known_classes = self._total_classes

        new_stats = dict(self._last_task_new_fisher_stats)
        accum_stats = self._summarize_tensor_dict(self.omega)
        logger.info(
            "[EWCGAMMergedInTaskEuc][FisherStats] task=%d new_tensors=%d new_numel=%d "
            "new_max=%.3e new_mean=%.3e new_sum=%.3e accum_tensors=%d accum_numel=%d "
            "accum_max=%.3e accum_mean=%.3e accum_sum=%.3e trace_mass=%.3e",
            self._cur_task,
            int(new_stats["tensors"]),
            int(new_stats["numel"]),
            new_stats["max"],
            new_stats["mean"],
            new_stats["sum"],
            int(accum_stats["tensors"]),
            int(accum_stats["numel"]),
            accum_stats["max"],
            accum_stats["mean"],
            accum_stats["sum"],
            float(self._fisher_past_trace_mass),
        )

        if self._compute_full_fisher:
            full_diff = self._compute_full_fisher_diff_metrics(net)
            if full_diff:
                full_diff["task"] = int(self._cur_task)
                full_diff["method_type"] = "ewc_gam_merged_intask_euc"
                self._last_full_fisher_diff = full_diff
                logging.info("[EWCGAMMergedInTaskEuc][FullFisherDiff] task=%d %s", self._cur_task, full_diff)

        self._last_mechanism_metrics = {
            "task": int(self._cur_task),
            "method_type": "ewc_gam_merged_intask_euc",
            "lambda_ewc": float(self.ewc_weight),
            "lambda_flat": float(self._lambda_flat),
            "ewc_gamma": float(self.gamma),
            "ewc_normalize_fisher": bool(self._normalize_fisher),
            "ewc_normalize_accumulated_fisher": bool(self._normalize_accumulated_fisher),
            "fisher_past_trace_mass": float(self._fisher_past_trace_mass),
            "approximate_fisher_in_task": bool(self._approximate_fisher_in_task),
            "euc_trust": bool(self._euc_trust),
            "euc_radius": float(self._resolve_euc_radius() or 0.0),
            "l2_penalty": bool(self._dpen_active),
            "l2_penalty_lambda": float(self._dpen_lambda),
            "l2_penalty_radius": float(self._resolve_dpen_radius() or 0.0),
            **self._in_task_fisher_last_metrics,
            **self._in_task_weight_merge_last_metrics,
            **self._summarize_train_log_stats(),
        }
        logging.info("[EWCGAMMergedInTaskEuc][Mechanism] %s", self._last_mechanism_metrics)

    def get_mechanism_metrics(self) -> dict:
        metrics = dict(self._last_full_fisher_diff)
        metrics.update(self._last_mechanism_metrics)
        return metrics


Learner = EWCGAMMergedInTaskEuc
