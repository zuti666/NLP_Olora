"""AS1 in-task Fisher with optional adaptive lambda and EWC projection.

This variant builds on ``as1_normfisher_gam_intask``:

- task Fisher can mix endpoint/path estimates and normalize them before mixing;
- accumulated cross-task Fisher can be trace-normalized with preserved gamma mass;
- the AS1 task direction can be projected away from the EWC restoring direction;
- lambda can optionally use multiplicative dual ascent after each task.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import torch

from models_LoRAbasedCL.as1_normfisher_gam_intask import (
    Learner as _AS1InTaskLearner,
)
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_AS1InTaskLearner):
    """AS1 in-task Fisher + projected EWC gradient composition."""

    def __init__(self, args):
        super().__init__(args)

        self._ewc_project_grad = as_bool(args.get("ewc_project_grad", True))
        self._ewc_project_eps = float(
            args.get("ewc_project_eps", args.get("fisher_project_eps", 1e-12))
        )
        self._dual_rho = float(args.get("ewc_dual_rho", args.get("dual_rho", 0.5)))
        self._ewc_project_rho = float(
            args.get("ewc_project_rho", args.get("fisher_project_rho", self._dual_rho))
        )
        self._ewc_project_boundary = as_bool(args.get("ewc_project_boundary", True))
        self._ewc_project_outward_only = as_bool(args.get("ewc_project_outward_only", True))
        self._ewc_project_drift_key = str(
            args.get("ewc_project_drift_key", "fisher_weighted_drift")
        )
        self._ewc_project_budget = float(
            args.get("ewc_project_budget", args.get("ewc_dual_delta", self._dual_delta))
        )

        self._dual_multiplicative = as_bool(
            args.get("ewc_dual_multiplicative", args.get("ewc_lambda_multiplicative", True))
        )
        self._dual_eta_up = float(args.get("ewc_dual_eta_up", 0.5))
        self._dual_eta_down = float(args.get("ewc_dual_eta_down", 0.05))
        self._dual_fisher_key = str(args.get("ewc_dual_fisher_key", "fisher_weighted_drift"))
        self._dual_euclidean_delta = float(
            args.get("ewc_dual_euclidean_delta", args.get("ewc_dual_r_sq", 0.0))
        )
        self._dual_spectral_delta = float(
            args.get("ewc_dual_spectral_delta", args.get("ewc_dual_fmax_l2_delta", 0.0))
        )
        self._last_dual_stats: Dict[str, float] = {}

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask-LambdaProject] project=%s rho=%.4g "
                "boundary=%s outward_only=%s drift_key=%s budget=%.4g "
                "dual=%s dual_multiplicative=%s dual_rho=%.4g "
                "dual_eta_up=%.4g dual_eta_down=%.4g "
                "lambda_bounds=[%.4g, %.4g] fisher_key=%s "
                "euclidean_budget=%.4g spectral_budget=%.4g",
                self._ewc_project_grad,
                self._ewc_project_rho,
                self._ewc_project_boundary,
                self._ewc_project_outward_only,
                self._ewc_project_drift_key,
                self._ewc_project_budget,
                self._dual_ascent,
                self._dual_multiplicative,
                self._dual_rho,
                self._dual_eta_up,
                self._dual_eta_down,
                float(self._dual_lambda_min),
                float(self._dual_lambda_max),
                self._dual_fisher_key,
                self._dual_euclidean_delta,
                self._dual_spectral_delta,
            )

    @staticmethod
    def _zeros_like_optional(reference_a, reference_b):
        ref = reference_a if reference_a is not None else reference_b
        return torch.zeros_like(ref)

    def _projection_drift_value(self) -> float:
        if not self._fisher_past_delta or not self._delta_reference:
            return 0.0
        metrics = self._compute_delta_drift_metrics()
        return float(metrics.get(self._ewc_project_drift_key, 0.0))

    def _build_direction_projector(
        self,
        clean_grad_cache: Dict[int, torch.Tensor],
        fisher_grad_cache: Dict[int, torch.Tensor],
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self._ewc_project_eps)
        drift_value = self._projection_drift_value()
        threshold = float(self._ewc_project_rho * self._ewc_project_budget)

        def projector(param_groups):
            clean_norm_sq = 0.0
            gam_norm_sq = 0.0
            flat_norm_sq = 0.0
            task_norm_sq = 0.0
            fisher_norm_sq = 0.0
            projected_task_norm_sq = 0.0
            removed_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = 0.0
            clean_flat_dot = 0.0
            clean_fisher_dot = 0.0
            flat_fisher_dot = 0.0
            task_fisher_dot = 0.0
            param_count = 0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    fisher_grad = fisher_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()

                    if clean_grad is None and gam_grad is None:
                        task_grad = None
                    elif clean_grad is None:
                        task_grad = gam_grad.clone()
                    elif gam_grad is None:
                        task_grad = clean_grad.clone()
                    else:
                        task_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if task_grad is None and fisher_grad is None:
                        continue

                    param_count += 1
                    if task_grad is not None:
                        task_norm_sq += float(task_grad.detach().float().pow(2).sum().item())
                    if fisher_grad is not None:
                        fisher_norm_sq += float(fisher_grad.detach().float().pow(2).sum().item())
                    if task_grad is not None and fisher_grad is not None:
                        task_fisher_dot += float(
                            (task_grad.detach().float() * fisher_grad.detach().float()).sum().item()
                        )

                    if self._mechanism_grad_stats:
                        if clean_grad is not None:
                            clean_norm_sq += float(clean_grad.pow(2).sum().item())
                        if gam_grad is not None:
                            gam_norm_sq += float(gam_grad.pow(2).sum().item())
                        if clean_grad is not None and gam_grad is not None:
                            flat_grad = gam_grad - clean_grad
                            flat_norm_sq += float(flat_grad.pow(2).sum().item())
                            clean_gam_dot += float((clean_grad * gam_grad).sum().item())
                            clean_flat_dot += float((clean_grad * flat_grad).sum().item())
                            if fisher_grad is not None:
                                flat_fisher_dot += float((flat_grad * fisher_grad).sum().item())
                        if clean_grad is not None and fisher_grad is not None:
                            clean_fisher_dot += float((clean_grad * fisher_grad).sum().item())

            boundary_active = (not self._ewc_project_boundary) or (drift_value > threshold)
            outward = task_fisher_dot < 0.0
            apply_projection = (
                self._ewc_project_grad
                and boundary_active
                and fisher_norm_sq > eps
                and ((not self._ewc_project_outward_only) or outward)
            )
            coeff = float(task_fisher_dot / (fisher_norm_sq + eps)) if apply_projection else 0.0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    fisher_grad = fisher_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()

                    if clean_grad is None and gam_grad is None:
                        task_grad = None
                    elif clean_grad is None:
                        task_grad = gam_grad.to(device=param.device, dtype=param.dtype)
                    elif gam_grad is None:
                        task_grad = clean_grad.to(device=param.device, dtype=param.dtype)
                    else:
                        clean_term = clean_grad.to(device=param.device, dtype=param.dtype)
                        gam_term = gam_grad.to(device=param.device, dtype=param.dtype)
                        task_grad = clean_term + lambda_flat * (gam_term - clean_term)

                    if fisher_grad is None:
                        if task_grad is None:
                            continue
                        fisher_term = self._zeros_like_optional(task_grad, None)
                    else:
                        fisher_term = fisher_grad.to(device=param.device, dtype=param.dtype)

                    if task_grad is None:
                        task_grad = self._zeros_like_optional(fisher_term, None)

                    projected_task = task_grad
                    removed = self._zeros_like_optional(projected_task, fisher_term)
                    if apply_projection and fisher_grad is not None:
                        removed = coeff * fisher_term
                        projected_task = task_grad - removed

                    combined_grad = projected_task + fisher_term

                    if self._mechanism_grad_stats:
                        projected_task_norm_sq += float(
                            projected_task.detach().float().pow(2).sum().item()
                        )
                        removed_norm_sq += float(removed.detach().float().pow(2).sum().item())
                        combined_norm_sq += float(
                            combined_grad.detach().float().pow(2).sum().item()
                        )

                    if param.grad is None:
                        param.grad = combined_grad.detach().clone()
                    else:
                        param.grad.data.copy_(combined_grad)

            if self._mechanism_grad_stats:
                clean_norm = clean_norm_sq ** 0.5
                gam_norm = gam_norm_sq ** 0.5
                flat_norm = flat_norm_sq ** 0.5
                task_norm = task_norm_sq ** 0.5
                fisher_norm = fisher_norm_sq ** 0.5
                projected_task_norm = projected_task_norm_sq ** 0.5
                removed_norm = removed_norm_sq ** 0.5
                combined_norm = combined_norm_sq ** 0.5
                self._accumulate_mechanism_grad_stats(
                    {
                        "clean_norm": clean_norm,
                        "gam_norm": gam_norm,
                        "flat_component_norm": flat_norm,
                        "task_direction_norm": task_norm,
                        "fisher_norm": fisher_norm,
                        "projected_task_norm": projected_task_norm,
                        "projection_removed_norm": removed_norm,
                        "projection_removal_ratio": removed_norm / (task_norm + eps),
                        "combined_norm": combined_norm,
                        "flat_to_clean_norm_ratio": flat_norm / (clean_norm + eps),
                        "fisher_to_clean_norm_ratio": fisher_norm / (clean_norm + eps),
                        "cos_clean_gam": self._cosine_from_sums(
                            clean_gam_dot, clean_norm_sq, gam_norm_sq, eps
                        ),
                        "cos_clean_flat": self._cosine_from_sums(
                            clean_flat_dot, clean_norm_sq, flat_norm_sq, eps
                        ),
                        "cos_clean_fisher": self._cosine_from_sums(
                            clean_fisher_dot, clean_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_flat_fisher": self._cosine_from_sums(
                            flat_fisher_dot, flat_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_task_fisher": self._cosine_from_sums(
                            task_fisher_dot, task_norm_sq, fisher_norm_sq, eps
                        ),
                        "project_enabled": float(self._ewc_project_grad),
                        "project_applied": float(apply_projection),
                        "project_boundary_active": float(boundary_active),
                        "project_outward": float(outward),
                        "project_threshold": float(threshold),
                        "project_drift": float(drift_value),
                        "project_coeff": float(coeff),
                        "project_param_count": float(param_count),
                    }
                )

        return projector

    def _dual_constraint_ratios(self, drift_metrics: Dict[str, float]) -> Dict[str, float]:
        eps = float(self._mechanism_eps)
        fisher_budget = float(self._dual_delta)
        fisher_value = float(drift_metrics.get(self._dual_fisher_key, 0.0))
        fisher_ratio = fisher_value / (fisher_budget + eps) if fisher_budget > 0.0 else 0.0

        euclidean_value = float(drift_metrics.get("delta_norm", 0.0))
        euclidean_ratio = (
            euclidean_value / (float(self._dual_euclidean_delta) + eps)
            if self._dual_euclidean_delta > 0.0
            else 0.0
        )

        fmax = float(drift_metrics.get("fisher_max", 0.0))
        spectral_value = fmax * euclidean_value
        spectral_ratio = (
            spectral_value / (float(self._dual_spectral_delta) + eps)
            if self._dual_spectral_delta > 0.0
            else 0.0
        )
        return {
            "dual_fisher_value": fisher_value,
            "dual_fisher_budget": fisher_budget,
            "dual_fisher_ratio": fisher_ratio,
            "dual_euclidean_value": euclidean_value,
            "dual_euclidean_budget": float(self._dual_euclidean_delta),
            "dual_euclidean_ratio": euclidean_ratio,
            "dual_spectral_value": spectral_value,
            "dual_spectral_budget": float(self._dual_spectral_delta),
            "dual_spectral_ratio": spectral_ratio,
            "dual_max_ratio": max(fisher_ratio, euclidean_ratio, spectral_ratio),
        }

    def _update_lambda_from_drift(self, drift_metrics: Dict[str, float]) -> Dict[str, float]:
        ratios = self._dual_constraint_ratios(drift_metrics)
        max_ratio = float(ratios["dual_max_ratio"])
        old_lambda = float(self._lambda_ewc)
        mode = 0.0

        if self._dual_multiplicative:
            new_lambda = old_lambda
            if max_ratio > 1.0:
                new_lambda = old_lambda * math.exp(float(self._dual_eta_up) * (max_ratio - 1.0))
                mode = 1.0
            elif max_ratio < float(self._dual_rho):
                new_lambda = old_lambda * math.exp(
                    -float(self._dual_eta_down) * (1.0 - max_ratio)
                )
                mode = -1.0
        else:
            fisher_value = float(ratios["dual_fisher_value"])
            violation = fisher_value - float(self._dual_delta)
            new_lambda = old_lambda + float(self._dual_alpha) * violation
            mode = 1.0 if violation > 0.0 else (-1.0 if violation < 0.0 else 0.0)

        new_lambda = min(
            float(self._dual_lambda_max),
            max(float(self._dual_lambda_min), float(new_lambda)),
        )
        self._lambda_ewc = new_lambda
        self._ewc_lambda = new_lambda

        stats = {
            **ratios,
            "dual_mode": mode,
            "dual_lambda_before": old_lambda,
            "dual_lambda_after": new_lambda,
        }
        self._last_dual_stats = stats
        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-InTask-LambdaProject] Dual update task=%d "
                "fisher=%.4g/%.4g ratio=%.4g "
                "euclidean=%.4g/%.4g ratio=%.4g "
                "spectral=%.4g/%.4g ratio=%.4g "
                "max_ratio=%.4g mode=%.0f lambda %.6g -> %.6g",
                self._cur_task,
                ratios["dual_fisher_value"],
                ratios["dual_fisher_budget"],
                ratios["dual_fisher_ratio"],
                ratios["dual_euclidean_value"],
                ratios["dual_euclidean_budget"],
                ratios["dual_euclidean_ratio"],
                ratios["dual_spectral_value"],
                ratios["dual_spectral_budget"],
                ratios["dual_spectral_ratio"],
                max_ratio,
                mode,
                old_lambda,
                new_lambda,
            )
        return stats

    def after_task(self):
        if self._mechanism_eval or self._dual_ascent:
            drift_before = self._compute_delta_drift_metrics()
        else:
            drift_before = {}

        dual_enabled = bool(self._dual_ascent)
        self._dual_ascent = False
        try:
            super().after_task()
        finally:
            self._dual_ascent = dual_enabled

        dual_stats: Dict[str, float] = {}
        if dual_enabled and self._cur_task > 0:
            dual_stats = self._update_lambda_from_drift(drift_before)

        if self._mechanism_eval and self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_gam_intask_lambda_project",
                    "ewc_project_grad": bool(self._ewc_project_grad),
                    "ewc_project_eps": float(self._ewc_project_eps),
                    "ewc_project_rho": float(self._ewc_project_rho),
                    "ewc_project_boundary": bool(self._ewc_project_boundary),
                    "ewc_project_outward_only": bool(self._ewc_project_outward_only),
                    "ewc_project_drift_key": self._ewc_project_drift_key,
                    "ewc_project_budget": float(self._ewc_project_budget),
                    "ewc_dual_multiplicative": bool(self._dual_multiplicative),
                    "ewc_dual_rho": float(self._dual_rho),
                    **dual_stats,
                }
            )
            if self._is_main_process:
                self._log(
                    "[AS1-NormFisher-InTask-LambdaProject][ProjectDual] "
                    f"task={self._cur_task} project={self._ewc_project_grad} "
                    f"boundary={self._ewc_project_boundary} "
                    f"outward_only={self._ewc_project_outward_only} "
                    f"drift_key={self._ewc_project_drift_key} "
                    f"budget={self._ewc_project_budget:.6g} "
                    f"dual_enabled={dual_enabled} "
                    f"lambda={float(self._lambda_ewc):.6g}"
                )
                self._log(
                    "[AS1-NormFisher-InTask-LambdaProject][Mechanism] "
                    f"{self._last_mechanism_metrics}"
                )


EWCLoRA = Learner
