"""SeqLoRA EWCLoRA with GAM, adaptive lambda, and projected EWC gradient.

This variant reuses the GAM lambda controller from ``GAM_EWCLoRA_lambda.py``:

    - delta-W Fisher/EWC geometry;
    - safe lambda cap ``lr * lambda * F_max <= c``;
    - optional Fisher/Euclidean/spectral dual constraints.

Only the per-step gradient composition is changed.  The base GAM version uses:

    g_final = g_gam + g_ewc

This projected version uses:

    g_task_proj = g_gam - <g_gam, g_ewc> / (||g_ewc||^2 + eps) * g_ewc
    g_final     = g_task_proj + g_ewc

by default only when the current Fisher drift is near the boundary and the
GAM direction is outward, i.e. ``g_gam^T g_ewc < 0``.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch

from models_LoRAbasedCL.GAM_EWCLoRA_lambda import Learner as _GAMLambdaLearner
from optimer_PerturabtionType.gam import GAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_GAMLambdaLearner):
    """GAM + delta-W EWC with projected GAM/EWC gradient composition."""

    def __init__(self, args):
        super().__init__(args)

        self._ewc_project_grad = as_bool(args.get("ewc_project_grad", True))
        self._ewc_project_eps = float(args.get("ewc_project_eps", args.get("fisher_project_eps", 1e-12)))
        self._ewc_project_rho = float(args.get("ewc_project_rho", args.get("fisher_project_rho", self._dual_rho)))
        self._ewc_project_boundary = as_bool(args.get("ewc_project_boundary", True))
        self._ewc_project_outward_only = as_bool(args.get("ewc_project_outward_only", True))

        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-lambda-project] project_grad=%s project_rho=%.4g "
                "outward_only=%s boundary_gate=%s eps=%.4g",
                self._ewc_project_grad,
                self._ewc_project_rho,
                self._ewc_project_outward_only,
                self._ewc_project_boundary,
                self._ewc_project_eps,
            )

    def _gam_ewc_step(
        self,
        model: torch.nn.Module,
        optimizer: GAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        params = self._flatten_optimizer_params(optimizer)
        safe_stats = self._apply_joint_lambda_clip(optimizer, reason="before_ewc_grad")
        ewc_value, ewc_grad_cache = self._compute_ewc_grad_cache(params, model)
        drift_value = (
            float(self._compute_fisher_drift_value(model).detach().item())
            if self._fisher_past_delta and self._delta_reference
            else 0.0
        )
        log_cache: Dict[str, float] = {
            "ce_clean": 0.0,
            "gam": 0.0,
            "ewc": ewc_value,
            "project_drift": drift_value,
        }

        optimizer.set_direction_projector(
            self._build_projected_ewc_projector(ewc_grad_cache, log_cache)
        )

        # GAM calls closure 4 times: [clean, perturb0, grad_norm_ascent, perturb1].
        # loss_value returned by step() is from the 1st (clean) call.
        # We capture the 2nd call (first perturbed point) as the GAM loss.
        _call_count = [0]

        def closure():
            _call_count[0] += 1
            optimizer.zero_grad()
            outputs = self._network(inputs)
            ce_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            if _call_count[0] == 1:
                log_cache["ce_clean"] = float(ce_loss.detach().item())
            else:
                log_cache["gam"] = float(ce_loss.detach().item())
            ce_loss.backward()
            return outputs, ce_loss.detach()

        try:
            outputs, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)
            optimizer.zero_grad()

        ce_value = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        log_cache["ce_clean"] = ce_value
        # do not overwrite log_cache["gam"] — it already holds a perturbed-point loss
        log_cache["lambda_ewc"] = float(self._ewc_lambda)
        log_cache["ewc_gam_ratio"] = float(ewc_value / (ce_value + self._mechanism_eps))
        if self._dual_update_by_epoch:
            dual_stats = self._default_dual_stats(ewc_ratio=log_cache["ewc_gam_ratio"], optimizer=optimizer)
        else:
            dual_stats = self._maybe_update_dual_lambda(
                model,
                current_loss=ce_value,
                current_ewc_ratio=log_cache["ewc_gam_ratio"],
                optimizer=optimizer,
            )
        log_cache.update(dual_stats)
        log_cache.update(safe_stats)

        return outputs["logits"].detach(), ce_value + ewc_value, log_cache

    def _build_projected_ewc_projector(
        self,
        ewc_grad_cache: Dict[int, torch.Tensor],
        log_cache: Dict[str, float],
    ):
        eps = float(self._ewc_project_eps)
        drift_value = float(log_cache.get("project_drift", 0.0))
        threshold = float(self._ewc_project_rho * self._dual_budget)

        def projector(param_groups):
            gam_norm_sq = 0.0
            ewc_norm_sq = 0.0
            final_norm_sq = 0.0
            gam_ewc_dot = 0.0
            param_count = 0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    gam_grad = None if param.grad is None else param.grad.detach().clone()
                    ewc_grad = ewc_grad_cache.get(id(param))
                    if gam_grad is None and ewc_grad is None:
                        continue

                    param_count += 1
                    if gam_grad is not None:
                        gam_f = gam_grad.detach().float()
                        gam_norm_sq += float((gam_f * gam_f).sum().item())
                    if ewc_grad is not None:
                        ewc_f = ewc_grad.detach().float()
                        ewc_norm_sq += float((ewc_f * ewc_f).sum().item())
                    if gam_grad is not None and ewc_grad is not None:
                        gam_ewc_dot += float((gam_grad.detach().float() * ewc_grad.detach().float()).sum().item())

            boundary_active = (not self._ewc_project_boundary) or (drift_value > threshold)
            outward = gam_ewc_dot < 0.0
            apply_projection = (
                self._ewc_project_grad
                and boundary_active
                and ewc_norm_sq > eps
                and ((not self._ewc_project_outward_only) or outward)
            )
            coeff = float(gam_ewc_dot / (ewc_norm_sq + eps)) if apply_projection else 0.0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    gam_grad = None if param.grad is None else param.grad.detach().clone()
                    ewc_grad = ewc_grad_cache.get(id(param))
                    if gam_grad is None and ewc_grad is None:
                        continue

                    if gam_grad is None:
                        gam_term = torch.zeros_like(ewc_grad, device=param.device, dtype=param.dtype)
                    else:
                        gam_term = gam_grad.to(device=param.device, dtype=param.dtype)

                    if ewc_grad is None:
                        ewc_term = torch.zeros_like(gam_term, device=param.device, dtype=param.dtype)
                    else:
                        ewc_term = ewc_grad.to(device=param.device, dtype=param.dtype)

                    if apply_projection:
                        gam_term = gam_term - coeff * ewc_term
                    final_grad = gam_term + ewc_term

                    if param.grad is None:
                        param.grad = final_grad.detach().clone()
                    else:
                        param.grad.data.copy_(final_grad)

                    final_f = final_grad.detach().float()
                    final_norm_sq += float((final_f * final_f).sum().item())

            log_cache.update(
                {
                    "gam_grad_norm": gam_norm_sq ** 0.5,
                    "ewc_grad_norm": ewc_norm_sq ** 0.5,
                    "final_grad_norm": final_norm_sq ** 0.5,
                    "cos_gam_ewc": self._cosine_from_sums(
                        gam_ewc_dot,
                        gam_norm_sq,
                        ewc_norm_sq,
                        eps,
                    ),
                    "project_enabled": float(self._ewc_project_grad),
                    "project_applied": float(apply_projection),
                    "project_boundary_active": float(boundary_active),
                    "project_outward": float(outward),
                    "project_threshold": float(threshold),
                    "project_dot": float(gam_ewc_dot),
                    "project_coeff": float(coeff),
                    "project_param_count": float(param_count),
                }
            )

        return projector

    # Keep the base step compatible with callers that still ask for this hook.
    def _build_additive_ewc_projector(
        self,
        ewc_grad_cache: Dict[int, torch.Tensor],
        log_cache: Dict[str, float],
    ):
        return self._build_projected_ewc_projector(ewc_grad_cache, log_cache)

    def after_task(self):
        super().after_task()
        if self._mechanism_eval and self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "gam_ewclora_lambda_project",
                    "ewc_project_grad": bool(self._ewc_project_grad),
                    "ewc_project_eps": float(self._ewc_project_eps),
                    "ewc_project_rho": float(self._ewc_project_rho),
                    "ewc_project_boundary": bool(self._ewc_project_boundary),
                    "ewc_project_outward_only": bool(self._ewc_project_outward_only),
                }
            )


GAMEWCLoRALambdaProject = Learner
GAMEWCLoRALambda = Learner
EWCLoRAProject = Learner
EWCLoRA = Learner
