"""AS1 NormFisher GAM + Euclidean cumulative hard trust-region or soft hinge penalty.

Thin subclass of ``as1_normfisher_gam_intask.Learner`` that adds one of:

Hard EucTR (``ewc_euclidean_trust=True``):
    After the parent projector writes ``param.grad = g_gam + g_ewc``, scale all
    gradients by a in [0, 1] so that the cumulative displacement from the
    task-start snapshot satisfies:
        ||theta_s - a * lr * v - theta_{t-1}||_2 <= rho

Soft delta-W penalty (``ewc_delta_penalty=True``):
    Add a hinge gradient to ``param.grad`` before the optimizer step:
        g_pen = 2 * lambda_pen * max(0, ||delta|| - rho) * delta / ||delta||
    where delta = theta_s - theta_{t-1}.

Only one of the two can be active at a time.  Task 0 is never constrained.
Classifier parameters are excluded by default.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner as _AS1InTask
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_AS1InTask):

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def __init__(self, args):
        super().__init__(args)

        # Hard EucTR
        self._euc_trust: bool = as_bool(
            args.get("ewc_euclidean_trust", args.get("euclidean_trust", False))
        )
        self._euc_radius: Optional[float] = self._parse_pos_float(
            args.get("ewc_euclidean_radius", args.get("euclidean_radius", None))
        )
        self._euc_radius_from_gam: bool = as_bool(
            args.get("ewc_euclidean_radius_from_gam", False)
        )
        self._euc_radius_source: str = str(
            args.get("ewc_euclidean_radius_source", "gam_grad_rho")
        )
        self._euc_radius_scale: float = float(
            args.get("ewc_euclidean_radius_scale", 1.0)
        )
        self._euc_outside_action: str = str(
            args.get("ewc_euclidean_outside_action", "zero")
        ).lower()
        self._euc_exclude_classifier: bool = as_bool(
            args.get("ewc_euclidean_exclude_classifier", True)
        )
        self._euc_eps: float = float(
            args.get("ewc_euclidean_eps", 1e-12)
        )

        # Soft delta-W hinge penalty
        self._dpen_active: bool = as_bool(
            args.get("ewc_delta_penalty", args.get("delta_w_penalty", False))
        )
        self._dpen_lambda: float = float(
            args.get("ewc_delta_penalty_lambda", args.get("delta_w_penalty_lambda", 0.0))
        )
        self._dpen_radius: Optional[float] = self._parse_pos_float(
            args.get("ewc_delta_penalty_radius", args.get("delta_w_penalty_radius", None))
        )
        self._dpen_exclude_classifier: bool = as_bool(
            args.get("ewc_delta_penalty_exclude_classifier", True)
        )
        self._dpen_eps: float = float(
            args.get("ewc_delta_penalty_eps", 1e-12)
        )

        # Disable hard EucTR if soft penalty is on (mutual exclusion)
        if self._dpen_active and self._euc_trust:
            logger.warning(
                "[AS1-EucTR] ewc_delta_penalty=True disables hard ewc_euclidean_trust."
            )
            self._euc_trust = False

        # Task-start parameter reference (raw A/B space)
        self._param_reference: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pos_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        v = float(value)
        return v if v > 0.0 else None

    @staticmethod
    def _is_classifier_param_name(name: str) -> bool:
        if name.startswith(("fc.", "head.", "classifier.")):
            return True
        return name.endswith(
            (".fc.weight", ".fc.bias", ".head.weight", ".head.bias",
             ".classifier.weight", ".classifier.bias")
        ) or ".classifiers." in name

    def _should_include(self, name: str, exclude_classifier: bool) -> bool:
        if not exclude_classifier:
            return True
        return not self._is_classifier_param_name(name)

    def _resolve_euc_radius(self) -> Optional[float]:
        if self._euc_radius is not None:
            return self._euc_radius
        if not self._euc_radius_from_gam:
            return None
        src = self._euc_radius_source.lower()
        if src in {"gam_grad_rho", "gam", "rho"}:
            rho = float(getattr(self, "_gam_grad_rho", 0.0))
        elif src in {"gam_grad_norm_rho", "norm_rho"}:
            rho = float(getattr(self, "_gam_grad_norm_rho", 0.0))
        else:
            return None
        return float(self._euc_radius_scale * rho) if rho > 0.0 else None

    def _snapshot_param_reference(self, param_groups) -> None:
        refs: Dict[int, torch.Tensor] = {}
        for group in param_groups:
            name_map = {id(p): n for n, p in
                        self._network.named_parameters() if p.requires_grad}
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pname = name_map.get(id(p), "")
                if self._euc_exclude_classifier and self._is_classifier_param_name(pname):
                    continue
                refs[id(p)] = p.detach().cpu().clone()
        self._param_reference = refs

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
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                total += 1
                ref = self._param_reference.get(id(p))
                if ref is None:
                    continue
                dev = p.device
                delta = p.detach().float() - ref.to(device=dev, dtype=torch.float32)
                v = param_to_v.get(id(p))
                step = (float(lr) * v.to(device=dev, dtype=torch.float32)
                        if v is not None else torch.zeros_like(delta))
                delta_norm_sq += float(delta.pow(2).sum().item())
                step_norm_sq += float(step.pow(2).sum().item())
                delta_dot_step += float((delta * step).sum().item())
                matched += 1

        A = step_norm_sq
        B = -2.0 * delta_dot_step
        C = delta_norm_sq - rho ** 2
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
                upper = max((-B - sq) / (2.0 * A + eps), (-B + sq) / (2.0 * A + eps))
                lower = min((-B - sq) / (2.0 * A + eps), (-B + sq) / (2.0 * A + eps))
                fl = max(0.0, lower)
                fu = min(1.0, upper)
                if fl <= fu:
                    scale = max(0.0, min(1.0, fu))
                    found = True

        if not found:
            scale = 0.0 if self._euc_outside_action == "zero" else 1.0

        after_sq = max(delta_norm_sq - 2.0 * scale * delta_dot_step
                       + scale ** 2 * step_norm_sq, 0.0)
        rho_eps = rho + eps
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
    # Projector override
    # ------------------------------------------------------------------

    def _build_direction_projector(self, clean_grad_cache, fisher_grad_cache):
        parent_proj = super()._build_direction_projector(
            clean_grad_cache, fisher_grad_cache
        )
        euc_active = self._euc_trust and bool(self._param_reference)
        pen_active = self._dpen_active and bool(self._param_reference)

        if not euc_active and not pen_active:
            return parent_proj

        euc_rho = self._resolve_euc_radius() if euc_active else None
        dpen_rho = self._dpen_radius
        dpen_lam = float(self._dpen_lambda)
        euc_eps = self._euc_eps
        dpen_eps = self._dpen_eps

        param_ref = self._param_reference  # snapshot captured at task start

        def projector(param_groups):
            # 1. Parent sets param.grad = g_gam + g_ewc
            parent_proj(param_groups)

            # 2. Hard EucTR: scale all grads by a in [0,1]
            if euc_active and euc_rho is not None:
                param_to_v: Dict[int, torch.Tensor] = {}
                for group in param_groups:
                    for p in group["params"]:
                        if p.requires_grad and p.grad is not None:
                            param_to_v[id(p)] = p.grad.detach().clone()
                scale, _ = self._compute_euclidean_cumulative_scale(
                    param_groups, param_to_v, euc_rho
                )
                if scale != 1.0:
                    for group in param_groups:
                        for p in group["params"]:
                            if p.requires_grad and p.grad is not None:
                                p.grad.data.mul_(scale)

            # 3. Soft penalty: add hinge gradient
            if pen_active and dpen_rho is not None and dpen_lam > 0.0:
                # Compute global delta norm (excluding classifier if configured)
                delta_norm_sq = 0.0
                for group in param_groups:
                    for p in group["params"]:
                        if not p.requires_grad:
                            continue
                        ref = param_ref.get(id(p))
                        if ref is None:
                            continue
                        delta = p.detach().float() - ref.to(device=p.device, dtype=torch.float32)
                        delta_norm_sq += float(delta.pow(2).sum().item())
                delta_norm = delta_norm_sq ** 0.5
                excess = delta_norm - dpen_rho
                if excess > 0.0:
                    coeff = 2.0 * dpen_lam * excess / (delta_norm + dpen_eps)
                    for group in param_groups:
                        for p in group["params"]:
                            if not p.requires_grad:
                                continue
                            ref = param_ref.get(id(p))
                            if ref is None:
                                continue
                            delta = (p.detach().float()
                                     - ref.to(device=p.device, dtype=torch.float32))
                            pen_grad = (coeff * delta).to(dtype=p.dtype)
                            if p.grad is None:
                                p.grad = pen_grad
                            else:
                                p.grad.data.add_(pen_grad)

        return projector

    # ------------------------------------------------------------------
    # Task-start snapshot with Task-0 guard
    # ------------------------------------------------------------------

    def _run_regularized_train(self, train_loader, test_loader, optimizer,
                               scheduler, *, epochs, class_offset):
        # Snapshot params at task start; skip Task 0 (no prior anchor)
        if self._cur_task > 0 and (self._euc_trust or self._dpen_active):
            self._snapshot_param_reference(optimizer.param_groups)
        else:
            self._param_reference = {}

        super()._run_regularized_train(
            train_loader, test_loader, optimizer, scheduler,
            epochs=epochs, class_offset=class_offset,
        )
