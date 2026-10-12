"""SeqLoRA EWCLoRA with GAM and adaptive EWC lambda.

This learner keeps the SeqLoRA EWCLoRA lifecycle from
``ewclora_youyue_fitArchitechture.py``:

    - Fisher is estimated in effective LoRA delta-W space.
    - The EWC anchor is the task-start delta-W snapshot.
    - ``after_task`` explicitly computes and accumulates the past-task Fisher.

Only the per-batch optimizer logic is changed from Fisher-SAM to GAM.  GAM is
run on the current-task CE loss only.  The EWC penalty is differentiated
separately at the clean weights and added to the final GAM direction before
GAM's base optimizer step:

    g_final = g_gam + g_ewc

Like ``FisherSAM_EWCLoRA_lambda.py``, this variant explicitly updates the
delta-W Fisher in ``after_task`` so in-task Fisher snapshots can be mixed into
the final task Fisher.  It also supports the same drift-budget dual update of
``_ewc_lambda`` using D_F <= sigma.

Optionally, a Euclidean hard trust-region layer can rescale the final
``g_gam + g_ewc`` direction so the cumulative trainable-parameter displacement
from the task-start snapshot remains within the configured rho.  As a softer
alternative, a mutually exclusive hinge penalty on the same cumulative
delta-W norm can be added as a third gradient term.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner as _SeqEWCLoRA
from optimer_PerturabtionType.gam import GAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_SeqEWCLoRA):
    """SeqLoRA + delta-W EWC + GAM current-task optimizer."""

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

        self._configured_optimizer_type = str(getattr(self, "_optimizer_type", "sgd")).lower()
        self._optimizer_type = "gam"
        self._base_optimizer_type = "gam"
        self._optimizer_type_by_task = None

        self._gam_base_optimizer = self._resolve_gam_base_optimizer(args)
        self._gam_adaptive = as_bool(args.get("gam_adaptive", False))
        self._gam_grad_reduce = str(args.get("gam_grad_reduce", "mean"))
        self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
        self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.02)))
        self._gam_grad_norm_rho = float(
            args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
        )
        self._gam_args = SimpleNamespace(
            grad_beta_1=float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1.0))),
            grad_beta_2=float(args.get("gam_grad_beta_2", args.get("grad_beta_2", 1.0))),
            grad_beta_3=float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1.0))),
            grad_gamma=float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.03))),
        )

        self._lambda_ewc = float(self._ewc_lambda)
        self._fisher_eta = float(
            getattr(self, "_ewc_eta", args.get("ewc_eta", args.get("eta", 1e-12)))
        )

        self._dual_ascent = as_bool(args.get("ewc_dual_ascent", args.get("gam_ewc_dual_ascent", True)))
        dual_budget_keys = ("ewc_dual_delta", "gamma_a_sq", "gam_ewc_gamma_a_sq")
        dual_budget_is_explicit = any(key in args for key in dual_budget_keys)
        dual_budget_default = float(args.get("ewc_dual_raw_budget_default", 1e-3))
        self._dual_budget = float(
            args.get(
                "ewc_dual_delta",
                args.get("gamma_a_sq", args.get("gam_ewc_gamma_a_sq", dual_budget_default)),
            )
        )
        if not dual_budget_is_explicit:
            logger.warning(
                "[GAM-EWCLoRA-lambda] Using raw-Fisher default ewc_dual_delta=%.4g. "
                "Delta-W Fisher drift is scale-dependent; tune ewc_dual_delta explicitly for final runs.",
                self._dual_budget,
            )
        self._dual_relative = as_bool(args.get("ewc_dual_relative", True))
        self._dual_lambda_min = float(args.get("ewc_dual_lambda_min", 1.0))
        self._dual_lambda_max = float(args.get("ewc_dual_lambda_max", 1e6))
        self._dual_eps = float(args.get("ewc_dual_eps", 1e-12))
        self._dual_use_eta = as_bool(args.get("ewc_dual_use_eta", True))
        self._dual_update_by_epoch = as_bool(args.get("ewc_dual_update_by_epoch", True))
        default_dual_interval = 5 if self._dual_update_by_epoch else 100
        self._dual_update_interval = max(1, int(args.get("ewc_dual_update_interval", default_dual_interval)))
        self._dual_eta_up = float(args.get("ewc_dual_eta_up", 0.05))
        self._dual_eta_down = float(args.get("ewc_dual_eta_down", 0.05))
        self._dual_rho = float(args.get("ewc_dual_rho", args.get("ewc_dual_low_ratio", 0.5)))
        self._dual_step_count = 0
        self._dual_euclidean_budget = self._optional_positive_float(
            args.get(
                "ewc_dual_euclidean_delta",
                args.get("ewc_dual_l2_delta", args.get("ewc_dual_r_e_sq", None)),
            )
        )
        self._dual_spectral_budget = self._optional_positive_float(
            args.get(
                "ewc_dual_spectral_delta",
                args.get("ewc_dual_fmax_l2_delta", args.get("ewc_dual_r_m_sq", None)),
            )
        )
        self._dual_use_joint_constraints = as_bool(
            args.get(
                "ewc_dual_use_joint_constraints",
                self._dual_euclidean_budget is not None or self._dual_spectral_budget is not None,
            )
        )

        self._safe_lambda_cap = as_bool(args.get("ewc_safe_lambda_cap", True))
        self._safe_lambda_c = float(args.get("ewc_safe_lambda_c", 0.5))
        self._safe_lambda_eps = float(args.get("ewc_safe_lambda_eps", 1e-12))
        self._safe_lambda_override_min = as_bool(args.get("ewc_safe_lambda_override_min", True))
        self._last_base_lr = float(
            args.get(
                "lrate",
                args.get(
                    "lr",
                    args.get(
                        "init_lr",
                        args.get("full_lr", args.get("learning_rate", 0.0)),
                    ),
                ),
            )
        )
        self._last_base_momentum = 0.0
        self._last_base_weight_decay = 0.0
        self._last_safe_lambda_max = float(self._dual_lambda_max)

        self._gam_ewc_euclidean_trust = as_bool(
            args.get(
                "gam_ewc_euclidean_trust",
                args.get("ewc_euclidean_trust", args.get("euclidean_trust", False)),
            )
        )
        self._gam_ewc_euclidean_exclude_classifier = as_bool(
            args.get(
                "gam_ewc_euclidean_exclude_classifier",
                args.get("ewc_euclidean_exclude_classifier", True),
            )
        )
        self._gam_ewc_euclidean_radius_from_gam = as_bool(
            args.get(
                "gam_ewc_euclidean_radius_from_gam",
                args.get("ewc_euclidean_radius_from_gam", True),
            )
        )
        self._gam_ewc_euclidean_radius = self._optional_positive_float(
            args.get(
                "gam_ewc_euclidean_radius",
                args.get(
                    "ewc_euclidean_radius",
                    args.get("gam_ewc_r_trust", args.get("ewc_r_trust", None)),
                ),
            )
        )
        self._gam_ewc_euclidean_radius_source = str(
            args.get(
                "gam_ewc_euclidean_radius_source",
                args.get("ewc_euclidean_radius_source", "gam_grad_rho"),
            )
        ).lower()
        self._gam_ewc_euclidean_radius_scale = float(
            args.get(
                "gam_ewc_euclidean_radius_scale",
                args.get("ewc_euclidean_radius_scale", 1.0),
            )
        )
        self._gam_ewc_euclidean_eps = float(
            args.get("gam_ewc_euclidean_eps", args.get("ewc_euclidean_eps", 1e-12))
        )
        self._gam_ewc_euclidean_outside_action = str(
            args.get(
                "gam_ewc_euclidean_outside_action",
                args.get("ewc_euclidean_outside_action", "zero"),
            )
        ).lower()
        if self._gam_ewc_euclidean_outside_action not in {"zero", "none"}:
            self._gam_ewc_euclidean_outside_action = "zero"

        self._gam_ewc_delta_penalty = as_bool(
            args.get(
                "gam_ewc_delta_penalty",
                args.get("ewc_delta_penalty", args.get("delta_w_penalty", False)),
            )
        )
        self._gam_ewc_delta_penalty_lambda = float(
            args.get(
                "gam_ewc_delta_penalty_lambda",
                args.get("ewc_delta_penalty_lambda", args.get("delta_w_penalty_lambda", 0.0)),
            )
        )
        if self._gam_ewc_delta_penalty_lambda < 0.0:
            logger.warning(
                "[GAM-EWCLoRA-lambda] Negative delta penalty lambda %.4g is invalid; "
                "clamping to 0.",
                self._gam_ewc_delta_penalty_lambda,
            )
            self._gam_ewc_delta_penalty_lambda = 0.0
        self._gam_ewc_delta_penalty_exclude_classifier = as_bool(
            args.get(
                "gam_ewc_delta_penalty_exclude_classifier",
                args.get("ewc_delta_penalty_exclude_classifier", True),
            )
        )
        self._gam_ewc_delta_penalty_radius_from_gam = as_bool(
            args.get(
                "gam_ewc_delta_penalty_radius_from_gam",
                args.get("ewc_delta_penalty_radius_from_gam", False),
            )
        )
        self._gam_ewc_delta_penalty_radius = self._optional_positive_float(
            args.get(
                "gam_ewc_delta_penalty_radius",
                args.get("ewc_delta_penalty_radius", args.get("delta_w_penalty_radius", None)),
            )
        )
        self._gam_ewc_delta_penalty_radius_source = str(
            args.get(
                "gam_ewc_delta_penalty_radius_source",
                args.get("ewc_delta_penalty_radius_source", "gam_grad_rho"),
            )
        ).lower()
        self._gam_ewc_delta_penalty_radius_scale = float(
            args.get(
                "gam_ewc_delta_penalty_radius_scale",
                args.get("ewc_delta_penalty_radius_scale", 1.0),
            )
        )
        self._gam_ewc_delta_penalty_eps = float(
            args.get("gam_ewc_delta_penalty_eps", args.get("ewc_delta_penalty_eps", 1e-12))
        )
        if self._gam_ewc_delta_penalty and self._gam_ewc_euclidean_trust:
            logger.warning(
                "[GAM-EWCLoRA-lambda] ewc_delta_penalty=True disables hard "
                "ewc_euclidean_trust to keep soft and hard delta-W controls mutually exclusive."
            )
            self._gam_ewc_euclidean_trust = False
        if self._gam_ewc_delta_penalty and (
            self._gam_ewc_delta_penalty_lambda <= 0.0
            or self._resolve_gam_ewc_delta_penalty_radius() is None
        ):
            logger.warning(
                "[GAM-EWCLoRA-lambda] ewc_delta_penalty=True requires positive "
                "ewc_delta_penalty_lambda and ewc_delta_penalty_radius; the penalty "
                "will stay inactive until both are configured."
            )
        self._param_reference: Dict[int, torch.Tensor] = {}
        self._param_name_by_id: Dict[int, str] = {}
        self._param_reference_skipped_classifier = 0
        self._warned_euclidean_optimizer = False

        self._approximate_fisher_in_task = as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get(
                    "approximate_fisher_inTask",
                    args.get("approximate_fisher_in_task", True),
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
        self._in_task_fisher_snapshots: List[
            Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}

        self._mechanism_eval = as_bool(args.get("mechanism_eval", False))
        self._mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._mechanism_top_fraction = float(args.get("mechanism_top_fraction", 0.1))
        topk = args.get("mechanism_topk", None)
        self._mechanism_topk = None if topk is None else int(topk)
        self._mechanism_sums: Dict[str, float] = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}

        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-lambda] base=%s grad_rho=%.4g grad_norm_rho=%.4g "
                "adaptive=%s grad_reduce=%s dual=%s budget(sigma)=%.4g "
                "update_interval=%d by_epoch=%s "
                "joint_constraints=%s euclidean_budget=%.4g spectral_budget=%.4g "
                "lambda_bounds=[%.4g, %.4g] lambda_init=%.4g "
                "euc_trust=%s euc_radius=%s euc_from_gam=%s euc_source=%s "
                "euc_radius_scale=%.3g euc_outside=%s euc_exclude_classifier=%s "
                "delta_penalty=%s delta_penalty_lambda=%.4g delta_penalty_radius=%s "
                "delta_penalty_from_gam=%s delta_penalty_source=%s "
                "delta_penalty_radius_scale=%.3g delta_penalty_exclude_classifier=%s "
                "approximte_fisher_inTask=%s interval=%d omega=%.3g",
                self._gam_base_optimizer,
                self._gam_grad_rho,
                self._gam_grad_norm_rho,
                self._gam_adaptive,
                self._gam_grad_reduce,
                self._dual_ascent,
                self._dual_budget,
                self._dual_update_interval,
                self._dual_update_by_epoch,
                self._dual_use_joint_constraints,
                0.0 if self._dual_euclidean_budget is None else self._dual_euclidean_budget,
                0.0 if self._dual_spectral_budget is None else self._dual_spectral_budget,
                self._dual_lambda_min,
                self._dual_lambda_max,
                self._ewc_lambda,
                self._gam_ewc_euclidean_trust,
                self._resolve_gam_ewc_euclidean_radius(),
                self._gam_ewc_euclidean_radius_from_gam,
                self._gam_ewc_euclidean_radius_source,
                self._gam_ewc_euclidean_radius_scale,
                self._gam_ewc_euclidean_outside_action,
                self._gam_ewc_euclidean_exclude_classifier,
                self._gam_ewc_delta_penalty,
                self._gam_ewc_delta_penalty_lambda,
                self._resolve_gam_ewc_delta_penalty_radius(),
                self._gam_ewc_delta_penalty_radius_from_gam,
                self._gam_ewc_delta_penalty_radius_source,
                self._gam_ewc_delta_penalty_radius_scale,
                self._gam_ewc_delta_penalty_exclude_classifier,
                self._approximate_fisher_in_task,
                self._in_task_fisher_interval,
                self._in_task_fisher_omega,
            )
        if (
            self._dual_use_joint_constraints
            and (self._dual_euclidean_budget is not None or self._dual_spectral_budget is not None)
            and self._fisher_eta <= 0.0
        ):
            logger.warning(
                "[GAM-EWCLoRA-lambda] Euclidean/spectral lambda constraints are enabled "
                "with ewc_eta<=0. A single Fisher-EWC lambda cannot control drift in low-Fisher "
                "or zero-Fisher directions unless an eta floor is used."
            )

    def _resolve_gam_base_optimizer(self, args) -> str:
        base = str(
            args.get(
                "gam_base_optimizer",
                args.get("optimizer", self._configured_optimizer_type),
            )
        ).lower()
        if base in {"gam", "fishersam", "sam", "cflat", "rwp", "arwp"}:
            return "sgd"
        if base not in {"sgd", "adam", "adamw"}:
            logger.warning(
                "[GAM-EWCLoRA-lambda] Unsupported gam_base_optimizer=%s; using SGD.",
                base,
            )
            return "sgd"
        return base

    def _resolve_gam_ewc_euclidean_radius(self) -> Optional[float]:
        """Return the Euclidean ball radius for the cumulative EucTR layer."""
        if self._gam_ewc_euclidean_radius is not None:
            return float(self._gam_ewc_euclidean_radius)
        if not self._gam_ewc_euclidean_radius_from_gam:
            return None

        src = self._gam_ewc_euclidean_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self._gam_grad_rho)
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self._gam_grad_norm_rho)
        else:
            return None
        return float(self._gam_ewc_euclidean_radius_scale * rho)

    def _resolve_gam_ewc_delta_penalty_radius(self) -> Optional[float]:
        """Return rho for the soft cumulative delta-W hinge penalty."""
        if self._gam_ewc_delta_penalty_radius is not None:
            return float(self._gam_ewc_delta_penalty_radius)
        if not self._gam_ewc_delta_penalty_radius_from_gam:
            return None

        src = self._gam_ewc_delta_penalty_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self._gam_grad_rho)
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self._gam_grad_norm_rho)
        else:
            return None
        return float(self._gam_ewc_delta_penalty_radius_scale * rho)

    def _refresh_param_name_map(self) -> None:
        model = self._unwrap_network()
        self._param_name_by_id = {id(param): name for name, param in model.named_parameters()}

    @staticmethod
    def _is_classifier_param_name(name: str) -> bool:
        name = name[7:] if name.startswith("module.") else name
        if name.startswith(("fc.", "head.", "classifier.")):
            return True
        classifier_suffixes = (
            ".fc.weight",
            ".fc.bias",
            ".head.weight",
            ".head.bias",
            ".classifier.weight",
            ".classifier.bias",
        )
        return name.endswith(classifier_suffixes) or ".classifiers." in name

    def _is_euclidean_tracked_param(self, param: torch.Tensor) -> bool:
        if not self._gam_ewc_euclidean_exclude_classifier:
            return True
        name = self._param_name_by_id.get(id(param), "")
        return not self._is_classifier_param_name(name)

    def _is_delta_penalty_tracked_param(self, param: torch.Tensor) -> bool:
        if not self._gam_ewc_delta_penalty_exclude_classifier:
            return True
        name = self._param_name_by_id.get(id(param), "")
        return not self._is_classifier_param_name(name)

    def _is_reference_tracked_param(self, param: torch.Tensor) -> bool:
        if self._gam_ewc_delta_penalty:
            return self._is_delta_penalty_tracked_param(param)
        return self._is_euclidean_tracked_param(param)

    def _snapshot_param_reference(self, param_groups) -> None:
        """Snapshot trainable parameters at task start as theta_{t-1}."""
        self._refresh_param_name_map()
        refs: Dict[int, torch.Tensor] = {}
        skipped_classifier = 0
        for group in param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                if not self._is_reference_tracked_param(param):
                    skipped_classifier += 1
                    continue
                refs[id(param)] = param.detach().cpu().clone()
        self._param_reference = refs
        self._param_reference_skipped_classifier = skipped_classifier

    def _compute_euclidean_cumulative_scale(
        self,
        param_groups,
        param_to_v: Dict[int, torch.Tensor],
        rho: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute a in [0, 1] so ||theta_s - a * lr * v - theta_ref||_2 <= rho.

        The helper uses each optimizer parameter group's own learning rate, so
        it remains well-defined when LoRA and classifier groups have different
        lrs.  The strict feasibility guarantee assumes the base optimizer step
        is plain SGD without momentum or weight decay.
        """
        eps = float(self._gam_ewc_euclidean_eps)
        delta_norm_sq = 0.0
        step_norm_sq = 0.0
        delta_dot_step = 0.0
        lr_max = 0.0
        lr_min = None
        matched = 0
        total = 0
        total_all = 0
        skipped_classifier = 0

        for group in param_groups:
            lr = float(group.get("lr", 0.0))
            if lr > 0.0:
                lr_max = max(lr_max, lr)
                lr_min = lr if lr_min is None else min(lr_min, lr)
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                total_all += 1
                if not self._is_euclidean_tracked_param(param):
                    skipped_classifier += 1
                    continue
                total += 1
                ref = self._param_reference.get(id(param))
                if ref is None:
                    continue

                dev = param.device
                v = param_to_v.get(id(param))
                delta = param.detach().float() - ref.to(device=dev, dtype=torch.float32)
                if v is None:
                    step = torch.zeros_like(delta)
                else:
                    step = float(lr) * v.to(device=dev, dtype=torch.float32)
                delta_norm_sq += float(delta.pow(2).sum().item())
                step_norm_sq += float(step.pow(2).sum().item())
                delta_dot_step += float((delta * step).sum().item())
                matched += 1

        a_coeff = step_norm_sq
        b_coeff = -2.0 * delta_dot_step
        c_coeff = delta_norm_sq - float(rho) ** 2
        current_feasible = c_coeff <= 0.0
        scale = 1.0
        found_feasible_scale = False

        if a_coeff <= eps:
            found_feasible_scale = current_feasible
            scale = 1.0 if current_feasible else 0.0
        else:
            disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
            if disc >= 0.0:
                sqrt_disc = disc ** 0.5
                root_a = (-b_coeff - sqrt_disc) / (2.0 * a_coeff + eps)
                root_b = (-b_coeff + sqrt_disc) / (2.0 * a_coeff + eps)
                lower = min(root_a, root_b)
                upper = max(root_a, root_b)
                feasible_lower = max(0.0, lower)
                feasible_upper = min(1.0, upper)
                if feasible_lower <= feasible_upper:
                    scale = max(0.0, min(1.0, feasible_upper))
                    found_feasible_scale = True

        if not found_feasible_scale:
            if self._gam_ewc_euclidean_outside_action == "zero":
                scale = 0.0
            else:
                scale = 1.0

        after_norm_sq = max(
            delta_norm_sq
            - 2.0 * float(scale) * delta_dot_step
            + float(scale) ** 2 * step_norm_sq,
            0.0,
        )
        rho_eps = float(rho) + eps
        stats = {
            "euc_scale": float(scale),
            "euc_before_norm": float(delta_norm_sq ** 0.5),
            "euc_after_norm": float(after_norm_sq ** 0.5),
            "euc_budget": float(rho),
            "euc_ratio_before": float(delta_norm_sq ** 0.5 / rho_eps),
            "euc_ratio_after": float(after_norm_sq ** 0.5 / rho_eps),
            "euc_step_norm": float(step_norm_sq ** 0.5),
            "euc_delta_dot_step": float(delta_dot_step),
            "euc_current_feasible": float(current_feasible),
            "euc_found_feasible_scale": float(found_feasible_scale),
            "euc_matched": float(matched),
            "euc_total": float(total),
            "euc_total_all": float(total_all),
            "euc_skipped_classifier": float(skipped_classifier),
            "euc_lr_min": float(lr_min if lr_min is not None else 0.0),
            "euc_lr_max": float(lr_max),
        }
        return float(scale), stats

    def _default_delta_penalty_stats(self) -> Dict[str, float]:
        return {
            "delta_penalty_enabled": float(self._gam_ewc_delta_penalty),
            "delta_penalty_configured": 0.0,
            "delta_penalty_active": 0.0,
            "delta_penalty_loss": 0.0,
            "delta_penalty_lambda": float(self._gam_ewc_delta_penalty_lambda),
            "delta_penalty_budget": float(self._resolve_gam_ewc_delta_penalty_radius() or 0.0),
            "delta_penalty_drift_norm": 0.0,
            "delta_penalty_ratio": 0.0,
            "delta_penalty_excess": 0.0,
            "delta_penalty_grad_norm": 0.0,
            "delta_penalty_matched": 0.0,
            "delta_penalty_total": 0.0,
            "delta_penalty_total_all": 0.0,
            "delta_penalty_skipped_classifier": 0.0,
            "delta_penalty_exclude_classifier": float(
                self._gam_ewc_delta_penalty_exclude_classifier
            ),
        }

    def _compute_delta_penalty_grad_cache(
        self,
        params: List[torch.nn.Parameter],
    ) -> Tuple[float, Dict[int, torch.Tensor], Dict[str, float]]:
        """Compute lambda * relu(||theta-theta_ref||_2-rho)^2 and its grads."""
        stats = self._default_delta_penalty_stats()
        if not self._gam_ewc_delta_penalty:
            return 0.0, {}, stats

        rho = self._resolve_gam_ewc_delta_penalty_radius()
        lam = float(self._gam_ewc_delta_penalty_lambda)
        if rho is None or lam <= 0.0 or not self._param_reference:
            return 0.0, {}, stats

        eps = float(self._gam_ewc_delta_penalty_eps)
        delta_norm_sq = None
        matched = 0
        total = 0
        total_all = 0
        skipped_classifier = 0

        with torch.enable_grad():
            for param in params:
                if not param.requires_grad:
                    continue
                total_all += 1
                if not self._is_delta_penalty_tracked_param(param):
                    skipped_classifier += 1
                    continue
                total += 1
                ref = self._param_reference.get(id(param))
                if ref is None:
                    continue

                delta = param.float() - ref.to(device=param.device, dtype=torch.float32)
                term = delta.pow(2).sum()
                delta_norm_sq = term if delta_norm_sq is None else delta_norm_sq + term
                matched += 1

            if delta_norm_sq is None:
                stats.update(
                    {
                        "delta_penalty_configured": 1.0,
                        "delta_penalty_total": float(total),
                        "delta_penalty_total_all": float(total_all),
                        "delta_penalty_skipped_classifier": float(skipped_classifier),
                    }
                )
                return 0.0, {}, stats

            delta_norm = torch.sqrt(delta_norm_sq + eps)
            excess = torch.clamp(delta_norm - float(rho), min=0.0)
            penalty_loss = lam * excess.pow(2)

        penalty_value = float(penalty_loss.detach().item())
        drift_norm = float(delta_norm.detach().item())
        excess_value = float(excess.detach().item())
        stats.update(
            {
                "delta_penalty_configured": 1.0,
                "delta_penalty_active": float(excess_value > 0.0),
                "delta_penalty_loss": penalty_value,
                "delta_penalty_budget": float(rho),
                "delta_penalty_drift_norm": drift_norm,
                "delta_penalty_ratio": float(drift_norm / (float(rho) + eps)),
                "delta_penalty_excess": excess_value,
                "delta_penalty_matched": float(matched),
                "delta_penalty_total": float(total),
                "delta_penalty_total_all": float(total_all),
                "delta_penalty_skipped_classifier": float(skipped_classifier),
            }
        )

        if excess_value <= 0.0 or not penalty_loss.requires_grad:
            return penalty_value, {}, stats

        grads = torch.autograd.grad(
            penalty_loss,
            params,
            allow_unused=True,
            retain_graph=False,
        )
        grad_cache: Dict[int, torch.Tensor] = {}
        grad_norm_sq = 0.0
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_detached = grad.detach().clone()
            grad_cache[id(param)] = grad_detached
            grad_norm_sq += float(grad_detached.pow(2).sum().item())

        stats["delta_penalty_grad_norm"] = float(grad_norm_sq ** 0.5)
        return penalty_value, grad_cache, stats

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        self._last_base_lr = float(lr)
        self._last_base_momentum = float(momentum)
        self._last_base_weight_decay = float(weight_decay)
        params = self._normalize_param_groups(params)
        base_name = self._gam_base_optimizer
        if (
            self._gam_ewc_euclidean_trust
            and not self._warned_euclidean_optimizer
            and (
                base_name != "sgd"
                or abs(float(momentum)) > 0.0
                or abs(float(weight_decay)) > 0.0
            )
        ):
            self._warned_euclidean_optimizer = True
            logger.warning(
                "[GAM-EWCLoRA-lambda][EucTR] Hard Euclidean scaling is exact only "
                "for SGD with momentum=0 and weight_decay=0. Current base=%s "
                "momentum=%.4g weight_decay=%.4g; scaling will still be applied "
                "to param.grad but the actual optimizer step may differ.",
                base_name,
                float(momentum),
                float(weight_decay),
            )

        if base_name == "adam":
            base_optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            )
        elif base_name == "adamw":
            base_optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            )
        else:
            base_optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        return GAM(
            params,
            base_optimizer=base_optimizer,
            model=self._network,
            grad_rho=float(self._gam_grad_rho),
            grad_norm_rho=float(self._gam_grad_norm_rho),
            adaptive=bool(self._gam_adaptive),
            perturb_eps=float(self._gam_perturb_eps),
            args=self._gam_args,
            grad_reduce=str(self._gam_grad_reduce),
        )

    @staticmethod
    def _normalize_param_groups(params):
        if not isinstance(params, (list, tuple)):
            return params

        normalized = []
        for group in params:
            if isinstance(group, dict):
                group_params = [p for p in group.get("params", []) if p.requires_grad]
                if not group_params:
                    continue
                new_group = dict(group)
                new_group["params"] = group_params
                normalized.append(new_group)
            elif getattr(group, "requires_grad", False):
                normalized.append(group)

        if not normalized:
            raise ValueError("No trainable parameters found for GAM-EWCLoRA.")
        return normalized

    # ------------------------------------------------------------------
    # GAM + EWC batch step
    # ------------------------------------------------------------------
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
        model = self._unwrap_network()
        self._snapshot_task_reference()
        # Skip cumulative delta-control snapshot on Task 0: no prior task to protect.
        if self._cur_task > 0:
            self._snapshot_param_reference(optimizer.param_groups)
        else:
            self._param_reference = {}
        self._reset_dual_controller_state()
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._apply_joint_lambda_clip(optimizer, reason="task_start")

        if self._is_main_process:
            has_f = bool(self._fisher_past_delta)
            has_r = bool(self._delta_reference)
            f_info = ""
            if has_f:
                f_vals = list(self._fisher_past_delta.values())
                f_max = float(max(v.max().item() for v in f_vals))
                f_sum = float(sum(v.sum().item() for v in f_vals))
                f_numel = sum(int(v.numel()) for v in f_vals)
                f_mean = f_sum / max(f_numel, 1)
                f_info = (
                    f"F_tensors={len(f_vals)} numel={f_numel} "
                    f"F_max={f_max:.3e} F_mean={f_mean:.3e} F_sum={f_sum:.3e}"
                )
            r_info = f"ref_tensors={len(self._delta_reference)}" if has_r else "ref=EMPTY"
            self._log(
                f"[GAM-EWCLoRA-lambda][TaskStart] task={self._cur_task} "
                f"lambda={self._ewc_lambda:.3e} "
                f"euc_trust={self._gam_ewc_euclidean_trust} "
                f"euc_radius={self._resolve_gam_ewc_euclidean_radius()} "
                f"delta_penalty={self._gam_ewc_delta_penalty} "
                f"delta_penalty_lambda={self._gam_ewc_delta_penalty_lambda:.3e} "
                f"delta_penalty_radius={self._resolve_gam_ewc_delta_penalty_radius()} "
                f"has_fisher={has_f} {f_info} | "
                f"has_ref={has_r} {r_info}"
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            ce_losses = 0.0
            gam_losses = 0.0
            ewc_losses = 0.0
            delta_penalty_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value, log_dict = self._gam_ewc_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )

                total_loss += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", 0.0))
                gam_losses += float(log_dict.get("gam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                delta_penalty_losses += float(log_dict.get("delta_penalty", 0.0))
                loss_count += 1

                if self._mechanism_eval:
                    self._record_mechanism_stats(log_dict)

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

            denom = max(1, loss_count)
            epoch_ewc_ratio = float(ewc_losses / (gam_losses + self._mechanism_eps))
            if self._dual_update_by_epoch:
                epoch_dual_stats = self._maybe_update_dual_lambda(
                    model,
                    current_loss=ce_losses / denom,
                    current_ewc_ratio=epoch_ewc_ratio,
                    optimizer=optimizer,
                )
                if self._mechanism_eval and epoch_dual_stats.get("dual_updated", 0.0):
                    self._record_mechanism_stats(epoch_dual_stats)

            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            drift_display = (
                float(self._compute_fisher_drift_value(model).detach().item())
                if self._fisher_past_delta and self._delta_reference
                else 0.0
            )
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / denom:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {ce_losses / denom:.3f}, GAM {gam_losses / denom:.3f}, "
                f"EWC {ewc_losses / denom:.3e}, DWpen {delta_penalty_losses / denom:.3e}, "
                f"D_F {drift_display:.3e}, "
                f"EWC/GAM {epoch_ewc_ratio:.3e}, lambda {float(self._ewc_lambda):.3g}"
            )
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._should_capture_in_task_fisher(epoch + 1):
                snapshot_count = self._capture_in_task_fisher_snapshot(
                    train_loader=train_loader,
                    model=model,
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

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        logits, loss_value, log_dict = self._gam_ewc_step(
            model=self._unwrap_network(),
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        if self._mechanism_eval:
            self._record_mechanism_stats(log_dict)
        return logits, loss_value

    def _gam_ewc_step(
        self,
        model: nn.Module,
        optimizer: GAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        params = self._flatten_optimizer_params(optimizer)
        safe_stats = self._apply_joint_lambda_clip(optimizer, reason="before_ewc_grad")
        ewc_value, ewc_grad_cache = self._compute_ewc_grad_cache(params, model)
        (
            delta_penalty_value,
            delta_penalty_grad_cache,
            delta_penalty_stats,
        ) = self._compute_delta_penalty_grad_cache(params)
        log_cache: Dict[str, float] = {
            "ce_clean": 0.0,
            "gam": 0.0,
            "ewc": ewc_value,
            "delta_penalty": delta_penalty_value,
        }
        log_cache.update(delta_penalty_stats)

        optimizer.set_direction_projector(
            self._build_additive_ewc_projector(
                ewc_grad_cache,
                delta_penalty_grad_cache,
                log_cache,
            )
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
        log_cache["delta_penalty_gam_ratio"] = float(
            delta_penalty_value / (ce_value + self._mechanism_eps)
        )
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

        return outputs["logits"].detach(), ce_value + ewc_value + delta_penalty_value, log_cache

    @staticmethod
    def _flatten_optimizer_params(optimizer) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(param)
        return params

    def _compute_ewc_grad_cache(
        self,
        params: List[torch.nn.Parameter],
        model: nn.Module,
    ) -> Tuple[float, Dict[int, torch.Tensor]]:
        if not self._fisher_past_delta or not self._delta_reference:
            return 0.0, {}

        with torch.enable_grad():
            ewc_loss = self._past_fisher_penalty(model)

        ewc_value = float(ewc_loss.detach().item())
        if not torch.is_tensor(ewc_loss) or not ewc_loss.requires_grad:
            return ewc_value, {}

        grads = torch.autograd.grad(
            ewc_loss,
            params,
            allow_unused=True,
            retain_graph=False,
        )
        grad_cache = {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }
        return ewc_value, grad_cache

    def _build_additive_ewc_projector(
        self,
        ewc_grad_cache: Dict[int, torch.Tensor],
        delta_penalty_grad_cache: Dict[int, torch.Tensor],
        log_cache: Dict[str, float],
    ):
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            gam_norm_sq = 0.0
            ewc_norm_sq = 0.0
            delta_penalty_norm_sq = 0.0
            unscaled_final_norm_sq = 0.0
            final_norm_sq = 0.0
            gam_ewc_dot = 0.0
            gam_delta_penalty_dot = 0.0
            ewc_delta_penalty_dot = 0.0
            param_to_v: Dict[int, torch.Tensor] = {}

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    gam_grad = None if param.grad is None else param.grad.detach().clone()
                    ewc_grad = ewc_grad_cache.get(id(param))
                    delta_penalty_grad = delta_penalty_grad_cache.get(id(param))

                    if gam_grad is None and ewc_grad is None and delta_penalty_grad is None:
                        continue
                    final_grad = None
                    for grad_part in (gam_grad, ewc_grad, delta_penalty_grad):
                        if grad_part is None:
                            continue
                        grad_part_t = grad_part.to(device=param.device, dtype=param.dtype)
                        final_grad = (
                            grad_part_t.clone()
                            if final_grad is None
                            else final_grad + grad_part_t
                        )
                    if final_grad is None:
                        continue
                    param_to_v[id(param)] = final_grad.detach().clone()

                    if self._mechanism_eval:
                        if gam_grad is not None:
                            gam_norm_sq += float(gam_grad.pow(2).sum().item())
                        if ewc_grad is not None:
                            ewc_norm_sq += float(ewc_grad.pow(2).sum().item())
                        if delta_penalty_grad is not None:
                            delta_penalty_norm_sq += float(delta_penalty_grad.pow(2).sum().item())
                        if gam_grad is not None and ewc_grad is not None:
                            gam_ewc_dot += float((gam_grad * ewc_grad).sum().item())
                        if gam_grad is not None and delta_penalty_grad is not None:
                            gam_delta_penalty_dot += float(
                                (gam_grad * delta_penalty_grad).sum().item()
                            )
                        if ewc_grad is not None and delta_penalty_grad is not None:
                            ewc_delta_penalty_dot += float(
                                (ewc_grad * delta_penalty_grad).sum().item()
                            )
                        unscaled_final_norm_sq += float(final_grad.pow(2).sum().item())

            euc_scale = 1.0
            euc_active = False
            if self._gam_ewc_euclidean_trust:
                rho_euc = self._resolve_gam_ewc_euclidean_radius()
                if rho_euc is not None and self._param_reference:
                    euc_scale, euc_stats = self._compute_euclidean_cumulative_scale(
                        param_groups=param_groups,
                        param_to_v=param_to_v,
                        rho=float(rho_euc),
                    )
                    log_cache.update(euc_stats)
                    log_cache["euc_active"] = 1.0
                    log_cache["euc_exclude_classifier"] = float(
                        self._gam_ewc_euclidean_exclude_classifier
                    )
                    euc_active = True
                else:
                    log_cache["euc_active"] = 0.0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue
                    final_grad = param_to_v.get(id(param))
                    if final_grad is None:
                        continue
                    if (
                        euc_active
                        and euc_scale != 1.0
                        and self._is_euclidean_tracked_param(param)
                    ):
                        final_grad = final_grad * float(euc_scale)

                    if param.grad is None:
                        param.grad = final_grad.clone()
                    else:
                        param.grad.data.copy_(final_grad)

                    if self._mechanism_eval:
                        final_norm_sq += float(final_grad.pow(2).sum().item())

            if self._mechanism_eval:
                log_cache.update(
                    {
                        "gam_grad_norm": gam_norm_sq ** 0.5,
                        "ewc_grad_norm": ewc_norm_sq ** 0.5,
                        "delta_penalty_projector_grad_norm": delta_penalty_norm_sq ** 0.5,
                        "unscaled_final_grad_norm": unscaled_final_norm_sq ** 0.5,
                        "final_grad_norm": final_norm_sq ** 0.5,
                        "cos_gam_ewc": self._cosine_from_sums(
                            gam_ewc_dot,
                            gam_norm_sq,
                            ewc_norm_sq,
                            eps,
                        ),
                        "cos_gam_delta_penalty": self._cosine_from_sums(
                            gam_delta_penalty_dot,
                            gam_norm_sq,
                            delta_penalty_norm_sq,
                            eps,
                        ),
                        "cos_ewc_delta_penalty": self._cosine_from_sums(
                            ewc_delta_penalty_dot,
                            ewc_norm_sq,
                            delta_penalty_norm_sq,
                            eps,
                        ),
                        "euc_scale": float(euc_scale),
                    }
                )

        return projector

    @staticmethod
    def _cosine_from_sums(dot: float, norm_a_sq: float, norm_b_sq: float, eps: float) -> float:
        denom = (max(norm_a_sq, 0.0) ** 0.5) * (max(norm_b_sq, 0.0) ** 0.5)
        if denom <= eps:
            return 0.0
        return float(dot / (denom + eps))

    # ------------------------------------------------------------------
    # In-task raw Fisher path averaging
    # ------------------------------------------------------------------
    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        if not self._approximate_fisher_in_task:
            return False
        if self._in_task_fisher_interval <= 0:
            return False
        if self._in_task_fisher_omega <= 0.0:
            return False
        return int(epoch_idx) % int(self._in_task_fisher_interval) == 0

    def _capture_lora_weight_snapshot(self, model: nn.Module) -> Dict[str, torch.Tensor]:
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
        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            lower_name = name.lower()
            if any(marker in lower_name for marker in markers):
                snapshot[name] = param.detach().cpu().clone()
        return snapshot

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: nn.Module,
        optimizer: GAM,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        was_training = bool(model.training)
        logger.info(
            "[GAM-EWCLoRA-lambda][InTaskFisher] Computing raw delta-W "
            "snapshot task=%d epoch=%d",
            self._cur_task,
            epoch_idx,
        )
        try:
            fisher = super()._compute_delta_fisher(train_loader, model, class_offset)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        fisher_snapshot = {
            key: value.detach().cpu().clone()
            for key, value in fisher.items()
        }
        lora_snapshot = self._capture_lora_weight_snapshot(model)
        self._in_task_fisher_snapshots.append((int(epoch_idx), fisher_snapshot, lora_snapshot))
        logger.info(
            "[GAM-EWCLoRA-lambda][InTaskFisher] Saved snapshot "
            "task=%d epoch=%d count=%d fisher_trace=%.6g lora_tensors=%d",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(fisher_snapshot),
            len(lora_snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    def _fisher_trace(self, fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        total = 0.0
        for value in fisher.values():
            total += float(value.detach().float().clamp_min(0).sum().item())
        return float(total)

    def _match_delta_tensor(self, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref = reference.detach().cpu()
        value_cpu = value.detach().cpu().to(dtype=ref.dtype)
        if tuple(value_cpu.shape) == tuple(ref.shape):
            return value_cpu

        matched = torch.zeros_like(ref)
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref.shape[dim])))
            for dim in range(min(value_cpu.ndim, ref.ndim))
        )
        if len(slices) == value_cpu.ndim == ref.ndim:
            matched[slices] = value_cpu[slices]
        return matched

    def _mean_in_task_fisher(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None

        mean_fisher: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            acc = None
            count = 0
            for _, fisher_snapshot, _ in self._in_task_fisher_snapshots:
                value = fisher_snapshot.get(key)
                if value is None:
                    continue
                value = self._match_delta_tensor(value, delta_now)
                acc = value.clone() if acc is None else acc + value
                count += 1
            if acc is not None and count > 0:
                mean_fisher[key] = acc / float(count)
        return mean_fisher or None

    def _mix_in_task_path_fisher(
        self,
        task_fisher: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        snapshot_epochs = [epoch for epoch, _, _ in self._in_task_fisher_snapshots]
        path_fisher = self._mean_in_task_fisher(model)
        final_trace = self._fisher_trace(task_fisher)
        path_trace = self._fisher_trace(path_fisher)
        lora_snapshot_tensors = sum(
            len(lora_snapshot)
            for _, _, lora_snapshot in self._in_task_fisher_snapshots
        )

        use_path = (
            self._approximate_fisher_in_task
            and self._in_task_fisher_omega > 0.0
            and path_fisher is not None
        )
        if not use_path:
            self._in_task_fisher_last_metrics = {
                "approximte_fisher_inTask": bool(self._approximate_fisher_in_task),
                "in_task_fisher_used_path": False,
                "in_task_fisher_interval": int(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(self._in_task_fisher_omega),
                "in_task_fisher_snapshots": int(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
                "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
                "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
            }
            return task_fisher

        omega = float(self._in_task_fisher_omega)
        mixed: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            final_value = task_fisher.get(key)
            path_value = path_fisher.get(key)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[key] = self._match_delta_tensor(path_value, delta_now).clone()
            elif path_value is None:
                mixed[key] = self._match_delta_tensor(final_value, delta_now).clone()
            else:
                final_t = self._match_delta_tensor(final_value, delta_now)
                path_t = self._match_delta_tensor(path_value, delta_now)
                mixed[key] = (1.0 - omega) * final_t + omega * path_t

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "approximte_fisher_inTask": bool(self._approximate_fisher_in_task),
            "in_task_fisher_used_path": True,
            "in_task_fisher_interval": int(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": int(len(snapshot_epochs)),
            "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
            "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
            "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
        }
        logger.info(
            "[GAM-EWCLoRA-lambda][InTaskFisher] Mixed final/path raw Fisher "
            "task=%d snapshots=%d omega=%.3g final_trace=%.6g path_trace=%.6g "
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

    # ------------------------------------------------------------------
    # Dual ascent and drift metrics
    # ------------------------------------------------------------------
    def _sync_lambda(self, value: float) -> None:
        self._ewc_lambda = float(value)
        self._lambda_ewc = float(value)

    def _get_current_base_lr(self, optimizer=None) -> float:
        if optimizer is not None:
            base_opt = getattr(optimizer, "base_optimizer", optimizer)
            if hasattr(base_opt, "param_groups") and len(base_opt.param_groups) > 0:
                lr_values = []
                for group in base_opt.param_groups:
                    lr = float(group.get("lr", 0.0))
                    if lr > 0.0:
                        lr_values.append(lr)
                if lr_values:
                    return max(lr_values)
        return float(getattr(self, "_last_base_lr", 0.0))

    @torch.no_grad()
    def _current_effective_fisher_max(self) -> float:
        if not self._fisher_past_delta:
            return 0.0

        f_max = 0.0
        for fisher in self._fisher_past_delta.values():
            if fisher is None or fisher.numel() == 0:
                continue
            value = float(fisher.detach().float().clamp_min(0).max().item())
            f_max = max(f_max, value)

        if self._dual_use_eta and self._fisher_eta != 0.0:
            f_max += float(self._fisher_eta)
        return float(f_max)

    def _compute_safe_lambda_bounds(self, optimizer=None) -> Tuple[float, float, Dict[str, float]]:
        static_min = float(self._dual_lambda_min)
        static_max = float(self._dual_lambda_max)

        if not self._safe_lambda_cap:
            return static_min, static_max, {
                "safe_lambda_enabled": 0.0,
                "safe_lambda_min_eff": static_min,
                "safe_lambda_max_eff": static_max,
                "safe_lambda_raw_max": static_max,
                "safe_lambda_fmax": 0.0,
                "safe_lambda_lr": 0.0,
                "safe_lambda_c": float(self._safe_lambda_c),
                "safe_lambda_min_overridden": 0.0,
            }

        lr = self._get_current_base_lr(optimizer)
        f_max = self._current_effective_fisher_max()
        if lr <= 0.0 or f_max <= 0.0:
            raw_safe_max = static_max
        else:
            raw_safe_max = float(self._safe_lambda_c) / (
                lr * (f_max + float(self._safe_lambda_eps))
            )

        capped_max = min(static_max, float(raw_safe_max))
        min_overridden = 0.0
        if capped_max < static_min:
            if self._safe_lambda_override_min:
                effective_min = capped_max
                effective_max = capped_max
                min_overridden = 1.0
            else:
                effective_min = static_min
                effective_max = static_min
        else:
            effective_min = static_min
            effective_max = capped_max

        self._last_safe_lambda_max = float(effective_max)
        return float(effective_min), float(effective_max), {
            "safe_lambda_enabled": 1.0,
            "safe_lambda_min_eff": float(effective_min),
            "safe_lambda_max_eff": float(effective_max),
            "safe_lambda_raw_max": float(raw_safe_max),
            "safe_lambda_fmax": float(f_max),
            "safe_lambda_lr": float(lr),
            "safe_lambda_c": float(self._safe_lambda_c),
            "safe_lambda_min_overridden": float(min_overridden),
        }

    def _clip_lambda_by_joint_bounds(
        self,
        lambda_value: float,
        optimizer=None,
        reason: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        effective_min, effective_max, stats = self._compute_safe_lambda_bounds(optimizer)
        old_value = float(lambda_value)
        clipped_value = min(effective_max, max(effective_min, old_value))
        stats.update({
            "safe_lambda_before_clip": old_value,
            "safe_lambda_after_clip": float(clipped_value),
            "safe_lambda_clipped": float(abs(clipped_value - old_value) > 0.0),
        })

        if abs(clipped_value - old_value) > 0.0 and self._is_main_process:
            self._log(
                f"[GAM-EWCLoRA-lambda][JointLambdaClip] "
                f"reason={reason} lambda {old_value:.6g} -> {clipped_value:.6g} "
                f"bounds=[{effective_min:.6g}, {effective_max:.6g}] "
                f"raw_safe_max={stats['safe_lambda_raw_max']:.6g} "
                f"F_max={stats['safe_lambda_fmax']:.6g} "
                f"lr={stats['safe_lambda_lr']:.6g} "
                f"c={stats['safe_lambda_c']:.6g}"
            )
        return float(clipped_value), stats

    def _apply_joint_lambda_clip(self, optimizer=None, reason: str = "") -> Dict[str, float]:
        old_lambda = float(self._ewc_lambda)
        new_lambda, stats = self._clip_lambda_by_joint_bounds(
            old_lambda,
            optimizer=optimizer,
            reason=reason,
        )
        self._sync_lambda(new_lambda)
        stats.update({
            "lambda_before_joint_clip": old_lambda,
            "lambda_after_joint_clip": float(new_lambda),
        })
        return stats

    @torch.no_grad()
    def _compute_fisher_drift_value(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher_past_delta or not self._delta_reference:
            return torch.zeros((), device=self._device)

        drift = torch.zeros((), device=self._device)
        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
            if self._dual_use_eta and self._fisher_eta != 0.0:
                fisher_t = fisher_t + self._fisher_eta
            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            drift = drift + 0.5 * (fisher_t * (delta_now.detach() - delta_ref_t).pow(2)).sum()
        return drift

    def _reset_dual_controller_state(self) -> None:
        self._dual_step_count = 0

    @torch.no_grad()
    def _compute_joint_drift_values(self, model: nn.Module) -> Dict[str, float]:
        """Compute Fisher, Euclidean, and F_max-scaled Euclidean drift.

            D_F = 0.5 * sum(F_eff * delta^2)
            D_E = ||delta||_2^2
            D_M = F_max * ||delta||_2^2
        """
        if not self._fisher_past_delta or not self._delta_reference:
            return {"fisher": 0.0, "euclidean": 0.0, "spectral": 0.0, "fmax": 0.0}

        fisher_drift = 0.0
        euclidean_drift = 0.0
        f_max = 0.0

        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
            if self._dual_use_eta and self._fisher_eta != 0.0:
                fisher_t = fisher_t + self._fisher_eta
            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            update_sq = (delta_now.detach() - delta_ref_t).pow(2)

            fisher_drift += float((0.5 * fisher_t * update_sq).sum().item())
            euclidean_drift += float(update_sq.sum().item())
            if fisher_t.numel() > 0:
                f_max = max(f_max, float(fisher_t.max().item()))

        return {
            "fisher": float(fisher_drift),
            "euclidean": float(euclidean_drift),
            "spectral": float(f_max * euclidean_drift),
            "fmax": float(f_max),
        }

    def _default_dual_constraint_stats(self) -> Dict[str, float]:
        return {
            "dual_fisher_drift": 0.0,
            "dual_fisher_budget": float(self._dual_budget),
            "dual_fisher_rho_budget": float(self._dual_rho * self._dual_budget),
            "dual_fisher_ratio": 0.0,
            "dual_fisher_violation": 0.0,
            "dual_euclidean_drift": 0.0,
            "dual_euclidean_budget": float(self._dual_euclidean_budget or 0.0),
            "dual_euclidean_rho_budget": float(self._dual_rho * (self._dual_euclidean_budget or 0.0)),
            "dual_euclidean_ratio": 0.0,
            "dual_euclidean_violation": 0.0,
            "dual_euclidean_active": float(
                self._dual_use_joint_constraints and self._dual_euclidean_budget is not None
            ),
            "dual_spectral_drift": 0.0,
            "dual_spectral_budget": float(self._dual_spectral_budget or 0.0),
            "dual_spectral_rho_budget": float(self._dual_rho * (self._dual_spectral_budget or 0.0)),
            "dual_spectral_ratio": 0.0,
            "dual_spectral_violation": 0.0,
            "dual_spectral_active": float(
                self._dual_use_joint_constraints and self._dual_spectral_budget is not None
            ),
            "dual_spectral_fmax": 0.0,
            "dual_max_ratio": 0.0,
            "dual_critical_constraint": 0.0,
        }

    def _compute_dual_constraint_stats(self, model: nn.Module) -> Dict[str, float]:
        values = self._compute_joint_drift_values(model)

        fisher_budget = float(self._dual_budget)
        fisher_ratio = float(values["fisher"] / (fisher_budget + self._dual_eps))
        constraints = [(fisher_ratio, 1.0)]

        euclidean_active = bool(
            self._dual_use_joint_constraints and self._dual_euclidean_budget is not None
        )
        euclidean_budget = float(self._dual_euclidean_budget or 0.0)
        euclidean_ratio = (
            float(values["euclidean"] / (euclidean_budget + self._dual_eps))
            if euclidean_active
            else 0.0
        )
        if euclidean_active:
            constraints.append((euclidean_ratio, 2.0))

        spectral_active = bool(
            self._dual_use_joint_constraints and self._dual_spectral_budget is not None
        )
        spectral_budget = float(self._dual_spectral_budget or 0.0)
        spectral_ratio = (
            float(values["spectral"] / (spectral_budget + self._dual_eps))
            if spectral_active
            else 0.0
        )
        if spectral_active:
            constraints.append((spectral_ratio, 3.0))

        max_ratio, critical_constraint = max(constraints, key=lambda item: item[0])

        return {
            "dual_fisher_drift": float(values["fisher"]),
            "dual_fisher_budget": fisher_budget,
            "dual_fisher_rho_budget": float(self._dual_rho * fisher_budget),
            "dual_fisher_ratio": fisher_ratio,
            "dual_fisher_violation": float(fisher_ratio - 1.0),
            "dual_euclidean_drift": float(values["euclidean"]),
            "dual_euclidean_budget": euclidean_budget,
            "dual_euclidean_rho_budget": float(self._dual_rho * euclidean_budget),
            "dual_euclidean_ratio": euclidean_ratio,
            "dual_euclidean_violation": float(euclidean_ratio - 1.0) if euclidean_active else 0.0,
            "dual_euclidean_active": float(euclidean_active),
            "dual_spectral_drift": float(values["spectral"]),
            "dual_spectral_budget": spectral_budget,
            "dual_spectral_rho_budget": float(self._dual_rho * spectral_budget),
            "dual_spectral_ratio": spectral_ratio,
            "dual_spectral_violation": float(spectral_ratio - 1.0) if spectral_active else 0.0,
            "dual_spectral_active": float(spectral_active),
            "dual_spectral_fmax": float(values["fmax"]),
            "dual_max_ratio": float(max_ratio),
            "dual_critical_constraint": float(critical_constraint),
        }

    def _default_dual_stats(self, ewc_ratio: float = 0.0, optimizer=None) -> Dict[str, float]:
        del ewc_ratio
        budget = float(self._dual_budget)
        _, _, safe_stats = self._compute_safe_lambda_bounds(optimizer)
        return {
            "dual_drift": 0.0,
            "dual_budget": budget,
            "dual_rho_budget": float(self._dual_rho * budget),
            "dual_violation": 0.0,
            "lambda_before": float(self._ewc_lambda),
            "lambda_after": float(self._ewc_lambda),
            "dual_updated": 0.0,
            "dual_mode": 0.0,
            **self._default_dual_constraint_stats(),
            **safe_stats,
        }

    def _maybe_update_dual_lambda(
        self,
        model: nn.Module,
        current_loss: float,
        current_ewc_ratio: float = 0.0,
        optimizer: Optional[GAM] = None,
    ) -> Dict[str, float]:
        del current_loss, current_ewc_ratio
        default_stats = self._default_dual_stats(optimizer=optimizer)
        if not self._dual_ascent or not self._fisher_past_delta or not self._delta_reference:
            return default_stats

        self._dual_step_count += 1
        if self._dual_step_count % self._dual_update_interval != 0:
            return default_stats
        return self._update_dual_lambda(model, optimizer=optimizer)

    def _update_dual_lambda(
        self,
        model: nn.Module,
        optimizer: Optional[GAM] = None,
    ) -> Dict[str, float]:
        """Version B dual ascent plus Fisher-aware stability clipping.

        Proposes lambda from the worst normalized active drift constraint
        (Fisher D_F, Euclidean D_E, or Spectral D_M), then clips to the
        stability range constrained by lr * lambda * F_max <= c.
        """
        if not self._dual_ascent or not self._fisher_past_delta or not self._delta_reference:
            return self._default_dual_stats(optimizer=optimizer)

        constraint_stats = self._compute_dual_constraint_stats(model)
        drift_value = float(constraint_stats["dual_fisher_drift"])
        budget = float(self._dual_budget)
        rho_budget = self._dual_rho * budget
        max_ratio = float(constraint_stats["dual_max_ratio"])

        old_lambda = float(self._ewc_lambda)
        base = old_lambda if old_lambda > 0.0 else max(float(self._dual_lambda_min), self._dual_eps)
        lambda_proposal = old_lambda
        updated = 0.0
        mode = 0.0

        if max_ratio > 1.0:
            # Any active constraint is over budget: respond to the worst normalized violation.
            rel = max_ratio - 1.0
            lambda_proposal = base * float(np.exp(np.clip(self._dual_eta_up * rel, -50.0, 50.0)))
            updated = 1.0
            mode = 1.0
        elif max_ratio < self._dual_rho:
            # All active constraints are well under budget: relax EWC.
            rel = 1.0 - max_ratio
            lambda_proposal = base * float(np.exp(np.clip(-self._dual_eta_down * rel, -50.0, 50.0)))
            updated = 1.0
            mode = -1.0
        # else: dead-band [ρ, 1] on the worst normalized constraint.

        new_lambda, safe_stats = self._clip_lambda_by_joint_bounds(
            lambda_proposal,
            optimizer=optimizer,
            reason="dual_update",
        )
        self._sync_lambda(new_lambda)
        violation = drift_value / (budget + self._dual_eps) - 1.0
        return {
            "dual_drift": drift_value,
            "dual_budget": budget,
            "dual_rho_budget": rho_budget,
            "dual_violation": float(max_ratio - 1.0),
            "dual_fisher_legacy_violation": violation,
            "lambda_before": old_lambda,
            "lambda_proposal": float(lambda_proposal),
            "lambda_after": float(new_lambda),
            "dual_updated": updated,
            "dual_mode": mode,
            **constraint_stats,
            **safe_stats,
        }

    def _compute_delta_drift_metrics(self) -> Dict[str, float]:
        if not self._fisher_past_delta or not self._delta_reference:
            return {
                "delta_drift_available": 0,
                "delta_tensors": 0,
                "delta_norm": 0.0,
                "fisher_weighted_drift": 0.0,
                "fisher_eta_weighted_drift": 0.0,
                "normalized_fisher_drift": 0.0,
                "normalized_fisher_eta_drift": 0.0,
                "high_fisher_delta_energy_ratio": 0.0,
                "high_fisher_weighted_energy_ratio": 0.0,
                "ewc_penalty_value": 0.0,
            }

        model = self._unwrap_network()
        eps = float(self._mechanism_eps)
        delta_norm = 0.0
        fisher_weighted = 0.0
        fisher_eta_weighted = 0.0
        high_delta_energy = 0.0
        high_weighted_energy = 0.0
        fisher_sum = 0.0
        fisher_max = 0.0
        fisher_numel = 0
        tensor_count = 0

        with torch.no_grad():
            for key, delta_now in self._iter_delta_terms(model):
                fisher = self._fisher_past_delta.get(key)
                delta_ref = self._delta_reference.get(key)
                if fisher is None or delta_ref is None:
                    continue

                fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
                delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
                update_sq = (delta_now.detach() - delta_ref_t).pow(2)
                fisher_update = fisher_t * update_sq

                delta_norm += float(update_sq.sum().item())
                fisher_weighted += float(fisher_update.sum().item())
                fisher_eta_weighted += float(((fisher_t + self._fisher_eta) * update_sq).sum().item())
                fisher_sum += float(fisher_t.sum().item())
                fisher_max = max(fisher_max, float(fisher_t.max().item()) if fisher_t.numel() else 0.0)
                fisher_numel += int(fisher_t.numel())
                tensor_count += 1

                flat_fisher = fisher_t.reshape(-1)
                flat_update_sq = update_sq.reshape(-1)
                if flat_fisher.numel() > 0:
                    if self._mechanism_topk is not None:
                        k = min(max(int(self._mechanism_topk), 1), int(flat_fisher.numel()))
                    else:
                        frac = min(max(float(self._mechanism_top_fraction), 0.0), 1.0)
                        k = min(max(int(round(float(flat_fisher.numel()) * frac)), 1), int(flat_fisher.numel()))
                    top_idx = torch.topk(flat_fisher, k=k, largest=True, sorted=False).indices
                    high_delta_energy += float(flat_update_sq[top_idx].sum().item())
                    high_weighted_energy += float((flat_fisher[top_idx] * flat_update_sq[top_idx]).sum().item())

        return {
            "delta_drift_available": 1,
            "delta_tensors": int(tensor_count),
            "delta_norm": float(delta_norm),
            "fisher_weighted_drift": float(fisher_weighted),
            "fisher_eta_weighted_drift": float(fisher_eta_weighted),
            "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
            "normalized_fisher_eta_drift": float(fisher_eta_weighted / (delta_norm + eps)),
            "high_fisher_delta_energy_ratio": float(high_delta_energy / (delta_norm + eps)),
            "high_fisher_weighted_energy_ratio": float(high_weighted_energy / (fisher_weighted + eps)),
            "fisher_mean": float(fisher_sum / max(fisher_numel, 1)),
            "fisher_max": float(fisher_max),
            "ewc_penalty_value": float(0.5 * self._ewc_lambda * fisher_eta_weighted),
        }

    # ------------------------------------------------------------------
    # Mechanism summaries and explicit Fisher update
    # ------------------------------------------------------------------
    def _reset_mechanism_stats(self) -> None:
        self._mechanism_sums = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics = {}

    def _snapshot_task_reference(self) -> None:
        super()._snapshot_task_reference()
        if self._mechanism_eval:
            self._reset_mechanism_stats()

    def _record_mechanism_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            try:
                self._mechanism_sums[key] = self._mechanism_sums.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        self._mechanism_count += 1

    def _summarize_mechanism_stats(self) -> Dict[str, float]:
        if self._mechanism_count <= 0:
            return {}
        denom = float(self._mechanism_count)
        return {
            f"train_{key}_mean": value / denom
            for key, value in sorted(self._mechanism_sums.items())
        }

    def after_task(self):
        drift_before = (
            self._compute_delta_drift_metrics()
            if (self._mechanism_eval or self._dual_ascent)
            else {}
        )
        base_model = self._unwrap_network()
        new_fisher = super()._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        new_fisher = self._mix_in_task_path_fisher(new_fisher, base_model)
        self._fisher_past_delta = super()._accumulate_fisher(self._fisher_past_delta, new_fisher)
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
                    f"[GAM-EWCLoRA-lambda][FisherStats] task={self._cur_task} "
                    f"new_fisher tensors={len(f_vals)} numel={nf_numel} "
                    f"max={nf_max:.3e} mean={nf_mean:.3e} sum={nf_sum:.3e}"
                )
            accum_vals = list(self._fisher_past_delta.values())
            if accum_vals:
                af_max = float(max(v.max().item() for v in accum_vals))
                af_sum = float(sum(v.sum().item() for v in accum_vals))
                af_numel = sum(int(v.numel()) for v in accum_vals)
                af_mean = af_sum / max(af_numel, 1)
                self._log(
                    f"[GAM-EWCLoRA-lambda][FisherStats] task={self._cur_task} "
                    f"accum_fisher tensors={len(accum_vals)} numel={af_numel} "
                    f"max={af_max:.3e} mean={af_mean:.3e} sum={af_sum:.3e}"
                )
            self._log(
                "[GAM-EWCLoRA-lambda] Updated raw delta-W Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}."
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
                "method_type": "gam_ewclora_lambda",
                "lambda_ewc": float(self._ewc_lambda),
                "gam_base_optimizer": self._gam_base_optimizer,
                "gam_grad_rho": float(self._gam_grad_rho),
                "gam_grad_norm_rho": float(self._gam_grad_norm_rho),
                "ewc_dual_ascent": bool(self._dual_ascent),
                "ewc_dual_delta": float(self._dual_budget),
                "ewc_dual_relative": bool(self._dual_relative),
                "ewc_dual_update_by_epoch": bool(self._dual_update_by_epoch),
                "ewc_dual_update_interval": int(self._dual_update_interval),
                "ewc_dual_eta_up": float(self._dual_eta_up),
                "ewc_dual_eta_down": float(self._dual_eta_down),
                "ewc_dual_lambda_min": float(self._dual_lambda_min),
                "ewc_dual_lambda_max": float(self._dual_lambda_max),
                "ewc_dual_use_joint_constraints": bool(self._dual_use_joint_constraints),
                "ewc_dual_euclidean_delta": float(self._dual_euclidean_budget or 0.0),
                "ewc_dual_spectral_delta": float(self._dual_spectral_budget or 0.0),
                "ewc_safe_lambda_cap": bool(self._safe_lambda_cap),
                "ewc_safe_lambda_c": float(self._safe_lambda_c),
                "ewc_safe_lambda_eps": float(self._safe_lambda_eps),
                "ewc_safe_lambda_override_min": bool(self._safe_lambda_override_min),
                "ewc_safe_lambda_last_max": float(self._last_safe_lambda_max),
                "gam_ewc_euclidean_trust": bool(self._gam_ewc_euclidean_trust),
                "gam_ewc_euclidean_radius": float(
                    self._resolve_gam_ewc_euclidean_radius() or 0.0
                ),
                "gam_ewc_euclidean_radius_from_gam": bool(
                    self._gam_ewc_euclidean_radius_from_gam
                ),
                "gam_ewc_euclidean_radius_source": self._gam_ewc_euclidean_radius_source,
                "gam_ewc_euclidean_radius_scale": float(
                    self._gam_ewc_euclidean_radius_scale
                ),
                "gam_ewc_euclidean_outside_action": self._gam_ewc_euclidean_outside_action,
                "gam_ewc_euclidean_exclude_classifier": bool(
                    self._gam_ewc_euclidean_exclude_classifier
                ),
                "gam_ewc_delta_penalty": bool(self._gam_ewc_delta_penalty),
                "gam_ewc_delta_penalty_lambda": float(
                    self._gam_ewc_delta_penalty_lambda
                ),
                "gam_ewc_delta_penalty_radius": float(
                    self._resolve_gam_ewc_delta_penalty_radius() or 0.0
                ),
                "gam_ewc_delta_penalty_radius_from_gam": bool(
                    self._gam_ewc_delta_penalty_radius_from_gam
                ),
                "gam_ewc_delta_penalty_radius_source": (
                    self._gam_ewc_delta_penalty_radius_source
                ),
                "gam_ewc_delta_penalty_radius_scale": float(
                    self._gam_ewc_delta_penalty_radius_scale
                ),
                "gam_ewc_delta_penalty_exclude_classifier": bool(
                    self._gam_ewc_delta_penalty_exclude_classifier
                ),
                **self._in_task_fisher_last_metrics,
                **self._summarize_mechanism_stats(),
                **drift_before,
                **full_fisher_diff_metrics,
            }
            if self._is_main_process:
                self._log(f"[GAM-EWCLoRA-lambda][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, float]:
        return dict(self._last_mechanism_metrics)


GAMEWCLoRALambda = Learner
EWCLoRA = Learner
