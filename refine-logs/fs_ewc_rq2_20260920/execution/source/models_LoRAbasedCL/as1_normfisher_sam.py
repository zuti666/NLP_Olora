"""CE-only SAM reference for AS1 trace-normalized delta-W EWC.

This learner is the direct-SAM counterpart of
``as1_normfisher_fishershape``.  It intentionally uses the same AS1
delta-W Fisher machinery and the same clean-point additive EWC gradient
injection, but the current-task shaping direction is ordinary SAM:

    g_sam = grad CE(theta + rho * g0 / ||g0||)
    d     = g_sam + g_EWC(theta)

The key distinction from the older ``ewclora_youyue_fitArchitechture``
``optimizer_type=sam`` path is that EWC is not included in either SAM
closure.  The perturbation and the second gradient are CE-only; the EWC
gradient is evaluated once at the clean point and added before the base
optimizer update.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from torch import optim

from models_LoRAbasedCL.as1_normfisher_gam import Learner as _AS1NormFisherGAM
from optimer_PerturabtionType.optimer_sam import SAM
from optimer_PerturabtionType.optimer_sam import logger as _sam_optimizer_logger
from optimer_PerturabtionType.util import disable_running_stats, enable_running_stats
from utils.mechanism_eval import as_bool


logger = logging.getLogger(__name__)


class Learner(_AS1NormFisherGAM):
    """AS1 delta-W EWC combined with CE-only factor-space SAM."""

    _OPTIMIZER_ALIASES = {
        "sam",
        "sam_ref",
        "ceonly_sam",
        "sam_ceonly",
    }

    def __init__(self, args):
        if "optimizer_type" not in args:
            args["optimizer_type"] = "sam"
        super().__init__(args)

        self._sam_rho = float(args.get("sam_rho", args.get("rho", 0.05)))
        self._sam_adaptive = as_bool(args.get("sam_adaptive", False))
        self._sam_base_optimizer = str(
            args.get("sam_base_optimizer", args.get("optimizer", "sgd"))
        ).lower()

        fisher_damping_raw = float(
            args.get(
                "ewc_fisher_damping_beta",
                args.get(
                    "ewc_fisher_damping_alpha",
                    args.get("fisher_damping_alpha", 0.0),
                ),
            )
        )
        self._ewc_fisher_damping_beta = max(0.0, min(1.0, fisher_damping_raw))
        self._last_task_fisher_damping_metrics: Dict[str, float] = {}

        if self._optimizer_type not in self._OPTIMIZER_ALIASES:
            raise ValueError(
                "as1_normfisher_sam requires optimizer_type in "
                f"{sorted(self._OPTIMIZER_ALIASES)}, got {self._optimizer_type!r}."
            )

        # optimer_sam.py configures a DEBUG module logger at import time.  Keep
        # long ImageNet-A runs readable while still using the shared SAM class.
        _sam_optimizer_logger.setLevel(logging.WARNING)

        if self._ewc_fisher_damping_beta > 0.0 and not self._normalize_fisher:
            logger.warning(
                "[AS1-SAM-LoRA] ewc_fisher_damping_beta=%.4g is intended for "
                "trace-normalized per-task delta-W Fisher, but "
                "ewc_normalize_fisher=False.",
                self._ewc_fisher_damping_beta,
            )

        if self._is_main_process:
            logger.info(
                "[AS1-SAM-LoRA] rho=%.4g adaptive=%s base=%s lambda=%.4g "
                "gamma=%.4g normalize_fisher=%s normalize_accumulated=%s "
                "fisher_damping_beta=%.4g ewc_in_sam_closure=False",
                self._sam_rho,
                self._sam_adaptive,
                self._sam_base_optimizer,
                self._lambda_ewc,
                self._ewc_gamma,
                self._normalize_fisher,
                self._normalize_accumulated_fisher,
                self._ewc_fisher_damping_beta,
            )

    # ------------------------------------------------------------------
    # Per-task delta-W Fisher damping, after AS1 trace normalization.
    # ------------------------------------------------------------------
    @staticmethod
    def _fisher_dict_numel(fisher: Dict[str, torch.Tensor]) -> int:
        return int(sum(int(v.numel()) for v in fisher.values()))

    @staticmethod
    def _fisher_dict_max(fisher: Dict[str, torch.Tensor]) -> float:
        vals = [v.detach().float().max().item() for v in fisher.values() if v.numel()]
        return float(max(vals)) if vals else 0.0

    def _damp_task_fisher(
        self,
        fisher: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        beta = float(getattr(self, "_ewc_fisher_damping_beta", 0.0))
        self._last_task_fisher_damping_metrics = {
            "ewc_fisher_damping_beta": beta,
            "task_fisher_damping_applied": False,
        }
        if beta <= 0.0 or not fisher:
            return fisher

        total_numel = self._fisher_dict_numel(fisher)
        if total_numel <= 0:
            return fisher

        trace_before = self._fisher_trace(fisher)
        max_before = self._fisher_dict_max(fisher)
        iso = beta / float(total_numel)
        damped = {key: (1.0 - beta) * value + iso for key, value in fisher.items()}
        trace_after = self._fisher_trace(damped)
        max_after = self._fisher_dict_max(damped)
        self._last_task_fisher_damping_metrics = {
            "ewc_fisher_damping_beta": beta,
            "task_fisher_damping_applied": True,
            "task_fisher_damping_numel": float(total_numel),
            "task_fisher_damping_iso": float(iso),
            "task_fisher_trace_before_damping": float(trace_before),
            "task_fisher_trace_after_damping": float(trace_after),
            "task_fisher_max_before_damping": float(max_before),
            "task_fisher_max_after_damping": float(max_after),
        }
        if self._is_main_process:
            logger.info(
                "[AS1-SAM-LoRA] Task delta-W Fisher damped beta=%.4g numel=%d "
                "trace %.4e -> %.4e max %.4e -> %.4e iso=%.4e",
                beta,
                total_numel,
                trace_before,
                trace_after,
                max_before,
                max_after,
                iso,
            )
        return damped

    def _compute_delta_fisher(self, loader, model, class_offset: int) -> Dict[str, torch.Tensor]:
        task_fisher = super()._compute_delta_fisher(loader, model, class_offset)
        return self._damp_task_fisher(task_fisher)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        if not isinstance(params, (list, tuple)):
            params = list(params)

        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        if self._sam_base_optimizer == "adam":
            base_optimizer = optim.Adam
            kwargs = {
                "lr": lr,
                "betas": (
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                "weight_decay": weight_decay,
            }
        elif self._sam_base_optimizer == "adamw":
            base_optimizer = optim.AdamW
            kwargs = {
                "lr": lr,
                "betas": (
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                "weight_decay": weight_decay,
            }
        else:
            base_optimizer = optim.SGD
            kwargs = {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            }

        return SAM(
            params,
            base_optimizer,
            rho=self._sam_rho,
            adaptive=self._sam_adaptive,
            **kwargs,
        )

    @staticmethod
    def _grad_norm_sq(params: List[torch.nn.Parameter]) -> float:
        total = 0.0
        for param in params:
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum().item())
        return float(total)

    @staticmethod
    def _restore_sam_params(optimizer: SAM) -> None:
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    old_param = optimizer.state[param].get("old_p", None)
                    if old_param is not None:
                        param.data.copy_(old_param)

    def _add_fisher_grad_to_current_grads(
        self,
        params: List[torch.nn.Parameter],
        fisher_grad_cache: Dict[int, torch.Tensor],
        *,
        clean_norm_sq: float,
    ) -> Dict[str, float]:
        collect = bool(self._mechanism_grad_stats)
        eps = float(self._mechanism_eps)
        sam_norm_sq = 0.0
        fisher_norm_sq = 0.0
        combined_norm_sq = 0.0
        sam_fisher_dot = 0.0

        with torch.no_grad():
            for param in params:
                if not param.requires_grad:
                    continue

                sam_grad = None if param.grad is None else param.grad.detach().clone()
                fisher_grad = fisher_grad_cache.get(id(param))
                if fisher_grad is not None:
                    fisher_grad = fisher_grad.to(device=param.device, dtype=param.dtype)

                if sam_grad is None and fisher_grad is None:
                    continue
                if sam_grad is None:
                    combined = fisher_grad.clone()
                elif fisher_grad is None:
                    combined = sam_grad
                else:
                    combined = sam_grad + fisher_grad

                if param.grad is None:
                    param.grad = combined.clone()
                else:
                    param.grad.copy_(combined)

                if collect:
                    if sam_grad is not None:
                        sam_norm_sq += float(sam_grad.pow(2).sum().item())
                    if fisher_grad is not None:
                        fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                    if sam_grad is not None and fisher_grad is not None:
                        sam_fisher_dot += float((sam_grad * fisher_grad).sum().item())
                    combined_norm_sq += float(combined.pow(2).sum().item())

        if not collect:
            return {}

        clean_norm = clean_norm_sq ** 0.5
        sam_norm = sam_norm_sq ** 0.5
        fisher_norm = fisher_norm_sq ** 0.5
        return {
            "clean_norm": float(clean_norm),
            "sam_dir_norm": float(sam_norm),
            "fisher_norm": float(fisher_norm),
            "combined_norm": float(combined_norm_sq ** 0.5),
            "sam_to_clean_norm_ratio": float(sam_norm / (clean_norm + eps)),
            "fisher_to_sam_norm_ratio": float(fisher_norm / (sam_norm + eps)),
            "cos_sam_fisher": self._cosine_from_sums(
                sam_fisher_dot,
                sam_norm_sq,
                fisher_norm_sq,
                eps,
            ),
        }

    # ------------------------------------------------------------------
    # One CE-only direct-SAM batch step.
    # ------------------------------------------------------------------
    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        if not isinstance(optimizer, SAM):
            raise TypeError(
                "as1_normfisher_sam requires a SAM optimizer, "
                f"got {type(optimizer).__name__}"
            )

        params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)

        # Clean-point delta-W EWC pullback.  This never enters the SAM
        # perturbation closure or the perturbed CE gradient.
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)

        optimizer.zero_grad()
        outputs_clean = self._network(inputs)
        clean_loss, _, _ = self._task_loss(
            outputs_clean["logits"],
            targets,
            class_offset,
        )
        clean_loss_value = float(clean_loss.detach().item())
        clean_loss.backward()
        clean_norm_sq = self._grad_norm_sq(params)

        optimizer.first_step(zero_grad=True)
        restored = False
        # Freeze normalization running stats for the perturbed pass, matching
        # FisherShape's pass-2 semantics (no-op for the BN-free ViT-LoRA path).
        disable_running_stats(self._network)
        try:
            outputs_sam = self._network(inputs)
            sam_loss, _, _ = self._task_loss(
                outputs_sam["logits"],
                targets,
                class_offset,
            )
            sam_loss_value = float(sam_loss.detach().item())
            sam_loss.backward()

            stats = self._add_fisher_grad_to_current_grads(
                params,
                fisher_grad_cache,
                clean_norm_sq=clean_norm_sq,
            )
            if stats:
                self._accumulate_mechanism_grad_stats(stats)

            # Restore all SAM-perturbed parameters, then run the wrapped base
            # optimizer with the CE-only SAM gradient plus clean-point EWC grad.
            self._restore_sam_params(optimizer)
            restored = True
            optimizer.base_optimizer.step()
            # The LR scheduler wraps the SAM object, but this learner drives
            # base_optimizer.step() directly; mark the wrapper as stepped so
            # torch's scheduler-ordering check doesn't warn.
            optimizer._opt_called = True
        finally:
            if not restored:
                self._restore_sam_params(optimizer)
            enable_running_stats(self._network)
            optimizer.zero_grad()

        self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)
        total_loss_value = clean_loss_value + float(fisher_loss_value)
        # Return the clean-point outputs so train-accuracy monitoring matches
        # the FisherShape learner (which reports pass-1 logits).
        return outputs_clean["logits"].detach(), float(total_loss_value)

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_sam",
                    "sam_ref": True,
                    "sam_space": "factor",
                    "sam_rho": float(self._sam_rho),
                    "sam_adaptive": bool(self._sam_adaptive),
                    "sam_base_optimizer": str(self._sam_base_optimizer),
                    "ewc_in_sam_closure": False,
                    **self._last_task_fisher_damping_metrics,
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
