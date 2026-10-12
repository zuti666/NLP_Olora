"""Factor-space FisherShape with momentum-decoupled AS1 EWC.

This is an experimental sibling of ``as1_normfisher_fishershape``.  It keeps
the original learner and FisherShape optimizer unchanged, but changes where
the already lambda-scaled EWC gradient is applied:

    v_t       = momentum(v_{t-1}, g_FS)
    theta_tmp = SGDStep(theta, v_t)
    theta_t   = theta_tmp - lr * g_EWC(theta)

The EWC gradient is evaluated at the same clean point as in the coupled
implementation.  It does not enter either CE-only FisherShape probe and,
unlike the coupled implementation, it is not stored in SGD's momentum
buffer.  With zero momentum and zero weight decay, one step is numerically
identical to adding ``g_EWC`` to ``g_FS`` before the optimizer step; subsequent
steps differ only through optimizer state.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch

from models_LoRAbasedCL.as1_normfisher_fishershape import (
    Learner as _CoupledFisherShapeLearner,
)
from optimer_PerturabtionType.fishershape import FisherShape


logger = logging.getLogger(__name__)


class Learner(_CoupledFisherShapeLearner):
    """AS1 FisherShape whose EWC correction bypasses SGD momentum."""

    def __init__(self, args):
        super().__init__(args)

        base_name = str(
            args.get(
                "fs_base_optimizer",
                args.get("gam_base_optimizer", args.get("optimizer", "sgd")),
            )
        ).lower()
        if base_name != "sgd":
            raise ValueError(
                "as1_normfisher_fishershape_decoupled currently supports only "
                "fs_base_optimizer=sgd. A post-step Adam correction would bypass "
                "Adam's preconditioner and has different optimization semantics."
            )

        self._fs_ewc_correction_lr_scale = float(
            args.get("fs_ewc_correction_lr_scale", 1.0)
        )
        if self._fs_ewc_correction_lr_scale < 0.0:
            raise ValueError("fs_ewc_correction_lr_scale must be non-negative")

        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA-Decoupled] ewc_update=post_step "
                "momentum_buffer=fs_only correction_lr_scale=%.4g",
                self._fs_ewc_correction_lr_scale,
            )

    def _synchronize_fisher_grad_cache(
        self,
        fisher_grad_cache: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Match FisherShape's DDP reduction before applying a separate update."""
        if not fisher_grad_cache or not torch.distributed.is_initialized():
            return fisher_grad_cache

        reduce_mode = str(self._fs_grad_reduce).lower()
        world_size = float(torch.distributed.get_world_size())
        synchronized: Dict[int, torch.Tensor] = {}
        for param_id, fisher_grad in fisher_grad_cache.items():
            grad = fisher_grad.detach().clone()
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            if reduce_mode == "mean":
                grad.div_(world_size)
            synchronized[param_id] = grad
        return synchronized

    def _build_stats_only_fisher_projector(
        self,
        fisher_grad_cache: Dict[int, torch.Tensor],
        step_stats: Dict[str, float],
    ):
        """Measure the virtual FS+EWC direction without changing ``param.grad``."""
        collect = bool(self._mechanism_grad_stats)
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            if not collect:
                return

            fs_norm_sq = 0.0
            fisher_norm_sq = 0.0
            combined_norm_sq = 0.0
            fs_fisher_dot = 0.0

            with torch.no_grad():
                for group in param_groups:
                    for param in group["params"]:
                        if not param.requires_grad:
                            continue

                        fs_grad = None if param.grad is None else param.grad.detach()
                        fisher_grad = fisher_grad_cache.get(id(param))
                        if fisher_grad is not None:
                            fisher_grad = fisher_grad.to(
                                device=param.device,
                                dtype=param.dtype,
                            )

                        if fs_grad is None and fisher_grad is None:
                            continue
                        if fs_grad is None:
                            combined = fisher_grad
                        elif fisher_grad is None:
                            combined = fs_grad
                        else:
                            combined = fs_grad + fisher_grad

                        if fs_grad is not None:
                            fs_norm_sq += float(fs_grad.pow(2).sum().item())
                        if fisher_grad is not None:
                            fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                        if fs_grad is not None and fisher_grad is not None:
                            fs_fisher_dot += float((fs_grad * fisher_grad).sum().item())
                        combined_norm_sq += float(combined.pow(2).sum().item())

            fs_norm = fs_norm_sq ** 0.5
            fisher_norm = fisher_norm_sq ** 0.5
            step_stats.update(
                {
                    "fs_dir_norm": float(fs_norm),
                    "fisher_norm": float(fisher_norm),
                    "combined_norm": float(combined_norm_sq ** 0.5),
                    "fisher_to_fs_norm_ratio": float(
                        fisher_norm / (fs_norm + eps)
                    ),
                    "cos_fs_fisher": self._cosine_from_sums(
                        fs_fisher_dot,
                        fs_norm_sq,
                        fisher_norm_sq,
                        eps,
                    ),
                }
            )

        return projector

    @torch.no_grad()
    def _apply_decoupled_fisher_correction(
        self,
        optimizer: FisherShape,
        fisher_grad_cache: Dict[int, torch.Tensor],
        step_stats: Dict[str, float],
    ) -> None:
        """Apply ``-group_lr * scale * g_EWC`` without touching optimizer state."""
        scale = float(self._fs_ewc_correction_lr_scale)
        correction_norm_sq = 0.0
        min_lr = None
        max_lr = None

        for group in optimizer.param_groups:
            correction_lr = float(group["lr"]) * scale
            min_lr = correction_lr if min_lr is None else min(min_lr, correction_lr)
            max_lr = correction_lr if max_lr is None else max(max_lr, correction_lr)
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                fisher_grad = fisher_grad_cache.get(id(param))
                if fisher_grad is None:
                    continue
                fisher_grad = fisher_grad.to(device=param.device, dtype=param.dtype)
                param.add_(fisher_grad, alpha=-correction_lr)
                if self._mechanism_grad_stats:
                    correction_norm_sq += float(
                        (correction_lr * fisher_grad).pow(2).sum().item()
                    )

        if self._mechanism_grad_stats:
            step_stats.update(
                {
                    "ewc_correction_norm": float(correction_norm_sq ** 0.5),
                    "ewc_correction_lr_min": float(min_lr or 0.0),
                    "ewc_correction_lr_max": float(max_lr or 0.0),
                }
            )

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        if not isinstance(optimizer, FisherShape):
            raise TypeError(
                "as1_normfisher_fishershape_decoupled requires a FisherShape "
                f"optimizer, got {type(optimizer).__name__}"
            )

        params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)

        # The parent cache is the clean-point delta-W EWC pullback and already
        # includes lambda. It is kept outside both CE-only probe passes.
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        fisher_grad_cache = self._synchronize_fisher_grad_cache(fisher_grad_cache)
        projector_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_stats_only_fisher_projector(
                fisher_grad_cache,
                projector_stats,
            )
        )

        def closure():
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(
                outputs["logits"],
                targets,
                class_offset,
            )
            loss_value = task_loss.detach()
            task_loss.backward()
            return outputs, loss_value

        try:
            # The base optimizer sees only g_FS, so its momentum buffer cannot
            # accumulate EWC. The analytic correction is applied afterwards.
            outputs, clean_loss = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        self._apply_decoupled_fisher_correction(
            optimizer,
            fisher_grad_cache,
            projector_stats,
        )

        clean_loss_value = float(
            clean_loss.item() if torch.is_tensor(clean_loss) else clean_loss
        )
        self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)

        if self._mechanism_grad_stats:
            fs_stats = dict(getattr(optimizer, "last_stats", {}) or {})
            fs_stats.update(projector_stats)
            self._accumulate_mechanism_grad_stats(fs_stats)

        total_loss_value = clean_loss_value + float(fisher_loss_value)
        return outputs["logits"].detach(), float(total_loss_value)

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_fishershape_decoupled",
                    "ewc_update_mode": "post_step_decoupled",
                    "momentum_buffer_direction": "fs_only",
                    "fs_ewc_correction_lr_scale": float(
                        self._fs_ewc_correction_lr_scale
                    ),
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
