"""Fresh-LoRA FisherShape EWC with cross-task *factor* merging.

This learner is the factor-merge counterpart of
``as1_normfisher_fishershape_freshlora_merge``.  Training, the zero-delta EWC
anchor, FisherShape, Fisher damping/normalization, and (most importantly) the
task-end Fisher timing are unchanged.  Only the cross-task integration rule is
different.

Let ``(A_bar, B_bar)`` be the frozen factors accumulated from completed tasks
and ``(A_t, B_t)`` the fresh trainable adapter for task ``t``.  The forward
weight is

    W_eff = W_0 + B_bar @ A_bar + B_t @ A_t.

After the Fisher of ``B_t @ A_t`` has been computed, this learner applies the
same factor accumulation rule as the reference EWCLoRA implementation:

    A_bar <- A_bar + A_t
    B_bar <- B_bar + B_t

and reinitializes ``A_t``/``B_t`` for the next task.  It deliberately does not
apply ``W_0 <- W_0 + B_t @ A_t``.

Factor addition is not equivalent to adding delta weights.  Starting with the
second merge it introduces the cross terms ``B_bar @ A_t + B_t @ A_bar``.
Those terms are intentional here: this method exists to isolate and measure
the reference factor-merge lifecycle against the weight-merge Fresh method.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch
import torch.nn as nn

from models_LoRAbasedCL.as1_normfisher_fishershape_freshlora_merge import (
    Learner as _FreshWeightMergeLearner,
)
from models_LoRAbasedCL.as1_normfisher_gam import Learner as _AS1NormFisherGAM
from utils.mechanism_eval import as_bool


logger = logging.getLogger(__name__)


class _MergedLoRAFactors(nn.Module):
    """Frozen, checkpointable cumulative q/v LoRA factors for one block."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.register_buffer("a_q", torch.zeros_like(module.linear_a_q.weight))
        self.register_buffer("b_q", torch.zeros_like(module.linear_b_q.weight))
        self.register_buffer("a_v", torch.zeros_like(module.linear_a_v.weight))
        self.register_buffer("b_v", torch.zeros_like(module.linear_b_v.weight))
        self.register_buffer("merge_steps", torch.zeros((), dtype=torch.long))


def _inject_merged_factor_delta(module: nn.Module, _inputs) -> None:
    """Expose B_bar@A_bar to the existing LoRA forward before every call.

    A forward-pre-hook is used so DataParallel/DDP replicas always construct
    the historical delta from their own registered buffers.  The existing
    LoRA forward then adds this frozen delta to the explicit current ``B@A``
    tensor.  Consequently its delta hook still returns dL/d(delta_W_t).
    """

    state = getattr(module, "_merged_lora_factors", None)
    if state is None:
        return
    with torch.no_grad():
        module.external_delta_w_q = state.b_q @ state.a_q
        module.external_delta_w_v = state.b_v @ state.a_v


class Learner(_FreshWeightMergeLearner):
    """Fresh delta-W EWC whose completed adapters are merged in factor space."""

    _METHOD_NAME = "as1_normfisher_fishershape_freshlora_merge_factor"

    def __init__(self, args):
        super().__init__(args)
        self._last_factor_merge_metrics: Dict[str, float] = {}
        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA-MergeFactor] lifecycle=fresh_adapter_merge_factors "
                "cross_task_update=Abar+=At,Bbar+=Bt base_weight_merge=False "
                "fisher_timing=before_factor_merge ewc_anchor=zero_delta"
            )

    # ------------------------------------------------------------------
    # Frozen cumulative-factor state and forward integration
    # ------------------------------------------------------------------
    @staticmethod
    def _install_factor_state(module: nn.Module) -> None:
        if not hasattr(module, "_merged_lora_factors"):
            module.add_module("_merged_lora_factors", _MergedLoRAFactors(module))
        if not getattr(module, "_merge_factor_hook_installed", False):
            module.register_forward_pre_hook(_inject_merged_factor_delta)
            module._merge_factor_hook_installed = True
        _inject_merged_factor_delta(module, ())

    def _ensure_factor_accumulators(self, model: nn.Module) -> int:
        count = 0
        for _, module in self._iter_delta_modules(model):
            self._install_factor_state(module)
            count += 1
        return count

    def build_lora_backbone(self, index=True, eval_mode=False):
        backbone = super().build_lora_backbone(index=index, eval_mode=eval_mode)
        count = self._ensure_factor_accumulators(backbone)
        if count == 0:
            raise RuntimeError(
                f"{self._METHOD_NAME} found no q/v LoRA modules in the new backbone"
            )
        return backbone

    # ------------------------------------------------------------------
    # The only algorithmic change relative to Fresh weight merge
    # ------------------------------------------------------------------
    def _merge_current_lora_factors(self) -> Dict[str, float]:
        """Accumulate old/new A and B, then reset only the current adapter."""

        base_model = self._unwrap_network()
        modules = list(self._iter_delta_modules(base_model))
        if not modules:
            raise RuntimeError(
                f"{self._METHOD_NAME} found no factor-mergeable LoRA q/v modules"
            )
        self._ensure_factor_accumulators(base_model)

        # Preserve the same current-delta diagnostics as Fresh weight merge.
        self._last_premerge_drift_metrics = self._direct_delta_drift_metrics()

        current_norm_sq = 0.0
        accumulated_norm_sq = 0.0
        cross_term_norm_sq = 0.0
        factor_a_norm_sq = 0.0
        factor_b_norm_sq = 0.0
        merged_numel = 0
        merge_steps = 0

        with torch.no_grad():
            for _, module in modules:
                state = module._merged_lora_factors
                scale = self._current_scale(module, module.linear_b_q.weight)

                old_q = state.b_q @ state.a_q
                old_v = state.b_v @ state.a_v
                current_q = scale * (
                    module.linear_b_q.weight @ module.linear_a_q.weight
                )
                current_v = scale * (
                    module.linear_b_v.weight @ module.linear_a_v.weight
                )
                effective_before_q = old_q + current_q
                effective_before_v = old_v + current_v

                # Reference EWCLoRA factor accumulation.  Scaling is folded
                # into B; with the project's standard IdentityScale this is
                # exactly A_bar += A_t, B_bar += B_t.
                state.a_q.add_(module.linear_a_q.weight)
                state.b_q.add_(scale * module.linear_b_q.weight)
                state.a_v.add_(module.linear_a_v.weight)
                state.b_v.add_(scale * module.linear_b_v.weight)
                state.merge_steps.add_(1)

                accumulated_q = state.b_q @ state.a_q
                accumulated_v = state.b_v @ state.a_v
                cross_q = accumulated_q - effective_before_q
                cross_v = accumulated_v - effective_before_v

                current_norm_sq += float(current_q.pow(2).sum().item())
                current_norm_sq += float(current_v.pow(2).sum().item())
                accumulated_norm_sq += float(accumulated_q.pow(2).sum().item())
                accumulated_norm_sq += float(accumulated_v.pow(2).sum().item())
                cross_term_norm_sq += float(cross_q.pow(2).sum().item())
                cross_term_norm_sq += float(cross_v.pow(2).sum().item())
                factor_a_norm_sq += float(state.a_q.pow(2).sum().item())
                factor_a_norm_sq += float(state.a_v.pow(2).sum().item())
                factor_b_norm_sq += float(state.b_q.pow(2).sum().item())
                factor_b_norm_sq += float(state.b_v.pow(2).sum().item())
                merged_numel += int(current_q.numel() + current_v.numel())
                merge_steps = max(merge_steps, int(state.merge_steps.item()))

                self._reset_current_lora(module)
                _inject_merged_factor_delta(module, ())

        residual_max = 0.0
        with torch.no_grad():
            for _, delta in self._iter_delta_terms(base_model):
                if delta.numel():
                    residual_max = max(residual_max, float(delta.abs().max().item()))
        if residual_max != 0.0:
            raise RuntimeError(
                f"factor-merge LoRA reset did not produce delta_W_t=0 "
                f"(max={residual_max:.3e})"
            )

        metrics = {
            "merge_task": int(self._cur_task),
            "merge_lora_modules": int(len(modules)),
            "merge_delta_tensors": int(2 * len(modules)),
            "merge_delta_numel": int(merged_numel),
            "merge_delta_norm": float(current_norm_sq**0.5),
            "factor_merge_steps": int(merge_steps),
            "factor_merge_accumulated_delta_norm": float(accumulated_norm_sq**0.5),
            "factor_merge_cross_term_norm": float(cross_term_norm_sq**0.5),
            "factor_merge_a_norm": float(factor_a_norm_sq**0.5),
            "factor_merge_b_norm": float(factor_b_norm_sq**0.5),
            "factor_merge_base_weight_changed": False,
            "fresh_delta_max_abs": float(residual_max),
        }
        self._last_factor_merge_metrics = metrics
        self._last_merge_metrics = metrics
        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA-MergeFactor] task=%d modules=%d current_delta_norm=%.6e "
                "accumulated_delta_norm=%.6e cross_term_norm=%.6e "
                "fresh_delta_max=%.1e",
                int(self._cur_task),
                int(len(modules)),
                float(metrics["merge_delta_norm"]),
                float(metrics["factor_merge_accumulated_delta_norm"]),
                float(metrics["factor_merge_cross_term_norm"]),
                float(residual_max),
            )
        return metrics

    def _save_factor_merged_final_model_if_requested(self) -> None:
        if not self._is_main_process or not as_bool(
            self.args.get("save_final_model", False)
        ):
            return
        nb_tasks = int(self.args.get("nb_tasks", self.args.get("total_sessions", 0)))
        if nb_tasks > 0 and int(self._cur_task) != nb_tasks - 1:
            return

        network = self._unwrap_network()
        checkpoint_dir = os.path.abspath(
            os.path.expanduser(str(self.args.get("filepath", "./")))
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        final_ckpt = {
            "method": self.args.get("model_name", self._METHOD_NAME),
            "dataset": self.args.get("dataset", "dataset"),
            "init_cls": self.args.get("init_cls", "init_cls"),
            "increment": self.args.get(
                "increment", self.args.get("init_cls", "init_cls")
            ),
            "seed": self.args.get("seed", "seed"),
            "tasks": int(self._cur_task),
            "model_state_dict": network.state_dict(),
        }
        torch.save(final_ckpt, final_path)
        logger.info(
            "[AS1-FS-LoRA-MergeFactor][FinalSave] Saved post-factor-merge model at %s",
            final_path,
        )

    def after_task(self):
        # Unchanged Fisher timing: compute F_t from the trained current B_t@A_t
        # before current factors are reset or integrated into historical state.
        _AS1NormFisherGAM.after_task(self)

        if getattr(self, "_fresh_merge_pending", True):
            self._maybe_barrier()
            self._network = self._unwrap_network()
            self._merge_current_lora_factors()
            self._fresh_merge_pending = False
            self._maybe_barrier()

            if self._is_main_process and hasattr(self._network, "save_fc"):
                self._network.save_fc(self.args.get("filepath", "./"), self._cur_task)
            self._save_factor_merged_final_model_if_requested()

        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": self._METHOD_NAME,
                    "lora_lifecycle": "fresh_adapter_merge_factors",
                    "cross_task_update": "Abar_plus_At__Bbar_plus_Bt",
                    "ewc_anchor": "zero_delta",
                    "stores_historical_lora_factors": True,
                    "base_weight_merge": False,
                    "factor_merge_has_cross_terms": True,
                    "fs_space": "factor",
                    "fs_rho": float(self._fs_rho),
                    "fs_alpha": float(self._fs_alpha),
                    "fs_alpha_mode": str(self._fs_alpha_mode),
                    "fs_sam_equivalent": bool(self._fs_sam_equivalent),
                    **self._last_task_fisher_damping_metrics,
                    **self._last_factor_merge_metrics,
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""

