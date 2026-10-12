"""AS1 EWC + periodic stochastic multi-directional Factor-FS for LoRA.

This learner preserves the EWC/Fisher protocol of
``as1_normfisher_fishershape`` and changes only the current-task optimizer.
On a probe step the current training batch is partitioned into M microbatches.
Each microbatch supplies one fixed-radius clean/perturbed gradient pair; the
optimizer aggregates their Fisher-shaping estimates and optionally gates the
result by cross-direction consistency.  Between probe steps it can reuse the
latest shaping component.

The delta-W EWC gradient is evaluated once at the clean parameters and added
after the CE-only multi-direction direction has been composed.  It never
affects a probing direction or finite difference.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from torch import optim

from models_LoRAbasedCL.as1_normfisher_fishershape import Learner as _BaseFSLearner
from optimer_PerturabtionType.fishershape_multidirectional import (
    StochasticMultiDirectionalFisherShape,
)
from utils.mechanism_eval import as_bool


logger = logging.getLogger(__name__)


class Learner(_BaseFSLearner):
    """Delta-W EWC with fixed-radius multi-direction Factor-FS."""

    _OPTIMIZER_ALIASES = {
        "fishershape_multidirectional",
        "multidirectional_fishershape",
        "stochastic_multidirectional_fs",
        "periodic_multidirectional_fs",
        "pmd_fishershape",
        "pmdfs",
    }

    def __init__(self, args):
        if "optimizer_type" not in args:
            args["optimizer_type"] = "fishershape_multidirectional"
        super().__init__(args)

        # PMD-FS is defined in the trainable LoRA factor space and its EWC
        # state is defined over effective delta-W tensors.  Fail immediately
        # if model-name routing accidentally created a plain ViT backbone;
        # otherwise training would silently become full-model fine-tuning and
        # every task-end Fisher dictionary would be empty.
        delta_module_count = sum(
            1 for _name, _module in self._iter_delta_modules(self._unwrap_network())
        )
        if delta_module_count <= 0:
            raise RuntimeError(
                "PMD-FS-LoRA requires a LoRA backbone with delta-W modules, "
                "but none were found. Check utils.inc_net LoRA model-name "
                f"routing for model_name={args.get('model_name')!r}."
            )
        self._pmdfs_delta_module_count = int(delta_module_count)

        self._fs_num_directions = max(
            1,
            int(args.get("fs_num_directions", args.get("fs_probe_repeats", 2))),
        )
        self._fs_probe_interval = max(1, int(args.get("fs_probe_interval", 1)))
        self._fs_reuse_mode = str(args.get("fs_reuse_mode", "look")).lower()
        self._fs_reuse_scale = float(args.get("fs_reuse_scale", 1.0))
        self._fs_confidence_mode = str(
            args.get("fs_confidence_mode", "consistency")
        ).lower()
        self._fs_match_probe_rng = as_bool(args.get("fs_match_probe_rng", True))
        self._fs_direction_source = str(
            args.get("fs_direction_source", "microbatch_split")
        ).lower()
        if self._fs_direction_source != "microbatch_split":
            raise ValueError(
                "The current PMD-FS implementation supports only "
                "fs_direction_source=microbatch_split, got "
                f"{self._fs_direction_source!r}."
            )

        if self._fs_adaptive:
            raise ValueError(
                "Multi-direction Factor-FS requires fs_adaptive=false because "
                "all directions must use the same Euclidean radius fs_rho."
            )

        if self._is_main_process:
            logger.info(
                "[AS1-PMDFS-LoRA] radius=fixed rho=%.4g directions=%d "
                "probe_interval=%d reuse_mode=%s reuse_scale=%.4g "
                "confidence_mode=%s direction_source=%s match_probe_rng=%s",
                self._fs_rho,
                self._fs_num_directions,
                self._fs_probe_interval,
                self._fs_reuse_mode,
                self._fs_reuse_scale,
                self._fs_confidence_mode,
                self._fs_direction_source,
                self._fs_match_probe_rng,
            )
            logger.info(
                "[AS1-PMDFS-LoRA] preflight delta_modules=%d delta_tensors=%d",
                self._pmdfs_delta_module_count,
                2 * self._pmdfs_delta_module_count,
            )

    def _build_optimizer(self, params, stage):
        if not isinstance(params, (list, tuple)):
            params = list(params)

        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base_name = str(
            self.args.get(
                "fs_base_optimizer",
                self.args.get("gam_base_optimizer", self.args.get("optimizer", "sgd")),
            )
        ).lower()

        if base_name == "adam":
            base_optimizer = optim.Adam(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        elif base_name == "adamw":
            base_optimizer = optim.AdamW(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        else:
            base_optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        return StochasticMultiDirectionalFisherShape(
            params,
            base_optimizer=base_optimizer,
            model=self._network,
            rho=self._fs_rho,
            alpha=self._fs_alpha,
            alpha_mode=self._fs_alpha_mode,
            sam_equivalent=self._fs_sam_equivalent,
            adaptive=self._fs_adaptive,
            perturb_eps=self._fs_xi,
            grad_reduce=self._fs_grad_reduce,
            num_directions=self._fs_num_directions,
            probe_interval=self._fs_probe_interval,
            reuse_mode=self._fs_reuse_mode,
            reuse_scale=self._fs_reuse_scale,
            confidence_mode=self._fs_confidence_mode,
            match_probe_rng=self._fs_match_probe_rng,
        )

    @staticmethod
    def _split_microbatches(inputs, targets, requested: int):
        batch_size = int(inputs.size(0))
        count = max(1, min(int(requested), batch_size))
        input_chunks = torch.tensor_split(inputs, count, dim=0)
        target_chunks = torch.tensor_split(targets, count, dim=0)
        return [
            (input_chunk, target_chunk)
            for input_chunk, target_chunk in zip(input_chunks, target_chunks)
            if int(input_chunk.size(0)) > 0
        ]

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        if not isinstance(optimizer, StochasticMultiDirectionalFisherShape):
            raise TypeError(
                "as1_normfisher_fishershape_multidirectional requires "
                "StochasticMultiDirectionalFisherShape, got "
                f"{type(optimizer).__name__}"
            )

        params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        projector_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_additive_fisher_projector(fisher_grad_cache, projector_stats)
        )

        def full_closure():
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

        microbatches = self._split_microbatches(
            inputs,
            targets,
            self._fs_num_directions,
        )
        probe_closures = []
        probe_weights = []
        batch_size = float(max(1, int(inputs.size(0))))
        for micro_inputs, micro_targets in microbatches:
            def micro_closure(
                micro_inputs=micro_inputs,
                micro_targets=micro_targets,
            ):
                optimizer.zero_grad()
                outputs = self._network(micro_inputs)
                task_loss, _, _ = self._task_loss(
                    outputs["logits"],
                    micro_targets,
                    class_offset,
                )
                loss_value = task_loss.detach()
                task_loss.backward()
                return outputs, loss_value

            probe_closures.append(micro_closure)
            probe_weights.append(float(micro_inputs.size(0)) / batch_size)

        try:
            outputs, clean_loss = optimizer.step(
                closure=full_closure,
                probe_closures=probe_closures,
                probe_weights=probe_weights,
            )
        finally:
            optimizer.set_direction_projector(None)

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
                    "method_type": "as1_normfisher_fishershape_multidirectional",
                    "fs_probe_geometry": "fixed_radius_stochastic_microbatch_directions",
                    "fs_num_directions": float(self._fs_num_directions),
                    "fs_probe_interval": float(self._fs_probe_interval),
                    "fs_reuse_mode": str(self._fs_reuse_mode),
                    "fs_reuse_scale": float(self._fs_reuse_scale),
                    "fs_confidence_mode": str(self._fs_confidence_mode),
                    "fs_direction_source": str(self._fs_direction_source),
                    "fs_match_probe_rng": bool(self._fs_match_probe_rng),
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
