"""Full-model EWC + GAM: weight-merge then Fisher.

This is the full-parameter counterpart of
``models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py``.

It reuses the full-model optimizer/Fisher machinery from
``ewc_gam_mergedintask_multifisher_norm`` but changes the task-end mechanism:

1. During training, checkpoint full-model trainable weights at the configured
   interval.
2. At task end, merge endpoint weights with the late-checkpoint path mean.
3. Estimate current-task Fisher once from the post-merge model.
4. Accumulate and damp the resulting Fisher through the parent implementation.

Unlike ``ewc_gam_mergedintask_multifisher_norm``, this variant does not mix
in-task Fisher snapshots.  It checkpoints weights only, so the stored Fisher is
matched to the reference weights used for the next task.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from models_CL.ewc_gam_mergedintask_multifisher_norm import (
    EWCGAMMergedInTaskMultiFisherNorm,
)
from models_CL.ewc_gam_projected import _as_bool

logger = logging.getLogger(__name__)


class EWCGAMMergedInTaskMultiFisherNormWeightMergeThenFisher(
    EWCGAMMergedInTaskMultiFisherNorm
):
    """Full-model normalized multi-Fisher with weight merge before Fisher."""

    @staticmethod
    def _unit_interval_float(value) -> float:
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _empty_snapshot_metrics() -> Dict[str, float]:
        return {
            "snapshot_train_ce": float("nan"),
            "snapshot_train_acc": 0.0,
            "snapshot_train_samples": 0.0,
            "snapshot_train_batches": 0.0,
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

    def __init__(self, args):
        parent_args = dict(args)
        if (
            "in_task_weight_merge" not in parent_args
            and "weight_merge_with_in_task_fisher" not in parent_args
        ):
            parent_args["in_task_weight_merge"] = True
        if (
            "in_task_fisher_omega" not in parent_args
            and "path_fisher_omega" not in parent_args
            and "approximate_fisher_omega" not in parent_args
            and "approximte_fisher_omega" not in parent_args
            and "in_task_weight_merge_omega" not in parent_args
        ):
            parent_args["in_task_fisher_omega"] = 0.2
        if (
            "in_task_path_weighted" not in parent_args
            and "path_fisher_weighted" not in parent_args
        ):
            parent_args["in_task_path_weighted"] = False

        super().__init__(parent_args)
        self.args = args

        self._in_task_weight_merge = _as_bool(
            args.get(
                "in_task_weight_merge",
                args.get("weight_merge_with_in_task_fisher", True),
            )
        )
        self._in_task_weight_checkpoint_interval = int(
            args.get(
                "in_task_weight_checkpoint_interval",
                args.get("in_task_fisher_interval", self._in_task_fisher_interval),
            )
        )
        self._in_task_weight_merge_omega = self._unit_interval_float(
            args.get(
                "in_task_weight_merge_omega",
                args.get(
                    "in_task_fisher_omega",
                    args.get(
                        "path_fisher_omega",
                        args.get(
                            "approximate_fisher_omega",
                            args.get("approximte_fisher_omega", 0.2),
                        ),
                    ),
                ),
            )
        )
        self._in_task_path_weighted = _as_bool(
            args.get("in_task_path_weighted", args.get("path_fisher_weighted", False))
        )

        # Parent weight-merge code reads these names.  In this variant they are
        # compatibility aliases for the weight checkpoint/merge schedule.
        self._in_task_fisher_interval = int(self._in_task_weight_checkpoint_interval)
        self._in_task_fisher_omega = float(self._in_task_weight_merge_omega)
        self._weight_merge_then_fisher_last_metrics: Dict[str, float] = {}
        self._in_task_weight_snapshot_tensors = 0

        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher] "
            "weight_merge=%s weight_checkpoint_interval=%d weight_merge_omega=%.4g "
            "path_weighted=%s fisher_snapshots=False fisher_source=post_weight_merge",
            self._in_task_weight_merge,
            self._in_task_weight_checkpoint_interval,
            self._in_task_weight_merge_omega,
            self._in_task_path_weighted,
        )

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        self._weight_merge_then_fisher_last_metrics = {}
        self._in_task_weight_snapshot_tensors = 0
        return super()._train_function(train_loader, test_loader, optimizer, scheduler)

    # ------------------------------------------------------------------
    # Checkpoint weights only during training
    # ------------------------------------------------------------------

    def _should_capture_in_task_weight(self, epoch_idx: int) -> bool:
        if not self._in_task_weight_merge:
            return False
        if self._in_task_weight_checkpoint_interval <= 0:
            return False
        if self._in_task_weight_merge_omega <= 0.0:
            return False
        return int(epoch_idx) % int(self._in_task_weight_checkpoint_interval) == 0

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        """Compatibility hook: parent loop calls this, but we save weights."""
        return self._should_capture_in_task_weight(epoch_idx)

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model,
        optimizer,
        epoch_idx: int,
    ) -> int:
        """Capture a full-model weight checkpoint; Fisher is computed at task end."""
        del optimizer

        task_start, task_end = self._task_range()
        if self._in_task_path_weighted:
            snapshot_metrics = self._estimate_snapshot_train_metrics(
                train_loader, model, task_start, task_end
            )
            metric_split = "train"
        else:
            snapshot_metrics = self._empty_snapshot_metrics()
            metric_split = "skipped"

        weight_snapshot: Dict[str, torch.Tensor] = {}
        if self._in_task_weight_merge:
            weight_snapshot = self._snapshot_model_weights()
            if weight_snapshot:
                self._in_task_weight_snapshots.append((int(epoch_idx), weight_snapshot))
                self._in_task_weight_snapshot_tensors += int(len(weight_snapshot))

        checkpoint_count = int(len(self._in_task_weight_snapshots))
        snapshot_metrics.update(
            {
                "snapshot_epoch": float(epoch_idx),
                "snapshot_index": float(checkpoint_count),
            }
        )
        if weight_snapshot:
            self._in_task_snapshot_metrics.append(snapshot_metrics)

        weight_stats = self._summarize_weight_snapshot(weight_snapshot)
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][Checkpoint] "
            "task=%d epoch=%d count=%d quality_split=%s ce=%.6g acc=%.3f "
            "samples=%d batches=%d weight_tensors=%d weight_l2=%.6g",
            self._cur_task,
            epoch_idx,
            checkpoint_count,
            metric_split,
            float(snapshot_metrics["snapshot_train_ce"]),
            float(snapshot_metrics["snapshot_train_acc"]),
            int(snapshot_metrics["snapshot_train_samples"]),
            int(snapshot_metrics["snapshot_train_batches"]),
            int(weight_stats["tensors"]),
            float(weight_stats["l2"]),
        )
        if self._in_task_weight_merge and not weight_snapshot:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][Checkpoint] "
                "weight merge is enabled but no full-model weights were captured "
                "for task=%d epoch=%d",
                self._cur_task,
                epoch_idx,
            )

        return checkpoint_count

    def _weight_snapshot_entries(
        self,
    ) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
        entries: List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]] = []
        for index, (epoch, weight_snapshot) in enumerate(self._in_task_weight_snapshots):
            metrics = (
                self._in_task_snapshot_metrics[index]
                if index < len(self._in_task_snapshot_metrics)
                else {}
            )
            entries.append((index, int(epoch), weight_snapshot, metrics))
        return entries

    def _select_late_weight_snapshot_entries(
        self,
    ) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
        entries = self._weight_snapshot_entries()
        if not entries:
            self._in_task_path_weight_last_metrics = {
                "in_task_path_selected_snapshots": 0.0,
                "in_task_path_total_snapshots": 0.0,
                "in_task_path_fallback_to_last": 0.0,
                "weight_merge_then_fisher_weight_checkpoints": 0.0,
                "in_task_weight_checkpoint_interval": float(
                    self._in_task_weight_checkpoint_interval
                ),
            }
            return []

        max_epoch = max(epoch for _, epoch, _, _ in entries)
        total_epochs = max(int(self._in_task_total_epochs), int(max_epoch), 1)
        min_epoch = int(math.ceil(float(self._in_task_path_start_ratio) * total_epochs))
        min_epoch = max(1, min(min_epoch, total_epochs))
        selected = [entry for entry in entries if int(entry[1]) >= min_epoch]
        fallback = False
        if not selected:
            selected = [entries[-1]]
            fallback = True

        self._in_task_path_weight_last_metrics = {
            "in_task_path_total_epochs": float(total_epochs),
            "in_task_path_start_ratio": float(self._in_task_path_start_ratio),
            "in_task_path_min_epoch": float(min_epoch),
            "in_task_path_total_snapshots": float(len(entries)),
            "in_task_path_selected_snapshots": float(len(selected)),
            "in_task_path_fallback_to_last": float(fallback),
            "in_task_path_selected_first_epoch": float(selected[0][1]),
            "in_task_path_selected_last_epoch": float(selected[-1][1]),
            "weight_merge_then_fisher_weight_checkpoints": float(len(entries)),
            "weight_merge_then_fisher_selected_weight_checkpoints": float(len(selected)),
            "in_task_weight_checkpoint_interval": float(
                self._in_task_weight_checkpoint_interval
            ),
        }
        return selected

    def _path_checkpoint_weights(
        self,
        selected: List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]],
    ) -> List[float]:
        if self._in_task_path_weighted:
            return self._loss_softmax_weights(selected)

        n = len(selected)
        if n <= 0:
            return []
        weights = [1.0 / float(n)] * n
        entropy = math.log(float(n))
        self._in_task_path_weight_last_metrics.update(
            {
                "in_task_path_weighted": 0.0,
                "in_task_path_weight_mode": "uniform",
                "in_task_path_beta": 0.0,
                "in_task_path_weight_min": float(weights[0]),
                "in_task_path_weight_max": float(weights[0]),
                "in_task_path_weight_entropy": float(entropy),
                "in_task_path_weight_effective_n": float(n),
                "in_task_path_loss_valid": 0.0,
            }
        )
        return weights

    def _mean_in_task_weight_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        selected = self._select_late_weight_snapshot_entries()
        if not selected:
            return None

        weights = self._path_checkpoint_weights(selected)
        if not weights:
            return None

        accum: Dict[str, torch.Tensor] = {}
        weight_sums: Dict[str, float] = {}
        for weight, (_, _, weight_snapshot, _) in zip(weights, selected):
            for name, value in weight_snapshot.items():
                value_cpu = value.detach().cpu()
                if name not in accum:
                    accum[name] = value_cpu * float(weight)
                    weight_sums[name] = float(weight)
                else:
                    value_t = self._match_shape(value_cpu, accum[name])
                    accum[name] = accum[name] + value_t * float(weight)
                    weight_sums[name] = weight_sums.get(name, 0.0) + float(weight)

        path_mean = {
            name: value
            / max(weight_sums.get(name, 0.0), float(self._in_task_path_loss_eps))
            for name, value in accum.items()
        }
        path_stats = self._summarize_weight_snapshot(path_mean)
        self._in_task_path_weight_last_metrics.update(
            {
                "weight_merge_then_fisher_path_weight_tensors": float(
                    path_stats["tensors"]
                ),
                "weight_merge_then_fisher_path_weight_numel": float(
                    path_stats["numel"]
                ),
                "weight_merge_then_fisher_path_weight_l2": float(path_stats["l2"]),
            }
        )
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][WeightPath] "
            "task=%d selected=%d/%d epochs=%s weighted=%s beta=%.4g mode=%s "
            "weight_min=%.4g weight_max=%.4g eff_n=%.4g path_l2=%.6g",
            self._cur_task,
            int(self._in_task_path_weight_last_metrics.get("in_task_path_selected_snapshots", 0)),
            int(self._in_task_path_weight_last_metrics.get("in_task_path_total_snapshots", 0)),
            [int(epoch) for _, epoch, _, _ in selected],
            self._in_task_path_weighted,
            float(self._in_task_path_beta),
            self._in_task_path_weight_last_metrics.get("in_task_path_weight_mode", "unknown"),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_min", 0.0)),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_max", 0.0)),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_effective_n", 0.0)),
            float(path_stats["l2"]),
        )
        return path_mean

    def _apply_in_task_weight_merge(self, model) -> Dict[str, float]:
        metrics = super()._apply_in_task_weight_merge(model)
        omega = float(self._in_task_weight_merge_omega)
        shift_l2 = float(metrics.get("in_task_weight_merge_shift_l2", 0.0))
        path_delta_l2 = float(shift_l2 / omega) if omega > 0.0 else 0.0
        path_to_merge_l2 = max(0.0, (1.0 - omega)) * path_delta_l2
        metrics.update(
            {
                "weight_merge_then_fisher": 1.0,
                "weight_merge_then_fisher_fisher_snapshots_disabled": 1.0,
                "weight_merge_then_fisher_fisher_source_post_merge": 1.0,
                "in_task_weight_checkpoint_interval": int(
                    self._in_task_weight_checkpoint_interval
                ),
                "weight_checkpoint_interval": int(self._in_task_weight_checkpoint_interval),
                "in_task_weight_merge_omega": float(self._in_task_weight_merge_omega),
                "weight_merge_omega": float(self._in_task_weight_merge_omega),
                "weight_merge_applied": bool(
                    metrics.get("in_task_weight_merge_applied", 0.0)
                ),
                "weight_merge_theta_end_to_merge_l2": float(shift_l2),
                "weight_merge_theta_path_to_merge_l2": float(path_to_merge_l2),
                "weight_merge_theta_end_to_path_l2": float(path_delta_l2),
                "in_task_weight_snapshot_tensors": float(
                    self._in_task_weight_snapshot_tensors
                ),
                **self._in_task_path_weight_last_metrics,
            }
        )
        self._in_task_weight_merge_last_metrics = metrics
        self._weight_merge_then_fisher_last_metrics.update(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Current-task Fisher from merged weights only
    # ------------------------------------------------------------------

    def _mix_in_task_fisher(self, final_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        final_trace = self._fisher_trace(final_raw)
        current_fisher = (
            self._trace_normalize_fisher(final_raw)
            if self._normalize_fisher and final_raw
            else final_raw
        )
        current_trace = self._fisher_trace(current_fisher)
        checkpoint_epochs = [epoch for epoch, _ in self._in_task_weight_snapshots]
        merge_applied = bool(
            self._in_task_weight_merge_last_metrics.get(
                "in_task_weight_merge_applied", 0.0
            )
        )

        self._last_task_new_fisher_stats = self._summarize_tensor_dict(current_fisher)
        self._in_task_fisher_last_metrics = {
            "current_fisher_source": "post_weight_merge",
            "weight_merge_then_fisher_current_fisher_source": "post_weight_merge",
            "current_fisher_raw_trace": float(final_trace),
            "current_fisher_trace": float(current_trace),
            "weight_checkpoints": int(len(checkpoint_epochs)),
            "weight_checkpoint_interval": int(self._in_task_weight_checkpoint_interval),
            "weight_merge_applied": bool(merge_applied),
            "weight_merge_omega": float(self._in_task_weight_merge_omega),
            "in_task_weight_checkpoint_interval": int(
                self._in_task_weight_checkpoint_interval
            ),
            "in_task_weight_merge_omega": float(self._in_task_weight_merge_omega),
            "in_task_fisher_used_path": 0.0,
            "in_task_fisher_interval": float(self._in_task_weight_checkpoint_interval),
            "in_task_fisher_omega": float(self._in_task_weight_merge_omega),
            "in_task_fisher_snapshots": 0.0,
            "in_task_fisher_first_epoch": 0.0,
            "in_task_fisher_last_epoch": 0.0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": 0.0,
            "in_task_fisher_mixed_trace": float(current_trace),
            "in_task_fisher_final_raw_trace": float(final_trace),
            "in_task_fisher_path_raw_trace": 0.0,
            "in_task_fisher_mixed_raw_trace": float(current_trace),
            "weight_merge_then_fisher": 1.0,
            "weight_merge_then_fisher_current_fisher_after_weight_merge": 1.0,
            "weight_merge_then_fisher_weight_checkpoints": float(len(checkpoint_epochs)),
            "weight_merge_then_fisher_first_checkpoint_epoch": float(
                checkpoint_epochs[0] if checkpoint_epochs else 0
            ),
            "weight_merge_then_fisher_last_checkpoint_epoch": float(
                checkpoint_epochs[-1] if checkpoint_epochs else 0
            ),
            "weight_merge_then_fisher_merge_applied": float(merge_applied),
            "weight_merge_then_fisher_degenerated_to_endpoint": float(
                self._in_task_weight_merge
                and self._in_task_weight_merge_omega > 0.0
                and not merge_applied
            ),
        }
        self._weight_merge_then_fisher_last_metrics.update(
            self._in_task_fisher_last_metrics
        )

        if (
            self._in_task_weight_merge
            and self._in_task_weight_merge_omega > 0.0
            and not merge_applied
        ):
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][Fisher] "
                "task=%d requested weight merge but no merge was applied; "
                "current Fisher is endpoint Fisher.",
                self._cur_task,
            )
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][Fisher] "
            "task=%d source=post_weight_merge checkpoints=%d merge_applied=%s "
            "omega=%.4g raw_trace=%.6g trace=%.6g normalize_task=%s",
            self._cur_task,
            len(checkpoint_epochs),
            merge_applied,
            float(self._in_task_weight_merge_omega),
            float(final_trace),
            float(current_trace),
            bool(self._normalize_fisher),
        )
        return current_fisher

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics["method_type"] = (
                "ewc_gam_mergedintask_multifisher_norm_weightmergethenfisher"
            )
            self._last_mechanism_metrics["weight_merge_then_fisher"] = True
            self._last_mechanism_metrics[
                "posterior_center_smoothing_before_fisher"
            ] = True
            self._last_mechanism_metrics.update(self._weight_merge_then_fisher_last_metrics)
            logging.info(
                "[EWCGAMMergedInTaskMultiFisherNorm-WMergeThenFisher][Mechanism] %s",
                self._last_mechanism_metrics,
            )


EWCGAMMergedInTaskMultiFisherNormWMergeThenFisher = (
    EWCGAMMergedInTaskMultiFisherNormWeightMergeThenFisher
)
Learner = EWCGAMMergedInTaskMultiFisherNormWeightMergeThenFisher
