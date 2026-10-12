"""AS1 NormFisher GAM InTask damp variant: weight-merge then Fisher.

This variant changes the task-end order used by
``as1_normfisher_gam_intask_damp``:

1. During training, checkpoint LoRA A/B weights at the configured weight
   checkpoint interval.
2. At task end, merge the final LoRA weights with the late-checkpoint path
   mean. The default path mean is a uniform average; train-CE softmax weighting
   is kept only as an opt-in ablation via ``in_task_path_weighted=True``.
3. Estimate the current-task Fisher once from the merged weights.
4. Accumulate that current Fisher into the previous accumulated Fisher, then
   apply the existing trace-normalized damping floor.

The important difference from the parent damp learner is that Fisher snapshots
are not mixed directly. The current-task Fisher is always computed from the
post-merge model, so the stored precision matches the weights that will be used
as the next task's reference.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

from models_LoRAbasedCL.as1_normfisher_gam_intask_damp import Learner as _DampLearner
from utils.mechanism_eval import (
    as_bool,
    average_accuracy_on_fixed_batches,
    average_loss_on_fixed_batches,
    collect_fixed_batches,
    delta_fisher_drift_metrics,
    delta_w_from_factor_pair,
    fisher_buffer_metrics,
    future_surrogate_residual_metrics,
    parse_float_list,
    preserve_training_mode,
    temporarily_loaded_named_params,
)

logger = logging.getLogger(__name__)


class Learner(_DampLearner):
    """Damp learner with task-end weight merge before current-task Fisher."""

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

    def __init__(self, args):
        super().__init__(args)
        self._in_task_weight_merge = as_bool(
            args.get(
                "in_task_weight_merge",
                args.get("weight_merge_with_in_task_fisher", True),
            )
        )
        self._in_task_weight_checkpoint_interval = int(
            args.get(
                "in_task_weight_checkpoint_interval",
                args.get("in_task_fisher_interval", 10),
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
        self._in_task_path_weighted = as_bool(
            args.get(
                "in_task_path_weighted",
                args.get("path_fisher_weighted", False),
            )
        )
        # Parent code still reads these names for the task-end weight merge and
        # mechanism metrics. Keep them as compatibility aliases only.
        self._in_task_fisher_interval = int(self._in_task_weight_checkpoint_interval)
        self._in_task_fisher_omega = float(self._in_task_weight_merge_omega)
        self._weight_merge_then_fisher_last_metrics: Dict[str, float] = {}

        # Center-Consistency / anchor mechanism study (default OFF; does not
        # change training numerics when disabled). See:
        #   refine-logs/EXPERIMENT_PLAN.md - Block 2/3.
        self._posterior_state_save = as_bool(args.get("posterior_state_save", False))
        self._posterior_state_dir = args.get("posterior_state_dir", None)
        self._mechanism_anchor_eval = as_bool(args.get("mechanism_anchor_eval", False))
        self._mechanism_future_surrogate_eval = as_bool(
            args.get("mechanism_future_surrogate_eval", False)
        )
        self._mechanism_anchor_eval_max_batches = int(
            args.get("mechanism_anchor_eval_max_batches", 20)
        )
        self._mechanism_future_surrogate_max_batches = int(
            args.get("mechanism_future_surrogate_max_batches", 20)
        )
        self._fisher_topk_fractions = parse_float_list(
            args.get("fisher_topk_fractions", None),
            default=[0.01, 0.05, 0.1],
        )
        self._posterior_theta_end: Dict[str, torch.Tensor] = {}
        self._posterior_mu_merge: Dict[str, torch.Tensor] = {}
        self._posterior_fisher_endpoint: Dict[str, torch.Tensor] = {}
        self._posterior_fisher_merge: Dict[str, torch.Tensor] = {}

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp-WMergeThenFisher] enabled: "
                "weight_merge=%s weight_checkpoint_interval=%d "
                "weight_merge_omega=%.4g path_weighted=%s "
                "fisher_snapshots=False fisher_source=post_weight_merge",
                self._in_task_weight_merge,
                self._in_task_weight_checkpoint_interval,
                self._in_task_weight_merge_omega,
                self._in_task_path_weighted,
            )

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
        self._weight_merge_then_fisher_last_metrics = {}
        return super()._run_regularized_train(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            class_offset=class_offset,
        )

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
        """Compatibility hook: this variant checkpoints weights, not Fisher."""
        return self._should_capture_in_task_weight(epoch_idx)

    def _capture_in_task_weight_snapshot(
        self,
        train_loader,
        model: torch.nn.Module,
        optimizer,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        """Capture a weight checkpoint; Fisher is computed only at task end."""
        del optimizer

        if self._in_task_path_weighted:
            quality_loader = self._get_current_fisher_loader(train_loader)
            metric_split = "fisher" if quality_loader is not train_loader else "train"
            snapshot_metrics = self._estimate_snapshot_train_metrics(
                train_loader=quality_loader,
                model=model,
                class_offset=class_offset,
            )
        else:
            metric_split = "skipped"
            snapshot_metrics = self._empty_snapshot_metrics()

        weight_snapshot: Dict[str, torch.Tensor] = {}
        if self._in_task_weight_merge:
            weight_snapshot = self._snapshot_lora_factor_weights(model)
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

        lora_tensors = self._count_lora_like_tensors(model)
        self._in_task_lora_snapshot_tensors += int(lora_tensors)

        if self._is_main_process:
            weight_stats = self._summarize_weight_snapshot(weight_snapshot)
            logger.info(
                "[AS1-NormFisher-Damp-WMergeThenFisher][Checkpoint] "
                "task=%d epoch=%d count=%d quality_split=%s ce=%.6g acc=%.3f "
                "samples=%d batches=%d lora_tensors=%d weight_tensors=%d "
                "weight_l2=%.6g",
                self._cur_task,
                epoch_idx,
                checkpoint_count,
                metric_split,
                float(snapshot_metrics["snapshot_train_ce"]),
                float(snapshot_metrics["snapshot_train_acc"]),
                int(snapshot_metrics["snapshot_train_samples"]),
                int(snapshot_metrics["snapshot_train_batches"]),
                lora_tensors,
                int(weight_stats["tensors"]),
                float(weight_stats["l2"]),
            )
            if self._in_task_weight_merge and not weight_snapshot:
                logger.warning(
                    "[AS1-NormFisher-Damp-WMergeThenFisher][Checkpoint] "
                    "weight merge is enabled but no LoRA factor weights were captured "
                    "for task=%d epoch=%d",
                    self._cur_task,
                    epoch_idx,
                )

        return checkpoint_count

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: torch.nn.Module,
        optimizer,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        """Compatibility hook for the parent training loop."""
        return self._capture_in_task_weight_snapshot(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch_idx=epoch_idx,
            class_offset=class_offset,
        )

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
            for key, value in weight_snapshot.items():
                value_cpu = value.detach().cpu()
                if key not in accum:
                    accum[key] = value_cpu * float(weight)
                    weight_sums[key] = float(weight)
                else:
                    value_t = self._match_shape(value_cpu, accum[key])
                    accum[key] = accum[key] + value_t * float(weight)
                    weight_sums[key] = weight_sums.get(key, 0.0) + float(weight)

        path_mean = {
            key: value / max(weight_sums.get(key, 0.0), float(self._in_task_path_loss_eps))
            for key, value in accum.items()
        }
        path_stats = self._summarize_weight_snapshot(path_mean)
        self._in_task_path_weight_last_metrics.update(
            {
                "weight_merge_then_fisher_path_weight_tensors": float(path_stats["tensors"]),
                "weight_merge_then_fisher_path_weight_numel": float(path_stats["numel"]),
                "weight_merge_then_fisher_path_weight_l2": float(path_stats["l2"]),
            }
        )

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp-WMergeThenFisher][WeightPath] "
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

    def _needs_posterior_state(self) -> bool:
        return bool(
            self._posterior_state_save
            or self._mechanism_anchor_eval
            or self._mechanism_future_surrogate_eval
        )

    def _capture_endpoint_posterior_state(self, model: torch.nn.Module) -> None:
        """Snapshot theta_end (raw, pre-merge) and optionally F(theta_end).

        Must run before ``super()._apply_in_task_weight_merge`` mutates the
        model's LoRA weights in place. This is an extra, opt-in Fisher
        estimation pass purely for mechanism analysis; it never feeds back
        into the accumulated EWC Fisher used for training.
        """
        if not self._needs_posterior_state():
            self._posterior_theta_end = {}
            self._posterior_fisher_endpoint = {}
            return

        self._posterior_theta_end = self._snapshot_lora_factor_weights(model)

        if not (self._posterior_state_save or self._mechanism_future_surrogate_eval):
            self._posterior_fisher_endpoint = {}
            return

        was_training = bool(model.training)
        fisher_loader = self._get_current_fisher_loader(self.train_loader)
        try:
            raw_endpoint_fisher = self._compute_raw_delta_fisher(
                fisher_loader, model, int(self._known_classes)
            )
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if self._normalize_fisher and raw_endpoint_fisher:
            self._posterior_fisher_endpoint = self._trace_normalize_fisher(raw_endpoint_fisher)
        else:
            self._posterior_fisher_endpoint = raw_endpoint_fisher

    def _apply_in_task_weight_merge(self, model: torch.nn.Module) -> Dict[str, float]:
        self._capture_endpoint_posterior_state(model)
        metrics = super()._apply_in_task_weight_merge(model)
        if self._needs_posterior_state():
            self._posterior_mu_merge = self._snapshot_lora_factor_weights(model)
        else:
            self._posterior_mu_merge = {}
        path_delta_l2 = float(metrics.get("in_task_weight_path_delta_l2", 0.0))
        shift_l2 = float(metrics.get("in_task_weight_merge_shift_l2", 0.0))
        path_to_merge_l2 = max(0.0, (1.0 - float(self._in_task_weight_merge_omega))) * path_delta_l2
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
                **self._in_task_path_weight_last_metrics,
            }
        )
        self._in_task_weight_merge_last_metrics = metrics
        self._weight_merge_then_fisher_last_metrics.update(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Current-task Fisher from merged weights only
    # ------------------------------------------------------------------

    def _compute_delta_fisher(
        self,
        loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, torch.Tensor]:
        final_raw = self._compute_raw_delta_fisher(loader, model, class_offset)
        final_trace = self._fisher_trace(final_raw)
        self._fisher_new_task_raw_trace = float(final_trace)

        if self._normalize_fisher and final_raw:
            current_fisher = self._trace_normalize_fisher(final_raw)
        else:
            current_fisher = final_raw

        current_trace = self._fisher_trace(current_fisher)
        checkpoint_epochs = [epoch for epoch, _ in self._in_task_weight_snapshots]
        self._last_task_new_fisher_stats = self._summarize_tensor_dict(current_fisher)
        merge_applied = bool(
            self._in_task_weight_merge_last_metrics.get(
                "in_task_weight_merge_applied", 0.0
            )
        )

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
            "in_task_fisher_used_path": False,
            "in_task_fisher_interval": int(self._in_task_weight_checkpoint_interval),
            "in_task_fisher_omega": float(self._in_task_weight_merge_omega),
            "in_task_fisher_snapshots": 0,
            "in_task_fisher_first_epoch": 0,
            "in_task_fisher_last_epoch": 0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": 0.0,
            "in_task_fisher_mixed_trace": float(current_trace),
            "in_task_fisher_final_raw_trace": float(final_trace),
            "in_task_fisher_path_raw_trace": 0.0,
            "in_task_fisher_mixed_raw_trace": float(current_trace),
            "in_task_lora_weight_snapshots": int(len(checkpoint_epochs)),
            "in_task_lora_snapshot_tensors": int(self._in_task_lora_snapshot_tensors),
            "weight_merge_then_fisher": True,
            "weight_merge_then_fisher_current_fisher_after_weight_merge": True,
            "weight_merge_then_fisher_weight_checkpoints": int(len(checkpoint_epochs)),
            "weight_merge_then_fisher_first_checkpoint_epoch": int(checkpoint_epochs[0])
            if checkpoint_epochs
            else 0,
            "weight_merge_then_fisher_last_checkpoint_epoch": int(checkpoint_epochs[-1])
            if checkpoint_epochs
            else 0,
            "weight_merge_then_fisher_merge_applied": bool(merge_applied),
            "weight_merge_then_fisher_degenerated_to_endpoint": bool(
                self._in_task_weight_merge
                and self._in_task_weight_merge_omega > 0.0
                and not merge_applied
            ),
        }
        self._weight_merge_then_fisher_last_metrics.update(
            self._in_task_fisher_last_metrics
        )

        if self._is_main_process:
            if (
                self._in_task_weight_merge
                and self._in_task_weight_merge_omega > 0.0
                and not merge_applied
            ):
                logger.warning(
                    "[AS1-NormFisher-Damp-WMergeThenFisher][Fisher] "
                    "task=%d requested weight merge but no merge was applied; "
                    "current Fisher is endpoint Fisher.",
                    self._cur_task,
                )
            logger.info(
                "[AS1-NormFisher-Damp-WMergeThenFisher][Fisher] "
                "task=%d source=post_weight_merge checkpoints=%d "
                "merge_applied=%s omega=%.4g raw_trace=%.6g trace=%.6g "
                "normalize_task=%s",
                self._cur_task,
                len(checkpoint_epochs),
                merge_applied,
                float(self._in_task_weight_merge_omega),
                float(final_trace),
                float(current_trace),
                bool(self._normalize_fisher),
            )

        if self._posterior_state_save or self._mechanism_future_surrogate_eval:
            self._posterior_fisher_merge = current_fisher
        else:
            self._posterior_fisher_merge = {}

        return current_fisher

    # ------------------------------------------------------------------
    # Center-Consistency / anchor mechanism study (opt-in, default OFF)
    # ------------------------------------------------------------------

    def _posterior_state_path(self, task_idx: int) -> str:
        base_dir = self._posterior_state_dir
        if not base_dir:
            base_dir = os.path.join(
                str(self.args.get("outputs_root", "outputs_logs")),
                "posterior_state",
                str(self.args.get("prefix", "run")),
            )
        base_dir = os.path.abspath(os.path.expanduser(str(base_dir)))
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"posterior_state_task{int(task_idx):02d}.pt")

    def _save_posterior_state(self) -> None:
        # ``future_surrogate_eval`` reads back the previous task's saved state,
        # so persisting to disk must also happen when only that flag is set
        # (not just when ``posterior_state_save`` is explicitly requested).
        needs_save = self._posterior_state_save or self._mechanism_future_surrogate_eval
        if not needs_save or not self._is_main_process:
            return
        if not self._posterior_theta_end or not self._posterior_mu_merge:
            return
        path = self._posterior_state_path(int(self._cur_task))
        torch.save(
            {
                "task": int(self._cur_task),
                "theta_end_lora": self._posterior_theta_end,
                "mu_merge_lora": self._posterior_mu_merge,
                "fisher_endpoint": self._posterior_fisher_endpoint,
                "fisher_merge": self._posterior_fisher_merge,
            },
            path,
        )
        logger.info(
            "[AS1-NormFisher-Damp-WMergeThenFisher][PosteriorState] "
            "saved task=%d -> %s",
            self._cur_task,
            path,
        )

    def _delta_terms_from_factor_state(
        self,
        factor_state: Dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Recompute per-module delta_w = B @ A from a saved raw factor snapshot.

        Mirrors ``_iter_delta_terms`` but reads from a CPU snapshot dict
        instead of the live model, so an anchor's delta_w can be recovered
        without mutating the model's current weights.
        """
        out: Dict[str, torch.Tensor] = {}
        for name, _module in self._iter_delta_modules(model):
            delta_q = delta_w_from_factor_pair(
                factor_state.get(f"{name}.linear_a_q.weight"),
                factor_state.get(f"{name}.linear_b_q.weight"),
            )
            if delta_q is not None:
                out[f"{name}.q"] = delta_q
            delta_v = delta_w_from_factor_pair(
                factor_state.get(f"{name}.linear_a_v.weight"),
                factor_state.get(f"{name}.linear_b_v.weight"),
            )
            if delta_v is not None:
                out[f"{name}.v"] = delta_v
        return out

    def _run_mechanism_anchor_eval(self) -> Dict[str, float]:
        """Compare theta_end vs mu_merge loss/acc on the just-finished task."""
        if not self._mechanism_anchor_eval:
            return {}
        if not self._posterior_theta_end or not self._posterior_mu_merge:
            return {}

        loader = getattr(self, "test_loader", None) or getattr(self, "train_loader", None)
        if loader is None:
            return {}

        model = self._unwrap_network()
        task_start, task_end = self._task_class_range(int(self._cur_task))
        fixed_batches = collect_fixed_batches(
            loader,
            self._device,
            task_start,
            task_end,
            max_batches=self._mechanism_anchor_eval_max_batches,
        )
        if not fixed_batches:
            return {}

        metrics: Dict[str, float] = {
            "anchor_eval_task": float(self._cur_task),
            "anchor_eval_batches": float(len(fixed_batches)),
        }
        with preserve_training_mode(model):
            model.eval()
            for anchor_name, anchor_state in (
                ("endpoint", self._posterior_theta_end),
                ("merge", self._posterior_mu_merge),
            ):
                with temporarily_loaded_named_params(model, anchor_state):
                    metrics[f"anchor_{anchor_name}_loss"] = average_loss_on_fixed_batches(
                        model, fixed_batches, class_slice=(task_start, task_end)
                    )
                    metrics[f"anchor_{anchor_name}_acc"] = average_accuracy_on_fixed_batches(
                        model, fixed_batches, class_slice=(task_start, task_end)
                    )
        return metrics

    @staticmethod
    @contextmanager
    def _network_task_pointer_as(model: torch.nn.Module, task_idx: int):
        """Defensive no-op for this model's actual backbone/head.

        ``self._network`` here is an ``IncrementalNet`` wrapping
        ``LoRA_ViT_timm`` (see ``utils/inc_net.py:get_backbone`` for
        ``model_name in {..., "as1_normfisher_gam_intask_damp"}``): the
        classifier is a single *cumulative* ``self.fc`` that grows every
        task, not a per-task ``classifier_pool`` indexed by a live
        ``_cur_task`` pointer (that indexing scheme belongs to the unrelated
        ``EWC_net``/``net_ewclora.py`` backbone, used only for
        ``model_name == "ewclora_youyue_github"``). So this network has no
        ``_cur_task`` attribute and this context manager is a no-op in
        practice; the actual fix for evaluating an old task's fixed batches
        against the right logit columns is ``class_slice=(task_start,
        task_end)`` on ``average_loss_on_fixed_batches`` /
        ``average_accuracy_on_fixed_batches`` (mirrors the proven-correct
        pattern in ``seqlora.Learner._compute_task_local_accuracy``). Kept as
        a defensive fallback in case this class is ever pointed at a
        classifier-pool-style backbone.
        """
        if not hasattr(model, "_cur_task"):
            yield
            return
        original = model._cur_task
        model._cur_task = int(task_idx)
        try:
            yield
        finally:
            model._cur_task = original

    def _run_mechanism_future_surrogate_eval(self) -> Dict[str, float]:
        """Endpoint / Merge-only / Merge-then-Fisher posterior surrogate check.

        Reads back task ``t = cur_task - 1``'s saved anchors and Fishers, then
        measures how well ``Q_F = 0.5 * (Delta)^T F Delta`` (computed at each
        anchor) predicts the actual loss drift on task ``t``'s data now that
        task ``cur_task`` has finished training.
        """
        if not self._mechanism_future_surrogate_eval or int(self._cur_task) < 1:
            return {}

        prev_task = int(self._cur_task) - 1
        path = self._posterior_state_path(prev_task)
        if not os.path.isfile(path):
            return {}
        state = torch.load(path, map_location="cpu")
        theta_end_ref = state.get("theta_end_lora") or {}
        mu_merge_ref = state.get("mu_merge_lora") or {}
        fisher_endpoint_ref = state.get("fisher_endpoint") or {}
        fisher_merge_ref = state.get("fisher_merge") or {}
        if not mu_merge_ref or not fisher_merge_ref:
            return {}

        loader = getattr(self, "test_loader", None)
        if loader is None:
            return {}

        model = self._unwrap_network()
        task_start, task_end = self._task_class_range(prev_task)
        fixed_batches = collect_fixed_batches(
            loader,
            self._device,
            task_start,
            task_end,
            max_batches=self._mechanism_future_surrogate_max_batches,
        )
        if not fixed_batches:
            return {}

        eps = float(self.args.get("mechanism_eps", 1e-12))
        current_state = self._snapshot_lora_factor_weights(model)
        delta_now = self._delta_terms_from_factor_state(current_state, model)

        with self._network_task_pointer_as(model, prev_task):
            with preserve_training_mode(model):
                model.eval()
                loss_current = average_loss_on_fixed_batches(
                    model, fixed_batches, class_slice=(task_start, task_end)
                )

        metrics: Dict[str, float] = {
            "future_surrogate_task": float(prev_task),
            "future_surrogate_batches": float(len(fixed_batches)),
            "future_surrogate_loss_current": float(loss_current),
        }

        # (anchor label, anchor weights, Fisher paired with that anchor)
        surrogate_variants = [
            ("endpoint", theta_end_ref, fisher_endpoint_ref),
            ("merge_only", mu_merge_ref, fisher_endpoint_ref),
            ("merge_then_fisher", mu_merge_ref, fisher_merge_ref),
        ]
        for name, anchor_state, fisher_ref in surrogate_variants:
            if not anchor_state or not fisher_ref:
                continue
            with self._network_task_pointer_as(model, prev_task):
                with preserve_training_mode(model):
                    model.eval()
                    with temporarily_loaded_named_params(model, anchor_state):
                        loss_anchor = average_loss_on_fixed_batches(
                            model, fixed_batches, class_slice=(task_start, task_end)
                        )

            delta_ref = self._delta_terms_from_factor_state(anchor_state, model)
            keys = sorted(set(delta_now) & set(delta_ref) & set(fisher_ref))
            if not keys:
                continue
            drift_terms = [
                delta_now[key].to(torch.float32)
                - delta_ref[key].to(device=delta_now[key].device, dtype=torch.float32)
                for key in keys
            ]
            fisher_terms = [fisher_ref[key] for key in keys]
            drift_metrics = delta_fisher_drift_metrics(fisher_terms, drift_terms, eps=eps)
            residual_metrics = future_surrogate_residual_metrics(
                delta_loss=(loss_current - loss_anchor),
                fisher_weighted_drift=drift_metrics["fisher_weighted_drift"],
                eps=eps,
                prefix=f"future_{name}",
            )
            metrics[f"future_{name}_loss_anchor"] = float(loss_anchor)
            metrics.update(residual_metrics)

        return metrics

    def _run_mechanism_fisher_profile(self) -> Dict[str, float]:
        """Fisher concentration profile (Block B7): tr(F), max/mean, top-k% mass,
        effective_dim, concentration -- for both the endpoint and merge Fisher
        when available. Cheap (pure tensor reduction, no forward pass).
        """
        metrics: Dict[str, float] = {}
        if self._posterior_fisher_endpoint:
            metrics.update(
                fisher_buffer_metrics(
                    list(self._posterior_fisher_endpoint.values()),
                    self._fisher_topk_fractions,
                    prefix="fisher_endpoint",
                )
            )
        if self._posterior_fisher_merge:
            metrics.update(
                fisher_buffer_metrics(
                    list(self._posterior_fisher_merge.values()),
                    self._fisher_topk_fractions,
                    prefix="fisher_merge",
                )
            )
        return metrics

    def after_task(self):
        super().after_task()
        mechanism_extra: Dict[str, float] = {}
        mechanism_extra.update(self._run_mechanism_anchor_eval())
        mechanism_extra.update(self._run_mechanism_future_surrogate_eval())
        mechanism_extra.update(self._run_mechanism_fisher_profile())
        self._save_posterior_state()

        if self._last_mechanism_metrics:
            self._last_mechanism_metrics["method_type"] = (
                "as1_normfisher_gam_intask_damp_weightmergethenfisher"
            )
            self._last_mechanism_metrics["weight_merge_then_fisher"] = True
            self._last_mechanism_metrics["posterior_center_smoothing_before_fisher"] = True
            self._last_mechanism_metrics.update(
                self._weight_merge_then_fisher_last_metrics
            )
            if mechanism_extra:
                self._last_mechanism_metrics.update(mechanism_extra)
            if self._is_main_process:
                self._log(
                    "[AS1-NormFisher-Damp-WMergeThenFisher][Mechanism] "
                    f"{self._last_mechanism_metrics}"
                )


EWCLoRA = Learner
