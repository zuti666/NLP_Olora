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
5. During later GAM updates, project the current-task direction away from the
   EWC restoring direction before adding the EWC gradient.

The important difference from the parent damp learner is that Fisher snapshots
are not mixed directly. The current-task Fisher is always computed from the
post-merge model, so the stored precision matches the weights that will be used
as the next task's reference.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from models_LoRAbasedCL.as1_normfisher_gam_intask_damp import Learner as _DampLearner
from utils.mechanism_eval import as_bool

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

    @staticmethod
    def _zeros_like_optional(reference_a, reference_b):
        ref = reference_a if reference_a is not None else reference_b
        return torch.zeros_like(ref)

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

        self._ewc_project_grad = as_bool(args.get("ewc_project_grad", True))
        self._ewc_project_eps = float(
            args.get("ewc_project_eps", args.get("fisher_project_eps", 1e-12))
        )
        self._dual_rho = float(args.get("ewc_dual_rho", args.get("dual_rho", 0.5)))
        self._ewc_project_rho = float(
            args.get("ewc_project_rho", args.get("fisher_project_rho", self._dual_rho))
        )
        self._ewc_project_boundary = as_bool(args.get("ewc_project_boundary", True))
        self._ewc_project_outward_only = as_bool(
            args.get("ewc_project_outward_only", True)
        )
        self._ewc_project_drift_key = str(
            args.get("ewc_project_drift_key", "fisher_weighted_drift")
        )
        self._ewc_project_budget = float(
            args.get("ewc_project_budget", args.get("ewc_dual_delta", self._dual_delta))
        )

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp-WMergeThenFisher] enabled: "
                "weight_merge=%s weight_checkpoint_interval=%d "
                "weight_merge_omega=%.4g path_weighted=%s "
                "fisher_snapshots=False fisher_source=post_weight_merge "
                "project=%s project_rho=%.4g boundary=%s outward_only=%s "
                "project_drift_key=%s project_budget=%.4g",
                self._in_task_weight_merge,
                self._in_task_weight_checkpoint_interval,
                self._in_task_weight_merge_omega,
                self._in_task_path_weighted,
                self._ewc_project_grad,
                self._ewc_project_rho,
                self._ewc_project_boundary,
                self._ewc_project_outward_only,
                self._ewc_project_drift_key,
                self._ewc_project_budget,
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

    def _apply_in_task_weight_merge(self, model: torch.nn.Module) -> Dict[str, float]:
        metrics = super()._apply_in_task_weight_merge(model)
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
    # Projected current-task / EWC gradient composition
    # ------------------------------------------------------------------

    def _projection_drift_value(self) -> float:
        if not self._fisher_past_delta or not self._delta_reference:
            return 0.0
        metrics = self._compute_delta_drift_metrics()
        return float(metrics.get(self._ewc_project_drift_key, 0.0))

    def _build_direction_projector(
        self,
        clean_grad_cache: Dict[int, torch.Tensor],
        fisher_grad_cache: Dict[int, torch.Tensor],
        delta_w_penalty_grad_cache: Optional[Dict[int, torch.Tensor]] = None,
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self._ewc_project_eps)
        delta_w_penalty_grad_cache = delta_w_penalty_grad_cache or {}
        drift_value = self._projection_drift_value()
        threshold = float(self._ewc_project_rho * self._ewc_project_budget)

        def projector(param_groups):
            clean_norm_sq = 0.0
            gam_norm_sq = 0.0
            flat_norm_sq = 0.0
            task_norm_sq = 0.0
            fisher_norm_sq = 0.0
            delta_w_penalty_norm_sq = 0.0
            projected_task_norm_sq = 0.0
            removed_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = 0.0
            clean_flat_dot = 0.0
            clean_fisher_dot = 0.0
            clean_delta_w_penalty_dot = 0.0
            flat_fisher_dot = 0.0
            flat_delta_w_penalty_dot = 0.0
            task_fisher_dot = 0.0
            task_delta_w_penalty_dot = 0.0
            fisher_delta_w_penalty_dot = 0.0
            param_count = 0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    fisher_grad = fisher_grad_cache.get(id(param))
                    delta_w_penalty_grad = delta_w_penalty_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()

                    if clean_grad is None and gam_grad is None:
                        task_grad = None
                    elif clean_grad is None:
                        task_grad = gam_grad.clone()
                    elif gam_grad is None:
                        task_grad = clean_grad.clone()
                    else:
                        task_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if (
                        task_grad is None
                        and fisher_grad is None
                        and delta_w_penalty_grad is None
                    ):
                        continue

                    param_count += 1
                    if task_grad is not None:
                        task_f = task_grad.detach().float()
                        task_norm_sq += float((task_f * task_f).sum().item())
                    if fisher_grad is not None:
                        fisher_f = fisher_grad.detach().float()
                        fisher_norm_sq += float((fisher_f * fisher_f).sum().item())
                    if delta_w_penalty_grad is not None:
                        delta_w_f = delta_w_penalty_grad.detach().float()
                        delta_w_penalty_norm_sq += float((delta_w_f * delta_w_f).sum().item())
                    if task_grad is not None and fisher_grad is not None:
                        task_fisher_dot += float(
                            (task_grad.detach().float() * fisher_grad.detach().float())
                            .sum()
                            .item()
                        )
                    if task_grad is not None and delta_w_penalty_grad is not None:
                        task_delta_w_penalty_dot += float(
                            (
                                task_grad.detach().float()
                                * delta_w_penalty_grad.detach().float()
                            )
                            .sum()
                            .item()
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
                                flat_fisher_dot += float(
                                    (flat_grad * fisher_grad).sum().item()
                                )
                            if delta_w_penalty_grad is not None:
                                flat_delta_w_penalty_dot += float(
                                    (flat_grad * delta_w_penalty_grad).sum().item()
                                )
                        if clean_grad is not None and fisher_grad is not None:
                            clean_fisher_dot += float((clean_grad * fisher_grad).sum().item())
                        if clean_grad is not None and delta_w_penalty_grad is not None:
                            clean_delta_w_penalty_dot += float(
                                (clean_grad * delta_w_penalty_grad).sum().item()
                            )
                        if fisher_grad is not None and delta_w_penalty_grad is not None:
                            fisher_delta_w_penalty_dot += float(
                                (fisher_grad * delta_w_penalty_grad).sum().item()
                            )

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
                    delta_w_penalty_grad = delta_w_penalty_grad_cache.get(id(param))
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
                        fisher_term = (
                            self._zeros_like_optional(task_grad, delta_w_penalty_grad)
                            if task_grad is not None or delta_w_penalty_grad is not None
                            else None
                        )
                    else:
                        fisher_term = fisher_grad.to(device=param.device, dtype=param.dtype)

                    if delta_w_penalty_grad is None:
                        delta_w_term = (
                            self._zeros_like_optional(task_grad, fisher_term)
                            if task_grad is not None or fisher_term is not None
                            else None
                        )
                    else:
                        delta_w_term = delta_w_penalty_grad.to(
                            device=param.device,
                            dtype=param.dtype,
                        )

                    if task_grad is None:
                        task_grad = (
                            self._zeros_like_optional(fisher_term, delta_w_term)
                            if fisher_term is not None or delta_w_term is not None
                            else None
                        )

                    if task_grad is None:
                        continue

                    projected_task = task_grad
                    removed = self._zeros_like_optional(projected_task, fisher_term)
                    if apply_projection and fisher_grad is not None and fisher_term is not None:
                        removed = coeff * fisher_term
                        projected_task = task_grad - removed

                    combined_grad = projected_task
                    if fisher_term is not None:
                        combined_grad = combined_grad + fisher_term
                    if delta_w_term is not None:
                        combined_grad = combined_grad + delta_w_term

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
                delta_w_penalty_norm = delta_w_penalty_norm_sq ** 0.5
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
                        "delta_w_penalty_norm": delta_w_penalty_norm,
                        "projected_task_norm": projected_task_norm,
                        "projection_removed_norm": removed_norm,
                        "projection_removal_ratio": removed_norm / (task_norm + eps),
                        "combined_norm": combined_norm,
                        "flat_to_clean_norm_ratio": flat_norm / (clean_norm + eps),
                        "fisher_to_clean_norm_ratio": fisher_norm / (clean_norm + eps),
                        "delta_w_penalty_to_clean_norm_ratio": (
                            delta_w_penalty_norm / (clean_norm + eps)
                        ),
                        "cos_clean_gam": self._cosine_from_sums(
                            clean_gam_dot, clean_norm_sq, gam_norm_sq, eps
                        ),
                        "cos_clean_flat": self._cosine_from_sums(
                            clean_flat_dot, clean_norm_sq, flat_norm_sq, eps
                        ),
                        "cos_clean_fisher": self._cosine_from_sums(
                            clean_fisher_dot, clean_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_clean_delta_w_penalty": self._cosine_from_sums(
                            clean_delta_w_penalty_dot,
                            clean_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "cos_flat_fisher": self._cosine_from_sums(
                            flat_fisher_dot, flat_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_flat_delta_w_penalty": self._cosine_from_sums(
                            flat_delta_w_penalty_dot,
                            flat_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "cos_task_fisher": self._cosine_from_sums(
                            task_fisher_dot, task_norm_sq, fisher_norm_sq, eps
                        ),
                        "cos_task_delta_w_penalty": self._cosine_from_sums(
                            task_delta_w_penalty_dot,
                            task_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "cos_fisher_delta_w_penalty": self._cosine_from_sums(
                            fisher_delta_w_penalty_dot,
                            fisher_norm_sq,
                            delta_w_penalty_norm_sq,
                            eps,
                        ),
                        "project_enabled": float(self._ewc_project_grad),
                        "project_applied": float(apply_projection),
                        "project_boundary_active": float(boundary_active),
                        "project_outward": float(outward),
                        "project_threshold": float(threshold),
                        "project_drift": float(drift_value),
                        "project_dot": float(task_fisher_dot),
                        "project_coeff": float(coeff),
                        "project_param_count": float(param_count),
                    }
                )

        return projector

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
        return current_fisher

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics["method_type"] = (
                "as1_normfisher_gam_intask_damp_weightmergethenfisher_project"
            )
            self._last_mechanism_metrics["weight_merge_then_fisher"] = True
            self._last_mechanism_metrics["weight_merge_then_fisher_project"] = True
            self._last_mechanism_metrics["posterior_center_smoothing_before_fisher"] = True
            self._last_mechanism_metrics.update(
                {
                    **self._weight_merge_then_fisher_last_metrics,
                    "ewc_project_grad": bool(self._ewc_project_grad),
                    "ewc_project_eps": float(self._ewc_project_eps),
                    "ewc_project_rho": float(self._ewc_project_rho),
                    "ewc_project_boundary": bool(self._ewc_project_boundary),
                    "ewc_project_outward_only": bool(self._ewc_project_outward_only),
                    "ewc_project_drift_key": self._ewc_project_drift_key,
                    "ewc_project_budget": float(self._ewc_project_budget),
                }
            )
            if self._is_main_process:
                self._log(
                    "[AS1-NormFisher-Damp-WMergeThenFisher-Project][Mechanism] "
                    f"{self._last_mechanism_metrics}"
                )


EWCLoRA = Learner
