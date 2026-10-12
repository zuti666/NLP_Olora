"""AS1 NormFisher GAM InTask + weighted path Fisher + damping.

Thin subclass of ``as1_normfisher_gam_intask.Learner``.  Instead of a
separate delta-W penalty, a precision floor is absorbed directly into the
trace-normalized accumulated Fisher:

    F̃ = (1 - alpha) * F  +  alpha * (1 / d)

where  d = total parameter count across all Fisher tensors  and
alpha ∈ [0, 1] is the damping coefficient.

Because trace-normalized F satisfies tr(F) = 1 and tr(I/d) = 1, the
convex mix keeps the same scale: tr(F̃) = 1 for any alpha.

    alpha = 0   →  pure original Fisher (no change)
    alpha > 0   →  lifts small-eigenvalue directions; isotropic floor
    alpha = 1   →  collapses to pure isotropic L2 (ridge regression)

Designed for use with ewc_normalize_accumulated_fisher=True.  A warning is
emitted when normalization is off, because the I/d term may be negligible
relative to un-normalized Fisher values.

This subclass also strengthens the parent in-task Fisher path estimate:

    F_path = sum_k pi_k * NormTrace(F_k)

where only late-task snapshots are used by default and pi_k is a softmax over
snapshot train CE.  Damping is intentionally applied after task Fisher merging
and cross-task accumulation.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner as _AS1InTask
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_AS1InTask):
    @staticmethod
    def _optional_positive_int(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "false", "off"}:
                return None
            value = stripped
        parsed = int(value)
        return parsed if parsed > 0 else None

    def __init__(self, args):
        super().__init__(args)

        alpha_raw = float(
            args.get(
                "ewc_fisher_damping_alpha",
                args.get("fisher_damping_alpha", 1e-3),
            )
        )
        self._ewc_fisher_damping_alpha = max(0.0, min(1.0, alpha_raw))
        self._in_task_path_start_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    args.get(
                        "in_task_path_start_ratio",
                        args.get("path_fisher_start_ratio", 0.6),
                    )
                ),
            ),
        )
        self._in_task_path_beta = max(
            0.0,
            float(
                args.get(
                    "in_task_path_beta",
                    args.get("path_fisher_beta", 3.0),
                )
            ),
        )
        self._in_task_path_weighted = as_bool(
            args.get(
                "in_task_path_weighted",
                args.get("path_fisher_weighted", True),
            )
        )
        self._in_task_path_normalize_snapshots = as_bool(
            args.get(
                "in_task_path_normalize_snapshots",
                args.get("path_fisher_normalize_snapshots", True),
            )
        )
        self._in_task_path_loss_eps = float(
            args.get("in_task_path_loss_eps", args.get("path_fisher_loss_eps", 1e-12))
        )
        self._in_task_path_loss_max_batches = self._optional_positive_int(
            args.get(
                "in_task_path_loss_max_batches",
                args.get("path_fisher_loss_max_batches", None),
            )
        )
        self._in_task_total_epochs = 0
        self._in_task_snapshot_metrics: List[Dict[str, float]] = []
        self._in_task_path_weight_last_metrics: Dict[str, float] = {}

        if self._in_task_path_normalize_snapshots and not self._normalize_fisher:
            logger.warning(
                "[AS1-NormFisher-Damp] path_fisher_normalize_snapshots=True but "
                "ewc_normalize_fisher=False; disabling per-snapshot path normalization "
                "to avoid mixing normalized path Fisher with raw endpoint Fisher."
            )
            self._in_task_path_normalize_snapshots = False

        if self._ewc_fisher_damping_alpha > 0.0 and not self._normalize_accumulated_fisher:
            logger.warning(
                "[AS1-NormFisher-Damp] ewc_fisher_damping_alpha=%.4g is set but "
                "ewc_normalize_accumulated_fisher=False; the I/d floor may be "
                "negligible relative to un-normalized Fisher values.",
                self._ewc_fisher_damping_alpha,
            )

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp] ewc_fisher_damping_alpha=%.6g "
                "normalize_accumulated=%s path_weighted=%s path_start_ratio=%.4g "
                "path_beta=%.4g path_norm_snapshots=%s path_loss_max_batches=%s",
                self._ewc_fisher_damping_alpha,
                self._normalize_accumulated_fisher,
                self._in_task_path_weighted,
                self._in_task_path_start_ratio,
                self._in_task_path_beta,
                self._in_task_path_normalize_snapshots,
                self._in_task_path_loss_max_batches,
            )

    # ------------------------------------------------------------------
    # In-task posterior path weighting
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
        self._in_task_total_epochs = int(epochs)
        self._in_task_snapshot_metrics = []
        self._in_task_path_weight_last_metrics = {}
        return super()._run_regularized_train(
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            class_offset=class_offset,
        )

    def _estimate_snapshot_train_metrics(
        self,
        train_loader,
        model: torch.nn.Module,
        class_offset: int,
    ) -> Dict[str, float]:
        """Estimate snapshot quality on train-source data only; never uses test data."""
        was_training = bool(model.training)
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        batches = 0
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    if (
                        self._in_task_path_loss_max_batches is not None
                        and batch_idx >= int(self._in_task_path_loss_max_batches)
                    ):
                        break

                    if len(batch) == 3:
                        _, inputs, targets = batch
                    else:
                        inputs, targets = batch
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)

                    outputs = model(inputs)
                    loss, eval_logits, eval_targets = self._task_loss(
                        outputs["logits"],
                        targets,
                        class_offset,
                    )

                    batch_size = int(eval_targets.numel())
                    total_loss += float(loss.detach().item()) * float(batch_size)
                    preds = torch.argmax(eval_logits, dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += batch_size
                    batches += 1
        finally:
            model.train(was_training)

        if total <= 0:
            return {
                "snapshot_train_ce": float("nan"),
                "snapshot_train_acc": 0.0,
                "snapshot_train_samples": 0.0,
                "snapshot_train_batches": float(batches),
            }

        return {
            "snapshot_train_ce": float(total_loss / float(total)),
            "snapshot_train_acc": float(100.0 * correct / float(total)),
            "snapshot_train_samples": float(total),
            "snapshot_train_batches": float(batches),
        }

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: torch.nn.Module,
        optimizer,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        quality_loader = self._get_current_fisher_loader(train_loader)
        metric_split = "fisher" if quality_loader is not train_loader else "train"
        snapshot_metrics = self._estimate_snapshot_train_metrics(
            train_loader=quality_loader,
            model=model,
            class_offset=class_offset,
        )
        count = super()._capture_in_task_fisher_snapshot(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch_idx=epoch_idx,
            class_offset=class_offset,
        )
        snapshot_metrics.update(
            {
                "snapshot_epoch": float(epoch_idx),
                "snapshot_index": float(count),
            }
        )
        self._in_task_snapshot_metrics.append(snapshot_metrics)

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp][PathWeight] task=%d epoch=%d "
                "split=%s ce=%.6g acc=%.3f samples=%d batches=%d",
                self._cur_task,
                epoch_idx,
                metric_split,
                float(snapshot_metrics["snapshot_train_ce"]),
                float(snapshot_metrics["snapshot_train_acc"]),
                int(snapshot_metrics["snapshot_train_samples"]),
                int(snapshot_metrics["snapshot_train_batches"]),
            )
        return count

    def _snapshot_entries(
        self,
    ) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
        entries: List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]] = []
        for index, (epoch, fisher_snapshot) in enumerate(self._in_task_fisher_snapshots):
            metrics = (
                self._in_task_snapshot_metrics[index]
                if index < len(self._in_task_snapshot_metrics)
                else {}
            )
            entries.append((index, int(epoch), fisher_snapshot, metrics))
        return entries

    def _select_late_snapshot_entries(
        self,
    ) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
        entries = self._snapshot_entries()
        if not entries:
            self._in_task_path_weight_last_metrics = {
                "in_task_path_selected_snapshots": 0.0,
                "in_task_path_total_snapshots": 0.0,
                "in_task_path_fallback_to_last": 0.0,
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
        }
        return selected

    def _loss_softmax_weights(
        self,
        entries: List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]],
    ) -> List[float]:
        n = len(entries)
        if n <= 0:
            return []

        losses = [float(metrics.get("snapshot_train_ce", float("nan"))) for *_, metrics in entries]
        valid_losses = all(math.isfinite(loss) for loss in losses)
        if (not self._in_task_path_weighted) or self._in_task_path_beta <= 0.0 or not valid_losses:
            weights = [1.0 / float(n)] * n
            mode = "uniform"
        else:
            mean_loss = sum(losses) / float(n)
            var_loss = sum((loss - mean_loss) ** 2 for loss in losses) / float(n)
            std_loss = math.sqrt(max(var_loss, 0.0))
            if std_loss <= float(self._in_task_path_loss_eps):
                weights = [1.0 / float(n)] * n
                mode = "uniform_constant_loss"
            else:
                logits = [
                    max(
                        -60.0,
                        min(
                            60.0,
                            -float(self._in_task_path_beta)
                            * ((loss - mean_loss) / (std_loss + float(self._in_task_path_loss_eps))),
                        ),
                    )
                    for loss in losses
                ]
                weight_tensor = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0)
                weights = [float(value) for value in weight_tensor.tolist()]
                mode = "loss_softmax"

        entropy = -sum(weight * math.log(max(weight, float(self._in_task_path_loss_eps))) for weight in weights)
        eff_n = math.exp(entropy) if weights else 0.0
        self._in_task_path_weight_last_metrics.update(
            {
                "in_task_path_weighted": float(bool(self._in_task_path_weighted)),
                "in_task_path_weight_mode": mode,
                "in_task_path_beta": float(self._in_task_path_beta),
                "in_task_path_weight_min": float(min(weights) if weights else 0.0),
                "in_task_path_weight_max": float(max(weights) if weights else 0.0),
                "in_task_path_weight_entropy": float(entropy),
                "in_task_path_weight_effective_n": float(eff_n),
                "in_task_path_loss_min": float(min(losses) if valid_losses and losses else 0.0),
                "in_task_path_loss_max": float(max(losses) if valid_losses and losses else 0.0),
                "in_task_path_loss_mean": float(sum(losses) / float(n) if valid_losses else 0.0),
                "in_task_path_loss_valid": float(valid_losses),
            }
        )
        return weights

    def _mean_in_task_raw_fisher(self) -> Optional[Dict[str, torch.Tensor]]:
        selected = self._select_late_snapshot_entries()
        if not selected:
            return None

        weights = self._loss_softmax_weights(selected)
        if not weights:
            return None

        accum: Dict[str, torch.Tensor] = {}
        for weight, (_, _, fisher_snapshot, _) in zip(weights, selected):
            if self._in_task_path_normalize_snapshots:
                fisher_values = self._trace_normalize_fisher(fisher_snapshot)
            else:
                fisher_values = fisher_snapshot
            for key, value in fisher_values.items():
                value_cpu = value.detach().cpu()
                weighted_value = value_cpu * float(weight)
                if key not in accum:
                    accum[key] = weighted_value.clone()
                else:
                    if tuple(weighted_value.shape) != tuple(accum[key].shape):
                        weighted_value = self._match_shape(weighted_value, accum[key])
                    accum[key] = accum[key] + weighted_value

        path_trace = self._fisher_trace(accum)
        self._in_task_path_weight_last_metrics.update(
            {
                "in_task_path_normalize_snapshots": float(
                    bool(self._in_task_path_normalize_snapshots)
                ),
                "in_task_path_weighted_trace": float(path_trace),
            }
        )

        if self._is_main_process:
            logger.info(
                "[AS1-NormFisher-Damp][PathWeight] task=%d selected=%d/%d "
                "epochs=%s beta=%.4g mode=%s weight_min=%.4g weight_max=%.4g "
                "eff_n=%.4g path_trace=%.6g norm_snapshots=%s",
                self._cur_task,
                int(self._in_task_path_weight_last_metrics.get("in_task_path_selected_snapshots", 0)),
                int(self._in_task_path_weight_last_metrics.get("in_task_path_total_snapshots", 0)),
                [int(epoch) for _, epoch, _, _ in selected],
                float(self._in_task_path_beta),
                self._in_task_path_weight_last_metrics.get("in_task_path_weight_mode", "unknown"),
                float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_min", 0.0)),
                float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_max", 0.0)),
                float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_effective_n", 0.0)),
                float(path_trace),
                self._in_task_path_normalize_snapshots,
            )
        return accum

    # ------------------------------------------------------------------
    # Fisher damping
    # ------------------------------------------------------------------

    def _damp_trace_normalized_fisher(
        self, fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convex mix of trace-normalized Fisher with scaled identity I/d.

        F̃_i = (1 - alpha) * F_i  +  alpha / d   for every element i.
        """
        alpha = float(self._ewc_fisher_damping_alpha)
        if alpha <= 0.0 or not fisher:
            return fisher

        total_numel = sum(v.numel() for v in fisher.values())
        if total_numel <= 0:
            return fisher

        iso = alpha / float(total_numel)
        return {k: (1.0 - alpha) * v + iso for k, v in fisher.items()}

    # ------------------------------------------------------------------
    # Override: apply damping after parent accumulation + normalization
    # ------------------------------------------------------------------

    def _accumulate_fisher(
        self,
        old: Dict[str, torch.Tensor],
        new: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        result = super()._accumulate_fisher(old, new)
        damped = self._damp_trace_normalized_fisher(result)

        if (
            self._is_main_process
            and float(self._ewc_fisher_damping_alpha) > 0.0
            and damped
        ):
            damped_stats = self._summarize_tensor_dict(damped)
            logger.info(
                "[AS1-NormFisher-Damp] task=%d alpha=%.6g damped_fisher "
                "tensors=%d numel=%d max=%.3e mean=%.3e sum=%.3e",
                self._cur_task,
                float(self._ewc_fisher_damping_alpha),
                int(damped_stats["tensors"]),
                int(damped_stats["numel"]),
                float(damped_stats["max"]),
                float(damped_stats["mean"]),
                float(damped_stats["sum"]),
            )

        return damped

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics["method_type"] = "as1_normfisher_gam_intask_damp"
            self._last_mechanism_metrics["ewc_fisher_damping_alpha"] = float(
                self._ewc_fisher_damping_alpha
            )
            self._last_mechanism_metrics.update(self._in_task_path_weight_last_metrics)


EWCLoRA = Learner
