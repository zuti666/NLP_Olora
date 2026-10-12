"""Full-model EWC + GAM with normalized multi-Fisher in-task estimates.

This learner is intentionally not a projected-gradient variant.  GAM still
builds its final direction, but before the optimizer step we replace the
gradient with a direct merge:

    g_final = g_clean + lambda_flat * (g_gam - g_clean) + g_ewc + g_l2_penalty

Optionally, hard Euclidean trust-region scaling is applied to this final merged
gradient so the cumulative task displacement remains inside an L2 budget.

This file is the full-model counterpart of
``models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py``:

* selectable full-model Fisher estimators, including batch, per-sample, true
  sample, NCM, and prototype modes;
* optional weighted multi-Fisher merge;
* late in-task path Fisher selection with CE-based softmax snapshot weights;
* trace normalization and optional F+I damping on the accumulated Fisher.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models_CL.ewc_gam_projected import (
    EWCGAMProjected,
    _BATCH_FISHER_MODES,
    _as_bool,
    _flatten_optimizer_params,
)
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


_FISHER_MODE_ALIASES = {
    "full": "batch_empirical",
    "max_pred": "max_pred_batch_empirical",
    "maxpred": "max_pred_batch_empirical",
    "sample": "sample_empirical",
    "lr_sample": "lr_sample_empirical",
    "true": "true_sample",
    "ncm": "ncm_empirical",
    "ncm_batch": "ncm_batch_empirical",
}
_SAMPLE_FISHER_MODES = {"sample_empirical", "lr_sample_empirical", "true_sample"}
_NCM_FISHER_MODES = {"ncm_empirical", "ncm_batch_empirical"}
_PROTOTYPE_FISHER_MODES = {"prototype", "prototype_all"}
_SUPPORTED_FISHER_MODES = (
    set(_BATCH_FISHER_MODES)
    | _SAMPLE_FISHER_MODES
    | _NCM_FISHER_MODES
    | _PROTOTYPE_FISHER_MODES
)

_FISHER_ESTIMATOR_INFO = {
    "batch_empirical": {
        "display_name": "batch-gradient Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "max_pred_batch_empirical": {
        "display_name": "max-pred batch-gradient Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "lr_batch_empirical": {
        "display_name": "logit-reversed batch-gradient Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "sample_empirical": {
        "display_name": "per-sample empirical Fisher",
        "estimator_family": "per_sample_empirical_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_gradients",
        "is_per_sample_empirical_fisher": True,
    },
    "lr_sample_empirical": {
        "display_name": "logit-reversed per-sample empirical Fisher",
        "estimator_family": "per_sample_empirical_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_gradients",
        "is_per_sample_empirical_fisher": True,
    },
    "true_sample": {
        "display_name": "model-distribution Fisher diagonal",
        "estimator_family": "model_distribution_fisher",
        "gradient_aggregation": "mean_of_per_sample_squared_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "ncm_empirical": {
        "display_name": "NCM batch-gradient Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "ncm_batch_empirical": {
        "display_name": "NCM batch-gradient Fisher proxy",
        "estimator_family": "batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "prototype": {
        "display_name": "prototype batch-gradient Fisher proxy",
        "estimator_family": "prototype_batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
    "prototype_all": {
        "display_name": "all-prototype batch-gradient Fisher proxy",
        "estimator_family": "prototype_batch_gradient_square_proxy",
        "gradient_aggregation": "square_of_batch_mean_gradient",
        "is_per_sample_empirical_fisher": False,
    },
}


def _canonical_fisher_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    return _FISHER_MODE_ALIASES.get(mode, mode)


def _fisher_estimator_info(mode: str) -> Dict:
    info = dict(_FISHER_ESTIMATOR_INFO.get(str(mode), {}))
    info.setdefault("display_name", str(mode))
    info.setdefault("estimator_family", "unknown")
    info.setdefault("gradient_aggregation", "unknown")
    info.setdefault("is_per_sample_empirical_fisher", False)
    return info


class EWCGAMMergedInTaskMultiFisherNorm(EWCGAMProjected):
    """Full-model normalized multi-Fisher + L2-constrained gradient merge.

    Unlike :class:`EWCGAMProjected`, this class does not project ``g_gam`` away
    from ``g_ewc``.  It only merges final gradient components.
    """

    def __init__(self, args):
        fisher_type = str(args.get("fisher_type", "")).strip().lower()
        default_fisher_mode = "ncm_empirical" if fisher_type == "ncm" else "batch_empirical"
        raw_requested_fisher_mode = args.get("fisher_mode", default_fisher_mode)
        if isinstance(raw_requested_fisher_mode, (list, tuple)):
            first_requested_fisher_mode = raw_requested_fisher_mode[0] if raw_requested_fisher_mode else default_fisher_mode
        else:
            first_requested_fisher_mode = str(raw_requested_fisher_mode).split(",")[0].strip()
        requested_fisher_mode = _canonical_fisher_mode(first_requested_fisher_mode or default_fisher_mode)
        parent_args = args
        raw_requested_is_multi = isinstance(raw_requested_fisher_mode, (list, tuple)) or "," in str(raw_requested_fisher_mode)
        parent_supported_modes = set(_BATCH_FISHER_MODES) | _PROTOTYPE_FISHER_MODES
        if raw_requested_is_multi or requested_fisher_mode not in parent_supported_modes:
            parent_args = dict(args)
            parent_args["fisher_mode"] = (
                requested_fisher_mode
                if requested_fisher_mode in parent_supported_modes
                else "batch_empirical"
            )
        super().__init__(parent_args)
        self.args = args
        self.fisher_mode = requested_fisher_mode

        self._lambda_flat = float(args.get("lambda_flat", 1.0))
        self.ewc_weight = float(args.get("ewc_lambda", args.get("lambda", self.ewc_weight)))
        self.gamma = float(args.get("ewc_gamma", args.get("gamma", self.gamma)))
        self._normalize_fisher = _as_bool(args.get("ewc_normalize_fisher", True))
        self._normalize_accumulated_fisher = _as_bool(
            args.get("ewc_normalize_accumulated_fisher", args.get("ewc_normalize_past_fisher", False))
        )
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))
        self._fisher_past_trace_mass = 0.0
        self._ewc_include_classifier = _as_bool(
            args.get("ewc_include_classifier", args.get("ewc_include_fc", False))
        )
        self._fisher_max_batches = self._optional_positive_int(
            args.get("ewc_max_batches", args.get("fisher_max_batches", None))
        )
        self._ncm_fisher_tau = float(
            args.get("ncm_fisher_tau", args.get("fisher_ncm_tau", args.get("ncm_temperature", 16.0)))
        )
        self._last_fisher_estimation_metrics: Dict[str, float] = {}
        self._last_fisher_norm_metrics: Dict[str, float] = {}
        self._last_multi_fisher_metrics: Dict[str, float] = {}
        self._last_fisher_damping_metrics: Dict[str, float] = {}

        raw_fisher_modes = args.get(
            "ewc_fisher_modes",
            args.get(
                "fisher_modes",
                args.get(
                    "multi_fisher_modes",
                    raw_requested_fisher_mode if "," in str(raw_requested_fisher_mode) else None,
                ),
            ),
        )
        self._fisher_modes = self._parse_fisher_modes(raw_fisher_modes, requested_fisher_mode)
        self._fisher_mode_weights = self._parse_fisher_mode_weights(
            args.get("ewc_fisher_mode_weights", args.get("fisher_mode_weights", None)),
            len(self._fisher_modes),
        )
        self._multi_fisher_normalize_each = _as_bool(
            args.get("ewc_multi_fisher_normalize_each", args.get("multi_fisher_normalize_each", self._normalize_fisher))
        )
        self._multi_fisher_merge = str(
            args.get("ewc_multi_fisher_merge", args.get("multi_fisher_merge", "weighted_mean"))
        ).strip().lower()
        self._uses_multi_fisher = len(self._fisher_modes) > 1
        self._uses_all_seen_fisher = "prototype_all" in self._fisher_modes
        self.fisher_mode = self._fisher_modes[0]

        self._approximate_fisher_in_task = _as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get("approximate_fisher_inTask", args.get("approximate_fisher_in_task", False)),
            )
        )
        self._in_task_fisher_interval = int(
            args.get(
                "in_task_fisher_interval",
                args.get("approximate_fisher_interval", args.get("approximte_fisher_interval", 10)),
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
                            args.get("approximate_fisher_omega", args.get("approximte_fisher_omega", 0.5)),
                        ),
                    )
                ),
            ),
        )
        self._in_task_fisher_start_epoch = self._optional_positive_int(
            args.get("in_task_fisher_start_epoch", args.get("path_fisher_start_epoch", None))
        )
        self._in_task_fisher_max_snapshots = self._optional_positive_int(
            args.get("in_task_fisher_max_snapshots", args.get("path_fisher_max_snapshots", None))
        )
        self._in_task_path_start_ratio = min(
            1.0,
            max(
                0.0,
                float(args.get("in_task_path_start_ratio", args.get("path_fisher_start_ratio", 0.6))),
            ),
        )
        self._in_task_path_beta = max(
            0.0,
            float(args.get("in_task_path_beta", args.get("path_fisher_beta", 3.0))),
        )
        self._in_task_path_weighted = _as_bool(
            args.get("in_task_path_weighted", args.get("path_fisher_weighted", True))
        )
        self._in_task_path_normalize_snapshots = _as_bool(
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

        self._in_task_weight_merge = _as_bool(
            args.get("in_task_weight_merge", args.get("weight_merge_with_in_task_fisher", False))
        )
        self._in_task_weight_merge_start_task = int(args.get("in_task_weight_merge_start_task", 0))
        self._in_task_weight_merge_exclude_classifier = _as_bool(
            args.get("in_task_weight_merge_exclude_classifier", True)
        )

        self._euc_trust = _as_bool(
            args.get("ewc_euclidean_trust", args.get("euclidean_trust", False))
        )
        self._euc_radius = self._parse_pos_float(
            args.get("ewc_euclidean_radius", args.get("euclidean_radius", None))
        )
        self._euc_radius_from_gam = _as_bool(args.get("ewc_euclidean_radius_from_gam", False))
        self._euc_radius_source = str(args.get("ewc_euclidean_radius_source", "gam_grad_rho")).lower()
        self._euc_radius_scale = float(args.get("ewc_euclidean_radius_scale", 1.0))
        self._euc_outside_action = str(args.get("ewc_euclidean_outside_action", "zero")).lower()
        self._euc_exclude_classifier = _as_bool(args.get("ewc_euclidean_exclude_classifier", True))
        self._euc_eps = float(args.get("ewc_euclidean_eps", 1e-12))

        self._dpen_active = _as_bool(
            args.get("ewc_delta_penalty", args.get("delta_w_penalty", False))
        )
        self._dpen_lambda = float(
            args.get("ewc_delta_penalty_lambda", args.get("delta_w_penalty_lambda", 0.0))
        )
        self._dpen_radius = self._parse_pos_float(
            args.get("ewc_delta_penalty_radius", args.get("delta_w_penalty_radius", None))
        )
        self._dpen_radius_from_gam = _as_bool(args.get("ewc_delta_penalty_radius_from_gam", False))
        self._dpen_radius_source = str(args.get("ewc_delta_penalty_radius_source", "gam_grad_rho")).lower()
        self._dpen_radius_scale = float(args.get("ewc_delta_penalty_radius_scale", 1.0))
        self._dpen_exclude_classifier = _as_bool(args.get("ewc_delta_penalty_exclude_classifier", True))
        self._dpen_eps = float(args.get("ewc_delta_penalty_eps", 1e-12))
        alpha_raw = float(
            args.get(
                "ewc_fisher_damping_alpha",
                args.get("fisher_damping_alpha", 1e-3),
            )
        )
        self._ewc_fisher_damping_alpha = max(0.0, min(1.0, alpha_raw))

        if self._dpen_active and self._euc_trust:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] ewc_delta_penalty=True disables hard "
                "ewc_euclidean_trust."
            )
            self._euc_trust = False
        if self._in_task_path_normalize_snapshots and not self._normalize_fisher:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] path_fisher_normalize_snapshots=True "
                "but ewc_normalize_fisher=False; disabling per-snapshot path normalization."
            )
            self._in_task_path_normalize_snapshots = False
        if self._ewc_fisher_damping_alpha > 0.0 and not self._normalize_accumulated_fisher:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] ewc_fisher_damping_alpha=%.4g is set "
                "but ewc_normalize_accumulated_fisher=False; the I/d floor may be "
                "negligible relative to un-normalized Fisher values.",
                self._ewc_fisher_damping_alpha,
            )
        if self._uses_all_seen_fisher and self._uses_multi_fisher:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] prototype_all is mixed with other "
                "Fisher modes; accumulated omega will be replaced after each task to "
                "avoid double-counting prior tasks."
            )

        self._param_reference: Dict[int, torch.Tensor] = {}
        self._in_task_fisher_snapshots: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_weight_snapshots: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}
        self._in_task_weight_merge_last_metrics: Dict[str, float] = {}
        self._last_task_new_fisher_stats: Dict[str, float] = {}
        self._train_log_sums: Dict[str, float] = {}
        self._train_log_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}

        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm] lambda=%.6g lambda_flat=%.6g gamma=%.4g "
            "normalize_fisher=%s normalize_accumulated=%s fisher_modes=%s "
            "mode_weights=%s multi_norm_each=%s damping_alpha=%.4g in_task=%s "
            "interval=%d omega=%.4g path_start_ratio=%.3g path_beta=%.3g "
            "euc_trust=%s euc_radius=%s dpen=%s dpen_lambda=%.4g dpen_radius=%s",
            self.ewc_weight,
            self._lambda_flat,
            self.gamma,
            self._normalize_fisher,
            self._normalize_accumulated_fisher,
            self._fisher_modes,
            self._fisher_mode_weights,
            self._multi_fisher_normalize_each,
            self._ewc_fisher_damping_alpha,
            self._approximate_fisher_in_task,
            self._in_task_fisher_interval,
            self._in_task_fisher_omega,
            self._in_task_path_start_ratio,
            self._in_task_path_beta,
            self._euc_trust,
            self._resolve_euc_radius(),
            self._dpen_active,
            self._dpen_lambda,
            self._resolve_dpen_radius(),
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _parse_pos_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false", "off"}:
            return None
        parsed = float(value)
        return parsed if parsed > 0.0 else None

    @staticmethod
    def _is_classifier_param_name(name: str) -> bool:
        if name.startswith(("fc.", "head.", "classifier.")):
            return True
        return name.endswith(
            (
                ".fc.weight",
                ".fc.bias",
                ".head.weight",
                ".head.bias",
                ".classifier.weight",
                ".classifier.bias",
            )
        ) or ".classifiers." in name

    @staticmethod
    def _parse_fisher_modes(value, default_mode: str) -> List[str]:
        if value is None:
            raw_modes = [default_mode]
        elif isinstance(value, (list, tuple)):
            raw_modes = list(value)
        else:
            raw_modes = [item.strip() for item in str(value).split(",") if item.strip()]
        modes: List[str] = []
        for raw_mode in raw_modes:
            mode = _canonical_fisher_mode(str(raw_mode))
            if mode not in _SUPPORTED_FISHER_MODES:
                raise ValueError(
                    f"Unsupported full-model fisher_mode={mode!r}. "
                    "Use batch_empirical, max_pred_batch_empirical, lr_batch_empirical, "
                    "sample_empirical, lr_sample_empirical, true_sample, ncm_empirical, "
                    "ncm_batch_empirical, prototype, or prototype_all."
                )
            if mode not in modes:
                modes.append(mode)
        return modes or [_canonical_fisher_mode(default_mode)]

    @staticmethod
    def _parse_fisher_mode_weights(value, count: int) -> List[float]:
        count = int(count)
        if count <= 0:
            return []
        if value is None:
            return [1.0 / float(count)] * count
        if isinstance(value, (list, tuple)):
            raw_weights = [float(v) for v in value]
        else:
            raw_weights = [float(item.strip()) for item in str(value).split(",") if item.strip()]
        if len(raw_weights) != count:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] fisher_mode_weights length %d "
                "does not match fisher_modes length %d; using uniform weights.",
                len(raw_weights),
                count,
            )
            return [1.0 / float(count)] * count
        weights = [max(0.0, float(v)) for v in raw_weights]
        total = sum(weights)
        if total <= 0.0:
            logger.warning(
                "[EWCGAMMergedInTaskMultiFisherNorm] fisher_mode_weights sum is non-positive; "
                "using uniform weights."
            )
            return [1.0 / float(count)] * count
        return [float(v / total) for v in weights]

    @staticmethod
    def _match_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value_cpu = value.detach().cpu().to(dtype=reference.dtype)
        ref_cpu = reference.detach().cpu()
        if tuple(value_cpu.shape) == tuple(ref_cpu.shape):
            return value_cpu

        matched = torch.zeros_like(ref_cpu)
        if value_cpu.ndim != ref_cpu.ndim:
            return matched
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref_cpu.shape[dim])))
            for dim in range(value_cpu.ndim)
        )
        matched[slices] = value_cpu[slices]
        return matched

    @staticmethod
    def _fisher_trace(fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        return float(sum(float(v.detach().float().sum().item()) for v in fisher.values()))

    def _trace_normalize_fisher(self, fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_trace = self._fisher_trace(fisher)
        if not fisher:
            self._last_fisher_norm_metrics = {
                "ewc_normalize_fisher": bool(self._normalize_fisher),
                "fisher_raw_trace": float(raw_trace),
                "fisher_normalized_trace": float(raw_trace),
            }
            return fisher
        denom = raw_trace + float(self._fisher_norm_eps)
        normalized = {name: value / denom for name, value in fisher.items()}
        self._last_fisher_norm_metrics = {
            "ewc_normalize_fisher": True,
            "ewc_fisher_norm_eps": float(self._fisher_norm_eps),
            "fisher_raw_trace": float(raw_trace),
            "fisher_normalized_trace": float(self._fisher_trace(normalized)),
        }
        return normalized

    @staticmethod
    def _summarize_tensor_dict(tensors: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        if not tensors:
            return {"tensors": 0, "numel": 0, "max": 0.0, "mean": 0.0, "sum": 0.0}

        total_sum = 0.0
        total_numel = 0
        max_value = 0.0
        for value in tensors.values():
            value_f = value.detach().float()
            if value_f.numel() == 0:
                continue
            total_sum += float(value_f.sum().item())
            total_numel += int(value_f.numel())
            max_value = max(max_value, float(value_f.max().item()))
        return {
            "tensors": int(len(tensors)),
            "numel": int(total_numel),
            "max": float(max_value),
            "mean": float(total_sum / max(total_numel, 1)),
            "sum": float(total_sum),
        }

    def _named_trainable_params(self):
        net = self._net()
        yield from ((name, p) for name, p in net.named_parameters() if p.requires_grad)

    def _iter_fisher_params(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not self._ewc_include_classifier and self._is_classifier_param_name(name):
                continue
            yield name, param

    def _new_fisher_buffer(self, model) -> Dict[str, torch.Tensor]:
        return {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in self._iter_fisher_params(model)
        }

    def _iter_loader_batches(self, loader):
        for batch_idx, batch in enumerate(loader):
            if self._fisher_max_batches is not None and batch_idx >= int(self._fisher_max_batches):
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            yield batch_idx, inputs, targets

    def _current_task_inputs(self, inputs, targets, task_start: int, task_end: int):
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
        if mask.numel() == 0:
            return None, None
        inputs = torch.index_select(inputs, 0, mask)
        local_targets = torch.index_select(targets, 0, mask) - int(task_start)
        return inputs, local_targets

    def _set_fisher_estimation_metrics(
        self,
        mode: str,
        *,
        num_samples: int,
        loss_reduction: str,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        info = _fisher_estimator_info(mode)
        metrics = {
            "fisher_mode": str(mode),
            "fisher_estimator": str(mode),
            "fisher_estimator_display": info["display_name"],
            "fisher_estimator_family": info["estimator_family"],
            "fisher_gradient_aggregation": info["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(info["is_per_sample_empirical_fisher"]),
            "fisher_num_samples": int(num_samples),
            "fisher_loss_reduction": str(loss_reduction),
        }
        if extra:
            metrics.update(extra)
        self._last_fisher_estimation_metrics = metrics

    def _compute_batch_fisher_mode(
        self,
        mode: str,
        loader,
        model,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        try:
            for _, inputs, targets in tqdm(
                self._iter_loader_batches(loader), desc=f"Computing Fisher [{mode}]"
            ):
                inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
                if inputs is None:
                    continue
                model.zero_grad(set_to_none=True)
                logits = model(inputs)["logits"][:, task_start:task_end]
                if mode == "max_pred_batch_empirical":
                    fisher_logits = logits
                    fisher_targets = logits.argmax(dim=1).detach()
                else:
                    fisher_logits = -logits if mode == "lr_batch_empirical" else logits
                    fisher_targets = local_targets
                F.cross_entropy(fisher_logits, fisher_targets).backward()
                batch_size = int(inputs.size(0))
                for name, param in self._iter_fisher_params(model):
                    if param.grad is not None and name in fisher:
                        fisher[name] += param.grad.detach().cpu().pow(2) * batch_size
                num_samples += batch_size
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        self._set_fisher_estimation_metrics(
            mode, num_samples=num_samples, loss_reduction="cross_entropy_mean"
        )
        return fisher

    def _compute_sample_fisher_mode(
        self,
        mode: str,
        loader,
        model,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        try:
            for _, inputs, targets in tqdm(self._iter_loader_batches(loader), desc=f"Computing Fisher [{mode}]"):
                inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
                if inputs is None:
                    continue
                for idx in range(int(inputs.size(0))):
                    x_i = inputs[idx : idx + 1]
                    y_i = local_targets[idx : idx + 1]
                    model.zero_grad(set_to_none=True)
                    logits = model(x_i)["logits"][:, task_start:task_end]
                    if mode == "true_sample":
                        probs = torch.softmax(logits.detach(), dim=-1)
                        fisher_targets = torch.multinomial(probs, num_samples=1).squeeze(1)
                        fisher_logits = logits
                    else:
                        fisher_targets = y_i
                        fisher_logits = -logits if mode == "lr_sample_empirical" else logits
                    F.cross_entropy(fisher_logits, fisher_targets).backward()
                    for name, param in self._iter_fisher_params(model):
                        if param.grad is not None and name in fisher:
                            fisher[name] += param.grad.detach().cpu().pow(2)
                    num_samples += 1
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        self._set_fisher_estimation_metrics(
            mode, num_samples=num_samples, loss_reduction="per_sample_cross_entropy"
        )
        return fisher

    @torch.no_grad()
    def _current_task_feature_prototypes(
        self,
        loader,
        model,
        task_start: int,
        task_end: int,
    ) -> torch.Tensor:
        was_training = bool(model.training)
        model.eval()
        sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        try:
            for _, inputs, targets in self._iter_loader_batches(loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                targets_m = torch.index_select(targets, 0, mask)
                if hasattr(model, "extract_vector"):
                    features = model.extract_vector(inputs)
                else:
                    outputs = model(inputs)
                    if not isinstance(outputs, dict) or "features" not in outputs:
                        raise ValueError("NCM Fisher requires model features or extract_vector().")
                    features = outputs["features"]
                if features.dim() > 2:
                    features = features.reshape(features.size(0), -1)
                features = features.detach().float().cpu()
                for feat, target in zip(features, targets_m.detach().cpu()):
                    local = int(target.item()) - int(task_start)
                    if local not in sums:
                        sums[local] = feat.clone()
                        counts[local] = 1
                    else:
                        sums[local] += feat
                        counts[local] += 1
        finally:
            model.train(was_training)

        num_classes = int(task_end) - int(task_start)
        if not sums:
            raise ValueError("NCM Fisher could not build prototypes: no current-task samples found.")
        feat_dim = int(next(iter(sums.values())).numel())
        prototypes = torch.zeros(num_classes, feat_dim, dtype=torch.float32)
        for local in range(num_classes):
            if local in sums and counts.get(local, 0) > 0:
                prototypes[local] = sums[local] / float(counts[local])
        return F.normalize(prototypes.to(self._device), p=2, dim=1, eps=1e-12)

    def _ncm_logits(self, model, inputs: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        outputs = model(inputs)
        if isinstance(outputs, dict) and "features" in outputs:
            features = outputs["features"]
        elif hasattr(model, "extract_vector"):
            features = model.extract_vector(inputs)
        else:
            raise ValueError("NCM Fisher requires model features or extract_vector().")
        if features.dim() > 2:
            features = features.reshape(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1, eps=1e-12)
        return float(self._ncm_fisher_tau) * features.matmul(prototypes.t())

    def _compute_ncm_fisher_mode(
        self,
        mode: str,
        loader,
        model,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        prototypes = self._current_task_feature_prototypes(loader, model, task_start, task_end)
        was_training = bool(model.training)
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        try:
            for _, inputs, targets in tqdm(
                self._iter_loader_batches(loader), desc=f"Computing Fisher [{mode}]"
            ):
                inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
                if inputs is None:
                    continue
                model.zero_grad(set_to_none=True)
                logits = self._ncm_logits(model, inputs, prototypes)
                F.cross_entropy(logits, local_targets).backward()
                batch_size = int(inputs.size(0))
                for name, param in self._iter_fisher_params(model):
                    if param.grad is not None and name in fisher:
                        fisher[name] += param.grad.detach().cpu().pow(2) * batch_size
                num_samples += batch_size
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        self._set_fisher_estimation_metrics(
            mode,
            num_samples=num_samples,
            loss_reduction="ncm_cross_entropy_mean",
            extra={
                "ncm_fisher_tau": float(self._ncm_fisher_tau),
                "ncm_fisher_num_classes": int(task_end - task_start),
            },
        )
        return fisher

    def _compute_fisher_from_absolute_loader(
        self,
        loader,
        model,
        all_classes: int,
        mode: str = "batch_empirical",
    ) -> Dict[str, torch.Tensor]:
        if loader is None:
            return {}
        was_training = bool(model.training)
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        try:
            for batch in tqdm(loader, desc=f"Computing Fisher [{mode}]"):
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                logits = model(inputs)["logits"][:, :all_classes]
                if mode == "max_pred_batch_empirical":
                    fisher_logits = logits
                    fisher_targets = logits.argmax(dim=1).detach()
                else:
                    fisher_logits = -logits if mode == "lr_batch_empirical" else logits
                    fisher_targets = targets
                F.cross_entropy(fisher_logits, fisher_targets).backward()
                batch_size = int(inputs.size(0))
                for name, param in self._iter_fisher_params(model):
                    if param.grad is not None and name in fisher:
                        fisher[name] += param.grad.detach().cpu().pow(2) * batch_size
                num_samples += batch_size
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        self._set_fisher_estimation_metrics(
            mode,
            num_samples=num_samples,
            loss_reduction="prototype_cross_entropy_mean",
        )
        return fisher

    def _compute_fisher_by_mode(
        self,
        mode: str,
        loader,
        model,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        mode = _canonical_fisher_mode(mode)
        if mode in _BATCH_FISHER_MODES:
            return self._compute_batch_fisher_mode(mode, loader, model, task_start, task_end)
        if mode in _SAMPLE_FISHER_MODES:
            return self._compute_sample_fisher_mode(mode, loader, model, task_start, task_end)
        if mode in _NCM_FISHER_MODES:
            return self._compute_ncm_fisher_mode(mode, loader, model, task_start, task_end)
        if mode == "prototype":
            self._build_prototype_set(model, task_start, task_end)
            fisher = self._compute_fisher_from_absolute_loader(
                self._make_prototype_loader(range(task_start, task_end)),
                model,
                all_classes=self._total_classes,
                mode="batch_empirical",
            )
            self._set_fisher_estimation_metrics(
                mode,
                num_samples=int(self._last_fisher_estimation_metrics.get("fisher_num_samples", 0)),
                loss_reduction="prototype_cross_entropy_mean",
            )
            return fisher
        if mode == "prototype_all":
            self._build_prototype_set(model, task_start, task_end)
            fisher = self._compute_fisher_from_absolute_loader(
                self._make_prototype_loader(range(0, self._total_classes)),
                model,
                all_classes=self._total_classes,
                mode="batch_empirical",
            )
            self._set_fisher_estimation_metrics(
                mode,
                num_samples=int(self._last_fisher_estimation_metrics.get("fisher_num_samples", 0)),
                loss_reduction="prototype_all_cross_entropy_mean",
            )
            return fisher
        raise ValueError(f"Unsupported full-model fisher_mode={mode!r}")

    def _combine_fisher_estimates(
        self,
        fishers: List[Dict[str, torch.Tensor]],
        weights: List[float],
        modes: List[str],
    ) -> Dict[str, torch.Tensor]:
        if not fishers:
            self._last_multi_fisher_metrics = {
                "multi_fisher_enabled": 0.0,
                "multi_fisher_num_modes": 0.0,
            }
            return {}
        if len(fishers) == 1:
            mode = modes[0] if modes else str(self.fisher_mode)
            trace = self._fisher_trace(fishers[0])
            self._last_multi_fisher_metrics = {
                "multi_fisher_enabled": 0.0,
                "multi_fisher_num_modes": 1.0,
                "multi_fisher_modes": mode,
                "multi_fisher_mode_0_trace": float(trace),
                "multi_fisher_mode_0_weight": 1.0,
            }
            return {name: value.detach().cpu().clone() for name, value in fishers[0].items()}

        accum: Dict[str, torch.Tensor] = {}
        mode_traces: Dict[str, float] = {}
        for idx, (mode, fisher, weight) in enumerate(zip(modes, fishers, weights)):
            mode_traces[f"multi_fisher_mode_{idx}_trace"] = float(self._fisher_trace(fisher))
            mode_traces[f"multi_fisher_mode_{idx}_weight"] = float(weight)
            src = self._trace_normalize_fisher(fisher) if self._multi_fisher_normalize_each else fisher
            for name, value in src.items():
                value_cpu = value.detach().cpu() * float(weight)
                if name not in accum:
                    accum[name] = value_cpu.clone()
                else:
                    if tuple(value_cpu.shape) != tuple(accum[name].shape):
                        value_cpu = self._match_shape(value_cpu, accum[name])
                    accum[name] = accum[name] + value_cpu

        self._last_multi_fisher_metrics = {
            "multi_fisher_enabled": 1.0,
            "multi_fisher_num_modes": float(len(fishers)),
            "multi_fisher_modes": ",".join(modes),
            "multi_fisher_merge": self._multi_fisher_merge,
            "multi_fisher_normalize_each": float(bool(self._multi_fisher_normalize_each)),
            "multi_fisher_merged_trace": float(self._fisher_trace(accum)),
            **mode_traces,
        }
        return accum

    def _compute_fisher_for_modes(
        self,
        loader,
        model,
        task_start: int,
        task_end: int,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        modes = list(modes or self._fisher_modes)
        fishers: List[Dict[str, torch.Tensor]] = []
        previous_metrics = {}
        for mode in modes:
            logger.info(
                "[EWCGAMMergedInTaskMultiFisherNorm] Computing Fisher mode=%s task=%d",
                mode,
                self._cur_task,
            )
            fisher = self._compute_fisher_by_mode(mode, loader, model, task_start, task_end)
            fishers.append(fisher)
            previous_metrics = dict(self._last_fisher_estimation_metrics)
        combined = self._combine_fisher_estimates(fishers, self._fisher_mode_weights, modes)
        if len(modes) > 1:
            self._last_fisher_estimation_metrics = {
                "fisher_mode": ",".join(modes),
                "fisher_estimator": "multi_fisher",
                "fisher_estimator_display": "weighted multi-Fisher",
                "fisher_estimator_family": "multi_fisher",
                "fisher_gradient_aggregation": "weighted_sum",
                "fisher_is_per_sample_empirical": any(
                    bool(_fisher_estimator_info(mode)["is_per_sample_empirical_fisher"])
                    for mode in modes
                ),
                **self._last_multi_fisher_metrics,
            }
        else:
            self._last_fisher_estimation_metrics = previous_metrics
        return combined

    def _snapshot_param_reference(self, param_groups) -> None:
        name_map = {id(p): name for name, p in self._named_trainable_params()}
        refs: Dict[int, torch.Tensor] = {}
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pname = name_map.get(id(p), "")
                if self._euc_exclude_classifier and self._is_classifier_param_name(pname):
                    continue
                refs[id(p)] = p.detach().cpu().clone()
        self._param_reference = refs

    def _snapshot_model_weights(self) -> Dict[str, torch.Tensor]:
        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in self._named_trainable_params():
            if self._in_task_weight_merge_exclude_classifier and self._is_classifier_param_name(name):
                continue
            snapshot[name] = param.detach().cpu().clone()
        return snapshot

    def _reset_train_log_stats(self) -> None:
        self._train_log_sums = {}
        self._train_log_count = 0

    def _record_train_log_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                self._train_log_sums[key] = self._train_log_sums.get(key, 0.0) + float(value)
        self._train_log_count += 1

    def _summarize_train_log_stats(self) -> Dict[str, float]:
        if self._train_log_count <= 0:
            return {}
        denom = float(self._train_log_count)
        return {f"train_{key}_mean": value / denom for key, value in sorted(self._train_log_sums.items())}

    # ------------------------------------------------------------------
    # Radius and L2 controls
    # ------------------------------------------------------------------

    def _resolve_euc_radius(self) -> Optional[float]:
        if self._euc_radius is not None:
            return float(self._euc_radius)
        if not self._euc_radius_from_gam:
            return None
        src = self._euc_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self.args.get("gam_grad_rho", 0.0))
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self.args.get("gam_grad_norm_rho", 0.0))
        else:
            return None
        return float(self._euc_radius_scale * rho) if rho > 0.0 else None

    def _resolve_dpen_radius(self) -> Optional[float]:
        if self._dpen_radius is not None:
            return float(self._dpen_radius)
        if not self._dpen_radius_from_gam:
            return None
        src = self._dpen_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self.args.get("gam_grad_rho", 0.0))
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self.args.get("gam_grad_norm_rho", 0.0))
        else:
            return None
        return float(self._dpen_radius_scale * rho) if rho > 0.0 else None

    def _default_dpen_stats(self) -> Dict[str, float]:
        radius = self._resolve_dpen_radius()
        return {
            "l2_penalty_enabled": float(self._dpen_active),
            "l2_penalty_configured": 0.0,
            "l2_penalty_active": 0.0,
            "l2_penalty_loss": 0.0,
            "l2_penalty_lambda": float(self._dpen_lambda),
            "l2_penalty_budget": float(radius or 0.0),
            "l2_penalty_drift_norm": 0.0,
            "l2_penalty_drift_sq": 0.0,
            "l2_penalty_ratio": 0.0,
            "l2_penalty_excess": 0.0,
            "l2_penalty_grad_norm": 0.0,
            "l2_penalty_tensors": 0.0,
        }

    def _delta_penalty_loss(self):
        stats = self._default_dpen_stats()
        zero = torch.zeros((), device=self._device)
        if not self._dpen_active or int(self._cur_task) <= 0 or not self._param_reference:
            return zero, stats

        rho = self._resolve_dpen_radius()
        lam = float(self._dpen_lambda)
        if rho is None or lam <= 0.0:
            return zero, stats

        eps = float(self._dpen_eps)
        delta_norm_sq = None
        tensor_count = 0
        for name, param in self._named_trainable_params():
            if self._dpen_exclude_classifier and self._is_classifier_param_name(name):
                continue
            ref = self._param_reference.get(id(param))
            if ref is None:
                continue
            delta = param.float() - ref.to(device=param.device, dtype=torch.float32)
            term = delta.pow(2).sum()
            delta_norm_sq = term if delta_norm_sq is None else delta_norm_sq + term
            tensor_count += 1

        if delta_norm_sq is None:
            stats["l2_penalty_configured"] = 1.0
            return zero, stats

        delta_norm = torch.sqrt(delta_norm_sq + eps)
        excess = torch.clamp(delta_norm - float(rho), min=0.0)
        penalty = lam * excess.pow(2)

        drift_norm = float(delta_norm.detach().item())
        drift_sq = float(delta_norm_sq.detach().item())
        excess_value = float(excess.detach().item())
        stats.update(
            {
                "l2_penalty_configured": 1.0,
                "l2_penalty_active": float(tensor_count > 0 and excess_value > 0.0),
                "l2_penalty_loss": float(penalty.detach().item()),
                "l2_penalty_budget": float(rho),
                "l2_penalty_drift_norm": drift_norm,
                "l2_penalty_drift_sq": drift_sq,
                "l2_penalty_ratio": float(drift_norm / (float(rho) + eps)),
                "l2_penalty_excess": excess_value,
                "l2_penalty_tensors": float(tensor_count),
            }
        )
        return penalty, stats

    def _compute_delta_penalty_grad_cache(self, params):
        with torch.enable_grad():
            penalty, stats = self._delta_penalty_loss()

        value = float(penalty.detach().item())
        if stats.get("l2_penalty_active", 0.0) <= 0.0 or not penalty.requires_grad:
            return value, {}, stats

        grads = torch.autograd.grad(penalty, params, allow_unused=True)
        grad_cache: Dict[int, torch.Tensor] = {}
        grad_norm_sq = 0.0
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_detached = grad.detach().clone()
            grad_cache[id(param)] = grad_detached
            grad_norm_sq += float(grad_detached.pow(2).sum().item())
        stats["l2_penalty_grad_norm"] = float(grad_norm_sq ** 0.5)
        return value, grad_cache, stats

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
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                total += 1
                ref = self._param_reference.get(id(param))
                if ref is None:
                    continue
                delta = param.detach().float() - ref.to(device=param.device, dtype=torch.float32)
                direction = param_to_v.get(id(param))
                step = (
                    float(lr) * direction.to(device=param.device, dtype=torch.float32)
                    if direction is not None
                    else torch.zeros_like(delta)
                )
                delta_norm_sq += float(delta.pow(2).sum().item())
                step_norm_sq += float(step.pow(2).sum().item())
                delta_dot_step += float((delta * step).sum().item())
                matched += 1

        A = step_norm_sq
        B = -2.0 * delta_dot_step
        C = delta_norm_sq - float(rho) ** 2
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
                roots = [(-B - sq) / (2.0 * A + eps), (-B + sq) / (2.0 * A + eps)]
                lower = min(roots)
                upper = max(roots)
                lo = max(0.0, lower)
                hi = min(1.0, upper)
                if lo <= hi:
                    scale = max(0.0, min(1.0, hi))
                    found = True

        if not found:
            scale = 0.0 if self._euc_outside_action == "zero" else 1.0

        after_sq = max(delta_norm_sq - 2.0 * scale * delta_dot_step + scale * scale * step_norm_sq, 0.0)
        rho_eps = float(rho) + eps
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
    # Gradient merge step
    # ------------------------------------------------------------------

    def _compute_clean_task_grad_cache(self, params, inputs, local_targets):
        self._network.zero_grad()
        with torch.enable_grad():
            _, ce_loss = self._compute_task_loss(inputs, local_targets)
        grads = torch.autograd.grad(ce_loss, params, allow_unused=True)
        cache = {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }
        self._network.zero_grad()
        return float(ce_loss.detach().item()), cache

    @staticmethod
    def _cosine_from_sums(dot: float, norm_a_sq: float, norm_b_sq: float, eps: float) -> float:
        return float(dot / ((norm_a_sq ** 0.5) * (norm_b_sq ** 0.5) + eps))

    def _build_merged_gradient_projector(
        self,
        clean_grad_cache,
        ewc_grad_cache,
        delta_penalty_grad_cache,
        step_stats: Dict[str, float],
    ):
        lambda_flat = float(self._lambda_flat)
        eps = float(self.args.get("mechanism_eps", 1e-12))
        euc_active = self._euc_trust and bool(self._param_reference)
        euc_radius = self._resolve_euc_radius() if euc_active else None

        def projector(param_groups):
            clean_norm_sq = gam_norm_sq = flat_norm_sq = ewc_norm_sq = dpen_norm_sq = 0.0
            combined_norm_sq = 0.0
            clean_gam_dot = clean_flat_dot = clean_ewc_dot = clean_dpen_dot = 0.0
            flat_ewc_dot = flat_dpen_dot = ewc_dpen_dot = 0.0
            param_to_v: Dict[int, torch.Tensor] = {}

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad:
                        continue

                    clean_grad = clean_grad_cache.get(id(param))
                    gam_grad = None if param.grad is None else param.grad.detach().clone()
                    ewc_grad = ewc_grad_cache.get(id(param))
                    dpen_grad = delta_penalty_grad_cache.get(id(param))

                    if clean_grad is None:
                        combined_grad = None if gam_grad is None else gam_grad.clone()
                    elif gam_grad is None:
                        combined_grad = clean_grad.clone()
                    else:
                        combined_grad = clean_grad + lambda_flat * (gam_grad - clean_grad)

                    if ewc_grad is not None:
                        combined_grad = ewc_grad.clone() if combined_grad is None else combined_grad + ewc_grad
                    if dpen_grad is not None:
                        combined_grad = dpen_grad.clone() if combined_grad is None else combined_grad + dpen_grad
                    if combined_grad is None:
                        continue

                    if clean_grad is not None:
                        clean_norm_sq += float(clean_grad.pow(2).sum().item())
                    if gam_grad is not None:
                        gam_norm_sq += float(gam_grad.pow(2).sum().item())
                    if ewc_grad is not None:
                        ewc_norm_sq += float(ewc_grad.pow(2).sum().item())
                    if dpen_grad is not None:
                        dpen_norm_sq += float(dpen_grad.pow(2).sum().item())
                    combined_norm_sq += float(combined_grad.pow(2).sum().item())

                    if clean_grad is not None and gam_grad is not None:
                        flat_grad = gam_grad - clean_grad
                        flat_norm_sq += float(flat_grad.pow(2).sum().item())
                        clean_gam_dot += float((clean_grad * gam_grad).sum().item())
                        clean_flat_dot += float((clean_grad * flat_grad).sum().item())
                        if ewc_grad is not None:
                            flat_ewc_dot += float((flat_grad * ewc_grad).sum().item())
                        if dpen_grad is not None:
                            flat_dpen_dot += float((flat_grad * dpen_grad).sum().item())
                    if clean_grad is not None and ewc_grad is not None:
                        clean_ewc_dot += float((clean_grad * ewc_grad).sum().item())
                    if clean_grad is not None and dpen_grad is not None:
                        clean_dpen_dot += float((clean_grad * dpen_grad).sum().item())
                    if ewc_grad is not None and dpen_grad is not None:
                        ewc_dpen_dot += float((ewc_grad * dpen_grad).sum().item())

                    param_to_v[id(param)] = combined_grad.detach().clone()
                    if param.grad is None:
                        param.grad = combined_grad.clone()
                    else:
                        param.grad.data.copy_(combined_grad)

            euc_stats = {
                "euc_scale": 1.0,
                "euc_before_norm": 0.0,
                "euc_after_norm": 0.0,
                "euc_budget": float(euc_radius or 0.0),
                "euc_ratio_before": 0.0,
                "euc_ratio_after": 0.0,
                "euc_current_feasible": 1.0,
                "euc_matched": 0.0,
                "euc_total": 0.0,
            }
            if euc_active and euc_radius is not None:
                scale, euc_stats = self._compute_euclidean_cumulative_scale(
                    param_groups, param_to_v, float(euc_radius)
                )
                if scale != 1.0:
                    for group in param_groups:
                        for param in group["params"]:
                            if param.requires_grad and param.grad is not None:
                                param.grad.data.mul_(scale)

            clean_norm = clean_norm_sq ** 0.5
            gam_norm = gam_norm_sq ** 0.5
            flat_norm = flat_norm_sq ** 0.5
            ewc_norm = ewc_norm_sq ** 0.5
            dpen_norm = dpen_norm_sq ** 0.5
            combined_norm = combined_norm_sq ** 0.5
            step_stats.update(
                {
                    "grad_clean_norm": float(clean_norm),
                    "grad_gam_norm": float(gam_norm),
                    "grad_flat_component_norm": float(flat_norm),
                    "grad_fisher_norm": float(ewc_norm),
                    "grad_l2_penalty_norm": float(dpen_norm),
                    "grad_combined_norm": float(combined_norm),
                    "grad_flat_to_clean_norm_ratio": float(flat_norm / (clean_norm + eps)),
                    "grad_fisher_to_clean_norm_ratio": float(ewc_norm / (clean_norm + eps)),
                    "grad_l2_penalty_to_clean_norm_ratio": float(dpen_norm / (clean_norm + eps)),
                    "grad_cos_clean_gam": self._cosine_from_sums(clean_gam_dot, clean_norm_sq, gam_norm_sq, eps),
                    "grad_cos_clean_flat": self._cosine_from_sums(clean_flat_dot, clean_norm_sq, flat_norm_sq, eps),
                    "grad_cos_clean_fisher": self._cosine_from_sums(clean_ewc_dot, clean_norm_sq, ewc_norm_sq, eps),
                    "grad_cos_clean_l2_penalty": self._cosine_from_sums(clean_dpen_dot, clean_norm_sq, dpen_norm_sq, eps),
                    "grad_cos_flat_fisher": self._cosine_from_sums(flat_ewc_dot, flat_norm_sq, ewc_norm_sq, eps),
                    "grad_cos_flat_l2_penalty": self._cosine_from_sums(flat_dpen_dot, flat_norm_sq, dpen_norm_sq, eps),
                    "grad_cos_fisher_l2_penalty": self._cosine_from_sums(ewc_dpen_dot, ewc_norm_sq, dpen_norm_sq, eps),
                    **euc_stats,
                }
            )

        return projector

    def _step_projected(self, optimizer, inputs, local_targets):
        params = _flatten_optimizer_params(optimizer)
        clean_loss_value, clean_grad_cache = self._compute_clean_task_grad_cache(
            params, inputs, local_targets
        )
        ewc_loss_value, ewc_grad_cache = self._compute_ewc_grad_cache_with_value(params)
        dpen_value, dpen_grad_cache, dpen_stats = self._compute_delta_penalty_grad_cache(params)

        log_cache: Dict[str, float] = {
            "ce_clean": float(clean_loss_value),
            "ce": float(clean_loss_value),
            "gam": float(clean_loss_value),
            "ewc": float(ewc_loss_value),
            "l2_penalty": float(dpen_value),
            "lambda_ewc": float(self.ewc_weight),
            "lambda_flat": float(self._lambda_flat),
        }
        log_cache.update(dpen_stats)

        step_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_merged_gradient_projector(
                clean_grad_cache, ewc_grad_cache, dpen_grad_cache, step_stats
            )
        )

        call_count = [0]

        def closure():
            call_count[0] += 1
            optimizer.zero_grad()
            logits, task_loss = self._compute_task_loss(inputs, local_targets)
            if call_count[0] == 1:
                log_cache["ce_clean"] = float(task_loss.detach().item())
                log_cache["ce"] = float(task_loss.detach().item())
            else:
                log_cache["gam"] = float(task_loss.detach().item())
            task_loss.backward()
            return logits, task_loss.detach()

        try:
            logits, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ce_value = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        gam_value = float(log_cache.get("gam", ce_value))
        log_cache.update(step_stats)
        log_cache["ewc_gam_ratio"] = float(ewc_loss_value / (gam_value + float(self.args.get("mechanism_eps", 1e-12))))
        log_cache["l2_penalty_gam_ratio"] = float(dpen_value / (gam_value + float(self.args.get("mechanism_eps", 1e-12))))

        total_loss_value = ce_value + float(ewc_loss_value) + float(dpen_value)
        return logits.detach(), float(total_loss_value), log_cache

    def _compute_ewc_grad_cache_with_value(self, params):
        if self.count_updates == 0 or not self.omega:
            return 0.0, {}

        with torch.enable_grad():
            ewc_loss = self._compute_ewc_loss()

        value = float(ewc_loss.detach().item())
        if not ewc_loss.requires_grad:
            return value, {}
        grads = torch.autograd.grad(ewc_loss, params, allow_unused=True, retain_graph=False)
        return value, {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }

    # ------------------------------------------------------------------
    # Training loop with in-task snapshots
    # ------------------------------------------------------------------

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task > 0 and (self._euc_trust or self._dpen_active):
            self._snapshot_param_reference(optimizer.param_groups)
        else:
            self._param_reference = {}

        self._in_task_fisher_snapshots = []
        self._in_task_weight_snapshots = []
        self._in_task_total_epochs = int(self._run_epochs)
        self._in_task_snapshot_metrics = []
        self._in_task_path_weight_last_metrics = {}
        self._in_task_fisher_last_metrics = {}
        self._in_task_weight_merge_last_metrics = {}
        self._last_task_new_fisher_stats = {}
        self._reset_train_log_stats()
        self._log_task_start_state()

        prog_bar = tqdm(range(self._run_epochs))
        info = ""
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            ce_losses = 0.0
            gam_losses = 0.0
            ewc_losses = 0.0
            l2_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                local_targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits, loss_value, log_dict = self._step_projected(optimizer, inputs, local_targets)
                losses += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", log_dict.get("ce", 0.0)))
                gam_losses += float(log_dict.get("gam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                l2_losses += float(log_dict.get("l2_penalty", 0.0))
                loss_count += 1
                self._record_train_log_stats(log_dict)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                total += len(local_targets)

            if scheduler is not None:
                scheduler.step()

            if self._should_capture_in_task_fisher(epoch + 1):
                self._capture_in_task_fisher_snapshot(
                    train_loader=train_loader,
                    model=self._net(),
                    optimizer=optimizer,
                    epoch_idx=epoch + 1,
                )

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            denom = max(1, loss_count)
            avg_loss = losses / denom
            avg_ce = ce_losses / denom
            avg_gam = gam_losses / denom
            avg_ewc = ewc_losses / denom
            avg_l2 = l2_losses / denom
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, "
                "CE {:.3f}, GAM {:.3f}, EWC {:.3e}, L2 {:.3e}, EWC/GAM {:.3e}"
            ).format(
                self._cur_task,
                epoch + 1,
                self._run_epochs,
                avg_loss,
                train_acc,
                avg_ce,
                avg_gam,
                avg_ewc,
                avg_l2,
                avg_ewc / (avg_gam + float(self.args.get("mechanism_eps", 1e-12))),
            )
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ", Test_accy {:.2f}".format(test_acc)
            prog_bar.set_description(info)

        logging.info(info)
        self._apply_in_task_weight_merge(self._net())

    # ------------------------------------------------------------------
    # In-task Fisher and accumulation
    # ------------------------------------------------------------------

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        if not self._approximate_fisher_in_task:
            return False
        if self._in_task_fisher_interval <= 0 or self._in_task_fisher_omega <= 0.0:
            return False
        if (
            self._in_task_fisher_start_epoch is not None
            and int(epoch_idx) < int(self._in_task_fisher_start_epoch)
        ):
            return False
        if int(epoch_idx) % int(self._in_task_fisher_interval) != 0:
            return False
        if (
            self._in_task_fisher_max_snapshots is not None
            and len(self._in_task_fisher_snapshots) >= int(self._in_task_fisher_max_snapshots)
        ):
            return False
        return True

    def _compute_endpoint_fisher(self, net) -> Dict[str, torch.Tensor]:
        task_start, task_end = self._task_range()
        train_loader = getattr(self, "train_loader", None)
        if train_loader is None:
            raise RuntimeError(
                "EWCGAMMergedInTaskMultiFisherNorm._compute_endpoint_fisher requires "
                "self.train_loader to be populated by the incremental training loop."
            )
        return self._compute_fisher_for_modes(train_loader, net, task_start, task_end)

    def _estimate_snapshot_train_metrics(self, train_loader, model, task_start: int, task_end: int) -> Dict[str, float]:
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
                    inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
                    if inputs is None:
                        continue
                    logits = model(inputs)["logits"][:, task_start:task_end]
                    loss = F.cross_entropy(logits, local_targets)
                    batch_size = int(local_targets.numel())
                    total_loss += float(loss.detach().item()) * float(batch_size)
                    preds = torch.argmax(logits, dim=1)
                    correct += int(preds.eq(local_targets).sum().item())
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

    def _capture_in_task_fisher_snapshot(self, train_loader, model, optimizer, epoch_idx: int) -> int:
        was_training = bool(model.training)
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm] Computing in-task Fisher snapshot task=%d epoch=%d modes=%s",
            self._cur_task,
            epoch_idx,
            self._fisher_modes,
        )
        try:
            task_start, task_end = self._task_range()
            snapshot_metrics = self._estimate_snapshot_train_metrics(
                train_loader, model, task_start, task_end
            )
            fisher = self._compute_fisher_for_modes(train_loader, model, task_start, task_end)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        snapshot = {name: value.detach().cpu().clone() for name, value in fisher.items()}
        self._in_task_fisher_snapshots.append((int(epoch_idx), snapshot))
        snapshot_metrics.update(
            {
                "snapshot_epoch": float(epoch_idx),
                "snapshot_index": float(len(self._in_task_fisher_snapshots)),
                "snapshot_fisher_trace": float(self._fisher_trace(snapshot)),
            }
        )
        self._in_task_snapshot_metrics.append(snapshot_metrics)
        if self._in_task_weight_merge:
            weights = self._snapshot_model_weights()
            if weights:
                self._in_task_weight_snapshots.append((int(epoch_idx), weights))

        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm] Saved in-task Fisher snapshot task=%d epoch=%d "
            "count=%d trace=%.6g tensors=%d ce=%.6g acc=%.3f",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(snapshot),
            len(snapshot),
            float(snapshot_metrics.get("snapshot_train_ce", float("nan"))),
            float(snapshot_metrics.get("snapshot_train_acc", 0.0)),
        )
        return len(self._in_task_fisher_snapshots)

    def _snapshot_entries(self) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
        entries: List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]] = []
        for index, (epoch, fisher_snapshot) in enumerate(self._in_task_fisher_snapshots):
            metrics = (
                self._in_task_snapshot_metrics[index]
                if index < len(self._in_task_snapshot_metrics)
                else {}
            )
            entries.append((index, int(epoch), fisher_snapshot, metrics))
        return entries

    def _select_late_snapshot_entries(self) -> List[Tuple[int, int, Dict[str, torch.Tensor], Dict[str, float]]]:
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
            fisher_values = (
                self._trace_normalize_fisher(fisher_snapshot)
                if self._in_task_path_normalize_snapshots
                else fisher_snapshot
            )
            for name, value in fisher_values.items():
                value_cpu = value.detach().cpu()
                weighted_value = value_cpu * float(weight)
                if name not in accum:
                    accum[name] = weighted_value.clone()
                else:
                    if tuple(weighted_value.shape) != tuple(accum[name].shape):
                        weighted_value = self._match_shape(weighted_value, accum[name])
                    accum[name] = accum[name] + weighted_value

        path_trace = self._fisher_trace(accum)
        self._in_task_path_weight_last_metrics.update(
            {
                "in_task_path_normalize_snapshots": float(bool(self._in_task_path_normalize_snapshots)),
                "in_task_path_weighted_trace": float(path_trace),
            }
        )
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm][PathWeight] task=%d selected=%d/%d "
            "epochs=%s mode=%s beta=%.4g weight_min=%.4g weight_max=%.4g "
            "eff_n=%.4g path_trace=%.6g norm_snapshots=%s",
            self._cur_task,
            int(self._in_task_path_weight_last_metrics.get("in_task_path_selected_snapshots", 0)),
            int(self._in_task_path_weight_last_metrics.get("in_task_path_total_snapshots", 0)),
            [int(epoch) for _, epoch, _, _ in selected],
            self._in_task_path_weight_last_metrics.get("in_task_path_weight_mode", "unknown"),
            float(self._in_task_path_beta),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_min", 0.0)),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_max", 0.0)),
            float(self._in_task_path_weight_last_metrics.get("in_task_path_weight_effective_n", 0.0)),
            float(path_trace),
            self._in_task_path_normalize_snapshots,
        )
        return accum

    def _mix_in_task_fisher(self, final_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        path_raw = self._mean_in_task_raw_fisher()
        final_trace = self._fisher_trace(final_raw)
        path_trace = self._fisher_trace(path_raw)
        snapshot_epochs = [epoch for epoch, _ in self._in_task_fisher_snapshots]

        use_path = (
            self._approximate_fisher_in_task
            and self._in_task_fisher_omega > 0.0
            and path_raw is not None
        )
        if not use_path:
            self._in_task_fisher_last_metrics = {
                "in_task_fisher_used_path": 0.0,
                "in_task_fisher_interval": float(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(self._in_task_fisher_omega),
                "in_task_fisher_snapshots": float(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
                "in_task_fisher_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_fisher_final_raw_trace": float(final_trace),
                "in_task_fisher_path_raw_trace": float(path_trace),
                "in_task_fisher_mixed_raw_trace": float(final_trace),
            }
            return self._trace_normalize_fisher(final_raw) if self._normalize_fisher and final_raw else final_raw

        omega = float(self._in_task_fisher_omega)
        if self._normalize_fisher:
            final_src = self._trace_normalize_fisher(final_raw)
            path_src = self._trace_normalize_fisher(path_raw)
        else:
            final_src = final_raw
            path_src = path_raw

        mixed: Dict[str, torch.Tensor] = {}
        for name in set(final_src.keys()) | set(path_src.keys()):
            final_value = final_src.get(name)
            path_value = path_src.get(name)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[name] = path_value.detach().cpu().clone()
            elif path_value is None:
                mixed[name] = final_value.detach().cpu().clone()
            else:
                final_t = final_value.detach().cpu()
                path_t = self._match_shape(path_value, final_t)
                mixed[name] = (1.0 - omega) * final_t + omega * path_t

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "in_task_fisher_used_path": 1.0,
            "in_task_fisher_interval": float(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": float(len(snapshot_epochs)),
            "in_task_fisher_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
            "in_task_fisher_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_fisher_final_raw_trace": float(final_trace),
            "in_task_fisher_path_raw_trace": float(path_trace),
            "in_task_fisher_mixed_raw_trace": float(mixed_trace),
        }
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm] Mixed endpoint/path Fisher task=%d snapshots=%d "
            "omega=%.4g final_trace=%.6g path_trace=%.6g mixed_trace=%.6g "
            "normalize_task=%s",
            self._cur_task,
            len(snapshot_epochs),
            omega,
            final_trace,
            path_trace,
            mixed_trace,
            self._normalize_fisher,
        )
        return mixed

    def _damp_trace_normalized_fisher(
        self, fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convex F+I mix: F_i <- (1-alpha) F_i + alpha / d."""
        alpha = float(self._ewc_fisher_damping_alpha)
        if alpha <= 0.0 or not fisher:
            self._last_fisher_damping_metrics = {
                "ewc_fisher_damping_alpha": float(alpha),
                "ewc_fisher_damped": 0.0,
                "ewc_fisher_damping_numel": 0.0,
                "ewc_fisher_damping_iso": 0.0,
            }
            return fisher
        total_numel = int(sum(value.numel() for value in fisher.values()))
        if total_numel <= 0:
            return fisher
        iso = alpha / float(total_numel)
        damped = {name: (1.0 - alpha) * value + iso for name, value in fisher.items()}
        self._last_fisher_damping_metrics = {
            "ewc_fisher_damping_alpha": float(alpha),
            "ewc_fisher_damped": 1.0,
            "ewc_fisher_damping_numel": float(total_numel),
            "ewc_fisher_damping_iso": float(iso),
            "ewc_fisher_damped_trace": float(self._fisher_trace(damped)),
        }
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm][Damp] task=%d alpha=%.6g numel=%d "
            "iso=%.3e trace=%.6g",
            self._cur_task,
            alpha,
            total_numel,
            iso,
            self._fisher_trace(damped),
        )
        return damped

    def _accumulate_fisher(self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not old:
            self._fisher_past_trace_mass = max(self._fisher_trace(new), 0.0)
            initial = self._trace_normalize_fisher(new) if self._normalize_accumulated_fisher else {
                name: value.detach().cpu().clone() for name, value in new.items()
            }
            return self._damp_trace_normalized_fisher(initial)

        if not self._normalize_accumulated_fisher:
            combined: Dict[str, torch.Tensor] = {}
            for name in set(old.keys()) | set(new.keys()):
                if name in old and name in new:
                    new_t = self._match_shape(new[name], old[name])
                    combined[name] = old[name] * float(self.gamma) + new_t
                elif name in old:
                    combined[name] = old[name] * float(self.gamma)
                else:
                    combined[name] = new[name].detach().cpu().clone()
            self._fisher_past_trace_mass = self._fisher_trace(combined)
            return self._damp_trace_normalized_fisher(combined)

        old_trace = self._fisher_trace(old)
        old_mass = float(self._fisher_past_trace_mass) if self._fisher_past_trace_mass > 0.0 else old_trace
        old_scale = old_mass / (old_trace + float(self._fisher_norm_eps))
        new_trace = self._fisher_trace(new)

        combined: Dict[str, torch.Tensor] = {}
        for name in set(old.keys()) | set(new.keys()):
            if name in old and name in new:
                old_raw = old[name] * float(old_scale)
                new_t = self._match_shape(new[name], old_raw)
                combined[name] = old_raw * float(self.gamma) + new_t
            elif name in old:
                combined[name] = old[name] * float(old_scale) * float(self.gamma)
            else:
                combined[name] = new[name].detach().cpu().clone()

        self._fisher_past_trace_mass = float(self.gamma) * old_mass + new_trace
        normalized = self._trace_normalize_fisher(combined)
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm] Normalized accumulated Fisher task=%d "
            "old_mass=%.6g new_trace=%.6g mass=%.6g trace=%.6g",
            self._cur_task,
            old_mass,
            new_trace,
            self._fisher_past_trace_mass,
            self._fisher_trace(normalized),
        )
        return self._damp_trace_normalized_fisher(normalized)

    def _mean_in_task_weight_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_weight_snapshots:
            return None
        accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _, weight_snapshot in self._in_task_weight_snapshots:
            for name, value in weight_snapshot.items():
                value_cpu = value.detach().cpu()
                if name not in accum:
                    accum[name] = value_cpu.clone()
                    counts[name] = 1
                else:
                    accum[name] = accum[name] + value_cpu
                    counts[name] += 1
        return {name: value / float(max(counts.get(name, 1), 1)) for name, value in accum.items()}

    def _apply_in_task_weight_merge(self, model) -> Dict[str, float]:
        snapshot_epochs = [epoch for epoch, _ in self._in_task_weight_snapshots]
        omega = float(self._in_task_fisher_omega)
        metrics: Dict[str, float] = {
            "in_task_weight_merge_enabled": float(bool(self._in_task_weight_merge)),
            "in_task_weight_merge_applied": 0.0,
            "in_task_weight_merge_omega": float(omega),
            "in_task_weight_merge_snapshots": float(len(snapshot_epochs)),
            "in_task_weight_merge_first_epoch": float(snapshot_epochs[0]) if snapshot_epochs else 0.0,
            "in_task_weight_merge_last_epoch": float(snapshot_epochs[-1]) if snapshot_epochs else 0.0,
        }
        if not self._in_task_weight_merge or int(self._cur_task) < int(self._in_task_weight_merge_start_task) or omega <= 0.0:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics

        path_mean = self._mean_in_task_weight_snapshot()
        if not path_mean:
            self._in_task_weight_merge_last_metrics = metrics
            return metrics

        final_norm_sq = path_norm_sq = merged_norm_sq = shift_norm_sq = 0.0
        tensor_count = numel = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if self._in_task_weight_merge_exclude_classifier and self._is_classifier_param_name(name):
                    continue
                path_value = path_mean.get(name)
                if path_value is None:
                    continue
                final_cpu = param.detach().cpu()
                path_t = self._match_shape(path_value, final_cpu)
                merged = final_cpu * (1.0 - omega) + path_t * omega
                final_f = final_cpu.float()
                path_f = path_t.float()
                merged_f = merged.float()
                final_norm_sq += float(final_f.pow(2).sum().item())
                path_norm_sq += float(path_f.pow(2).sum().item())
                merged_norm_sq += float(merged_f.pow(2).sum().item())
                shift_norm_sq += float((merged_f - final_f).pow(2).sum().item())
                tensor_count += 1
                numel += int(final_cpu.numel())
                param.data.copy_(merged.to(device=param.device, dtype=param.dtype))

        metrics.update(
            {
                "in_task_weight_merge_applied": float(tensor_count > 0),
                "in_task_weight_merge_tensors": float(tensor_count),
                "in_task_weight_merge_numel": float(numel),
                "in_task_weight_final_l2": float(final_norm_sq ** 0.5),
                "in_task_weight_path_l2": float(path_norm_sq ** 0.5),
                "in_task_weight_merged_l2": float(merged_norm_sq ** 0.5),
                "in_task_weight_merge_shift_l2": float(shift_norm_sq ** 0.5),
            }
        )
        self._in_task_weight_merge_last_metrics = metrics
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm][WeightMerge] task=%d %s",
            self._cur_task,
            metrics,
        )
        return metrics

    def _log_task_start_state(self) -> None:
        fisher_stats = self._summarize_tensor_dict(self.omega)
        ref_count = len(self.theta_star) if self.theta_star else 0
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm][TaskStart] task=%d lambda=%.3e lambda_flat=%.3g "
            "has_fisher=%s F_tensors=%d F_numel=%d F_max=%.3e F_mean=%.3e F_sum=%.3e "
            "has_ref=%s ref_tensors=%d euc_trust=%s euc_radius=%s dpen=%s dpen_radius=%s",
            self._cur_task,
            float(self.ewc_weight),
            float(self._lambda_flat),
            bool(self.omega),
            int(fisher_stats["tensors"]),
            int(fisher_stats["numel"]),
            fisher_stats["max"],
            fisher_stats["mean"],
            fisher_stats["sum"],
            bool(self.theta_star),
            ref_count,
            self._euc_trust,
            self._resolve_euc_radius(),
            self._dpen_active,
            self._resolve_dpen_radius(),
        )

    def after_task(self):
        print("=== Computing Fisher Information ===")
        net = self._net()
        final_raw = self._compute_endpoint_fisher(net)
        fisher = self._mix_in_task_fisher(final_raw)
        self._last_task_new_fisher_stats = self._summarize_tensor_dict(fisher)

        if self._uses_all_seen_fisher:
            replacement = (
                self._trace_normalize_fisher(fisher)
                if self._normalize_accumulated_fisher
                else {name: value.detach().cpu().clone() for name, value in fisher.items()}
            )
            self.omega = self._damp_trace_normalized_fisher(replacement)
            self._fisher_past_trace_mass = self._fisher_trace(self.omega)
        else:
            self.omega = self._accumulate_fisher(self.omega, fisher)

        self.count_updates += 1
        self.theta_star = {
            name: param.detach().clone()
            for name, param in net.named_parameters()
            if param.requires_grad and (self._ewc_include_classifier or not self._is_classifier_param_name(name))
        }
        self._known_classes = self._total_classes

        new_stats = dict(self._last_task_new_fisher_stats)
        accum_stats = self._summarize_tensor_dict(self.omega)
        logger.info(
            "[EWCGAMMergedInTaskMultiFisherNorm][FisherStats] task=%d new_tensors=%d new_numel=%d "
            "new_max=%.3e new_mean=%.3e new_sum=%.3e accum_tensors=%d accum_numel=%d "
            "accum_max=%.3e accum_mean=%.3e accum_sum=%.3e trace_mass=%.3e",
            self._cur_task,
            int(new_stats["tensors"]),
            int(new_stats["numel"]),
            new_stats["max"],
            new_stats["mean"],
            new_stats["sum"],
            int(accum_stats["tensors"]),
            int(accum_stats["numel"]),
            accum_stats["max"],
            accum_stats["mean"],
            accum_stats["sum"],
            float(self._fisher_past_trace_mass),
        )

        if self._compute_full_fisher:
            full_diff = self._compute_full_fisher_diff_metrics(net)
            if full_diff:
                full_diff["task"] = int(self._cur_task)
                full_diff["method_type"] = "ewc_gam_mergedintask_multifisher_norm"
                self._last_full_fisher_diff = full_diff
                logging.info(
                    "[EWCGAMMergedInTaskMultiFisherNorm][FullFisherDiff] task=%d %s",
                    self._cur_task,
                    full_diff,
                )

        self._last_mechanism_metrics = {
            "task": int(self._cur_task),
            "method_type": "ewc_gam_mergedintask_multifisher_norm",
            "lambda_ewc": float(self.ewc_weight),
            "lambda_flat": float(self._lambda_flat),
            "ewc_gamma": float(self.gamma),
            "ewc_normalize_fisher": bool(self._normalize_fisher),
            "ewc_normalize_accumulated_fisher": bool(self._normalize_accumulated_fisher),
            "ewc_include_classifier": bool(self._ewc_include_classifier),
            "fisher_mode": str(self.fisher_mode),
            "fisher_modes": ",".join(self._fisher_modes),
            "fisher_past_trace_mass": float(self._fisher_past_trace_mass),
            "approximate_fisher_in_task": bool(self._approximate_fisher_in_task),
            "ewc_fisher_damping_alpha": float(self._ewc_fisher_damping_alpha),
            "euc_trust": bool(self._euc_trust),
            "euc_radius": float(self._resolve_euc_radius() or 0.0),
            "l2_penalty": bool(self._dpen_active),
            "l2_penalty_lambda": float(self._dpen_lambda),
            "l2_penalty_radius": float(self._resolve_dpen_radius() or 0.0),
            **self._last_fisher_estimation_metrics,
            **self._last_fisher_norm_metrics,
            **self._last_multi_fisher_metrics,
            **self._last_fisher_damping_metrics,
            **self._in_task_fisher_last_metrics,
            **self._in_task_path_weight_last_metrics,
            **self._in_task_weight_merge_last_metrics,
            **self._summarize_train_log_stats(),
        }
        logging.info("[EWCGAMMergedInTaskMultiFisherNorm][Mechanism] %s", self._last_mechanism_metrics)

    def get_mechanism_metrics(self) -> dict:
        metrics = dict(self._last_full_fisher_diff)
        metrics.update(self._last_mechanism_metrics)
        return metrics


EWCGAMMergedInTaskEuc = EWCGAMMergedInTaskMultiFisherNorm
Learner = EWCGAMMergedInTaskMultiFisherNorm
