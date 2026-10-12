"""Full-model online EWC with mechanism/FQA diagnostics.

This is the non-LoRA counterpart of the EWCLoRA probing path.  It keeps the
training objective and optimizer plumbing close to ``models_CL/EWCon.py`` while
adding:

* per-batch CE/EWC/drift summaries for ``mechanism`` metrics;
* endpoint/anchor FQA via ``utils.mechanism_eval`` in trainable-parameter space;
* Fisher-buffer and Fisher-weighted parameter-drift diagnostics.

The EWC penalty is the standard online diagonal form:

    L = L_task + 0.5 * lambda * sum_i F_i * (theta_i - theta_i*)^2
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation_weight_sharpness.eval_flatness_weight_Loss import (
    FlatnessConfig,
    evaluate_flatness_metrics,
)
from evaluation_weight_sharpness.param_utils import _select_params_by_name
from models_CL.baseLearner import BaseLearner
from utils.inc_net import IncrementalNet
from utils.mechanism_eval import (
    as_bool,
    delta_fisher_drift_metrics,
    evaluate_anchor_metrics,
    fisher_buffer_metrics,
    parse_float_list,
    parse_int_list,
    parse_max_batches,
    parse_str_list,
    select_eval_named_params,
)
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


_LORA_SCOPES = {
    "lora",
    "new_lora",
    "delta_w",
    "dw",
    "effective_delta_w",
    "effective_update",
    "lora_factor",
    "lora_factors",
    "lora_ab",
    "ab",
    "a_b",
}

# Default Fisher estimation modes for multi-Fisher FQA comparison.
_MULTI_FISHER_MODES: tuple = (
    "ncm_empirical",
    "batch_empirical",
    "max_pred_batch_empirical",
    "lr_batch_empirical",
    "sample_empirical",
    "lr_sample_empirical",
    "true_sample",
    "kfac_empirical",
    "kfac_full_empirical",
    "lr_kfac_empirical",
    "lr_kfac_full_empirical",
)
_FULL_KFAC_FISHER_MODES = {"kfac_full_empirical", "lr_kfac_full_empirical"}
_KFAC_FISHER_MODES = {"kfac_empirical", "lr_kfac_empirical"} | _FULL_KFAC_FISHER_MODES
_FISHER_MODE_ALIASES = {"full": "batch_empirical"}

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
    "kfac_empirical": {
        "display_name": "legacy Linear KFAC batch-gradient proxy",
        "estimator_family": "kfac_batch_gradient_proxy",
        "gradient_aggregation": "batch_loss_output_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "lr_kfac_empirical": {
        "display_name": "logit-reversed legacy Linear KFAC batch-gradient proxy",
        "estimator_family": "kfac_batch_gradient_proxy",
        "gradient_aggregation": "batch_loss_output_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "kfac_full_empirical": {
        "display_name": "full-block KFAC batch-gradient proxy",
        "estimator_family": "kfac_full_batch_gradient_proxy",
        "gradient_aggregation": "batch_loss_output_gradients",
        "is_per_sample_empirical_fisher": False,
    },
    "lr_kfac_full_empirical": {
        "display_name": "logit-reversed full-block KFAC batch-gradient proxy",
        "estimator_family": "kfac_full_batch_gradient_proxy",
        "gradient_aggregation": "batch_loss_output_gradients",
        "is_per_sample_empirical_fisher": False,
    },
}


def _fisher_estimator_info(mode: str) -> Dict:
    info = dict(_FISHER_ESTIMATOR_INFO.get(str(mode), {}))
    info.setdefault("display_name", str(mode))
    info.setdefault("estimator_family", "unknown")
    info.setdefault("gradient_aggregation", "unknown")
    info.setdefault("is_per_sample_empirical_fisher", False)
    return info


def _canonical_fisher_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    return _FISHER_MODE_ALIASES.get(mode, mode)


class _CurrentTaskLogitView(nn.Module):
    """Wrap a full classifier so mechanism_eval sees current-task logits only."""

    def __init__(self, model: nn.Module, task_start: int, task_end: int):
        super().__init__()
        self.model = model
        self.task_start = int(task_start)
        self.task_end = int(task_end)

    def forward(self, inputs, *_, **__):
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            out = dict(outputs)
            out["logits"] = out["logits"][:, self.task_start : self.task_end]
            return out
        return outputs[:, self.task_start : self.task_end]

    def extract_vector(self, inputs):
        if hasattr(self.model, "extract_vector"):
            return self.model.extract_vector(inputs)
        outputs = self.model(inputs)
        if isinstance(outputs, dict) and "features" in outputs:
            return outputs["features"]
        return outputs


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

        self._ewc_lambda = float(args.get("ewc_lambda", args.get("lambda_ewc", args.get("lambda", 20.0))))
        self._ewc_gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))
        self._ewc_max_batches = parse_max_batches(args.get("ewc_max_batches", 100), default=100)
        fisher_type = str(args.get("fisher_type", "")).lower()
        default_fisher_mode = "ncm_empirical" if fisher_type == "ncm" else "batch_empirical"
        self._fisher_mode = _canonical_fisher_mode(args.get("fisher_mode", default_fisher_mode))
        self._kfac_grad_scale = str(args.get("kfac_grad_scale", "mean_x_bs")).lower()
        self._ncm_fisher_tau = float(
            args.get("ncm_fisher_tau", args.get("fisher_ncm_tau", args.get("ncm_temperature", 16.0)))
        )
        self._use_ewc = as_bool(args.get("use_ewc", True))
        self._compute_fisher_enabled = as_bool(args.get("compute_fisher", True))
        self._ewc_include_fc = as_bool(args.get("ewc_include_fc", False))
        self._ewc_normalize_fisher = as_bool(args.get("ewc_normalize_fisher", False))
        self._ewc_normalize_accumulated_fisher = as_bool(
            args.get(
                "ewc_normalize_accumulated_fisher",
                args.get("ewc_renormalize_accumulated_fisher", False),
            )
        )
        self._ewc_fisher_normalize_type = str(args.get("ewc_fisher_normalize_type", "trace")).lower()
        self._ewc_fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))
        fisher_damping_raw = float(
            args.get(
                "ewc_fisher_damping_beta",
                args.get("ewc_fisher_damping_alpha", args.get("fisher_damping_alpha", 0.0)),
            )
        )
        self._ewc_fisher_damping_beta = max(0.0, min(1.0, fisher_damping_raw))
        self._last_task_fisher_damping_metrics: Dict[str, float] = {}
        if (
            self._ewc_fisher_damping_beta > 0.0
            and (
                not self._ewc_normalize_fisher
                or self._ewc_fisher_normalize_type in {"none", "false", "off"}
            )
        ):
            logger.warning(
                "[EWConProbe] ewc_fisher_damping_beta=%.4g is intended for "
                "trace-normalized per-task Fisher, but ewc_normalize_fisher=%s "
                "normalize_type=%s.",
                self._ewc_fisher_damping_beta,
                self._ewc_normalize_fisher,
                self._ewc_fisher_normalize_type,
            )

        self._nme_eval_enabled = as_bool(args.get("eval_nme", True))

        self._mechanism_eval = as_bool(args.get("mechanism_eval", False))
        self._mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._mechanism_param_drift_interval_steps = int(
            args.get("mechanism_param_drift_interval_steps", 1)
        )
        if self._mechanism_param_drift_interval_steps < 0:
            raise ValueError("mechanism_param_drift_interval_steps must be >= 0")
        self._anchor_eval = as_bool(args.get("anchor_eval", args.get("flatness_eval", False)))
        default_anchor_task = int(args.get("phase_a_tasks", 1)) - 1
        self._anchor_eval_task = int(args.get("anchor_eval_task", default_anchor_task))
        self._endpoint_eval = as_bool(args.get("endpoint_eval", False))
        self._endpoint_eval_tasks = parse_int_list(
            args.get("endpoint_eval_tasks", "0,1"), default=[0, 1]
        )
        self._probe_flatness_eval = as_bool(
            args.get("probe_flatness_eval", args.get("ewcon_probe_flatness_eval", False))
        )
        self._mechanism_sums: Dict[str, float] = {}
        self._mechanism_key_counts: Dict[str, int] = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}
        self._epoch_dynamics_eval = as_bool(
            args.get("epoch_dynamics_eval", args.get("train_dynamics_eval", False))
        )
        self._epoch_dynamics_current_test = as_bool(
            args.get("epoch_dynamics_current_test", False)
        )
        self._last_epoch_dynamics: List[Dict[str, float]] = []

        # Independent endpoint probe used to test the Fisher-shaping claim.
        # It uses a deterministic fixed subset of the current task with test
        # transforms and never feeds its Fisher values into EWC accumulation.
        self._endpoint_sample_fisher_eval = as_bool(
            args.get("endpoint_sample_fisher_eval", False)
        )
        self._endpoint_sample_fisher_samples_per_class = int(
            args.get("endpoint_sample_fisher_samples_per_class", 5)
        )
        self._endpoint_sample_fisher_seed = int(
            args.get("endpoint_sample_fisher_seed", 20260717)
        )
        self._endpoint_sample_fisher_batch_size = int(
            args.get("endpoint_sample_fisher_batch_size", 16)
        )
        self._endpoint_sample_fisher_num_workers = int(
            args.get(
                "endpoint_sample_fisher_num_workers",
                args.get("linear_probe_eval_num_workers", 0),
            )
        )
        self._endpoint_sample_fisher_loader = None
        self._endpoint_sample_fisher_subset_info: Dict = {}

        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._optimizer_type == "cflat":
            self._cflat_rho = float(args.get("cflat_rho", 0.2))
            self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
            self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
            self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
            self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")
        elif self._optimizer_type == "gam":
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.02)))
            self._gam_grad_norm_rho = float(
                args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
            )
            self._gam_beta1 = float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1.0)))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", args.get("grad_beta_2", -1.0)))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1.0)))
            self._gam_gamma = float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.03)))
            self._gam_args = SimpleNamespace(
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
                use_sgv=bool(args.get("gam_use_sgv", False)),
                sgv_lambda=float(args.get("gam_sgv_lambda", 0.0)),
            )
        elif self._optimizer_type in {"ngd", "naturalgradient"}:
            self._ngd_damping = float(args.get("ngd_damping", 1e-3))
            self._ngd_ema_decay = float(args.get("ngd_ema_decay", 0.95))
            self._ngd_eps = float(args.get("ngd_eps", 1e-8))

        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._checkpoint: Optional[Dict[str, torch.Tensor]] = None
        self._last_fisher_estimation_metrics: Dict[str, float] = {}

        # Multi-Fisher FQA: compute all requested Fisher variants after each task
        # and run FQA with each variant to compare quadratic predictions.
        self._multi_fisher_eval = as_bool(args.get("multi_fisher_eval", False))
        _raw_modes = str(args.get("multi_fisher_modes", ",".join(_MULTI_FISHER_MODES)))
        self._multi_fisher_modes: List[str] = [
            _canonical_fisher_mode(m.strip()) for m in _raw_modes.split(",") if m.strip()
        ]
        self._multi_fisher_kfac_factors: Dict[str, List[Dict]] = {}

        # Full Fisher (all seen tasks) vs accumulated Fisher comparison
        self._compute_full_fisher = as_bool(args.get("compute_full_fisher", False))
        self._full_fisher_max_batches = parse_max_batches(
            args.get("full_fisher_max_batches", None), default=None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def incremental_train(self, data_manager):
        self._data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._align_buffers_to_model(self._network)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args.get("train_num_workers", 8),
        )

        self._build_endpoint_sample_fisher_loader(data_manager)

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args.get("train_num_workers", 8),
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._memory_size > 0 or self._memory_per_class:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if self._nme_eval_enabled:
            try:
                self.compute_all_seen_class_means(data_manager)
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("[EWConProbe] Failed to compute class means: %s", exc)

    def after_task(self):
        model = self._unwrap_network()
        task_start, task_end = int(self._known_classes), int(self._total_classes)
        pre_update_drift = self._compute_param_drift_metrics(include_top1pct=True)
        task_fisher: Optional[Dict[str, torch.Tensor]] = None
        anchor_metrics: Dict[str, float] = {}
        endpoint_metrics: Dict[str, float] = {}
        accumulated_metrics: Dict[str, float] = {}
        main_fisher_metrics: Dict[str, float] = {}
        full_fisher_diff_metrics: Dict[str, float] = {}
        endpoint_sample_fisher_metrics: Dict[str, float] = {}

        # ── Fisher: load cache (independent of compute_fisher) or compute ──────
        _fisher_load_dir = self.args.get("load_fisher_dir", None)
        _strict_load = as_bool(self.args.get("strict_load_fisher", False))
        _loaded_fisher = None

        if _fisher_load_dir:
            _loaded_fisher = self._load_fisher_cache(_fisher_load_dir)
            if _loaded_fisher is None and _strict_load:
                raise FileNotFoundError(
                    f"[EWConProbe][FisherCache] strict_load_fisher=true but cache not found: "
                    f"{self._fisher_cache_path(_fisher_load_dir)}"
                )
            if _loaded_fisher is not None:
                cached_mode = str(_loaded_fisher.get("fisher_mode", "")).lower()
                current_mode = str(self._fisher_mode).lower()
                if cached_mode and cached_mode != current_mode:
                    raise ValueError(
                        f"[EWConProbe][FisherCache] Cached fisher_mode='{cached_mode}' "
                        f"does not match requested fisher_mode='{current_mode}'. "
                        f"Regenerate the cache with --main-mode {current_mode}."
                    )

        _has_fisher_source = (
            (_loaded_fisher is not None)
            or (self._compute_fisher_enabled and hasattr(self, "train_loader"))
        )

        if _has_fisher_source:
            if _loaded_fisher is not None:
                logger.info(
                    "[EWConProbe] Loaded Fisher cache from %s (task %d)",
                    _fisher_load_dir, self._cur_task,
                )
                task_fisher = _loaded_fisher.get("main_fisher")
                task_fisher = self._maybe_damp_cached_task_fisher(
                    task_fisher, _loaded_fisher
                )
                task_blocks = (
                    self._fisher_blocks(task_fisher, model)
                    if task_fisher is not None
                    else _loaded_fisher.get("main_fisher_blocks", [])
                )
                main_fisher_metrics = {}
            else:
                logger.info(
                    "[EWConProbe] Computing %s Fisher after task %d",
                    self._fisher_mode, self._cur_task,
                )
                task_fisher = self._compute_fisher(self.train_loader, model, task_start, task_end)
                main_fisher_metrics = dict(self._last_fisher_estimation_metrics)
                if self._ewc_normalize_fisher and task_fisher is not None:
                    task_fisher = self._normalize_task_fisher(task_fisher)
                task_fisher = self._damp_task_fisher(task_fisher)
                task_blocks = self._fisher_blocks(task_fisher, model)

            _mf_vars: Dict[str, List[torch.Tensor]] = {}
            _mf_kfac_factors: Dict[str, List[Dict]] = {}
            if self._multi_fisher_eval:
                if _loaded_fisher is not None and "mf_variants" in _loaded_fisher:
                    _mf_vars = _loaded_fisher.get("mf_variants") or {}
                    _mf_kfac_factors = _loaded_fisher.get("mf_kfac_factors") or {}
                    self._multi_fisher_kfac_factors = _mf_kfac_factors
                    logger.info(
                        "[EWConProbe] Using cached multi-Fisher variants: %s full-KFAC=%s",
                        list(_mf_vars.keys()), list(_mf_kfac_factors.keys()),
                    )
                else:
                    _mf_vars = self._compute_multi_fisher_if_needed(
                        self.train_loader,
                        model,
                        task_start,
                        task_end,
                        precomputed_blocks={self._fisher_mode: task_blocks},
                    )
                    _mf_kfac_factors = dict(self._multi_fisher_kfac_factors)
                # Save Fisher cache before long FQA evaluation starts.
                _fisher_save_dir = self.args.get("save_fisher_dir", None)
                if _fisher_save_dir and _loaded_fisher is None:
                    self._save_fisher_cache(
                        _fisher_save_dir, task_fisher, task_blocks, _mf_vars, _mf_kfac_factors
                    )

            anchor_metrics = self._compute_anchor_eval_metrics_if_needed(task_blocks)
            endpoint_fqa_can_co_compute = (
                bool(_mf_vars)
                and self._endpoint_fqa_can_co_compute_extra_fishers(prefix="endpoint")
            )
            endpoint_metrics = self._compute_endpoint_eval_metrics_if_needed(
                task_blocks,
                extra_omega_W=(
                    self._multi_fisher_extra_omega(_mf_vars)
                    if endpoint_fqa_can_co_compute
                    else None
                ),
                extra_kfac_factors=(
                    self._multi_fisher_extra_kfac(_mf_kfac_factors)
                    if endpoint_fqa_can_co_compute
                    else None
                ),
            )
            if self._multi_fisher_eval and _mf_vars:
                if endpoint_fqa_can_co_compute:
                    logger.info(
                        "[EWConProbe][MultiFisherFQA] co-computed with endpoint FQA; "
                        "skipping separate multi-Fisher FQA pass",
                    )
                    endpoint_metrics.update(
                        self._multi_fisher_summary_metrics(_mf_vars, endpoint_metrics)
                    )
                else:
                    endpoint_metrics.update(
                        self._run_multi_fisher_fqa(
                            _mf_vars, task_start, task_end,
                            kfac_factor_variants=_mf_kfac_factors,
                            endpoint_metrics=endpoint_metrics,
                        )
                    )
            self._accumulate_fisher(task_fisher, model)
            self._checkpoint = {
                name: p.detach().cpu().clone()
                for name, p in self._iter_trainable(model)
            }
            accumulated_metrics = (
                self._compute_accumulated_fisher_metrics()
                if (
                    self._mechanism_eval
                    or self._anchor_eval
                    or self._endpoint_eval
                    or self._endpoint_sample_fisher_eval
                )
                else {}
            )
            if "accumulated_fisher_max" in accumulated_metrics:
                next_task_lr = float(
                    self.args.get("lrate", self.args.get("init_lr", 0.0))
                )
                accumulated_metrics.update(
                    {
                        "next_task_lr": next_task_lr,
                        "next_task_explicit_sgd_ewc_stability_index": float(
                            next_task_lr
                            * (self._ewc_lambda if self._use_ewc else 0.0)
                            * accumulated_metrics["accumulated_fisher_max"]
                        ),
                    }
                )
            if self._compute_full_fisher:
                full_fisher_diff_metrics = self._compute_full_fisher_diff_metrics(model)
        else:
            anchor_metrics = self._compute_anchor_eval_metrics_if_needed([])
            endpoint_metrics = self._compute_endpoint_eval_metrics_if_needed([])

        endpoint_sample_fisher_metrics = self._compute_endpoint_sample_fisher_metrics(
            model, task_start, task_end
        )

        if (self._mechanism_eval or anchor_metrics or endpoint_metrics
                or accumulated_metrics or full_fisher_diff_metrics
                or endpoint_sample_fisher_metrics):
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "ewcon_probe",
                "lambda_ewc": float(self._ewc_lambda if self._use_ewc else 0.0),
                "ewc_gamma": float(self._ewc_gamma),
                "fisher_mode": self._fisher_mode,
                "ewc_include_fc": bool(self._ewc_include_fc),
                "use_ewc": bool(self._use_ewc),
                "compute_fisher": bool(self._compute_fisher_enabled),
                "ewc_normalize_fisher": bool(self._ewc_normalize_fisher),
                "ewc_normalize_accumulated_fisher": bool(
                    self._ewc_normalize_accumulated_fisher
                ),
                "ewc_fisher_normalize_type": str(self._ewc_fisher_normalize_type),
                **main_fisher_metrics,
                **self._last_task_fisher_damping_metrics,
                **self._summarize_mechanism_stats(),
                **pre_update_drift,
                **anchor_metrics,
                **endpoint_metrics,
                **accumulated_metrics,
                **full_fisher_diff_metrics,
                **endpoint_sample_fisher_metrics,
            }
            logging.info("[EWConProbe][Mechanism] %s", self._last_mechanism_metrics)

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Task checkpoint: save / load for staged training
    # ------------------------------------------------------------------
    def save_task_checkpoint(self, ckpt_dir: str, task_idx: int) -> None:
        """Persist model weights + EWC state after task_idx so another run can resume."""
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"task_{task_idx:02d}.pt")
        model = self._unwrap_network()
        state = {
            "task_idx": task_idx,
            "known_classes": int(self._known_classes),
            "cur_task": int(self._cur_task),
            "network_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "fisher": (
                {k: v.cpu().clone() for k, v in self._fisher.items()}
                if self._fisher else None
            ),
            "checkpoint_params": (
                {k: v.cpu().clone() for k, v in self._checkpoint.items()}
                if self._checkpoint else None
            ),
        }
        torch.save(state, path)
        logging.info(
            "[TaskCheckpoint] Saved task %d to %s (known_classes=%d)",
            task_idx, path, self._known_classes,
        )

    def load_task_checkpoint(self, ckpt_dir: str, task_idx: int, data_manager) -> None:
        """Restore model + EWC state from a staged checkpoint; training resumes at task_idx+1."""
        path = os.path.join(ckpt_dir, f"task_{task_idx:02d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)

        self._known_classes = int(state["known_classes"])
        self._total_classes = self._known_classes
        self._cur_task = int(state["cur_task"])

        # Extend the FC head to the saved size before loading weights.
        model = self._unwrap_network()
        model.update_fc(self._known_classes)
        model.load_state_dict(state["network_state"])

        self._fisher = state.get("fisher")
        self._checkpoint = state.get("checkpoint_params")
        logging.info(
            "[TaskCheckpoint] Loaded task %d from %s (known_classes=%d, cur_task=%d)",
            task_idx, path, self._known_classes, self._cur_task,
        )

    def _task_class_range(self, task_idx: int) -> Tuple[int, int]:
        increments = self.args.get("task_increments", None)
        if increments:
            sizes = [int(value) for value in increments]
            task_idx = max(0, min(int(task_idx), len(sizes) - 1))
            start = int(sum(sizes[:task_idx]))
            return start, int(start + sizes[task_idx])

        init_cls = int(self.args.get("init_cls", 0))
        increment = int(self.args.get("increment", init_cls))
        task_idx = int(task_idx)
        if task_idx <= 0:
            return 0, init_cls
        start = init_cls + (task_idx - 1) * increment
        return int(start), int(start + increment)

    def evaluate_loaded_checkpoint(self, task_idx: int) -> Dict[str, float]:
        """Run endpoint/FQA diagnostics on a checkpoint loaded via resume_from_task."""
        task_idx = int(task_idx)
        task_start, task_end = self._task_class_range(task_idx)
        old_known, old_total, old_cur = self._known_classes, self._total_classes, self._cur_task

        try:
            self._known_classes = int(task_start)
            self._total_classes = int(task_end)
            self._cur_task = int(task_idx)
            fisher_blocks = self._fisher_blocks_for_eval_scope(self._full_model_eval_args())
            endpoint_metrics = self._compute_endpoint_eval_metrics_if_needed(fisher_blocks)
            accumulated_metrics = (
                self._compute_accumulated_fisher_metrics()
                if self._fisher is not None else {}
            )
            drift_metrics = self._compute_param_drift_metrics(include_top1pct=True)
        finally:
            self._known_classes = old_known
            self._total_classes = old_total
            self._cur_task = old_cur

        metrics = {
            "task": int(task_idx),
            "loaded_checkpoint_eval": True,
            "method_type": "ewcon_probe",
            "lambda_ewc": float(self._ewc_lambda if self._use_ewc else 0.0),
            "ewc_gamma": float(self._ewc_gamma),
            "fisher_mode": self._fisher_mode,
            "ewc_include_fc": bool(self._ewc_include_fc),
            "use_ewc": bool(self._use_ewc),
            "compute_fisher": bool(self._compute_fisher_enabled),
            **drift_metrics,
            **endpoint_metrics,
            **accumulated_metrics,
        }
        self._last_mechanism_metrics = metrics
        logger.info("[EWConProbe][LoadedCheckpointEval] %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)
        self._reset_mechanism_stats()

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", self.args.get("init_lr_decay", 1.0))),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            ce_losses = 0.0
            ewc_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                closure_stats: Dict[str, float] = {"ce": 0.0, "ewc": 0.0}

                if self._optimizer_type == "cflat":
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        loss_task = self._task_loss_from_logits(logits, targets, stage)
                        loss_ewc = self._ewc_penalty(model)
                        loss = loss_task + loss_ewc
                        closure_stats["ce"] = float(loss_task.detach().item())
                        closure_stats["ewc"] = float(loss_ewc.detach().item())
                        loss.backward()
                        return outputs, [loss]

                    _, loss_list = optimizer.step(closure=closure)
                    loss_value = torch.stack([term.detach() for term in loss_list]).sum()
                    with torch.no_grad():
                        logits = model(inputs)["logits"]
                elif self._optimizer_type == "gam":
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        loss_task = self._task_loss_from_logits(logits, targets, stage)
                        loss_ewc = self._ewc_penalty(model)
                        loss = loss_task + loss_ewc
                        closure_stats["ce"] = float(loss_task.detach().item())
                        closure_stats["ewc"] = float(loss_ewc.detach().item())
                        loss_value_inner = loss.detach()
                        loss.backward()
                        return outputs, loss_value_inner

                    outputs, loss_value = optimizer.step(closure=closure)
                    logits = outputs["logits"].detach()
                else:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    logits = outputs["logits"]
                    loss_task = self._task_loss_from_logits(logits, targets, stage)
                    loss_ewc = self._ewc_penalty(model)
                    loss = loss_task + loss_ewc
                    closure_stats["ce"] = float(loss_task.detach().item())
                    closure_stats["ewc"] = float(loss_ewc.detach().item())

                    if self._optimizer_type == "sam":
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        outputs = model(inputs)
                        logits = outputs["logits"]
                        second_task = self._task_loss_from_logits(logits, targets, stage)
                        second_ewc = self._ewc_penalty(model)
                        second_loss = second_task + second_ewc
                        closure_stats["ce"] = float(second_task.detach().item())
                        closure_stats["ewc"] = float(second_ewc.detach().item())
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                        loss_value = second_loss.detach()
                    else:
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.detach()

                eval_logits, eval_targets = self._eval_view_from_logits(logits, targets, stage)
                total_loss += float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
                ce_losses += float(closure_stats["ce"])
                ewc_losses += float(closure_stats["ewc"])
                loss_count += 1

                if self._mechanism_eval:
                    stats = {
                        "ce": float(closure_stats["ce"]),
                        "ewc": float(closure_stats["ewc"]),
                        "ewc_ce_ratio": float(
                            closure_stats["ewc"] / (closure_stats["ce"] + self._mechanism_eps)
                        ),
                    }
                    if self._should_measure_batch_param_drift():
                        stats.update(self._compute_param_drift_metrics())
                    self._record_mechanism_stats(stats, increment=True)

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = total_loss / max(1, len(train_loader))
            avg_ce = ce_losses / max(1, loss_count)
            avg_ewc = ewc_losses / max(1, loss_count)
            ewc_ce_ratio = float(avg_ewc / (avg_ce + self._mechanism_eps))
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {avg_loss:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {avg_ce:.3f}, EWC {avg_ewc:.3e}"
            )
            all_seen_test_acc = None
            cur_test_acc = None
            if epoch % 5 == 4:
                all_seen_test_acc = self._compute_accuracy(model, test_loader)
                cur_test_acc = self._compute_current_task_accuracy(model, test_loader, stage)
                info += f", Test_accy {all_seen_test_acc:.2f}, CurTask_Test {cur_test_acc:.2f}"
            if self._epoch_dynamics_eval:
                drift = self._compute_param_drift_metrics()
                epoch_metrics = {
                    "task": int(self._cur_task),
                    "epoch": int(epoch + 1),
                    "epochs": int(epochs),
                    "stage": stage,
                    "lambda_ewc": float(self._ewc_lambda if self._use_ewc else 0.0),
                    "use_ewc": bool(self._use_ewc),
                    "loss_total": float(avg_loss),
                    "loss_ce": float(avg_ce),
                    "loss_ewc": float(avg_ewc),
                    "ewc_ce_ratio": float(ewc_ce_ratio),
                    "train_acc_current": float(train_acc),
                    "loss_total_finite": bool(np.isfinite(avg_loss)),
                    "loss_ce_finite": bool(np.isfinite(avg_ce)),
                    "loss_ewc_finite": bool(np.isfinite(avg_ewc)),
                    **drift,
                }
                # delta_fisher_drift_metrics historically calls squared L2 drift
                # "delta_norm"; expose an explicit alias for mechanism plots.
                epoch_metrics["delta_norm_sq"] = float(epoch_metrics.get("delta_norm", 0.0))
                epoch_metrics["param_delta_norm_sq"] = float(
                    epoch_metrics.get("param_delta_norm", epoch_metrics["delta_norm_sq"])
                )
                if all_seen_test_acc is not None:
                    epoch_metrics["test_acc_seen"] = float(all_seen_test_acc)
                if cur_test_acc is not None:
                    epoch_metrics["test_acc_current"] = float(cur_test_acc)
                elif self._epoch_dynamics_current_test:
                    epoch_metrics["test_acc_current"] = float(
                        self._compute_current_task_accuracy(model, test_loader, stage)
                    )
                self._last_epoch_dynamics.append(epoch_metrics)
                logger.info("[EWConProbe][EpochDynamics] %s", epoch_metrics)
            prog_bar.set_description(info)
        if epochs > 0:
            logging.info(info)

    def _task_loss_from_logits(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            return F.cross_entropy(logits, targets)
        fake_targets = targets - self._known_classes
        return F.cross_entropy(logits[:, self._known_classes :], fake_targets)

    def _eval_view_from_logits(self, logits: torch.Tensor, targets: torch.Tensor, stage: str):
        if stage == "init":
            return logits, targets
        fake_targets = targets - self._known_classes
        return logits[:, self._known_classes :], fake_targets

    def _compute_current_task_accuracy(self, model: nn.Module, loader, stage: str) -> float:
        was_training = bool(model.training)
        model.eval()
        correct = 0
        total = 0
        task_start = 0 if stage == "init" else int(self._known_classes)
        task_end = int(self._total_classes)
        try:
            with torch.no_grad():
                for batch in loader:
                    if len(batch) == 3:
                        _, inputs, targets = batch
                    else:
                        inputs, targets = batch
                    mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
                    if mask.numel() == 0:
                        continue
                    inputs = torch.index_select(inputs, 0, mask).to(self._device, non_blocking=True)
                    targets = torch.index_select(targets, 0, mask).to(self._device, non_blocking=True)
                    outputs = model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    if stage == "init":
                        eval_logits = logits[:, :task_end]
                        eval_targets = targets
                    else:
                        eval_logits = logits[:, task_start:task_end]
                        eval_targets = targets - task_start
                    preds = eval_logits.argmax(dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += int(eval_targets.numel())
        finally:
            model.train(was_training)
        return 100.0 * float(correct) / max(1, int(total))

    # ------------------------------------------------------------------
    # EWC buffers and diagnostics
    # ------------------------------------------------------------------
    def _build_endpoint_sample_fisher_loader(self, data_manager) -> None:
        """Build the same deterministic current-task probe subset for every method."""
        self._endpoint_sample_fisher_loader = None
        self._endpoint_sample_fisher_subset_info = {}
        if not self._endpoint_sample_fisher_eval:
            return
        if self._endpoint_sample_fisher_samples_per_class <= 0:
            raise ValueError(
                "endpoint_sample_fisher_samples_per_class must be positive when "
                "endpoint_sample_fisher_eval=true"
            )

        subset_seed = int(self._endpoint_sample_fisher_seed + self._cur_task)
        _, probe_dataset, split_info = data_manager.get_dataset_with_split(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            val_per_class=self._endpoint_sample_fisher_samples_per_class,
            seed=subset_seed,
            train_mode="train",
            val_mode="test",
            ret_indices=True,
            min_train_per_class=0,
        )
        source_indices = np.asarray(
            split_info.get("val_source_indices", []), dtype=np.int64
        )
        subset_sha1 = hashlib.sha1(source_indices.tobytes()).hexdigest()[:12]
        self._endpoint_sample_fisher_subset_info = {
            "seed": subset_seed,
            "samples": int(len(probe_dataset)),
            "samples_per_class": int(self._endpoint_sample_fisher_samples_per_class),
            "source_indices_sha1": subset_sha1,
        }
        self._endpoint_sample_fisher_loader = DataLoader(
            probe_dataset,
            batch_size=self._endpoint_sample_fisher_batch_size,
            shuffle=False,
            num_workers=self._endpoint_sample_fisher_num_workers,
        )
        logger.info(
            "[EWConProbe][EndpointSampleFisherSubset] task=%d classes=%d-%d "
            "samples=%d samples_per_class=%d seed=%d source_indices_sha1=%s "
            "transform=test",
            self._cur_task,
            self._known_classes,
            self._total_classes,
            len(probe_dataset),
            self._endpoint_sample_fisher_samples_per_class,
            subset_seed,
            subset_sha1,
        )

    def _compute_endpoint_sample_fisher_metrics(
        self, model: nn.Module, task_start: int, task_end: int
    ) -> Dict[str, float]:
        """Measure raw per-sample Fisher trace and endpoint fit on one subset.

        This is deliberately trace-only: it avoids allocating another full
        86M-parameter Fisher buffer.  The metric is
        mean_n ||grad_theta CE(x_n, y_n)||^2 over the EWC parameter scope.
        It is diagnostic only and is never normalized or accumulated.
        """
        loader = self._endpoint_sample_fisher_loader
        if not self._endpoint_sample_fisher_eval or loader is None:
            return {}

        was_training = bool(model.training)
        model.eval()
        trace_sum = 0.0
        loss_sum = 0.0
        correct = 0
        num_samples = 0
        try:
            for batch in tqdm(loader, desc="Endpoint sample-Fisher"):
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_inputs(
                    inputs, targets, task_start, task_end
                )
                if inputs is None:
                    continue
                for idx in range(int(inputs.size(0))):
                    model.zero_grad(set_to_none=True)
                    outputs = model(inputs[idx : idx + 1])
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    task_logits = logits[:, task_start:task_end]
                    target_i = local_targets[idx : idx + 1]
                    loss_i = F.cross_entropy(task_logits, target_i)
                    loss_i.backward()

                    sample_trace = None
                    for _, param in self._iter_trainable(model):
                        if param.grad is not None:
                            term = param.grad.detach().float().pow(2).sum()
                            sample_trace = term if sample_trace is None else sample_trace + term
                    if sample_trace is not None:
                        trace_sum += float(sample_trace.item())
                    loss_sum += float(loss_i.detach().item())
                    correct += int(task_logits.detach().argmax(dim=1).eq(target_i).item())
                    num_samples += 1
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if num_samples <= 0:
            logger.warning(
                "[EWConProbe][EndpointSampleFisher] task=%d produced no samples",
                self._cur_task,
            )
            return {}

        metrics = {
            "endpoint_sample_fisher_raw_trace": float(trace_sum / num_samples),
            "endpoint_sample_fisher_ce": float(loss_sum / num_samples),
            "endpoint_sample_fisher_accuracy": float(100.0 * correct / num_samples),
            "endpoint_sample_fisher_num_samples": int(num_samples),
            "endpoint_sample_fisher_samples_per_class": int(
                self._endpoint_sample_fisher_samples_per_class
            ),
            "endpoint_sample_fisher_seed": int(
                self._endpoint_sample_fisher_subset_info.get(
                    "seed", self._endpoint_sample_fisher_seed + self._cur_task
                )
            ),
            "endpoint_sample_fisher_subset_sha1": str(
                self._endpoint_sample_fisher_subset_info.get("source_indices_sha1", "")
            ),
            "endpoint_sample_fisher_excludes_fc": bool(not self._ewc_include_fc),
        }
        logger.info(
            "[EWConProbe][EndpointSampleFisher] %s",
            {"task": int(self._cur_task), **metrics},
        )
        return metrics

    def _iter_trainable(self, model: nn.Module):
        for name, p in model.named_parameters():
            is_fc = name.startswith("fc.") or name.startswith("module.fc.")
            if p.requires_grad and (self._ewc_include_fc or not is_fc):
                yield name, p

    def _ewc_penalty(self, model: nn.Module):
        if (
            not self._use_ewc
            or self._fisher is None
            or self._checkpoint is None
        ):
            return torch.tensor(0.0, device=self._device)
        pen = None
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            diff = p - ref.to(device=p.device, dtype=p.dtype)
            fisher_d = fisher.to(device=p.device, dtype=p.dtype)
            term = (fisher_d * diff.pow(2)).sum()
            pen = term if pen is None else pen + term
        if pen is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * pen

    def _compute_param_drift_metrics(self, include_top1pct: bool = False) -> Dict[str, float]:
        if self._fisher is None or self._checkpoint is None:
            out = {
                "delta_norm": 0.0,
                "delta_norm_sq": 0.0,
                "delta_l2_norm": 0.0,
                "fisher_weighted_drift": 0.0,
                "normalized_fisher_drift": 0.0,
                "ewc_penalty_value": 0.0,
                "param_delta_norm": 0.0,
                "param_delta_norm_sq": 0.0,
                "param_delta_l2_norm": 0.0,
            }
            if include_top1pct:
                out.update({
                    "t1_fisher_top1pct_numel": 0,
                    "delta_w_energy_in_t1_top1pct": 0.0,
                    "d_f_contribution_from_t1_top1pct": 0.0,
                })
            return out

        model = self._unwrap_network()
        fisher_blocks: List[torch.Tensor] = []
        delta_blocks: List[torch.Tensor] = []
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            fisher_blocks.append(fisher)
            delta_blocks.append(p.detach() - ref.to(device=p.device, dtype=p.dtype))

        metrics = delta_fisher_drift_metrics(
            fisher_blocks,
            delta_blocks,
            eps=self._mechanism_eps,
            include_top1pct=include_top1pct,
        )
        metrics["ewc_penalty_value"] = float(
            0.5 * float(self._ewc_lambda if self._use_ewc else 0.0)
            * float(metrics["fisher_weighted_drift"])
        )
        # ``delta_norm`` is a historical name for sum_i Delta-theta_i^2.
        # Preserve it for compatibility and expose unambiguous squared/L2 aliases.
        metrics["delta_norm_sq"] = float(metrics["delta_norm"])
        metrics["delta_l2_norm"] = math.sqrt(max(0.0, float(metrics["delta_norm"])))
        metrics["param_delta_norm"] = float(metrics["delta_norm"])
        metrics["param_delta_norm_sq"] = float(metrics["delta_norm"])
        metrics["param_delta_l2_norm"] = float(metrics["delta_l2_norm"])
        metrics["param_fisher_weighted_drift"] = float(metrics["fisher_weighted_drift"])
        metrics["param_normalized_fisher_drift"] = float(metrics["normalized_fisher_drift"])
        if include_top1pct:
            metrics["param_energy_in_t1_top1pct"] = float(
                metrics.get("delta_w_energy_in_t1_top1pct", 0.0)
            )
            metrics["param_d_f_contribution_from_t1_top1pct"] = float(
                metrics.get("d_f_contribution_from_t1_top1pct", 0.0)
            )
        return metrics

    def _match_tensor(self, stored: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if stored.shape == param.shape:
            return stored
        new = torch.zeros_like(param.detach().cpu())
        slices = tuple(slice(0, min(a, b)) for a, b in zip(stored.shape, param.shape))
        new[slices] = stored[slices]
        return new

    def _align_buffers_to_model(self, model: nn.Module):
        if self._fisher is not None:
            for name, p in model.named_parameters():
                if name in self._fisher:
                    self._fisher[name] = self._match_tensor(self._fisher[name], p)
        if self._checkpoint is not None:
            for name, p in model.named_parameters():
                if name in self._checkpoint:
                    self._checkpoint[name] = self._match_tensor(self._checkpoint[name], p)

    def _normalize_task_fisher(
        self, fisher: Dict[str, torch.Tensor], label: str = "Task Fisher"
    ) -> Dict[str, torch.Tensor]:
        if not fisher:
            return fisher
        norm_type = self._ewc_fisher_normalize_type
        eps = self._ewc_fisher_norm_eps
        if norm_type in {"none", "false", "off"}:
            return fisher
        if norm_type in {"trace", "sum", "l1"}:
            norm = sum(v.float().sum().item() for v in fisher.values())
        elif norm_type in {"fro", "l2"}:
            norm = torch.sqrt(
                sum(v.float().pow(2).sum() for v in fisher.values())
            ).item()
        elif norm_type == "mean":
            total_numel = sum(v.numel() for v in fisher.values())
            norm = sum(v.float().sum().item() for v in fisher.values()) / max(1, total_numel)
        else:
            raise ValueError(
                f"[EWConProbe] Unknown ewc_fisher_normalize_type: {norm_type!r}"
            )
        scale = float(norm) + float(eps)
        logger.info(
            "[EWConProbe] %s normalized (type=%s, norm=%.4e, scale=%.4e)",
            label, norm_type, float(norm), scale,
        )
        return {name: v / scale for name, v in fisher.items()}

    @staticmethod
    def _fisher_dict_trace(fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        return float(sum(v.detach().float().sum().item() for v in fisher.values()))

    @staticmethod
    def _fisher_dict_numel(fisher: Optional[Dict[str, torch.Tensor]]) -> int:
        if not fisher:
            return 0
        return int(sum(int(v.numel()) for v in fisher.values()))

    @staticmethod
    def _fisher_dict_max(fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        vals = [v.detach().float().max().item() for v in (fisher or {}).values() if v.numel()]
        return float(max(vals)) if vals else 0.0

    def _damp_task_fisher(
        self,
        fisher: Optional[Dict[str, torch.Tensor]],
        label: str = "Task Fisher",
    ) -> Optional[Dict[str, torch.Tensor]]:
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

        trace_before = self._fisher_dict_trace(fisher)
        max_before = self._fisher_dict_max(fisher)
        iso = beta / float(total_numel)
        damped = {
            name: (1.0 - beta) * value + iso
            for name, value in fisher.items()
        }
        trace_after = self._fisher_dict_trace(damped)
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
        logger.info(
            "[EWConProbe] %s damped beta=%.4g numel=%d trace %.4e -> %.4e "
            "max %.4e -> %.4e iso=%.4e",
            label,
            beta,
            total_numel,
            trace_before,
            trace_after,
            max_before,
            max_after,
            iso,
        )
        return damped

    def _maybe_damp_cached_task_fisher(
        self,
        fisher: Optional[Dict[str, torch.Tensor]],
        payload: Optional[Dict],
    ) -> Optional[Dict[str, torch.Tensor]]:
        beta = float(getattr(self, "_ewc_fisher_damping_beta", 0.0))
        cached_beta = None
        if payload is not None and "main_fisher_damping_beta" in payload:
            try:
                cached_beta = float(payload.get("main_fisher_damping_beta", 0.0))
            except (TypeError, ValueError):
                cached_beta = None

        if cached_beta is not None and abs(cached_beta - beta) <= 1e-12:
            self._last_task_fisher_damping_metrics = {
                "ewc_fisher_damping_beta": beta,
                "task_fisher_damping_applied": bool(beta > 0.0),
                "task_fisher_damping_from_cache": True,
            }
            return fisher

        if cached_beta is not None and cached_beta > 0.0:
            logger.warning(
                "[EWConProbe][FisherCache] cached task Fisher was damped with "
                "beta=%.4g, but current beta=%.4g. Using cached Fisher as-is.",
                cached_beta,
                beta,
            )
            self._last_task_fisher_damping_metrics = {
                "ewc_fisher_damping_beta": beta,
                "cached_task_fisher_damping_beta": cached_beta,
                "task_fisher_damping_applied": False,
                "task_fisher_damping_from_cache": True,
            }
            return fisher

        return self._damp_task_fisher(fisher, label="Cached task Fisher")

    def _accumulate_fisher(self, task_fisher: Dict[str, torch.Tensor], model: nn.Module):
        if not task_fisher:
            return
        if self._fisher is None:
            self._fisher = {name: f.detach().cpu().clone() for name, f in task_fisher.items()}
            if self._ewc_normalize_accumulated_fisher:
                self._fisher = self._normalize_task_fisher(
                    self._fisher, label="Accumulated Fisher"
                )
            return

        next_fisher: Dict[str, torch.Tensor] = {}
        for name, p in self._iter_trainable(model):
            old = self._fisher.get(name)
            new = task_fisher.get(name)
            if old is None and new is None:
                continue
            if old is None:
                next_fisher[name] = new.detach().cpu().clone()
            elif new is None:
                next_fisher[name] = self._ewc_gamma * self._match_tensor(old, p)
            else:
                next_fisher[name] = self._ewc_gamma * self._match_tensor(old, p) + new.detach().cpu()
        if self._ewc_normalize_accumulated_fisher:
            next_fisher = self._normalize_task_fisher(
                next_fisher, label="Accumulated Fisher"
            )
        self._fisher = next_fisher

    def _fisher_blocks(
        self,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        model: Optional[nn.Module] = None,
    ) -> List[torch.Tensor]:
        source = self._fisher if fisher is None else fisher
        if not source:
            return []
        model = self._unwrap_network() if model is None else model
        blocks = []
        for name, _ in self._iter_trainable(model):
            block = source.get(name)
            if block is not None:
                blocks.append(block)
        return blocks

    def _fisher_blocks_for_eval_scope(
        self,
        eval_args: Dict,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        model: Optional[nn.Module] = None,
    ) -> List[torch.Tensor]:
        source = self._fisher if fisher is None else fisher
        if not source:
            return []
        model = self._unwrap_network() if model is None else model
        scope = str(
            eval_args.get("flat_eval_param_scope", eval_args.get("anchor_eval_param_scope", "trainable"))
        )
        blocks: List[torch.Tensor] = []
        for name, param in select_eval_named_params(model, scope):
            block = source.get(name)
            if block is None and name.startswith("module."):
                block = source.get(name[len("module."):])
            if block is None and name.startswith("model."):
                block = source.get(name[len("model."):])
            if block is None:
                block = torch.zeros_like(param.detach().cpu())
            else:
                block = self._match_tensor(block, param)
            blocks.append(block)
        return blocks

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------
    def _compute_fisher(
        self,
        loader,
        model: nn.Module,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        info = _fisher_estimator_info(self._fisher_mode)
        self._last_fisher_estimation_metrics = {
            "fisher_estimator": str(self._fisher_mode),
            "fisher_estimator_display": info["display_name"],
            "fisher_estimator_family": info["estimator_family"],
            "fisher_gradient_aggregation": info["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(info["is_per_sample_empirical_fisher"]),
        }
        if self._fisher_mode in {"ncm_empirical", "ncm_batch_empirical"}:
            return self._compute_ncm_fisher_mode(
                self._fisher_mode, loader, model, task_start, task_end
            )
        if self._fisher_mode in {"sample_empirical", "lr_sample_empirical", "true_sample"}:
            return self._compute_sample_fisher(loader, model, task_start, task_end)
        if self._fisher_mode not in {
            "batch_empirical",
            "max_pred_batch_empirical",
            "lr_batch_empirical",
        }:
            raise ValueError(
                f"Unsupported full-model fisher_mode={self._fisher_mode!r}; "
                "use ncm_empirical, batch_empirical, max_pred_batch_empirical, "
                "lr_batch_empirical, sample_empirical, lr_sample_empirical, "
                "or true_sample."
            )
        return self._compute_batch_fisher(loader, model, task_start, task_end)

    def _new_fisher_buffer(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in self._iter_trainable(model)
        }

    def _current_task_inputs(self, inputs, targets, task_start: int, task_end: int):
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
        if mask.numel() == 0:
            return None, None
        inputs = torch.index_select(inputs, 0, mask)
        targets = torch.index_select(targets, 0, mask) - task_start
        return inputs, targets

    def _ncm_prototypes(
        self,
        task_start: int,
        task_end: int,
        device,
        dtype,
    ) -> torch.Tensor:
        class_means = getattr(self, "_class_means", None)
        if class_means is None:
            raise ValueError(
                "NCM Fisher requires class means. "
                "compute_all_seen_class_means() must run before after_task()."
            )
        if int(class_means.shape[0]) < int(task_end):
            raise ValueError(
                f"NCM Fisher class means cover {class_means.shape[0]} classes, "
                f"but task_end={task_end}."
            )
        prototypes = torch.as_tensor(
            class_means[task_start:task_end],
            device=device,
            dtype=dtype,
        )
        return F.normalize(prototypes, p=2, dim=1, eps=1e-12)

    def _ncm_logits(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
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
        model: nn.Module,
        task_start: int,
        task_end: int,
    ) -> Dict[str, torch.Tensor]:
        """Batch-gradient-square proxy from CE over NCM cosine logits."""
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        prototypes: Optional[torch.Tensor] = None

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Computing Fisher [{mode}]")):
            if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
            if inputs is None:
                continue
            if prototypes is None:
                first_param = next(model.parameters())
                prototypes = self._ncm_prototypes(
                    task_start,
                    task_end,
                    device=self._device,
                    dtype=first_param.dtype,
                )

            model.zero_grad(set_to_none=True)
            logits = self._ncm_logits(model, inputs, prototypes)
            F.cross_entropy(logits, local_targets).backward()

            batch_size = int(inputs.size(0))
            for name, p in self._iter_trainable(model):
                if p.grad is not None:
                    fisher[name] += (p.grad.detach().cpu() ** 2) * batch_size
            num_samples += batch_size

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        self._last_fisher_estimation_metrics = {
            "fisher_estimator": str(mode),
            "fisher_estimator_display": _fisher_estimator_info(mode)["display_name"],
            "fisher_estimator_family": _fisher_estimator_info(mode)["estimator_family"],
            "fisher_gradient_aggregation": _fisher_estimator_info(mode)["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(
                _fisher_estimator_info(mode)["is_per_sample_empirical_fisher"]
            ),
            "fisher_num_samples": int(num_samples),
            "ncm_fisher_tau": float(self._ncm_fisher_tau),
            "ncm_fisher_num_classes": int(task_end - task_start),
        }
        logger.info(
            "[EWConProbe] NCM Fisher estimated over %d samples, classes=%d, tau=%.4f",
            num_samples,
            int(task_end - task_start),
            float(self._ncm_fisher_tau),
        )
        return fisher

    def _compute_batch_fisher(self, loader, model: nn.Module, task_start: int, task_end: int):
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Computing Fisher [{self._fisher_mode}]")):
            if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
            if inputs is None:
                continue

            model.zero_grad(set_to_none=True)
            logits = model(inputs)["logits"][:, task_start:task_end]
            if self._fisher_mode == "max_pred_batch_empirical":
                fisher_logits = logits
                fisher_targets = logits.argmax(dim=1).detach()
            else:
                fisher_logits = -logits if self._fisher_mode == "lr_batch_empirical" else logits
                fisher_targets = local_targets
            loss = F.cross_entropy(fisher_logits, fisher_targets)
            loss.backward()

            batch_size = int(inputs.size(0))
            for name, p in self._iter_trainable(model):
                if p.grad is not None:
                    fisher[name] += (p.grad.detach().cpu() ** 2) * batch_size
            num_samples += batch_size

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        info = _fisher_estimator_info(self._fisher_mode)
        self._last_fisher_estimation_metrics = {
            "fisher_estimator": str(self._fisher_mode),
            "fisher_estimator_display": info["display_name"],
            "fisher_estimator_family": info["estimator_family"],
            "fisher_gradient_aggregation": info["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(info["is_per_sample_empirical_fisher"]),
            "fisher_num_samples": int(num_samples),
            "fisher_loss_reduction": "cross_entropy_mean",
        }
        logger.info(
            "[EWConProbe] %s estimated over %d samples",
            info["display_name"], num_samples,
        )
        return fisher

    def _compute_sample_fisher(self, loader, model: nn.Module, task_start: int, task_end: int):
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Computing Fisher [{self._fisher_mode}]")):
            if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
            if inputs is None:
                continue

            for idx in range(int(inputs.size(0))):
                x_i = inputs[idx : idx + 1]
                y_i = local_targets[idx : idx + 1]
                model.zero_grad(set_to_none=True)
                logits = model(x_i)["logits"][:, task_start:task_end]
                if self._fisher_mode == "true_sample":
                    probs = torch.softmax(logits.detach(), dim=-1)
                    if not torch.isfinite(probs).all():
                        # model has diverged (NaN/inf logits); fall back to uniform sampling
                        probs = torch.ones_like(probs) / probs.size(-1)
                    fisher_target = torch.multinomial(probs, num_samples=1).squeeze(1)
                    fisher_logits = logits
                else:
                    fisher_target = y_i
                    fisher_logits = -logits if self._fisher_mode == "lr_sample_empirical" else logits
                loss = F.cross_entropy(fisher_logits, fisher_target)
                loss.backward()

                for name, p in self._iter_trainable(model):
                    if p.grad is not None:
                        fisher[name] += p.grad.detach().cpu() ** 2
                num_samples += 1

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        info = _fisher_estimator_info(self._fisher_mode)
        self._last_fisher_estimation_metrics = {
            "fisher_estimator": str(self._fisher_mode),
            "fisher_estimator_display": info["display_name"],
            "fisher_estimator_family": info["estimator_family"],
            "fisher_gradient_aggregation": info["gradient_aggregation"],
            "fisher_is_per_sample_empirical": bool(info["is_per_sample_empirical_fisher"]),
            "fisher_num_samples": int(num_samples),
            "fisher_loss_reduction": "per_sample_cross_entropy",
        }
        logger.info(
            "[EWConProbe] %s estimated over %d samples",
            info["display_name"], num_samples,
        )
        return fisher

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Multi-Fisher computation helpers
    # ------------------------------------------------------------------

    def _compute_fisher_by_mode(
        self, mode: str, loader, model: nn.Module, task_start: int, task_end: int
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to the correct Fisher estimator by mode name."""
        if mode in {"ncm_empirical", "ncm_batch_empirical"}:
            return self._compute_ncm_fisher_mode(mode, loader, model, task_start, task_end)
        if mode in {"batch_empirical", "max_pred_batch_empirical", "lr_batch_empirical"}:
            return self._compute_batch_fisher_mode(mode, loader, model, task_start, task_end)
        if mode in {"sample_empirical", "lr_sample_empirical", "true_sample"}:
            return self._compute_sample_fisher_mode(mode, loader, model, task_start, task_end)
        if mode in _KFAC_FISHER_MODES:
            return self._compute_kfac_fisher_mode(mode, loader, model, task_start, task_end)
        raise ValueError(f"[EWConProbe] Unknown fisher_mode: {mode!r}")

    def _compute_batch_fisher_mode(
        self, mode: str, loader, model: nn.Module, task_start: int, task_end: int
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-gradient-square Fisher proxy.

        Ω_i = (1/N) Σ_b |B_b| · (mean_{i∈b} ∂L/∂θ_i)²

        max_pred_batch_empirical uses argmax(model logits) as the batch target,
        matching ref/ewc.py fi_sampling_type=max_pred.

        lr_batch_empirical negates logits so CE(-o, y) has near-uniform softmax,
        preventing gradient vanishing when the model is over-confident.
        """
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Fisher [{mode}]")):
            if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
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
            bs = int(inputs.size(0))
            for name, p in self._iter_trainable(model):
                if p.grad is not None:
                    fisher[name] += (p.grad.detach().cpu() ** 2) * bs
            num_samples += bs
        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        return fisher

    def _compute_sample_fisher_mode(
        self, mode: str, loader, model: nn.Module, task_start: int, task_end: int
    ) -> Dict[str, torch.Tensor]:
        """
        Per-sample Fisher (sample_empirical, lr_sample_empirical, true_sample).

        Ω_i = (1/N) Σ_n (∂L_n/∂θ_i)²  — avoids intra-batch gradient cancellation.

        true_sample draws ỹ ~ Cat(softmax(o)) instead of using ground truth,
        recovering the FIM definition E_{ỹ~p}[(∂ log p(ỹ|x)/∂θ)²].
        """
        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Fisher [{mode}]")):
            if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
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
                    fisher_target = torch.multinomial(probs, num_samples=1).squeeze(1)
                    fisher_logits = logits
                else:
                    fisher_target = y_i
                    fisher_logits = -logits if mode == "lr_sample_empirical" else logits
                F.cross_entropy(fisher_logits, fisher_target).backward()
                for name, p in self._iter_trainable(model):
                    if p.grad is not None:
                        fisher[name] += p.grad.detach().cpu() ** 2
                num_samples += 1
        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        return fisher

    def _compute_kfac_fisher_mode(
        self,
        mode: str,
        loader,
        model: nn.Module,
        task_start: int,
        task_end: int,
        return_factors: bool = False,
    ):
        """
        KFAC Fisher for full-model Linear/Conv2d layers.

        For each layer weight matrix W (out_f × in_f_flat):
            F_diag,ij ≈ E_tokens[g_i²] · E_tokens[a_j²]
        where  a = flattened layer input,  g = ∂L/∂(layer output).

        With return_factors=True, also store full KFAC factors:
            G = E_tokens[gg^T], A = E_tokens[aa^T]
        so FQA can compute tr(G dW A dW^T).  If a trainable bias exists,
        A uses the augmented activation [a, 1] and the same KFAC block covers
        both weight and bias.

        Non-covered params fall back to batch_empirical diagonal.
        """
        model.eval()
        trainable_names = {n for n, _ in self._iter_trainable(model)}
        lr_variant = mode in {"lr_kfac_empirical", "lr_kfac_full_empirical"}

        # Per-module KFAC accumulators.
        kfac_acc: Dict[str, Dict] = {}
        hooks = []
        _current_kfac_bs: dict = {"value": 1}  # updated per batch before backward

        for mod_name, module in model.named_modules():
            w_key = f"{mod_name}.weight" if mod_name else "weight"
            b_key = f"{mod_name}.bias" if mod_name else "bias"
            if not isinstance(module, (nn.Linear, nn.Conv2d)) or w_key not in trainable_names:
                continue
            if isinstance(module, nn.Conv2d) and int(module.groups) != 1:
                continue
            out_f = int(module.weight.shape[0])
            in_f = int(module.weight[0].numel())
            include_bias = bool(module.bias is not None and b_key in trainable_names)
            a_dim = in_f + (1 if include_bias else 0)
            if return_factors:
                kfac_acc[mod_name] = {
                    "A_sum": torch.zeros(a_dim, a_dim, dtype=torch.float64),
                    "G_sum": torch.zeros(out_f, out_f, dtype=torch.float64),
                    "n_act": 0,
                    "n_grad": 0,
                    "w_key": w_key,
                    "b_key": b_key if include_bias else None,
                    "weight_shape": tuple(module.weight.shape),
                    "matrix_shape": (out_f, a_dim),
                    "include_bias": include_bias,
                    "module_type": module.__class__.__name__,
                }
            else:
                kfac_acc[mod_name] = {
                    "act2": torch.zeros(a_dim, dtype=torch.float64),
                    "grad2": torch.zeros(out_f, dtype=torch.float64),
                    "n_act": 0,
                    "n_grad": 0,
                    "w_key": w_key,
                    "b_key": b_key if include_bias else None,
                    "weight_shape": tuple(module.weight.shape),
                    "matrix_shape": (out_f, a_dim),
                    "include_bias": include_bias,
                    "module_type": module.__class__.__name__,
                }

            def _fwd(mn: str):
                def hook(mod, inp, _out):
                    x = inp[0].detach().float()
                    if isinstance(mod, nn.Conv2d):
                        a = F.unfold(
                            x,
                            kernel_size=mod.kernel_size,
                            dilation=mod.dilation,
                            padding=mod.padding,
                            stride=mod.stride,
                        ).transpose(1, 2).reshape(-1, kfac_acc[mn]["matrix_shape"][1] - int(kfac_acc[mn]["include_bias"]))
                    else:
                        # Linear input: (batch, ..., in_f) — flatten all leading dims.
                        a = x.reshape(-1, x.shape[-1])
                    if kfac_acc[mn]["include_bias"]:
                        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                        a = torch.cat([a, ones], dim=1)
                    if return_factors:
                        kfac_acc[mn]["A_sum"] += (a.T @ a).double().cpu()
                    else:
                        kfac_acc[mn]["act2"] += a.pow(2).sum(0).double().cpu()
                    kfac_acc[mn]["n_act"] += a.shape[0]
                return hook

            def _bwd(mn: str):
                def hook(mod, _gin, gout):
                    # gout[0]: gradient of loss w.r.t. layer output.
                    g_out = gout[0].detach().float()
                    if isinstance(mod, nn.Conv2d):
                        g = g_out.permute(0, 2, 3, 1).reshape(-1, g_out.shape[1])
                    else:
                        g = g_out.reshape(-1, g_out.shape[-1])
                    # CE(mean) backward gives δ/B; multiply by B to recover δ before outer product.
                    if self._kfac_grad_scale == "mean_x_bs":
                        g = g * float(_current_kfac_bs["value"])
                    if return_factors:
                        kfac_acc[mn]["G_sum"] += (g.T @ g).double().cpu()
                    else:
                        kfac_acc[mn]["grad2"] += g.pow(2).sum(0).double().cpu()
                    kfac_acc[mn]["n_grad"] += g.shape[0]
                return hook

            hooks.append(module.register_forward_hook(_fwd(mod_name)))
            hooks.append(module.register_full_backward_hook(_bwd(mod_name)))

        num_samples = 0
        try:
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Fisher [{mode}]")):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_inputs(
                    inputs, targets, task_start, task_end
                )
                if inputs is None:
                    continue
                _current_kfac_bs["value"] = int(inputs.size(0))
                model.zero_grad(set_to_none=True)
                logits = model(inputs)["logits"][:, task_start:task_end]
                fisher_logits = -logits if lr_variant else logits
                F.cross_entropy(fisher_logits, local_targets).backward()
                num_samples += int(inputs.size(0))
        finally:
            for h in hooks:
                h.remove()

        # Start from batch_empirical as fallback for all non-covered params
        # (LayerNorm, head, etc.).
        fisher = self._compute_batch_fisher_mode(
            "lr_batch_empirical" if lr_variant else "batch_empirical",
            loader, model, task_start, task_end,
        )

        # Overwrite covered layer params with the KFAC outer-product diagonal.
        kfac_by_param: Dict[str, Dict] = {}
        kfac_covered_biases = set()
        if num_samples > 0:
            for mod_name, acc in kfac_acc.items():
                w_key = acc["w_key"]
                b_key = acc.get("b_key", None)
                n_a, n_g = acc["n_act"], acc["n_grad"]
                if n_a == 0 or n_g == 0 or w_key not in fisher:
                    continue
                if return_factors:
                    A = (acc["A_sum"] / n_a).float()
                    G = (acc["G_sum"] / n_g).float()
                    act2_mean = torch.diagonal(A)
                    grad2_mean = torch.diagonal(G)
                else:
                    A = None
                    G = None
                    act2_mean = (acc["act2"] / n_a).float()    # (in_f,)
                    grad2_mean = (acc["grad2"] / n_g).float()  # (out_f,)
                weight_cols = int(torch.tensor(acc["weight_shape"][1:]).prod().item())
                weight_diag_mat = torch.outer(grad2_mean, act2_mean[:weight_cols])
                fisher[w_key] = weight_diag_mat.reshape(acc["weight_shape"])
                if b_key is not None and b_key in fisher:
                    bias_diag = grad2_mean * act2_mean[weight_cols]
                    fisher[b_key] = bias_diag.reshape_as(fisher[b_key])
                    kfac_covered_biases.add(b_key)
                if return_factors:
                    kfac_by_param[w_key] = {
                        "shape": tuple(fisher[w_key].shape),
                        "weight_shape": tuple(acc["weight_shape"]),
                        "matrix_shape": tuple(acc["matrix_shape"]),
                        "A": A.cpu(),
                        "G": G.cpu(),
                        "diag": fisher[w_key].detach().cpu(),
                        "weight_name": w_key,
                        "bias_name": b_key,
                        "param_name": w_key,
                        "module_name": mod_name,
                        "module_type": acc.get("module_type", ""),
                        "include_bias": bool(b_key is not None),
                        "stat_type": "empirical_average",
                    }

        model.train()
        if not return_factors:
            return fisher

        kfac_blocks: List[Dict] = []
        for name, p in self._iter_trainable(model):
            if name in kfac_by_param:
                kfac_blocks.append(kfac_by_param[name])
                continue
            if name in kfac_covered_biases:
                continue
            diag = fisher.get(name, None)
            block = {
                "shape": tuple(p.shape),
                "param_name": name,
                "fallback": True,
                "stat_type": "empirical_average",
            }
            if diag is not None:
                block["diag"] = diag.detach().cpu()
            kfac_blocks.append(block)
        return fisher, kfac_blocks

    def _compute_all_fisher_fused(
        self,
        loader,
        model: nn.Module,
        task_start: int,
        task_end: int,
        modes: List[str],
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, List[Dict]]]:
        """
        Single-pass multi-Fisher: shares forward+backward across batch and KFAC modes.

        Groups:
          std  : batch_empirical, kfac_empirical, kfac_full_empirical  → CE(logits, y)
          max  : max_pred_batch_empirical -> CE(logits, argmax(logits))
          lr   : lr_batch_empirical, lr_kfac_empirical, lr_kfac_full_empirical → CE(-logits, y)
                 (reuses retained graph from std backward; no second forward needed)
          ps   : sample_empirical, lr_sample_empirical, true_sample
                 (per-sample forward + up to 3 backward with retain_graph)

        ncm_empirical is excluded; it requires a separate prototype-building pass.

        Returns:
            fisher_by_mode  – {mode: {param_name: Fisher tensor}}
            kfac_blocks_map – {mode: [kfac_block_dict]}  (full KFAC modes only)
        """
        mode_set = set(modes)
        _FULL_K = {"kfac_full_empirical", "lr_kfac_full_empirical"}

        active_std_diag = {"batch_empirical"}                             & mode_set
        active_max_diag = {"max_pred_batch_empirical"}                    & mode_set
        active_lr_diag  = {"lr_batch_empirical"}                         & mode_set
        active_std_kfac = {"kfac_empirical", "kfac_full_empirical"}      & mode_set
        active_lr_kfac  = {"lr_kfac_empirical", "lr_kfac_full_empirical"} & mode_set
        active_ps       = {"sample_empirical", "lr_sample_empirical", "true_sample"} & mode_set

        need_std  = bool(active_std_diag | active_std_kfac)
        need_max  = bool(active_max_diag)
        need_lr   = bool(active_lr_diag  | active_lr_kfac)
        need_kfac = bool(active_std_kfac | active_lr_kfac)
        # Full G/A matrices (for FQA) when any full-KFAC mode is requested.
        return_factors = bool(_FULL_K & mode_set)
        need_ps = bool(active_ps)

        trainable = list(self._iter_trainable(model))
        trainable_names = {n for n, _ in trainable}

        # ── batch-group diagonal accumulators ────────────────────────────────
        # Always accumulate std/lr diagonals when the corresponding KFAC group
        # is active; these serve as fallback for non-covered params.
        _implicit_std = "batch_empirical"
        _implicit_lr  = "lr_batch_empirical"
        batch_keys_needed = set()
        if need_std:
            batch_keys_needed.add(_implicit_std)
        if need_lr:
            batch_keys_needed.add(_implicit_lr)
        batch_keys_needed |= active_std_diag | active_max_diag | active_lr_diag

        diag_acc: Dict[str, Dict[str, torch.Tensor]] = {
            m: {n: torch.zeros_like(p.data.cpu().float()) for n, p in trainable}
            for m in batch_keys_needed
        }
        diag_n: Dict[str, int] = {m: 0 for m in batch_keys_needed}

        # ── per-sample accumulators ───────────────────────────────────────────
        ps_acc: Dict[str, Dict[str, torch.Tensor]] = {
            m: {n: torch.zeros_like(p.data.cpu().float()) for n, p in trainable}
            for m in active_ps
        }
        ps_n: Dict[str, int] = {m: 0 for m in active_ps}

        # ── KFAC accumulators + hooks ─────────────────────────────────────────
        # A hooks fire once per forward pass (shared between std and lr groups).
        # G hooks route to G_sum_std or G_sum_lr via _pass_flag.
        kfac_acc: Dict[str, Dict] = {}
        _pass_flag: Dict[str, object] = {"current": "std", "enabled": False}
        _current_kfac_bs: dict = {"value": 1}  # updated per batch before backward
        hooks: List = []

        if need_kfac:
            for mod_name, module in model.named_modules():
                w_key = f"{mod_name}.weight" if mod_name else "weight"
                b_key = f"{mod_name}.bias"   if mod_name else "bias"
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                if w_key not in trainable_names:
                    continue
                if isinstance(module, nn.Conv2d) and int(module.groups) != 1:
                    continue
                is_linear = isinstance(module, nn.Linear)
                is_conv   = isinstance(module, nn.Conv2d)

                # Conv2d only covered by full-KFAC modes.
                in_std = (is_linear and bool(active_std_kfac)) or \
                         (is_conv   and bool(active_std_kfac & _FULL_K))
                in_lr  = (is_linear and bool(active_lr_kfac)) or \
                         (is_conv   and bool(active_lr_kfac  & _FULL_K))
                if not (in_std or in_lr):
                    continue

                out_f = int(module.weight.shape[0])
                in_f  = int(module.weight[0].numel())
                include_bias = bool(module.bias is not None and b_key in trainable_names)
                a_dim = in_f + (1 if include_bias else 0)

                if return_factors:
                    acc_entry: Dict = {
                        "A_sum":     torch.zeros(a_dim, a_dim, dtype=torch.float64),
                        "G_sum_std": torch.zeros(out_f, out_f, dtype=torch.float64) if in_std else None,
                        "G_sum_lr":  torch.zeros(out_f, out_f, dtype=torch.float64) if in_lr  else None,
                    }
                else:
                    acc_entry = {
                        "A_sum":     torch.zeros(a_dim, dtype=torch.float64),
                        "G_sum_std": torch.zeros(out_f, dtype=torch.float64) if in_std else None,
                        "G_sum_lr":  torch.zeros(out_f, dtype=torch.float64) if in_lr  else None,
                    }
                acc_entry.update({
                    "n_act": 0, "n_grad_std": 0, "n_grad_lr": 0,
                    "w_key": w_key, "b_key": b_key if include_bias else None,
                    "weight_shape": tuple(module.weight.shape),
                    "matrix_shape": (out_f, a_dim),
                    "include_bias": include_bias,
                    "module_type": module.__class__.__name__,
                })
                kfac_acc[mod_name] = acc_entry

                def _fwd(mn: str):
                    def hook(mod, inp, _out):
                        if not bool(_pass_flag.get("enabled", False)):
                            return
                        x = inp[0].detach().float()
                        bias_col = kfac_acc[mn]["include_bias"]
                        wc = kfac_acc[mn]["matrix_shape"][1] - int(bias_col)
                        if isinstance(mod, nn.Conv2d):
                            a = F.unfold(
                                x, kernel_size=mod.kernel_size, dilation=mod.dilation,
                                padding=mod.padding, stride=mod.stride,
                            ).transpose(1, 2).reshape(-1, wc)
                        else:
                            a = x.reshape(-1, x.shape[-1])
                        if bias_col:
                            ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                            a = torch.cat([a, ones], dim=1)
                        if return_factors:
                            kfac_acc[mn]["A_sum"] += (a.T @ a).double().cpu()
                        else:
                            kfac_acc[mn]["A_sum"] += a.pow(2).sum(0).double().cpu()
                        kfac_acc[mn]["n_act"] += int(a.shape[0])
                    return hook

                def _bwd(mn: str):
                    def hook(mod, _gin, gout):
                        if not bool(_pass_flag.get("enabled", False)):
                            return
                        g_out = gout[0].detach().float()
                        if isinstance(mod, nn.Conv2d):
                            g = g_out.permute(0, 2, 3, 1).reshape(-1, g_out.shape[1])
                        else:
                            g = g_out.reshape(-1, g_out.shape[-1])
                        # CE(mean) backward gives δ/B; multiply by B to recover δ before outer product.
                        if self._kfac_grad_scale == "mean_x_bs":
                            g = g * float(_current_kfac_bs["value"])
                        cur = _pass_flag["current"]
                        g_key = f"G_sum_{cur}"
                        if kfac_acc[mn].get(g_key) is not None:
                            if return_factors:
                                kfac_acc[mn][g_key] += (g.T @ g).double().cpu()
                            else:
                                kfac_acc[mn][g_key] += g.pow(2).sum(0).double().cpu()
                            kfac_acc[mn][f"n_grad_{cur}"] += int(g.shape[0])
                    return hook

                hooks.append(module.register_forward_hook(_fwd(mod_name)))
                hooks.append(module.register_full_backward_hook(_bwd(mod_name)))

        # ── batch loop ────────────────────────────────────────────────────────
        model.eval()
        try:
            for batch_idx, batch in enumerate(tqdm(loader, desc="Fisher [fused]")):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, local_targets = self._current_task_inputs(inputs, targets, task_start, task_end)
                if inputs is None:
                    continue
                bs = int(inputs.size(0))
                _current_kfac_bs["value"] = bs

                # ── std group: CE(logits, y) ──────────────────────────────
                logits = None
                if need_std:
                    model.zero_grad(set_to_none=True)
                    _pass_flag["current"] = "std"
                    _pass_flag["enabled"] = True
                    logits = model(inputs)["logits"][:, task_start:task_end]
                    F.cross_entropy(logits, local_targets).backward(retain_graph=need_lr)
                    _pass_flag["enabled"] = False
                    for n, p in trainable:
                        if p.grad is not None:
                            diag_acc[_implicit_std][n] += (p.grad.detach().cpu().float() ** 2) * bs
                    diag_n[_implicit_std] += bs

                # ── lr group: CE(-logits, y) — reuses retained graph ──────
                if need_lr:
                    model.zero_grad(set_to_none=True)
                    _pass_flag["current"] = "lr"
                    _pass_flag["enabled"] = True
                    if logits is None:
                        # std was skipped → fresh forward (A hooks fire here too)
                        logits = model(inputs)["logits"][:, task_start:task_end]
                    F.cross_entropy(-logits, local_targets).backward()
                    _pass_flag["enabled"] = False
                    for n, p in trainable:
                        if p.grad is not None:
                            diag_acc[_implicit_lr][n] += (p.grad.detach().cpu().float() ** 2) * bs
                    diag_n[_implicit_lr] += bs

                # -- max-pred group: CE(logits, argmax(logits)) ------------
                if need_max:
                    model.zero_grad(set_to_none=True)
                    _pass_flag["enabled"] = False
                    logits_max = model(inputs)["logits"][:, task_start:task_end]
                    max_targets = logits_max.argmax(dim=1).detach()
                    F.cross_entropy(logits_max, max_targets).backward()
                    for n, p in trainable:
                        if p.grad is not None:
                            diag_acc["max_pred_batch_empirical"][n] += (
                                p.grad.detach().cpu().float() ** 2
                            ) * bs
                    diag_n["max_pred_batch_empirical"] += bs

                # ── per-sample group ──────────────────────────────────────
                if need_ps:
                    _pass_flag["enabled"] = False
                    need_std_ps = "sample_empirical"    in active_ps
                    need_lr_ps  = "lr_sample_empirical" in active_ps
                    need_true   = "true_sample"         in active_ps
                    for idx in range(bs):
                        x_i = inputs[idx : idx + 1]
                        y_i = local_targets[idx : idx + 1]
                        model.zero_grad(set_to_none=True)
                        logits_i = model(x_i)["logits"][:, task_start:task_end]

                        if need_std_ps:
                            F.cross_entropy(logits_i, y_i).backward(
                                retain_graph=(need_lr_ps or need_true)
                            )
                            for n, p in trainable:
                                if p.grad is not None:
                                    ps_acc["sample_empirical"][n] += p.grad.detach().cpu().float() ** 2
                            ps_n["sample_empirical"] += 1
                            model.zero_grad(set_to_none=True)

                        if need_lr_ps:
                            F.cross_entropy(-logits_i, y_i).backward(retain_graph=need_true)
                            for n, p in trainable:
                                if p.grad is not None:
                                    ps_acc["lr_sample_empirical"][n] += p.grad.detach().cpu().float() ** 2
                            ps_n["lr_sample_empirical"] += 1
                            model.zero_grad(set_to_none=True)

                        if need_true:
                            probs   = torch.softmax(logits_i.detach(), dim=-1)
                            y_tilde = torch.multinomial(probs, num_samples=1).squeeze(1)
                            F.cross_entropy(logits_i, y_tilde).backward()
                            for n, p in trainable:
                                if p.grad is not None:
                                    ps_acc["true_sample"][n] += p.grad.detach().cpu().float() ** 2
                            ps_n["true_sample"] += 1
                            model.zero_grad(set_to_none=True)

        finally:
            for h in hooks:
                h.remove()
        model.train()

        # ── assemble results ──────────────────────────────────────────────────
        fisher_by_mode: Dict[str, Dict[str, torch.Tensor]] = {}
        kfac_blocks_map: Dict[str, List[Dict]] = {}

        def _norm(acc: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
            n = max(1, n)
            return {k: v / n for k, v in acc.items()}

        # Explicitly requested batch-diagonal modes
        for m in active_std_diag:
            fisher_by_mode[m] = _norm(diag_acc[_implicit_std], diag_n[_implicit_std])
        for m in active_max_diag:
            fisher_by_mode[m] = _norm(diag_acc[m], diag_n[m])
        for m in active_lr_diag:
            fisher_by_mode[m] = _norm(diag_acc[_implicit_lr], diag_n[_implicit_lr])

        # Per-sample modes
        for m in active_ps:
            fisher_by_mode[m] = _norm(ps_acc[m], ps_n[m])

        # KFAC modes: start from batch-diagonal fallback, overwrite covered params.
        def _assemble_kfac_modes(is_lr: bool, active_kfac: set) -> None:
            if not active_kfac:
                return
            base_key = _implicit_lr if is_lr else _implicit_std
            base_n   = diag_n.get(base_key, 0)
            base_acc = diag_acc.get(base_key, {})
            g_key   = "G_sum_lr"  if is_lr else "G_sum_std"
            n_g_key = "n_grad_lr" if is_lr else "n_grad_std"

            active_full   = active_kfac & _FULL_K   # full matrices, Linear+Conv2d
            active_legacy = active_kfac - _FULL_K   # diagonal, Linear only

            def _overwrite(full_mode: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], set]:
                fisher = _norm(base_acc, base_n)
                kfac_by_param: Dict[str, Dict] = {}
                covered_biases: set = set()
                for mod_name, acc in kfac_acc.items():
                    w_key  = acc["w_key"]
                    b_key  = acc.get("b_key")
                    n_a    = int(acc["n_act"])
                    n_g    = int(acc[n_g_key])
                    G_sum  = acc.get(g_key)
                    if n_a == 0 or n_g == 0 or G_sum is None or w_key not in fisher:
                        continue
                    if acc.get("module_type", "") == "Conv2d" and not full_mode:
                        continue  # legacy KFAC covers Linear only
                    if return_factors:
                        A = (acc["A_sum"] / n_a).float()
                        G = (G_sum / n_g).float()
                        act2_mean  = torch.diagonal(A)
                        grad2_mean = torch.diagonal(G)
                    else:
                        A = G = None
                        act2_mean  = (acc["A_sum"] / n_a).float()
                        grad2_mean = (G_sum / n_g).float()
                    weight_cols = int(torch.tensor(acc["weight_shape"][1:]).prod().item())
                    fisher[w_key] = torch.outer(
                        grad2_mean, act2_mean[:weight_cols]
                    ).reshape(acc["weight_shape"])
                    if b_key is not None and b_key in fisher and acc["include_bias"]:
                        fisher[b_key] = (
                            grad2_mean * act2_mean[weight_cols]
                        ).reshape_as(fisher[b_key])
                        covered_biases.add(b_key)
                    if full_mode and return_factors and A is not None and G is not None:
                        kfac_by_param[w_key] = {
                            "shape":        tuple(fisher[w_key].shape),
                            "weight_shape": tuple(acc["weight_shape"]),
                            "matrix_shape": tuple(acc["matrix_shape"]),
                            "A":            A.cpu(),
                            "G":            G.cpu(),
                            "diag":         fisher[w_key].detach().cpu(),
                            "weight_name":  w_key,
                            "bias_name":    b_key,
                            "param_name":   w_key,
                            "module_name":  mod_name,
                            "module_type":  acc.get("module_type", ""),
                            "include_bias": bool(b_key is not None),
                            "stat_type":    "empirical_average",
                        }
                return fisher, kfac_by_param, covered_biases

            def _make_blocks(fisher: Dict, kfac_by_param: Dict, covered_biases: set) -> List[Dict]:
                blocks: List[Dict] = []
                for name, p in trainable:
                    if name in kfac_by_param:
                        blocks.append(kfac_by_param[name])
                        continue
                    if name in covered_biases:
                        continue
                    block: Dict = {
                        "shape": tuple(p.shape), "param_name": name,
                        "fallback": True, "stat_type": "empirical_average",
                    }
                    d = fisher.get(name)
                    if d is not None:
                        block["diag"] = d.detach().cpu()
                    blocks.append(block)
                return blocks

            if active_full:
                fisher_f, kp_f, cb_f = _overwrite(full_mode=True)
                blocks_f = _make_blocks(fisher_f, kp_f, cb_f)
                for m in active_full:
                    fisher_by_mode[m] = fisher_f
                    kfac_blocks_map[m] = blocks_f

            if active_legacy:
                fisher_l, _, _ = _overwrite(full_mode=False)
                for m in active_legacy:
                    fisher_by_mode[m] = fisher_l

        _assemble_kfac_modes(is_lr=False, active_kfac=active_std_kfac)
        _assemble_kfac_modes(is_lr=True,  active_kfac=active_lr_kfac)

        ordered_fishers = {m: fisher_by_mode[m] for m in modes if m in fisher_by_mode}
        ordered_kfac_blocks = {m: kfac_blocks_map[m] for m in modes if m in kfac_blocks_map}
        return ordered_fishers, ordered_kfac_blocks

    def _compute_multi_fisher_if_needed(
        self,
        loader,
        model: nn.Module,
        task_start: int,
        task_end: int,
        precomputed_blocks: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Compute all configured Fisher variants; returns {mode: fisher_blocks}.

        All non-NCM modes share a single fused forward+backward pass per batch,
        cutting data passes from N_modes down to 1 (std group) + 1 (lr group, optional)
        + 1/sample (per-sample group, optional).  NCM needs a separate prototype pass
        and is handled serially before the fused call.
        """
        if not self._multi_fisher_eval:
            return {}
        result: Dict[str, List[torch.Tensor]] = {}
        self._multi_fisher_kfac_factors = {}
        precomputed_blocks = precomputed_blocks or {}

        # Register precomputed blocks (e.g. main Fisher reused for its own mode).
        for mode in self._multi_fisher_modes:
            if mode in precomputed_blocks:
                logger.info(
                    "[EWConProbe][MultiFisher] mode=%s task=%d using precomputed blocks",
                    mode, self._cur_task,
                )
                result[mode] = precomputed_blocks[mode]

        to_compute = [m for m in self._multi_fisher_modes if m not in precomputed_blocks]
        if not to_compute:
            return result

        # NCM requires a prototype-building pass; keep it separate.
        ncm_modes   = [m for m in to_compute if m == "ncm_empirical"]
        fused_modes = [m for m in to_compute if m != "ncm_empirical"]

        for mode in ncm_modes:
            logger.info("[EWConProbe][MultiFisher] mode=%s task=%d (ncm pass)", mode, self._cur_task)
            try:
                fdict = self._compute_fisher_by_mode(mode, loader, model, task_start, task_end)
                result[mode] = self._fisher_blocks(fdict, model)
            except Exception as exc:
                logger.exception("[EWConProbe][MultiFisher] mode=%s failed: %s", mode, exc)

        if fused_modes:
            logger.info(
                "[EWConProbe][MultiFisher] fused pass for modes=%s task=%d",
                fused_modes, self._cur_task,
            )
            try:
                fused_fishers, kfac_blocks_map = self._compute_all_fisher_fused(
                    loader, model, task_start, task_end, fused_modes
                )
                for mode in fused_modes:
                    fdict = fused_fishers.get(mode)
                    if fdict is not None:
                        result[mode] = self._fisher_blocks(fdict, model)
                    blocks = kfac_blocks_map.get(mode)
                    if blocks is not None:
                        self._multi_fisher_kfac_factors[mode] = blocks
            except Exception as exc:
                logger.exception("[EWConProbe][MultiFisher] fused pass failed: %s", exc)
                logger.warning("[EWConProbe][MultiFisher] falling back to serial per-mode computation")
                for mode in fused_modes:
                    logger.info("[EWConProbe][MultiFisher] mode=%s task=%d (serial fallback)", mode, self._cur_task)
                    try:
                        if mode in _FULL_KFAC_FISHER_MODES:
                            fdict, kfac_blocks = self._compute_kfac_fisher_mode(
                                mode, loader, model, task_start, task_end, return_factors=True
                            )
                            self._multi_fisher_kfac_factors[mode] = kfac_blocks
                        else:
                            fdict = self._compute_fisher_by_mode(mode, loader, model, task_start, task_end)
                        result[mode] = self._fisher_blocks(fdict, model)
                    except Exception as exc2:
                        logger.exception("[EWConProbe][MultiFisher] mode=%s serial fallback failed: %s", mode, exc2)

        return result

    def _run_multi_fisher_fqa(
        self,
        fisher_variants: Dict[str, List[torch.Tensor]],
        task_start: int,
        task_end: int,
        kfac_factor_variants: Optional[Dict[str, List[Dict]]] = None,
        endpoint_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Run FQA with each Fisher variant and return combined metrics.

        Each variant gets prefix 'mf_{mode}' so metrics are distinguished, e.g.:
            mf_batch_empirical_fqa_r0p01_error_mean
            mf_kfac_empirical_fqa_r0p01_error_mean
        """
        if not fisher_variants:
            return {}
        if not self._endpoint_eval or self._cur_task not in set(self._endpoint_eval_tasks):
            return {}

        model_view = _CurrentTaskLogitView(self._unwrap_network(), task_start, task_end)
        base_args = self._full_model_eval_args()
        base_args = self._mechanism_probe_eval_args(base_args)
        # Disable expensive second-order evals; keep only FQA per variant
        base_args["flat_eval_hessian"] = False
        base_args["flat_eval_fisher"] = False
        base_args["flat_eval_ggn"] = False
        base_args["probe_flatness_eval"] = False
        base_args["flat_eval_fisher_quadratic_error"] = True

        combined: Dict[str, float] = {}
        modes = list(fisher_variants.keys())
        if not modes:
            return combined
        kfac_factor_variants = kfac_factor_variants or {}

        # Extract Hessian / curvature scalars already computed in the endpoint eval.
        # These are passed into the FQA loop so each per-radius JSONL record can
        # include a Hessian upper-bound reference without recomputing anything.
        _HESSIAN_KEYS = [
            "endpoint_flat_lambda_max_lanczos",   # most accurate λ_max (Lanczos)
            "endpoint_flat_lambda_max",            # power-iteration λ_max
            "endpoint_hessian_top_eigen",          # λ_max from mechanism_eval Hessian
            "endpoint_flat_hessian_trace",
            "endpoint_hessian_trace",
            "endpoint_flat_emp_fisher_lambda_max",
            "endpoint_flat_emp_fisher_trace",
        ]
        hessian_scalars: Dict[str, float] = {}
        for k in _HESSIAN_KEYS:
            v = (endpoint_metrics or {}).get(k, None)
            if v is not None:
                try:
                    hessian_scalars[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if hessian_scalars:
            logger.info(
                "[EWConProbe][MultiFisherFQA] passing %d Hessian scalars to FQA: %s",
                len(hessian_scalars), list(hessian_scalars.keys()),
            )

        # Run all Fisher variants in a single FQA pass: the first mode is the main
        # prefix; remaining modes are passed as extra_omega_W and evaluated using the
        # same sampled directions and forward passes (no extra model evaluations).
        first_mode = modes[0]
        first_prefix = f"mf_{first_mode}"
        first_is_full_kfac = first_mode in _FULL_KFAC_FISHER_MODES
        extra_omega_W = {
            f"mf_{m}": fisher_variants[m]
            for m in modes[1:]
            if m not in _FULL_KFAC_FISHER_MODES
        }
        extra_kfac_factors = {
            f"mf_{m}": kfac_factor_variants[m]
            for m in modes[1:]
            if m in kfac_factor_variants
        }
        args_v = dict(base_args)
        args_v["flat_eval_fqa_prefixes"] = first_prefix
        args_v["flat_eval_fqa_tasks"] = str(self._cur_task)
        logger.info(
            "[EWConProbe][MultiFisherFQA] mode=%s prefix=%s "
            "(co-computing %d diagonal variants: %s, %d full-KFAC variants: %s)",
            first_mode, first_prefix,
            len(extra_omega_W), list(extra_omega_W.keys()),
            len(extra_kfac_factors), list(extra_kfac_factors.keys()),
        )
        try:
            metrics = evaluate_anchor_metrics(
                model_view,
                self.train_loader,
                self._device,
                args_v,
                cur_task=self._cur_task,
                task_start=task_start,
                task_end=task_end,
                omega_W=fisher_variants[first_mode],
                prefix=first_prefix,
                extra_omega_W=extra_omega_W or None,
                kfac_factors=(
                    kfac_factor_variants.get(first_mode)
                    if first_is_full_kfac else None
                ),
                extra_kfac_factors=extra_kfac_factors or None,
                partial_results_path=self._fqa_partial_results_path(first_prefix),
                hessian_scalars=hessian_scalars or None,
            )
            combined.update(metrics)
            combined.update(self._multi_fisher_summary_metrics(fisher_variants, None))
        except Exception as exc:
            logger.exception(
                "[EWConProbe][MultiFisherFQA] mode=%s failed: %s", first_mode, exc
            )
            for m in modes:
                combined[f"mf_{m}_fqa_failed"] = True
        return combined

    # Fisher cache save / load
    # ------------------------------------------------------------------
    def _fisher_cache_path(self, cache_dir: str) -> str:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"fisher_task{self._cur_task:02d}.pt")

    def _save_fisher_cache(
        self,
        save_dir: str,
        main_fisher: Optional[Dict[str, torch.Tensor]],
        main_blocks: List[torch.Tensor],
        mf_variants: Dict[str, List[torch.Tensor]],
        mf_kfac_factors: Optional[Dict[str, List[Dict]]] = None,
    ) -> None:
        path = self._fisher_cache_path(save_dir)
        payload = {
            "task": int(self._cur_task),
            "fisher_mode": self._fisher_mode,
            "main_fisher_damping_beta": float(
                getattr(self, "_ewc_fisher_damping_beta", 0.0)
            ),
            "main_fisher": {k: v.cpu() for k, v in (main_fisher or {}).items()},
            "main_fisher_blocks": [b.cpu() for b in main_blocks],
            "mf_variants": {
                mode: [b.cpu() for b in blocks]
                for mode, blocks in mf_variants.items()
            },
            "mf_kfac_factors": {
                mode: [
                    {
                        key: (value.cpu() if torch.is_tensor(value) else value)
                        for key, value in block.items()
                    }
                    for block in blocks
                ]
                for mode, blocks in (mf_kfac_factors or {}).items()
            },
        }
        torch.save(payload, path)
        logger.info(
            "[EWConProbe][FisherCache] Saved task %d Fisher to %s "
            "(main_blocks=%d, mf_modes=%s, kfac_full_modes=%s)",
            self._cur_task, path, len(main_blocks), list(mf_variants.keys()),
            list((mf_kfac_factors or {}).keys()),
        )

    def _load_fisher_cache(self, load_dir: str) -> Optional[Dict]:
        import os
        path = self._fisher_cache_path(load_dir)
        if not os.path.exists(path):
            logger.warning(
                "[EWConProbe][FisherCache] Cache not found: %s — will recompute", path
            )
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # Move tensors to model device
        device = self._device
        payload["main_fisher_blocks"] = [b.to(device) for b in payload.get("main_fisher_blocks", [])]
        payload["mf_variants"] = {
            mode: [b.to(device) for b in blocks]
            for mode, blocks in payload.get("mf_variants", {}).items()
        }
        payload["mf_kfac_factors"] = payload.get("mf_kfac_factors", {}) or {}
        if payload.get("main_fisher"):
            payload["main_fisher"] = {
                k: v.to(device) for k, v in payload["main_fisher"].items()
            }
        logger.info(
            "[EWConProbe][FisherCache] Loaded task %d Fisher from %s "
            "(main_blocks=%d, mf_modes=%s, kfac_full_modes=%s)",
            self._cur_task, path,
            len(payload["main_fisher_blocks"]),
            list(payload["mf_variants"].keys()),
            list(payload["mf_kfac_factors"].keys()),
        )
        return payload

    # Mechanism/FQA bridge
    # ------------------------------------------------------------------
    def get_mechanism_metrics(self):
        return dict(self._last_mechanism_metrics)

    def get_epoch_dynamics_metrics(self):
        return [dict(row) for row in self._last_epoch_dynamics]

    def _reset_mechanism_stats(self):
        self._mechanism_sums = {}
        self._mechanism_key_counts = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics = {}
        self._last_epoch_dynamics = []

    def _should_measure_batch_param_drift(self) -> bool:
        """Whether this batch should scan the full protected parameter vector.

        Interval 0 disables the expensive per-batch scan while preserving the
        task-end drift computed in ``after_task``.  The legacy default of 1
        retains the old behavior for existing configurations.
        """
        interval = int(self._mechanism_param_drift_interval_steps)
        return interval > 0 and self._mechanism_count % interval == 0

    def _record_mechanism_stats(self, stats: Dict[str, float], increment: bool = False):
        if not self._mechanism_eval:
            return
        for key, value in stats.items():
            self._mechanism_sums[key] = self._mechanism_sums.get(key, 0.0) + float(value)
            self._mechanism_key_counts[key] = self._mechanism_key_counts.get(key, 0) + 1
        if increment:
            self._mechanism_count += 1

    def _summarize_mechanism_stats(self):
        if self._mechanism_count <= 0:
            return {}
        return {
            f"train_{key}_mean": value / float(self._mechanism_key_counts.get(key, 1))
            for key, value in sorted(self._mechanism_sums.items())
        }

    def _compute_anchor_eval_metrics_if_needed(self, fisher_blocks: Optional[List[torch.Tensor]] = None):
        if not self._anchor_eval or self._cur_task != self._anchor_eval_task:
            return {}
        return self._evaluate_probe_metrics(fisher_blocks or [], prefix="anchor")

    def _compute_endpoint_eval_metrics_if_needed(
        self,
        fisher_blocks: Optional[List[torch.Tensor]] = None,
        extra_omega_W: Optional[Dict[str, List[torch.Tensor]]] = None,
        extra_kfac_factors: Optional[Dict[str, List[Dict]]] = None,
    ):
        if not self._endpoint_eval or self._cur_task not in set(self._endpoint_eval_tasks):
            return {}
        return self._evaluate_probe_metrics(
            fisher_blocks or [],
            prefix="endpoint",
            extra_omega_W=extra_omega_W,
            extra_kfac_factors=extra_kfac_factors,
        )

    def _compute_accumulated_fisher_metrics(self):
        topk_fractions = parse_float_list(
            self.args.get("fisher_topk_fractions", "0.001,0.01,0.05"),
            default=[0.001, 0.01, 0.05],
        )
        return fisher_buffer_metrics(
            self._fisher_blocks(),
            topk_fractions,
            prefix="accumulated_fisher",
            eps=self._mechanism_eps,
        )

    def _compute_fisher_all_seen(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Batch-empirical Fisher over all seen tasks' training data (no batch cap by default)."""
        if not hasattr(self, "_data_manager"):
            return {}, 0
        dataset = self._data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        batch_size = int(self.args.get("batch_size", 128))
        num_workers = int(self.args.get("train_num_workers", 4))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model.eval()
        fisher = self._new_fisher_buffer(model)
        num_samples = 0
        max_batches = self._full_fisher_max_batches

        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Computing full Fisher [all seen]")
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            logits = model(inputs)["logits"][:, : self._total_classes]
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            bs = int(inputs.size(0))
            for name, p in self._iter_trainable(model):
                if p.grad is not None:
                    fisher[name] += (p.grad.detach().cpu() ** 2) * bs
            num_samples += bs

        if num_samples > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(num_samples)
        model.train()
        logger.info(
            "[EWConProbe] Full Fisher (all seen tasks) estimated over %d samples", num_samples
        )
        return fisher, num_samples

    def _compute_full_fisher_diff_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compare accumulated Fisher vs full Fisher over all seen training data.

        Metrics (all scalar, logged per task):
          full_fisher_*          — statistics of the true joint Fisher F_full
          accum_fisher_*         — statistics of the online accumulated Fisher Ω
          full_vs_accum_*        — element-wise comparison between the two
        """
        if self._fisher is None:
            return {}
        full_fisher, num_samples = self._compute_fisher_all_seen(model)
        if not full_fisher:
            return {}

        accum_vals: List[torch.Tensor] = []
        full_vals: List[torch.Tensor] = []
        for name in self._fisher:
            if name in full_fisher:
                accum_vals.append(self._fisher[name].float().view(-1))
                full_vals.append(full_fisher[name].float().view(-1))
        if not accum_vals:
            return {}

        accum_flat = torch.cat(accum_vals)
        full_flat = torch.cat(full_vals)

        accum_trace = float(accum_flat.sum().item())
        full_trace  = float(full_flat.sum().item())
        diff        = full_flat - accum_flat

        # ── norms ──────────────────────────────────────────────────────────────
        accum_fro = float(accum_flat.pow(2).sum().sqrt().item())
        full_fro  = float(full_flat.pow(2).sum().sqrt().item())
        diff_fro  = float(diff.pow(2).sum().sqrt().item())

        # ── L1 ─────────────────────────────────────────────────────────────────
        accum_l1    = float(accum_flat.abs().sum().item())
        full_l1     = float(full_flat.abs().sum().item())
        abs_diff_l1 = float(diff.abs().sum().item())

        # ── trace / sum difference ──────────────────────────────────────────────
        trace_diff = float(diff.sum().item())

        # ── cosine similarity ───────────────────────────────────────────────────
        cosine_sim = float(
            (accum_flat * full_flat).sum().item()
            / max(accum_fro * full_fro, 1e-12)
        )

        # ── sign agreement: fraction of elements where sign matches ────────────
        sign_agree = float(
            ((accum_flat > 0) == (full_flat > 0)).float().mean().item()
        )

        # ── max element-wise absolute difference ───────────────────────────────
        max_abs_diff = float(diff.abs().max().item())

        # ── top-1% Fisher concentration overlap ────────────────────────────────
        # How much of the energy in the top-1% of F_full is captured by Ω?
        numel = int(accum_flat.numel())
        k1pct = max(1, int(numel * 0.01))
        full_top_idx = full_flat.topk(k1pct, largest=True).indices
        top1pct_full_sum  = float(full_flat[full_top_idx].sum().item())
        top1pct_accum_sum = float(accum_flat[full_top_idx].sum().item())

        return {
            # provenance
            "full_fisher_num_samples":       int(num_samples),
            "full_fisher_num_params":        int(numel),
            # full Fisher statistics
            "full_fisher_trace":             full_trace,
            "full_fisher_fro":               full_fro,
            "full_fisher_l1":                full_l1,
            "full_fisher_top1pct_trace":     top1pct_full_sum,
            # accumulated Fisher statistics (snapshot at comparison time)
            "accum_fisher_trace":            accum_trace,
            "accum_fisher_fro":              accum_fro,
            "accum_fisher_l1":               accum_l1,
            "accum_fisher_top1pct_by_full":  top1pct_accum_sum,
            # differences
            "full_vs_accum_trace_diff":      trace_diff,
            "full_vs_accum_abs_trace_diff":  abs(trace_diff),
            "full_vs_accum_trace_ratio":     full_trace / max(abs(accum_trace), 1e-12),
            "full_vs_accum_fro_diff":        diff_fro,
            # normalised by accumulated norm (how big is the error relative to Ω?)
            "full_vs_accum_rel_fro":         diff_fro / max(accum_fro, 1e-12),
            # normalised by full-Fisher norm (how big is the error relative to F_full?)
            "full_vs_accum_rel_fro_wrt_full": diff_fro / max(full_fro, 1e-12),
            "full_vs_accum_rel_l1":          abs_diff_l1 / max(accum_l1, 1e-12),
            "full_vs_accum_rel_l1_wrt_full": abs_diff_l1 / max(full_l1, 1e-12),
            "full_vs_accum_max_abs_diff":    max_abs_diff,
            # direction alignment
            "full_vs_accum_cosine_sim":      cosine_sim,
            "full_vs_accum_sign_agree":      sign_agree,
            # concentration: does Ω assign high weight where F_full does?
            "full_vs_accum_top1pct_accum_coverage": (
                top1pct_accum_sum / max(top1pct_full_sum, 1e-12)
            ),
        }

    def _fqa_partial_results_path(self, prefix: str) -> Optional[str]:
        """Return a file path for incremental FQA result saving, or None if not configured."""
        import os
        partial_dir = self.args.get("flat_eval_fqa_partial_dir", None)
        if not partial_dir:
            return None
        os.makedirs(partial_dir, exist_ok=True)
        run_prefix = self.args.get("prefix", "run")
        fname = f"{run_prefix}_task{self._cur_task:02d}_{prefix}_fqa_partial.json"
        return os.path.join(partial_dir, fname)

    def _evaluate_probe_metrics(
        self,
        fisher_blocks: List[torch.Tensor],
        prefix: str,
        extra_omega_W: Optional[Dict[str, List[torch.Tensor]]] = None,
        extra_kfac_factors: Optional[Dict[str, List[Dict]]] = None,
    ):
        task_start, task_end = int(self._known_classes), int(self._total_classes)
        model = _CurrentTaskLogitView(self._unwrap_network(), task_start, task_end)
        eval_args = self._full_model_eval_args()
        anchor_eval_args = self._mechanism_probe_eval_args(eval_args)
        metrics = evaluate_anchor_metrics(
            model,
            self.train_loader,
            self._device,
            anchor_eval_args,
            cur_task=self._cur_task,
            task_start=task_start,
            task_end=task_end,
            omega_W=fisher_blocks,
            prefix=prefix,
            extra_omega_W=extra_omega_W,
            extra_kfac_factors=extra_kfac_factors,
            partial_results_path=self._fqa_partial_results_path(prefix),
        )
        metrics.update(self._compute_flatness_probe_metrics_if_needed(prefix, eval_args))
        return metrics

    def _endpoint_fqa_active(self, prefix: str = "endpoint") -> bool:
        if not self._endpoint_eval or self._cur_task not in set(self._endpoint_eval_tasks):
            return False
        args = self._mechanism_probe_eval_args(self._full_model_eval_args())
        enabled = as_bool(
            args.get(
                "flat_eval_fisher_quadratic_error",
                args.get("flat_eval_unit_expected_sharpness", False),
            )
        )
        radii = parse_float_list(
            args.get("flat_eval_fqa_radii", args.get("flat_eval_unit_esh_radii", "")),
            default=[],
        )
        samples = int(
            args.get(
                "flat_eval_fqa_samples",
                args.get("flat_eval_unit_esh_samples", 10),
            )
        )
        tasks = set(parse_int_list(
            args.get("flat_eval_fqa_tasks", args.get("flat_eval_unit_esh_tasks", "0")),
            default=[0],
        ))
        prefixes = {
            item.lower()
            for item in parse_str_list(
                args.get("flat_eval_fqa_prefixes", args.get("flat_eval_unit_esh_prefixes", "endpoint")),
                default=["endpoint"],
            )
        }
        return (
            enabled
            and bool(radii)
            and samples > 0
            and int(self._cur_task) in tasks
            and str(prefix).lower() in prefixes
        )

    def _endpoint_fqa_can_co_compute_extra_fishers(self, prefix: str = "endpoint") -> bool:
        if not self._endpoint_fqa_active(prefix=prefix):
            return False
        args = self._full_model_eval_args()
        scope = str(
            args.get("flat_eval_param_scope", args.get("anchor_eval_param_scope", "lora"))
        ).lower()
        fqa_fisher_space = str(args.get("flat_eval_fqa_fisher_space", "")).lower()
        requested_delta_w = scope in {"delta_w", "dw", "effective_delta_w", "effective_update"}
        requested_lora_factor = scope in {
            "lora_factor",
            "lora_factors",
            "lora_ab",
            "ab",
            "a_b",
        }
        lora_factor_delta_w_fqa = requested_lora_factor and fqa_fisher_space in {
            "delta_w",
            "dw",
            "delta_w_induced",
            "induced_delta_w",
        }
        return not requested_delta_w and not lora_factor_delta_w_fqa

    @staticmethod
    def _multi_fisher_extra_omega(
        fisher_variants: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        return {
            f"mf_{mode}": blocks
            for mode, blocks in fisher_variants.items()
            if mode not in _FULL_KFAC_FISHER_MODES
        }

    @staticmethod
    def _multi_fisher_extra_kfac(
        kfac_factor_variants: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        return {
            f"mf_{mode}": blocks
            for mode, blocks in (kfac_factor_variants or {}).items()
        }

    def _multi_fisher_summary_metrics(
        self,
        fisher_variants: Dict[str, List[torch.Tensor]],
        shared_endpoint_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Add lightweight per-variant metadata when FQA is co-computed as extras."""
        metrics: Dict[str, float] = {}
        eval_args = self._full_model_eval_args()
        scope = str(
            eval_args.get("flat_eval_param_scope", eval_args.get("anchor_eval_param_scope", "trainable"))
        ).lower()
        topk_fractions = parse_float_list(
            eval_args.get(
                "fisher_topk_fractions",
                eval_args.get("anchor_fisher_topk_fractions", "0.001,0.01,0.05"),
            ),
            default=[0.001, 0.01, 0.05],
        )
        task_start, task_end = int(self._known_classes), int(self._total_classes)
        fqa_fisher_space = str(eval_args.get("flat_eval_fqa_fisher_space", "")).lower()
        shared_suffixes = (
            "base_loss",
            "grad_norm",
            "first_order_sharpness",
            "sh0_perturbed_loss",
            "sh0_max",
            "esh_mean",
            "esh_std",
            "fqa_base_loss",
            "fqa_grad_base_loss",
            "fqa_grad_norm",
        )
        for mode, blocks in fisher_variants.items():
            prefix = f"mf_{mode}"
            info = _fisher_estimator_info(mode)
            metrics[f"{prefix}_task"] = int(self._cur_task)
            metrics[f"{prefix}_task_start"] = task_start
            metrics[f"{prefix}_task_end"] = task_end
            metrics[f"{prefix}_param_count"] = int(sum(int(block.numel()) for block in blocks))
            metrics[f"{prefix}_param_space"] = scope
            metrics[f"{prefix}_requested_param_scope"] = scope
            metrics[f"{prefix}_fqa_fisher_space"] = fqa_fisher_space
            metrics[f"{prefix}_estimator_display"] = info["display_name"]
            metrics[f"{prefix}_estimator_family"] = info["estimator_family"]
            metrics[f"{prefix}_gradient_aggregation"] = info["gradient_aggregation"]
            metrics[f"{prefix}_is_per_sample_empirical"] = bool(
                info["is_per_sample_empirical_fisher"]
            )
            metrics.update(fisher_buffer_metrics(blocks, topk_fractions, prefix=f"{prefix}_fisher"))
            for suffix in shared_suffixes:
                endpoint_key = f"endpoint_{suffix}"
                if shared_endpoint_metrics and endpoint_key in shared_endpoint_metrics:
                    metrics[f"{prefix}_{suffix}"] = shared_endpoint_metrics[endpoint_key]
        return metrics

    def _mechanism_probe_eval_args(self, eval_args: Dict):
        anchor_args = dict(eval_args)
        if not as_bool(anchor_args.get("probe_mechanism_hessian", False)):
            anchor_args["flat_eval_hessian"] = False
            anchor_args["anchor_eval_hessian"] = False
            anchor_args["flat_eval_emp_fisher_spectrum"] = False
        fqa_max_batches = anchor_args.get(
            "flat_eval_fqa_max_batches",
            anchor_args.get("flat_eval_fqa_loss_max_batches", None),
        )
        if fqa_max_batches is not None:
            anchor_args["flat_eval_loss_max_batches"] = fqa_max_batches
        fqa_grad_batches = anchor_args.get(
            "flat_eval_fqa_grad_max_batches",
            anchor_args.get("flat_eval_fqa_first_order_grad_batches", None),
        )
        if fqa_grad_batches is not None:
            anchor_args["flat_eval_first_order_grad_batches"] = fqa_grad_batches
        return anchor_args

    def _compute_flatness_probe_metrics_if_needed(self, prefix: str, eval_args: Dict):
        if not self._should_run_flatness_probe(eval_args):
            return {}

        model = self._unwrap_network()
        task_start = int(self._known_classes)
        loader, flat_known_classes, data_scope = self._make_flatness_probe_loader(task_start)
        flat_args = self._normalize_flatness_args(eval_args)
        save_path = flat_args.get(
            "probe_flatness_save_path",
            flat_args.get("ewcon_probe_flatness_save_path", None),
        )
        if save_path:
            os.makedirs(str(save_path), exist_ok=True)
        flat_cfg = FlatnessConfig(
            args=flat_args,
            save_metrics_path=save_path,
            save_prefix=f"ewcon_probe_{prefix}_t{self._cur_task:02d}",
        )
        self._normalize_flatness_config(flat_cfg, flat_args)
        self._apply_flatness_param_scope(flat_cfg, flat_args)

        saved_requires = [(p, bool(p.requires_grad)) for _, p in model.named_parameters()]
        try:
            requested_scope = str(
                flat_args.get(
                    "flat_eval_param_scope",
                    flat_args.get("anchor_eval_param_scope", "trainable"),
                )
            ).lower()
            if requested_scope in {
                "no_fc",
                "trainable_no_fc",
                "ewc_params",
                "backbone_trainable",
                "protected",
                "ewc",
            } and not self._ewc_include_fc:
                effective_scope = "no_fc"
                flat_named_params = select_eval_named_params(model, effective_scope)
                flat_params = [param for _, param in flat_named_params]
            else:
                effective_scope = requested_scope
                flat_params = _select_params_by_name(
                    model,
                    getattr(flat_cfg, "param_name_substrings", None),
                    include_frozen=bool(getattr(flat_cfg, "include_frozen_params", False)),
                )
            if not flat_params:
                raise ValueError(
                    f"flatness parameter selection is empty for scope={effective_scope!r}"
                )
            flat_metrics = evaluate_flatness_metrics(
                model,
                loader,
                device=self._device,
                config=flat_cfg,
                known_classes=flat_known_classes,
                params_override=flat_params,
            )
            flat_metrics["param_scope"] = effective_scope
            flat_metrics["param_numel"] = int(sum(param.numel() for param in flat_params))
            flat_metrics["param_tensor_count"] = int(len(flat_params))
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("[EWConProbe] Flatness probe failed: %s", exc)
            return {
                f"{prefix}_flatness_failed": True,
                f"{prefix}_flatness_error": str(exc),
            }
        finally:
            for param, old_requires_grad in saved_requires:
                if bool(param.requires_grad) != bool(old_requires_grad):
                    param.requires_grad_(old_requires_grad)

        prefixed = {
            f"{prefix}_flat_{key}": value
            for key, value in flat_metrics.items()
        }
        prefixed[f"{prefix}_flat_data_scope"] = data_scope
        prefixed[f"{prefix}_flat_known_classes"] = (
            -1 if flat_known_classes is None else int(flat_known_classes)
        )
        return prefixed

    def _should_run_flatness_probe(self, eval_args: Dict) -> bool:
        if self._probe_flatness_eval:
            return True
        for key in ("flat_eval_hessian", "flat_eval_GGN", "flat_eval_ggn", "flat_eval_fisher"):
            if key in eval_args and as_bool(eval_args.get(key)):
                return True
        return False

    def _make_flatness_probe_loader(self, task_start: int):
        data_scope = str(
            self.args.get(
                "probe_flatness_data_scope",
                self.args.get("ewcon_probe_flatness_data_scope", "current_task"),
            )
        ).lower()
        known_classes = int(task_start)
        dataset = None

        if data_scope in {"seen", "all_seen", "seen_classes", "global"} and hasattr(
            self, "_data_manager"
        ):
            source = str(
                self.args.get(
                    "probe_flatness_data_source",
                    self.args.get("flat_eval_data_source", "train"),
                )
            )
            mode = str(
                self.args.get(
                    "probe_flatness_data_mode",
                    self.args.get("flat_eval_data_mode", "train"),
                )
            )
            dataset = self._data_manager.get_dataset(
                np.arange(0, self._total_classes),
                source=source,
                mode=mode,
            )
            known_classes = None
            data_scope = "seen"
        else:
            data_scope = "current_task"
            dataset = getattr(self.train_loader, "dataset", None)

        if dataset is None:
            return self.train_loader, known_classes, data_scope

        batch_size = parse_max_batches(
            self.args.get("flat_eval_batch_size", getattr(self.train_loader, "batch_size", None)),
            default=getattr(self.train_loader, "batch_size", None),
        )
        if batch_size is None:
            batch_size = getattr(self.train_loader, "batch_size", self.args.get("batch_size", 32))
        num_workers = int(
            self.args.get("flat_eval_num_workers", getattr(self.train_loader, "num_workers", 0) or 0)
        )
        kwargs = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.train_loader, "pin_memory", False)),
            "drop_last": False,
            "collate_fn": getattr(self.train_loader, "collate_fn", None),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(
                getattr(self.train_loader, "persistent_workers", False)
            )
        return DataLoader(**kwargs), known_classes, data_scope

    def _normalize_flatness_args(self, eval_args: Dict):
        flat_args = dict(eval_args)
        for key in ("flat_eval_loss_max_batches", "flat_eval_first_order_grad_batches"):
            if key in flat_args:
                flat_args[key] = parse_max_batches(flat_args.get(key), default=None)
        if "flat_eval_batch_size" in flat_args:
            flat_args["flat_eval_batch_size"] = parse_max_batches(
                flat_args.get("flat_eval_batch_size"), default=None
            )
        if "flat_eval_max_examples_per_batch" in flat_args:
            flat_args["flat_eval_max_examples_per_batch"] = parse_max_batches(
                flat_args.get("flat_eval_max_examples_per_batch"), default=None
            )
        for key in ("flat_eval_hessian_power_iters", "flat_eval_hessian_trace_samples"):
            if key in flat_args and flat_args[key] is not None:
                flat_args[key] = max(1, int(flat_args[key]))
        return flat_args

    def _normalize_flatness_config(self, flat_cfg: FlatnessConfig, flat_args: Dict):
        for attr in (
            "eval_sharpness",
            "eval_expected_sharpness",
            "eval_hessian",
            "eval_ggn",
            "eval_fisher",
            "disable_power",
            "include_frozen_params",
            "eig_save_vectors",
            "fisher_rao",
            "relative_flatness",
        ):
            if hasattr(flat_cfg, attr):
                setattr(flat_cfg, attr, as_bool(getattr(flat_cfg, attr)))
        flat_cfg.loss_eval_max_batches = parse_max_batches(
            getattr(flat_cfg, "loss_eval_max_batches", None), default=None
        )
        flat_cfg.first_order_grad_batches = parse_max_batches(
            getattr(flat_cfg, "first_order_grad_batches", None), default=None
        )
        flat_cfg.flat_batch_size = parse_max_batches(
            getattr(flat_cfg, "flat_batch_size", None), default=None
        )
        flat_cfg.max_examples_per_batch = parse_max_batches(
            getattr(flat_cfg, "max_examples_per_batch", None), default=None
        )
        flat_cfg.hessian_power_iters = max(1, int(getattr(flat_cfg, "hessian_power_iters", 5)))
        flat_cfg.hessian_trace_samples = max(1, int(getattr(flat_cfg, "hessian_trace_samples", 5)))
        if not as_bool(flat_args.get("probe_flatness_eval_sharpness", True)):
            flat_cfg.eval_sharpness = False

    def _apply_flatness_param_scope(self, flat_cfg: FlatnessConfig, flat_args: Dict):
        substrings = getattr(flat_cfg, "param_name_substrings", None)
        if isinstance(substrings, str):
            lowered = substrings.strip().lower()
            if lowered in {"", "none", "all"}:
                flat_cfg.param_name_substrings = None
            else:
                flat_cfg.param_name_substrings = [
                    item.strip() for item in substrings.split(",") if item.strip()
                ]
                return
        elif substrings is not None:
            return
        scope = str(
            flat_args.get(
                "flat_eval_param_scope",
                flat_args.get("anchor_eval_param_scope", "trainable"),
            )
        ).lower()
        if scope in {
            "trainable",
            "full",
            "full_model",
            "all",
            "params",
            "weights",
            "ewc",
            "protected",
        }:
            flat_cfg.param_name_substrings = None
        elif scope in {"head", "classifier", "fc"}:
            flat_cfg.param_name_substrings = ["fc", "classifier"]
        elif scope in {"backbone", "encoder", "feature", "features"}:
            flat_cfg.param_name_substrings = ["convnet", "backbone", "encoder", "feature"]
        elif scope in {"no_fc", "trainable_no_fc", "ewc_params", "backbone_trainable"}:
            # The exact exclusion is applied through params_override in
            # _compute_flatness_probe_metrics_if_needed; substring filters
            # cannot express "all trainable except fc.*".
            flat_cfg.param_name_substrings = None
        elif "," in scope:
            flat_cfg.param_name_substrings = [
                item.strip() for item in scope.split(",") if item.strip()
            ]
        else:
            flat_cfg.param_name_substrings = [scope]

    def _full_model_eval_args(self):
        eval_args = dict(self.args)
        scope = str(
            eval_args.get("flat_eval_param_scope", eval_args.get("anchor_eval_param_scope", "trainable"))
        ).lower()
        if scope in _LORA_SCOPES:
            scope = "trainable"
        if scope in {"protected", "ewc"}:
            scope = "trainable" if self._ewc_include_fc else "no_fc"
        eval_args["flat_eval_param_scope"] = scope
        eval_args["anchor_eval_param_scope"] = scope
        if str(eval_args.get("flat_eval_fqa_fisher_space", "")).lower() in {
            "delta_w",
            "dw",
            "delta_w_induced",
            "induced_delta_w",
        }:
            eval_args["flat_eval_fqa_fisher_space"] = "params"
        return eval_args

    def _unwrap_network(self) -> nn.Module:
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network
