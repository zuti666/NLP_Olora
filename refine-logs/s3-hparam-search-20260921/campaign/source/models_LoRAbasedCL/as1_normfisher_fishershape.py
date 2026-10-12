"""Factor-space FisherShape with AS1 trace-normalized delta-W EWC.

This learner deliberately keeps the two geometries used by the existing
``as1_normfisher_gam`` implementation:

* current-task shaping is performed in the trainable parameter space (LoRA
  A/B factors and the classifier head);
* past-task Fisher and the EWC quadratic are defined in effective
  ``delta_W = B @ A`` space;
* autograd pulls the delta-W EWC gradient back to A/B, and that gradient is
  added only after the two CE-only FisherShape probes have been composed.

Consequently one implementation supplies the controlled directions

    fs_direction_mode=native_sgd : g_0 + g_EWC (one CE pass)
    fs_direction_mode=alpha0     : g_0 + g_EWC (two-pass wrapper sanity)
    fs_direction_mode=sam        : g_1 + g_EWC
    fs_direction_mode=fixed      : g_0 + alpha r_F + g_EWC
    fs_direction_mode=ratio      : legacy normalized shaping + g_EWC

without allowing EWC to affect the probing point or finite difference.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import torch
from torch import optim

from models_LoRAbasedCL.as1_normfisher_gam import Learner as _AS1NormFisherGAM
from optimer_PerturabtionType.fishershape import FisherShape
from utils.mechanism_eval import as_bool


logger = logging.getLogger(__name__)


class Learner(_AS1NormFisherGAM):
    """AS1 delta-W EWC combined with factor-space FisherShape."""

    _OPTIMIZER_ALIASES = {
        "fishershape",
        "fishershape_factor",
        "factor_fishershape",
        "fs_factor",
    }

    _DIRECTION_ALIASES = {
        "sgd": "native_sgd",
        "native": "native_sgd",
        "native-sgd": "native_sgd",
        "native_sgd": "native_sgd",
        "alpha0": "alpha0",
        "alpha_0": "alpha0",
        "fixed": "fixed",
        "fixed_fs": "fixed",
        "fixed-fs": "fixed",
        "ratio": "ratio",
        "ratio_fs": "ratio",
        "ratio-fs": "ratio",
        "sam": "sam",
    }

    _FS_DISTRIBUTION_STATS = (
        "fs_shaping_ratio",
        "fs_curvature_ratio",
        "fs_s0",
        "fs_alpha_eff",
    )

    def __init__(self, args):
        # Make the dedicated learner usable without an explicit optimizer key.
        if "optimizer_type" not in args:
            args["optimizer_type"] = "fishershape"
        super().__init__(args)

        self._fs_rho = float(args.get("fs_rho", args.get("gam_grad_rho", 0.05)))
        self._fs_alpha = float(args.get("fs_alpha", 0.2))
        self._fs_alpha_mode = str(args.get("fs_alpha_mode", "ratio")).lower()
        self._fs_xi = float(args.get("fs_xi", args.get("fs_perturb_eps", 1e-12)))
        self._fs_sam_equivalent = as_bool(args.get("fs_sam_equivalent", False))
        self._fs_adaptive = as_bool(args.get("fs_adaptive", False))
        self._fs_grad_reduce = str(args.get("fs_grad_reduce", "mean"))
        self._fs_preserve_probe_rng = as_bool(
            args.get("fs_preserve_probe_rng", False)
        )
        requested_direction = str(args.get("fs_direction_mode", "")).strip().lower()
        if requested_direction:
            self._fs_direction_mode = self._DIRECTION_ALIASES.get(
                requested_direction, requested_direction
            )
        elif self._fs_sam_equivalent:
            self._fs_direction_mode = "sam"
        else:
            # Backward compatibility: existing configurations continue to be
            # interpreted solely through fs_alpha_mode/fs_alpha.
            self._fs_direction_mode = self._fs_alpha_mode

        if self._fs_direction_mode not in set(self._DIRECTION_ALIASES.values()):
            raise ValueError(
                "fs_direction_mode must be one of "
                f"{sorted(set(self._DIRECTION_ALIASES.values()))}, "
                f"got {self._fs_direction_mode!r}"
            )
        if self._fs_direction_mode == "fixed":
            self._fs_alpha_mode = "fixed"
            self._fs_sam_equivalent = False
        elif self._fs_direction_mode == "ratio":
            self._fs_alpha_mode = "ratio"
            self._fs_sam_equivalent = False
        elif self._fs_direction_mode == "sam":
            self._fs_sam_equivalent = True
        elif self._fs_direction_mode == "alpha0":
            self._fs_sam_equivalent = False
            self._fs_alpha = 0.0

        self._fs_mechanism_samples: Dict[str, List[float]] = {}
        fisher_damping_raw = float(
            args.get(
                "ewc_fisher_damping_beta",
                args.get("ewc_fisher_damping_alpha", args.get("fisher_damping_alpha", 0.0)),
            )
        )
        self._ewc_fisher_damping_beta = max(0.0, min(1.0, fisher_damping_raw))
        self._last_task_fisher_damping_metrics: Dict[str, float] = {}

        if self._fs_alpha_mode not in {"fixed", "ratio"}:
            raise ValueError(
                f'fs_alpha_mode must be "fixed" or "ratio", got {self._fs_alpha_mode!r}'
            )
        if self._optimizer_type not in self._OPTIMIZER_ALIASES:
            raise ValueError(
                "as1_normfisher_fishershape requires optimizer_type in "
                f"{sorted(self._OPTIMIZER_ALIASES)}, got {self._optimizer_type!r}. "
                "Use fs_alpha/fs_sam_equivalent for the controlled "
                "SGD/SAM/FS comparison."
            )
        if self._ewc_fisher_damping_beta > 0.0 and not self._normalize_fisher:
            logger.warning(
                "[AS1-FS-LoRA] ewc_fisher_damping_beta=%.4g is intended for "
                "trace-normalized per-task delta-W Fisher, but "
                "ewc_normalize_fisher=False.",
                self._ewc_fisher_damping_beta,
            )

        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA] space=factor direction=%s rho=%.4g alpha=%.4g alpha_mode=%s "
                "xi=%.3g sam_equivalent=%s adaptive=%s preserve_probe_rng=%s "
                "lambda=%.4g gamma=%.4g "
                "normalize_fisher=%s normalize_accumulated=%s fisher_damping_beta=%.4g",
                self._fs_direction_mode,
                self._fs_rho,
                self._fs_alpha,
                self._fs_alpha_mode,
                self._fs_xi,
                self._fs_sam_equivalent,
                self._fs_adaptive,
                self._fs_preserve_probe_rng,
                self._lambda_ewc,
                self._ewc_gamma,
                self._normalize_fisher,
                self._normalize_accumulated_fisher,
                self._ewc_fisher_damping_beta,
            )

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
                "[AS1-FS-LoRA] Task delta-W Fisher damped beta=%.4g numel=%d "
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
        # ``params`` is normally a list (possibly containing FC/LoRA parameter
        # group dicts). Materialize generators so both optimizers see the same
        # objects.
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

        if self._fs_direction_mode == "native_sgd":
            if base_name != "sgd":
                raise ValueError(
                    "fs_direction_mode=native_sgd requires fs_base_optimizer=sgd; "
                    f"got {base_name!r}"
                )
            self._last_optimizer_class = type(base_optimizer).__name__
            return base_optimizer

        optimizer = FisherShape(
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
            preserve_probe_rng=self._fs_preserve_probe_rng,
        )
        self._last_optimizer_class = type(optimizer).__name__
        return optimizer

    # ------------------------------------------------------------------
    # Add delta-W EWC pullback after the factor-space FS direction
    # ------------------------------------------------------------------
    def _build_additive_fisher_projector(
        self,
        fisher_grad_cache: Dict[int, torch.Tensor],
        step_stats: Dict[str, float],
    ):
        collect = bool(self._mechanism_grad_stats)
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            fs_norm_sq = 0.0
            fisher_norm_sq = 0.0
            combined_norm_sq = 0.0
            fs_fisher_dot = 0.0

            with torch.no_grad():
                for group in param_groups:
                    for param in group["params"]:
                        if not param.requires_grad:
                            continue

                        fs_grad = None if param.grad is None else param.grad.detach().clone()
                        fisher_grad = fisher_grad_cache.get(id(param))
                        if fisher_grad is not None:
                            fisher_grad = fisher_grad.to(
                                device=param.device,
                                dtype=param.dtype,
                            )

                        if fs_grad is None and fisher_grad is None:
                            continue
                        if fs_grad is None:
                            combined = fisher_grad.clone()
                        elif fisher_grad is None:
                            combined = fs_grad
                        else:
                            combined = fs_grad + fisher_grad

                        if param.grad is None:
                            param.grad = combined.clone()
                        else:
                            param.grad.copy_(combined)

                        if collect:
                            if fs_grad is not None:
                                fs_norm_sq += float(fs_grad.pow(2).sum().item())
                            if fisher_grad is not None:
                                fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                            if fs_grad is not None and fisher_grad is not None:
                                fs_fisher_dot += float((fs_grad * fisher_grad).sum().item())
                            combined_norm_sq += float(combined.pow(2).sum().item())

            if collect:
                fs_norm = fs_norm_sq ** 0.5
                fisher_norm = fisher_norm_sq ** 0.5
                step_stats.update(
                    {
                        "fs_dir_norm": float(fs_norm),
                        "fisher_norm": float(fisher_norm),
                        "combined_norm": float(combined_norm_sq ** 0.5),
                        "fisher_to_fs_norm_ratio": float(fisher_norm / (fs_norm + eps)),
                        "cos_fs_fisher": self._cosine_from_sums(
                            fs_fisher_dot,
                            fs_norm_sq,
                            fisher_norm_sq,
                            eps,
                        ),
                    }
                )

        return projector

    # ------------------------------------------------------------------
    # One controlled FS/SAM/SGD batch step
    # ------------------------------------------------------------------
    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        if self._fs_direction_mode == "native_sgd":
            if isinstance(optimizer, FisherShape):
                raise TypeError(
                    "fs_direction_mode=native_sgd requires a plain base optimizer"
                )
            params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)
            fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)

            optimizer.zero_grad()
            outputs = self._network(inputs)
            clean_loss, _, _ = self._task_loss(
                outputs["logits"], targets, class_offset
            )
            clean_loss_value = float(clean_loss.detach().item())
            clean_loss.backward()

            projector_stats: Dict[str, float] = {}
            self._build_additive_fisher_projector(
                fisher_grad_cache, projector_stats
            )(optimizer.param_groups)
            optimizer.step()

            self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)
            if self._mechanism_grad_stats:
                projector_stats.update(
                    {
                        "fs_native_one_pass": 1.0,
                        "fs_ce_backward_count": 1.0,
                    }
                )
                self._accumulate_mechanism_grad_stats(projector_stats)

            return outputs["logits"].detach(), float(
                clean_loss_value + float(fisher_loss_value)
            )

        if not isinstance(optimizer, FisherShape):
            raise TypeError(
                "as1_normfisher_fishershape requires a FisherShape optimizer, "
                f"got {type(optimizer).__name__}"
            )

        params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)

        # This is the existing AS1 delta-W penalty. autograd.grad performs the
        # exact chain-rule pullback from delta_W=B@A to the A/B factors. It is
        # evaluated at the clean point and never enters either CE closure.
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        projector_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_additive_fisher_projector(fisher_grad_cache, projector_stats)
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
            # FisherShape returns the first, clean-point outputs and CE value.
            outputs, clean_loss = optimizer.step(closure=closure)
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

    def _reset_mechanism_stats(self) -> None:
        super()._reset_mechanism_stats()
        self._fs_mechanism_samples = {}

    def _accumulate_mechanism_grad_stats(self, stats: Dict[str, float]) -> None:
        super()._accumulate_mechanism_grad_stats(stats)
        if not self._mechanism_grad_stats:
            return
        for key in self._FS_DISTRIBUTION_STATS:
            value = stats.get(key)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                self._fs_mechanism_samples.setdefault(key, []).append(value)

    def _summarize_mechanism_grad_stats(self) -> Dict[str, float]:
        summary = super()._summarize_mechanism_grad_stats()
        for key, values in sorted(self._fs_mechanism_samples.items()):
            if not values:
                continue
            sample = torch.tensor(values, dtype=torch.float64)
            for label, quantile in (
                ("p50", 0.50),
                ("p90", 0.90),
                ("p95", 0.95),
                ("p99", 0.99),
            ):
                summary[f"grad_{key}_{label}"] = float(
                    torch.quantile(sample, quantile).item()
                )
            summary[f"grad_{key}_max"] = float(sample.max().item())
        return summary

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_fishershape",
                    "fs_space": "factor",
                    "fs_direction_mode": str(self._fs_direction_mode),
                    "fs_rho": float(self._fs_rho),
                    "fs_alpha": float(self._fs_alpha),
                    "fs_alpha_mode": str(self._fs_alpha_mode),
                    "fs_sam_equivalent": bool(self._fs_sam_equivalent),
                    "fs_preserve_probe_rng": bool(self._fs_preserve_probe_rng),
                    "mechanism_final": True,
                    **self._last_task_fisher_damping_metrics,
                }
            )
            # Parent classes enrich this dictionary in successive after_task
            # layers, after the legacy mechanism line has already been
            # printed.  Emit one final, uniquely named record so collectors
            # can recover the direction, fixed-alpha and Fisher metadata from
            # the actual log rather than guessing from a filename.
            if self._is_main_process:
                self._log(
                    f"[AS1-FS-LoRA][FinalMechanism] {self._last_mechanism_metrics}"
                )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
