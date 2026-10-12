"""Efficient / multi-probe Factor-FS LoRA learner.

This learner keeps the AS1 delta-W EWC protocol from
``as1_normfisher_fishershape`` and only changes the current-task optimizer:

* expensive FisherShape/SAM probes can be computed every K steps;
* on non-probe steps the optimizer can either use pure SGD or reuse the latest
  cached sharpness component, following LookSAM's motivation;
* probe steps can average several perturbed finite differences before composing
  the final factor-space direction.

EWC is still clean-point additive and never enters the probing closure.
"""

from __future__ import annotations

import logging

from torch import optim

from models_LoRAbasedCL.as1_normfisher_fishershape import Learner as _BaseFSLearner
from optimer_PerturabtionType.fishershape_efficient import EfficientFisherShape


logger = logging.getLogger(__name__)


class Learner(_BaseFSLearner):
    """AS1 delta-W EWC with efficient multi-probe factor-space FisherShape."""

    _OPTIMIZER_ALIASES = {
        "fishershape_efficient",
        "efficient_fishershape",
        "look_fishershape",
        "lookfs",
        "efs_factor",
    }

    def __init__(self, args):
        if "optimizer_type" not in args:
            args["optimizer_type"] = "fishershape_efficient"
        super().__init__(args)
        self._fs_probe_interval = max(1, int(args.get("fs_probe_interval", 1)))
        self._fs_probe_repeats = max(1, int(args.get("fs_probe_repeats", 1)))
        self._fs_probe_mode = str(args.get("fs_probe_mode", "multiradius")).lower()
        self._fs_reuse_mode = str(args.get("fs_reuse_mode", "look")).lower()
        self._fs_reuse_scale = float(args.get("fs_reuse_scale", 1.0))
        self._fs_cache_momentum = float(args.get("fs_cache_momentum", 0.0))

        if self._is_main_process:
            logger.info(
                "[AS1-EfficientFS-LoRA] probe_interval=%d probe_repeats=%d "
                "probe_mode=%s reuse_mode=%s reuse_scale=%.4g cache_momentum=%.4g",
                self._fs_probe_interval,
                self._fs_probe_repeats,
                self._fs_probe_mode,
                self._fs_reuse_mode,
                self._fs_reuse_scale,
                self._fs_cache_momentum,
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

        return EfficientFisherShape(
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
            probe_interval=self._fs_probe_interval,
            probe_repeats=self._fs_probe_repeats,
            probe_mode=self._fs_probe_mode,
            reuse_mode=self._fs_reuse_mode,
            reuse_scale=self._fs_reuse_scale,
            cache_momentum=self._fs_cache_momentum,
        )

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_fishershape_efficient",
                    "fs_probe_interval": float(self._fs_probe_interval),
                    "fs_probe_repeats": float(self._fs_probe_repeats),
                    "fs_probe_mode": str(self._fs_probe_mode),
                    "fs_reuse_mode": str(self._fs_reuse_mode),
                    "fs_reuse_scale": float(self._fs_reuse_scale),
                    "fs_cache_momentum": float(self._fs_cache_momentum),
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
