"""Fixed-objective finite-difference Fisher shaping with AS1 delta-W EWC.

The current-task direction is g + alpha * H g.  Richardson uses three CE
gradient evaluations at the clean, half-radius, and full-radius parameters.
The inherited learner supplies the same clean-point delta-W EWC pullback,
task Fisher estimation, training loop, and evaluation as the references.
"""

from __future__ import annotations

import logging

from torch import optim

from models_LoRAbasedCL.as1_normfisher_fishershape import Learner as _BaseFSLearner
from optimer_PerturabtionType.fishershape_richardson import RichardsonFisherShape
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_BaseFSLearner):
    """AS1 EWC with fixed-alpha forward / Richardson / central Hg estimation."""

    _OPTIMIZER_ALIASES = {"fishershape_richardson"}
    _OPTIMIZER_CLASS = RichardsonFisherShape

    def __init__(self, args):
        args.setdefault("optimizer_type", "fishershape_richardson")
        args.setdefault("fs_alpha", 0.01)
        args.setdefault("fs_alpha_mode", "fixed")
        args.setdefault("fs_direction_mode", "fixed")
        args.setdefault("fs_preserve_probe_rng", True)
        if str(args["fs_alpha_mode"]).lower() != "fixed":
            raise ValueError("Richardson FS optimizes a fixed-alpha objective; fs_alpha_mode must be fixed.")
        if str(args["fs_direction_mode"]).lower() != "fixed":
            raise ValueError("Richardson FS requires fs_direction_mode=fixed.")
        if as_bool(args.get("fs_adaptive", False)) or as_bool(args.get("fs_sam_equivalent", False)):
            raise ValueError("Richardson FS requires fs_adaptive=false and fs_sam_equivalent=false.")
        if not as_bool(args["fs_preserve_probe_rng"]):
            raise ValueError("Richardson FS requires fs_preserve_probe_rng=true.")
        self._fs_estimator = str(args.get("fs_estimator", "richardson")).lower()
        if self._fs_estimator not in {"forward", "richardson", "central"}:
            raise ValueError("fs_estimator must be forward, richardson, or central.")
        super().__init__(args)
        if self._is_main_process:
            logger.info(
                "[AS1-FS-FD] estimator=%s objective=CE+alpha/2*||grad_CE||^2+EWC "
                "alpha=%.6g h=%.6g CE_evaluations=%d EWC_at_clean=True",
                self._fs_estimator, self._fs_alpha, self._fs_rho,
                2 if self._fs_estimator == "forward" else 3,
            )

    def _build_optimizer(self, params, stage):
        if not isinstance(params, (list, tuple)):
            params = list(params)
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base_name = str(
            self.args.get("fs_base_optimizer", self.args.get("optimizer", "sgd"))
        ).lower()
        if base_name in {"adam", "adamw"}:
            base_class = optim.Adam if base_name == "adam" else optim.AdamW
            base_optimizer = base_class(
                params, lr=lr, weight_decay=weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            )
        elif base_name == "sgd":
            base_optimizer = optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported fs_base_optimizer: {base_name!r}")

        optimizer = self._OPTIMIZER_CLASS(
            params, base_optimizer=base_optimizer, model=self._network,
            rho=self._fs_rho, alpha=self._fs_alpha, alpha_mode="fixed",
            sam_equivalent=False, adaptive=False, perturb_eps=self._fs_xi,
            grad_reduce=self._fs_grad_reduce, preserve_probe_rng=True,
            estimator=self._fs_estimator,
        )
        self._last_optimizer_class = type(optimizer).__name__
        return optimizer

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update({
                "method_type": "as1_normfisher_fishershape_richardson",
                "fs_estimator": self._fs_estimator,
                "fs_objective": "CE + alpha/2 * ||grad_CE||^2 + EWC",
                "fs_hg_scale_restored": True,
                "fs_expected_ce_evaluations": 2 if self._fs_estimator == "forward" else 3,
            })
            if self._is_main_process:
                self._log(f"[AS1-FS-FD][FinalMechanism] {self._last_mechanism_metrics}")


class EWCLoRA(Learner):
    """Alias consistent with the existing AS1 learner modules."""
