"""Default central-difference sensitivity regularization with AS1 EWC.

The trainable factor-space objective is CE + alpha/2 * ||grad CE||^2 + EWC.
With s=||g0||+eps and the fixed u=g0/s, use clean CE gradients at theta and
theta +/- h*u, then d = g0 + alpha*s*(g_plus-g_minus)/(2*h) + clean EWC.
"""
from __future__ import annotations

from models_LoRAbasedCL.as1_normfisher_fishershape_richardson import Learner as _FDLearner
from optimer_PerturabtionType.fishershape_central import CentralFisherShape


class Learner(_FDLearner):
    """AS1 delta-W EWC with a dedicated three-gradient central default."""

    _OPTIMIZER_ALIASES = {"fishershape_central"}
    _OPTIMIZER_CLASS = CentralFisherShape

    def __init__(self, args):
        args.setdefault("optimizer_type", "fishershape_central")
        args.setdefault("fs_estimator", "central")
        if str(args["fs_estimator"]).lower() != "central":
            raise ValueError("The central learner requires fs_estimator=central.")
        super().__init__(args)

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update({
                "method_type": "as1_normfisher_fishershape_central",
                "fs_estimator": "central",
                "fs_expected_ce_evaluations": 3,
                "fs_probe_offsets": [-1.0, 0.0, 1.0],
            })
            if self._is_main_process:
                self._log(f"[AS1-CentralFS][FinalMechanism] {self._last_mechanism_metrics}")


class EWCLoRA(Learner):
    """Alias consistent with the other AS1 learners."""

