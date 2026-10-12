"""GAMFisherShape optimizer variants.

Phase 1 intentionally implements only the SAM-equivalent member of the
GAM/FisherShape direction family.  It reuses the hardened two-pass
FisherShape machinery but fixes the composed direction to the probing-point
gradient::

    g0 = grad L(theta)
    theta_probe = theta + rho * g0 / (||g0|| + xi)
    g1 = grad L(theta_probe)
    g_final = g0 + 1 * (g1 - g0) = g1

No four-probe GAM decomposition is run in this mode.  EWCLoRA can use the
trainer's clean-point additive projector to append OLoRA and EWC gradients;
those terms never enter the two CE probes.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from optimer_fishershape import FisherShape


class GAMFisherShapeSAM(FisherShape):
    """Two-pass GAMFisherShape mode whose direction is exactly SAM's ``g1``."""

    SUPPORTED_MODE = "sam"

    def __init__(
        self,
        params: Iterable,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        rho: float = 0.05,
        mode: str = SUPPORTED_MODE,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        grad_reduce: str = "mean",
        **kwargs,
    ) -> None:
        normalized_mode = str(mode).strip().lower().replace("-", "_")
        if normalized_mode not in {"sam", "sam_equivalent"}:
            raise ValueError(
                "GAMFisherShape phase 1 only supports mode='sam'; "
                f"got {mode!r}."
            )
        self.mode = self.SUPPORTED_MODE
        super().__init__(
            params,
            base_optimizer=base_optimizer,
            model=model,
            rho=rho,
            alpha=0.0,
            alpha_mode="ratio",
            sam_equivalent=True,
            adaptive=adaptive,
            perturb_eps=perturb_eps,
            grad_reduce=grad_reduce,
            **kwargs,
        )

    @torch.no_grad()
    def _compose_direction(self, s0: float) -> None:
        super()._compose_direction(s0)
        # A zero/near-zero clean gradient intentionally uses FisherShape's
        # stable g0 fallback.  Every non-degenerate SAM step must be coeff=1.
        if abs(float(s0)) > self.perturb_eps:
            coefficient = float(self.last_stats.get("fs_coefficient", 0.0))
            if coefficient != 1.0:
                raise RuntimeError(
                    "GAMFisherShape-SAM expected coefficient=1 for a "
                    f"non-degenerate probe, got {coefficient}."
                )
        self.last_stats.update(
            {
                "gamfs_sam_equivalent": 1.0,
                "gamfs_gam_probe_count": 0.0,
            }
        )

    def __repr__(self) -> str:
        return (
            "GAMFisherShapeSAM("
            f"{self.base_optimizer.__class__.__name__}, rho={self.rho}, "
            f"adaptive={self.adaptive})"
        )
