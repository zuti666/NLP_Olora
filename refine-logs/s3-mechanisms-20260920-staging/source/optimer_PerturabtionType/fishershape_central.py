"""Dedicated central-difference entry for fixed-alpha Fisher shaping.

The task direction is g0 + alpha * Hg, where

    s = ||g0|| + eps,  u = g0 / s,
    Hg ~= s * (grad L(theta + h*u) - grad L(theta - h*u)) / (2*h).

The clean gradient and the two symmetric probes require three CE gradient
evaluations.  This estimates the gradient of the minibatch surrogate
L + alpha/2 * ||grad L||^2, not a neighborhood maximum or the per-example
Fisher trace.  It uses the shared core's fixed batch closure, common RNG,
BN control, exact parameter restoration and separate clean EWC callback.
"""

from .fishershape_richardson import RichardsonFisherShape


class CentralFisherShape(RichardsonFisherShape):
    """Three-gradient central estimator; other estimators use separate entries."""

    def __init__(self, params, base_optimizer, model, *, estimator="central", **kwargs):
        if str(estimator).lower() != "central":
            raise ValueError("CentralFisherShape requires estimator='central'")
        super().__init__(params, base_optimizer=base_optimizer, model=model,
                         estimator="central", **kwargs)

    def __repr__(self):
        return (f"CentralFisherShape({self.base_optimizer.__class__.__name__}, "
                f"rho={self.rho}, alpha={self.alpha})")
