"""Fixed-weight gradient-square shaping with controlled finite differences.

For a *single fixed minibatch loss* L, this estimates the gradient of
L + alpha/2 * ||grad L||^2.  A minibatch gradient square is not generally
the per-example empirical Fisher trace, and this is not GAM's neighborhood
maximum objective.  No correction normalization, clipping or SAM endpoint
substitution is applied.

With g0 = grad L(theta), h = rho, and detached u = g0/(||g0||+eps):
  richardson: Hg ~= (||g0||+eps) * (4*g_half - g_plus - 3*g0)/h
  central:    Hg ~= (||g0||+eps) * (g_plus - g_minus)/(2*h)
  forward:    Hg ~= (||g0||+eps) * (g_plus - g0)/h

The caller must supply a closure over the same inputs on every evaluation.
Each probe replays Torch, Python and NumPy RNG states from the clean pass;
only the clean pass advances the external stream and BN running statistics.
Parameters are restored by copying saved values, including on exceptions.
The optional inherited direction callback runs at the restored clean point
and can add the separately computed EWC gradient exactly once.

As in FisherShape, distributed training averages the final locally shaped
directions, i.e. it shapes each rank's minibatch gradient, not the square of
the globally averaged minibatch gradient.
"""

import math
import random

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from .fishershape import FisherShape


class RichardsonFisherShape(FisherShape):
    def __init__(self, params, base_optimizer, model, rho=0.05, alpha=0.2,
                 estimator="richardson", perturb_eps=1e-12,
                 rho_scheduler=None, grad_reduce="mean", alpha_mode="fixed",
                 adaptive=False, sam_equivalent=False, preserve_probe_rng=True,
                 **kwargs):
        estimator = str(estimator).lower()
        if estimator not in {"forward", "richardson", "central"}:
            raise ValueError("estimator must be forward, richardson, or central")
        if str(alpha_mode).lower() != "fixed" or adaptive or sam_equivalent:
            raise ValueError("RichardsonFisherShape requires fixed alpha, "
                             "adaptive=False, and sam_equivalent=False")
        if not preserve_probe_rng:
            raise ValueError("Finite differences require preserve_probe_rng=True")
        if not math.isfinite(float(alpha)) or float(alpha) < 0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(float(perturb_eps)) or float(perturb_eps) <= 0:
            raise ValueError("perturb_eps must be finite and positive")
        self._validate_radius(rho)
        self.estimator = estimator
        super().__init__(params, base_optimizer=base_optimizer, model=model,
                         rho=float(rho), alpha=float(alpha), alpha_mode="fixed",
                         adaptive=False, sam_equivalent=False,
                         perturb_eps=float(perturb_eps),
                         preserve_probe_rng=True, rho_scheduler=rho_scheduler,
                         grad_reduce=grad_reduce, **kwargs)
        self._validate_radius(self.rho)

    @staticmethod
    def _validate_radius(radius):
        if not math.isfinite(float(radius)) or float(radius) <= 0:
            raise ValueError("rho (finite-difference h) must be finite and positive")

    @staticmethod
    def _capture_rng():
        return (torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                random.getstate(), np.random.get_state())

    @staticmethod
    def _restore_rng(state):
        cpu, cuda, python, numpy = state
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        np.random.set_state(numpy)

    def _capture_bn(self):
        return [(module, module.momentum,
                 {name: value.detach().clone() for name, value in
                  module.named_buffers(recurse=False) if value is not None})
                for module in self.model.modules() if isinstance(module, _BatchNorm)]

    @staticmethod
    @torch.no_grad()
    def _restore_bn(snapshot, probe=False):
        for module, momentum, buffers in snapshot:
            module.momentum = 0.0 if probe else momentum
            for name, value in buffers.items():
                getattr(module, name).copy_(value)

    @staticmethod
    def _copy_gradients(params):
        gradients = []
        for param in params:
            grad = param.grad
            if grad is not None:
                if grad.is_sparse:
                    raise RuntimeError("RichardsonFisherShape requires dense gradients")
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("Non-finite CE gradient; step aborted")
                grad = grad.detach().clone()
            gradients.append(grad)
        return gradients

    @staticmethod
    @torch.no_grad()
    def _restore_parameters(params, values):
        for param, value in zip(params, values):
            param.copy_(value)

    @staticmethod
    def _norm(gradients):
        # Accumulate scalar norms in float64, including for float32 training.
        result = math.hypot(*(float(torch.linalg.vector_norm(g, dtype=torch.float64))
                              for g in gradients if g is not None))
        if not math.isfinite(result):
            raise FloatingPointError("Non-finite global gradient norm; step aborted")
        return result

    def _evaluate(self, closure):
        self.last_stats["fs_ce_evaluation_count"] += 1.0
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            result = closure()
        self.last_stats["fs_ce_backward_count"] += 1.0
        loss = result[1]
        finite = bool(torch.isfinite(loss).all()) if torch.is_tensor(loss) else math.isfinite(float(loss))
        if not finite:
            raise FloatingPointError("Non-finite CE loss; step aborted")
        return result

    @torch.no_grad()
    def _compose(self, params, clean, probes, denominator):
        estimates, dots = [], []
        for idx, (param, g0) in enumerate(zip(params, clean)):
            if g0 is None and all(probe[idx] is None for probe in probes):
                param.grad = None
                continue
            zero = torch.zeros_like(param)
            origin = zero if g0 is None else g0
            values = [zero if probe[idx] is None else probe[idx] for probe in probes]
            if self.estimator == "richardson":
                # Difference-first form avoids the gratuitous 4*g cancellation.
                difference = 4.0 * (values[0] - origin) - (values[1] - origin)
            elif self.estimator == "central":
                difference = 0.5 * (values[0] - values[1])
            else:
                difference = values[0] - origin
            estimate = difference * (denominator / self.rho)
            if not bool(torch.isfinite(estimate).all()):
                raise FloatingPointError("Non-finite finite-difference HVP; step aborted")
            estimates.append(estimate)
            dots.append(float((origin.to(torch.float64) * estimate.to(torch.float64)).sum()))
            # alpha=0 preserves clean unused-gradient semantics as well.
            param.grad = (None if g0 is None else g0.clone()) if self.alpha == 0 else (
                origin + self.alpha * estimate)
        g0_norm, rf_norm = self._norm(clean), self._norm(estimates)
        shape_norm = self.alpha * rf_norm
        if not math.isfinite(shape_norm):
            raise FloatingPointError("Non-finite shaping norm; step aborted")
        self.last_stats.update({
            "fs_g0_norm": g0_norm, "fs_rf_norm": rf_norm,
            "fs_delta_norm": rf_norm * self.rho / denominator,
            "fs_cos_g0_rf": sum(dots) / (g0_norm * rf_norm + self.perturb_eps),
            "fs_s0": self.rho / denominator, "fs_alpha_eff": self.alpha,
            "fs_shaping_component_norm": shape_norm,
            "fs_shaping_ratio": shape_norm / denominator,
            "fs_curvature_ratio": rf_norm / denominator,
            "fs_coeff_on_delta": self.alpha * denominator / self.rho,
            "fs_estimator_order": 1.0 if self.estimator == "forward" else 2.0,
        })

    def step(self, closure=None):
        self._validate_radius(self.rho)
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        get_grad = closure if closure is not None else self.forward_backward_func
        params = [p for group in self.param_groups for p in group["params"]]
        original = [p.detach().clone() for p in params]
        rng_before = self._capture_rng()
        bn_before = self._capture_bn()
        rng_after = bn_after = clean = None
        self.last_stats = {"fs_ce_evaluation_count": 0.0, "fs_ce_backward_count": 0.0}
        try:
            with self.maybe_no_sync():
                outputs, loss = self._evaluate(get_grad)
                rng_after, bn_after = self._capture_rng(), self._capture_bn()
                clean = self._copy_gradients(params)
                denominator = self._norm(clean) + self.perturb_eps
                offsets = {"richardson": (0.5, 1.0),
                           "central": (1.0, -1.0), "forward": (1.0,)}[self.estimator]
                probes = []
                for offset in offsets:
                    self._restore_parameters(params, original)
                    with torch.no_grad():
                        for param, grad in zip(params, clean):
                            if grad is not None:
                                param.add_(grad, alpha=offset * self.rho / denominator)
                    self._restore_bn(bn_after, probe=True)
                    self._restore_rng(rng_before)
                    self._evaluate(get_grad)
                    probes.append(self._copy_gradients(params))
                self._restore_parameters(params, original)
                self._restore_bn(bn_after)
                self._restore_rng(rng_after)
                self._compose(params, clean, probes, denominator)
                if callable(self._direction_projector):
                    self._direction_projector(self.param_groups)
                for param in params:
                    if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                        raise FloatingPointError("Non-finite final direction; step aborted")
            self._sync_grad()
            self._copy_gradients(params)  # Fail before applying an invalid reduced direction.
        except BaseException:
            self._restore_parameters(params, original)
            for param, grad in zip(params, clean or [None] * len(params)):
                param.grad = None if grad is None else grad.clone()
            raise
        finally:
            self._restore_bn(bn_after if bn_after is not None else bn_before)
            self._restore_rng(rng_after if rng_after is not None else rng_before)
        self.base_optimizer.step()
        return outputs, loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def __repr__(self):
        return (f"RichardsonFisherShape({self.base_optimizer.__class__.__name__}, "
                f"estimator={self.estimator}, rho={self.rho}, alpha={self.alpha})")
