"""Two-pass Fisher-trace shaping optimizer used by FisherShape-EWCLoRA.

The optimizer composes two gradients of the same task-only objective::

    g0 = grad L(theta)
    s0 = rho / (||g0|| + xi)
    g1 = grad L(theta + s0 * g0)
    d  = g0 + coeff * (g1 - g0)

``coeff`` is one of:

* ``1`` for the SAM-equivalent ablation;
* ``alpha / s0`` for the original fixed-alpha finite-difference objective;
* ``alpha * ||g0|| / (||g1-g0|| + xi)`` for the stable ratio mode.

An optional direction projector is invoked after the parameters have returned
to the clean point.  EWCLoRA uses that seam to add its analytic, clean-point
EWC gradient without putting EWC inside either FisherShape probe.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import ReduceOp


def disable_running_stats(model: nn.Module) -> None:
    """Disable BatchNorm running-stat updates during the probing pass."""

    def _disable(module: nn.Module) -> None:
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not hasattr(module, "_fishershape_backup_momentum"):
                module._fishershape_backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    """Restore BatchNorm momentum after the probing pass."""

    def _enable(module: nn.Module) -> None:
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and hasattr(
            module, "_fishershape_backup_momentum"
        ):
            module.momentum = module._fishershape_backup_momentum
            del module._fishershape_backup_momentum

    model.apply(_enable)


def fisher_trace(fisher: Dict[str, torch.Tensor]) -> float:
    """Return the global trace/sum of a diagonal Fisher dictionary."""

    return float(sum(value.detach().float().sum().item() for value in fisher.values()))


def normalize_fisher_by_trace(
    fisher: Dict[str, torch.Tensor], eps: float = 1e-12
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Trace-normalize a Fisher dictionary and return its raw trace."""

    raw_trace = fisher_trace(fisher)
    denominator = raw_trace + float(eps)
    return ({key: value / denominator for key, value in fisher.items()}, raw_trace)


def damp_trace_normalized_fisher(
    fisher: Dict[str, torch.Tensor], beta: float
) -> Dict[str, torch.Tensor]:
    """Mix a per-task Fisher with an isotropic, trace-one diagonal floor.

    For a trace-normalized Fisher this implements

    ``F <- (1 - beta) * F + beta / d``.
    """

    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"Fisher damping beta must be in [0, 1], got {beta}.")
    if beta == 0.0 or not fisher:
        return dict(fisher)

    total_numel = sum(int(value.numel()) for value in fisher.values())
    if total_numel <= 0:
        return dict(fisher)
    isotropic = beta / float(total_numel)
    return {
        key: (1.0 - beta) * value + isotropic
        for key, value in fisher.items()
    }


class FisherShape(torch.optim.Optimizer):
    """Compose a Fisher-shaping direction and delegate to a base optimizer."""

    def __init__(
        self,
        params: Iterable,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        rho: float = 0.05,
        alpha: float = 0.2,
        alpha_mode: str = "ratio",
        sam_equivalent: bool = False,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        grad_reduce: str = "mean",
        **kwargs,
    ) -> None:
        if not math.isfinite(float(rho)) or rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}.")
        if not math.isfinite(float(alpha)) or alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if not math.isfinite(float(perturb_eps)) or perturb_eps <= 0.0:
            raise ValueError(f"perturb_eps must be positive, got {perturb_eps}.")
        alpha_mode = str(alpha_mode).lower()
        if alpha_mode not in {"fixed", "ratio"}:
            raise ValueError(
                f'alpha_mode must be "fixed" or "ratio", got {alpha_mode!r}.'
            )

        defaults = dict(adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.alpha_mode = alpha_mode
        self.sam_equivalent = bool(sam_equivalent)
        self.adaptive = bool(adaptive)
        self.perturb_eps = float(perturb_eps)
        self._direction_projector: Optional[Callable] = None
        self.last_stats: Dict[str, float] = {}
        self._configure_grad_reduce(grad_reduce)

    def _configure_grad_reduce(self, grad_reduce: str) -> None:
        grad_reduce = str(grad_reduce).lower()
        if grad_reduce == "mean":
            if hasattr(ReduceOp, "AVG"):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce == "sum":
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('grad_reduce must be either "mean" or "sum".')

    def set_direction_projector(self, projector: Optional[Callable]) -> None:
        self._direction_projector = projector

    def maybe_no_sync(self):
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and hasattr(self.model, "no_sync")
        ):
            return self.model.no_sync()
        return nullcontext()

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False) -> torch.Tensor:
        norm_sq = None
        reference = None
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                reference = param
                grad = param.grad
                if weight_adaptive:
                    grad = grad * param.detach().abs()
                term = grad.detach().float().pow(2).sum()
                norm_sq = term if norm_sq is None else norm_sq + term
        if norm_sq is None:
            device = reference.device if reference is not None else torch.device("cpu")
            norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        return torch.sqrt(norm_sq)

    @torch.no_grad()
    def _perturb_weights(self) -> Tuple[float, float]:
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        if self.rho == 0.0 or float(grad_norm.item()) <= self.perturb_eps:
            scale = torch.zeros_like(grad_norm)
        else:
            scale = self.rho / (grad_norm + self.perturb_eps)
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["fs_g0"] = param.grad.detach().clone()
                perturbation = param.grad * scale.to(param)
                if self.adaptive:
                    perturbation = perturbation * param.detach().pow(2)
                param.add_(perturbation)
                self.state[param]["fs_perturbation"] = perturbation
        return float(grad_norm.item()), float(scale.item())

    @torch.no_grad()
    def _unperturb(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                perturbation = self.state[param].pop("fs_perturbation", None)
                if perturbation is not None:
                    param.sub_(perturbation)

    @torch.no_grad()
    def _clear_transient_state(self) -> None:
        """Drop failed-probe state so a later step cannot reuse stale gradients."""

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param].pop("fs_g0", None)
                self.state[param].pop("fs_perturbation", None)

    @torch.no_grad()
    def _compose_direction(self, s0: float) -> None:
        entries = []
        g0_norm_sq = 0.0
        diff_norm_sq = 0.0
        g0_dot_diff = 0.0

        for group in self.param_groups:
            for param in group["params"]:
                g0 = self.state[param].pop("fs_g0", None)
                if g0 is None:
                    # A parameter absent from the clean graph must not receive a
                    # probing-only update.
                    param.grad = None
                    continue
                if param.grad is None:
                    param.grad = g0.clone()
                diff = param.grad.detach() - g0
                entries.append((param, g0, diff))
                g0_norm_sq += float(g0.detach().float().pow(2).sum().item())
                diff_norm_sq += float(diff.detach().float().pow(2).sum().item())
                g0_dot_diff += float((g0.detach().float() * diff.detach().float()).sum().item())

        g0_norm = g0_norm_sq ** 0.5
        diff_norm = diff_norm_sq ** 0.5
        if abs(float(s0)) <= self.perturb_eps:
            # rho=0 is the unshaped baseline.  This guard also prevents
            # stochastic probe noise from being multiplied by alpha / eps in
            # fixed mode.
            coefficient = 0.0
        elif self.sam_equivalent:
            coefficient = 1.0
        elif diff_norm <= self.perturb_eps:
            coefficient = 0.0
        elif self.alpha_mode == "ratio":
            coefficient = self.alpha * g0_norm / (diff_norm + self.perturb_eps)
        else:
            coefficient = self.alpha / max(float(s0), self.perturb_eps)

        for param, g0, diff in entries:
            param.grad.copy_(g0 + coefficient * diff)

        cosine_denom = g0_norm * diff_norm + self.perturb_eps
        shaping_norm = abs(coefficient) * diff_norm
        self.last_stats = {
            "fs_g0_norm": float(g0_norm),
            "fs_delta_grad_norm": float(diff_norm),
            "fs_rf_norm": float(
                diff_norm / float(s0)
                if abs(float(s0)) > self.perturb_eps
                else 0.0
            ),
            "fs_cos_g0_rf": float(g0_dot_diff / cosine_denom),
            "fs_s0": float(s0),
            "fs_coefficient": float(coefficient),
            "fs_alpha_eff": float(coefficient * s0),
            "fs_shaping_to_g0_ratio": float(
                shaping_norm / (g0_norm + self.perturb_eps)
            ),
        }

    @torch.no_grad()
    def _sync_grad(self) -> None:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                torch.distributed.all_reduce(param.grad, op=self.grad_reduce)
                if self.manual_average:
                    param.grad.div_(float(torch.distributed.get_world_size()))

    def step(self, closure=None, clip_grad_norm=None, first_grad_ready: bool = False):
        if closure is None:
            raise ValueError("FisherShape requires a forward-backward closure.")

        outputs = None
        loss_value = None
        running_stats_disabled = False
        direction_ready = False
        if first_grad_ready and not any(
            param.grad is not None
            for group in self.param_groups
            for param in group["params"]
        ):
            raise RuntimeError(
                "first_grad_ready=True requires clean-point gradients on the optimizer parameters."
            )
        try:
            # In the standard two-closure path, suppress DDP synchronization
            # until the final composed direction. In the Trainer integration,
            # g0 is already accumulated/synchronized by the normal first pass;
            # its replay helper synchronizes g1, so no outer no_sync is used.
            sync_context = nullcontext() if first_grad_ready else self.maybe_no_sync()
            with sync_context:
                # The closure owns zero_grad so each pass represents the same
                # effective accumulation window rather than a gradient sum.
                if not first_grad_ready:
                    outputs, loss_value = closure()
                    if not any(
                        param.grad is not None
                        for group in self.param_groups
                        for param in group["params"]
                    ):
                        raise RuntimeError(
                            "FisherShape clean closure produced no parameter gradients."
                        )
                _, s0 = self._perturb_weights()
                disable_running_stats(self.model)
                running_stats_disabled = True
                probe_outputs, probe_loss = closure()
                if first_grad_ready:
                    outputs, loss_value = probe_outputs, probe_loss
                self._unperturb()
                self._compose_direction(s0)
                if callable(self._direction_projector):
                    self._direction_projector(self.param_groups)
                direction_ready = True
        finally:
            # Idempotent and also protects against an exception part-way
            # through perturbation construction.
            self._unperturb()
            if running_stats_disabled:
                enable_running_stats(self.model)
            if not direction_ready:
                self._clear_transient_state()
                self.zero_grad(set_to_none=True)

        self._sync_grad()
        if clip_grad_norm is not None and float(clip_grad_norm) > 0.0:
            parameters = [
                param
                for group in self.param_groups
                for param in group["params"]
                if param.grad is not None
            ]
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, float(clip_grad_norm))
        self.base_optimizer.step()
        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def __repr__(self) -> str:
        return (
            f"FisherShape({self.base_optimizer.__class__.__name__}, rho={self.rho}, "
            f"alpha={self.alpha}, alpha_mode={self.alpha_mode!r}, "
            f"sam_equivalent={self.sam_equivalent})"
        )
