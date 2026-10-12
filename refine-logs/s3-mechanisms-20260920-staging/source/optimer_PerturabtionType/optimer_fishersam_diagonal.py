from __future__ import annotations

from typing import Any, Optional

import torch


class DiagonalFisherSAM(torch.optim.Optimizer):
    """
    Diagonal Fisher-SAM optimizer with a SAM-compatible two-step API.

    Fisher-SAM solves the first-order inner problem

        max_eps g^T eps  subject to  eps^T F eps <= gamma^2

    with the closed-form perturbation

        eps = gamma * F^{-1} g / sqrt(g^T F^{-1} g).

    This implementation uses the paper's diagonal minibatch approximation
    F ~= Diag(g^2).  By default it follows the experimental implementation
    note in the paper and uses diag(F^{-1}) = 1 / (1 + eta * g^2).

    The public methods intentionally mirror optimer_sam.SAM:
        loss.backward()
        optimizer.first_step(zero_grad=True)
        second_loss.backward()
        optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: Optional[float] = None,
        gamma: Optional[float] = None,
        eta: float = 1.0,
        damping: float = 1e-12,
        norm_eps: float = 1e-12,
        adaptive: bool = False,
        inverse_mode: str = "paper",
        **kwargs,
    ):
        if gamma is None:
            gamma = 0.1 if rho is None else rho
        if rho is None:
            rho = gamma

        gamma = float(gamma)
        rho = float(rho)
        eta = float(eta)
        damping = float(damping)
        norm_eps = float(norm_eps)

        if gamma < 0.0:
            raise ValueError(f"Invalid gamma, should be non-negative: {gamma}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta, should be non-negative: {eta}")
        if damping <= 0.0:
            raise ValueError(f"Invalid damping, should be positive: {damping}")
        if norm_eps <= 0.0:
            raise ValueError(f"Invalid norm_eps, should be positive: {norm_eps}")
        if inverse_mode not in {"paper", "damping"}:
            raise ValueError(
                f"Invalid inverse_mode: {inverse_mode}. "
                "Choose from ['paper', 'damping']."
            )

        defaults = dict(
            rho=rho,
            gamma=gamma,
            eta=eta,
            damping=damping,
            norm_eps=norm_eps,
            adaptive=bool(adaptive),
            inverse_mode=inverse_mode,
            **kwargs,
        )
        super().__init__(params, defaults)

        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._set_param_group_defaults(
            rho=rho,
            gamma=gamma,
            eta=eta,
            damping=damping,
            norm_eps=norm_eps,
            adaptive=bool(adaptive),
            inverse_mode=inverse_mode,
        )

    def _set_param_group_defaults(self, **defaults: Any) -> None:
        for group in self.param_groups:
            for key, value in defaults.items():
                group.setdefault(key, value)

    def _first_param(self) -> torch.Tensor:
        for group in self.param_groups:
            for p in group["params"]:
                return p
        raise ValueError("DiagonalFisherSAM received an empty parameter list.")

    def _has_any_gradient(self) -> bool:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return True
        return False

    @staticmethod
    def _extract_loss(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in reversed(value):
                loss = DiagonalFisherSAM._extract_loss(item)
                if torch.is_tensor(loss):
                    return loss
        return None

    def _backward_if_closure_only_returned_loss(self, closure_result) -> None:
        if self._has_any_gradient():
            return
        loss = self._extract_loss(closure_result)
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()

    @torch.no_grad()
    def _fisher_preconditioned_grad(self, p: torch.Tensor, group) -> torch.Tensor:
        grad = p.grad
        fisher_diag = grad.pow(2)

        if group["inverse_mode"] == "paper":
            inv_fisher_diag = 1.0 / (1.0 + group["eta"] * fisher_diag + group["damping"])
        else:
            inv_fisher_diag = 1.0 / (fisher_diag + group["damping"])

        direction = inv_fisher_diag * grad
        if group["adaptive"]:
            direction = direction * p.detach().pow(2)
        return direction

    @torch.no_grad()
    def _fisher_sam_norm(self) -> torch.Tensor:
        first_param = self._first_param()
        shared_device = first_param.device
        norm_sq = torch.zeros((), device=shared_device, dtype=torch.float32)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                direction = self._fisher_preconditioned_grad(p, group)
                term = (p.grad * direction).sum().detach()
                norm_sq.add_(term.to(device=shared_device, dtype=torch.float32))

        return torch.sqrt(torch.clamp(norm_sq, min=0.0))

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Move from theta to theta + eps_FSAM using gradients at theta.
        """
        fisher_sam_norm = self._fisher_sam_norm()

        for group in self.param_groups:
            norm_eps = group.get("norm_eps", 1e-12)
            scale = group["gamma"] / (fisher_sam_norm + norm_eps)
            scale = scale.to(device=fisher_sam_norm.device, dtype=fisher_sam_norm.dtype)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.detach().clone(memory_format=torch.preserve_format)
                e_w = self._fisher_preconditioned_grad(p, group)
                e_w = e_w * scale.to(device=p.device, dtype=p.dtype)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Restore theta and apply the base optimizer using gradients from theta + eps.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" not in state:
                    continue
                p.copy_(state["old_p"])

        self.base_optimizer.step()
        self._clear_perturbation_state()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self):
        """
        Restore theta from theta + eps without applying a base optimizer step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" not in state:
                    continue
                p.copy_(state["old_p"])
        self._clear_perturbation_state()

    def _clear_perturbation_state(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state.pop("old_p", None)
                state.pop("e_w", None)

    @torch.no_grad()
    def step(self, closure=None, zero_grad: bool = True):
        """
        Convenience wrapper compatible with SAM-style closures.

        If gradients are already populated, this behaves like optimer_sam.SAM.step:
        it uses the existing gradients for the first perturbation, runs closure at
        theta + eps, then updates at theta. If gradients are absent, closure is
        run once first to populate them.
        """
        if closure is None:
            raise RuntimeError("DiagonalFisherSAM requires a closure.")

        closure = torch.enable_grad()(closure)

        if not self._has_any_gradient():
            clean_result = closure()
            self._backward_if_closure_only_returned_loss(clean_result)

        self.first_step(zero_grad=True)
        fisher_sam_result = closure()
        self._backward_if_closure_only_returned_loss(fisher_sam_result)
        self.second_step(zero_grad=zero_grad)
        return fisher_sam_result

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups


FisherSAM = DiagonalFisherSAM
FSAM = DiagonalFisherSAM
DiagonalFSAM = DiagonalFisherSAM

__all__ = ["DiagonalFisherSAM", "FisherSAM", "FSAM", "DiagonalFSAM"]
