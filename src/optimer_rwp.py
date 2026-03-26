"""
RWP (Random Weight Perturbation) optimizer.

Two-level design (mirrors optimer_sam.py):
  - RWP(torch.optim.Optimizer)  — wraps *any* base optimizer (SGD, Adam, AdamW …)
  - RWPAdamW(AdamW)             — convenience single-class variant

Two-pass update per step
--------------------------
  1. Clean forward-backward  → g₀
  2. Add noise ε to target params (controlled by ``rwp_range``)
  3. Noisy forward-backward  → g₁
  4. Restore clean parameters
  5. Mix: g = rwp_lambda · g₁ + (1 − rwp_lambda) · g₀
  6. Base-optimizer step with the mixed gradient

References
----------
  mARWP: https://github.com/nblt/mARWP
  LPF-SGD: https://github.com/devansh20la/LPF-SGD
"""

import math
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------
# BatchNorm helpers (consistent with optimer_sam.py)
# ---------------------------------------------------------------------------

def disable_running_stats(model: nn.Module) -> None:
    """Disable BatchNorm running-statistics accumulation (call before noisy pass)."""
    def _disable(m):
        if isinstance(m, nn.BatchNorm2d):
            m.backup_momentum = m.momentum
            m.momentum = 0
    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    """Restore BatchNorm momentum (call after second step)."""
    def _enable(m):
        if isinstance(m, nn.BatchNorm2d) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum
    model.apply(_enable)


# ---------------------------------------------------------------------------
# RWP noise generator
# ---------------------------------------------------------------------------

def generate_pertubation(param: torch.Tensor,
                         pertubation_mode: str,
                         std: float,
                         fisher_param: torch.Tensor = None,
                         fisher_scaler: float = 2.0) -> torch.Tensor:
    """
    Generate a noise perturbation tensor for RWP.

    Parameters
    ----------
    param : torch.Tensor
        The parameter whose shape/device/dtype the noise should match.
    pertubation_mode : str
        One of the modes listed below.
    std : float
        Base noise standard deviation σ.
    fisher_param : torch.Tensor, optional
        Fisher-information tensor (same shape as ``param``).
        Only used by Fisher-based modes.
    fisher_scaler : float
        Scaling factor η for Fisher-based attenuation. Default 2.0.

    Supported modes
    ---------------
    "Gauss_standard"   ε ~ N(0, σ²)  — iid, same σ everywhere
    "Gauss_element"    ε ~ N(0, σ² |W_ij|²)  — element-wise adaptive
    "Gauss_matrix"     ε ~ N(0, σ² ‖W‖_F²)  — Frobenius-scaled
    "lpf_sgd_Gauss"    per-row L2-norm scaled Gaussian  (LPF-SGD style)
    "mARWP_fisher"     row-norm Gaussian ÷ sqrt(1 + η · fisher_row)
    "flatLoRA_Gauss"   row-norm Gaussian with √n normalisation
    (anything else)    fallback → "Gauss_standard"

    Returns
    -------
    torch.Tensor
        Noise tensor, same shape / device / dtype as ``param``.
    """
    if param.numel() == 0:
        return torch.zeros_like(param)

    device = param.device
    dtype  = param.dtype

    # ── Gauss_standard ───────────────────────────────────────────────
    if pertubation_mode == "Gauss_standard":
        return torch.randn_like(param, device=device, dtype=dtype) * float(std)

    # ── Gauss_element ────────────────────────────────────────────────
    elif pertubation_mode == "Gauss_element":
        scaler = float(std) * (param.abs() + 1e-16)
        return torch.randn_like(param, device=device, dtype=dtype) * scaler

    # ── Gauss_matrix ─────────────────────────────────────────────────
    elif pertubation_mode == "Gauss_matrix":
        fro_norm = torch.norm(param, p='fro')
        scaler   = float(std) * fro_norm
        return torch.randn_like(param, device=device, dtype=dtype) * scaler

    # ── lpf_sgd_Gauss ────────────────────────────────────────────────
    elif pertubation_mode == "lpf_sgd_Gauss":
        if param.dim() > 1:
            sh     = param.shape
            sh_mul = int(np.prod(sh[1:]))
            row_norms       = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            row_norms_param = row_norms.repeat(1, sh_mul).view(sh)
            scaler = float(std) * row_norms_param
            return torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            scale = float(std) * (param.view(-1).norm().item() + 1e-16)
            return torch.randn_like(param, device=device, dtype=dtype) * scale

    # ── mARWP_fisher ─────────────────────────────────────────────────
    elif pertubation_mode == "mARWP_fisher":
        sh = param.shape
        if param.dim() > 1:
            sh_mul    = int(np.prod(sh[1:]))
            row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            std_mat   = row_norms.repeat(1, sh_mul).view(sh)
            scaler    = float(std) * std_mat
            noise     = torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            scale = float(std) * (param.view(-1).norm().item() + 1e-16)
            noise = torch.randn_like(param, device=device, dtype=dtype) * scale
        if fisher_param is not None:
            if param.dim() > 1:
                sh_mul     = int(np.prod(sh[1:]))
                fisher_row = (fisher_param.view(sh[0], -1)
                              .sum(dim=1, keepdim=True)
                              .repeat(1, sh_mul).view(sh))
            else:
                fisher_row = fisher_param
            noise = noise / torch.sqrt(1.0 + float(fisher_scaler) * fisher_row)
        return noise

    # ── flatLoRA_Gauss ───────────────────────────────────────────────
    elif pertubation_mode == "flatLoRA_Gauss":
        if param.dim() > 1:
            sh     = param.shape
            sh_mul = int(np.prod(sh[1:]))
            row_norms  = param.view(sh[0], -1).norm(dim=1, keepdim=True)
            std_matrix = row_norms.repeat(1, sh_mul).view(sh)
            scaler     = float(std) / math.sqrt(sh[1]) * std_matrix
            return torch.randn_like(param, device=device, dtype=dtype) * scaler
        else:
            n     = param.shape[0]
            scale = float(std) / math.sqrt(max(n, 1)) * (param.view(-1).norm().item() + 1e-16)
            return torch.randn_like(param, device=device, dtype=dtype) * scale

    # ── fallback: standard Gaussian ──────────────────────────────────
    else:
        logger.debug(
            f"[RWP] Unknown pertubation_mode '{pertubation_mode}', "
            "falling back to Gauss_standard"
        )
        return torch.randn_like(param, device=device, dtype=dtype) * float(std)


# ---------------------------------------------------------------------------
# RWP — wraps any base optimizer  (mirrors SAM interface in optimer_sam.py)
# ---------------------------------------------------------------------------

class RWP(torch.optim.Optimizer):
    """
    Random Weight Perturbation optimizer wrapper.

    Wraps *any* PyTorch base optimizer and performs the two-pass RWP update
    automatically inside ``step(closure)``.  The interface mirrors
    :class:`optimer_sam.SAM`: a ``first_step`` / ``second_step`` pair plus
    a combined ``step(closure)`` shortcut.

    Parameters
    ----------
    params :
        Iterable of parameters or parameter groups.
    base_optimizer : type
        Base optimizer **class** (not instance), e.g. ``torch.optim.AdamW``.
    std : float
        Noise standard deviation σ. Default 0.01.
    rwp_lambda : float
        Gradient mixing weight λ ∈ [0, 1].
        Effective gradient = λ · g₁(noisy) + (1−λ) · g₀(clean).
        Default 0.5.
    rwp_range : str
        ``"lora"`` — perturb only params with ``requires_grad=True``.
        ``"all"``  — perturb every non-empty parameter.
        Default ``"lora"``.
    pertubation_mode : str
        Noise mode forwarded to :func:`generate_pertubation`.
        Default ``"Gauss_standard"``.
    rwp_eta : float
        Fisher scaling factor η (only used by Fisher-aware modes).
        Default 1.0.
    **kwargs :
        Extra keyword arguments forwarded verbatim to ``base_optimizer``.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        std: float = 0.01,
        rwp_lambda: float = 0.5,
        rwp_range: str = "lora",
        pertubation_mode: str = "Gauss_standard",
        rwp_eta: float = 1.0,
        **kwargs,
    ):
        assert std >= 0.0, f"Invalid noise std, must be non-negative: {std}"
        assert 0.0 <= rwp_lambda <= 1.0, (
            f"rwp_lambda must be in [0, 1], got {rwp_lambda}"
        )

        defaults = dict(std=std, rwp_lambda=rwp_lambda, **kwargs)
        super().__init__(params, defaults)

        # Mirror SAM pattern: base optimizer owns param_groups
        self.base_optimizer   = base_optimizer(self.param_groups, **kwargs)
        self.param_groups     = self.base_optimizer.param_groups

        self.std              = std
        self.rwp_lambda       = rwp_lambda
        self.rwp_range        = rwp_range
        self.pertubation_mode = pertubation_mode
        self.rwp_eta          = rwp_eta

        # Fisher EMA buffer: param id → Tensor.  Updated externally when needed.
        self._fisher: dict = {}

    # ------------------------------------------------------------------
    def _is_perturb_target(self, p: torch.Tensor) -> bool:
        """True if this parameter should receive noise perturbation."""
        if p.numel() == 0:
            return False
        if self.rwp_range == "lora":
            return bool(p.requires_grad)
        return True   # "all"

    # ------------------------------------------------------------------
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        **Pass 1** — save clean gradient g₀ and add noise to target params.

        Call *after* the clean forward-backward pass so that ``.grad``
        tensors are populated.

        Parameters
        ----------
        zero_grad : bool
            Zero gradients after saving them. Default False.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._is_perturb_target(p):
                    continue
                # Snapshot clean param and gradient
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["g_0"]   = p.grad.detach().clone()
                # Generate perturbation and add it in-place
                fisher_param = self._fisher.get(id(p), None)
                e = generate_pertubation(
                    p,
                    pertubation_mode=self.pertubation_mode,
                    std=self.std,
                    fisher_param=fisher_param,
                    fisher_scaler=self.rwp_eta,
                )
                p.data.add_(e)

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        **Pass 2** — mix g₀ and g₁, restore params, run base-optimizer step.

        Call *after* the noisy forward-backward pass so that ``.grad``
        tensors hold g₁.

        Parameters
        ----------
        zero_grad : bool
            Zero gradients after the update. Default False.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore clean weights
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
                # Mix: g = λ · g₁ + (1−λ) · g₀
                if "g_0" in self.state[p]:
                    p.grad.data.mul_(self.rwp_lambda).add_(
                        self.state[p]["g_0"], alpha=(1.0 - self.rwp_lambda)
                    )

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        """
        Full two-pass RWP step driven by a closure.

        Parameters
        ----------
        closure : callable() -> Tensor
            Must perform a full forward + backward pass and return the loss.
            Called **twice** internally: once on clean weights, once on
            perturbed weights.

        Returns
        -------
        torch.Tensor
            Clean loss value (detached scalar).
        """
        assert closure is not None, (
            "RWP requires a closure for the two-pass update, but none was provided."
        )
        closure = torch.enable_grad()(closure)

        # ── Pass 1: clean ────────────────────────────────────────────
        loss = closure()
        self.first_step(zero_grad=True)

        # ── Pass 2: noisy ────────────────────────────────────────────
        closure()

        # ── Mix + restore + base-optimizer step ──────────────────────
        self.second_step()

        return loss.detach() if torch.is_tensor(loss) else loss

    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def __repr__(self):
        return (
            f"RWP({self.base_optimizer.__class__.__name__}, "
            f"std={self.std}, λ={self.rwp_lambda}, "
            f"mode={self.pertubation_mode})"
        )


# ---------------------------------------------------------------------------
# RWPAdamW — convenience single-class variant  (mirrors SAMAdamW)
# ---------------------------------------------------------------------------

class RWPAdamW(AdamW):
    """
    RWP wrapped directly around AdamW (no separate base-optimizer instance).

    Drop-in replacement for AdamW that performs the two-pass RWP update.
    Mirrors :class:`optimer_sam.SAMAdamW`.

    Parameters
    ----------
    params :
        Parameter groups.
    lr : float
        Learning rate. Default 1e-3.
    std : float
        Noise std σ. Default 0.01.
    rwp_lambda : float
        Gradient mixing weight λ ∈ [0, 1]. Default 0.5.
    rwp_range : str
        ``"lora"`` or ``"all"``. Default ``"lora"``.
    pertubation_mode : str
        Noise mode passed to :func:`generate_pertubation`.
        Default ``"Gauss_standard"``.
    rwp_eta : float
        Fisher η factor. Default 1.0.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        std: float = 0.01,
        rwp_lambda: float = 0.5,
        rwp_range: str = "lora",
        pertubation_mode: str = "Gauss_standard",
        rwp_eta: float = 1.0,
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        self.std              = std
        self.rwp_lambda       = rwp_lambda
        self.rwp_range        = rwp_range
        self.pertubation_mode = pertubation_mode
        self.rwp_eta          = rwp_eta
        self._fisher: dict    = {}

    def _is_perturb_target(self, p: torch.Tensor) -> bool:
        if p.numel() == 0:
            return False
        if self.rwp_range == "lora":
            return bool(p.requires_grad)
        return True

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:   # type: ignore[override]
        """Two-pass RWP step.  ``closure`` must do forward + backward."""
        closure = torch.enable_grad()(closure)

        # ── Pass 1: clean ────────────────────────────────────────────
        self.zero_grad()
        loss     = closure()
        loss_val = loss.detach().clone()

        g0: dict = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and self._is_perturb_target(p):
                    g0[id(p)] = p.grad.detach().clone()

        # ── Add noise ────────────────────────────────────────────────
        saved: dict = {}
        for group in self.param_groups:
            for p in group["params"]:
                if not self._is_perturb_target(p):
                    continue
                saved[id(p)] = p.data.clone()
                fisher_param = self._fisher.get(id(p), None)
                e = generate_pertubation(
                    p,
                    pertubation_mode=self.pertubation_mode,
                    std=self.std,
                    fisher_param=fisher_param,
                    fisher_scaler=self.rwp_eta,
                )
                p.data.add_(e)

        # ── Pass 2: noisy ────────────────────────────────────────────
        self.zero_grad()
        closure()

        # ── Restore + mix ────────────────────────────────────────────
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in saved:
                    p.data.copy_(saved[pid])
                if p.grad is not None and pid in g0:
                    p.grad.data.mul_(self.rwp_lambda).add_(
                        g0[pid], alpha=(1.0 - self.rwp_lambda)
                    )

        # ── AdamW update with mixed gradient ─────────────────────────
        super().step()
        return loss_val
