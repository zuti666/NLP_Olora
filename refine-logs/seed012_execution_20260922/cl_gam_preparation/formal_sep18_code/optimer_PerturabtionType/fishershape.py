"""Fisher-Shape optimizer: two-gradient Fisher-trace shaping (FS-EWC method).

Implements the current-task Fisher-sensitivity surrogate from the FS-EWC
design (``newmethod_design_wiki``):

    R_F(theta) = 0.5 * tr(F_t(theta))  ~  0.5 * ||g_0||^2   (minibatch surrogate)
    grad R_F  ~=  H_0 g_0              (Hessian-vector product)

The HVP is estimated with a forward finite difference on the SAM ascent point,
so each step needs exactly two forward-backward passes (same cost as SAM,
half of GAM):

    g_0 = grad L_CE(theta; B)
    s_0 = rho / (||g_0||_2 + xi)
    theta_+ = theta + s_0 * g_0          # probing point (SAM ascent step)
    g_1 = grad L_CE(theta_+; B)          # same minibatch
    r_F = (g_1 - g_0) / s_0  ~=  H_0 g_0
    d   = g_0 + alpha * r_F              # final update direction

Relations to SAM / GAM:
  * ``sam_equivalent=True`` sets alpha = s_0 each step, so d = g_1 exactly:
    the optimizer reduces to SAM.  With a fixed ``alpha`` the shaping weight
    is decoupled from ||g_0||, which is the FS-EWC contribution.
  * Compared to GAM, the norm-ascent branch and ``gradient_decompose``
    (beta_1..3 / gamma projection) are removed: no g_2/g_3 passes, the trace
    surrogate replaces the neighborhood max-gradient-norm objective.

Interface is kept drop-in compatible with ``optimer_PerturabtionType.gam.GAM``:
closure-based ``step`` returning ``(outputs, loss_value)``,
``set_direction_projector`` for analytic EWC gradient injection, DDP grad
sync, and BN running-stat handling on the second pass.
"""

import contextlib

import torch
from torch.distributed import ReduceOp

from .util import enable_running_stats, disable_running_stats


class FisherShape(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho=0.05, alpha=0.2,
                 alpha_mode="fixed", sam_equivalent=False, adaptive=False,
                 perturb_eps=1e-12, rho_scheduler=None, grad_reduce='mean',
                 preserve_probe_rng=False,
                 **kwargs):
        """
        Args:
            params: trainable parameters (or param groups).
            base_optimizer: constructed optimizer instance (SGD/Adam/AdamW)
                performing the actual update with the shaped direction.
            model: the network, needed for BN running-stat toggling and DDP
                no_sync during the extra pass.
            rho: nominal finite-difference probing radius.  With the numerical
                floor xi, the non-adaptive perturbation norm is at most rho
                (and approaches rho when ||g_0|| >> xi).
            alpha: Fisher-shaping weight. Meaning depends on alpha_mode:
                'fixed' - absolute weight on r_F: d = g_0 + alpha * r_F.
                    Has curvature units; diverges unless alpha << 1/lambda_max.
                'ratio' - dimensionless relative strength c: the shaping
                    component has norm at most c * ||g_0|| at every step,
                        d = g_0 + c * ||g_0|| * delta_g/(||delta_g||+eps).
                    The leading finite-difference scale cancels to first order;
                    at finite rho, the direction of delta_g may still depend on
                    rho through curvature beyond the local linearization.
            alpha_mode: 'fixed' | 'ratio'. Ignored when sam_equivalent.
            sam_equivalent: use alpha = s_0 per step, recovering exactly SAM
                (d = g_1). Ablation switch; overrides alpha_mode.
            adaptive: SAM-style per-parameter rescaling of the perturbation.
                The finite-difference identity r_F ~= H_0 g_0 is exact only
                for adaptive=False; keep False for the FS-EWC method.
            perturb_eps: numerical floor xi added to ||g_0||.
            rho_scheduler: optional object with .step() returning a new rho.
            grad_reduce: 'mean' or 'sum' for DDP gradient reduction.
            preserve_probe_rng: replay the clean-pass Torch RNG state for the
                probe pass, then restore the post-clean state.  This gives the
                finite difference common random numbers and makes alpha=0
                consume exactly the same Torch RNG stream as one-pass SGD.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(adaptive=adaptive, **kwargs)
        super(FisherShape, self).__init__(params, defaults)

        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        alpha_mode = str(alpha_mode).lower()
        if alpha_mode not in {"fixed", "ratio"}:
            raise ValueError(f'alpha_mode must be "fixed" or "ratio", got {alpha_mode!r}')

        self.rho = rho
        self.alpha = alpha
        self.alpha_mode = alpha_mode
        self.sam_equivalent = sam_equivalent
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.preserve_probe_rng = bool(preserve_probe_rng)
        self.rho_scheduler = rho_scheduler
        self.get_grad_reduce(grad_reduce)

        self._direction_projector = None
        # Per-step diagnostics for mechanism logging (read-only for learners).
        self.last_stats = {}

        self.update_rho_t()

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, average manually
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        if self.rho_scheduler is not None:
            self.rho = self.rho_scheduler.step()

    def set_direction_projector(self, projector):
        """Attach a callback that can modify gradients before the base step."""
        self._direction_projector = projector

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False):
        device = None
        dtype = None
        norm = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.device
                    dtype = p.dtype
                if weight_adaptive:
                    term = torch.sum((p.grad * torch.abs(p)) ** 2)
                else:
                    term = torch.sum(p.grad ** 2)
                norm = term if norm is None else norm + term
        if norm is None:
            norm = torch.zeros((), device=device or 'cpu', dtype=dtype or torch.float32)
        return torch.sqrt(norm)

    @torch.no_grad()
    def perturb_weights(self):
        """Save g_0 and move to the probing point theta_+ = theta + s_0 g_0."""
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_0"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        return float(grad_norm.item()), float(scale.item())

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if "e_w" in self.state[p]:
                    p.data.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]

    @torch.no_grad()
    def compose_direction(self, s_0: float):
        """Build d = g_0 + coeff * (g_1 - g_0) in-place on p.grad.

        p.grad currently holds g_1 (gradient at the probing point), so
        (g_1 - g_0) = s_0 * r_F.  The coefficient on this difference is:
            sam_equivalent : 1                        -> d = g_1 (exact SAM)
            fixed          : alpha / s_0              -> d = g_0 + alpha*r_F
            ratio          : c*||g_0||/(||g_1-g_0||)  -> d = g_0 + c*||g_0||*r_F/||r_F||
        In ratio mode the leading s_0 scale cancels.  This does not imply exact
        rho-invariance at finite probing radius because delta_g can rotate.
        """
        # Pass 1: gather global norms (needed before writing in ratio mode).
        entries = []
        g0_norm_sq = 0.0
        diff_norm_sq = 0.0
        g0_dot_diff = 0.0
        for group in self.param_groups:
            for p in group['params']:
                g_0 = self.state[p].get("g_0")
                if g_0 is None:
                    continue
                if p.grad is None:
                    # Second pass produced no gradient; fall back to g_0.
                    p.grad = g_0.clone()
                    del self.state[p]["g_0"]
                    continue
                entries.append((p, g_0))
                diff = p.grad.data - g_0  # s_0 * H_0 g_0
                g0_norm_sq += float(torch.sum(g_0 ** 2).item())
                diff_norm_sq += float(torch.sum(diff ** 2).item())
                g0_dot_diff += float(torch.sum(g_0 * diff).item())

        g0_norm = g0_norm_sq ** 0.5
        diff_norm = diff_norm_sq ** 0.5

        if self.sam_equivalent:
            coeff = 1.0
        elif self.alpha_mode == "ratio":
            coeff = self.alpha * g0_norm / (diff_norm + self.perturb_eps)
        else:  # fixed
            coeff = self.alpha / max(s_0, self.perturb_eps)

        # Pass 2: write the shaped direction.
        for p, g_0 in entries:
            p.grad.data = g_0 + coeff * (p.grad.data - g_0)
            del self.state[p]["g_0"]

        rf_scale = 1.0 / max(s_0, self.perturb_eps)
        denom = g0_norm * diff_norm + self.perturb_eps
        rf_norm = diff_norm * rf_scale
        alpha_eff = coeff * s_0
        shaping_component_norm = abs(coeff) * diff_norm
        self.last_stats = {
            "fs_g0_norm": g0_norm,
            "fs_delta_norm": diff_norm,
            "fs_rf_norm": rf_norm,
            "fs_cos_g0_rf": g0_dot_diff / denom,
            "fs_s0": s_0,
            # alpha_eff is the weight actually applied to r_F this step
            "fs_alpha_eff": alpha_eff,
            # q_t is the realized relative norm of the shaping component.
            # In ratio mode q_t=alpha (up to eps); in fixed mode it exposes
            # the batch-dependent curvature amplification alpha*||r_F||/||g0||.
            "fs_shaping_component_norm": shaping_component_norm,
            "fs_shaping_ratio": shaping_component_norm / (g0_norm + self.perturb_eps),
            "fs_curvature_ratio": rf_norm / (g0_norm + self.perturb_eps),
            "fs_coeff_on_delta": coeff,
            "fs_ce_backward_count": 2.0,
        }

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # Mirror GAM's helper: build a forward-backward closure once.
        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def step(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            rng_before_clean = None
            rng_after_clean = None
            if self.preserve_probe_rng:
                rng_before_clean = self._capture_torch_rng_state()

            # Pass 1: g_0 at the clean point theta.
            outputs, loss_value = get_grad()
            if self.preserve_probe_rng:
                rng_after_clean = self._capture_torch_rng_state()

            # Probing point theta_+ = theta + s_0 g_0 (SAM ascent step).
            grad_norm, s_0 = self.perturb_weights()

            # Pass 2: g_1 at theta_+, same minibatch, frozen BN stats.
            disable_running_stats(self.model)
            if rng_before_clean is not None:
                self._restore_torch_rng_state(rng_before_clean)
            try:
                get_grad()
            finally:
                if rng_after_clean is not None:
                    self._restore_torch_rng_state(rng_after_clean)

            # Back to theta, then d = g_0 + alpha * (g_1 - g_0)/s_0.
            self.unperturb()
            self.compose_direction(s_0)

            # External direction injection (e.g. analytic EWC gradient).
            # Called at the clean weights, unlike GAM which projects while
            # still perturbed.
            if callable(self._direction_projector):
                self._direction_projector(self.param_groups)

        # synchronize gradients across workers
        self._sync_grad()

        # update with the shaped direction
        self.base_optimizer.step()

        enable_running_stats(self.model)

        return outputs, loss_value

    @staticmethod
    def _capture_torch_rng_state():
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_torch_rng_state(state):
        torch.set_rng_state(state["cpu"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return (f'FisherShape({self.base_optimizer.__class__.__name__}, '
                f'rho={self.rho}, alpha={self.alpha}, '
                f'sam_equivalent={self.sam_equivalent}, '
                f'preserve_probe_rng={self.preserve_probe_rng})')
