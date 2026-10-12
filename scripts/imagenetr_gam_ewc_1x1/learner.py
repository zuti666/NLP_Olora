"""Continuous Q/V LoRA: native SGD/Adam EWC versus four-pass GAM + EWC.

The caller puts its CopyFlatCV source directory on sys.path before import.
Both methods use exactly the same effective-weight EWC implementation.
"""
from types import SimpleNamespace

import torch

from models_LoRAbasedCL.as1_normfisher_gam import Learner as BaseLearner
from optimer_PerturabtionType.gam import GAM
from optimer_PerturabtionType.util import enable_running_stats


def capture_rng():
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def additive_step(network, optimizer, inputs, targets, class_offset,
                  penalty_fn, task_loss_fn):
    """One native update or four CE GAM probes plus CLEAN-point EWC gradient.

    No EWC term enters any probe. GAM's own decomposition is preserved;
    its callback only adds the cached EWC gradient, without EWC projection.
    Every probe replays the first probe's RNG; afterward only one clean-pass
    RNG advance remains. This keeps stochastic layers comparable to EWC.
    """
    params = [p for group in optimizer.param_groups for p in group['params']
              if p.requires_grad]
    with torch.enable_grad():
        penalty = penalty_fn()
    ewc_grads = (torch.autograd.grad(penalty, params, allow_unused=True)
                 if penalty.requires_grad else [None] * len(params))
    ewc_cache = {id(p): g.detach().clone() for p, g in zip(params, ewc_grads)
                 if g is not None}
    penalty_value = float(penalty.detach())
    before = capture_rng()
    post_clean = None
    clean_outputs = clean_loss = None
    calls = 0

    def closure():
        nonlocal post_clean, clean_outputs, clean_loss, calls
        restore_rng(before)
        optimizer.zero_grad()
        with torch.enable_grad():
            outputs = network(inputs)
            loss, _, _ = task_loss_fn(outputs['logits'], targets, class_offset)
        loss.backward()
        calls += 1
        if calls == 1:
            clean_outputs, clean_loss = outputs['logits'].detach(), float(loss.detach())
            post_clean = capture_rng()
        return outputs, loss.detach()

    def add_protection(groups):
        with torch.no_grad():
            for group in groups:
                for p in group['params']:
                    g = ewc_cache.get(id(p))
                    if g is not None:
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.add_(g)

    is_gam = isinstance(optimizer, GAM)
    originals = [p.detach().clone() for p in params] if is_gam else None
    try:
        if is_gam:
            optimizer.set_direction_projector(add_protection)
            optimizer.step(closure=closure)
        else:
            closure()
            add_protection(optimizer.param_groups)
            optimizer.step()
    except BaseException:
        # GAM's upstream step has no exceptional unperturb: never leave a
        # probe point behind when a closure/OOM fails.
        if originals is not None:
            with torch.no_grad():
                for p, value in zip(params, originals):
                    p.copy_(value)
        raise
    finally:
        if is_gam:
            optimizer.set_direction_projector(None)
        enable_running_stats(network)
        restore_rng(post_clean if post_clean is not None else before)
    expected_calls = 4 if is_gam else 1
    if calls != expected_calls:
        raise RuntimeError(f'Expected {expected_calls} CE calls, observed {calls}')
    return clean_outputs, clean_loss + penalty_value, {
        'ce_backward_count': calls, 'clean_ce': clean_loss,
        'ewc_penalty': penalty_value, 'ewc_gradient_at_clean_point': True,
    }


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # The upstream checkpoint identity reads this attribute, not args.
        self._ewc_fisher_damping_beta = float(args['ewc_fisher_damping_beta'])

    def _build_optimizer(self, params, stage):
        params = list(params)
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base_name = self.args['comparison_base_optimizer']
        if base_name == 'sgd':
            base = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)
        elif base_name == 'adam':
            base = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(base_name)
        if self.args['comparison_method'] == 'ewc':
            self._last_optimizer_class = type(base).__name__
            return base
        args = SimpleNamespace(
            grad_beta_1=float(self.args['gam_grad_beta_1']),
            grad_beta_2=float(self.args['gam_grad_beta_2']),
            grad_beta_3=float(self.args['gam_grad_beta_3']),
            grad_gamma=float(self.args['gam_grad_gamma']), use_sgv=False)
        self._last_optimizer_class = 'GAM'
        return GAM(params, base_optimizer=base, model=self._network,
                   grad_rho=float(self.args['gam_grad_rho']),
                   grad_norm_rho=float(self.args['gam_grad_norm_rho']),
                   adaptive=False, args=args)

    def _compute_delta_fisher(self, loader, model, class_offset):
        # Same trace normalization + isotropic damping as reference R protocol.
        fisher = super()._compute_delta_fisher(loader, model, class_offset)
        beta = float(self.args['ewc_fisher_damping_beta'])
        count = sum(v.numel() for v in fisher.values())
        if beta and count:
            fisher = {k: (1.0 - beta) * v + beta / count for k, v in fisher.items()}
        return fisher

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset):
        logits, loss, stats = additive_step(
            self._network, optimizer, inputs, targets, class_offset,
            lambda: self._past_fisher_penalty(self._unwrap_network()), self._task_loss)
        self._record_mechanism_loss_stats(stats['clean_ce'], stats['ewc_penalty'])
        self._accumulate_mechanism_grad_stats({'ce_backward_count': stats['ce_backward_count']})
        return logits, loss
