"""Current-G2: a descent predictor followed by a response-directed gradient.

    g0 = grad CE(theta)
    theta_inner = theta - q * g0 / ||g0||
    g1 = grad CE(theta_inner)
    a = g0 - g1
    theta_outer = theta + R * a / ||a||
    g2 = grad CE(theta_outer)
    restore theta; submit g2 to the base optimizer exactly once

In ``sgd_step`` mode q/R = eta * ||g0||. The inner predictor uses the
literal ``theta.add_(g0, alpha=-eta)`` operation in that mode. Fixed radii
are independent global parameter-space L2 distances. The AdamW adapter is
explicitly a predictive-SGD probe followed by an AdamW commit; it does not
claim to predict the next AdamW parameter position.

The Trainer supplies the same realized effective batch and per-microbatch
RNG for both CE replays. No Fisher, EWC, history, Hessian, orthogonalization,
or decomposition enters these probes. The pure Current-G2 implementation
has no additive regularization or direction-projection hook.
"""
from __future__ import annotations

import copy
import math
import random
from contextlib import nullcontext
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


def _rng_get():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            (int(torch.cuda.current_device()),
             torch.cuda.get_rng_state(torch.cuda.current_device()).clone())
            if torch.cuda.is_initialized() else None)


def _rng_set(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3][1], device=state[3][0])


def _clone(value):
    if isinstance(value, nn.Parameter):
        return value
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    return copy.deepcopy(value)


def _controlled(name):
    return ('cache' in name.lower() or name.startswith('delta_w_')
            or name.startswith('external_delta_w_') or name in {
                'momentum', '_register_delta_hook', 'backup_momentum',
                'current_task', 'task_id', 'task_idx', 'task_name',
                'active_adapter', 'active_adapters', 'disable_adapters',
                'merged', 'merged_adapters', 'fused_attn'})


class _Transaction:
    """Save trainables and forward state, without cloning a frozen Llama.

    Frozen parameters are immutable by contract and protected by version
    counters. A frozen mutation is rejected; arbitrary malicious writes to
    their .data cannot be rolled back without copying the entire backbone.
    """
    def __init__(self, model):
        self.model = model
        self.parameters = tuple(model.parameters())
        self.trainable = [(p, p.detach().clone()) for p in self.parameters if p.requires_grad]
        self.frozen = [(p, p._version) for p in self.parameters if not p.requires_grad]
        self.grads = [(p, None if p.grad is None else p.grad.detach().clone())
                      for p in self.parameters]
        self.requires_grad = [(p, p.requires_grad) for p in self.parameters]
        self.modules = []
        for module in model.modules():
            buffers = dict(module._buffers)
            self.modules.append((module, module.training, buffers,
                                 {k: None if v is None else v.detach().clone()
                                  for k, v in buffers.items()},
                                 set(module._non_persistent_buffers_set),
                                 {k: _clone(v) for k, v in module.__dict__.items()
                                  if _controlled(k)}))
        self.rng = _rng_get()

    @torch.no_grad()
    def restore(self):
        for param, value in self.trainable:
            param.copy_(value)
        for param, flag in self.requires_grad:
            param.requires_grad_(flag)
        for param, grad in self.grads:
            param.grad = None if grad is None else grad.clone()
        for module, training, buffers, values, nonpersistent, controls in self.modules:
            module.training = training
            module._buffers.clear()
            module._buffers.update(buffers)
            module._non_persistent_buffers_set = set(nonpersistent)
            for name, buffer in buffers.items():
                if buffer is not None:
                    buffer.copy_(values[name])
            for name in tuple(module.__dict__):
                if _controlled(name) and name not in controls:
                    delattr(module, name)
            for name, value in controls.items():
                setattr(module, name, _clone(value))
        _rng_set(self.rng)
        if any(p._version != version for p, version in self.frozen):
            raise AssertionError('A Current-G2 CE probe mutated a frozen parameter')


def _norm(values):
    # FP64 accumulation, including each tensor's sum. Gradients remain FP32.
    terms = [v.detach().double().pow(2).sum() for v in values]
    if not terms:
        return 0.0
    value = float(torch.stack(terms).sum().item())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite Current-G2 global gradient norm')
    return math.sqrt(max(0., value))


def _finite(values, label):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(label + ' contains nonfinite values')


def _ratio(numerator, denominator):
    """A zero denominator is undefined, including 0/0; never emit NaN/Inf."""
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise FloatingPointError('Nonfinite Current-G2 diagnostic input')
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite Current-G2 diagnostic ratio')
    return value


def _cosine(left, right, left_norm, right_norm):
    """Cosine of complete trainable vectors, with FP64 scalar accumulation."""
    if left_norm == 0 or right_norm == 0:
        return None
    dot = float(torch.stack([
        (a.detach().double() * b.detach().double()).sum()
        for a, b in zip(left, right)
    ]).sum().item())
    if not math.isfinite(dot):
        raise FloatingPointError('Nonfinite Current-G2 diagnostic dot product')
    # Divide successively, rather than overflow the product of two norms.
    cosine = (dot / left_norm) / right_norm
    if not math.isfinite(cosine):
        raise FloatingPointError('Nonfinite Current-G2 diagnostic cosine')
    return max(-1., min(1., cosine))


@torch.no_grad()
def _bf16_weight_cast_proxy(base, point):
    """Hypothetical weight-cast visibility, NOT a measurement of the forward.

    The actual stored parameter dtype is retained. Only temporary copies
    are cast to BF16. Autocast kernels, activation casts, LoRA composition
    and stochastic/model state can all change actual forward visibility.
    No optimizer branch or fallback decision uses this proxy.
    """
    total = sum(value.numel() for value in base)
    intended = tuple(value.detach().double() - old.detach().double()
                     for old, value in zip(base, point))
    intended_norm = _norm(intended)
    casts = [(old.detach().to(torch.bfloat16), value.detach().to(torch.bfloat16))
             for old, value in zip(base, point)]
    result = {
        'status': 'finite_cast',
        'source_parameter_dtypes': sorted({str(value.dtype) for value in base}),
        'cast_dtype': 'torch.bfloat16',
        'total_coordinates': total,
        'intended_displacement_norm': intended_norm,
    }
    if not all(bool(torch.isfinite(old).all() & torch.isfinite(value).all())
               for old, value in casts):
        # A diagnostic cast must not change whether the actual optimizer
        # step is accepted. Flag overflow, and leave undefined metrics null.
        result.update(status='nonfinite_bf16_cast', changed_coordinates=None,
                      changed_fraction=None, represented_displacement_norm=None,
                      represented_over_intended_norm=None,
                      cosine_represented_vs_intended=None)
        return result
    changed = int(torch.stack([(old != value).sum() for old, value in casts]).sum().item())
    represented = tuple(value.double() - old.double() for old, value in casts)
    represented_norm = _norm(represented)
    result.update(
        changed_coordinates=changed,
        changed_fraction=_ratio(float(changed), float(total)),
        represented_displacement_norm=represented_norm,
        represented_over_intended_norm=_ratio(represented_norm, intended_norm),
        cosine_represented_vs_intended=_cosine(
            represented, intended, represented_norm, intended_norm),
    )
    return result


class CurrentG2(torch.optim.Optimizer):
    """Three-CE-gradient optimizer adapter with a single base-optimizer commit.

    ``first_grad_ready=True`` means g0 is already accumulated and globally
    mean reduced by the native Trainer. ``closure_gradients_reduced=True``
    means each replay closure also owns exactly one global mean reduction;
    CurrentG2 never all-reduces the final direction again. Otherwise it runs
    replay closures in ``no_sync`` and performs each phase's mean itself.
    Trainables must use actual FP32 coordinates in production; BF16 forward
    autocast is controlled by the Trainer, not by this optimizer.
    """
    FORMAT = 'CurrentG2_v1'

    def __init__(self, params: Iterable, base_optimizer: torch.optim.Optimizer,
                 model: nn.Module, inner_radius_mode: str = 'sgd_step',
                 outer_radius_mode: str = 'sgd_step',
                 inner_radius: Optional[float] = None,
                 outer_radius: Optional[float] = None,
                 eps: float = 1e-12, grad_reduce: str = 'mean',
                 closure_gradients_reduced: bool = True,
                 require_fp32_trainable: bool = True,
                 quantization_diagnostics_interval: int = 0, **kwargs):
        if kwargs:
            raise TypeError('Unknown CurrentG2 options: ' + ', '.join(sorted(kwargs)))
        if str(grad_reduce).lower() != 'mean':
            raise ValueError('Current-G2 requires global mean gradients')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('eps must be finite and positive')
        if (not isinstance(quantization_diagnostics_interval, int)
                or isinstance(quantization_diagnostics_interval, bool)
                or quantization_diagnostics_interval < 0):
            raise ValueError('quantization_diagnostics_interval must be a nonnegative integer')
        modes = (str(inner_radius_mode).lower(), str(outer_radius_mode).lower())
        radii = (inner_radius, outer_radius)
        for label, mode, radius in zip(('inner', 'outer'), modes, radii):
            if mode not in {'sgd_step', 'fixed_l2'}:
                raise ValueError(label + '_radius_mode must be sgd_step or fixed_l2')
            if mode == 'sgd_step' and radius is not None:
                raise ValueError(label + '_radius must be None for sgd_step')
            if mode == 'fixed_l2' and (radius is None or not math.isfinite(float(radius))
                                      or float(radius) < 0):
                raise ValueError(label + '_radius must be finite and nonnegative for fixed_l2')
        supplied = list(params)
        if not supplied:
            raise ValueError('Current-G2 needs nonempty trainable parameters')
        super().__init__(supplied, {})
        supplied_ids = {id(p) for group in self.param_groups for p in group['params']}
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.model = model
        self.params = [p for group in self.param_groups for p in group['params']]
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Duplicate optimizer parameters')
        if {id(p) for p in self.params} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Current-G2 scope must cover all and only trainable model parameters')
        if {id(p) for p in self.params} != supplied_ids:
            raise ValueError('Supplied parameters and base optimizer scope must match')
        if require_fp32_trainable and any(p.dtype != torch.float32 for p in self.params):
            raise ValueError('Current-G2 requires actual FP32 trainable parameters')
        if any(not p.is_floating_point() for p in self.params):
            raise ValueError('Current-G2 trainables must be floating point')
        is_hf_adamw = (type(base_optimizer).__name__ == 'AdamW'
                       and type(base_optimizer).__module__ == 'transformers.optimization')
        if not isinstance(base_optimizer, (torch.optim.SGD, torch.optim.AdamW)) and not is_hf_adamw:
            raise ValueError('Current-G2 supports torch SGD, torch AdamW or transformers AdamW commits')
        if isinstance(base_optimizer, torch.optim.SGD) and any(
                float(group.get('momentum', 0)) != 0 or float(group.get('dampening', 0)) != 0
                or float(group.get('weight_decay', 0)) != 0 or bool(group.get('nesterov', False))
                or bool(group.get('maximize', False)) or bool(group.get('differentiable', False))
                or bool(group.get('foreach', False)) or bool(group.get('fused', False))
                for group in self.param_groups):
            raise ValueError('The faithful Current-G2 SGD commit requires plain momentum=weight_decay=0 SGD')
        self.inner_radius_mode, self.outer_radius_mode = modes
        self.inner_radius = None if inner_radius is None else float(inner_radius)
        self.outer_radius = None if outer_radius is None else float(outer_radius)
        self.eps = float(eps)
        self.closure_gradients_reduced = bool(closure_gradients_reduced)
        self.require_fp32_trainable = bool(require_fp32_trainable)
        self.quantization_diagnostics_interval = quantization_diagnostics_interval
        self._busy = False
        self.real_updates = 0
        self.last_stats: Dict = {}
        self._layout = self._current_layout()

    def _current_layout(self):
        return tuple((id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad) for p in self.params)

    def maybe_no_sync(self):
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and hasattr(self.model, 'no_sync')):
            return self.model.no_sync()
        return nullcontext()

    @torch.no_grad()
    def _sync_grad(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 0
        world = float(torch.distributed.get_world_size())
        for param in self.params:
            # All ranks participate even when a parameter is locally unused.
            present = torch.tensor(int(param.grad is not None), device=param.device, dtype=torch.int64)
            torch.distributed.all_reduce(present, op=torch.distributed.ReduceOp.SUM)
            if int(present.item()) == 0:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            param.grad.div_(world)
        return 1

    def _grads(self):
        masks = tuple(p.grad is not None for p in self.params)
        values = tuple(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                       for p in self.params)
        _finite(values, 'Current-G2 CE gradient')
        return values, masks

    def _lr(self):
        rates = [float(group['lr']) for group in self.param_groups]
        if any(not math.isfinite(v) or v < 0 for v in rates) or len(set(rates)) != 1:
            raise ValueError('Current-G2 requires one finite nonnegative LR across parameter groups')
        return rates[0]

    def step(self, closure=None, clip_grad_norm=None, first_grad_ready: bool = False):
        if closure is None:
            raise ValueError('Current-G2 requires a task-only forward-backward closure')
        if self._busy:
            raise RuntimeError('Nested Current-G2 step is not allowed')
        if self._current_layout() != self._layout or {id(p) for p in self.params} != {
                id(p) for p in self.model.parameters() if p.requires_grad}:
            raise ValueError('Trainable layout changed: reconstruct Current-G2 for the new task')
        if clip_grad_norm is not None and (not math.isfinite(float(clip_grad_norm))
                                          or float(clip_grad_norm) < 0):
            raise ValueError('clip_grad_norm must be finite and nonnegative')
        eta = self._lr()
        entry = _Transaction(self.model)
        base = tuple(v for _, v in entry.trainable)
        # Entry trainable order can differ from optimizer groups.
        base_by_id = {id(p): v for p, v in entry.trainable}
        base = tuple(base_by_id[id(p)] for p in self.params)
        calls = 1 if first_grad_ready else 0
        core_reductions = 0
        outputs = loss_value = None
        rng_after_base = entry.rng
        self._busy = True

        def run_probe(point):
            nonlocal calls, core_reductions
            entry.restore()
            with torch.no_grad():
                for param, value in zip(self.params, point):
                    param.copy_(value)
            self.zero_grad(set_to_none=True)
            context = nullcontext() if self.closure_gradients_reduced else self.maybe_no_sync()
            with context, torch.enable_grad():
                result = closure()
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError('Current-G2 closure must return (outputs, scalar loss)')
            probe_outputs, probe_loss = result
            if torch.is_tensor(probe_loss) and (probe_loss.numel() != 1
                                                or not bool(torch.isfinite(probe_loss))):
                raise FloatingPointError('Current-G2 closure loss must be finite scalar')
            if not torch.is_tensor(probe_loss):
                if not isinstance(probe_loss, (float, int)) or isinstance(probe_loss, bool):
                    raise ValueError('Current-G2 closure must return a finite scalar loss')
                if not math.isfinite(float(probe_loss)):
                    raise FloatingPointError('Current-G2 closure loss must be finite scalar')
            if not self.closure_gradients_reduced:
                core_reductions += self._sync_grad()
            gradients, mask = self._grads()
            if not any(mask):
                raise RuntimeError('Current-G2 closure produced no parameter gradients')
            if not all(torch.equal(p, v) for p, v in zip(self.params, point)):
                raise AssertionError('Current-G2 CE closure mutated trainable parameters')
            if self._current_layout() != self._layout:
                raise AssertionError('Current-G2 CE closure mutated trainable flags or layout')
            calls += 1
            return probe_outputs, probe_loss, gradients, mask, _rng_get()

        fallback = None
        g1 = g2 = None
        inner_actual = outer_actual = 0.
        inner_distinguishable = outer_distinguishable = None
        q = R = h = 0.
        response_norm = 0.
        quantization_sampled = bool(self.quantization_diagnostics_interval
                                   and self.real_updates % self.quantization_diagnostics_interval == 0)
        quantization_proxy = {
            'schema': 'CurrentG2BF16WeightCastProxy_v1',
            'kind': 'parameter_cast_to_bfloat16',
            'scope': 'all_trainable_parameters',
            'proves_forward_visibility': False,
            'sampling_due': quantization_sampled,
            'sampled': False,
            'interval': self.quantization_diagnostics_interval,
            'inner': None, 'outer': None,
        }
        restored = False
        try:
            if first_grad_ready:
                g0, mask0 = self._grads()
                if not any(mask0):
                    raise RuntimeError('first_grad_ready=True requires clean-point gradients')
            else:
                outputs, loss_value, g0, mask0, rng_after_base = run_probe(base)
            g0_norm = _norm(g0)
            h = eta * g0_norm
            q = h if self.inner_radius_mode == 'sgd_step' else self.inner_radius
            R = h if self.outer_radius_mode == 'sgd_step' else self.outer_radius
            if not math.isfinite(h):
                raise FloatingPointError('Nonfinite predicted-SGD length')
            final, final_mask = g0, mask0
            if g0_norm <= self.eps:
                fallback = 'base_gradient_at_or_below_eps'
            elif q <= self.eps:
                fallback = 'inner_radius_at_or_below_eps'
            else:
                if self.inner_radius_mode == 'sgd_step':
                    inner = tuple(v.clone().add_(g, alpha=-eta) for v, g in zip(base, g0))
                else:
                    inner = tuple(v.clone().add_(g, alpha=-q / g0_norm) for v, g in zip(base, g0))
                _finite(inner, 'Current-G2 inner point')
                inner_distinguishable = any(not torch.equal(v, x) for v, x in zip(base, inner))
                inner_actual = _norm(tuple(x - v for x, v in zip(inner, base)))
                if quantization_sampled:
                    quantization_proxy['inner'] = _bf16_weight_cast_proxy(base, inner)
                    quantization_proxy['sampled'] = True
                if not inner_distinguishable or inner_actual <= self.eps:
                    fallback = 'inner_point_bitwise_indistinguishable_or_actual_length_at_or_below_eps'
                else:
                    outputs, loss_value, g1, _, _ = run_probe(inner)
                    response = tuple(a - b for a, b in zip(g0, g1))
                    response_norm = _norm(response)
                    if response_norm <= self.eps:
                        fallback = 'response_at_or_below_eps'
                    elif R <= self.eps:
                        fallback = 'outer_radius_at_or_below_eps'
                    else:
                        outer = tuple(v.clone().add_(a, alpha=R / response_norm)
                                      for v, a in zip(base, response))
                        _finite(outer, 'Current-G2 outer point')
                        outer_distinguishable = any(not torch.equal(v, x) for v, x in zip(base, outer))
                        outer_actual = _norm(tuple(x - v for x, v in zip(outer, base)))
                        if quantization_sampled:
                            quantization_proxy['outer'] = _bf16_weight_cast_proxy(base, outer)
                        if not outer_distinguishable or outer_actual <= self.eps:
                            fallback = 'outer_point_bitwise_indistinguishable_or_actual_length_at_or_below_eps'
                        else:
                            outputs, loss_value, g2, final_mask, _ = run_probe(outer)
                            final = g2
            # These compare globally mean-reduced, pre-clipping task gradients.
            # Missing parameter gradients are zero-filled by _grads(), so no
            # coordinate is silently excluded from the comparison.
            g1_norm = _norm(g1) if g1 is not None else None
            g2_norm = _norm(g2) if g2 is not None else None
            g2_delta_norm = (_norm(tuple(b - a for a, b in zip(g0, g2)))
                             if g2 is not None else None)
            cos_g0_g1 = _cosine(g0, g1, g0_norm, g1_norm) if g1 is not None else None
            cos_g0_g2 = _cosine(g0, g2, g0_norm, g2_norm) if g2 is not None else None
            entry.restore()
            restored = True
            _rng_set(rng_after_base)
            with torch.no_grad():
                for param, value, present in zip(self.params, final, final_mask):
                    param.grad = value.clone() if present else None
            submitted, _ = self._grads()
            submitted_norm = _norm(submitted)
            if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(self.params, float(clip_grad_norm), error_if_nonfinite=True)
            clipped, _ = self._grads()
            final_norm = _norm(clipped)
            # No optimizer step, momentum update, scheduler call, or weight decay
            # occurs before this single commit at the restored clean point.
            self.base_optimizer.step()
            _finite(self.params, 'Committed Current-G2 parameters')
            self.real_updates += 1
            actual_update_norm = _norm(tuple(p.detach() - v for p, v in zip(self.params, base)))
            stats = {
                'g2_method': 'current_g2', 'g2_final_direction': 'g2' if g2 is not None else 'g0_fallback',
                'g2_commit_optimizer': type(self.base_optimizer).__name__,
                'g2_commit_optimizer_module': type(self.base_optimizer).__module__,
                'g2_update_is_sgd': isinstance(self.base_optimizer, torch.optim.SGD),
                'g2_commit_semantics': ('ordinary_sgd' if isinstance(self.base_optimizer, torch.optim.SGD)
                                        else 'predictive_sgd_probe_then_adamw_commit'),
                'g2_inner_radius_mode': self.inner_radius_mode,
                'g2_outer_radius_mode': self.outer_radius_mode,
                'g2_inner_radius': self.inner_radius, 'g2_outer_radius': self.outer_radius,
                'g2_lr': eta, 'g2_h_ref': h,
                'g2_inner_radius_theoretical': q, 'g2_outer_radius_theoretical': R,
                'g2_inner_radius_actual': inner_actual, 'g2_outer_radius_actual': outer_actual,
                'g2_inner_distinguishable': inner_distinguishable,
                'g2_outer_distinguishable': outer_distinguishable,
                'g2_g0_norm': g0_norm, 'g2_g1_norm': g1_norm,
                'g2_g2_norm': g2_norm,
                'g2_cos_g0_g1': cos_g0_g1, 'g2_cos_g0_g2': cos_g0_g2,
                'g2_g2_minus_g0_norm': g2_delta_norm,
                'g2_g2_minus_g0_relative_norm': (
                    _ratio(g2_delta_norm, g0_norm) if g2_delta_norm is not None else None),
                'g2_response_relative_norm': (
                    _ratio(response_norm, g0_norm) if g1 is not None else None),
                'g2_response_norm': response_norm, 'g2_submitted_preclip_norm': submitted_norm,
                'g2_submitted_postclip_norm': final_norm,
                'g2_actual_update_norm': actual_update_norm, 'g2_ce_gradient_phases': calls,
                'g2_actual_update_over_h_ref': _ratio(actual_update_norm, h),
                'g2_actual_update_over_inner_radius': _ratio(actual_update_norm, inner_actual),
                'g2_actual_update_over_outer_radius': _ratio(actual_update_norm, outer_actual),
                'g2_gradient_clip_scale': _ratio(final_norm, submitted_norm),
                'g2_gradient_metrics_scope': 'all_trainable_global_mean_preclip_gradients',
                'g2_gradient_diagnostics_schema': 'CurrentG2GradientDiagnostics_v1',
                'g2_quantization_proxy': quantization_proxy,
                'g2_replay_closure_calls': calls - 1,
                'g2_core_explicit_mean_reductions': core_reductions,
                'g2_gradient_reduction_owner': 'trainer' if self.closure_gradients_reduced else 'core',
                'g2_base_optimizer_steps': 1, 'g2_restore_base_exact': restored,
                'g2_regularization_hook_used': False,
                'g2_final_clip_enabled': bool(clip_grad_norm and float(clip_grad_norm) > 0),
                'g2_history_used': False, 'g2_fisher_probe_calls': 0,
                'g2_projection_calls': 0, 'g2_hvp_calls': 0,
                'g2_relocation_origin': 'base' if g2 is not None else None,
                'g2_fallback_reason': fallback,
                'g2_real_updates': self.real_updates,
            }
            self.last_stats = stats
        except BaseException:
            # Probe failures restore pre-step weights, flags, gradients,
            # buffers/caches and RNG. A base optimizer raising mid-commit is not
            # recoverable without cloning its full AdamW moment state.
            entry.restore()
            raise
        finally:
            self._busy = False
        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _config(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return dict(inner_radius_mode=self.inner_radius_mode, outer_radius_mode=self.outer_radius_mode,
                    inner_radius=self.inner_radius, outer_radius=self.outer_radius, eps=self.eps,
                    require_fp32_trainable=self.require_fp32_trainable,
                    commit_optimizer=type(self.base_optimizer).__name__,
                    commit_optimizer_module=type(self.base_optimizer).__module__, history=False,
                    closure_gradients_reduced=self.closure_gradients_reduced,
                    portable_layout=[dict(name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype))
                                     for p in self.params])

    def state_dict(self):
        if self._busy:
            raise RuntimeError('Cannot checkpoint a pending Current-G2 transaction')
        state = self.base_optimizer.state_dict()
        state['_current_g2'] = dict(format=self.FORMAT, config=self._config(),
                                    real_updates=self.real_updates)
        return state

    def load_state_dict(self, state_dict):
        if self._busy:
            raise RuntimeError('Cannot load a pending Current-G2 transaction')
        metadata = state_dict.get('_current_g2')
        if not isinstance(metadata, dict) or metadata.get('format') != self.FORMAT:
            raise ValueError('Current-G2 checkpoint metadata is required')
        if metadata.get('config') != self._config():
            raise ValueError('Current-G2 checkpoint algorithm/config/layout mismatch')
        updates = metadata.get('real_updates')
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
            raise ValueError('Invalid Current-G2 checkpoint update counter')
        self.base_optimizer.load_state_dict({k: v for k, v in state_dict.items() if k != '_current_g2'})
        self.param_groups = self.base_optimizer.param_groups
        self.real_updates = updates

    def __repr__(self):
        return (f'CurrentG2({type(self.base_optimizer).__name__}, '
                f'inner={self.inner_radius_mode}:{self.inner_radius}, '
                f'outer={self.outer_radius_mode}:{self.outer_radius})')
