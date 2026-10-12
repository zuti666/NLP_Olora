"""Clean two-point gradient probes followed by one ordinary optimizer commit.

``reverse`` evaluates g_minus at theta-rho*g0/(||g0||+eps), restores
theta, and submits 2*g0-g_minus. ``sam`` evaluates the corresponding
ascent point and submits its gradient. ``sgd`` submits g0 without a probe.
The mode names describe gradients, not the commit backend: in the Llama
experiments all three modes use ordinary Transformers AdamW. No virtual
AdamW state update, outer probe, G2/G3, Fisher, history or projection is used.

The native Trainer owns realized effective batches and per-microbatch RNG
replay. Standalone closures instead have the entire BASE RNG replayed.
Probe construction uses pre-clipping global-mean gradients; only the final
submitted gradient is clipped. The denominator is literally norm+eps.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Dict, Iterable

import torch
import torch.nn as nn

from optimer_g2 import (
    _Transaction, _rng_get, _rng_set, _norm, _finite, _ratio, _cosine,
    _bf16_weight_cast_proxy,
)


class TwoPointProbe(torch.optim.Optimizer):
    """Two-point reverse/SAM gradient adapter, or one-point clean control.

    ``first_grad_ready`` declares that BASE gradients are already globally
    mean reduced. ``closure_gradients_reduced`` declares that each replay
    closure owns that same mean reduction. Otherwise this adapter handles
    replay reductions under model.no_sync(). Never reduce the submitted
    direction again. Actual FP32 trainable master coordinates are required
    by default; autocast precision remains the Trainer's responsibility.

    Frozen parameters are immutable and checked using the shared transaction
    version guard. As in CurrentG2, malicious mutation of a frozen backbone
    cannot be rolled back without copying that entire backbone. Supported
    forward buffers, caches, flags, trainables, gradients, RNG and optimizer
    moments are restored on failure, including a failed real commit.
    """
    FORMAT = 'TwoPointProbe_v1'

    def __init__(self, params: Iterable, base_optimizer: torch.optim.Optimizer,
                 model: nn.Module, mode: str = 'reverse', radius: float = .05,
                 eps: float = 1e-12, grad_reduce: str = 'mean',
                 closure_gradients_reduced: bool = True,
                 require_fp32_trainable: bool = True,
                 quantization_diagnostics_interval: int = 0, **kwargs):
        if kwargs:
            raise TypeError('Unknown TwoPointProbe options: ' + ', '.join(sorted(kwargs)))
        mode = str(mode).lower()
        if mode not in {'reverse', 'sam', 'sgd'}:
            raise ValueError('mode must be reverse, sam or sgd')
        if not math.isfinite(float(radius)) or float(radius) < 0:
            raise ValueError('radius must be finite and nonnegative')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('eps must be finite and positive')
        if str(grad_reduce).lower() != 'mean':
            raise ValueError('TwoPointProbe requires global mean gradients')
        if (not isinstance(quantization_diagnostics_interval, int)
                or isinstance(quantization_diagnostics_interval, bool)
                or quantization_diagnostics_interval < 0):
            raise ValueError('quantization_diagnostics_interval must be a nonnegative integer')
        supplied = list(params)
        if not supplied:
            raise ValueError('TwoPointProbe requires nonempty trainable parameters')
        super().__init__(supplied, {})
        supplied_ids = {id(p) for group in self.param_groups for p in group['params']}
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.model = model
        self.params = [p for group in self.param_groups for p in group['params']]
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Duplicate optimizer parameters')
        if supplied_ids != {id(p) for p in self.params}:
            raise ValueError('Supplied and base-optimizer parameter scopes must match')
        if supplied_ids != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('TwoPointProbe must cover all and only trainable parameters')
        if require_fp32_trainable and any(p.dtype != torch.float32 for p in self.params):
            raise ValueError('TwoPointProbe requires actual FP32 trainable parameters')
        if any(not p.is_floating_point() for p in self.params):
            raise ValueError('TwoPointProbe trainables must be floating point')
        is_hf_adamw = (type(base_optimizer).__name__ == 'AdamW'
                      and type(base_optimizer).__module__ == 'transformers.optimization')
        if not isinstance(base_optimizer, (torch.optim.SGD, torch.optim.AdamW)) and not is_hf_adamw:
            raise ValueError('TwoPointProbe supports torch SGD/AdamW and Transformers AdamW')
        if isinstance(base_optimizer, torch.optim.SGD) and any(
                float(group.get('momentum', 0)) != 0 or float(group.get('dampening', 0)) != 0
                or float(group.get('weight_decay', 0)) != 0 or bool(group.get('nesterov', False))
                or bool(group.get('maximize', False)) or bool(group.get('differentiable', False))
                for group in self.param_groups):
            raise ValueError('The plain SGD commit requires momentum=weight_decay=0')
        self.mode, self.radius, self.eps = mode, float(radius), float(eps)
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
            # Participate on every rank, including locally unused parameters.
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
        _finite(values, 'TwoPointProbe CE gradient')
        return values, masks

    def _lr(self):
        rates = [float(group['lr']) for group in self.param_groups]
        if any(not math.isfinite(v) or v < 0 for v in rates) or len(set(rates)) != 1:
            raise ValueError('TwoPointProbe requires one finite nonnegative LR')
        return rates[0]

    def step(self, closure=None, clip_grad_norm=None, first_grad_ready: bool = False):
        if closure is None:
            raise ValueError('TwoPointProbe requires a task-only forward-backward closure')
        if self._busy:
            raise RuntimeError('Nested TwoPointProbe step is not allowed')
        if self._current_layout() != self._layout or {id(p) for p in self.params} != {
                id(p) for p in self.model.parameters() if p.requires_grad}:
            raise ValueError('Trainable layout changed: reconstruct TwoPointProbe')
        if clip_grad_norm is not None and (not math.isfinite(float(clip_grad_norm))
                                          or float(clip_grad_norm) < 0):
            raise ValueError('clip_grad_norm must be finite and nonnegative')
        eta = self._lr()
        entry = _Transaction(self.model)
        base_by_id = {id(p): value for p, value in entry.trainable}
        base = tuple(base_by_id[id(p)] for p in self.params)
        calls = 1 if first_grad_ready else 0
        reductions = 0
        outputs = loss_value = None
        rng_after_base = entry.rng
        optimizer_snapshot = None
        previous_stats = self.last_stats
        previous_updates = self.real_updates
        self._busy = True

        def evaluate(point):
            nonlocal calls, reductions
            entry.restore()
            with torch.no_grad():
                for param, value in zip(self.params, point):
                    param.copy_(value)
            self.zero_grad(set_to_none=True)
            context = nullcontext() if self.closure_gradients_reduced else self.maybe_no_sync()
            with context, torch.enable_grad():
                result = closure()
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError('TwoPointProbe closure must return (outputs, scalar loss)')
            probe_outputs, loss = result
            if torch.is_tensor(loss):
                if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError('TwoPointProbe closure loss must be finite scalar')
            elif (not isinstance(loss, (float, int)) or isinstance(loss, bool)
                  or not math.isfinite(float(loss))):
                raise FloatingPointError('TwoPointProbe closure loss must be finite scalar')
            if not self.closure_gradients_reduced:
                reductions += self._sync_grad()
            gradients, masks = self._grads()
            if not any(masks):
                raise RuntimeError('TwoPointProbe closure produced no parameter gradients')
            if not all(torch.equal(p, v) for p, v in zip(self.params, point)):
                raise AssertionError('TwoPointProbe CE closure mutated trainable parameters')
            if self._current_layout() != self._layout:
                raise AssertionError('TwoPointProbe CE closure mutated trainable layout')
            calls += 1
            return probe_outputs, loss, gradients, masks, _rng_get()

        sampled = (self.mode != 'sgd' and self.quantization_diagnostics_interval > 0
                   and self.real_updates % self.quantization_diagnostics_interval == 0)
        cast_proxy = {
            'schema': 'TwoPointBF16WeightCastProxy_v1',
            'kind': 'parameter_cast_to_bfloat16', 'scope': 'all_trainable_parameters',
            'proves_forward_visibility': False, 'sampling_due': sampled,
            'sampled': False, 'interval': self.quantization_diagnostics_interval, 'probe': None,
        }
        try:
            if first_grad_ready:
                g0, mask0 = self._grads()
                if not any(mask0):
                    raise RuntimeError('first_grad_ready=True requires clean-point gradients')
            else:
                outputs, loss_value, g0, mask0, rng_after_base = evaluate(base)
            g0_norm = _norm(g0)
            g1 = None
            actual_radius = 0.
            distinguishable = None
            final, final_mask = g0, mask0
            # A zero radius or zero gradient still performs the requested
            # second CE phase. This gives a real same-point noise diagnostic
            # and keeps the two-point execution budget explicit and fixed.
            if self.mode != 'sgd':
                signed_scale = (-1. if self.mode == 'reverse' else 1.) * self.radius / (g0_norm + self.eps)
                point = tuple(value.clone().add_(grad, alpha=signed_scale)
                              for value, grad in zip(base, g0))
                _finite(point, 'TwoPointProbe probe point')
                actual_radius = _norm(tuple(value - old for value, old in zip(point, base)))
                distinguishable = any(not torch.equal(old, value) for old, value in zip(base, point))
                if sampled:
                    cast_proxy['probe'] = _bf16_weight_cast_proxy(base, point)
                    cast_proxy['sampled'] = True
                outputs, loss_value, g1, mask1, _ = evaluate(point)
                if self.mode == 'reverse':
                    final = tuple(a.mul(2).sub(b) for a, b in zip(g0, g1))
                    # Absent in both phases remains None: AdamW must not
                    # create state/decay for completely unused coordinates.
                    final_mask = tuple(a or b for a, b in zip(mask0, mask1))
                else:
                    final, final_mask = g1, mask1
            _finite(final, 'TwoPointProbe submitted gradient')
            g1_norm = _norm(g1) if g1 is not None else None
            response_norm = _norm(tuple(a - b for a, b in zip(g0, g1))) if g1 is not None else None
            final_norm = _norm(final)
            delta_norm = _norm(tuple(b - a for a, b in zip(g0, final)))
            entry.restore()
            _rng_set(rng_after_base)
            with torch.no_grad():
                for param, grad, present in zip(self.params, final, final_mask):
                    param.grad = grad.clone() if present else None
            if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(self.params, float(clip_grad_norm), error_if_nonfinite=True)
            clipped, _ = self._grads()
            clipped_norm = _norm(clipped)
            # Snapshot LoRA optimizer moments only immediately before the
            # unique real commit. Probe failures never touch those moments.
            optimizer_snapshot = copy.deepcopy(self.base_optimizer.state_dict())
            self.base_optimizer.step()
            _finite(self.params, 'Committed TwoPointProbe parameters')
            actual_update_norm = _norm(tuple(p.detach() - old for p, old in zip(self.params, base)))
            self.real_updates += 1
            direction = {'reverse': '2g0_minus_g1_descent', 'sam': 'g1_ascent', 'sgd': 'g0'}[self.mode]
            self.last_stats = {
                'probe_schema': self.FORMAT, 'probe_method': 'two_point_probe',
                'probe_mode': self.mode, 'probe_final_direction': direction,
                'probe_sign': -1 if self.mode == 'reverse' else (1 if self.mode == 'sam' else 0),
                'probe_radius': self.radius, 'probe_rho': self.radius,
                'probe_radius_mode': 'fixed_l2_normalized_with_eps',
                'probe_radius_theoretical': (self.radius * g0_norm / (g0_norm + self.eps)
                                             if self.mode != 'sgd' else 0.),
                'probe_radius_actual': actual_radius, 'probe_distinguishable': distinguishable,
                'probe_lr': eta, 'probe_eps': self.eps, 'probe_g0_norm': g0_norm,
                'probe_g1_norm': g1_norm,
                'probe_cos_g0_g1': _cosine(g0, g1, g0_norm, g1_norm) if g1 is not None else None,
                'probe_cos_g0_submitted': _cosine(g0, final, g0_norm, final_norm),
                'probe_response_norm': response_norm,
                'probe_response_relative_norm': _ratio(response_norm, g0_norm) if g1 is not None else None,
                'probe_submitted_minus_g0_norm': delta_norm,
                'probe_submitted_minus_g0_relative_norm': _ratio(delta_norm, g0_norm),
                'probe_submitted_preclip_norm': final_norm,
                'probe_submitted_postclip_norm': clipped_norm,
                'probe_gradient_clip_scale': _ratio(clipped_norm, final_norm),
                'probe_final_clip_enabled': bool(clip_grad_norm and float(clip_grad_norm) > 0),
                'probe_actual_update_norm': actual_update_norm,
                'probe_actual_update_over_probe_radius': _ratio(actual_update_norm, actual_radius),
                'probe_ce_gradient_phases': calls, 'probe_replay_closure_calls': calls - 1,
                'probe_core_explicit_mean_reductions': reductions,
                'probe_gradient_reduction_owner': 'trainer' if self.closure_gradients_reduced else 'core',
                'probe_gradient_metrics_scope': 'all_trainable_global_mean_preclip_gradients',
                'probe_base_optimizer_steps': 1, 'probe_real_updates': self.real_updates,
                'probe_restore_base_exact': True, 'probe_fallback_reason': None,
                'probe_commit_optimizer': type(self.base_optimizer).__name__,
                'probe_commit_optimizer_module': type(self.base_optimizer).__module__,
                'probe_commit_semantics': ('ordinary_sgd' if isinstance(self.base_optimizer, torch.optim.SGD)
                                           else 'virtual_sgd_probe_then_adamw_commit'),
                'probe_outer_probe_calls': 0, 'probe_g2_calls': 0, 'probe_g3_calls': 0,
                'probe_fisher_probe_calls': 0, 'probe_projection_calls': 0,
                'probe_hvp_calls': 0, 'probe_history_used': False,
                'probe_regularization_hook_used': False, 'probe_quantization_proxy': cast_proxy,
            }
        except BaseException:
            if optimizer_snapshot is not None:
                self.base_optimizer.load_state_dict(optimizer_snapshot)
                self.param_groups = self.base_optimizer.param_groups
            self.real_updates, self.last_stats = previous_updates, previous_stats
            entry.restore()
            raise
        finally:
            self._busy = False
        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _config(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return dict(mode=self.mode, radius=self.radius, eps=self.eps,
                    require_fp32_trainable=self.require_fp32_trainable,
                    closure_gradients_reduced=self.closure_gradients_reduced,
                    commit_optimizer=type(self.base_optimizer).__name__,
                    commit_optimizer_module=type(self.base_optimizer).__module__, history=False,
                    portable_layout=[dict(name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype))
                                     for p in self.params])

    def state_dict(self):
        if self._busy:
            raise RuntimeError('Cannot checkpoint a pending TwoPointProbe transaction')
        state = self.base_optimizer.state_dict()
        state['_two_point_probe'] = dict(format=self.FORMAT, config=self._config(), real_updates=self.real_updates)
        return state

    def load_state_dict(self, state_dict):
        if self._busy:
            raise RuntimeError('Cannot load a pending TwoPointProbe transaction')
        metadata = state_dict.get('_two_point_probe')
        if not isinstance(metadata, dict) or metadata.get('format') != self.FORMAT:
            raise ValueError('TwoPointProbe checkpoint metadata is required')
        if metadata.get('config') != self._config():
            raise ValueError('TwoPointProbe checkpoint algorithm/config/layout mismatch')
        updates = metadata.get('real_updates')
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
            raise ValueError('Invalid TwoPointProbe checkpoint update counter')
        self.base_optimizer.load_state_dict({k: v for k, v in state_dict.items() if k != '_two_point_probe'})
        self.param_groups = self.base_optimizer.param_groups
        self.real_updates = updates

    def __repr__(self):
        return f'TwoPointProbe({type(self.base_optimizer).__name__}, mode={self.mode}, radius={self.radius})'
