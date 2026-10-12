"""Matched three-point CE probes with a single ordinary optimizer commit.

Every mode evaluates BASE, BASE+rho*N(g0), and BASE-rho*N(g0), using
the same realized effective batch and replayed stochastic state. The four
submitted directions are gplus, 2*g0-gminus, g0+.5*(gplus-gminus), and g0.
Only the submitted direction is clipped. Production commits use HF AdamW;
the probe displacement is a normalized virtual SGD displacement, independent
of the learning rate and optimizer moments. No g2/g3, projection, history,
Fisher or explicit Hessian-vector product is evaluated.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn

from optimer_reverse_probe import TwoPointProbe
from optimer_g2 import (
    _Transaction, _rng_get, _rng_set, _norm, _finite, _ratio, _cosine,
    _bf16_weight_cast_proxy,
)


class ThreePointProbe(TwoPointProbe):
    """Three matched gradient phases followed by one AdamW/SGD commit.

    Reduction, trainable validation and transaction helpers are shared with
    TwoPointProbe, but its algorithm is never executed. The Trainer owns
    per-microbatch replay and global averaging when
    ``closure_gradients_reduced=True``. Standalone closures are replayed from
    the BASE RNG/state; only the clean phase advances the outer RNG stream.

    All modes execute both sides even when rho=0 or g0=0. ``g0`` is therefore
    a matched-probe-budget control, not a cheap one-pass implementation.
    """
    FORMAT = 'ThreePointProbe_v1'
    MODES = ('g1_plus', 'backward_reflected', 'central', 'g0')

    def __init__(self, params: Iterable, base_optimizer: torch.optim.Optimizer,
                 model: nn.Module, mode: str = 'central', radius: float = .05,
                 eps: float = 1e-12, grad_reduce: str = 'mean',
                 closure_gradients_reduced: bool = True,
                 require_fp32_trainable: bool = True,
                 quantization_diagnostics_interval: int = 0, **kwargs):
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError('mode must be g1_plus, backward_reflected, central or g0')
        # Reuse validation and ordinary optimizer delegation only. The
        # two-point step is never called and no legacy source is modified.
        super().__init__(params, base_optimizer, model, mode='reverse', radius=radius,
                         eps=eps, grad_reduce=grad_reduce,
                         closure_gradients_reduced=closure_gradients_reduced,
                         require_fp32_trainable=require_fp32_trainable,
                         quantization_diagnostics_interval=quantization_diagnostics_interval,
                         **kwargs)
        self.mode = mode

    def step(self, closure=None, clip_grad_norm=None, first_grad_ready: bool = False):
        if closure is None:
            raise ValueError('ThreePointProbe requires a task-only forward-backward closure')
        if self._busy:
            raise RuntimeError('Nested ThreePointProbe step is not allowed')
        if self._current_layout() != self._layout or {id(p) for p in self.params} != {
                id(p) for p in self.model.parameters() if p.requires_grad}:
            raise ValueError('Trainable layout changed: reconstruct ThreePointProbe')
        if clip_grad_norm is not None and (not math.isfinite(float(clip_grad_norm))
                                          or float(clip_grad_norm) < 0):
            raise ValueError('clip_grad_norm must be finite and nonnegative')
        eta = self._lr()
        entry = _Transaction(self.model)
        base_by_id = {id(p): value for p, value in entry.trainable}
        base = tuple(base_by_id[id(p)] for p in self.params)
        calls = 1 if first_grad_ready else 0
        reductions = 0
        rng_after_base = entry.rng
        optimizer_snapshot = None
        previous_stats, previous_updates = self.last_stats, self.real_updates
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
                raise ValueError('ThreePointProbe closure must return (outputs, scalar loss)')
            outputs, loss = result
            if torch.is_tensor(loss):
                if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError('ThreePointProbe closure loss must be finite scalar')
            elif (not isinstance(loss, (float, int)) or isinstance(loss, bool)
                  or not math.isfinite(float(loss))):
                raise FloatingPointError('ThreePointProbe closure loss must be finite scalar')
            if not self.closure_gradients_reduced:
                reductions += self._sync_grad()
            gradients, masks = self._grads()
            if not any(masks):
                raise RuntimeError('ThreePointProbe closure produced no parameter gradients')
            if not all(torch.equal(p, v) for p, v in zip(self.params, point)):
                raise AssertionError('ThreePointProbe CE closure mutated trainable parameters')
            if self._current_layout() != self._layout:
                raise AssertionError('ThreePointProbe CE closure mutated trainable layout')
            calls += 1
            return outputs, loss, gradients, masks, _rng_get()

        sampled = (self.quantization_diagnostics_interval > 0
                   and self.real_updates % self.quantization_diagnostics_interval == 0)
        cast_proxy = {
            'schema': 'ThreePointBF16WeightCastProxy_v1',
            'kind': 'parameter_cast_to_bfloat16', 'scope': 'all_trainable_parameters',
            'proves_forward_visibility': False, 'sampling_due': sampled,
            'sampled': False, 'interval': self.quantization_diagnostics_interval,
            'plus': None, 'minus': None,
        }
        try:
            if first_grad_ready:
                g0, mask0 = self._grads()
                if not any(mask0):
                    raise RuntimeError('first_grad_ready=True requires clean-point gradients')
            else:
                _, _, g0, mask0, rng_after_base = evaluate(base)
            g0_norm = _norm(g0)
            scale = self.radius / (g0_norm + self.eps)
            plus_point = tuple(value.clone().add_(grad, alpha=scale)
                               for value, grad in zip(base, g0))
            minus_point = tuple(value.clone().add_(grad, alpha=-scale)
                                for value, grad in zip(base, g0))
            _finite(plus_point, 'ThreePointProbe plus point')
            _finite(minus_point, 'ThreePointProbe minus point')
            plus_radius = _norm(tuple(value-old for value, old in zip(plus_point, base)))
            minus_radius = _norm(tuple(value-old for value, old in zip(minus_point, base)))
            plus_distinguishable = any(not torch.equal(v, b) for v, b in zip(plus_point, base))
            minus_distinguishable = any(not torch.equal(v, b) for v, b in zip(minus_point, base))
            if sampled:
                cast_proxy.update(plus=_bf16_weight_cast_proxy(base, plus_point),
                                  minus=_bf16_weight_cast_proxy(base, minus_point), sampled=True)
            # Both sides originate at BASE and the original clean g0. The
            # minus side never branches from the plus parameters or gradient.
            _, _, gplus, maskplus, _ = evaluate(plus_point)
            outputs, loss_value, gminus, maskminus, _ = evaluate(minus_point)
            if self.mode == 'g1_plus':
                final, final_mask = gplus, maskplus
                direction = 'gplus'
            elif self.mode == 'backward_reflected':
                final = tuple(a.mul(2).sub(c) for a, c in zip(g0, gminus))
                final_mask = tuple(a or c for a, c in zip(mask0, maskminus))
                direction = '2g0_minus_gminus'
            elif self.mode == 'central':
                final = tuple(a + .5*(b-c) for a, b, c in zip(g0, gplus, gminus))
                final_mask = tuple(a or b or c for a, b, c in zip(mask0, maskplus, maskminus))
                direction = 'g0_plus_half_gplus_minus_gminus'
            else:
                final, final_mask, direction = g0, mask0, 'g0'
            _finite(final, 'ThreePointProbe submitted gradient')
            plus_norm, minus_norm, final_norm = _norm(gplus), _norm(gminus), _norm(final)
            delta = tuple(b-a for a, b in zip(g0, final))
            delta_norm = _norm(delta)
            plus_response = tuple(b-a for a, b in zip(g0, gplus))
            minus_response = tuple(a-c for a, c in zip(g0, gminus))
            plus_response_norm, minus_response_norm = _norm(plus_response), _norm(minus_response)
            side_difference_norm = _norm(tuple(b-c for b, c in zip(gplus, gminus)))
            symmetric_response = tuple(.5*(b-c) for b, c in zip(gplus, gminus))
            even_response = tuple(.5*(b+c-2*a) for a, b, c in zip(g0, gplus, gminus))
            symmetric_response_norm, even_response_norm = _norm(symmetric_response), _norm(even_response)
            entry.restore()
            _rng_set(rng_after_base)
            if not all(torch.equal(p, old) for p, old in zip(self.params, base)):
                raise AssertionError('ThreePointProbe failed to restore BASE before commit')
            with torch.no_grad():
                for param, grad, present in zip(self.params, final, final_mask):
                    param.grad = grad.clone() if present else None
            if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(self.params, float(clip_grad_norm), error_if_nonfinite=True)
            clipped, _ = self._grads()
            clipped_norm = _norm(clipped)
            optimizer_snapshot = copy.deepcopy(self.base_optimizer.state_dict())
            self.base_optimizer.step()
            _finite(self.params, 'Committed ThreePointProbe parameters')
            actual_update_norm = _norm(tuple(p.detach()-old for p, old in zip(self.params, base)))
            self.real_updates += 1
            actual_radius = max(plus_radius, minus_radius)
            self.last_stats = {
                'probe_schema': self.FORMAT, 'probe_method': 'paired_three_point',
                'probe_protocol': 'paired_three_point',
                'probe_mode': self.mode, 'probe_final_direction': direction,
                'probe_sign': 0, 'probe_signs': [1, -1],
                'probe_plus_sign': 1, 'probe_minus_sign': -1,
                'probe_radius': self.radius, 'probe_rho': self.radius,
                'probe_radius_mode': 'fixed_l2_normalized_with_eps',
                'probe_radius_theoretical': self.radius*g0_norm/(g0_norm+self.eps),
                'probe_radius_actual': actual_radius,
                'probe_plus_radius_actual': plus_radius, 'probe_minus_radius_actual': minus_radius,
                'probe_radius_actual_definition': 'maximum_of_actual_plus_and_minus_displacements',
                'probe_distinguishable': plus_distinguishable or minus_distinguishable,
                'probe_plus_distinguishable': plus_distinguishable,
                'probe_minus_distinguishable': minus_distinguishable,
                'probe_lr': eta, 'probe_eps': self.eps,
                'probe_g0_norm': g0_norm,
                # Retained compatibility keys have an explicit plus reference;
                # both side-specific gradients are reported independently.
                'probe_g1_reference': 'gplus', 'probe_g1_norm': plus_norm,
                'probe_cos_g0_g1': _cosine(g0, gplus, g0_norm, plus_norm),
                'probe_gplus_norm': plus_norm, 'probe_gminus_norm': minus_norm,
                'probe_cos_g0_gplus': _cosine(g0, gplus, g0_norm, plus_norm),
                'probe_cos_g0_gminus': _cosine(g0, gminus, g0_norm, minus_norm),
                'probe_cos_gplus_gminus': _cosine(gplus, gminus, plus_norm, minus_norm),
                'probe_plus_response_norm': plus_response_norm,
                'probe_minus_response_norm': minus_response_norm,
                'probe_plus_response_relative_norm': _ratio(plus_response_norm, g0_norm),
                'probe_minus_response_relative_norm': _ratio(minus_response_norm, g0_norm),
                'probe_cos_plus_minus_response': _cosine(
                    plus_response, minus_response, plus_response_norm, minus_response_norm),
                'probe_symmetric_response_definition': '(gplus-gminus)/2',
                'probe_symmetric_response_norm': symmetric_response_norm,
                'probe_symmetric_response_relative_norm': _ratio(symmetric_response_norm, g0_norm),
                'probe_even_response_definition': '(gplus+gminus-2g0)/2',
                'probe_even_response_norm': even_response_norm,
                'probe_even_response_relative_norm': _ratio(even_response_norm, g0_norm),
                'probe_side_difference_norm': side_difference_norm,
                'probe_cos_g0_submitted': _cosine(g0, final, g0_norm, final_norm),
                'probe_response_definition': 'submitted_minus_g0',
                'probe_response_norm': delta_norm,
                'probe_response_relative_norm': _ratio(delta_norm, g0_norm),
                'probe_submitted_minus_g0_norm': delta_norm,
                'probe_submitted_minus_g0_relative_norm': _ratio(delta_norm, g0_norm),
                'probe_submitted_preclip_norm': final_norm,
                'probe_submitted_postclip_norm': clipped_norm,
                'probe_gradient_clip_scale': _ratio(clipped_norm, final_norm),
                'probe_final_clip_enabled': bool(clip_grad_norm and float(clip_grad_norm) > 0),
                'probe_actual_update_norm': actual_update_norm,
                'probe_actual_update_over_probe_radius': _ratio(actual_update_norm, actual_radius),
                'probe_ce_gradient_phases': calls, 'probe_replay_closure_calls': calls-1,
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
                'probe_regularization_hook_used': False,
                'probe_quantization_proxy': cast_proxy,
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

    def _config(self):
        config = super()._config()
        config.update(protocol='paired_three_point', probe_signs=[1, -1],
                      ce_gradient_phases=3, final_direction=self.mode)
        return config

    def state_dict(self):
        if self._busy:
            raise RuntimeError('Cannot checkpoint a pending ThreePointProbe transaction')
        state = self.base_optimizer.state_dict()
        state['_three_point_probe'] = dict(format=self.FORMAT, config=self._config(),
                                          real_updates=self.real_updates)
        return state

    def load_state_dict(self, state_dict):
        if self._busy:
            raise RuntimeError('Cannot load a pending ThreePointProbe transaction')
        metadata = state_dict.get('_three_point_probe')
        if not isinstance(metadata, dict) or metadata.get('format') != self.FORMAT:
            raise ValueError('ThreePointProbe checkpoint metadata is required')
        if metadata.get('config') != self._config():
            raise ValueError('ThreePointProbe checkpoint algorithm/config/layout mismatch')
        updates = metadata.get('real_updates')
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
            raise ValueError('Invalid ThreePointProbe checkpoint update counter')
        self.base_optimizer.load_state_dict({k: v for k, v in state_dict.items()
                                             if k != '_three_point_probe'})
        self.param_groups = self.base_optimizer.param_groups
        self.real_updates = updates

    def __repr__(self):
        return f'ThreePointProbe({type(self.base_optimizer).__name__}, mode={self.mode}, radius={self.radius})'
