"""Current-G2 with independently registered inner and outer probe geometry.

The frozen ResponseStep implements real ordinary SGD and checkpoint mechanics.
This subclass changes only virtual probe positions, never the submitted update
rule, history handling, trainable scope, stochastic replay, or commit backend.
Fixed values are absolute global FP32 trainable-parameter L2 distances.
"""
import math

import torch

from response_step import ResponseStep, _clone, _cos, _digest, _dot, _finite, _norm
from state_transaction import State, backend_context, rng_equal, rng_get, rng_set


class IndependentResponseStep(ResponseStep):
    def __init__(self, model, params, geometry, config):
        if config.get('method', 'current_g2') != 'current_g2':
            raise ValueError('Independent probes are registered only for current_g2')
        if float(config.get('history_weight', 0.)) != 0.:
            raise ValueError('Independent current response must not use history')
        if config.get('radius_mode', 'sgd_step') not in ('sgd_step', 'independent_inner_outer_v1'):
            raise ValueError('Unknown independent probe radius protocol')
        parent = dict(config, method='current_g2', history_weight=0., radius_mode='sgd_step')
        super().__init__(model, params, geometry, parent)
        self.inner_probe_radius_mode = config.get('inner_probe_radius_mode', 'sgd_step')
        self.outer_probe_radius_mode = config.get('outer_probe_radius_mode', 'sgd_step')
        for prefix in ('inner', 'outer'):
            mode = getattr(self, prefix + '_probe_radius_mode')
            value = config.get('fixed_' + prefix + '_probe_radius')
            if mode not in ('sgd_step', 'fixed_l2'):
                raise ValueError('Unknown ' + prefix + ' probe radius mode')
            if mode == 'fixed_l2':
                if isinstance(value, bool) or value is None:
                    raise ValueError('A fixed radius requires a numeric absolute L2 value')
                value = float(value)
                if not math.isfinite(value) or value <= 0.:
                    raise ValueError('Fixed probe radii must be finite and positive')
            elif value is not None:
                raise ValueError('A coupled radius must not also specify a fixed value')
            setattr(self, 'fixed_' + prefix + '_probe_radius', value)
        mask = (self.inner_probe_radius_mode == 'fixed_l2', self.outer_probe_radius_mode == 'fixed_l2')
        self.radius_variant = {(False, False): 'coupled_reference', (True, False): 'fixed_inner',
                               (False, True): 'fixed_outer', (True, True): 'fixed_both'}[mask]
        if config.get('radius_variant', self.radius_variant) != self.radius_variant:
            raise ValueError('Radius variant disagrees with the two independent modes')
        self.response_probe_variant = config.get('response_probe_variant', self.radius_variant)

    def _radius_config(self):
        return {key: getattr(self, key) for key in ('inner_probe_radius_mode', 'outer_probe_radius_mode',
                'fixed_inner_probe_radius', 'fixed_outer_probe_radius', 'radius_variant', 'response_probe_variant')}

    def _decorate(self, out, inner_length, outer_length):
        r, c = out['record'], out['context']
        gnorm = r['g_grad_norm']
        difference = tuple(a - b for a, b in zip(c['g2'], c['g0'])) if c['g2'] is not None else None
        r.update(self._radius_config())
        r.update({'radius_protocol': 'independent_inner_outer_v1',
                  'radius_mode': 'independent_inner_outer_v1', 'h_ref': r['h'],
                  'h_ref_definition': 'current_lr*global_parameter_L2_norm(g0), reference only',
                  'inner_probe_radius_theoretical': inner_length,
                  'outer_probe_radius_theoretical': outer_length,
                  'inner_probe_radius_actual': r['future_actual_parameter_displacement_norm'],
                  'outer_probe_radius_actual': r['relocation_actual_parameter_displacement_norm'],
                  'a_relative_norm': _norm(c['response']) / gnorm if gnorm > self.eps else None,
                  'g2_minus_g0_relative_norm': _norm(difference) / gnorm if difference is not None and gnorm > self.eps else None,
                  'probe_direction_normalization': 'global_L2_above_eps_without_final_update_rescaling'})
        c.update({'inner_probe_radius_theoretical': inner_length, 'outer_probe_radius_theoretical': outer_length})
        return out

    def compute(self, inputs, targets, loss_closure, current_lr):
        if self.radius_variant == 'coupled_reference':
            # Invoke the unmodified frozen implementation, including FP32 order.
            out = super().compute(inputs, targets, loss_closure, current_lr)
            return self._decorate(out, out['record']['h'], out['record']['h'])
        if self._pending is not None:
            raise RuntimeError('The previously computed step has not been applied')
        self._check_layout()
        eta = float(current_lr)
        if not math.isfinite(eta) or eta < 0.:
            raise ValueError('current_lr must be finite and nonnegative')
        if self._history is not None:
            raise AssertionError('Independent current response must never carry a history cache')
        entry = State(self.model)
        base, rng0 = _clone(self.params), entry.rng
        calls = 0

        def probe(point):
            nonlocal calls
            entry.restore()
            with torch.no_grad():
                for p, value in zip(self.params, point):
                    p.copy_(value)
                for p in self.model.parameters():
                    p.grad = None
            with backend_context(self.model, self.backend):
                loss, logits = loss_closure(inputs, targets)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Closure must return a finite scalar loss')
                if not loss.requires_grad:
                    raise ValueError('Closure loss must retain its first-order autograd graph')
                gradients = torch.autograd.grad(loss, self.params, allow_unused=True, create_graph=False)
            if any(p.grad is not None for p in self.model.parameters()):
                raise AssertionError('loss_closure must not call backward')
            grads = tuple(torch.zeros_like(p) if g is None else g.detach().clone()
                          for p, g in zip(self.params, gradients))
            _finite(grads, 'Probe gradient')
            if not all(torch.equal(p, v) for p, v in zip(self.params, point)):
                raise AssertionError('loss_closure mutated trainable parameters')
            calls += 1
            return float(loss.detach().item()), logits.detach().clone(), grads, rng_get()

        try:
            loss0, logits0, g0, rng1 = probe(base)
            gnorm = _norm(g0)
            h_ref = eta * gnorm
            inner_length = self.fixed_inner_probe_radius if self.inner_probe_radius_mode == 'fixed_l2' else h_ref
            outer_length = self.fixed_outer_probe_radius if self.outer_probe_radius_mode == 'fixed_l2' else h_ref
            if not all(math.isfinite(v) for v in (h_ref, inner_length, outer_length)):
                raise FloatingPointError('Nonfinite registered probe length')
            g1 = g2 = future = relocation = None
            loss1 = loss2 = None
            inner_actual = outer_actual = 0.
            inner_distinct = outer_distinct = None
            response = tuple(torch.zeros_like(g) for g in g0)
            fallback_reason = None
            if self.inner_probe_radius_mode == 'sgd_step':
                future = tuple(v.clone().add_(g, alpha=-eta) for v, g in zip(base, g0))
            elif gnorm > self.eps:
                future = tuple(v.clone().add_(g, alpha=-inner_length / gnorm) for v, g in zip(base, g0))
            if future is not None:
                _finite(future, 'Inner descending probe point')
                inner_distinct = any(not torch.equal(v, f) for v, f in zip(base, future))
                inner_actual = _norm(tuple(f - v for f, v in zip(future, base)))
            if gnorm <= self.eps and self.inner_probe_radius_mode == 'fixed_l2':
                fallback_reason = 'base_gradient_length_at_or_below_eps'
            elif inner_length <= self.eps:
                fallback_reason = 'inner_probe_length_at_or_below_eps'
            elif not inner_distinct:
                fallback_reason = 'predicted_point_bitwise_indistinguishable_in_FP32'
            U = _clone(g0)
            if fallback_reason is None:
                loss1, _, g1, _ = probe(future)
                response = tuple(a - b for a, b in zip(g0, g1))
                response_norm = _norm(response)
                if response_norm <= self.eps:
                    fallback_reason = 'current_response_length_at_or_below_eps'
                elif outer_length <= self.eps:
                    fallback_reason = 'outer_probe_length_at_or_below_eps'
                else:
                    relocation = tuple(v.clone().add_(a, alpha=outer_length / response_norm)
                                       for v, a in zip(base, response))
                    _finite(relocation, 'Outer relocated response point')
                    outer_distinct = any(not torch.equal(v, f) for v, f in zip(base, relocation))
                    outer_actual = _norm(tuple(f - v for f, v in zip(relocation, base)))
                    if not outer_distinct:
                        fallback_reason = 'relocated_point_bitwise_indistinguishable_in_FP32'
                    else:
                        loss2, _, g2, _ = probe(relocation)
                        U = _clone(g2)
            _finite(U, 'Final update vector')
        finally:
            entry.restore()
            restored = entry.check()
        rng_set(rng1)
        if not rng_equal(rng_get(), rng1):
            raise AssertionError('Failed to commit exactly the base-probe RNG state')
        if fallback_reason is not None:
            self.fallback_count += 1
        U = _clone(U)
        unorm, anorm = _norm(U), _norm(response)
        record = {
            'method': self.method, 'direction': self.method, 'cache_source': None,
            'cache_source_description': 'disabled',
            'formula': 'g2 at BASE+outer_radius*N(g0-gradL(BASE-inner_radius*N(g0)))',
            'lambda': outer_length, 'lambda_definition': 'registered_outer_probe_radius',
            'h': h_ref, 'ell': None, 'ell_f': inner_length, 'ell_b': None,
            'current_lr': eta, 'history_weight': 0., 'effective_history_weight': 0.,
            'h_over_ell': None, 'ell_f_over_ell_b': None, 'history_coefficient': 0.,
            'future_actual_parameter_displacement_norm': inner_actual,
            'relocation_actual_parameter_displacement_norm': outer_actual,
            'predicted_point_FP32_distinguishable': inner_distinct,
            'relocated_point_FP32_distinguishable': outer_distinct,
            'relocation_origin': 'base' if relocation is not None else None,
            'history_available': False, 'history_used': False, 'history_reason': 'history_disabled',
            'history_incoming_vector_sha256': None, 'history_incoming_vector_norm': None,
            'history_incoming_vector_source': None, 'history_cache_raw_gradient_sha256': None,
            'history_cache_actual_displacement_sha256': None, 'history_cache_after_sha256': None,
            'base_parameter_sha256': _digest(base),
            'future_parameter_sha256': _digest(future) if future is not None else None,
            'relocation_parameter_sha256': _digest(relocation) if relocation is not None else None,
            'g0_sha256': _digest(g0), 'g1_sha256': _digest(g1) if g1 is not None else None,
            'g2_sha256': _digest(g2) if g2 is not None else None, 'raw_gradient_sha256': _digest(g0),
            'future_gradient_sha256': _digest(g1) if g1 is not None else None,
            'U_sha256': _digest(U), 'final_gradient_sha256': _digest(U),
            'gradient_norms': {'g': gnorm, 'g0': gnorm, 'gf': _norm(g1) if g1 is not None else None,
                               'g1': _norm(g1) if g1 is not None else None,
                               'g2': _norm(g2) if g2 is not None else None,
                               'U': unorm, 'final': unorm, 'response': anorm,
                               'future_response': anorm, 'history_response': 0.},
            'g_grad_norm': gnorm, 'U_grad_norm': unorm,
            'u_g_ratio': unorm / gnorm if gnorm > self.eps else None,
            'g_dot_U': _dot(g0, U), 'g_U_cosine': _cos(g0, U, self.eps),
            'u_v_cosine': None, 'cos_u_v_definition': 'history_disabled', 'response_cosine': None,
            'future_component_relative_norm': anorm / gnorm if gnorm > self.eps else None,
            'history_component_relative_norm': 0. if gnorm > self.eps else None,
            'loss0': loss0, 'loss_future': loss1, 'loss_relocation': loss2,
            'closure_count': calls, 'gradient_calls': calls, 'ordinary_gradient_calls': calls,
            'training_ce_gradient_passes': calls, 'training_forward_count': calls,
            'fallback': fallback_reason is not None, 'fallback_reason': fallback_reason,
            'response_fallback_reason': None, 'fallback_count': self.fallback_count,
            'history_reset_count': self.history_reset_count,
            'state_restore_checks': restored, 'only_base_R1_rng_advance': True,
            'fisher_estimation_passes': 0, 'weight_merge_updates': 0, 'projection_calls': 0,
            'hvp_calls': 0, 'qr_calls': 0, 'create_graph_calls': 0,
            'gradient_clipping_calls': 0, 'final_norm_rescale_calls': 0,
        }
        context = {'base': base, 'future': future, 'relocation': relocation,
                   'g0': g0, 'g1': g1, 'g2': g2, 'g': g0, 'gf': g1, 'U': U,
                   'response': response, 'rng0': rng0, 'rng1': rng1,
                   'entry_state': entry, 'loss0': loss0, 'loss_future': loss1}
        geometry_before = self.geometry.capture() if self.geometry is not None else None
        self._pending = {'eta': eta, 'base': base, 'g0': g0, 'U': U,
                         'record': record, 'context': context, 'geometry_before': geometry_before}
        return self._decorate({'loss0': loss0, 'logits0': logits0, 'U': U,
                               'record': record, 'context': context}, inner_length, outer_length)

    def state_dict(self):
        state = super().state_dict()
        state.update({'format': 'IndependentResponseStep_v1', 'independent_radius_config': self._radius_config()})
        return state

    def load_state_dict(self, state):
        if state.get('format') != 'IndependentResponseStep_v1':
            raise ValueError('Independent probes require their own explicit checkpoint format')
        if state.get('independent_radius_config') != self._radius_config():
            raise ValueError('Saved independent probe geometry differs from this configuration')
        if state.get('history') is not None:
            raise ValueError('Independent current response cannot restore a history cache')
        parent_state = dict(state, format='ResponseStep_v1')
        super().load_state_dict(parent_state)
