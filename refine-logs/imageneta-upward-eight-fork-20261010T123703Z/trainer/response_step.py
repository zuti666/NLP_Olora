"""Transactional ordinary-SGD implementation of four response update signals.

``history_fusion`` caches the base gradient. ``history_g2`` instead caches the
entire actually submitted vector, with its amplitude. Both independently cache
the actual FP32 parameter displacement; neither caches a unit direction or an
idealized -lr*gradient displacement. All virtual probes start from one saved
base state and replay its RNG; only the base probe advances the training RNG.
"""
import hashlib
import math

import torch

from state_transaction import State, backend_context, rng_equal, rng_get, rng_set


METHODS = ('sgd', 'current_g2', 'history_fusion', 'history_g2')
CACHE_SOURCE = {'sgd': None, 'current_g2': None,
                'history_fusion': 'base', 'history_g2': 'final'}


def _clone(values):
    return tuple(v.detach().clone() for v in values)


def _digest(values):
    h = hashlib.sha256()
    for value in values:
        v = value.detach().contiguous()
        h.update(str((tuple(v.shape), str(v.dtype))).encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def _dot(a, b):
    value = float(torch.stack([(x.double() * y.double()).sum()
                               for x, y in zip(a, b)]).sum().item())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite response inner product')
    return value


def _norm(a):
    return math.sqrt(max(0., _dot(a, a)))


def _cos(a, b, eps):
    den = _norm(a) * _norm(b)
    return _dot(a, b) / den if den > eps else None


def _finite(values, label):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(label + ' has nonfinite tensors')


class ResponseStep:
    """Separate virtual construction from exactly one committed torch SGD step.

    closure(inputs, targets) returns a finite scalar loss and logits. It must
    retain the loss graph, must not call backward or update parameters, and uses
    the supplied realized batch. Construct a fresh instance at each task/head
    expansion. Ordinary per-epoch LR scheduling belongs to the outer trainer.
    """
    def __init__(self, model, params, geometry, config):
        self.model, self.params, self.geometry = model, list(params), geometry
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Nonempty unique trainable parameters required')
        if set(map(id, self.params)) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Parameter scope must cover all and only trainable parameters')
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError('Response training uses actual FP32 coordinates')
        if geometry is not None and set(map(id, geometry.params)) != set(map(id, self.params)):
            raise ValueError('Effective geometry does not cover exactly all trainables')
        self.method = config.get('method', 'history_fusion')
        if self.method not in METHODS:
            raise ValueError('Unknown response method')
        self.direction = self.method
        self.cache_source = CACHE_SOURCE[self.method]
        self.history_weight = float(config.get('history_weight', .5 if self.cache_source else 0.))
        self.eps = float(config.get('eps', 1e-12))
        self.backend = config.get('backend', 'native')
        if config.get('radius_mode', 'sgd_step') != 'sgd_step' or float(config.get('radius_scale', 1.)) != 1.:
            raise ValueError('Registered response radius must be r=h=lr*global_norm(g0)')
        if not math.isfinite(self.history_weight) or not 0. <= self.history_weight <= 1.:
            raise ValueError('History weight must be finite in [0,1]')
        if self.cache_source is None and self.history_weight != 0.:
            raise ValueError('Methods without history require history_weight=0')
        if not math.isfinite(self.eps) or self.eps <= 0.:
            raise ValueError('eps must be finite and positive')
        if self.backend != 'native':
            raise ValueError('The registered response protocol requires native backend')
        self._layout = self._layout_now()
        self._history = None
        self._pending = None
        self.fallback_count = self.history_reset_count = self.real_updates = 0

    def _layout_now(self):
        return tuple((id(p), tuple(p.shape), str(p.dtype), str(p.device), p.requires_grad)
                     for p in self.params)

    def _portable_layout(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return [{'name': names[id(p)], 'shape': list(p.shape), 'dtype': str(p.dtype)}
                for p in self.params]

    def reset_history(self, reason='explicit_reset'):
        if self._pending is not None:
            raise RuntimeError('Cannot reset history while a computed update is pending')
        self._history = None
        self.history_reset_count += 1
        self.last_reset_reason = str(reason)

    def _check_layout(self):
        if (self._layout_now() != self._layout or
                set(map(id, self.params)) != {id(p) for p in self.model.parameters() if p.requires_grad}):
            self._history = None
            self.history_reset_count += 1
            self.last_reset_reason = 'trainable_layout_changed'
            raise ValueError('Trainable layout changed: construct a new ResponseStep for this task')

    def compute(self, inputs, targets, loss_closure, current_lr):
        if self._pending is not None:
            raise RuntimeError('The previously computed step has not been applied')
        self._check_layout()
        eta = float(current_lr)
        if not math.isfinite(eta) or eta < 0.:
            raise ValueError('current_lr must be finite and nonnegative')
        entry = State(self.model)
        base, rng0 = _clone(self.params), entry.rng
        calls = 0

        def probe(point):
            nonlocal calls
            entry.restore()
            with torch.no_grad():
                for p, v in zip(self.params, point):
                    p.copy_(v)
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

        reset_reason = None
        try:
            loss0, logits0, g0, rng1 = probe(base)
            g_norm = _norm(g0)
            h_length = eta * g_norm
            if not math.isfinite(h_length):
                raise FloatingPointError('Nonfinite predicted displacement length')
            h = self._history
            history_reason = 'no_previous_update'
            if self.cache_source is None or self.history_weight == 0.:
                h, history_reason = None, 'history_disabled'
            elif h is not None:
                if h['after_sha256'] != _digest(base):
                    reset_reason = history_reason = 'parameter_continuity_changed'
                    h = None
                elif h['layout'] != self._layout or h['length'] <= self.eps:
                    reset_reason = history_reason = 'invalid_previous_displacement_or_layout'
                    h = None
                elif h['cache_source'] != self.cache_source:
                    raise ValueError('Historical vector source differs from this method')
                else:
                    history_reason = 'valid_previous_actual_update'
            ell = h['length'] if h is not None else None
            z = tuple(torch.zeros_like(v) for v in g0)
            g1 = g2 = future = relocation = None
            future_loss = relocation_loss = None
            future_actual_length = relocation_actual_length = 0.
            future_distinguishable = relocation_distinguishable = None
            current_response = history_response = response = z
            history_available = h is not None
            history_used = False
            w_eff = coefficient = 0.
            fallback_reason = response_fallback_reason = None
            if self.method == 'sgd':
                U = _clone(g0)
            else:
                future = tuple(v.clone().add_(grad, alpha=-eta) for v, grad in zip(base, g0))
                _finite(future, 'Predicted SGD point')
                future_distinguishable = any(not torch.equal(v, f) for v, f in zip(base, future))
                future_actual_length = _norm(tuple(f - v for f, v in zip(future, base)))
                if h_length <= self.eps:
                    fallback_reason = 'predicted_displacement_length_at_or_below_eps'
                elif not future_distinguishable:
                    fallback_reason = 'predicted_point_bitwise_indistinguishable_in_FP32'
                if fallback_reason is not None:
                    U = _clone(g0)
                else:
                    future_loss, _, g1, _ = probe(future)
                    current_response = tuple(a - b for a, b in zip(g0, g1))
                    response = current_response
                    history_used = history_available
                    w_eff = self.history_weight if history_used else 0.
                    coefficient = h_length * w_eff / ell if history_used else 0.
                    history_response = (tuple(old - cur for old, cur in zip(h['vector'], g0))
                                        if history_used else z)
                    if history_used:
                        response = tuple((1. - w_eff) * fr + coefficient * hr
                                         for fr, hr in zip(current_response, history_response))
                    if self.method == 'history_fusion':
                        # Preserve the prior HRF implementation's FP32 operation order.
                        if history_used:
                            U = tuple(gi + (1. - w_eff) * fr + coefficient * hr
                                      for gi, fr, hr in zip(g0, current_response, history_response))
                        else:
                            U = tuple(2. * gi - fi for gi, fi in zip(g0, g1))
                    else:
                        response_norm = _norm(response)
                        if response_norm <= self.eps:
                            response_fallback_reason = 'blended_response_zero_try_current_response'
                            response = current_response
                            response_norm = _norm(response)
                            history_used = False
                            w_eff = coefficient = 0.
                        if response_norm <= self.eps:
                            fallback_reason = 'current_response_length_at_or_below_eps'
                            U = _clone(g0)
                        else:
                            # Deliberately from BASE, not the virtual SGD predictor.
                            relocation = tuple(v.clone().add_(a, alpha=h_length / response_norm)
                                               for v, a in zip(base, response))
                            _finite(relocation, 'Relocated response point')
                            relocation_distinguishable = any(not torch.equal(v, f)
                                                            for v, f in zip(base, relocation))
                            relocation_actual_length = _norm(tuple(f - v for f, v in zip(relocation, base)))
                            if not relocation_distinguishable:
                                fallback_reason = 'relocated_point_bitwise_indistinguishable_in_FP32'
                                U = _clone(g0)
                                history_used = False
                                w_eff = coefficient = 0.
                            else:
                                relocation_loss, _, g2, _ = probe(relocation)
                                U = _clone(g2)
            _finite(U, 'Final update vector')
        finally:
            entry.restore()
            restored = entry.check()
        # Only commit bookkeeping after all probes and the complete restoration succeed.
        rng_set(rng1)
        if not rng_equal(rng_get(), rng1):
            raise AssertionError('Failed to commit exactly the base-probe RNG state')
        if reset_reason is not None:
            self._history = None
            self.history_reset_count += 1
            self.last_reset_reason = reset_reason
        if fallback_reason is not None or response_fallback_reason is not None:
            self.fallback_count += 1
        U = _clone(U)
        u_norm = _norm(U)
        future_component = tuple((1. - w_eff) * v for v in current_response)
        history_component = tuple(coefficient * v for v in history_response)
        record = {
            'method': self.method, 'direction': self.method, 'cache_source': self.cache_source,
            'cache_source_description': {'base': 'raw_base_gradient_g0', 'final': 'full_amplitude_final_update_vector_d', None: 'disabled'}[self.cache_source],
            'formula': {'sgd': 'g0', 'current_g2': 'g2 at base+h*N(g0-g1)',
                        'history_fusion': 'g0+(1-w)*(g0-g1)+w*h/ell*(previous_base_gradient-g0)',
                        'history_g2': 'g2 at base+h*N((1-w)*(g0-g1)+w*h/ell*(previous_final_vector-g0))'}[self.method],
            'lambda': h_length, 'lambda_definition': 'h=current_lr*global_parameter_L2_norm(g0)',
            'h': h_length, 'ell': ell, 'ell_f': h_length, 'ell_b': ell,
            'current_lr': eta, 'history_weight': self.history_weight, 'effective_history_weight': w_eff,
            'h_over_ell': h_length / ell if ell is not None else None,
            'ell_f_over_ell_b': h_length / ell if ell is not None else None,
            'history_coefficient': coefficient,
            'future_actual_parameter_displacement_norm': future_actual_length,
            'relocation_actual_parameter_displacement_norm': relocation_actual_length,
            'predicted_point_FP32_distinguishable': future_distinguishable,
            'relocated_point_FP32_distinguishable': relocation_distinguishable,
            'relocation_origin': 'base' if relocation is not None else None,
            'history_available': history_available, 'history_used': history_used,
            'history_reason': history_reason,
            'history_incoming_vector_sha256': h['vector_sha256'] if h is not None else None,
            'history_incoming_vector_norm': _norm(h['vector']) if h is not None else None,
            'history_incoming_vector_source': h['cache_source'] if h is not None else None,
            'history_cache_raw_gradient_sha256': h['vector_sha256'] if h is not None and self.cache_source == 'base' else None,
            'history_cache_actual_displacement_sha256': h['displacement_sha256'] if h is not None else None,
            'history_cache_after_sha256': h['after_sha256'] if h is not None else None,
            'base_parameter_sha256': _digest(base),
            'future_parameter_sha256': _digest(future) if future is not None else None,
            'relocation_parameter_sha256': _digest(relocation) if relocation is not None else None,
            'g0_sha256': _digest(g0), 'g1_sha256': _digest(g1) if g1 is not None else None,
            'g2_sha256': _digest(g2) if g2 is not None else None,
            'raw_gradient_sha256': _digest(g0),
            'future_gradient_sha256': _digest(g1) if g1 is not None else None,
            'U_sha256': _digest(U), 'final_gradient_sha256': _digest(U),
            'gradient_norms': {'g': g_norm, 'g0': g_norm,
                               'gf': _norm(g1) if g1 is not None else None,
                               'g1': _norm(g1) if g1 is not None else None,
                               'g2': _norm(g2) if g2 is not None else None,
                               'U': u_norm, 'final': u_norm, 'response': _norm(response),
                               'future_response': _norm(current_response),
                               'history_response': _norm(history_response)},
            'g_grad_norm': g_norm, 'U_grad_norm': u_norm,
            'u_g_ratio': u_norm / g_norm if g_norm > self.eps else None,
            'g_dot_U': _dot(g0, U), 'g_U_cosine': _cos(g0, U, self.eps),
            'u_v_cosine': _cos(g0, tuple(-v for v in h['displacement']), self.eps) if history_used else None,
            'cos_u_v_definition': 'cos(g0,-actual_previous_displacement)',
            'response_cosine': _cos(current_response, history_response, self.eps) if history_available else None,
            'future_component_relative_norm': _norm(future_component) / g_norm if g_norm > self.eps else None,
            'history_component_relative_norm': _norm(history_component) / g_norm if g_norm > self.eps else None,
            'loss0': loss0, 'loss_future': future_loss, 'loss_relocation': relocation_loss,
            'closure_count': calls, 'gradient_calls': calls, 'ordinary_gradient_calls': calls,
            'training_ce_gradient_passes': calls, 'training_forward_count': calls,
            'fallback': fallback_reason is not None or response_fallback_reason is not None,
            'fallback_reason': fallback_reason, 'response_fallback_reason': response_fallback_reason,
            'fallback_count': self.fallback_count, 'history_reset_count': self.history_reset_count,
            'state_restore_checks': restored, 'only_base_R1_rng_advance': True,
            'fisher_estimation_passes': 0, 'weight_merge_updates': 0, 'projection_calls': 0,
            'hvp_calls': 0, 'qr_calls': 0, 'create_graph_calls': 0,
            'gradient_clipping_calls': 0, 'final_norm_rescale_calls': 0,
        }
        context = {'base': base, 'future': future, 'relocation': relocation,
                   'g0': g0, 'g1': g1, 'g2': g2, 'g': g0, 'gf': g1, 'U': U,
                   'response': response, 'rng0': rng0, 'rng1': rng1,
                   'entry_state': entry, 'loss0': loss0, 'loss_future': future_loss}
        geometry_before = self.geometry.capture() if self.geometry is not None else None
        self._pending = {'eta': eta, 'base': base, 'g0': g0, 'U': U,
                         'record': record, 'context': context, 'geometry_before': geometry_before}
        return {'loss0': loss0, 'logits0': logits0, 'U': U, 'record': record, 'context': context}

    def state_dict(self):
        if self._pending is not None:
            raise RuntimeError('Cannot checkpoint between compute and apply_update')
        self._check_layout()
        history = None
        if self._history is not None:
            h = self._history
            history = {'vector': tuple(v.detach().cpu().clone() for v in h['vector']),
                       'displacement': tuple(v.detach().cpu().clone() for v in h['displacement']),
                       'length': h['length'], 'vector_sha256': h['vector_sha256'],
                       'displacement_sha256': h['displacement_sha256'],
                       'after_sha256': h['after_sha256'], 'cache_source': h['cache_source'],
                       'valid': True, 'layout': self._portable_layout()}
        return {'format': 'ResponseStep_v1', 'method': self.method,
                'history_weight': self.history_weight, 'eps': self.eps, 'backend': self.backend,
                'cache_source': self.cache_source, 'layout': self._portable_layout(), 'history': history,
                'fallback_count': self.fallback_count, 'history_reset_count': self.history_reset_count,
                'real_updates': self.real_updates, 'last_reset_reason': getattr(self, 'last_reset_reason', None),
                'cache_coordinate': 'full_amplitude_vector_and_actual_FP32_trainable_parameter_displacement'}

    def load_state_dict(self, state):
        if self._pending is not None:
            raise RuntimeError('Cannot restore history while an update is pending')
        self._check_layout()
        if state.get('format') != 'ResponseStep_v1':
            raise ValueError('Unknown response state format')
        for key in ('method', 'history_weight', 'eps', 'backend', 'cache_source'):
            if state.get(key) != getattr(self, key):
                raise ValueError('Saved response configuration differs: ' + key)
        if state.get('layout') != self._portable_layout():
            raise ValueError('Saved response parameter names/shapes/dtypes differ')
        counters = {}
        for key in ('fallback_count', 'history_reset_count', 'real_updates'):
            value = state.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError('Invalid saved response counter: ' + key)
            counters[key] = value
        h, loaded = state.get('history'), None
        if h is not None:
            if self.cache_source is None or self.history_weight <= 0.:
                raise ValueError('History-disabled method cannot restore a history cache')
            if h.get('valid') is not True or h.get('layout') != self._portable_layout():
                raise ValueError('Saved history cache is invalid or has incompatible layout')
            if h.get('cache_source') != self.cache_source:
                raise ValueError('Saved historical vector must preserve this method-specific source')
            vectors = {}
            for key in ('vector', 'displacement'):
                values = h.get(key)
                if not isinstance(values, (list, tuple)) or len(values) != len(self.params):
                    raise ValueError('Invalid saved history tensor count: ' + key)
                if any(not torch.is_tensor(v) or v.shape != p.shape or v.dtype != p.dtype
                       for v, p in zip(values, self.params)):
                    raise ValueError('Invalid saved history tensor shape/dtype: ' + key)
                vectors[key] = tuple(v.detach().to(device=p.device).clone() for v, p in zip(values, self.params))
                _finite(vectors[key], 'Saved ' + key)
            vector_hash, delta_hash = _digest(vectors['vector']), _digest(vectors['displacement'])
            if vector_hash != h.get('vector_sha256') or delta_hash != h.get('displacement_sha256'):
                raise ValueError('Saved history tensor digest mismatch')
            length = _norm(vectors['displacement'])
            saved_length = float(h.get('length', float('nan')))
            if (length <= self.eps or not math.isfinite(saved_length)
                    or not math.isclose(length, saved_length, rel_tol=1e-12, abs_tol=0.)):
                raise ValueError('Saved history length disagrees with actual displacement')
            if _digest(self.params) != h.get('after_sha256'):
                raise ValueError('Load the exact saved current model before restoring its history')
            loaded = {'vector': vectors['vector'], 'displacement': vectors['displacement'],
                      'length': length, 'vector_sha256': vector_hash, 'displacement_sha256': delta_hash,
                      'after_sha256': h['after_sha256'], 'layout': self._layout,
                      'cache_source': self.cache_source}
        self._history = loaded
        for key, value in counters.items():
            setattr(self, key, value)
        self.last_reset_reason = state.get('last_reset_reason')

    @torch.no_grad()
    def apply_update(self, current_lr, optimizer):
        if self._pending is None:
            raise RuntimeError('compute must precede the single real SGD update')
        self._check_layout()
        pending, eta = self._pending, float(current_lr)
        if not math.isfinite(eta) or eta != pending['eta']:
            raise ValueError('SGD learning rate changed between prediction and update')
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError('Only ordinary torch.optim.SGD is supported')
        optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
        if len(optimizer_params) != len(self.params) or set(map(id, optimizer_params)) != set(map(id, self.params)):
            raise ValueError('SGD parameter scope mismatch')
        for group in optimizer.param_groups:
            if float(group['lr']) != eta:
                raise ValueError('All SGD groups must retain current_lr')
            if (any(float(group.get(key, 0.)) != 0. for key in ('momentum', 'dampening', 'weight_decay'))
                    or any(bool(group.get(key, False)) for key in ('nesterov', 'maximize', 'differentiable', 'foreach', 'fused'))):
                raise ValueError('Natural SGD requires zero momentum/decay, no alternate update backend')
        if any(optimizer.state[p] for p in optimizer_params):
            raise ValueError('Ordinary SGD must not contain retained optimizer moments')
        if not all(torch.equal(p, b) for p, b in zip(self.params, pending['base'])):
            self._history = None
            self.history_reset_count += 1
            self.last_reset_reason = 'parameters_changed_between_compute_and_update'
            raise AssertionError('Parameters changed between compute and the real update')
        before_commit = State(self.model)
        try:
            for p, value in zip(self.params, pending['U']):
                p.grad = value.clone()
            expected = tuple(p.detach().clone().add_(value, alpha=-eta)
                             for p, value in zip(self.params, pending['U']))
            optimizer.step()
            exact = all(torch.equal(p, value) for p, value in zip(self.params, expected))
            if not exact:
                raise AssertionError('Actual torch SGD differs from natural FP32 single-step specification')
            _finite(self.params, 'Updated parameters')
            displacement = tuple(p.detach() - before for p, before in zip(self.params, pending['base']))
            _finite(displacement, 'Actual historical displacement')
            ell = _norm(displacement)
            effective = self.geometry.actual_delta(pending['geometry_before']) if self.geometry is not None else None
        except BaseException:
            before_commit.restore()
            before_commit.check()
            raise
        cache_enabled = self.cache_source is not None and self.history_weight > 0.
        cache_valid = cache_enabled and ell > self.eps
        vector = pending['g0'] if self.cache_source == 'base' else pending['U']
        if cache_valid:
            vector, delta = _clone(vector), _clone(displacement)
            self._history = {'vector': vector, 'displacement': delta, 'length': ell,
                             'vector_sha256': _digest(vector), 'displacement_sha256': _digest(delta),
                             'after_sha256': _digest(self.params), 'layout': self._layout,
                             'cache_source': self.cache_source}
        else:
            self._history = None
        self.real_updates += 1
        result = {
            'sgd_updates': 1, 'natural_sgd_exact': exact, 'base_sgd_exact': exact,
            'base_sgd_max_abs_error': 0., 'actual_parameter_step_norm': ell,
            'actual_step_displacement_sha256': _digest(displacement),
            'history_cache_valid_after_update': cache_valid,
            'history_cache_reason_after_update': ('valid_actual_FP32_displacement' if cache_valid else
                                                  'history_disabled' if not cache_enabled else 'zero_actual_displacement'),
            'history_cached_vector_sha256': _digest(vector) if cache_valid else None,
            'history_cached_vector_norm': _norm(vector) if cache_valid else None,
            'history_cached_vector_source': self.cache_source if cache_valid else None,
            'history_cached_raw_gradient_sha256': _digest(vector) if cache_valid and self.cache_source == 'base' else None,
            'history_cache_uses_raw_gradient_not_U': self.cache_source == 'base',
            'history_cache_uses_full_final_vector': self.cache_source == 'final',
            'history_cache_keeps_full_amplitude': cache_valid,
            'real_updates': self.real_updates,
        }
        if effective is not None:
            result['actual_effective_delta'] = effective
        pending['record'].update(result)
        self._pending = None
        return result
