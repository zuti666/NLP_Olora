"""One SGD prediction and direct historical/future response fusion.

There is no Fisher, replay buffer, weight merge, projection, HVP, QR, clipping,
or update normalization. The only persistent history is the last raw gradient
and the *actual* trainable-parameter displacement of the last SGD update.
"""
import hashlib
import math

import torch

from state_transaction import State, backend_context, rng_get, rng_set, rng_equal


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
    value = float(torch.stack([(x.double() * y.double()).sum() for x, y in zip(a, b)]).sum().item())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite history-response inner product')
    return value


def _norm(a):
    return math.sqrt(max(0., _dot(a, a)))


def _cos(a, b, eps):
    den = _norm(a) * _norm(b)
    return _dot(a, b) / den if den > eps else None


def _finite(values, label):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(label + ' has nonfinite tensors')


class HistoryResponseStep:
    """State-transactional two-gradient response step; no autograd graph persists.

    loss_closure(inputs, targets) must return (scalar loss, logits) and must not
    backward, update the optimizer, or intentionally modify model parameters.
    compute returns a dict; apply_update is the sole real optimizer update.
    Construct a new instance whenever the task expands the classifier.
    """
    def __init__(self, model, params, geometry, config):
        self.model, self.params, self.geometry = model, list(params), geometry
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Nonempty unique trainable parameters required')
        if set(map(id, self.params)) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Parameter scope must cover all and only trainable parameters')
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError('History-response training uses actual FP32 coordinates')
        if geometry is not None and set(map(id, geometry.params)) != set(map(id, self.params)):
            raise ValueError('Effective geometry does not cover exactly all trainables')
        self.direction = config.get('direction', 'history_fused')
        if self.direction not in ('history_fused', 'future_only'):
            raise ValueError('Unknown history-response direction')
        self.history_weight = float(config.get('history_weight', .5 if self.direction == 'history_fused' else 0.))
        self.eps = float(config.get('eps', 1e-12))
        self.backend = config.get('backend', 'native')
        if not math.isfinite(self.history_weight) or not 0. <= self.history_weight <= 1.:
            raise ValueError('History weight must be finite in [0,1]')
        if self.direction == 'future_only' and self.history_weight != 0.:
            raise ValueError('future_only requires history_weight=0')
        if not math.isfinite(self.eps) or self.eps <= 0.:
            raise ValueError('eps must be finite and positive')
        if self.backend != 'native':
            raise ValueError('The registered history-response protocol requires native backend')
        self._layout = self._layout_now()
        self._history = None
        self._pending = None
        self.fallback_count = 0
        self.history_reset_count = 0
        self.real_updates = 0

    def _layout_now(self):
        return tuple((id(p), tuple(p.shape), str(p.dtype), str(p.device), p.requires_grad) for p in self.params)

    def reset_history(self, reason='explicit_reset'):
        self._history = None
        self.history_reset_count += 1
        self.last_reset_reason = str(reason)

    def _check_layout(self):
        if (self._layout_now() != self._layout or
                set(map(id, self.params)) != {id(p) for p in self.model.parameters() if p.requires_grad}):
            self.reset_history('trainable_layout_changed')
            raise ValueError('Trainable layout changed: construct a new HistoryResponseStep for this task')

    def compute(self, inputs, targets, loss_closure, current_lr):
        if self._pending is not None:
            raise RuntimeError('The previously computed step has not been applied')
        self._check_layout()
        eta = float(current_lr)
        if not math.isfinite(eta) or eta < 0.:
            raise ValueError('current_lr must be finite and nonnegative')
        entry = State(self.model)
        base = _clone(self.params)
        rng0 = entry.rng
        calls = 0
        outputs = {}

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

        try:
            loss0, logits0, g, rng1 = probe(base)
            g_norm = _norm(g)
            ell_f = eta * g_norm
            if not math.isfinite(ell_f):
                raise FloatingPointError('Nonfinite predicted displacement length')
            future = tuple(v.clone().add_(grad, alpha=-eta) for v, grad in zip(base, g))
            _finite(future, 'Predicted SGD point')
            distinguishable = any(not torch.equal(v, f) for v, f in zip(base, future))
            future_actual_delta = tuple(f - v for f, v in zip(future, base))
            future_actual_length = _norm(future_actual_delta)
            history_reason = 'no_previous_update'
            h = self._history
            if self.direction == 'future_only' or self.history_weight == 0.:
                history_reason = 'history_disabled'
                h = None
            elif h is not None:
                if h['after_sha256'] != _digest(base):
                    self.reset_history('parameter_continuity_changed')
                    history_reason = 'parameter_continuity_changed'
                    h = None
                elif h['layout'] != self._layout or h['length'] <= self.eps:
                    self.reset_history('invalid_previous_displacement_or_layout')
                    history_reason = 'invalid_previous_displacement_or_layout'
                    h = None
                else:
                    history_reason = 'valid_previous_actual_update'
            ell_b = h['length'] if h is not None else None
            fallback_reason = None
            if ell_f <= self.eps:
                fallback_reason = 'predicted_displacement_length_at_or_below_eps'
            elif not distinguishable:
                fallback_reason = 'predicted_point_bitwise_indistinguishable_in_FP32'
            if fallback_reason is not None:
                self.fallback_count += 1
                gf = None
                future_loss = None
                U = _clone(g)
                future_response = tuple(torch.zeros_like(v) for v in g)
                history_response = tuple(torch.zeros_like(v) for v in g)
                w_eff = 0.
                lam = ell_f
                coefficient = 0.
                used_history = False
            else:
                future_loss, _, gf, _ = probe(future)
                future_response = tuple(gi - fi for gi, fi in zip(g, gf))
                used_history = h is not None
                w_eff = self.history_weight if used_history else 0.
                lam = ell_f
                coefficient = lam * w_eff / ell_b if used_history else 0.
                history_response = (tuple(old - cur for old, cur in zip(h['raw_gradient'], g))
                                    if used_history else tuple(torch.zeros_like(v) for v in g))
                if used_history:
                    U = tuple(gi + (1. - w_eff) * fr + coefficient * hr
                              for gi, fr, hr in zip(g, future_response, history_response))
                else:
                    U = tuple(2. * gi - fi for gi, fi in zip(g, gf))
            _finite(U, 'Fused update direction')
        finally:
            entry.restore()
            restored = entry.check()
        # All state has returned to entry; only the original g0 stochastic stream
        # is committed once. Future evaluation's random draws never advance it.
        rng_set(rng1)
        if not rng_equal(rng_get(), rng1):
            raise AssertionError('Failed to commit exactly the base-probe RNG state')
        U = _clone(U)
        u_norm = _norm(U)
        future_component = tuple((1. - w_eff) * v for v in future_response)
        history_component = tuple(coefficient * v for v in history_response)
        record = {
            'direction': self.direction,
            'formula': 'g + (1-w_eff)*(g-gf) + lambda*w_eff/ell_b*(g_prev-g)',
            'lambda': lam, 'lambda_definition': 'ell_f=current_lr*global_parameter_L2_norm(g)',
            'history_weight': self.history_weight, 'effective_history_weight': w_eff,
            'current_lr': eta, 'ell_f': ell_f, 'ell_b': ell_b,
            'ell_f_over_ell_b': ell_f / ell_b if ell_b is not None else None,
            'history_coefficient': coefficient,
            'future_actual_parameter_displacement_norm': future_actual_length,
            'predicted_point_FP32_distinguishable': distinguishable,
            'history_used': used_history, 'history_reason': history_reason,
            'history_cache_raw_gradient_sha256': h['raw_gradient_sha256'] if h is not None else None,
            'history_cache_actual_displacement_sha256': h['displacement_sha256'] if h is not None else None,
            'history_cache_after_sha256': h['after_sha256'] if h is not None else None,
            'base_parameter_sha256': _digest(base), 'future_parameter_sha256': _digest(future),
            'raw_gradient_sha256': _digest(g), 'future_gradient_sha256': _digest(gf) if gf is not None else None,
            'U_sha256': _digest(U),
            'gradient_norms': {'g': g_norm, 'gf': _norm(gf) if gf is not None else None,
                               'U': u_norm, 'future_response': _norm(future_response),
                               'history_response': _norm(history_response)},
            'u_g_ratio': u_norm / g_norm if g_norm > self.eps else None,
            'g_dot_U': _dot(g, U), 'g_U_cosine': _cos(g, U, self.eps),
            'u_v_cosine': _cos(g, tuple(-d for d in h['displacement']), self.eps) if used_history else None,
            'cos_u_v_definition': 'u=g/||g||, v=-actual_previous_displacement/ell_b',
            'response_cosine': _cos(future_response, history_response, self.eps) if used_history else None,
            'response_cosine_definition': 'cos(g-gf, g_prev-g)',
            'future_component_relative_norm': _norm(future_component) / g_norm if g_norm > self.eps else None,
            'history_component_relative_norm': _norm(history_component) / g_norm if g_norm > self.eps else None,
            'g_grad_norm': g_norm, 'U_grad_norm': u_norm,
            'loss0': loss0, 'loss_future': future_loss,
            'closure_count': calls, 'gradient_calls': calls, 'ordinary_gradient_calls': calls,
            'training_ce_gradient_passes': calls, 'training_forward_count': calls,
            'fallback': fallback_reason is not None, 'fallback_reason': fallback_reason,
            'fallback_count': self.fallback_count, 'history_reset_count': self.history_reset_count,
            'state_restore_checks': restored, 'only_base_R1_rng_advance': True,
            'fisher_estimation_passes': 0, 'weight_merge_updates': 0, 'projection_calls': 0,
            'hvp_calls': 0, 'qr_calls': 0, 'create_graph_calls': 0,
            'gradient_clipping_calls': 0, 'final_norm_rescale_calls': 0,
        }
        context = {'base': base, 'future': future, 'g': g, 'gf': gf, 'U': U,
                   'rng0': rng0, 'rng1': rng1, 'entry_state': entry,
                   'loss0': loss0, 'loss_future': future_loss}
        self._pending = {'eta': eta, 'base': base, 'g': g, 'U': U, 'record': record,
                         'context': context, 'geometry_before': self.geometry.capture() if self.geometry is not None else None}
        return {'loss0': loss0, 'logits0': logits0, 'U': U, 'record': record, 'context': context}

    def _portable_layout(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return [{'name': names[id(p)], 'shape': list(p.shape), 'dtype': str(p.dtype)} for p in self.params]

    def state_dict(self):
        """Portable raw-gradient and actual-displacement history, without Python ids.

        A checkpoint is legal only after a completed real update. Model tensors,
        optimizer, scheduler and global RNG are saved by the outer trainer.
        """
        if self._pending is not None:
            raise RuntimeError('Cannot checkpoint between compute and apply_update')
        self._check_layout()
        history = None
        if self._history is not None:
            h = self._history
            history = {'raw_gradient': tuple(v.detach().cpu().clone() for v in h['raw_gradient']),
                       'displacement': tuple(v.detach().cpu().clone() for v in h['displacement']),
                       'length': h['length'], 'raw_gradient_sha256': h['raw_gradient_sha256'],
                       'displacement_sha256': h['displacement_sha256'], 'after_sha256': h['after_sha256'],
                       'valid': True, 'layout': self._portable_layout()}
        return {'format': 'HistoryResponseStep_v1', 'direction': self.direction,
                'history_weight': self.history_weight, 'eps': self.eps, 'backend': self.backend,
                'layout': self._portable_layout(), 'history': history,
                'fallback_count': self.fallback_count, 'history_reset_count': self.history_reset_count,
                'real_updates': self.real_updates, 'last_reset_reason': getattr(self, 'last_reset_reason', None),
                'cache_coordinate': 'raw_gradient_and_actual_FP32_trainable_parameter_displacement'}

    def load_state_dict(self, state):
        """Restore history only against the exact saved current model parameters.

        Load the model checkpoint before this call. An incompatible task/classifier
        or model point is rejected rather than silently misusing the cached history.
        """
        if self._pending is not None:
            raise RuntimeError('Cannot restore history while an update is pending')
        self._check_layout()
        if state.get('format') != 'HistoryResponseStep_v1':
            raise ValueError('Unknown history-response state format')
        for key in ('direction', 'history_weight', 'eps', 'backend'):
            if state.get(key) != getattr(self, key):
                raise ValueError('Saved history-response configuration differs: ' + key)
        if state.get('layout') != self._portable_layout():
            raise ValueError('Saved history-response parameter names/shapes/dtypes differ')
        counters = {}
        for key in ('fallback_count', 'history_reset_count', 'real_updates'):
            value = state.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid saved history-response counter: ' + key)
            counters[key] = value
        h = state.get('history')
        loaded = None
        if h is not None:
            if self.direction != 'history_fused' or self.history_weight <= 0.:
                raise ValueError('History-disabled configuration cannot restore a history cache')
            if h.get('valid') is not True or h.get('layout') != self._portable_layout():
                raise ValueError('Saved history cache is invalid or has incompatible layout')
            vectors = {}
            for key in ('raw_gradient', 'displacement'):
                values = h.get(key)
                if not isinstance(values, (list, tuple)) or len(values) != len(self.params):
                    raise ValueError('Invalid saved history tensor count: ' + key)
                if any(not torch.is_tensor(v) or v.shape != p.shape or v.dtype != p.dtype
                       for v, p in zip(values, self.params)):
                    raise ValueError('Invalid saved history tensor shape/dtype: ' + key)
                vectors[key] = tuple(v.detach().to(device=p.device).clone() for v, p in zip(values, self.params))
                _finite(vectors[key], 'Saved ' + key)
            raw_hash = _digest(vectors['raw_gradient'])
            delta_hash = _digest(vectors['displacement'])
            if raw_hash != h.get('raw_gradient_sha256') or delta_hash != h.get('displacement_sha256'):
                raise ValueError('Saved history tensor digest mismatch')
            length = _norm(vectors['displacement'])
            saved_length = float(h.get('length', float('nan')))
            if length <= self.eps or not math.isfinite(saved_length) or not math.isclose(length, saved_length, rel_tol=1e-12, abs_tol=0.):
                raise ValueError('Saved history length disagrees with actual displacement')
            if _digest(self.params) != h.get('after_sha256'):
                raise ValueError('Load the exact saved current model before restoring its history')
            loaded = {'raw_gradient': vectors['raw_gradient'], 'displacement': vectors['displacement'],
                      'length': length, 'raw_gradient_sha256': raw_hash, 'displacement_sha256': delta_hash,
                      'after_sha256': h['after_sha256'], 'layout': self._layout}
        self._history = loaded
        for key, value in counters.items():
            setattr(self, key, value)
        self.last_reset_reason = state.get('last_reset_reason')

    @torch.no_grad()
    def apply_update(self, current_lr, optimizer):
        if self._pending is None:
            raise RuntimeError('compute must precede the single real SGD update')
        self._check_layout()
        pending = self._pending
        eta = float(current_lr)
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
            self.reset_history('parameters_changed_between_compute_and_update')
            raise AssertionError('Parameters changed between compute and the real update')
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
        ell_b = _norm(displacement)
        cache_enabled = self.direction == 'history_fused' and self.history_weight > 0.
        cache_valid = cache_enabled and ell_b > self.eps
        if cache_valid:
            raw = _clone(pending['g'])
            delta = _clone(displacement)
            self._history = {'raw_gradient': raw, 'displacement': delta, 'length': ell_b,
                             'raw_gradient_sha256': _digest(raw), 'displacement_sha256': _digest(delta),
                             'after_sha256': _digest(self.params), 'layout': self._layout}
        else:
            self._history = None
        self.real_updates += 1
        result = {'sgd_updates': 1, 'natural_sgd_exact': exact, 'base_sgd_exact': exact,
                  'base_sgd_max_abs_error': 0., 'actual_parameter_step_norm': ell_b,
                  'actual_step_displacement_sha256': _digest(displacement),
                  'history_cache_valid_after_update': cache_valid,
                  'history_cache_reason_after_update': ('valid_actual_FP32_displacement' if cache_valid else
                                                       'history_disabled' if not cache_enabled else 'zero_actual_displacement'),
                  'history_cached_raw_gradient_sha256': _digest(pending['g']) if cache_valid else None,
                  'history_cache_uses_raw_gradient_not_U': True,
                  'real_updates': self.real_updates}
        if self.geometry is not None:
            result['actual_effective_delta'] = self.geometry.actual_delta(pending['geometry_before'])
        pending['record'].update(result)
        self._pending = None
        return result
