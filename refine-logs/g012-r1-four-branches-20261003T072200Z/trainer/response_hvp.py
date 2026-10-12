"""Fixed, arbitrary displacement HVPs with transactional state restoration.

This module never differentiates endpoint construction or normalisation.  The
vector already has its physical factor-coordinate length; no radius is applied.
All loss evaluations in a comparison use one attention backend and RNG state.
"""
import contextlib
import copy
import math
import random
import time

import numpy as np
import torch


def rng_get():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_set(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3]: torch.cuda.set_rng_state_all(state[3])


def rng_equal(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2]) and len(a[3]) == len(b[3])
            and all(torch.equal(v, w) for v, w in zip(a[3], b[3])))


def _cache_name(name):
    return ('cache' in name.lower() or name == 'backup_momentum'
            or name.startswith('delta_w_') or name.startswith('external_delta_w_')
            or name == '_register_delta_hook')


def _clone(value):
    if isinstance(value, torch.nn.Parameter): return value
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list): return [_clone(v) for v in value]
    if isinstance(value, tuple): return tuple(_clone(v) for v in value)
    return copy.deepcopy(value)


def _equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(_equal(v, w) for v, w in zip(a, b))
    return a == b


class State:
    """Entry state; frozen values are protected by immutable version counters.

    Trainable tensors, every gradient, registered buffers, module modes, BN
    controls, named forward caches, fused-attention flags and all RNG streams
    are restored.  Copying the entire frozen ViT is avoided on every update.
    """
    def __init__(self, model):
        self.model = model
        self.trainable = [(p, p.detach().clone()) for p in model.parameters() if p.requires_grad]
        self.frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
        self.grads = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]
        self.modes = [(m, m.training) for m in model.modules()]
        self.controls = [(m, {k: _clone(v) for k, v in m.__dict__.items()
                              if _cache_name(k) or k in ('momentum', 'fused_attn')}) for m in model.modules()]
        self.rng = rng_get()

    @torch.no_grad()
    def restore(self):
        if any(p._version != version for p, version in self.frozen):
            raise AssertionError('HVP mutated a frozen parameter')
        for p, value in self.trainable: p.copy_(value)
        for p, grad in self.grads: p.grad = None if grad is None else grad.clone()
        for b, value in self.buffers: b.copy_(value)
        for m, mode in self.modes: m.training = mode
        for m, attrs in self.controls:
            for key in list(m.__dict__):
                if (_cache_name(key) or key in ('momentum', 'fused_attn')) and key not in attrs:
                    delattr(m, key)
            for key, value in attrs.items(): setattr(m, key, _clone(value))
        rng_set(self.rng)

    def check(self):
        checks = {
            'trainable_parameters_exact': all(torch.equal(p, v) for p, v in self.trainable),
            'frozen_versions_exact': all(p._version == version for p, version in self.frozen),
            'gradients_exact': all((p.grad is None and v is None) or
                                  (p.grad is not None and v is not None and torch.equal(p.grad, v)) for p, v in self.grads),
            'buffers_exact': all(torch.equal(b, v) for b, v in self.buffers),
            'modes_exact': all(m.training == mode for m, mode in self.modes),
            'controls_caches_exact': all({k for k in m.__dict__ if _cache_name(k) or k in ('momentum', 'fused_attn')} == set(attrs)
                                        and all(_equal(getattr(m, k), v) for k, v in attrs.items()) for m, attrs in self.controls),
            'rng_exact': rng_equal(self.rng, rng_get()),
        }
        if not all(checks.values()): raise AssertionError('HVP entry restore failed: ' + repr(checks))
        return checks


capture_state = State


@contextlib.contextmanager
def backend_context(model, backend):
    if backend == 'native':
        yield
    elif backend == 'math':
        fused = [(m, m.fused_attn) for m in model.modules() if hasattr(m, 'fused_attn')]
        try:
            # timm's explicit attention is differentiable twice.  Dispatch flags
            # also cover any direct SDPA use remaining in the model.
            for m, _ in fused: m.fused_attn = False
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                    enable_mem_efficient=False, enable_cudnn=False):
                yield
        finally:
            for m, value in fused: m.fused_attn = value
    else:
        raise ValueError('backend must be native or math')


attention_backend = backend_context


def _validate_vector(params, vector, label):
    if len(params) != len(vector): raise ValueError(label + ' tensor count mismatch')
    for p, v in zip(params, vector):
        if v.requires_grad or v.grad_fn is not None: raise ValueError(label + ' must be detached')
        if v.shape != p.shape or v.dtype != p.dtype or v.device != p.device:
            raise ValueError(label + ' shape/dtype/device mismatch')
        if not bool(torch.isfinite(v).all()): raise FloatingPointError(label + ' nonfinite')


def vector_norm(values):
    return math.sqrt(max(0., sum(float(v.detach().double().square().sum()) for v in values)))


def compare(values, reference):
    if len(values) != len(reference): raise ValueError('comparison tensor count mismatch')
    na, nb = vector_norm(values), vector_norm(reference)
    dot = sum(float((a.detach().double() * b.detach().double()).sum()) for a, b in zip(values, reference))
    error = vector_norm([a-b for a, b in zip(values, reference)])
    return {'first_norm': na, 'reference_norm': nb, 'relative_l2': error/nb if nb > 1e-30 else None,
            'cosine': dot/(na*nb) if na > 1e-30 and nb > 1e-30 else None,
            'norm_ratio': na/nb if nb > 1e-30 else None,
            'max_abs': max(float((a-b).abs().max()) for a, b in zip(values, reference)),
            'degenerate': na <= 1e-30 or nb <= 1e-30}


def assert_same_gradient(observed, reference, tolerance=2e-5):
    stats = compare(observed, reference)
    if stats['relative_l2'] is None:
        if stats['max_abs'] > 1e-10: raise AssertionError('Degenerate same-point gradient mismatch: ' + repr(stats))
    elif stats['relative_l2'] > tolerance:
        raise AssertionError('Same-backend same-point CE gradient mismatch: ' + repr(stats))
    return stats


def gradient_at(model, params, point, closure, rng_state=None, backend='math', need_hvp=False, direction=None):
    """Evaluate one scalar closure at a detached parameter point transactionally.

    Returns (loss_float, detached_grads, detached_Hd_or_None, report).  No caller
    graph survives parameter restoration.  No differentiation through point/d.
    """
    params = tuple(params)
    if not params or len({id(p) for p in params}) != len(params) or any(not p.requires_grad for p in params):
        raise ValueError('Expected unique trainable parameter coordinates')
    if point is not None: _validate_vector(params, point, 'point')
    if need_hvp:
        if direction is None: raise ValueError('HVP direction missing')
        _validate_vector(params, direction, 'direction')
    live = State(model); started = time.perf_counter(); report = {}
    try:
        with torch.no_grad():
            if point is not None:
                for p, value in zip(params, point): p.copy_(value)
        if rng_state is not None: rng_set(rng_state)
        with backend_context(model, backend), torch.enable_grad():
            loss = closure()
            if not torch.is_tensor(loss) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError('closure must return one finite scalar loss')
            graph = torch.autograd.grad(loss, params, create_graph=need_hvp, allow_unused=False)
            gradients = tuple(v.detach().clone() for v in graph)
            if any(not bool(torch.isfinite(v).all()) for v in gradients): raise FloatingPointError('Nonfinite CE gradient')
            hvp = None
            if need_hvp:
                inner = sum((g * d.detach()).sum() for g, d in zip(graph, direction))
                second = torch.autograd.grad(inner, params, allow_unused=True) if inner.requires_grad else (None,) * len(params)
                hvp = tuple(torch.zeros_like(p) if h is None else h.detach().clone() for p, h in zip(params, second))
                if any(not bool(torch.isfinite(v).all()) for v in hvp): raise FloatingPointError('Nonfinite HVP')
            report = {'backend': backend, 'ce': float(loss.detach()), 'ce_forward_count': 1,
                      'first_derivative_passes': 1, 'second_derivative_passes': int(need_hvp),
                      'direction_detached': bool(need_hvp), 'direction_norm': vector_norm(direction) if need_hvp else None,
                      'direction_radius_multiplier': 1., 'gradient_norm': vector_norm(gradients),
                      'hvp_norm': vector_norm(hvp) if hvp is not None else None,
                      'seconds': time.perf_counter()-started, 'fallback_used': False}
            report['attention_backend_flags'] = {
                'flash': torch.backends.cuda.flash_sdp_enabled(),
                'math': torch.backends.cuda.math_sdp_enabled(),
                'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                'cudnn': torch.backends.cuda.cudnn_sdp_enabled(),
                'fused_attn_true_modules': sum(bool(m.fused_attn) for m in model.modules() if hasattr(m, 'fused_attn'))}
            return float(loss.detach()), gradients, hvp, report
    finally:
        live.restore(); restored = live.check()
        if report: report['entry_state_restore'] = restored


def arbitrary_Hd(model, params, point, direction, closure, rng_state=None, backend='math',
                 reference_gradient=None, agreement_tol=2e-5):
    """Arbitrary detached H(x)d, validated against a same-backend CE gradient."""
    reference_report = None
    if reference_gradient is None:
        _, reference_gradient, _, reference_report = gradient_at(model, params, point, closure, rng_state, backend)
    _, gradient, hvp, report = gradient_at(model, params, point, closure, rng_state, backend,
                                          need_hvp=True, direction=direction)
    report['same_point_gradient'] = assert_same_gradient(gradient, reference_gradient, agreement_tol)
    report['ordinary_reference_passes'] = int(reference_report is not None)
    report['ordinary_reference_report'] = reference_report
    report['hvp_graph_forward_count'] = 1
    report['ordinary_reference_forward_count'] = int(reference_report is not None)
    report['total_ce_forward_count'] = 1 + int(reference_report is not None)
    report['total_first_derivative_passes'] = 1 + int(reference_report is not None)
    report['status'] = 'passed'
    return hvp, report


def response_observer(context, response):
    """FourProbeStep observer: (U,report); supports hstart and hmid only."""
    if response not in ('hstart', 'hmid'): raise ValueError('Unknown HVP response')
    params = context['params']; points = context['points']; grads = context['grads']
    d12 = tuple((p2-p1).detach() for p1, p2 in zip(points['phi1_plus'], points['phi2']))
    point = points['phi1_plus'] if response == 'hstart' else tuple(((p1+p2)*.5).detach() for p1, p2 in zip(points['phi1_plus'], points['phi2']))
    reference = grads['g1_plus'] if response == 'hstart' else None
    hvp, report = arbitrary_Hd(context['model'], params, point, d12, context['closure'],
                              context['rng_state'], context['backend'], reference)
    sec = tuple(b-a for a, b in zip(grads['g1_plus'], grads['g2']))
    report.update(response=response, hvp_vs_sec=compare(hvp, sec), d12_source='actual_phi2_minus_phi1_snapshots')
    return tuple((g+h).detach() for g, h in zip(grads['g0'], hvp)), report


def center_difference_gate(model, params, point, direction, closure, rng_state=None,
                           backend='math', h_grid=(.1,.03,.01,.003,.001)):
    """Frozen numerical gate; finite-r Sec agreement is intentionally irrelevant."""
    hvp, info = arbitrary_Hd(model, params, point, direction, closure, rng_state, backend)
    scan = []
    for h in h_grid:
        plus = tuple((x+h*d).detach() for x, d in zip(point, direction))
        minus = tuple((x-h*d).detach() for x, d in zip(point, direction))
        _, gp, _, _ = gradient_at(model, params, plus, closure, rng_state, backend)
        _, gm, _, _ = gradient_at(model, params, minus, closure, rng_state, backend)
        fd = tuple((a-b)/(2*h) for a, b in zip(gp, gm))
        stats = compare(fd, hvp)
        ok = stats['relative_l2'] is not None and stats['relative_l2'] <= 5e-3 and stats['cosine'] is not None and stats['cosine'] >= .999
        scan.append({'h': h, 'central_vs_hvp': stats, 'accepted': ok})
    stable = any(a['accepted'] and b['accepted'] for a, b in zip(scan, scan[1:]))
    degenerate = vector_norm(hvp) <= 1e-30
    report = {'status': 'passed' if stable else 'failed', 'hvp': info, 'scan': scan,
              'two_adjacent_h_accepted': stable, 'hvp_near_zero': degenerate,
              'criteria': {'relative_l2_max': .005, 'cosine_min': .999, 'same_point_gradient_relative_max': .00002},
              'note': 'Near-zero HVP requires separate absolute-error/float64 review; no automatic pass.'}
    return report
