"""Transactional state and attention contexts for four ordinary gradient probes."""
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
            raise AssertionError('A probe mutated a frozen parameter')
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
        if not all(checks.values()): raise AssertionError('Probe entry restore failed: ' + repr(checks))
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


