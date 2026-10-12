"""KFAC precision history in effective LoRA weight coordinates.

Each block is a *sum* of Kronecker products, not a product of summed factors.
For D with shape (out, in), its quadratic form is tr(D.T @ G @ D @ C).
Anchors, lambda, and the conventional factor 1/2 belong to the caller.
"""

import math
from collections.abc import Mapping

import torch


class KFACEWCState:
    """Detached CPU factors and an optional isotropic precision per block.

    Treat ``blocks`` and ``isotropic`` as immutable; ``accumulate`` creates a
    new state. Factors are FP32, except FP64 inputs retain their precision.
    Device copies are cached lazily for repeated differentiable penalties.
    """

    VERSION = 1

    def __init__(self, blocks=None, isotropic=None, tasks=0):
        self.blocks = {} if blocks is None else blocks
        self.isotropic = {} if isotropic is None else isotropic
        self.tasks = tasks
        self._cache = {}

    @property
    def shapes(self):
        return {
            key: (terms[0]["G"].shape[0], terms[0]["C"].shape[0])
            for key, terms in self.blocks.items()
        }

    @staticmethod
    def _nonnegative(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite nonnegative number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite nonnegative number") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        return result

    @staticmethod
    def _factor(value, name, validate_psd=False):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor")
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
            raise ValueError(f"{name} must be a nonempty square matrix")
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        result = value.detach().to(device="cpu", dtype=dtype).clone()
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} contains nonfinite values")
        scale = float(result.abs().max())
        # Relative tolerance allows normal FP32 covariance roundoff, including
        # rank deficient matrices, without accepting materially indefinite data.
        tolerance = max(scale * 1e-6, torch.finfo(dtype).tiny)
        if float((result - result.T).abs().max()) > tolerance:
            raise ValueError(f"{name} must be symmetric")
        result = (result + result.T) * 0.5
        if float(result.diag().min()) < -tolerance:
            raise ValueError(f"{name} has a negative diagonal")
        if validate_psd and scale > 0:
            shifted = result.double() + torch.eye(result.shape[0], dtype=torch.float64) * tolerance
            _, info = torch.linalg.cholesky_ex(shifted, check_errors=False)
            if int(info) != 0:
                raise ValueError(f"{name} is not positive semidefinite")
        return result

    @classmethod
    def from_estimate(cls, factors, normalize=True, beta=0.0, eps=1e-12):
        """Build one task's precision, optionally with unit global trace.

        With normalization enabled, damping produces
        (1-beta) * F / tr(F) + (beta/P) * I over all P covered weights.
        The estimator's ``C`` name and conventional ``A`` alias are accepted.
        """
        if not isinstance(factors, Mapping) or not factors:
            raise ValueError("KFAC estimate must contain at least one block")
        beta = cls._nonnegative(beta, "beta")
        if beta > 1:
            raise ValueError("beta must be in [0, 1]")
        if beta > 0 and not normalize:
            raise ValueError("isotropic damping requires normalized KFAC")
        eps = cls._nonnegative(eps, "eps")
        blocks = {}
        for key, estimate in factors.items():
            if not isinstance(key, str) or not key or not isinstance(estimate, Mapping):
                raise ValueError("KFAC blocks require nonempty string keys and factor mappings")
            if "G" not in estimate or ("C" not in estimate and "A" not in estimate):
                raise ValueError(f"Missing G/C factors for {key}")
            blocks[key] = [{
                "G": cls._factor(estimate["G"], f"{key}.G"),
                "C": cls._factor(estimate.get("C", estimate.get("A")), f"{key}.C"),
                "weight": 1.0,
            }]
        state = cls(blocks, {key: 0.0 for key in blocks}, tasks=1)
        if normalize:
            state._scale_to_unit_trace(eps)
        if beta:
            count = sum(out_dim * in_dim for out_dim, in_dim in state.shapes.values())
            for key, terms in state.blocks.items():
                terms[0]["weight"] *= 1.0 - beta
                state.isotropic[key] = beta / count
        return state

    def _scale_to_unit_trace(self, eps):
        total = self.trace()
        if not math.isfinite(total) or total <= eps:
            raise ValueError(f"Cannot normalize KFAC with trace {total} <= eps {eps}")
        for key, terms in self.blocks.items():
            for term in terms:
                term["weight"] /= total
            self.isotropic[key] /= total
        self._cache.clear()

    def accumulate(self, new, gamma=1.0, normalize=True, eps=1e-12):
        """Return gamma * self + new, optionally renormalized globally.

        No factors are merged: merging G and C separately creates spurious
        cross-task Kronecker terms. The caller supplies its current anchor.
        """
        if not isinstance(new, KFACEWCState) or not new.blocks:
            raise ValueError("new must be a nonempty KFACEWCState")
        gamma = self._nonnegative(gamma, "gamma")
        eps = self._nonnegative(eps, "eps")
        if self.blocks and self.shapes != new.shapes:
            raise ValueError("KFAC history keys/shapes do not match the new estimate")
        blocks = {}
        isotropic = {}
        for key, terms in new.blocks.items():
            blocks[key] = []
            for source, scale in ((self.blocks.get(key, []), gamma), (terms, 1.0)):
                for term in source:
                    # CPU tensors are immutable and can be shared safely.
                    blocks[key].append({
                        "G": term["G"], "C": term["C"],
                        "weight": term["weight"] * scale,
                    })
            isotropic[key] = gamma * self.isotropic.get(key, 0.0) + new.isotropic[key]
        result = type(self)(blocks, isotropic, self.tasks + new.tasks)
        if normalize:
            result._scale_to_unit_trace(eps)
        return result

    def trace(self):
        """Global trace of the implied precision; no Kronecker allocation."""
        total = 0.0
        for key, terms in self.blocks.items():
            out_dim, in_dim = self.shapes[key]
            total += self.isotropic[key] * out_dim * in_dim
            for term in terms:
                total += term["weight"] * float(term["G"].double().trace()) * float(term["C"].double().trace())
        return total

    def diagonal(self):
        """Return block diagonals in D's matrix layout for legacy bookkeeping."""
        result = {}
        for key, terms in self.blocks.items():
            dtype = torch.float64 if any(
                t["G"].dtype == torch.float64 or t["C"].dtype == torch.float64 for t in terms
            ) else torch.float32
            value = torch.full(self.shapes[key], self.isotropic[key], dtype=dtype)
            for term in terms:
                value.add_(torch.outer(term["G"].diag().to(dtype), term["C"].diag().to(dtype)), alpha=term["weight"])
            result[key] = value
        return result

    def penalty(self, deltas):
        """Differentiable sum tr(D.T G D C) plus isotropic terms.

        Deliberately excludes lambda and 1/2. Half-precision D is promoted to
        FP32 while retaining its autograd graph. Every history key is required.
        """
        if not isinstance(deltas, Mapping) or set(deltas) != set(self.blocks):
            raise ValueError("KFAC delta keys must exactly match all history blocks")
        total = None
        for key, terms in self.blocks.items():
            delta = deltas[key]
            if not isinstance(delta, torch.Tensor) or not delta.is_floating_point() or tuple(delta.shape) != self.shapes[key]:
                raise ValueError(f"KFAC delta {key} has the wrong shape or dtype")
            dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
            value = delta.to(dtype=dtype)
            block_loss = value.square().sum() * self.isotropic[key]
            for index, term in enumerate(terms):
                if term["weight"] == 0.0:
                    continue
                cache_key = (key, index, value.device, dtype)
                if cache_key not in self._cache:
                    self._cache[cache_key] = (
                        term["G"].to(device=value.device, dtype=dtype),
                        term["C"].to(device=value.device, dtype=dtype),
                    )
                g_factor, c_factor = self._cache[cache_key]
                block_loss = block_loss + (value * (g_factor @ value @ c_factor)).sum() * term["weight"]
            total = block_loss if total is None else total + block_loss
        return torch.tensor(0.0) if total is None else total

    def state_dict(self):
        """Portable tensor-only CPU payload; excludes device cache and anchors."""
        return {
            "version": self.VERSION,
            "tasks": self.tasks,
            "blocks": {
                key: [{"G": t["G"].detach().cpu().clone(), "C": t["C"].detach().cpu().clone(), "weight": t["weight"]} for t in terms]
                for key, terms in self.blocks.items()
            },
            "isotropic": dict(self.isotropic),
        }

    @classmethod
    def from_state_dict(cls, payload, shapes=None):
        """Validate checkpoint schema, keys, shapes, finiteness, and PSD."""
        if not isinstance(payload, Mapping) or payload.get("version") != cls.VERSION:
            raise ValueError("Unsupported KFAC history checkpoint version")
        if not {"tasks", "blocks", "isotropic"}.issubset(payload):
            raise ValueError("Incomplete KFAC history checkpoint")
        tasks = payload["tasks"]
        if isinstance(tasks, bool) or not isinstance(tasks, int) or tasks < 0:
            raise ValueError("KFAC tasks must be a nonnegative integer")
        raw_blocks, raw_isotropic = payload["blocks"], payload["isotropic"]
        if not isinstance(raw_blocks, Mapping) or not isinstance(raw_isotropic, Mapping):
            raise ValueError("KFAC blocks/isotropic must be mappings")
        if set(raw_blocks) != set(raw_isotropic):
            raise ValueError("KFAC blocks/isotropic keys do not match")
        if bool(raw_blocks) != bool(tasks):
            raise ValueError("KFAC task count and empty state disagree")
        blocks, isotropic = {}, {}
        for key, terms in raw_blocks.items():
            if not isinstance(key, str) or not key or not isinstance(terms, (list, tuple)) or not terms:
                raise ValueError("KFAC blocks require nonempty string keys and term lists")
            if len(terms) != tasks:
                raise ValueError("KFAC history requires one factor term per block per task")
            blocks[key] = []
            expected = None
            for index, term in enumerate(terms):
                if not isinstance(term, Mapping) or not {"G", "C", "weight"}.issubset(term):
                    raise ValueError(f"Incomplete KFAC term {key}[{index}]")
                g_factor = cls._factor(term["G"], f"{key}[{index}].G", validate_psd=True)
                c_factor = cls._factor(term["C"], f"{key}[{index}].C", validate_psd=True)
                shape = (g_factor.shape[0], c_factor.shape[0])
                if expected is not None and shape != expected:
                    raise ValueError(f"KFAC term shapes disagree in {key}")
                expected = shape
                blocks[key].append({"G": g_factor, "C": c_factor, "weight": cls._nonnegative(term["weight"], f"{key}.weight")})
            isotropic[key] = cls._nonnegative(raw_isotropic[key], f"{key}.isotropic")
        state = cls(blocks, isotropic, tasks)
        if shapes is not None:
            if not isinstance(shapes, Mapping) or {k: tuple(v) for k, v in shapes.items()} != state.shapes:
                raise ValueError("KFAC checkpoint keys/shapes do not match the model")
        if not math.isfinite(state.trace()):
            raise ValueError("KFAC checkpoint trace is nonfinite")
        return state

    def tensor_bytes(self):
        """Canonical factor storage bytes, excluding transient device cache."""
        return sum(
            term[name].numel() * term[name].element_size()
            for terms in self.blocks.values() for term in terms for name in ("G", "C")
        )
