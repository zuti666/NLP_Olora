"""CPU online averages of the *current* LoRA A/B factors at fixed epochs.

This averages A and B separately, not their products B @ A.  The caller must
apply the average before estimating historical importance and saving an anchor.
Classifier parameters, historical adapters and scaling parameters are excluded.
"""

from __future__ import annotations

import hashlib
import math
import re

import torch


_FACTOR_NAME = re.compile(r"(?:^|\.)linear_[ab]_[qv]\.weight$")


def _factors(model):
    factors = {name: param for name, param in model.named_parameters() if _FACTOR_NAME.search(name)}
    if not factors:
        raise ValueError("No current LoRA linear_[ab]_[qv].weight factors found")
    return factors


def _hash(factors):
    digest = hashlib.sha256()
    for name, tensor in sorted(factors.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ABFactorAverager:
    """Average snapshots at explicit one-based epochs, then mix with endpoint.

    ``record(model, epoch)`` returns whether this epoch was recorded.  ``apply``
    performs ``factor <- (1 - omega) * endpoint + omega * mean`` and returns
    JSON-compatible provenance. At least two snapshots and every requested
    epoch are required, so a single endpoint cannot masquerade as averaging.
    """

    def __init__(self, epochs=(3, 4, 5)):
        epochs = tuple(epochs)
        if len(epochs) < 2 or any(isinstance(e, bool) or not isinstance(e, int) or e <= 0 for e in epochs):
            raise ValueError("Averaging needs at least two positive integer epochs")
        if len(set(epochs)) != len(epochs) or tuple(sorted(epochs)) != epochs:
            raise ValueError("Averaging epochs must be unique and increasing")
        self.epochs = epochs
        self.recorded_epochs = []
        self.mean = {}
        self.merge_applied = False
        self._merge_metadata = {}

    @property
    def count(self):
        return len(self.recorded_epochs)

    def record(self, model, epoch):
        if self.merge_applied:
            raise RuntimeError("Cannot record after applying the average")
        if epoch not in self.epochs:
            return False
        if epoch in self.recorded_epochs:
            raise ValueError(f"Epoch {epoch} has already been recorded")
        if self.recorded_epochs and epoch <= self.recorded_epochs[-1]:
            raise ValueError("Snapshots must be recorded in increasing epoch order")
        factors = _factors(model)
        cpu = {name: param.detach().to(device="cpu", dtype=torch.float32).clone() for name, param in factors.items()}
        if any(not torch.isfinite(tensor).all() for tensor in cpu.values()):
            raise ValueError("LoRA snapshots must be finite")
        if self.mean and (set(cpu) != set(self.mean) or any(cpu[name].shape != self.mean[name].shape for name in cpu)):
            raise ValueError("LoRA factor names/shapes changed between snapshots")
        if not self.mean:
            self.mean = cpu
        else:
            for name, tensor in cpu.items():
                self.mean[name].add_((tensor - self.mean[name]) / (self.count + 1))
        self.recorded_epochs.append(epoch)
        return True

    @torch.no_grad()
    def apply(self, model, omega=0.5):
        omega = float(omega)
        if not math.isfinite(omega) or not 0.0 <= omega <= 1.0:
            raise ValueError("omega must be finite and in [0, 1]")
        if self.merge_applied:
            raise RuntimeError("Average has already been applied")
        if self.count < 2:
            raise ValueError("At least two recorded snapshots are required")
        if tuple(self.recorded_epochs) != self.epochs:
            raise ValueError("All requested averaging epochs must be recorded")
        factors = _factors(model)
        if set(factors) != set(self.mean) or any(factors[name].shape != self.mean[name].shape for name in factors):
            raise ValueError("LoRA factor names/shapes changed before applying average")
        before = {name: param.detach().cpu().clone() for name, param in factors.items()}
        if any(not torch.isfinite(tensor).all() for tensor in before.values()):
            raise ValueError("LoRA endpoints must be finite")
        # Compute everything before modifying the model to keep validation failures atomic.
        merged = {name: ((1.0 - omega) * tensor.float() + omega * self.mean[name]).to(tensor.dtype)
                  for name, tensor in before.items()}
        if any(not torch.isfinite(tensor).all() for tensor in merged.values()):
            raise ValueError("Merged LoRA factors must be finite")
        displacement = sum(float((merged[name].double() - tensor.double()).square().sum())
                           for name, tensor in before.items())
        endpoint_norm = sum(float(tensor.double().square().sum()) for tensor in before.values())
        for name, param in factors.items():
            param.copy_(merged[name].to(device=param.device))
        self.merge_applied = True
        self._merge_metadata = {
            "omega": omega,
            "endpoint_hash": _hash(before),
            "merged_hash": _hash(merged),
            "mean_hash": _hash(self.mean),
            "factor_displacement_l2": math.sqrt(displacement),
            "factor_relative_displacement_l2": math.sqrt(displacement / endpoint_norm) if endpoint_norm else None,
        }
        return self.metadata()

    def metadata(self):
        return {
            "coordinate": "current_lora_A_B_factors",
            "epochs": list(self.epochs),
            "recorded_epochs": list(self.recorded_epochs),
            "count": self.count,
            "merge_applied": self.merge_applied,
            "mean_dtype": "float32",
            "mean_device": "cpu",
            "classifier": "endpoint_unchanged",
            "factor_count": len(self.mean),
            "mean_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in self.mean.values()),
            **self._merge_metadata,
        }
