"""Streaming, effective-Q/V, per-example empirical-Fisher evaluation.

This module deliberately has no imports from the training repository at import
time. The synthetic CPU self-test therefore needs only PyTorch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def select_scope(logits, targets, scope):
    if scope == "old_local20":
        if not bool(((targets >= 0) & (targets < 20)).all()):
            raise ValueError("old_local20 requires labels 0..19")
        return logits[:, :20], targets
    if scope == "new_local20":
        if logits.shape[1] < 40 or not bool(((targets >= 20) & (targets < 40)).all()):
            raise ValueError("new_local20 requires a 40-class head and labels 20..39")
        return logits[:, 20:40], targets - 20
    if scope == "seen40":
        if logits.shape[1] != 40 or not bool(((targets >= 0) & (targets < 40)).all()):
            raise ValueError("seen40 requires a 40-class head and labels 0..39")
        return logits, targets
    raise ValueError(f"Unknown loss scope: {scope}")


def unpack_batch(batch):
    if len(batch) == 3:
        return batch
    if len(batch) == 2:
        x, y = batch
        return None, x, y
    raise ValueError("Expected (index, image, label) or (image, label)")


class PerExampleCollector:
    """Recover every sample's linear-weight gradient, without storing all F."""

    def __init__(self, block_count):
        self.block_count = int(block_count)
        self.active = False

    def start(self, count, device, directions):
        self.active = True
        self.trace = torch.zeros(count, device=device, dtype=torch.float64)
        self.dots = {name: torch.zeros_like(self.trace) for name in directions}
        self.directions = directions
        self.blocks_seen = set()

    def add(self, index, inputs, adjoint):
        # Losses are summed, so adjoint retains each unscaled sample gradient.
        with torch.no_grad():
            gradient = torch.bmm(adjoint.detach().transpose(1, 2), inputs)
            self.trace.add_(gradient.square().sum(dim=(1, 2), dtype=torch.float64))
            for name, direction in self.directions.items():
                self.dots[name].add_((gradient * direction[index]).sum(
                    dim=(1, 2), dtype=torch.float64))
            if index in self.blocks_seen:
                raise RuntimeError("A measured linear block was reused in the same forward")
            self.blocks_seen.add(index)
        return adjoint

    def finish(self):
        self.active = False
        if self.blocks_seen != set(range(self.block_count)):
            raise RuntimeError(f"Missing measured blocks: {self.blocks_seen}")
        return self.trace.detach(), {key: value.detach() for key, value in self.dots.items()}


class EffectiveQKV(nn.Module):
    def __init__(self, original, collector, offset, effective_values):
        super().__init__()
        self.qkv = original.qkv
        device = original.qkv.weight.device
        self.effective_q = nn.Parameter(effective_values[0].detach().clone().to(device))
        self.effective_v = nn.Parameter(effective_values[1].detach().clone().to(device))
        self.collector, self.offset = collector, offset

    def forward(self, inputs):
        query = F.linear(inputs, self.effective_q)
        value = F.linear(inputs, self.effective_v)
        if self.collector.active and torch.is_grad_enabled():
            saved_inputs = inputs.detach()
            query.register_hook(lambda adj: self.collector.add(self.offset, saved_inputs, adj))
            value.register_hook(lambda adj: self.collector.add(self.offset + 1, saved_inputs, adj))
        base_q, base_k, base_v = self.qkv(inputs).chunk(3, dim=-1)
        return torch.cat((base_q + query, base_k, base_v + value), dim=-1)


class RestoredNetwork(nn.Module):
    def __init__(self, backbone, classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(768, classes)

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))


def read_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("network_state")
    if not isinstance(state, dict):
        raise ValueError(f"{path} has no complete network_state")
    return state


def factor_names():
    return [f"backbone.lora_vit.blocks.{layer}.attn.qkv.linear_{factor}_{channel}.weight"
            for layer in range(12) for channel in ("q", "v") for factor in ("a", "b")]


def effective_weights(state):
    values = []
    for layer in range(12):
        for channel in ("q", "v"):
            stem = f"backbone.lora_vit.blocks.{layer}.attn.qkv.linear_"
            a, b = state[f"{stem}a_{channel}.weight"], state[f"{stem}b_{channel}.weight"]
            if a.shape != (10, 768) or b.shape != (768, 10):
                raise ValueError(f"Unsupported LoRA shape in layer {layer}/{channel}")
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                raise ValueError("Expected FP32 saved LoRA factors")
            values.append(b @ a)
    return values


def audit_checkpoint_pair(paths):
    states = [read_state(path) for path in paths]
    old, new = states
    if set(old) != set(new):
        raise ValueError("Endpoint state key sets differ")
    if old["fc.weight"].shape != (20, 768) or new["fc.weight"].shape != (40, 768):
        raise ValueError("Expected exactly 20-class and 40-class checkpoints")
    for key in ("fc.weight", "fc.bias"):
        if not torch.equal(old[key], new[key][:20]):
            raise ValueError(f"Old classifier changed: {key}; fixed-head comparison invalid")
    excluded = set(factor_names()) | {"fc.weight", "fc.bias"}
    unchanged = [key for key in old if key not in excluded]
    for key in unchanged:
        if not torch.equal(old[key], new[key]):
            raise ValueError(f"Backbone/frozen tensor changed: {key}")
    values = [effective_weights(state) for state in states]
    delta = [end - start for start, end in zip(*values)]
    squared = sum(float(value.double().square().sum()) for value in delta)
    audit = {
        "checkpoints": [{"path": str(path), "sha256": sha_file(path),
                         "state_tensors": len(state)} for path, state in zip(paths, states)],
        "frozen_tensors_equal": len(unchanged), "old_head_bitwise_equal": True,
        "effective_blocks": 24, "effective_parameters": 14155776,
        "coordinate": "12 layers x Q,V effective BA matrices; scale=1; excludes head",
        "effective_product_precision": "CPU FP32 BA; FP32 subtraction; float64 norm reduction",
        "delta_squared_norm": squared, "delta_norm": math.sqrt(squared),
        "delta_sha256": tensor_hash(delta),
    }
    return values, delta, audit


def load_network(path, source, device):
    source = str(Path(source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    import timm
    from backbone.lora import IdentityScale, LoRA_ViT_timm
    state = read_state(path)
    if any("scaling" in key or "wrapped_param" in key for key in state):
        raise ValueError("Learned LoRA scaling is unsupported")
    base = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    backbone = LoRA_ViT_timm(base.eval(), r=10, num_classes=0, increment=20,
                            filepath="/unused_offline_no_sidecars/", learn_alpha=False)
    model = RestoredNetwork(backbone, int(state["fc.weight"].shape[0]))
    model.load_state_dict(state, strict=True)
    if len(model.backbone.lora_vit.blocks) != 12:
        raise ValueError("Expected ViT-B/16 with twelve blocks")
    for block in model.backbone.lora_vit.blocks:
        qkv = block.attn.qkv
        if qkv.task_id != 0 or qkv.saved_A or qkv.saved_B:
            raise ValueError("Expected continuous, single Q/V adapter without history")
        if len(qkv.scaling_factor) != 1 or not isinstance(qkv.scaling_factor[0], IdentityScale):
            raise ValueError("Expected identity LoRA scale")
    del model.backbone.base_vit  # Unused copy; all its tensors were strictly restored above.
    return model.requires_grad_(False).eval().to(device)


def convert_effective(model, values):
    collector = PerExampleCollector(24)
    for index, block in enumerate(model.backbone.lora_vit.blocks):
        block.attn.qkv = EffectiveQKV(block.attn.qkv, collector, 2 * index,
                                     values[2 * index:2 * index + 2])
    parameters = [parameter for block in model.backbone.lora_vit.blocks
                  for parameter in (block.attn.qkv.effective_q, block.attn.qkv.effective_v)]
    if sum(parameter.numel() for parameter in parameters) != 14155776:
        raise ValueError("Effective coordinate dimension mismatch")
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}:
        raise ValueError("Unexpected trainable parameters in the evaluation model")
    model.eval()
    return collector, parameters


def proportional_indices(targets, samples, seed):
    targets = torch.as_tensor(targets, dtype=torch.long)
    if not 0 < samples <= len(targets):
        raise ValueError(f"Requested {samples} Fisher samples from {len(targets)} available")
    labels, counts = torch.unique(targets, sorted=True, return_counts=True)
    quotas = counts.double() * samples / len(targets)
    allocation = quotas.floor().long()
    remainder_order = sorted(range(len(labels)), key=lambda i: (-float(quotas[i] - allocation[i]),
                                                               int(labels[i])))
    for index in remainder_order[:samples - int(allocation.sum())]:
        allocation[index] += 1
    generator = torch.Generator().manual_seed(int(seed))
    chosen = []
    for label, amount in zip(labels, allocation):
        candidates = torch.where(targets == label)[0]
        selected = candidates[torch.randperm(len(candidates), generator=generator)[:int(amount)]]
        chosen.extend(selected.tolist())
    chosen.sort()
    if len(chosen) != samples or len(set(chosen)) != samples:
        raise RuntimeError("Invalid stratified sample selection")
    return chosen, {str(int(label)): {"available": int(count), "selected": int(amount)}
                    for label, count, amount in zip(labels, counts, allocation)}


def prepare_data(config, manifest, source, data_root, samples, sample_seed, batch_size, workers, device):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import numpy as np
    import PIL
    import torchvision
    from utils.data_manager import DataManager
    from torch.utils.data import DataLoader, Subset

    args = dict(config)
    args["data_root"] = str(data_root)
    args.setdefault("model_name", "ewclora_normfisher_gam")
    order = manifest.get("class_order", config.get("class_order"))
    if order is not None:
        args["class_order_override"] = order
    seed = config.get("seed", 0)
    if isinstance(seed, (list, tuple)):
        if len(seed) != 1:
            raise ValueError("Evaluation requires one run seed, not a seed sweep")
        seed = seed[0]
    manager = DataManager("imagenetr", bool(config.get("class_shuffle", False)), int(seed),
                          20, 20, args)
    if order is not None and list(order) != list(manager._class_order):
        raise ValueError("Manifest class order differs from the data manager")
    datasets = {
        "old_train": manager.get_dataset(np.arange(20), source="train", mode="test"),
        "old_test": manager.get_dataset(np.arange(20), source="test", mode="test"),
        "new_train": manager.get_dataset(np.arange(20, 40), source="train", mode="test"),
        "new_test": manager.get_dataset(np.arange(20, 40), source="test", mode="test"),
    }
    data_manifest = {
        "dataset": "imagenetr", "data_root": str(data_root),
        "class_order": list(map(int, manager._class_order)),
        "source": "training samples with deterministic test transforms",
        "transform": repr(datasets["old_train"].trsf),
        "torchvision": torchvision.__version__, "Pillow": PIL.__version__,
        "loader_mode": manager.loader_mode,
        "sampler": "Hamilton proportional allocation; lower label wins remainder ties; torch randperm",
        "sample_seed": sample_seed, "fisher_samples_per_task": samples,
        "datasets": {}, "fisher_subsets": {},
    }
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"Empty dataset: {name}")
        records = [{"id": str(Path(path).resolve()), "label": int(label)}
                   for path, label in zip(dataset.images, dataset.labels)]
        data_manifest["datasets"][name] = {"count": len(dataset), "records": records,
                                           "records_sha256": canonical_hash(records)}
    for task in ("old", "new"):
        dataset = datasets[f"{task}_train"]
        chosen, allocation = proportional_indices(dataset.labels, samples, sample_seed)
        records = [{"dataset_index": index, "id": str(Path(dataset.images[index]).resolve()),
                    "label": int(dataset.labels[index]), "sha256": sha_file(dataset.images[index])}
                   for index in chosen]
        data_manifest["fisher_subsets"][task] = {"indices": chosen, "allocation": allocation,
                                                "records": records, "records_sha256": canonical_hash(records)}
        datasets[f"{task}_fisher"] = Subset(dataset, chosen)
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                  pin_memory=torch.device(device).type == "cuda", drop_last=False)
    loaders = {name: DataLoader(dataset, **kwargs) for name, dataset in datasets.items()}
    return loaders, data_manifest


@torch.no_grad()
def evaluate_scopes(model, loader, scopes, device):
    accum = {scope: {"loss_sum": 0.0, "correct": 0, "samples": 0} for scope in scopes}
    for batch in loader:
        _, inputs, targets = unpack_batch(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        for scope in scopes:
            selected, labels = select_scope(logits, targets, scope)
            accum[scope]["loss_sum"] += float(F.cross_entropy(selected, labels, reduction="none").double().sum())
            accum[scope]["correct"] += int((selected.argmax(1) == labels).sum())
            accum[scope]["samples"] += len(labels)
    return {scope: {"loss": value["loss_sum"] / value["samples"],
                    "accuracy": 100.0 * value["correct"] / value["samples"],
                    "samples": value["samples"], "correct": value["correct"], "loss_scope": scope}
            for scope, value in accum.items()}


def collect_batch(model, collector, inputs, targets, scope, directions):
    model.zero_grad(set_to_none=True)
    collector.start(len(targets), inputs.device, directions)
    try:
        logits, labels = select_scope(model(inputs), targets, scope)
        losses = F.cross_entropy(logits, labels, reduction="none")
        losses.sum().backward()
        trace, dots = collector.finish()
    finally:
        collector.active = False
    return losses.detach(), trace, dots


def fisher(model, collector, loader, scope, directions, device):
    started = time.monotonic()
    loss_sum = trace_sum = 0.0
    q_sums = {name: 0.0 for name in directions}
    count, input_hashes = 0, []
    for batch in loader:
        indices, inputs, targets = unpack_batch(batch)
        for index, sample, label in zip(indices, inputs, targets):
            input_hashes.append({"dataset_index": int(index), "label": int(label),
                                 "tensor_sha256": tensor_hash([sample])})
        inputs, targets = inputs.to(device), targets.to(device)
        losses, trace, dots = collect_batch(model, collector, inputs, targets, scope, directions)
        trace_sum += float(trace.sum())
        loss_sum += float(losses.double().sum())
        for name, dot in dots.items():
            # Sum over coordinates/blocks BEFORE squaring: full outer-product q.
            q_sums[name] += float(dot.square().sum())
        count += len(targets)
    model.zero_grad(set_to_none=True)
    if count == 0 or not math.isfinite(trace_sum) or trace_sum < 0:
        raise RuntimeError("Invalid empirical Fisher estimate")
    return {
        "raw_trace": trace_sum / count, "subset_loss": loss_sum / count,
        "q": {name: value / count for name, value in q_sums.items()},
        "samples": count, "seconds": time.monotonic() - started,
        "gradient_batch_size": loader.batch_size, "loss_scope": scope,
        "parameter_scope": "24 effective Q/V weight matrices; fixed backbone and classifier",
        "estimator": "mean of true-label per-example CE gradient outer products",
        "normalization": "sample mean only", "normalized": False,
        "floor": None, "damping": None, "regularizer": None,
        "input_tensor_records": input_hashes, "input_tensors_sha256": canonical_hash(input_hashes),
    }


def compare_hook_autograd(model, collector, parameters, inputs, targets, scope, directions):
    _, actual_trace, actual_dots = collect_batch(model, collector, inputs, targets, scope, directions)
    expected_trace = []
    expected_dots = {name: [] for name in directions}
    for index in range(len(targets)):
        logits, labels = select_scope(model(inputs[index:index + 1]), targets[index:index + 1], scope)
        gradients = torch.autograd.grad(F.cross_entropy(logits, labels), parameters)
        expected_trace.append(sum(gradient.double().square().sum() for gradient in gradients))
        for name, direction in directions.items():
            expected_dots[name].append(sum((gradient.double() * value.double()).sum()
                                           for gradient, value in zip(gradients, direction)))
    expected_trace = torch.stack(expected_trace)
    torch.testing.assert_close(actual_trace, expected_trace, rtol=5e-4, atol=1e-7)
    dot_errors = {}
    for name in directions:
        expected = torch.stack(expected_dots[name])
        torch.testing.assert_close(actual_dots[name], expected, rtol=5e-4, atol=2e-5)
        dot_errors[name] = float((actual_dots[name] - expected).abs().max())
    model.zero_grad(set_to_none=True)
    return {"passed": True, "samples": len(targets), "loss_scope": scope,
            "trace_max_absolute_error": float((actual_trace - expected_trace).abs().max()),
            "trace_max_relative_error": float(((actual_trace - expected_trace).abs() /
                                                  expected_trace.abs().clamp_min(1e-30)).max()),
            "direction_dot_max_absolute_error": dot_errors}


def self_test():
    """Tiny sample-independent network; no ViT, dataset, training source or GPU."""
    torch.manual_seed(20260920)
    collector = PerExampleCollector(2)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
                                              for _ in range(2)])
            self.register_buffer("head", torch.randn(40, 4, dtype=torch.float64))

        def forward(self, inputs):
            branches = []
            for index, weight in enumerate(self.weights):
                output = F.linear(inputs, weight)
                if collector.active:
                    output.register_hook(lambda adj, block=index: collector.add(block, inputs.detach(), adj))
                branches.append(output)
            features = torch.tanh(branches[0] + 0.37 * branches[1]).mean(dim=1)
            return F.linear(features, self.head)

    model = Tiny().eval()
    inputs = torch.randn(5, 2, 3, dtype=torch.float64)
    directions = {"delta": [torch.randn_like(weight) for weight in model.weights]}
    results = {}
    for scope, targets in (("old_local20", torch.tensor([0, 1, 2, 3, 4])),
                           ("new_local20", torch.tensor([20, 21, 22, 23, 24])),
                           ("seen40", torch.tensor([0, 21, 2, 23, 4]))):
        results[scope] = compare_hook_autograd(model, collector, list(model.weights),
                                             inputs, targets, scope, directions)
        values = {}
        for batch_size in (1, 2, 5):
            traces, dots = [], []
            for start in range(0, len(targets), batch_size):
                _, trace, dot = collect_batch(model, collector, inputs[start:start + batch_size],
                                              targets[start:start + batch_size], scope, directions)
                traces.append(trace)
                dots.append(dot["delta"])
            values[batch_size] = torch.stack((torch.cat(traces).mean(), torch.cat(dots).square().mean()))
        torch.testing.assert_close(values[1], values[2], rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(values[1], values[5], rtol=1e-11, atol=1e-11)
        results[scope]["batch_partition_check"] = "1,2,5 agree, including incomplete final microbatch"
    labels = [0] * 7 + [1] * 11 + [2] * 3
    indices, allocation = proportional_indices(labels, 9, 101)
    assert indices == proportional_indices(labels, 9, 101)[0]
    assert sum(value["selected"] for value in allocation.values()) == 9
    return {"status": "complete", "passed": True, "device": "cpu", "dtype": "float64",
            "scope_checks": results, "sampler_check": "fixed, without replacement, exact sample count"}
