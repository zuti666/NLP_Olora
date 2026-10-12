"""Matched diagonal and partial-trace Kronecker gradient-moment estimates.

This is a gradient-moment Kronecker approximation, not activation KFAC. For a
matrix gradient H, accumulate R=E[HH.T], S=E[H.T H], t=tr(R), and return
G=R/sqrt(t), C=S/sqrt(t). Thus C (x) G has both exact partial traces and
the same trace as E[vec(H)vec(H).T], without constructing that dense matrix.
The caller owns normalization, damping, task accumulation, and anchors.
"""

import hashlib
import itertools
import json
import math
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


_MODES = {"batch_empirical", "sample_empirical", "true_sample"}
_REPRESENTATIONS = {"diagonal", "kronecker"}
_HOOK_ATTRS = ("_register_delta_hook", "delta_w_q_new_grad", "delta_w_v_new_grad")
_MISSING = object()


@contextmanager
def _preserve_estimation_state(model, loader, seed):
    """Estimate reproducibly without disturbing the surrounding training run."""
    if bool(getattr(loader, "persistent_workers", False)):
        raise ValueError("Fisher pairing requires persistent_workers=False")
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(param, param.grad) for param in model.parameters()]
    hooks = [
        (module, {attr: getattr(module, attr, _MISSING) for attr in _HOOK_ATTRS})
        for module in model.modules()
        if any(hasattr(module, attr) for attr in _HOOK_ATTRS)
    ]
    generators = {}
    for owner in (loader, getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in generators:
            generators[id(generator)] = (generator, generator.get_state())
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        for generator, _ in generators.values():
            generator.manual_seed(seed)
        model.eval()
        yield
    finally:
        for param, gradient in gradients:
            param.grad = gradient
        for module, state in hooks:
            for attr, value in state.items():
                if value is _MISSING:
                    if hasattr(module, attr):
                        delattr(module, attr)
                else:
                    setattr(module, attr, value)
        for module, training in modes:
            module.training = training
        for generator, state in generators.values():
            generator.set_state(state)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(f"Fisher sample IDs must be JSON serializable, got {type(value).__name__}")


def _tensor_hash_update(hasher, value):
    value = value.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(json.dumps(list(value.shape)).encode("ascii"))
    # Viewing bytes also supports bfloat16, which NumPy cannot represent.
    hasher.update(value.view(torch.uint8).numpy().tobytes())


def estimate_importance(*, learner, model, loader, mode, representation,
                        class_offset, seed, mc_samples=1, max_batches=None):
    """Return ``(payload, metadata)`` from exactly matched gradient draws.

    ``batch_empirical`` uses the gradient of mean batch CE, weighted by batch
    size and divided by the number of examples. The other rows average
    single-example gradients, with ``true_sample`` labels independently drawn
    from the frozen model's current-task softmax. Label draws use their own CPU
    generator, so both representations consume identical labels.

    Payload tensors are detached CPU FP64. Diagonal payloads map block names to
    tensors; Kronecker payloads map them to ``{"G": tensor, "C": tensor}``.
    Zero-trace blocks remain zero. Empty data or missing hook gradients fail.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown computation mode {mode!r}")
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"Unknown matrix representation {representation!r}")
    if isinstance(mc_samples, bool) or int(mc_samples) != mc_samples or int(mc_samples) < 1:
        raise ValueError("mc_samples must be a positive integer")
    mc_samples = int(mc_samples)
    if mode != "true_sample" and mc_samples != 1:
        raise ValueError("mc_samples applies only to true_sample")
    if max_batches is not None:
        if isinstance(max_batches, bool) or int(max_batches) != max_batches or int(max_batches) < 1:
            raise ValueError("max_batches must be a positive integer or None")
        max_batches = int(max_batches)
    seed = int(seed)
    class_offset = int(class_offset)
    label_generator = torch.Generator(device="cpu").manual_seed(seed)
    input_hasher = hashlib.sha256()
    accumulators = {}
    block_traces = {}
    expected_keys = None
    gradient_dtypes = set()
    product_dtypes = set()
    sample_ids = []
    label_draws = []
    sample_count = batch_count = gradient_draws = denominator = 0
    data_position = 0
    id_kinds = set()
    started = time.perf_counter()

    def collect(weight):
        nonlocal expected_keys, gradient_draws
        hook_grads = dict(learner._iter_hook_grads(model))
        if not hook_grads:
            raise RuntimeError("No effective Delta-W gradients were captured")
        keys = set(hook_grads)
        if expected_keys is None:
            expected_keys = keys
            if hasattr(learner, "_iter_delta_terms"):
                declared = {key for key, _ in learner._iter_delta_terms(model)}
                if declared != keys:
                    raise RuntimeError(f"Incomplete Delta-W hooks: expected {sorted(declared)}, got {sorted(keys)}")
        elif keys != expected_keys:
            raise RuntimeError("Delta-W hook keys changed between gradient draws")
        for key, gradient in hook_grads.items():
            if gradient.ndim != 2 or not gradient.is_floating_point():
                raise RuntimeError(f"Delta-W gradient {key} must be a floating matrix")
            gradient_dtypes.add(str(gradient.dtype))
            # GPU matrix products stay FP32; the running sums live in CPU FP64.
            h = gradient.detach().to(dtype=torch.float32 if gradient.is_cuda else torch.float64)
            product_dtypes.add(str(h.dtype))
            if representation == "diagonal":
                moment = h.square().to(device="cpu", dtype=torch.float64)
                if not bool(torch.isfinite(moment).all()):
                    raise RuntimeError(f"Nonfinite Delta-W moment for {key}")
                if key not in accumulators:
                    accumulators[key] = torch.zeros_like(moment)
                accumulators[key].add_(moment, alpha=float(weight))
                trace = float(moment.sum())
            else:
                row = (h @ h.T).to(device="cpu", dtype=torch.float64)
                col = (h.T @ h).to(device="cpu", dtype=torch.float64)
                if not bool(torch.isfinite(row).all() and torch.isfinite(col).all()):
                    raise RuntimeError(f"Nonfinite Delta-W moment for {key}")
                if key not in accumulators:
                    accumulators[key] = {"R": torch.zeros_like(row), "S": torch.zeros_like(col)}
                accumulators[key]["R"].add_(row, alpha=float(weight))
                accumulators[key]["S"].add_(col, alpha=float(weight))
                trace = float(row.trace())
            block_traces[key] = block_traces.get(key, 0.0) + float(weight) * trace
        gradient_draws += 1

    with _preserve_estimation_state(model, loader, seed):
        learner._set_delta_hook_state(model, True)
        batches = iter(loader)
        if max_batches is not None:
            batches = itertools.islice(batches, max_batches)
        for batch in batches:
            if len(batch) == 3:
                ids, raw_inputs, raw_targets = batch
                ids = _json_value(ids)
                id_kinds.add("dataset_id")
            elif len(batch) == 2:
                raw_inputs, raw_targets = batch
                ids = list(range(data_position, data_position + int(raw_targets.numel())))
                id_kinds.add("loader_position")
            else:
                raise ValueError("Fisher loader must yield (IDs, inputs, targets) or (inputs, targets)")
            raw_count = int(raw_targets.numel())
            if not isinstance(ids, list) or len(ids) != raw_count:
                raise ValueError("Fisher sample IDs do not match batch size")
            data_position += raw_count
            class_end = int(getattr(learner, "_total_classes", 0))
            if class_end <= class_offset:
                class_end = int(raw_targets.max().item()) + 1 if raw_count else class_offset
            keep = ((raw_targets >= class_offset) & (raw_targets < class_end)).detach().cpu().nonzero().flatten().tolist()
            inputs, targets = learner._current_task_batch(raw_inputs, raw_targets, class_offset)
            if inputs is None:
                continue
            kept_ids = [ids[index] for index in keep]
            count = int(inputs.shape[0])
            if count != len(kept_ids):
                raise RuntimeError("Current-task batch filtering disagrees with sample IDs")
            sample_ids.extend(kept_ids)
            _tensor_hash_update(input_hasher, inputs)
            _tensor_hash_update(input_hasher, targets)
            batch_count += 1
            sample_count += count
            if mode == "batch_empirical":
                model.zero_grad(set_to_none=True)
                learner._clear_delta_hook_grads(model)
                with torch.enable_grad():
                    logits = learner._current_task_logits(model(inputs), class_offset)
                    F.cross_entropy(logits, targets, reduction="mean").backward()
                collect(count)
                denominator += count
                label_draws.append({"sample_ids": kept_ids, "labels": targets.detach().cpu().tolist(), "mc_index": 0})
            else:
                draws = mc_samples if mode == "true_sample" else 1
                for index in range(count):
                    with torch.enable_grad():
                        logits = learner._current_task_logits(model(inputs[index:index + 1]), class_offset)
                        if mode == "true_sample":
                            probabilities = logits.detach().float().softmax(-1).cpu()
                            labels = torch.multinomial(probabilities, draws, replacement=True, generator=label_generator).flatten()
                        else:
                            labels = targets[index:index + 1].detach().cpu()
                        for draw_index, label in enumerate(labels.tolist()):
                            model.zero_grad(set_to_none=True)
                            learner._clear_delta_hook_grads(model)
                            target = torch.tensor([label], dtype=torch.long, device=logits.device)
                            F.cross_entropy(logits, target).backward(retain_graph=(draw_index + 1 < draws))
                            collect(1)
                            denominator += 1
                            label_draws.append({"sample_ids": [kept_ids[index]], "labels": [int(label)], "mc_index": draw_index})
        learner._clear_delta_hook_grads(model)

    if denominator == 0 or not accumulators:
        raise ValueError("No current-task examples were available for importance estimation")
    payload = {}
    partial_trace_error = {}
    for key, accumulator in accumulators.items():
        block_traces[key] /= float(denominator)
        if representation == "diagonal":
            payload[key] = accumulator / float(denominator)
        else:
            row = accumulator["R"] / float(denominator)
            col = accumulator["S"] / float(denominator)
            trace = float(row.trace())
            col_trace = float(col.trace())
            partial_trace_error[key] = abs(trace - col_trace) / max(abs(trace), abs(col_trace), torch.finfo(torch.float64).tiny)
            if trace > 0:
                # The analytically equal traces differ slightly after FP32 GEMM.
                # Rescaling S reconciles this roundoff and preserves total trace.
                if col_trace <= 0 or not math.isfinite(trace):
                    raise RuntimeError(f"Inconsistent partial traces for {key}")
                col = col * (trace / col_trace)
                scale = math.sqrt(trace)
                payload[key] = {"G": (row + row.T) * (0.5 / scale), "C": (col + col.T) * (0.5 / scale)}
            else:
                payload[key] = {"G": torch.zeros_like(row), "C": torch.zeros_like(col)}
    draws_json = json.dumps(label_draws, sort_keys=True, separators=(",", ":"))
    metadata = {
        "computation": mode,
        "representation": representation,
        "estimator": "gradient_moment_partial_trace_kronecker" if representation == "kronecker" else "gradient_moment_diagonal",
        "parameter_space": "unscaled_delta_w_ba",
        "sample_count": sample_count,
        "num_samples": sample_count,
        "batch_count": batch_count,
        "gradient_draws": gradient_draws,
        "normalization_denominator": denominator,
        "mc_samples": mc_samples,
        "class_offset": class_offset,
        "seed": seed,
        "sample_ids": sample_ids,
        "sample_id_kinds": sorted(id_kinds),
        "input_sha256": input_hasher.hexdigest(),
        "label_draws": label_draws,
        "label_draw_sha256": hashlib.sha256(draws_json.encode("utf-8")).hexdigest(),
        "raw_trace": sum(block_traces.values()),
        "block_raw_traces": block_traces,
        "partial_trace_relative_roundoff": partial_trace_error,
        "gradient_dtypes": sorted(gradient_dtypes),
        "product_dtypes": sorted(product_dtypes),
        "accumulation_dtype": "torch.float64",
        "elapsed_seconds": time.perf_counter() - started,
        "loss_reduction": "batch_mean_weighted_by_samples" if mode == "batch_empirical" else "per_example_cross_entropy",
    }
    return payload, metadata
