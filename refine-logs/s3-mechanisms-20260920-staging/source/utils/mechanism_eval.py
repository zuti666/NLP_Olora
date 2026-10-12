import logging
import math
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_float_list(value, default: Sequence[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return list(default)
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [float(item) for item in value]
    return [float(value)]


def parse_int_list(value, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return list(default)
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    return [int(value)]


def parse_str_list(value, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return list(default)
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def parse_max_batches(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.lower() in {"none", "all", "full"}:
        return None
    value = int(value)
    return None if value <= 0 else value


def parse_positive_int(value, default=None):
    if value is None:
        return default
    value = int(value)
    return value if value > 0 else default


def _make_eval_loader(loader, batch_size: Optional[int]):
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader
    if batch_size is None:
        batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        return loader

    num_workers = int(getattr(loader, "num_workers", 0) or 0)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
        "drop_last": False,
        "collate_fn": getattr(loader, "collate_fn", None),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(loader, "persistent_workers", False))
    return DataLoader(**kwargs)


@contextmanager
def preserve_training_mode(model):
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def select_eval_named_params(model, scope: str = "lora") -> List[Tuple[str, torch.nn.Parameter]]:
    model = unwrap_model(model)
    scope = str(scope or "lora").lower()
    lora_tokens = (
        "lora_new_A_k",
        "lora_new_A_v",
        "lora_new_B_k",
        "lora_new_B_v",
    )

    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if scope in {
            "lora",
            "new_lora",
            "delta_w",
            "lora_factor",
            "lora_factors",
            "lora_ab",
            "ab",
            "a_b",
        }:
            if any(token in name for token in lora_tokens):
                named_params.append((name, param))
        elif scope in {"classifier", "head"}:
            if "classifier_pool" in name or name.endswith("head.weight"):
                named_params.append((name, param))
        elif scope in {"trainable", "all_trainable"}:
            named_params.append((name, param))
        elif scope in {"no_fc", "trainable_no_fc", "ewc_params", "backbone_trainable"}:
            # All trainable params EXCEPT the linear classifier head (fc.*).
            # Mirrors EWCon_Probe._iter_trainable with ewc_include_fc=False.
            # Use this scope when loading a Fisher cache that was computed without fc.
            _is_fc = name.startswith("fc.") or name.startswith("module.fc.")
            if not _is_fc:
                named_params.append((name, param))
        elif scope in {"full", "full_model", "all", "params", "weights", "ewc", "protected"}:
            named_params.append((name, param))
        else:
            if scope in name:
                named_params.append((name, param))

    if not named_params and scope != "trainable":
        return select_eval_named_params(model, "trainable")
    return named_params


def select_eval_params(model, scope: str = "lora") -> List[torch.nn.Parameter]:
    return [param for _, param in select_eval_named_params(model, scope)]


def _param_slice_map(named_params: Sequence[Tuple[str, torch.nn.Parameter]]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    ptr = 0
    for name, param in named_params:
        n = int(param.numel())
        meta = {
            "start": ptr,
            "end": ptr + n,
            "shape": tuple(param.shape),
            "numel": n,
            "aliases": [name],
        }
        out[name] = meta
        for prefix in ("model.", "module."):
            if name.startswith(prefix):
                alias = name[len(prefix):]
                meta["aliases"].append(alias)
                out.setdefault(alias, meta)
        ptr += n
    return out


def _flatten_tensors_like(params, tensors, device):
    parts = []
    for param, tensor in zip(params, tensors):
        if tensor is None:
            parts.append(torch.zeros(param.numel(), device=device, dtype=param.dtype))
        else:
            parts.append(tensor.reshape(-1))
    if not parts:
        return torch.zeros((), device=device)
    return torch.cat(parts)


def _add_vector_to_params(params: List[torch.nn.Parameter], vec: torch.Tensor):
    ptr = 0
    for param in params:
        n = param.numel()
        param.data.add_(vec[ptr: ptr + n].view_as(param))
        ptr += n


def _clone_params(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [param.detach().cpu().clone() for param in params]


def _restore_params(params: List[torch.nn.Parameter], saved: List[torch.Tensor]):
    for param, value in zip(params, saved):
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def collect_fixed_batches(
    loader,
    device,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Cache (inputs, targets) tensors so all FQA loss calls see identical data.

    Bypasses dataset-level random augmentation by storing the already-drawn
    tensors.  All subsequent loss evaluations on these batches are deterministic.
    """
    fixed: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        fixed.append((inputs.detach(), targets.detach()))
        if max_batches is not None and len(fixed) >= max_batches:
            break
    return fixed


def average_loss_on_fixed_batches(
    model,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    class_slice: Optional[Tuple[int, int]] = None,
) -> float:
    """Mean cross-entropy over pre-cached (inputs, targets) pairs.

    Model must already be in eval mode before calling. ``targets`` in
    ``fixed_batches`` are task-local (shifted by ``task_start``, see
    ``_current_task_batch``/``collect_fixed_batches``). Most incremental
    heads used in this codebase (``IncrementalNet`` + a single growing
    ``fc``) return *cumulative* logits spanning every class seen so far, not
    a per-task-local slice — pass ``class_slice=(task_start, task_end)`` so
    the local targets line up with the matching logit columns. Omitting it
    is only correct for task 0 (where local and global ranges coincide).
    """
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in fixed_batches:
            loss = _forward_current_task_loss(model, inputs, targets, class_slice=class_slice)
            total_loss += float(loss.detach().item())
            total_samples += int(targets.numel())
    return total_loss / max(1, total_samples)


def average_accuracy_on_fixed_batches(
    model,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    class_slice: Optional[Tuple[int, int]] = None,
) -> float:
    """Mean top-1 accuracy (0-100) over pre-cached (inputs, targets) pairs.

    See ``average_loss_on_fixed_batches`` for why ``class_slice`` matters for
    any task beyond task 0.
    """
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in fixed_batches:
            try:
                outputs = model(inputs, use_new=True)
            except TypeError:
                outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            if class_slice is not None:
                logits = logits[:, class_slice[0]:class_slice[1]]
            preds = logits.argmax(dim=1)
            total_correct += int(preds.eq(targets).sum().item())
            total_samples += int(targets.numel())
    return 100.0 * total_correct / max(1, total_samples)


def grad_vector_on_fixed_batches(
    model,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """Gradient of the same fixed-batch loss used by paired FQA."""
    if not tensors:
        return torch.zeros((), device=next(model.parameters()).device), 0.0

    grad_sums = [torch.zeros_like(tensor, device=tensor.device) for tensor in tensors]
    total_loss = 0.0
    total_samples = 0
    model.eval()
    for inputs, targets in fixed_batches:
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets)
            grads = torch.autograd.grad(
                loss,
                tensors,
                allow_unused=True,
                retain_graph=False,
            )
        for idx, grad in enumerate(grads):
            if grad is not None:
                grad_sums[idx].add_(grad.detach())
        total_loss += float(loss.detach().item())
        total_samples += int(targets.numel())

    denom = max(1, total_samples)
    grad_vec = torch.cat([(grad / denom).reshape(-1) for grad in grad_sums])
    return grad_vec, total_loss / denom


def _iter_delta_w_external_slots(model):
    model = unwrap_model(model)
    for module in model.modules():
        if (
            hasattr(module, "lora_new_B_k")
            and hasattr(module, "lora_new_A_k")
            and hasattr(module, "external_delta_w_k")
        ):
            base = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
            yield module, "external_delta_w_k", base
        if (
            hasattr(module, "lora_new_B_v")
            and hasattr(module, "lora_new_A_v")
            and hasattr(module, "external_delta_w_v")
        ):
            base = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
            yield module, "external_delta_w_v", base
        if (
            hasattr(module, "linear_b_q")
            and hasattr(module, "linear_a_q")
            and hasattr(module, "external_delta_w_q")
        ):
            base = module.linear_b_q.weight @ module.linear_a_q.weight
            yield module, "external_delta_w_q", base
        if (
            hasattr(module, "linear_b_v")
            and hasattr(module, "linear_a_v")
            and hasattr(module, "external_delta_w_v")
        ):
            base = module.linear_b_v.weight @ module.linear_a_v.weight
            yield module, "external_delta_w_v", base


@contextmanager
def external_delta_w_coordinates(model):
    """Expose effective Delta-W matrices as additive leaf tensors.

    Attention_LoRA supports an ``external_delta_w_*`` additive path.  Setting a
    zero leaf tensor there makes the loss differentiable directly with respect
    to the dense effective update ΔW = B@A, while leaving the LoRA factors
    unchanged.
    """
    slots = list(_iter_delta_w_external_slots(model))
    saved = [(module, attr, getattr(module, attr, None)) for module, attr, _ in slots]
    coords = []
    for module, attr, base in slots:
        tensor = torch.zeros_like(base.detach(), requires_grad=True)
        setattr(module, attr, tensor)
        coords.append(tensor)
    try:
        yield coords
    finally:
        for module, attr, value in saved:
            setattr(module, attr, value)


def _delta_w_coordinate_count(model) -> int:
    return int(sum(base.numel() for _, _, base in _iter_delta_w_external_slots(model)))


def _iter_lora_factor_pairs(model):
    model = unwrap_model(model)
    for module in model.modules():
        if hasattr(module, "lora_new_A_k") and hasattr(module, "lora_new_B_k"):
            yield module, "k", module.lora_new_A_k.weight, module.lora_new_B_k.weight
        if hasattr(module, "lora_new_A_v") and hasattr(module, "lora_new_B_v"):
            yield module, "v", module.lora_new_A_v.weight, module.lora_new_B_v.weight
        if hasattr(module, "linear_a_q") and hasattr(module, "linear_b_q"):
            yield module, "q", module.linear_a_q.weight, module.linear_b_q.weight
        if hasattr(module, "linear_a_v") and hasattr(module, "linear_b_v"):
            yield module, "v", module.linear_a_v.weight, module.linear_b_v.weight


def _lora_factor_delta_w_shapes(model) -> List[torch.Size]:
    return [
        torch.Size((b_param.shape[0], a_param.shape[1]))
        for _, _, a_param, b_param in _iter_lora_factor_pairs(model)
    ]


def _split_vector_to_param_map(
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    pieces: Dict[int, torch.Tensor] = {}
    ptr = 0
    for param in params:
        n = param.numel()
        pieces[id(param)] = vec[ptr: ptr + n].view_as(param)
        ptr += n
    return pieces


def _lora_factor_delta_w_components(
    model,
    params: List[torch.nn.Parameter],
    factor_vec: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return raw (tangent, normal) Delta-W components induced by a factor vector.

    For a factor perturbation (dA, dB), the effective update is
    dW = B dA + dB A + dB dA.  The first two terms are the LoRA tangent-space
    direction; the last term is the bilinear normal component from applying an
    actual A/B perturbation rather than an infinitesimal tangent vector.
    """
    pieces = _split_vector_to_param_map(params, factor_vec)
    components: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for _, _, a_param, b_param in _iter_lora_factor_pairs(model):
            d_a = pieces.get(id(a_param))
            d_b = pieces.get(id(b_param))
            if d_a is None:
                d_a = torch.zeros_like(a_param)
            if d_b is None:
                d_b = torch.zeros_like(b_param)

            a_base = a_param.detach()
            b_base = b_param.detach()
            d_a = d_a.detach().to(device=a_base.device, dtype=a_base.dtype)
            d_b = d_b.detach().to(device=b_base.device, dtype=b_base.dtype)

            tangent = b_base @ d_a + d_b @ a_base
            normal = d_b @ d_a
            components.append((tangent, normal))
    return components


def _prepare_delta_w_fisher_blocks(
    omega_W,
    shapes: Sequence[torch.Size],
    device,
    dtype,
) -> Optional[List[torch.Tensor]]:
    if not omega_W or len(omega_W) != len(shapes):
        return None
    blocks: List[torch.Tensor] = []
    for fisher, shape in zip(omega_W, shapes):
        if fisher is None or tuple(fisher.shape) != tuple(shape):
            return None
        blocks.append(fisher.detach().to(device=device, dtype=dtype).clamp_min(0))
    return blocks


def _unflatten_delta_w_vector_to_shapes(
    vec: Optional[torch.Tensor],
    shapes: Sequence[torch.Size],
    device,
    dtype,
) -> Optional[List[torch.Tensor]]:
    if vec is None:
        return None
    expected = int(sum(math.prod(tuple(shape)) for shape in shapes))
    if int(vec.numel()) != expected:
        return None
    blocks: List[torch.Tensor] = []
    ptr = 0
    flat = vec.detach().to(device=device, dtype=dtype)
    for shape in shapes:
        n = int(math.prod(tuple(shape)))
        blocks.append(flat[ptr: ptr + n].view(shape))
        ptr += n
    return blocks


def _lora_factor_component_stats(
    components: List[Tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    fisher_blocks: Optional[List[torch.Tensor]] = None,
    delta_w_grad_blocks: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    scale = float(scale)
    scale2 = scale * scale
    tangent_norm_sq = 0.0
    normal_norm_sq = 0.0
    actual_norm_sq = 0.0
    fisher_quad_tangent = 0.0
    fisher_quad_actual = 0.0
    delta_w_linear_tangent = 0.0
    delta_w_linear_actual = 0.0
    delta_w_linear_normal = 0.0

    fisher_available = fisher_blocks is not None and len(fisher_blocks) == len(components)
    linear_available = delta_w_grad_blocks is not None and len(delta_w_grad_blocks) == len(components)

    for idx, (raw_tangent, raw_normal) in enumerate(components):
        tangent = raw_tangent * scale
        normal = raw_normal * scale2
        actual = tangent + normal

        tangent_norm_sq += float(tangent.pow(2).sum().detach().item())
        normal_norm_sq += float(normal.pow(2).sum().detach().item())
        actual_norm_sq += float(actual.pow(2).sum().detach().item())

        if fisher_available:
            fisher = fisher_blocks[idx].to(device=tangent.device, dtype=tangent.dtype)
            if tuple(fisher.shape) == tuple(tangent.shape):
                fisher_quad_tangent += float((fisher * tangent.pow(2)).sum().detach().item())
                fisher_quad_actual += float((fisher * actual.pow(2)).sum().detach().item())
            else:
                fisher_available = False

        if linear_available:
            grad = delta_w_grad_blocks[idx].to(device=tangent.device, dtype=tangent.dtype)
            if tuple(grad.shape) == tuple(tangent.shape):
                delta_w_linear_tangent += float((grad * tangent).sum().detach().item())
                delta_w_linear_actual += float((grad * actual).sum().detach().item())
                delta_w_linear_normal += float((grad * normal).sum().detach().item())
            else:
                linear_available = False

    return {
        "delta_w_tangent_norm": math.sqrt(max(0.0, tangent_norm_sq)),
        "delta_w_normal_norm": math.sqrt(max(0.0, normal_norm_sq)),
        "delta_w_actual_norm": math.sqrt(max(0.0, actual_norm_sq)),
        "fisher_available": bool(fisher_available),
        "fisher_quad_tangent": 0.5 * fisher_quad_tangent if fisher_available else 0.0,
        "fisher_quad_actual": 0.5 * fisher_quad_actual if fisher_available else 0.0,
        "delta_w_linear_available": bool(linear_available),
        "delta_w_linear_tangent": delta_w_linear_tangent if linear_available else 0.0,
        "delta_w_linear_actual": delta_w_linear_actual if linear_available else 0.0,
        "delta_w_linear_normal": delta_w_linear_normal if linear_available else 0.0,
    }


def _add_vector_to_tensors(tensors: List[torch.Tensor], vec: torch.Tensor):
    ptr = 0
    for tensor in tensors:
        n = tensor.numel()
        tensor.data.add_(vec[ptr: ptr + n].view_as(tensor))
        ptr += n


def _zero_tensors(tensors: List[torch.Tensor]):
    for tensor in tensors:
        tensor.data.zero_()


def _flatten_fisher_diag(omega_W, device, dtype, eps: float = 1e-12) -> Optional[torch.Tensor]:
    if not omega_W:
        return None
    parts = []
    for fisher in omega_W:
        if fisher is None:
            continue
        parts.append(fisher.detach().to(device=device, dtype=dtype).reshape(-1).clamp_min(0))
    if not parts:
        return None
    flat = torch.cat(parts)
    return flat if int(flat.numel()) > 0 else None


def _prepare_kfac_quadratic_blocks(
    kfac_blocks,
    dim: int,
    device,
    dtype,
    eps: float = 1e-12,
    param_slices: Optional[Dict[str, Dict]] = None,
    require_full_coverage: bool = True,
) -> Dict:
    """Prepare full-block KFAC factors aligned to the flattened parameter vector.

    Prefer name-based alignment through ``param_slices``.  Full KFAC blocks may
    cover a layer weight plus its bias by using an augmented activation factor.
    Diagonal fallback blocks cover one parameter by name.
    """
    result = {
        "available": False,
        "blocks": [],
        "total_numel": 0,
        "covered_numel": 0,
        "fallback_numel": 0,
        "missing_numel": 0,
        "kfac_blocks": 0,
        "fallback_blocks": 0,
        "missing_blocks": 0,
        "coverage_ratio": 0.0,
        "fallback_ratio": 0.0,
        "missing_ratio": 0.0,
    }
    if not kfac_blocks:
        return result

    factor_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    if factor_dtype not in {torch.float32, torch.float64}:
        factor_dtype = torch.float32

    seen_names = set()

    def _shape_numel(shape) -> int:
        return int(math.prod(tuple(int(x) for x in shape)))

    def _add_missing(numel: int):
        result["missing_numel"] += int(numel)
        result["missing_blocks"] += 1

    def _slice_for_name(name: Optional[str]):
        if not name or not param_slices:
            return None
        return param_slices.get(name)

    def _mark_seen(meta: Dict):
        for alias in meta.get("aliases", []):
            seen_names.add(alias)

    ptr = 0
    for block_idx, block in enumerate(kfac_blocks):
        if not isinstance(block, dict):
            result["error"] = f"block {block_idx} is not a dict"
            return result

        param_name = block.get("param_name", None)
        weight_name = block.get("weight_name", param_name)
        bias_name = block.get("bias_name", None)
        G = block.get("G", None)
        A = block.get("A", None)
        diag = block.get("diag", None)

        if param_slices and G is not None and A is not None and weight_name:
            w_meta = _slice_for_name(weight_name)
            b_meta = _slice_for_name(bias_name)
            if w_meta is None:
                continue
            weight_shape = tuple(block.get("weight_shape", w_meta["shape"]))
            out_dim = int(weight_shape[0]) if weight_shape else 0
            weight_cols = _shape_numel(weight_shape[1:]) if len(weight_shape) > 1 else 1
            matrix_shape = tuple(block.get(
                "matrix_shape",
                (out_dim, weight_cols + (1 if b_meta is not None else 0)),
            ))
            use_full = (
                len(matrix_shape) == 2
                and tuple(G.shape) == (matrix_shape[0], matrix_shape[0])
                and tuple(A.shape) == (matrix_shape[1], matrix_shape[1])
            )
            if use_full:
                G_t = G.detach().to(device=device, dtype=factor_dtype)
                A_t = A.detach().to(device=device, dtype=factor_dtype)
                G_t = 0.5 * (G_t + G_t.transpose(0, 1))
                A_t = 0.5 * (A_t + A_t.transpose(0, 1))
                if eps > 0:
                    G_t = G_t + eps * torch.eye(G_t.shape[0], device=device, dtype=factor_dtype)
                    A_t = A_t + eps * torch.eye(A_t.shape[0], device=device, dtype=factor_dtype)
                result["blocks"].append({
                    "kind": "kfac",
                    "weight_name": weight_name,
                    "bias_name": bias_name if b_meta is not None else None,
                    "weight_slice": (int(w_meta["start"]), int(w_meta["end"])),
                    "bias_slice": (
                        (int(b_meta["start"]), int(b_meta["end"]))
                        if b_meta is not None else None
                    ),
                    "weight_shape": weight_shape,
                    "matrix_shape": matrix_shape,
                    "G": G_t,
                    "A": A_t,
                })
                result["covered_numel"] += int(w_meta["numel"])
                _mark_seen(w_meta)
                if b_meta is not None:
                    result["covered_numel"] += int(b_meta["numel"])
                    _mark_seen(b_meta)
                result["kfac_blocks"] += 1
                continue

        if param_slices and diag is not None and param_name:
            meta = _slice_for_name(param_name)
            if meta is None:
                continue
            shape = tuple(meta["shape"])
            diag_t = diag.detach().to(device=device, dtype=factor_dtype).reshape(shape).clamp_min(eps)
            result["blocks"].append({
                "kind": "diag",
                "param_name": param_name,
                "start": int(meta["start"]),
                "end": int(meta["end"]),
                "shape": shape,
                "diag": diag_t,
            })
            result["fallback_numel"] += int(meta["numel"])
            result["fallback_blocks"] += 1
            _mark_seen(meta)
            continue

        shape = block.get("shape", None)
        if shape is None and diag is not None:
            shape = tuple(diag.shape)
        if shape is None:
            result["error"] = f"block {block_idx} has no shape"
            return result
        shape = tuple(int(x) for x in shape)
        numel = _shape_numel(shape)
        start, end = ptr, ptr + numel
        ptr = end
        if end > int(dim):
            result["error"] = f"block {block_idx} exceeds dim: end={end} dim={int(dim)}"
            return result

        use_full = (
            G is not None
            and A is not None
            and len(shape) == 2
            and tuple(G.shape) == (shape[0], shape[0])
            and tuple(A.shape) == (shape[1], shape[1])
        )
        if use_full:
            G_t = G.detach().to(device=device, dtype=factor_dtype)
            A_t = A.detach().to(device=device, dtype=factor_dtype)
            G_t = 0.5 * (G_t + G_t.transpose(0, 1))
            A_t = 0.5 * (A_t + A_t.transpose(0, 1))
            if eps > 0:
                G_t = G_t + eps * torch.eye(G_t.shape[0], device=device, dtype=factor_dtype)
                A_t = A_t + eps * torch.eye(A_t.shape[0], device=device, dtype=factor_dtype)
            result["blocks"].append({
                "kind": "kfac",
                "start": start,
                "end": end,
                "shape": shape,
                "matrix_shape": shape,
                "G": G_t,
                "A": A_t,
            })
            result["covered_numel"] += numel
            result["kfac_blocks"] += 1
        elif diag is not None:
            diag_t = diag.detach().to(device=device, dtype=factor_dtype).reshape(shape).clamp_min(eps)
            result["blocks"].append({
                "kind": "diag",
                "start": start,
                "end": end,
                "shape": shape,
                "diag": diag_t,
            })
            result["fallback_numel"] += numel
            result["fallback_blocks"] += 1
        else:
            _add_missing(numel)

    if param_slices:
        result["total_numel"] = int(dim)
        unique = {}
        for meta in param_slices.values():
            unique[(int(meta["start"]), int(meta["end"]))] = meta
        for meta in unique.values():
            if not any(alias in seen_names for alias in meta.get("aliases", [])):
                _add_missing(int(meta["numel"]))
    else:
        result["total_numel"] = ptr
        if ptr < int(dim):
            _add_missing(int(dim) - ptr)

    denom = max(1, int(dim))
    result["coverage_ratio"] = float(result["covered_numel"] / denom)
    result["fallback_ratio"] = float(result["fallback_numel"] / denom)
    result["missing_ratio"] = float(result["missing_numel"] / denom)
    result["available"] = bool(result["blocks"]) and (
        result["missing_numel"] == 0 if require_full_coverage else True
    )
    return result


def _kfac_quadratic_from_direction(
    prepared_kfac: Dict,
    direction: torch.Tensor,
    include_half: bool = False,
) -> float:
    if not prepared_kfac.get("available", False):
        return float("nan")
    total = direction.new_tensor(0.0)
    for block in prepared_kfac.get("blocks", []):
        if block["kind"] == "kfac":
            if "weight_slice" in block:
                ws, we = block["weight_slice"]
                w = direction[ws:we].view(block["weight_shape"]).reshape(block["matrix_shape"][0], -1)
                bs = block.get("bias_slice", None)
                if bs is not None:
                    b_start, b_end = bs
                    b = direction[b_start:b_end].reshape(block["matrix_shape"][0], 1)
                    u = torch.cat([w, b], dim=1)
                else:
                    u = w
            else:
                u = direction[block["start"]: block["end"]].view(block["shape"])
            u = u.to(dtype=block["A"].dtype)
            # For Linear/Conv2d weight matrix shaped [out, in_flat]:
            # vec(dW)^T (A kron G) vec(dW) = tr(G dW A dW^T).
            total = total + ((block["G"].matmul(u).matmul(block["A"])) * u).sum()
        elif block["kind"] == "diag":
            u = direction[block["start"]: block["end"]].view(block["shape"]).to(dtype=block["diag"].dtype)
            total = total + (block["diag"] * u.pow(2)).sum()
    if include_half:
        total = 0.5 * total
    return float(total.detach().item())


def _add_kfac_quadratic_metadata(out: Dict[str, float], prefix: str, prepared_kfac: Dict) -> None:
    out[f"{prefix}_fisher_kind"] = "kfac_full_plus_diag_fallback"
    out[f"{prefix}_kfac_full_available"] = bool(prepared_kfac.get("available", False))
    out[f"{prefix}_kfac_full_blocks"] = int(prepared_kfac.get("kfac_blocks", 0))
    out[f"{prefix}_kfac_full_fallback_blocks"] = int(prepared_kfac.get("fallback_blocks", 0))
    out[f"{prefix}_kfac_full_missing_blocks"] = int(prepared_kfac.get("missing_blocks", 0))
    out[f"{prefix}_kfac_full_numel"] = int(prepared_kfac.get("total_numel", 0))
    out[f"{prefix}_kfac_full_covered_numel"] = int(prepared_kfac.get("covered_numel", 0))
    out[f"{prefix}_kfac_full_fallback_numel"] = int(prepared_kfac.get("fallback_numel", 0))
    out[f"{prefix}_kfac_full_missing_numel"] = int(prepared_kfac.get("missing_numel", 0))
    out[f"{prefix}_kfac_full_coverage_ratio"] = float(prepared_kfac.get("coverage_ratio", 0.0))
    out[f"{prefix}_kfac_full_fallback_ratio"] = float(prepared_kfac.get("fallback_ratio", 0.0))
    out[f"{prefix}_kfac_full_missing_ratio"] = float(prepared_kfac.get("missing_ratio", 0.0))
    if "error" in prepared_kfac:
        out[f"{prefix}_kfac_full_error"] = str(prepared_kfac["error"])


def _sample_unit_direction(dim: int, device, dtype, eps: float = 1e-12) -> torch.Tensor:
    direction = torch.randn(dim, device=device, dtype=dtype)
    return direction / (direction.norm() + eps)


def _sample_fisher_direction(
    dim: int,
    device,
    dtype,
    fisher_diag: Optional[torch.Tensor],
    mode: str = "random",
    topk_frac: float = 0.01,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sample a unit perturbation direction according to `mode`.

    Modes:
      random          – uniform on S^{d-1} (default)
      fisher_topk     – random in top-k% Fisher coordinates, zeros elsewhere
      fisher_bottomk  – random in bottom-k% Fisher coordinates, zeros elsewhere
      fisher_weighted – Gaussian scaled by sqrt(F_i)
    """
    mode = str(mode or "random").lower()

    if mode == "random" or fisher_diag is None or int(fisher_diag.numel()) != int(dim):
        return _sample_unit_direction(dim, device, dtype, eps=eps)

    f = fisher_diag.detach().to(device=device, dtype=dtype).reshape(-1).clamp_min(0)
    k = max(1, min(int(math.ceil(topk_frac * int(dim))), int(dim)))

    if mode == "fisher_topk":
        idx = torch.topk(f, k=k, largest=True, sorted=False).indices
    elif mode == "fisher_bottomk":
        idx = torch.topk(f, k=k, largest=False, sorted=False).indices
    elif mode == "fisher_weighted":
        z = torch.randn(dim, device=device, dtype=dtype)
        direction = z * torch.sqrt(f + eps)
        return direction / (direction.norm() + eps)
    else:
        return _sample_unit_direction(dim, device, dtype, eps=eps)

    direction = torch.zeros(dim, device=device, dtype=dtype)
    direction[idx] = torch.randn(int(idx.numel()), device=device, dtype=dtype)
    return direction / (direction.norm() + eps)


def _radius_tag(radius: float) -> str:
    return _fraction_tag(radius)


def _current_task_batch(inputs, targets, task_start: int, task_end: int, device):
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
    if mask.numel() == 0:
        return None, None
    inputs = torch.index_select(inputs, 0, mask)
    targets = torch.index_select(targets, 0, mask) - task_start
    return inputs, targets


def _forward_current_task_loss(model, inputs, targets, class_slice: Optional[Tuple[int, int]] = None):
    try:
        outputs = model(inputs, use_new=True)
    except TypeError:
        outputs = model(inputs)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    if class_slice is not None:
        logits = logits[:, class_slice[0]:class_slice[1]]
    return F.cross_entropy(logits, targets, reduction="sum")


def average_current_task_loss(
    model,
    loader,
    device,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> float:
    total_loss = 0.0
    total_samples = 0
    batches = 0
    model.eval()
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
            if inputs is None:
                continue
            loss = _forward_current_task_loss(model, inputs, targets)
            total_loss += float(loss.detach().item())
            total_samples += int(targets.numel())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return total_loss / max(1, total_samples)


def current_task_grad_vector(
    model,
    loader,
    device,
    params: List[torch.nn.Parameter],
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    grad_sums = [torch.zeros_like(param, device=param.device) for param in params]
    total_loss = 0.0
    total_samples = 0
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets)
            grads = torch.autograd.grad(
                loss,
                params,
                allow_unused=True,
                retain_graph=False,
            )
        for idx, grad in enumerate(grads):
            if grad is not None:
                grad_sums[idx].add_(grad.detach())
        total_loss += float(loss.detach().item())
        total_samples += int(targets.numel())
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    denom = max(1, total_samples)
    grad_vec = torch.cat([(grad / denom).reshape(-1) for grad in grad_sums])
    return grad_vec, total_loss / denom


def current_task_delta_w_grad_vector(
    model,
    loader,
    device,
    coords: List[torch.Tensor],
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    grad_sums = [torch.zeros_like(coord, device=coord.device) for coord in coords]
    total_loss = 0.0
    total_samples = 0
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets)
            grads = torch.autograd.grad(
                loss,
                coords,
                allow_unused=True,
                retain_graph=False,
            )
        for idx, grad in enumerate(grads):
            if grad is not None:
                grad_sums[idx].add_(grad.detach())
        total_loss += float(loss.detach().item())
        total_samples += int(targets.numel())
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    denom = max(1, total_samples)
    grad_vec = torch.cat([(grad / denom).reshape(-1) for grad in grad_sums])
    return grad_vec, total_loss / denom


def hessian_vector_product(
    model,
    loader,
    device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    vec = vec.to(device=device)
    out = torch.zeros_like(vec)
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets) / max(1, targets.numel())
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                allow_unused=True,
                retain_graph=True,
            )
            flat_grad = _flatten_tensors_like(params, grads, device)
            dot = torch.dot(flat_grad, vec)
            if not dot.requires_grad:
                hv = torch.zeros_like(vec)
            else:
                hv_tensors = torch.autograd.grad(
                    dot,
                    params,
                    allow_unused=True,
                    retain_graph=False,
                )
                hv = _flatten_tensors_like(params, hv_tensors, device).detach()
        out.add_(hv)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return out / float(max(1, batches))


def delta_w_hessian_vector_product(
    model,
    loader,
    device,
    coords: List[torch.Tensor],
    vec: torch.Tensor,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    vec = vec.to(device=device)
    out = torch.zeros_like(vec)
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets) / max(1, targets.numel())
            grads = torch.autograd.grad(
                loss,
                coords,
                create_graph=True,
                allow_unused=True,
                retain_graph=True,
            )
            flat_grad = _flatten_tensors_like(coords, grads, device)
            dot = torch.dot(flat_grad, vec)
            if not dot.requires_grad:
                hv = torch.zeros_like(vec)
            else:
                hv_tensors = torch.autograd.grad(
                    dot,
                    coords,
                    allow_unused=True,
                    retain_graph=False,
                )
                hv = _flatten_tensors_like(coords, hv_tensors, device).detach()
        out.add_(hv)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return out / float(max(1, batches))


def empirical_fisher_vector_product(
    model,
    loader,
    device,
    params: List[torch.nn.Parameter],
    vec: torch.Tensor,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    vec = vec.to(device=device)
    out = torch.zeros_like(vec)
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets) / max(1, targets.numel())
            grads = torch.autograd.grad(
                loss,
                params,
                allow_unused=True,
                retain_graph=False,
            )
        flat_grad = _flatten_tensors_like(params, grads, device).detach()
        out.add_(flat_grad * torch.dot(flat_grad, vec))
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return out / float(max(1, batches))


def delta_w_empirical_fisher_vector_product(
    model,
    loader,
    device,
    coords: List[torch.Tensor],
    vec: torch.Tensor,
    task_start: int,
    task_end: int,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    vec = vec.to(device=device)
    out = torch.zeros_like(vec)
    batches = 0
    model.eval()
    for _, inputs, targets in loader:
        inputs, targets = _current_task_batch(inputs, targets, task_start, task_end, device)
        if inputs is None:
            continue
        with torch.enable_grad():
            loss = _forward_current_task_loss(model, inputs, targets) / max(1, targets.numel())
            grads = torch.autograd.grad(
                loss,
                coords,
                allow_unused=True,
                retain_graph=False,
            )
        flat_grad = _flatten_tensors_like(coords, grads, device).detach()
        out.add_(flat_grad * torch.dot(flat_grad, vec))
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return out / float(max(1, batches))


def power_top_eigen(mvp_fn, dim: int, device, iters: int, eps: float = 1e-12) -> float:
    if dim <= 0 or iters <= 0:
        return 0.0
    vec = torch.randn(dim, device=device)
    vec = vec / (vec.norm() + eps)
    eig = 0.0
    for _ in range(iters):
        hv = mvp_fn(vec)
        norm = hv.norm()
        if float(norm.detach().item()) <= eps:
            return 0.0
        vec = hv / (norm + eps)
        eig = float(torch.dot(vec, mvp_fn(vec)).detach().item())
    return eig


def hutchinson_trace(mvp_fn, dim: int, device, samples: int) -> float:
    if dim <= 0 or samples <= 0:
        return 0.0
    vals = []
    for _ in range(samples):
        vec = torch.empty(dim, device=device).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        vals.append(float(torch.dot(vec, mvp_fn(vec)).detach().item()))
    return float(sum(vals) / max(1, len(vals)))


def summarize_radius_fisher_quadratic_error(
    *,
    prefix: str,
    radii: Sequence[float],
    samples: int,
    base_loss: float,
    dim: int,
    device,
    dtype,
    fisher_diag: Optional[torch.Tensor],
    grad_vec: Optional[torch.Tensor],
    perturb_fn,
    restore_fn,
    loss_fn,
    acc_fn=None,
    extra_fishers: Optional[List[Tuple[str, Optional[torch.Tensor]]]] = None,
    kfac_factors: Optional[List[Dict]] = None,
    extra_kfac_factors: Optional[List[Tuple[str, Optional[List[Dict]]]]] = None,
    param_slices: Optional[Dict[str, Dict]] = None,
    partial_results_path: Optional[str] = None,
    hessian_scalars: Optional[Dict[str, float]] = None,
    direction_mode: str = "random",
    direction_topk_frac: float = 0.01,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Estimate loss-vs-Fisher-quadratic error on fixed-radius perturbation directions.

    The direction distribution is controlled by direction_mode. The default uses
    random unit directions uniformly on the sphere. ``fisher_topk`` restricts
    perturbations to the top-k% coordinates ranked by the Fisher diagonal;
    ``fisher_bottomk`` uses the bottom-k% (Fisher blind-spot) directions;
    ``fisher_weighted`` scales Gaussian noise by sqrt(F_i).

    acc_fn: optional callable () -> float that returns top-1 accuracy (0–100) for
    the current model state.  When provided, accuracy is evaluated after every
    weight perturbation and the per-radius statistics are recorded alongside loss.

    extra_fishers: optional list of (prefix, fisher_diag) pairs computed in the
    same forward-pass loop without additional model evaluations.

    kfac_factors / extra_kfac_factors: optional full-block KFAC variants aligned
    to the same flattened parameter vector.  Their quadratic uses
    tr(G dW A dW^T) on covered Linear weights and diagonal fallback blocks for
    uncovered parameters.

    partial_results_path: if set, append one JSONL record per completed radius so
    results are preserved even if the process is killed.

    hessian_scalars: pre-computed Hessian scalars (lambda_max, trace, emp_fisher_*)
    from the earlier sharpness/flatness evaluation.  Included in each JSONL record
    as reference values (e.g. Q_H upper bound = 0.5 * r^2 * lambda_max) without
    requiring any additional HVP computation.
    """
    out: Dict[str, float] = {}
    if dim <= 0 or samples <= 0 or not radii:
        return out

    out[f"{prefix}_fqa_direction_mode"] = str(direction_mode)
    out[f"{prefix}_fqa_direction_topk_frac"] = float(direction_topk_frac)
    if fisher_diag is not None and int(fisher_diag.numel()) == int(dim):
        _topk_numel = max(1, min(int(math.ceil(float(direction_topk_frac) * int(dim))), int(dim)))
    else:
        _topk_numel = 0
    out[f"{prefix}_fqa_direction_topk_numel"] = int(_topk_numel)

    prepared_main_kfac = _prepare_kfac_quadratic_blocks(
        kfac_factors, int(dim), device, dtype, eps=eps, param_slices=param_slices
    ) if kfac_factors else None
    kfac_available = bool(prepared_main_kfac and prepared_main_kfac.get("available", False))
    if prepared_main_kfac is not None:
        _add_kfac_quadratic_metadata(out, prefix, prepared_main_kfac)
        if not kfac_available:
            logging.warning(
                "[AnchorEval] skip full-KFAC quadratic term for %s: kfac numel=%s dim=%s",
                prefix,
                int(prepared_main_kfac.get("total_numel", 0)),
                int(dim),
            )

    diag_available = fisher_diag is not None and int(fisher_diag.numel()) == int(dim)
    fisher_available = bool(kfac_available or diag_available)
    if fisher_diag is not None and not diag_available:
        logging.warning(
            "[AnchorEval] skip Fisher quadratic term for %s: fisher numel=%s does not match dim=%s",
            prefix,
            int(fisher_diag.numel()),
            int(dim),
        )

    sample_count = max(0, samples)
    radius_count = len(radii)

    # Pre-compute base accuracy once before any perturbation.
    base_acc: Optional[float] = None
    if acc_fn is not None:
        base_acc = float(acc_fn())
        out[f"{prefix}_fqa_base_acc"] = base_acc
        logging.info("[FQA][%s] base_acc=%.4f%%", prefix, base_acc)

    logging.info(
        "[FQA][%s] start: radii=%s samples=%d dim=%d fisher_available=%s fisher_kind=%s",
        prefix,
        [float(radius) for radius in radii],
        sample_count,
        int(dim),
        bool(fisher_available),
        "kfac_full" if kfac_available else ("diag" if diag_available else "none"),
    )

    linear_available = grad_vec is not None and int(grad_vec.numel()) == int(dim)
    if grad_vec is not None and not linear_available:
        logging.warning(
            "[AnchorEval] skip FQA first-order term for %s: grad numel=%s does not match dim=%s",
            prefix,
            int(grad_vec.numel()),
            int(dim),
        )
    grad_vec = grad_vec.detach().to(device=device, dtype=dtype) if linear_available else None
    grad_norm = float(grad_vec.norm().item()) if grad_vec is not None else 0.0
    random_projection_coeff = math.sqrt(2.0 / (math.pi * max(1, int(dim))))

    # Prepare extra Fisher variants: validate dims, move to device.
    _extra: List[Tuple[str, bool, Optional[torch.Tensor]]] = []
    for ef_prefix, ef_fd in (extra_fishers or []):
        ef_ok = ef_fd is not None and int(ef_fd.numel()) == int(dim)
        if ef_fd is not None and not ef_ok:
            logging.warning(
                "[AnchorEval] skip Fisher quadratic term for %s: fisher numel=%s does not match dim=%s",
                ef_prefix, int(ef_fd.numel()), int(dim),
            )
        _extra.append((ef_prefix, ef_ok, ef_fd.detach().to(device=device, dtype=dtype) if ef_ok else None))
    if _extra:
        logging.info(
            "[FQA][%s] co-computing %d extra Fisher variants in same pass: %s",
            prefix, len(_extra), [p for p, _, _ in _extra],
        )

    _extra_kfac: List[Tuple[str, bool, Optional[Dict]]] = []
    for kf_prefix, kf_blocks in (extra_kfac_factors or []):
        prepared = _prepare_kfac_quadratic_blocks(
            kf_blocks, int(dim), device, dtype, eps=eps, param_slices=param_slices
        ) if kf_blocks else None
        kf_ok = bool(prepared and prepared.get("available", False))
        if prepared is not None:
            _add_kfac_quadratic_metadata(out, kf_prefix, prepared)
        if kf_blocks and not kf_ok:
            logging.warning(
                "[AnchorEval] skip full-KFAC quadratic term for %s: kfac numel=%s dim=%s",
                kf_prefix,
                int((prepared or {}).get("total_numel", 0)),
                int(dim),
            )
        _extra_kfac.append((kf_prefix, kf_ok, prepared if kf_ok else None))
    if _extra_kfac:
        logging.info(
            "[FQA][%s] co-computing %d extra full-KFAC variants in same pass: %s",
            prefix, len(_extra_kfac), [p for p, _, _ in _extra_kfac],
        )

    for radius_idx, radius in enumerate(radii, start=1):
        radius = float(radius)
        tag = _radius_tag(radius)
        loss_deltas = []
        perturbed_losses = []
        perturbed_accs = []
        fisher_quads = []
        linear_terms = []
        linear_corrected_deltas = []
        linear_corrected_errors = []
        linear_corrected_signed_errors = []
        errors = []
        signed_errors = []
        direction_norms = []
        progress_every = max(1, sample_count // 10) if sample_count else 1

        # Per-radius accumulators for extra Fisher variants.
        extra_rd = [
            {"quads": [], "signed": [], "errors": [], "lc_signed": [], "lc_errors": []}
            for _ in _extra
        ]
        extra_kfac_rd = [
            {"quads": [], "signed": [], "errors": [], "lc_signed": [], "lc_errors": []}
            for _ in _extra_kfac
        ]

        logging.info(
            "[FQA][%s] radius %d/%d r=%g start",
            prefix,
            radius_idx,
            radius_count,
            radius,
        )

        for sample_idx in range(1, sample_count + 1):
            direction = _sample_fisher_direction(
                dim, device, dtype,
                fisher_diag=fisher_diag if diag_available else None,
                mode=direction_mode,
                topk_frac=direction_topk_frac,
                eps=eps,
            )
            perturb = direction * radius
            perturb_fn(perturb)
            try:
                loss = loss_fn()
                if acc_fn is not None:
                    acc = acc_fn()
                    perturbed_accs.append(float(acc))
            finally:
                restore_fn()

            loss_delta = float(loss - base_loss)
            perturbed_losses.append(float(loss))
            direction_norms.append(float(direction.norm().detach().item()))
            loss_deltas.append(loss_delta)

            linear_term = None
            if grad_vec is not None:
                linear_term = float(torch.dot(grad_vec, perturb).detach().item())
                linear_corrected_delta = loss_delta - linear_term
                linear_terms.append(float(linear_term))
                linear_corrected_deltas.append(float(linear_corrected_delta))

            dir_sq = direction.pow(2)

            if fisher_available:
                if kfac_available:
                    u_f_u = _kfac_quadratic_from_direction(prepared_main_kfac, direction)
                else:
                    u_f_u = float((fisher_diag * dir_sq).sum().detach().item())
                quad = 0.5 * radius * radius * u_f_u
                signed_error = loss_delta - quad
                fisher_quads.append(float(quad))
                signed_errors.append(float(signed_error))
                errors.append(float(abs(signed_error)))
                if linear_term is not None:
                    linear_corrected_signed_error = loss_delta - linear_term - quad
                    linear_corrected_signed_errors.append(float(linear_corrected_signed_error))
                    linear_corrected_errors.append(float(abs(linear_corrected_signed_error)))

            # Extra Fisher variants: cheap dot-product, no additional forward pass.
            for ef_idx, (_, ef_ok, ef_f) in enumerate(_extra):
                if ef_ok:
                    ef_u_f_u = float((ef_f * dir_sq).sum().detach().item())
                    ef_quad = 0.5 * radius * radius * ef_u_f_u
                    ef_signed_error = loss_delta - ef_quad
                    extra_rd[ef_idx]["quads"].append(ef_quad)
                    extra_rd[ef_idx]["signed"].append(ef_signed_error)
                    extra_rd[ef_idx]["errors"].append(abs(ef_signed_error))
                    if linear_term is not None:
                        ef_lc = loss_delta - linear_term - ef_quad
                        extra_rd[ef_idx]["lc_signed"].append(ef_lc)
                        extra_rd[ef_idx]["lc_errors"].append(abs(ef_lc))

            # Extra full-KFAC variants: same sampled directions and forward pass.
            for kf_idx, (_, kf_ok, kf_prepared) in enumerate(_extra_kfac):
                if kf_ok:
                    kf_u_f_u = _kfac_quadratic_from_direction(kf_prepared, direction)
                    kf_quad = 0.5 * radius * radius * kf_u_f_u
                    kf_signed_error = loss_delta - kf_quad
                    extra_kfac_rd[kf_idx]["quads"].append(kf_quad)
                    extra_kfac_rd[kf_idx]["signed"].append(kf_signed_error)
                    extra_kfac_rd[kf_idx]["errors"].append(abs(kf_signed_error))
                    if linear_term is not None:
                        kf_lc = loss_delta - linear_term - kf_quad
                        extra_kfac_rd[kf_idx]["lc_signed"].append(kf_lc)
                        extra_kfac_rd[kf_idx]["lc_errors"].append(abs(kf_lc))

            if sample_idx == sample_count or sample_idx % progress_every == 0:
                logging.info(
                    "[FQA][%s] radius %d/%d r=%g samples %d/%d",
                    prefix,
                    radius_idx,
                    radius_count,
                    radius,
                    sample_idx,
                    sample_count,
                )

        out[f"{prefix}_fqa_r{tag}_radius"] = radius
        out[f"{prefix}_fqa_r{tag}_samples"] = int(samples)
        out[f"{prefix}_fqa_r{tag}_fisher_available"] = bool(fisher_available)
        out[f"{prefix}_fqa_r{tag}_linear_available"] = bool(linear_available)
        if linear_available:
            out[f"{prefix}_fqa_r{tag}_grad_norm"] = float(grad_norm)
            out[f"{prefix}_fqa_r{tag}_stationarity_bound"] = float(grad_norm * radius)
            out[f"{prefix}_fqa_r{tag}_stationarity_bound_over_r2"] = (
                float(grad_norm / radius) if abs(radius) > eps else 0.0
            )
            random_abs = float(radius * grad_norm * random_projection_coeff)
            out[f"{prefix}_fqa_r{tag}_random_linear_abs_theory"] = random_abs
            out[f"{prefix}_fqa_r{tag}_random_linear_abs_theory_over_r2"] = (
                float(random_abs / (radius * radius)) if abs(radius) > eps else 0.0
            )
        # Sanity check: at r=0 every loss_delta must be ~0 (paired eval guarantee).
        if abs(radius) < 1e-12 and loss_deltas:
            zero_bias = float(torch.tensor(loss_deltas).mean().item())
            out[f"{prefix}_fqa_r{tag}_zero_radius_bias"] = zero_bias
            if abs(zero_bias) > 1e-4:
                logging.warning(
                    "[FQA][%s] r=0 bias=%.6f > 1e-4: base_loss and perturbed_loss "
                    "are NOT from the same data. Use collect_fixed_batches for "
                    "deterministic paired evaluation.",
                    prefix,
                    zero_bias,
                )
        if direction_norms:
            norm_tensor = torch.tensor(direction_norms)
            out[f"{prefix}_fqa_r{tag}_direction_norm_mean"] = float(norm_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_direction_norm_std"] = float(norm_tensor.std(unbiased=False).item())
        if perturbed_losses:
            perturbed_tensor = torch.tensor(perturbed_losses)
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_mean"] = float(perturbed_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_std"] = float(perturbed_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_min"] = float(perturbed_tensor.min().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_max"] = float(perturbed_tensor.max().item())
        if perturbed_accs:
            acc_tensor = torch.tensor(perturbed_accs)
            out[f"{prefix}_fqa_r{tag}_perturbed_acc_mean"] = float(acc_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_acc_std"] = float(acc_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_perturbed_acc_min"] = float(acc_tensor.min().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_acc_max"] = float(acc_tensor.max().item())
            if base_acc is not None:
                acc_delta_tensor = acc_tensor - base_acc
                out[f"{prefix}_fqa_r{tag}_acc_delta_mean"] = float(acc_delta_tensor.mean().item())
                out[f"{prefix}_fqa_r{tag}_acc_delta_std"] = float(acc_delta_tensor.std(unbiased=False).item())
                out[f"{prefix}_fqa_r{tag}_acc_delta_min"] = float(acc_delta_tensor.min().item())
                out[f"{prefix}_fqa_r{tag}_acc_delta_max"] = float(acc_delta_tensor.max().item())
        if loss_deltas:
            loss_tensor = torch.tensor(loss_deltas)
            out[f"{prefix}_fqa_r{tag}_loss_delta_mean"] = float(loss_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_std"] = float(loss_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_min"] = float(loss_tensor.min().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_max"] = float(loss_tensor.max().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_abs_mean"] = float(loss_tensor.abs().mean().item())
        if linear_terms:
            linear_tensor = torch.tensor(linear_terms)
            linear_abs = linear_tensor.abs()
            corrected_tensor = torch.tensor(linear_corrected_deltas)
            corrected_abs = corrected_tensor.abs()
            out[f"{prefix}_fqa_r{tag}_linear_term_mean"] = float(linear_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_std"] = float(linear_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_mean"] = float(linear_abs.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_max"] = float(linear_abs.max().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_over_r2_mean"] = (
                float(linear_abs.mean().item() / (radius * radius)) if abs(radius) > eps else 0.0
            )
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_mean"] = float(corrected_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_std"] = float(corrected_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_abs_mean"] = float(corrected_abs.mean().item())
            if loss_deltas:
                loss_abs_mean = float(torch.tensor(loss_deltas).abs().mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_term_abs_to_loss_abs_ratio"] = float(
                    linear_abs.mean().item() / (loss_abs_mean + eps)
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_abs_to_loss_abs_ratio"] = float(
                    corrected_abs.mean().item() / (loss_abs_mean + eps)
                )
        if fisher_quads:
            quad_tensor = torch.tensor(fisher_quads)
            err_tensor = torch.tensor(errors)
            signed_tensor = torch.tensor(signed_errors)
            out[f"{prefix}_fqa_r{tag}_fisher_quad_mean"] = float(quad_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_std"] = float(quad_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_signed_error_mean"] = float(signed_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_signed_error_std"] = float(signed_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_error_mean"] = float(err_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_error_std"] = float(err_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_error_max"] = float(err_tensor.max().item())
            denom = quad_tensor.abs() + torch.tensor(loss_deltas).abs() + eps
            out[f"{prefix}_fqa_r{tag}_relative_error_mean"] = float((err_tensor / denom).mean().item())
            if linear_corrected_errors:
                lc_err_tensor = torch.tensor(linear_corrected_errors)
                lc_signed_tensor = torch.tensor(linear_corrected_signed_errors)
                lc_delta_tensor = torch.tensor(linear_corrected_deltas)
                out[f"{prefix}_fqa_r{tag}_linear_corrected_signed_error_mean"] = float(lc_signed_tensor.mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_corrected_signed_error_std"] = float(
                    lc_signed_tensor.std(unbiased=False).item()
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_mean"] = float(lc_err_tensor.mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_std"] = float(
                    lc_err_tensor.std(unbiased=False).item()
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_max"] = float(lc_err_tensor.max().item())
                lc_denom = quad_tensor.abs() + lc_delta_tensor.abs() + eps
                out[f"{prefix}_fqa_r{tag}_linear_corrected_relative_error_mean"] = float(
                    (lc_err_tensor / lc_denom).mean().item()
                )
        logging.info(
            "[FQA][%s] radius %d/%d r=%g done loss_delta_mean=%s fisher_quad_mean=%s",
            prefix,
            radius_idx,
            radius_count,
            radius,
            out.get(f"{prefix}_fqa_r{tag}_loss_delta_mean", ""),
            out.get(f"{prefix}_fqa_r{tag}_fisher_quad_mean", ""),
        )

        # Aggregate extra Fisher variants for this radius.
        # Shared metrics (loss_delta, perturbed_loss, direction_norm, linear_term) are
        # copied from the main prefix; only Fisher-specific metrics differ.
        _shared_suffixes = {
            "radius", "samples", "linear_available",
            "grad_norm", "stationarity_bound", "stationarity_bound_over_r2",
            "random_linear_abs_theory", "random_linear_abs_theory_over_r2",
            "zero_radius_bias",
            "direction_norm_mean", "direction_norm_std",
            "perturbed_loss_mean", "perturbed_loss_std",
            "perturbed_loss_min", "perturbed_loss_max",
            "perturbed_acc_mean", "perturbed_acc_std",
            "perturbed_acc_min", "perturbed_acc_max",
            "acc_delta_mean", "acc_delta_std", "acc_delta_min", "acc_delta_max",
            "loss_delta_mean", "loss_delta_std", "loss_delta_min",
            "loss_delta_max", "loss_delta_abs_mean",
            "linear_term_mean", "linear_term_std",
            "linear_term_abs_mean", "linear_term_abs_max",
            "linear_term_abs_over_r2_mean",
            "linear_corrected_delta_mean", "linear_corrected_delta_std",
            "linear_corrected_delta_abs_mean",
            "linear_term_abs_to_loss_abs_ratio",
            "linear_corrected_abs_to_loss_abs_ratio",
        }
        for ef_idx, (ef_prefix, ef_ok, _) in enumerate(_extra):
            # Copy shared metrics.
            for suf in _shared_suffixes:
                main_key = f"{prefix}_fqa_r{tag}_{suf}"
                if main_key in out:
                    out[f"{ef_prefix}_fqa_r{tag}_{suf}"] = out[main_key]
            out[f"{ef_prefix}_fqa_r{tag}_fisher_available"] = bool(ef_ok)
            # Write Fisher-quadratic-specific metrics.
            ef_data = extra_rd[ef_idx]
            ef_quads = ef_data["quads"]
            if ef_quads:
                ef_quad_t = torch.tensor(ef_quads)
                ef_err_t = torch.tensor(ef_data["errors"])
                ef_signed_t = torch.tensor(ef_data["signed"])
                out[f"{ef_prefix}_fqa_r{tag}_fisher_quad_mean"] = float(ef_quad_t.mean().item())
                out[f"{ef_prefix}_fqa_r{tag}_fisher_quad_std"] = float(ef_quad_t.std(unbiased=False).item())
                out[f"{ef_prefix}_fqa_r{tag}_signed_error_mean"] = float(ef_signed_t.mean().item())
                out[f"{ef_prefix}_fqa_r{tag}_signed_error_std"] = float(ef_signed_t.std(unbiased=False).item())
                out[f"{ef_prefix}_fqa_r{tag}_error_mean"] = float(ef_err_t.mean().item())
                out[f"{ef_prefix}_fqa_r{tag}_error_std"] = float(ef_err_t.std(unbiased=False).item())
                out[f"{ef_prefix}_fqa_r{tag}_error_max"] = float(ef_err_t.max().item())
                ef_denom = ef_quad_t.abs() + torch.tensor(loss_deltas).abs() + eps
                out[f"{ef_prefix}_fqa_r{tag}_relative_error_mean"] = float((ef_err_t / ef_denom).mean().item())
                ef_lc_errors = ef_data["lc_errors"]
                if ef_lc_errors:
                    ef_lc_err_t = torch.tensor(ef_lc_errors)
                    ef_lc_signed_t = torch.tensor(ef_data["lc_signed"])
                    ef_lc_delta_t = torch.tensor(linear_corrected_deltas)
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_signed_error_mean"] = float(ef_lc_signed_t.mean().item())
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_signed_error_std"] = float(ef_lc_signed_t.std(unbiased=False).item())
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_error_mean"] = float(ef_lc_err_t.mean().item())
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_error_std"] = float(ef_lc_err_t.std(unbiased=False).item())
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_error_max"] = float(ef_lc_err_t.max().item())
                    ef_lc_denom = ef_quad_t.abs() + ef_lc_delta_t.abs() + eps
                    out[f"{ef_prefix}_fqa_r{tag}_linear_corrected_relative_error_mean"] = float(
                        (ef_lc_err_t / ef_lc_denom).mean().item()
                    )
            logging.info(
                "[FQA][%s] radius %d/%d r=%g done (co-computed) fisher_quad_mean=%s",
                ef_prefix, radius_idx, radius_count, radius,
                out.get(f"{ef_prefix}_fqa_r{tag}_fisher_quad_mean", ""),
            )

        for kf_idx, (kf_prefix, kf_ok, _) in enumerate(_extra_kfac):
            for suf in _shared_suffixes:
                main_key = f"{prefix}_fqa_r{tag}_{suf}"
                if main_key in out:
                    out[f"{kf_prefix}_fqa_r{tag}_{suf}"] = out[main_key]
            out[f"{kf_prefix}_fqa_r{tag}_fisher_available"] = bool(kf_ok)
            kf_data = extra_kfac_rd[kf_idx]
            kf_quads = kf_data["quads"]
            if kf_quads:
                kf_quad_t = torch.tensor(kf_quads)
                kf_err_t = torch.tensor(kf_data["errors"])
                kf_signed_t = torch.tensor(kf_data["signed"])
                out[f"{kf_prefix}_fqa_r{tag}_fisher_quad_mean"] = float(kf_quad_t.mean().item())
                out[f"{kf_prefix}_fqa_r{tag}_fisher_quad_std"] = float(kf_quad_t.std(unbiased=False).item())
                out[f"{kf_prefix}_fqa_r{tag}_signed_error_mean"] = float(kf_signed_t.mean().item())
                out[f"{kf_prefix}_fqa_r{tag}_signed_error_std"] = float(kf_signed_t.std(unbiased=False).item())
                out[f"{kf_prefix}_fqa_r{tag}_error_mean"] = float(kf_err_t.mean().item())
                out[f"{kf_prefix}_fqa_r{tag}_error_std"] = float(kf_err_t.std(unbiased=False).item())
                out[f"{kf_prefix}_fqa_r{tag}_error_max"] = float(kf_err_t.max().item())
                kf_denom = kf_quad_t.abs() + torch.tensor(loss_deltas).abs() + eps
                out[f"{kf_prefix}_fqa_r{tag}_relative_error_mean"] = float((kf_err_t / kf_denom).mean().item())
                kf_lc_errors = kf_data["lc_errors"]
                if kf_lc_errors:
                    kf_lc_err_t = torch.tensor(kf_lc_errors)
                    kf_lc_signed_t = torch.tensor(kf_data["lc_signed"])
                    kf_lc_delta_t = torch.tensor(linear_corrected_deltas)
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_signed_error_mean"] = float(
                        kf_lc_signed_t.mean().item()
                    )
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_signed_error_std"] = float(
                        kf_lc_signed_t.std(unbiased=False).item()
                    )
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_error_mean"] = float(kf_lc_err_t.mean().item())
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_error_std"] = float(
                        kf_lc_err_t.std(unbiased=False).item()
                    )
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_error_max"] = float(kf_lc_err_t.max().item())
                    kf_lc_denom = kf_quad_t.abs() + kf_lc_delta_t.abs() + eps
                    out[f"{kf_prefix}_fqa_r{tag}_linear_corrected_relative_error_mean"] = float(
                        (kf_lc_err_t / kf_lc_denom).mean().item()
                    )
            logging.info(
                "[FQA][%s] radius %d/%d r=%g done (full-KFAC) fisher_quad_mean=%s",
                kf_prefix, radius_idx, radius_count, radius,
                out.get(f"{kf_prefix}_fqa_r{tag}_fisher_quad_mean", ""),
            )

        # Append one JSONL record for this radius.  Written AFTER extra fisher aggregation
        # so all prefixes are complete.  Each line is self-contained and requires no
        # re-reading of earlier lines; the file grows by one line per radius.
        if partial_results_path is not None:
            try:
                import json as _json

                # ── per-u arrays (ΔL, g^T r u, Q_F per fisher variant) ──────
                _per_u: dict = {
                    "loss_delta": loss_deltas,           # ΔL_A(r_j, u_k)
                }
                if perturbed_accs:
                    _per_u["perturbed_acc"] = perturbed_accs
                    if base_acc is not None:
                        _per_u["acc_delta"] = [float(x - base_acc) for x in perturbed_accs]
                if linear_terms:
                    _per_u["linear_term"] = linear_terms             # g_A^T (r_j u_k), signed
                    _per_u["linear_term_abs"] = [abs(x) for x in linear_terms]  # |g_A^T (r_j u_k)|
                if fisher_quads:
                    _per_u[f"{prefix}_fisher_quad"] = fisher_quads   # Q_F(r_j, u_k) main
                for _ef_idx, (_ef_prefix, _ef_ok, _) in enumerate(_extra):
                    _ef_qs = extra_rd[_ef_idx]["quads"]
                    if _ef_qs:
                        _per_u[f"{_ef_prefix}_fisher_quad"] = _ef_qs  # Q_F extra variant
                for _kf_idx, (_kf_prefix, _kf_ok, _) in enumerate(_extra_kfac):
                    _kf_qs = extra_kfac_rd[_kf_idx]["quads"]
                    if _kf_qs:
                        _per_u[f"{_kf_prefix}_fisher_quad"] = _kf_qs  # Q_F full-KFAC variant

                # ── per-radius means (same quantities averaged over K) ───────
                _mean: dict = {}
                if loss_deltas:
                    _mean["loss_delta_abs"] = out.get(f"{prefix}_fqa_r{tag}_loss_delta_abs_mean")
                    _mean["loss_delta_signed"] = out.get(f"{prefix}_fqa_r{tag}_loss_delta_mean")
                if perturbed_accs:
                    _mean["base_acc"] = base_acc
                    _mean["perturbed_acc_mean"] = out.get(f"{prefix}_fqa_r{tag}_perturbed_acc_mean")
                    _mean["perturbed_acc_std"] = out.get(f"{prefix}_fqa_r{tag}_perturbed_acc_std")
                    _mean["acc_delta_mean"] = out.get(f"{prefix}_fqa_r{tag}_acc_delta_mean")
                    _mean["acc_delta_std"] = out.get(f"{prefix}_fqa_r{tag}_acc_delta_std")
                if linear_terms:
                    _mean["linear_term_abs"] = out.get(f"{prefix}_fqa_r{tag}_linear_term_abs_mean")   # Ḡ_A^abs
                    _mean["linear_term_signed"] = out.get(f"{prefix}_fqa_r{tag}_linear_term_mean")
                # Main Fisher
                if f"{prefix}_fqa_r{tag}_fisher_quad_mean" in out:
                    _mean[f"{prefix}_fisher_quad"] = out[f"{prefix}_fqa_r{tag}_fisher_quad_mean"]     # Q̄_F
                    _mean[f"{prefix}_error"] = out.get(f"{prefix}_fqa_r{tag}_error_mean")             # ē_A
                    _mean[f"{prefix}_signed_error"] = out.get(f"{prefix}_fqa_r{tag}_signed_error_mean")
                    _mean[f"{prefix}_relative_error"] = out.get(f"{prefix}_fqa_r{tag}_relative_error_mean")
                # Extra Fishers
                for _ef_prefix, _ef_ok, _ in _extra:
                    if f"{_ef_prefix}_fqa_r{tag}_fisher_quad_mean" in out:
                        _mean[f"{_ef_prefix}_fisher_quad"] = out[f"{_ef_prefix}_fqa_r{tag}_fisher_quad_mean"]
                        _mean[f"{_ef_prefix}_error"] = out.get(f"{_ef_prefix}_fqa_r{tag}_error_mean")
                        _mean[f"{_ef_prefix}_signed_error"] = out.get(f"{_ef_prefix}_fqa_r{tag}_signed_error_mean")
                        _mean[f"{_ef_prefix}_relative_error"] = out.get(f"{_ef_prefix}_fqa_r{tag}_relative_error_mean")
                for _kf_prefix, _kf_ok, _ in _extra_kfac:
                    if f"{_kf_prefix}_fqa_r{tag}_fisher_quad_mean" in out:
                        _mean[f"{_kf_prefix}_fisher_quad"] = out[f"{_kf_prefix}_fqa_r{tag}_fisher_quad_mean"]
                        _mean[f"{_kf_prefix}_error"] = out.get(f"{_kf_prefix}_fqa_r{tag}_error_mean")
                        _mean[f"{_kf_prefix}_signed_error"] = out.get(f"{_kf_prefix}_fqa_r{tag}_signed_error_mean")
                        _mean[f"{_kf_prefix}_relative_error"] = out.get(f"{_kf_prefix}_fqa_r{tag}_relative_error_mean")

                # Hessian reference: pre-computed scalars + per-radius upper bound.
                # lambda_max priority: Lanczos > power-iter > mechanism_eval Hessian.
                _hessian_ref: dict = {}
                if hessian_scalars:
                    _hessian_ref.update(hessian_scalars)
                    _lmax = (
                        hessian_scalars.get("endpoint_flat_lambda_max_lanczos")
                        or hessian_scalars.get("endpoint_flat_lambda_max")
                        or hessian_scalars.get("endpoint_hessian_top_eigen")
                    )
                    if _lmax is not None:
                        # Q_H(r, u) <= 0.5 * r^2 * lambda_max  for any unit vector u
                        _hessian_ref["hessian_quad_upper_bound"] = 0.5 * radius * radius * float(_lmax)

                _record = {
                    "radius": radius,
                    "radius_idx": radius_idx,
                    "radius_count": radius_count,
                    "K": len(loss_deltas),
                    "main_prefix": prefix,
                    "extra_prefixes": [ep for ep, _, _ in _extra] + [ep for ep, _, _ in _extra_kfac],
                    "per_u": _per_u,
                    "mean": {k: v for k, v in _mean.items() if v is not None},
                    "hessian_ref": _hessian_ref if _hessian_ref else None,
                }

                with open(partial_results_path, "a", encoding="utf-8") as _fh:
                    _fh.write(_json.dumps(_record, ensure_ascii=False) + "\n")
            except Exception as _e:
                logging.warning("[FQA] Failed to append partial results to %s: %s", partial_results_path, _e)

    return out


def summarize_radius_lora_factor_fisher_quadratic_error(
    *,
    prefix: str,
    radii: Sequence[float],
    samples: int,
    base_loss: float,
    model,
    params: List[torch.nn.Parameter],
    device,
    dtype,
    omega_W,
    grad_vec: Optional[torch.Tensor],
    delta_w_grad_vec: Optional[torch.Tensor],
    perturb_fn,
    restore_fn,
    loss_fn,
    radius_space: str = "delta_w_tangent",
    fisher_delta: str = "tangent",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """FQA for LoRA-factor perturbations with a Delta-W Fisher quadratic.

    Directions are sampled in factor coordinates (A/B), then evaluated through
    the induced Delta-W tangent direction.  This keeps the first-order Taylor
    term in the actual perturbed coordinates while still testing the EWC
    surrogate stored in Delta-W space.
    """
    out: Dict[str, float] = {}
    factor_dim = int(sum(param.numel() for param in params))
    if factor_dim <= 0 or samples <= 0 or not radii:
        return out

    radius_space = str(radius_space or "delta_w_tangent").lower()
    fisher_delta = str(fisher_delta or "tangent").lower()
    use_delta_w_radius = radius_space in {
        "delta_w",
        "delta_w_tangent",
        "tangent_delta_w",
        "induced_delta_w",
        "dw",
    }
    use_actual_fisher_delta = fisher_delta in {"actual", "actual_delta_w", "full"}

    shapes = _lora_factor_delta_w_shapes(model)
    fisher_blocks = _prepare_delta_w_fisher_blocks(omega_W or [], shapes, device, dtype)
    fisher_available = fisher_blocks is not None
    if omega_W and not fisher_available:
        logging.warning(
            "[FQA][%s] LoRA-factor Delta-W Fisher unavailable: fisher blocks=%s shapes=%s",
            prefix,
            len(omega_W or []),
            len(shapes),
        )

    delta_w_grad_blocks = _unflatten_delta_w_vector_to_shapes(
        delta_w_grad_vec, shapes, device, dtype
    )
    delta_w_linear_available = delta_w_grad_blocks is not None
    if delta_w_grad_vec is not None and not delta_w_linear_available:
        logging.warning(
            "[FQA][%s] skip Delta-W first-order diagnostics: grad numel=%s expected=%s",
            prefix,
            int(delta_w_grad_vec.numel()),
            int(sum(math.prod(tuple(shape)) for shape in shapes)),
        )

    linear_available = grad_vec is not None and int(grad_vec.numel()) == factor_dim
    if grad_vec is not None and not linear_available:
        logging.warning(
            "[FQA][%s] skip factor first-order term: grad numel=%s dim=%s",
            prefix,
            int(grad_vec.numel()),
            factor_dim,
        )
    grad_vec = grad_vec.detach().to(device=device, dtype=dtype) if linear_available else None
    grad_norm = float(grad_vec.norm().item()) if grad_vec is not None else 0.0
    random_projection_coeff = math.sqrt(2.0 / (math.pi * max(1, factor_dim)))

    logging.info(
        "[FQA][%s] LoRA-factor start: radii=%s samples=%d factor_dim=%d "
        "radius_space=%s fisher_delta=%s fisher_available=%s delta_w_linear=%s",
        prefix,
        [float(radius) for radius in radii],
        int(samples),
        factor_dim,
        radius_space,
        "actual" if use_actual_fisher_delta else "tangent",
        bool(fisher_available),
        bool(delta_w_linear_available),
    )

    for radius_idx, radius in enumerate(radii, start=1):
        radius = float(radius)
        tag = _radius_tag(radius)
        loss_deltas = []
        perturbed_losses = []
        fisher_quads = []
        fisher_quads_tangent = []
        fisher_quads_actual = []
        linear_terms = []
        linear_corrected_deltas = []
        linear_corrected_errors = []
        linear_corrected_signed_errors = []
        delta_w_linear_tangent_terms = []
        delta_w_linear_actual_terms = []
        delta_w_linear_normal_terms = []
        errors = []
        signed_errors = []
        direction_norms = []
        factor_perturb_norms = []
        factor_scales = []
        raw_tangent_norms = []
        delta_w_tangent_norms = []
        delta_w_actual_norms = []
        delta_w_normal_norms = []
        progress_every = max(1, int(samples) // 10) if samples else 1

        logging.info(
            "[FQA][%s] LoRA-factor radius %d/%d r=%g start",
            prefix,
            radius_idx,
            len(radii),
            radius,
        )

        attempts = 0
        max_attempts = max(int(samples) * 20, int(samples) + 1)
        while len(loss_deltas) < int(samples) and attempts < max_attempts:
            attempts += 1
            direction = _sample_unit_direction(factor_dim, device, dtype, eps=eps)
            components = _lora_factor_delta_w_components(model, params, direction)
            raw_stats = _lora_factor_component_stats(components, 1.0)
            raw_tangent_norm = float(raw_stats["delta_w_tangent_norm"])
            if use_delta_w_radius:
                if abs(radius) <= eps:
                    scale = 0.0
                elif raw_tangent_norm <= eps:
                    continue
                else:
                    scale = radius / (raw_tangent_norm + eps)
            else:
                scale = radius

            perturb = direction * scale
            stats = _lora_factor_component_stats(
                components,
                scale,
                fisher_blocks=fisher_blocks,
                delta_w_grad_blocks=delta_w_grad_blocks,
            )

            perturb_fn(perturb)
            try:
                loss = loss_fn()
            finally:
                restore_fn()

            loss_delta = float(loss - base_loss)
            loss_deltas.append(loss_delta)
            perturbed_losses.append(float(loss))
            direction_norms.append(float(direction.norm().detach().item()))
            factor_perturb_norms.append(float(abs(scale)))
            factor_scales.append(float(scale))
            raw_tangent_norms.append(raw_tangent_norm)
            delta_w_tangent_norms.append(float(stats["delta_w_tangent_norm"]))
            delta_w_actual_norms.append(float(stats["delta_w_actual_norm"]))
            delta_w_normal_norms.append(float(stats["delta_w_normal_norm"]))

            linear_term = None
            if grad_vec is not None:
                linear_term = float(torch.dot(grad_vec, perturb).detach().item())
                linear_corrected_delta = loss_delta - linear_term
                linear_terms.append(float(linear_term))
                linear_corrected_deltas.append(float(linear_corrected_delta))

            if stats["delta_w_linear_available"]:
                delta_w_linear_tangent_terms.append(float(stats["delta_w_linear_tangent"]))
                delta_w_linear_actual_terms.append(float(stats["delta_w_linear_actual"]))
                delta_w_linear_normal_terms.append(float(stats["delta_w_linear_normal"]))

            if stats["fisher_available"]:
                quad_tangent = float(stats["fisher_quad_tangent"])
                quad_actual = float(stats["fisher_quad_actual"])
                quad = quad_actual if use_actual_fisher_delta else quad_tangent
                signed_error = loss_delta - quad
                fisher_quads.append(float(quad))
                fisher_quads_tangent.append(float(quad_tangent))
                fisher_quads_actual.append(float(quad_actual))
                signed_errors.append(float(signed_error))
                errors.append(float(abs(signed_error)))
                if linear_term is not None:
                    linear_corrected_signed_error = loss_delta - linear_term - quad
                    linear_corrected_signed_errors.append(float(linear_corrected_signed_error))
                    linear_corrected_errors.append(float(abs(linear_corrected_signed_error)))

            sample_idx = len(loss_deltas)
            if sample_idx == int(samples) or sample_idx % progress_every == 0:
                logging.info(
                    "[FQA][%s] LoRA-factor radius %d/%d r=%g samples %d/%d",
                    prefix,
                    radius_idx,
                    len(radii),
                    radius,
                    sample_idx,
                    int(samples),
                )

        actual_samples = len(loss_deltas)
        out[f"{prefix}_fqa_r{tag}_radius"] = radius
        out[f"{prefix}_fqa_r{tag}_samples"] = int(actual_samples)
        out[f"{prefix}_fqa_r{tag}_fisher_available"] = bool(fisher_available)
        out[f"{prefix}_fqa_r{tag}_linear_available"] = bool(linear_available)
        out[f"{prefix}_fqa_r{tag}_delta_w_linear_available"] = bool(delta_w_linear_available)
        out[f"{prefix}_fqa_r{tag}_radius_space"] = radius_space
        out[f"{prefix}_fqa_r{tag}_fisher_delta"] = "actual" if use_actual_fisher_delta else "tangent"

        if linear_available:
            mean_factor_norm = float(torch.tensor(factor_perturb_norms).mean().item()) if factor_perturb_norms else 0.0
            out[f"{prefix}_fqa_r{tag}_grad_norm"] = float(grad_norm)
            out[f"{prefix}_fqa_r{tag}_stationarity_bound"] = float(grad_norm * mean_factor_norm)
            out[f"{prefix}_fqa_r{tag}_stationarity_bound_over_r2"] = (
                float((grad_norm * mean_factor_norm) / (radius * radius)) if abs(radius) > eps else 0.0
            )
            random_abs = float(radius * grad_norm * random_projection_coeff) if not use_delta_w_radius else 0.0
            out[f"{prefix}_fqa_r{tag}_random_linear_abs_theory"] = random_abs
            out[f"{prefix}_fqa_r{tag}_random_linear_abs_theory_over_r2"] = (
                float(random_abs / (radius * radius)) if abs(radius) > eps else 0.0
            )

        if abs(radius) < 1e-12 and loss_deltas:
            zero_bias = float(torch.tensor(loss_deltas).mean().item())
            out[f"{prefix}_fqa_r{tag}_zero_radius_bias"] = zero_bias
            if abs(zero_bias) > 1e-4:
                logging.warning("[FQA][%s] LoRA-factor r=0 bias=%.6f > 1e-4", prefix, zero_bias)

        def add_stats(name: str, values: List[float]):
            if not values:
                return
            tensor = torch.tensor(values)
            out[f"{prefix}_fqa_r{tag}_{name}_mean"] = float(tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_{name}_std"] = float(tensor.std(unbiased=False).item())

        add_stats("direction_norm", direction_norms)
        add_stats("factor_perturb_norm", factor_perturb_norms)
        add_stats("factor_scale", factor_scales)
        add_stats("raw_delta_w_tangent_norm", raw_tangent_norms)
        add_stats("delta_w_tangent_norm", delta_w_tangent_norms)
        add_stats("delta_w_actual_norm", delta_w_actual_norms)
        add_stats("delta_w_normal_norm", delta_w_normal_norms)

        if perturbed_losses:
            perturbed_tensor = torch.tensor(perturbed_losses)
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_mean"] = float(perturbed_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_std"] = float(perturbed_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_min"] = float(perturbed_tensor.min().item())
            out[f"{prefix}_fqa_r{tag}_perturbed_loss_max"] = float(perturbed_tensor.max().item())
        if loss_deltas:
            loss_tensor = torch.tensor(loss_deltas)
            out[f"{prefix}_fqa_r{tag}_loss_delta_mean"] = float(loss_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_std"] = float(loss_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_min"] = float(loss_tensor.min().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_max"] = float(loss_tensor.max().item())
            out[f"{prefix}_fqa_r{tag}_loss_delta_abs_mean"] = float(loss_tensor.abs().mean().item())
        if linear_terms:
            linear_tensor = torch.tensor(linear_terms)
            linear_abs = linear_tensor.abs()
            corrected_tensor = torch.tensor(linear_corrected_deltas)
            corrected_abs = corrected_tensor.abs()
            out[f"{prefix}_fqa_r{tag}_linear_term_mean"] = float(linear_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_std"] = float(linear_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_mean"] = float(linear_abs.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_max"] = float(linear_abs.max().item())
            out[f"{prefix}_fqa_r{tag}_linear_term_abs_over_r2_mean"] = (
                float(linear_abs.mean().item() / (radius * radius)) if abs(radius) > eps else 0.0
            )
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_mean"] = float(corrected_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_std"] = float(corrected_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_linear_corrected_delta_abs_mean"] = float(corrected_abs.mean().item())
            if loss_deltas:
                loss_abs_mean = float(torch.tensor(loss_deltas).abs().mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_term_abs_to_loss_abs_ratio"] = float(
                    linear_abs.mean().item() / (loss_abs_mean + eps)
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_abs_to_loss_abs_ratio"] = float(
                    corrected_abs.mean().item() / (loss_abs_mean + eps)
                )
        if delta_w_linear_tangent_terms:
            tangent_tensor = torch.tensor(delta_w_linear_tangent_terms)
            actual_tensor = torch.tensor(delta_w_linear_actual_terms)
            normal_tensor = torch.tensor(delta_w_linear_normal_terms)
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_tangent_mean"] = float(tangent_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_tangent_abs_mean"] = float(tangent_tensor.abs().mean().item())
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_actual_mean"] = float(actual_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_actual_abs_mean"] = float(actual_tensor.abs().mean().item())
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_normal_mean"] = float(normal_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_delta_w_linear_normal_abs_mean"] = float(normal_tensor.abs().mean().item())
        if fisher_quads:
            quad_tensor = torch.tensor(fisher_quads)
            quad_tangent_tensor = torch.tensor(fisher_quads_tangent)
            quad_actual_tensor = torch.tensor(fisher_quads_actual)
            err_tensor = torch.tensor(errors)
            signed_tensor = torch.tensor(signed_errors)
            out[f"{prefix}_fqa_r{tag}_fisher_quad_mean"] = float(quad_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_std"] = float(quad_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_tangent_mean"] = float(quad_tangent_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_tangent_std"] = float(quad_tangent_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_actual_mean"] = float(quad_actual_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_fisher_quad_actual_std"] = float(quad_actual_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_signed_error_mean"] = float(signed_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_signed_error_std"] = float(signed_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_error_mean"] = float(err_tensor.mean().item())
            out[f"{prefix}_fqa_r{tag}_error_std"] = float(err_tensor.std(unbiased=False).item())
            out[f"{prefix}_fqa_r{tag}_error_max"] = float(err_tensor.max().item())
            denom = quad_tensor.abs() + torch.tensor(loss_deltas).abs() + eps
            out[f"{prefix}_fqa_r{tag}_relative_error_mean"] = float((err_tensor / denom).mean().item())
            if linear_corrected_errors:
                lc_err_tensor = torch.tensor(linear_corrected_errors)
                lc_signed_tensor = torch.tensor(linear_corrected_signed_errors)
                lc_delta_tensor = torch.tensor(linear_corrected_deltas)
                out[f"{prefix}_fqa_r{tag}_linear_corrected_signed_error_mean"] = float(lc_signed_tensor.mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_corrected_signed_error_std"] = float(
                    lc_signed_tensor.std(unbiased=False).item()
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_mean"] = float(lc_err_tensor.mean().item())
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_std"] = float(
                    lc_err_tensor.std(unbiased=False).item()
                )
                out[f"{prefix}_fqa_r{tag}_linear_corrected_error_max"] = float(lc_err_tensor.max().item())
                lc_denom = quad_tensor.abs() + lc_delta_tensor.abs() + eps
                out[f"{prefix}_fqa_r{tag}_linear_corrected_relative_error_mean"] = float(
                    (lc_err_tensor / lc_denom).mean().item()
                )

        logging.info(
            "[FQA][%s] LoRA-factor radius %d/%d r=%g done loss_delta_mean=%s fisher_quad_mean=%s",
            prefix,
            radius_idx,
            len(radii),
            radius,
            out.get(f"{prefix}_fqa_r{tag}_loss_delta_mean", ""),
            out.get(f"{prefix}_fqa_r{tag}_fisher_quad_mean", ""),
        )

    return out


def fisher_buffer_metrics(
    omega_W,
    topk_fractions: Sequence[float],
    prefix: str = "fisher",
    eps: float = 1e-12,
) -> Dict[str, float]:
    if not omega_W:
        out = {
            f"{prefix}_blocks": 0,
            f"{prefix}_numel": 0,
            f"{prefix}_total_mass": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_effective_dim": 0.0,
            f"{prefix}_concentration": 0.0,
        }
        for frac in topk_fractions:
            out[f"{prefix}_top{_fraction_tag(frac)}_mass_ratio"] = 0.0
        return out

    flats = [fw.detach().float().reshape(-1).cpu().clamp_min(0) for fw in omega_W]
    flat = torch.cat(flats) if flats else torch.zeros(0)
    total = float(flat.sum().item())
    numel = int(flat.numel())
    mean = float(total / max(1, numel))
    max_val = float(flat.max().item()) if numel > 0 else 0.0
    sum_sq = float(flat.pow(2).sum().item())
    out = {
        f"{prefix}_blocks": int(len(omega_W)),
        f"{prefix}_numel": numel,
        f"{prefix}_total_mass": total,
        f"{prefix}_mean": mean,
        f"{prefix}_max": max_val,
        # effective_dim = tr(F)^2 / ||F||_F^2: how many coordinates carry most
        # of the Fisher mass (low value = concentrated on few directions).
        f"{prefix}_effective_dim": float((total * total) / (sum_sq + eps)),
        f"{prefix}_concentration": float(max_val / (mean + eps)),
    }

    for frac in topk_fractions:
        tag = _fraction_tag(frac)
        if numel <= 0:
            ratio = 0.0
        else:
            k = max(1, int(math.ceil(numel * float(frac))))
            k = min(k, numel)
            ratio = float(torch.topk(flat, k).values.sum().item() / (total + eps))
        out[f"{prefix}_top{tag}_mass_ratio"] = ratio
    return out


def delta_fisher_drift_metrics(
    omega_W,
    delta_tensors,
    eps: float = 1e-12,
    include_top1pct: bool = False,
) -> Dict[str, float]:
    """Measure Delta-W drift under a diagonal Fisher metric.

    ``delta_w_energy_in_t1_top1pct`` and
    ``d_f_contribution_from_t1_top1pct`` are ratios computed on the global top
    1% Fisher coordinates.  For tasks after task 2, "t1" should be read as the
    previous accumulated Fisher used by EWC.
    """
    delta_norm = 0.0
    fisher_weighted = 0.0
    fisher_parts = []
    delta_sq_parts = []
    numel = 0

    with torch.no_grad():
        for idx, delta_now in enumerate(delta_tensors):
            if idx >= len(omega_W):
                break
            fisher = omega_W[idx].to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
            delta_sq = delta_now.pow(2)
            delta_norm += float(delta_sq.sum().item())
            fisher_weighted += float((fisher * delta_sq).sum().item())

            if include_top1pct:
                fisher_parts.append(fisher.detach().float().reshape(-1).cpu())
                delta_sq_parts.append(delta_sq.detach().float().reshape(-1).cpu())
                numel += int(fisher.numel())

    out = {
        "delta_norm": float(delta_norm),
        "fisher_weighted_drift": float(fisher_weighted),
        "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
    }

    if include_top1pct:
        if numel <= 0 or not fisher_parts:
            out.update({
                "t1_fisher_top1pct_numel": 0,
                "delta_w_energy_in_t1_top1pct": 0.0,
                "d_f_contribution_from_t1_top1pct": 0.0,
            })
        else:
            flat_fisher = torch.cat(fisher_parts)
            flat_delta_sq = torch.cat(delta_sq_parts)
            k = min(max(1, int(math.ceil(numel * 0.01))), numel)
            top_idx = torch.topk(flat_fisher, k=k, largest=True, sorted=False).indices
            top_delta_energy = float(flat_delta_sq[top_idx].sum().item())
            top_fisher_weighted = float(
                (flat_fisher[top_idx] * flat_delta_sq[top_idx]).sum().item()
            )
            out.update({
                "t1_fisher_top1pct_numel": int(k),
                "delta_w_energy_in_t1_top1pct": float(top_delta_energy / (delta_norm + eps)),
                "d_f_contribution_from_t1_top1pct": float(
                    top_fisher_weighted / (fisher_weighted + eps)
                ),
            })

    return out


@contextmanager
def temporarily_loaded_named_params(model, state: Dict[str, torch.Tensor]):
    """Temporarily overwrite named parameters, restoring them on exit.

    Used to evaluate loss/accuracy at a saved anchor (e.g. a pre-merge
    ``theta_end`` or post-merge ``mu_merge`` snapshot) without permanently
    disturbing the live training state.
    """
    model = unwrap_model(model)
    named = dict(model.named_parameters())
    saved: Dict[str, torch.Tensor] = {}
    touched: List[str] = []
    for key, value in state.items():
        param = named.get(key)
        if param is None:
            continue
        saved[key] = param.detach().clone()
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        touched.append(key)
    try:
        yield touched
    finally:
        for key in touched:
            param = named[key]
            param.data.copy_(saved[key].to(device=param.device, dtype=param.dtype))


def delta_w_from_factor_pair(
    a: Optional[torch.Tensor], b: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Recompute an effective LoRA update ``B @ A`` from two saved factor tensors."""
    if a is None or b is None:
        return None
    return b.detach().float() @ a.detach().float()


def future_surrogate_residual_metrics(
    delta_loss: float,
    fisher_weighted_drift: float,
    eps: float = 1e-12,
    prefix: str = "future",
) -> Dict[str, float]:
    """Compare an observed future loss change to its quadratic Fisher surrogate.

    ``delta_loss`` is the observed ``L_t(current) - L_t(anchor)`` after training
    has moved the model away from a saved task-``t`` anchor. ``fisher_weighted_drift``
    is ``sum(F * Delta^2)`` at that anchor (e.g. from ``delta_fisher_drift_metrics``);
    the quadratic Laplace surrogate is ``Q_F = 0.5 * fisher_weighted_drift``.
    """
    quad_approx = 0.5 * float(fisher_weighted_drift)
    signed_residual = float(delta_loss) - quad_approx
    abs_residual = abs(signed_residual)
    return {
        f"{prefix}_delta_loss": float(delta_loss),
        f"{prefix}_quad_approx": quad_approx,
        f"{prefix}_signed_residual": signed_residual,
        f"{prefix}_abs_residual": abs_residual,
        f"{prefix}_rel_residual": float(abs_residual / (abs(float(delta_loss)) + eps)),
        f"{prefix}_quad_ratio": float(quad_approx / (float(delta_loss) + eps)),
    }


def _fraction_tag(frac: float) -> str:
    text = f"{float(frac):g}".replace(".", "p").replace("-", "m")
    return text


def evaluate_anchor_metrics(
    model,
    loader,
    device,
    args,
    cur_task: int,
    task_start: int,
    task_end: int,
    omega_W=None,
    prefix: str = "anchor",
    extra_omega_W: Optional[Dict[str, List[torch.Tensor]]] = None,
    kfac_factors: Optional[List[Dict]] = None,
    extra_kfac_factors: Optional[Dict[str, List[Dict]]] = None,
    partial_results_path: Optional[str] = None,
    hessian_scalars: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    model = unwrap_model(model)
    scope = str(args.get("flat_eval_param_scope", args.get("anchor_eval_param_scope", "lora")))
    scope_lower = scope.lower()
    requested_delta_w = scope_lower in {"delta_w", "dw", "effective_delta_w", "effective_update"}
    requested_lora_factor = scope_lower in {
        "lora_factor",
        "lora_factors",
        "lora_ab",
        "ab",
        "a_b",
    }
    fqa_fisher_space = str(args.get("flat_eval_fqa_fisher_space", "")).lower()
    use_lora_factor_delta_w_fqa = requested_lora_factor and fqa_fisher_space in {
        "delta_w",
        "dw",
        "delta_w_induced",
        "induced_delta_w",
    }
    delta_w_dim = _delta_w_coordinate_count(model)
    use_delta_w_coords = requested_delta_w and delta_w_dim > 0
    named_params = [] if use_delta_w_coords else select_eval_named_params(
        model, "lora" if requested_delta_w else scope
    )
    params = [param for _, param in named_params]
    param_slices = _param_slice_map(named_params) if named_params else None
    topk_fractions = parse_float_list(
        args.get("fisher_topk_fractions", args.get("anchor_fisher_topk_fractions", "0.001,0.01,0.05")),
        default=[0.001, 0.01, 0.05],
    )

    param_count = delta_w_dim if use_delta_w_coords else int(sum(param.numel() for param in params))
    param_space = "delta_w" if use_delta_w_coords else (
        "lora_factor_fallback" if requested_delta_w else (
            "lora_factor_delta_w_fisher" if use_lora_factor_delta_w_fqa else scope_lower
        )
    )
    metrics: Dict[str, float] = {
        f"{prefix}_task": int(cur_task),
        f"{prefix}_task_start": int(task_start),
        f"{prefix}_task_end": int(task_end),
        f"{prefix}_param_count": int(param_count),
        f"{prefix}_param_space": param_space,
        f"{prefix}_requested_param_scope": scope_lower,
        f"{prefix}_fqa_fisher_space": fqa_fisher_space,
    }
    metrics.update(fisher_buffer_metrics(omega_W or [], topk_fractions, prefix=f"{prefix}_fisher"))

    if param_count <= 0:
        return metrics

    rho = float(args.get("flat_eval_sharpness_radius", args.get("gam_grad_rho", 0.05)))
    loss_max_batches = parse_max_batches(args.get("flat_eval_loss_max_batches", 10), default=10)
    grad_max_batches = parse_max_batches(
        args.get("flat_eval_first_order_grad_batches", loss_max_batches),
        default=loss_max_batches,
    )
    eps = float(args.get("mechanism_eps", 1e-12))
    unit_fqa_enabled = as_bool(
        args.get(
            "flat_eval_fisher_quadratic_error",
            args.get("flat_eval_unit_expected_sharpness", False),
        )
    )
    unit_fqa_radii = parse_float_list(
        args.get("flat_eval_fqa_radii", args.get("flat_eval_unit_esh_radii", "")),
        default=[],
    )
    unit_fqa_samples = int(
        args.get(
            "flat_eval_fqa_samples",
            args.get("flat_eval_unit_esh_samples", 10),
        )
    )
    unit_fqa_tasks = set(parse_int_list(
        args.get("flat_eval_fqa_tasks", args.get("flat_eval_unit_esh_tasks", "0")),
        default=[0],
    ))
    unit_fqa_prefixes = {
        item.lower()
        for item in parse_str_list(
            args.get("flat_eval_fqa_prefixes", args.get("flat_eval_unit_esh_prefixes", "endpoint")),
            default=["endpoint"],
        )
    }
    unit_fqa_active = (
        unit_fqa_enabled
        and int(cur_task) in unit_fqa_tasks
        and str(prefix).lower() in unit_fqa_prefixes
    )
    second_order_batch_size = parse_positive_int(
        args.get("flat_eval_second_order_batch_size", args.get("flat_eval_batch_size", None)),
        default=None,
    )
    hessian_batch_size = parse_positive_int(
        args.get("flat_eval_hessian_batch_size", None),
        default=second_order_batch_size,
    )
    emp_fisher_batch_size = parse_positive_int(
        args.get("flat_eval_emp_fisher_batch_size", None),
        default=hessian_batch_size,
    )
    if hessian_batch_size is None and emp_fisher_batch_size is not None:
        hessian_batch_size = emp_fisher_batch_size
    if hessian_batch_size != emp_fisher_batch_size:
        logging.warning(
            "[AnchorEval] flat_eval_emp_fisher_batch_size=%s differs from "
            "flat_eval_hessian_batch_size=%s; using the Hessian batch size for both.",
            emp_fisher_batch_size,
            hessian_batch_size,
        )
        emp_fisher_batch_size = hessian_batch_size

    eval_loader = _make_eval_loader(loader, None)

    with preserve_training_mode(model):
        model.eval()
        if use_delta_w_coords:
            with external_delta_w_coordinates(model) as coords:
                grad_vec, base_loss = current_task_delta_w_grad_vector(
                    model,
                    eval_loader,
                    device,
                    coords,
                    task_start,
                    task_end,
                    max_batches=grad_max_batches,
                )
                base_loss = average_current_task_loss(
                    model,
                    eval_loader,
                    device,
                    task_start,
                    task_end,
                    max_batches=loss_max_batches,
                )
                grad_norm = float(grad_vec.norm().item())
                metrics[f"{prefix}_base_loss"] = float(base_loss)
                metrics[f"{prefix}_grad_norm"] = grad_norm
                metrics[f"{prefix}_first_order_sharpness"] = float(rho * grad_norm)

                if grad_norm > eps:
                    _add_vector_to_tensors(coords, grad_vec * (rho / (grad_norm + eps)))
                    perturbed_loss = average_current_task_loss(
                        model,
                        eval_loader,
                        device,
                        task_start,
                        task_end,
                        max_batches=loss_max_batches,
                    )
                    _zero_tensors(coords)
                    metrics[f"{prefix}_sh0_perturbed_loss"] = float(perturbed_loss)
                    metrics[f"{prefix}_sh0_max"] = float(perturbed_loss - base_loss)
                else:
                    metrics[f"{prefix}_sh0_perturbed_loss"] = float(base_loss)
                    metrics[f"{prefix}_sh0_max"] = 0.0

                if as_bool(args.get("flat_eval_expected_sharpness", args.get("flat_eval_esh", False))):
                    dim = int(grad_vec.numel())
                    samples = int(args.get("flat_eval_esh_samples", 10))
                    sigma = args.get("flat_eval_esh_gaussian_std", None)
                    sigma = float(sigma) if sigma is not None else rho / math.sqrt(max(1, dim))
                    losses = []
                    for _ in range(max(0, samples)):
                        noise = torch.randn(dim, device=device, dtype=grad_vec.dtype) * sigma
                        _add_vector_to_tensors(coords, noise)
                        loss = average_current_task_loss(
                            model,
                            eval_loader,
                            device,
                            task_start,
                            task_end,
                            max_batches=loss_max_batches,
                        )
                        losses.append(float(loss - base_loss))
                        _zero_tensors(coords)
                    if losses:
                        tensor = torch.tensor(losses)
                        metrics[f"{prefix}_esh_mean"] = float(tensor.mean().item())
                        metrics[f"{prefix}_esh_std"] = float(tensor.std(unbiased=False).item())
                else:
                    metrics[f"{prefix}_esh_mean"] = 0.0
                    metrics[f"{prefix}_esh_std"] = 0.0

                if unit_fqa_active:
                    dim = int(grad_vec.numel())
                    fisher_diag = _flatten_fisher_diag(omega_W or [], device, grad_vec.dtype, eps=eps)

                    # Collect fixed batches once; both base_loss and every
                    # perturbed_loss call will use identical input tensors,
                    # guaranteeing loss_delta == 0 at r=0.
                    fqa_fixed = collect_fixed_batches(
                        eval_loader, device, task_start, task_end,
                        max_batches=loss_max_batches,
                    )
                    base_loss_fqa = average_loss_on_fixed_batches(model, fqa_fixed)
                    fqa_grad_vec, fqa_grad_base_loss = grad_vector_on_fixed_batches(
                        model, fqa_fixed, coords
                    )
                    metrics[f"{prefix}_fqa_base_loss"] = float(base_loss_fqa)
                    metrics[f"{prefix}_fqa_grad_base_loss"] = float(fqa_grad_base_loss)
                    metrics[f"{prefix}_fqa_grad_norm"] = float(fqa_grad_vec.norm().item())
                    logging.info(
                        "[FQA][%s] paired base_loss=%.6f grad_loss=%.6f "
                        "grad_norm=%.6f (non-paired was %.6f)",
                        prefix,
                        base_loss_fqa,
                        fqa_grad_base_loss,
                        float(fqa_grad_vec.norm().item()),
                        base_loss,
                    )

                    def perturb_fn(perturb):
                        _add_vector_to_tensors(coords, perturb)

                    def restore_fn():
                        _zero_tensors(coords)

                    def loss_fn():
                        return average_loss_on_fixed_batches(model, fqa_fixed)

                    def acc_fn():
                        return average_accuracy_on_fixed_batches(model, fqa_fixed)

                    metrics.update(summarize_radius_fisher_quadratic_error(
                        prefix=prefix,
                        radii=unit_fqa_radii,
                        samples=unit_fqa_samples,
                        base_loss=base_loss_fqa,
                        dim=dim,
                        device=device,
                        dtype=grad_vec.dtype,
                        fisher_diag=fisher_diag,
                        kfac_factors=kfac_factors,
                        grad_vec=fqa_grad_vec,
                        perturb_fn=perturb_fn,
                        restore_fn=restore_fn,
                        loss_fn=loss_fn,
                        acc_fn=acc_fn,
                        extra_kfac_factors=(
                            list((extra_kfac_factors or {}).items()) if extra_kfac_factors else None
                        ),
                        param_slices=param_slices,
                        partial_results_path=partial_results_path,
                        hessian_scalars=hessian_scalars,
                        direction_mode=str(args.get("flat_eval_fqa_direction_mode", "random")),
                        direction_topk_frac=float(args.get("flat_eval_fqa_direction_topk_frac", 0.01)),
                        eps=eps,
                    ))

                if as_bool(args.get("flat_eval_hessian", args.get("anchor_eval_hessian", False))):
                    dim = int(grad_vec.numel())
                    hess_batches = parse_max_batches(
                        args.get("flat_eval_hessian_max_batches", loss_max_batches),
                        default=loss_max_batches,
                    )
                    iters = int(args.get("flat_eval_hessian_power_iters", 10))
                    trace_samples = int(args.get("flat_eval_hessian_trace_samples", 5))
                    second_order_loader = _make_eval_loader(eval_loader, hessian_batch_size)
                    metrics[f"{prefix}_hessian_batch_size"] = int(
                        getattr(second_order_loader, "batch_size", 0) or 0
                    )
                    metrics[f"{prefix}_emp_fisher_batch_size"] = metrics[
                        f"{prefix}_hessian_batch_size"
                    ]

                    def hess_mvp(vec):
                        _zero_tensors(coords)
                        return delta_w_hessian_vector_product(
                            model,
                            second_order_loader,
                            device,
                            coords,
                            vec,
                            task_start,
                            task_end,
                            hess_batches,
                        )

                    metrics[f"{prefix}_hessian_top_eigen"] = power_top_eigen(
                        hess_mvp, dim, device, iters, eps=eps
                    )
                    metrics[f"{prefix}_hessian_trace"] = hutchinson_trace(
                        hess_mvp, dim, device, trace_samples
                    )

                    if as_bool(args.get("flat_eval_emp_fisher_spectrum", True)):
                        def fisher_mvp(vec):
                            _zero_tensors(coords)
                            return delta_w_empirical_fisher_vector_product(
                                model,
                                second_order_loader,
                                device,
                                coords,
                                vec,
                                task_start,
                                task_end,
                                hess_batches,
                            )

                        metrics[f"{prefix}_emp_fisher_top_eigen"] = power_top_eigen(
                            fisher_mvp, dim, device, iters, eps=eps
                        )
                        metrics[f"{prefix}_emp_fisher_trace"] = hutchinson_trace(
                            fisher_mvp, dim, device, trace_samples
                        )
                        metrics[f"{prefix}_hessian_emp_fisher_top_eigen_ratio"] = (
                            metrics[f"{prefix}_hessian_top_eigen"]
                            / (metrics[f"{prefix}_emp_fisher_top_eigen"] + eps)
                        )
                        metrics[f"{prefix}_hessian_emp_fisher_trace_ratio"] = (
                            metrics[f"{prefix}_hessian_trace"]
                            / (metrics[f"{prefix}_emp_fisher_trace"] + eps)
                        )
        else:
            grad_vec, base_loss = current_task_grad_vector(
                model,
                eval_loader,
                device,
                params,
                task_start,
                task_end,
                max_batches=grad_max_batches,
            )
            base_loss = average_current_task_loss(
                model,
                eval_loader,
                device,
                task_start,
                task_end,
                max_batches=loss_max_batches,
            )
            grad_norm = float(grad_vec.norm().item())
            metrics[f"{prefix}_base_loss"] = float(base_loss)
            metrics[f"{prefix}_grad_norm"] = grad_norm
            metrics[f"{prefix}_first_order_sharpness"] = float(rho * grad_norm)

            saved = _clone_params(params)
            if grad_norm > eps:
                _add_vector_to_params(params, grad_vec * (rho / (grad_norm + eps)))
                perturbed_loss = average_current_task_loss(
                    model,
                    eval_loader,
                    device,
                    task_start,
                    task_end,
                    max_batches=loss_max_batches,
                )
                _restore_params(params, saved)
                metrics[f"{prefix}_sh0_perturbed_loss"] = float(perturbed_loss)
                metrics[f"{prefix}_sh0_max"] = float(perturbed_loss - base_loss)
            else:
                metrics[f"{prefix}_sh0_perturbed_loss"] = float(base_loss)
                metrics[f"{prefix}_sh0_max"] = 0.0

            if as_bool(args.get("flat_eval_expected_sharpness", args.get("flat_eval_esh", False))):
                dim = int(grad_vec.numel())
                samples = int(args.get("flat_eval_esh_samples", 10))
                sigma = args.get("flat_eval_esh_gaussian_std", None)
                sigma = float(sigma) if sigma is not None else rho / math.sqrt(max(1, dim))
                losses = []
                for _ in range(max(0, samples)):
                    noise = torch.randn(dim, device=device, dtype=grad_vec.dtype) * sigma
                    _add_vector_to_params(params, noise)
                    loss = average_current_task_loss(
                        model,
                        eval_loader,
                        device,
                        task_start,
                        task_end,
                        max_batches=loss_max_batches,
                    )
                    losses.append(float(loss - base_loss))
                    _restore_params(params, saved)
                if losses:
                    tensor = torch.tensor(losses)
                    metrics[f"{prefix}_esh_mean"] = float(tensor.mean().item())
                    metrics[f"{prefix}_esh_std"] = float(tensor.std(unbiased=False).item())
                else:
                    metrics[f"{prefix}_esh_mean"] = 0.0
                    metrics[f"{prefix}_esh_std"] = 0.0

            if unit_fqa_active:
                dim = int(grad_vec.numel())

                fqa_fixed = collect_fixed_batches(
                    eval_loader, device, task_start, task_end,
                    max_batches=loss_max_batches,
                )
                base_loss_fqa = average_loss_on_fixed_batches(model, fqa_fixed)
                fqa_grad_vec, fqa_grad_base_loss = grad_vector_on_fixed_batches(
                    model, fqa_fixed, params
                )
                metrics[f"{prefix}_fqa_base_loss"] = float(base_loss_fqa)
                metrics[f"{prefix}_fqa_grad_base_loss"] = float(fqa_grad_base_loss)
                metrics[f"{prefix}_fqa_grad_norm"] = float(fqa_grad_vec.norm().item())
                logging.info(
                    "[FQA][%s] paired base_loss=%.6f grad_loss=%.6f "
                    "grad_norm=%.6f (non-paired was %.6f)",
                    prefix,
                    base_loss_fqa,
                    fqa_grad_base_loss,
                    float(fqa_grad_vec.norm().item()),
                    base_loss,
                )

                def perturb_fn(perturb):
                    _add_vector_to_params(params, perturb)

                def restore_fn():
                    _restore_params(params, saved)

                def loss_fn():
                    return average_loss_on_fixed_batches(model, fqa_fixed)

                def acc_fn():
                    return average_accuracy_on_fixed_batches(model, fqa_fixed)

                if use_lora_factor_delta_w_fqa:
                    with external_delta_w_coordinates(model) as coords:
                        fqa_delta_w_grad_vec, fqa_delta_w_grad_base_loss = grad_vector_on_fixed_batches(
                            model, fqa_fixed, coords
                        )
                    metrics[f"{prefix}_fqa_delta_w_grad_base_loss"] = float(fqa_delta_w_grad_base_loss)
                    metrics[f"{prefix}_fqa_delta_w_grad_norm"] = float(fqa_delta_w_grad_vec.norm().item())
                    logging.info(
                        "[FQA][%s] Delta-W grad on paired batches: loss=%.6f grad_norm=%.6f",
                        prefix,
                        fqa_delta_w_grad_base_loss,
                        float(fqa_delta_w_grad_vec.norm().item()),
                    )

                    metrics.update(summarize_radius_lora_factor_fisher_quadratic_error(
                        prefix=prefix,
                        radii=unit_fqa_radii,
                        samples=unit_fqa_samples,
                        base_loss=base_loss_fqa,
                        model=model,
                        params=params,
                        device=device,
                        dtype=grad_vec.dtype,
                        omega_W=omega_W or [],
                        grad_vec=fqa_grad_vec,
                        delta_w_grad_vec=fqa_delta_w_grad_vec,
                        perturb_fn=perturb_fn,
                        restore_fn=restore_fn,
                        loss_fn=loss_fn,
                        radius_space=args.get("flat_eval_factor_radius_space", "delta_w_tangent"),
                        fisher_delta=args.get("flat_eval_factor_fisher_delta", "tangent"),
                        eps=eps,
                    ))
                else:
                    fisher_diag = _flatten_fisher_diag(omega_W or [], device, grad_vec.dtype, eps=eps)
                    _extra_fqa_fishers = [
                        (ef_prefix, _flatten_fisher_diag(ef_blocks, device, grad_vec.dtype, eps=eps))
                        for ef_prefix, ef_blocks in (extra_omega_W or {}).items()
                    ]
                    _extra_fqa_kfac = (
                        list((extra_kfac_factors or {}).items()) if extra_kfac_factors else None
                    )
                    metrics.update(summarize_radius_fisher_quadratic_error(
                        prefix=prefix,
                        radii=unit_fqa_radii,
                        samples=unit_fqa_samples,
                        base_loss=base_loss_fqa,
                        dim=dim,
                        device=device,
                        dtype=grad_vec.dtype,
                        fisher_diag=fisher_diag,
                        kfac_factors=kfac_factors,
                        grad_vec=fqa_grad_vec,
                        perturb_fn=perturb_fn,
                        restore_fn=restore_fn,
                        loss_fn=loss_fn,
                        acc_fn=acc_fn,
                        extra_fishers=_extra_fqa_fishers or None,
                        extra_kfac_factors=_extra_fqa_kfac,
                        param_slices=param_slices,
                        partial_results_path=partial_results_path,
                        hessian_scalars=hessian_scalars,
                        direction_mode=str(args.get("flat_eval_fqa_direction_mode", "random")),
                        direction_topk_frac=float(args.get("flat_eval_fqa_direction_topk_frac", 0.01)),
                        eps=eps,
                    ))

            if as_bool(args.get("flat_eval_hessian", args.get("anchor_eval_hessian", False))):
                dim = int(grad_vec.numel())
                hess_batches = parse_max_batches(
                    args.get("flat_eval_hessian_max_batches", loss_max_batches),
                    default=loss_max_batches,
                )
                iters = int(args.get("flat_eval_hessian_power_iters", 10))
                trace_samples = int(args.get("flat_eval_hessian_trace_samples", 5))
                second_order_loader = _make_eval_loader(eval_loader, hessian_batch_size)
                metrics[f"{prefix}_hessian_batch_size"] = int(
                    getattr(second_order_loader, "batch_size", 0) or 0
                )
                metrics[f"{prefix}_emp_fisher_batch_size"] = metrics[
                    f"{prefix}_hessian_batch_size"
                ]

                def hess_mvp(vec):
                    return hessian_vector_product(
                        model,
                        second_order_loader,
                        device,
                        params,
                        vec,
                        task_start,
                        task_end,
                        hess_batches,
                    )

                metrics[f"{prefix}_hessian_top_eigen"] = power_top_eigen(
                    hess_mvp, dim, device, iters, eps=eps
                )
                metrics[f"{prefix}_hessian_trace"] = hutchinson_trace(
                    hess_mvp, dim, device, trace_samples
                )

                if as_bool(args.get("flat_eval_emp_fisher_spectrum", True)):
                    def fisher_mvp(vec):
                        return empirical_fisher_vector_product(
                            model,
                            second_order_loader,
                            device,
                            params,
                            vec,
                            task_start,
                            task_end,
                            hess_batches,
                        )

                    metrics[f"{prefix}_emp_fisher_top_eigen"] = power_top_eigen(
                        fisher_mvp, dim, device, iters, eps=eps
                    )
                    metrics[f"{prefix}_emp_fisher_trace"] = hutchinson_trace(
                        fisher_mvp, dim, device, trace_samples
                    )
                    metrics[f"{prefix}_hessian_emp_fisher_top_eigen_ratio"] = (
                        metrics[f"{prefix}_hessian_top_eigen"]
                        / (metrics[f"{prefix}_emp_fisher_top_eigen"] + eps)
                    )
                    metrics[f"{prefix}_hessian_emp_fisher_trace_ratio"] = (
                        metrics[f"{prefix}_hessian_trace"]
                        / (metrics[f"{prefix}_emp_fisher_trace"] + eps)
                    )

    logging.info("[AnchorEval] %s", metrics)
    return metrics
