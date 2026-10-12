import sys
import logging
import copy
import re
import torch
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from evaluation.metrics import compute_sequence_metrics, save_metrics_and_vectors
from evaluation_feature.eval_flat_feature import (
    extract_features_and_labels,
    linear_cka,
    compare_first_last_features,
    compare_task_last_features,
)
import os
import numpy as np
from evaluation_weight_sharpness.eval_flatness_weight_Loss import FlatnessConfig, evaluate_flatness_metrics
from evaluation_weight_sharpness.param_utils import _select_params_by_name
import gc, torch
from evaluation_feature.eval_flat_feature import FeatureFlatnessConfig, evaluate_feature_metrics, save_task_feature_cache
from evaluation_attention.attention_probe import run_attention_probe
from evaluation_ood.ood_eval import run_ood_evaluation
from typing import Optional
import json
from utils.data_manager import fractional_loader
from evaluation_performance.probe import (
    LinearProbeConfig,
    LinearProbeRunner,
    fit_linear_probe_softmax_head,
    evaluate_linear_probe_softmax_with_head,
    _FeatureView,
)
from utils.random_reproduce import set_device, set_random, json_safe
from utils.metrics_book import (
    metrics_json_path,
    write_step_metrics,
    write_final_metrics,
    assemble_eval_matrix,
    log_eval_matrix,
    save_eval_matrix,
)

# ---------------------------
# Metrics helpers live in utils.metrics_book
# ---------------------------




def _parse_task_list(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        try:
            return [int(x) for x in val]
        except Exception:
            return None
    if isinstance(val, int):
        return [val]
    if isinstance(val, str):
        s = val.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                import json as _json
                return [int(x) for x in _json.loads(s)]
        except Exception:
            return None
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        except Exception:
            return None
    return None
    logging.info("Saved %s to %s and %s", tag, npy_path, csv_path)


def _resolve_task_indices(spec, nb_tasks):
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip().lower()
        if spec in {"", "all"}:
            return None
        if spec in {"first_last", "first-last", "firstlast"}:
            indices = [0, nb_tasks - 1]
        else:
            indices = [int(s) for s in spec.split(",") if s.strip() != ""]
    elif isinstance(spec, (list, tuple, set)):
        indices = [int(s) for s in spec]
    else:
        indices = [int(spec)]
    resolved = []
    for idx in indices:
        if idx < 0:
            idx = nb_tasks + idx
        if 0 <= idx < nb_tasks:
            resolved.append(idx)
    if not resolved:
        return []
    return sorted(set(resolved))


def _resolve_conditioned_task_indices(spec, current_task, nb_tasks):
    if spec is None:
        return list(range(current_task + 1))
    if isinstance(spec, str):
        lowered = spec.strip().lower()
        if lowered in {"", "seen", "all_seen", "seen_so_far"}:
            return list(range(current_task + 1))
        if lowered in {"old", "previous"}:
            return list(range(current_task))
        if lowered in {"current", "cur"}:
            return [current_task]
    resolved = _resolve_task_indices(spec, nb_tasks)
    if resolved is None:
        resolved = list(range(nb_tasks))
    return [idx for idx in resolved if 0 <= idx <= current_task]


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _optimizer_schedule_tag(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = [item.strip() for item in text.split(",") if item.strip()]
        else:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
    elif isinstance(spec, (list, tuple)):
        parsed = list(spec)
    else:
        parsed = [spec]
    items = [str(item).strip().lower() for item in parsed if str(item).strip()]
    if not items:
        return None
    tag = "_".join(items)
    return "taskwise_" + re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)


def _matrix_rows_from_json(json_path, section, resume_task):
    """Load prefix evaluation rows from a previous consolidated metrics JSON."""
    if not json_path:
        return []
    json_path = os.path.abspath(os.path.expanduser(str(json_path)))
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"resume metrics JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        book = json.load(f)
    matrices = book.get(section, {}).get("matrices", {})
    key = f"t{int(resume_task):02d}"
    matrix = matrices.get(key, matrices.get("final", None))
    if matrix is None:
        raise KeyError(f"metrics JSON has no {section}.matrices.{key} or final: {json_path}")
    matrix = np.asarray(matrix, dtype=float)
    rows = []
    upto = min(int(resume_task) + 1, matrix.shape[0])
    for row_idx in range(upto):
        width = min(row_idx + 1, matrix.shape[1])
        row = matrix[row_idx, :width]
        rows.append([float(x) for x in row])
    return rows


def _top1_curve_from_rows(rows):
    curve = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        curve.append(float(np.nanmean(arr)) if arr.size else float("nan"))
    return curve


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_max_batches(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "all", "full"}:
        return None
    value = int(value)
    return None if value <= 0 else value


def _collect_task_fixed_batches_cpu(data_manager, task_start, task_end, split, batch_size, num_workers, max_batches):
    source = "train" if split == "train" else "test"
    mode = "train" if split == "train" else "test"
    dataset = data_manager.get_dataset(
        np.arange(int(task_start), int(task_end)),
        source=source,
        mode=mode,
    )
    loader = data_manager.build_dataloader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    fixed = []
    for _, inputs, targets in loader:
        mask = ((targets >= int(task_start)) & (targets < int(task_end))).nonzero().view(-1)
        if mask.numel() == 0:
            continue
        fixed.append((
            torch.index_select(inputs, 0, mask).detach().cpu(),
            torch.index_select(targets, 0, mask).detach().cpu(),
        ))
        if max_batches is not None and len(fixed) >= int(max_batches):
            break
    return fixed


def _evaluate_fixed_task_batches(model, fixed_batches, task_start, task_end):
    net = getattr(model, "_network", model)
    was_training = bool(getattr(net, "training", False))
    net.eval()
    total = 0
    local_loss_sum = 0.0
    full_loss_sum = 0.0
    local_correct = 0
    full_correct = 0
    device = getattr(model, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        with torch.no_grad():
            for inputs_cpu, targets_cpu in fixed_batches:
                inputs = inputs_cpu.to(device, non_blocking=True)
                targets = targets_cpu.to(device, non_blocking=True)
                try:
                    outputs = net(inputs, use_new=True)
                except TypeError:
                    outputs = net(inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                local_targets = targets - int(task_start)
                local_logits = logits[:, int(task_start): int(task_end)]
                local_loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        local_logits,
                        local_targets,
                        reduction="sum",
                    ).detach().item()
                )
                full_loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        logits,
                        targets,
                        reduction="sum",
                    ).detach().item()
                )
                local_correct += int(local_logits.argmax(dim=1).eq(local_targets).sum().item())
                full_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                total += int(targets.numel())
    finally:
        net.train(was_training)

    denom = max(1, total)
    return {
        "samples": int(total),
        "local_loss": float(local_loss_sum / denom),
        "full_loss": float(full_loss_sum / denom),
        "local_acc": float(100.0 * local_correct / denom),
        "full_acc": float(100.0 * full_correct / denom),
    }


def _snapshot_delta_terms_cpu(model):
    base_model = model._unwrap_network() if hasattr(model, "_unwrap_network") else getattr(model, "_network", model)
    if hasattr(model, "_iter_delta_terms"):
        return {
            key: delta.detach().cpu().clone()
            for key, delta in model._iter_delta_terms(base_model)
        }
    if hasattr(model, "_iter_trainable"):
        return {
            key: param.detach().cpu().clone()
            for key, param in model._iter_trainable(base_model)
        }
    return {
        key: param.detach().cpu().clone()
        for key, param in base_model.named_parameters()
        if param.requires_grad
    }


def _fisher_quadratic_from_delta(delta_terms, fisher_delta, eps=1e-12):
    if not delta_terms:
        return {
            "available": 0,
            "delta_tensors": 0,
            "delta_norm_sq": 0.0,
            "delta_norm": 0.0,
            "fisher_raw_quadratic": 0.0,
            "fisher_quad": 0.0,
            "normalized_fisher_drift": 0.0,
        }

    delta_norm_sq = sum(
        float(delta.detach().cpu().float().pow(2).sum().item())
        for delta in delta_terms.values()
    )
    if not fisher_delta:
        return {
            "available": int(bool(delta_terms)),
            "delta_tensors": int(len(delta_terms)),
            "delta_norm_sq": float(delta_norm_sq),
            "delta_norm": float(delta_norm_sq ** 0.5),
            "fisher_raw_quadratic": 0.0,
            "fisher_quad": 0.0,
            "normalized_fisher_drift": 0.0,
        }

    fisher_raw = 0.0
    tensors = 0
    with torch.no_grad():
        for key, delta in delta_terms.items():
            fisher = fisher_delta.get(key)
            if fisher is None:
                continue
            delta_t = delta.detach().cpu().float()
            fisher_t = _match_tensor_shape_cpu(fisher, delta_t).float().clamp_min(0)
            update_sq = delta_t.pow(2)
            fisher_raw += float((fisher_t * update_sq).sum().item())
            tensors += 1

    return {
        "available": int(tensors > 0),
        "delta_tensors": int(tensors),
        "delta_norm_sq": float(delta_norm_sq),
        "delta_norm": float(delta_norm_sq ** 0.5),
        "fisher_raw_quadratic": float(fisher_raw),
        "fisher_quad": float(0.5 * fisher_raw),
        "normalized_fisher_drift": float(fisher_raw / (delta_norm_sq + float(eps))),
    }


def _delta_terms_between(current_terms, reference_terms):
    out = {}
    for key, cur in current_terms.items():
        ref = reference_terms.get(key)
        if ref is None:
            continue
        ref_t = _match_tensor_shape_cpu(ref, cur.detach().cpu())
        out[key] = cur.detach().cpu().float() - ref_t.float()
    return out


def _task0_fisher_quadratic_from_anchor(model, anchor_delta, fisher_delta, eps=1e-12):
    current_terms = _snapshot_delta_terms_cpu(model)
    return _fisher_quadratic_from_delta(
        _delta_terms_between(current_terms, anchor_delta),
        fisher_delta,
        eps=eps,
    )


def _clone_fisher_cpu(fisher):
    if not fisher:
        return {}
    return {
        key: value.detach().cpu().clone()
        for key, value in fisher.items()
        if hasattr(value, "detach")
    }


def _get_accumulated_fisher_for_task0(model):
    fisher = getattr(model, "_fisher_past_delta", None)
    if fisher:
        return fisher
    fisher = getattr(model, "_fisher", None)
    if fisher:
        return fisher
    return {}


def _fisher_trace_cpu(fisher):
    return sum(float(value.detach().cpu().float().sum().item()) for value in fisher.values())


def _normalize_fisher_cpu(fisher, eps=1e-12):
    if not fisher:
        return {}
    denom = _fisher_trace_cpu(fisher) + float(eps)
    return {key: value.detach().cpu() / denom for key, value in fisher.items()}


def _fisher_stats(prefix, fisher):
    if not fisher:
        return {
            f"{prefix}_available": 0,
            f"{prefix}_num_tensors": 0,
            f"{prefix}_numel": 0,
            f"{prefix}_trace": 0.0,
            f"{prefix}_fro": 0.0,
            f"{prefix}_l1": 0.0,
        }
    flat = torch.cat([value.detach().cpu().float().reshape(-1) for value in fisher.values()])
    return {
        f"{prefix}_available": 1,
        f"{prefix}_num_tensors": int(len(fisher)),
        f"{prefix}_numel": int(flat.numel()),
        f"{prefix}_trace": float(flat.sum().item()),
        f"{prefix}_fro": float(flat.pow(2).sum().sqrt().item()),
        f"{prefix}_l1": float(flat.abs().sum().item()),
    }


def _match_tensor_shape_cpu(value, reference):
    value_cpu = value.detach().cpu().to(dtype=reference.dtype)
    ref_cpu = reference.detach().cpu()
    if tuple(value_cpu.shape) == tuple(ref_cpu.shape):
        return value_cpu
    if value_cpu.ndim != ref_cpu.ndim:
        flat_value = value_cpu.reshape(-1)
        matched_flat = torch.zeros(ref_cpu.numel(), dtype=ref_cpu.dtype)
        n = min(int(flat_value.numel()), int(matched_flat.numel()))
        if n > 0:
            matched_flat[:n] = flat_value[:n]
        return matched_flat.view_as(ref_cpu)
    matched = torch.zeros_like(ref_cpu)
    slices = tuple(
        slice(0, min(int(value_cpu.shape[dim]), int(ref_cpu.shape[dim])))
        for dim in range(value_cpu.ndim)
    )
    matched[slices] = value_cpu[slices]
    return matched


def _flatten_fisher_pair(lhs, rhs):
    lhs_vals, rhs_vals = [], []
    for key in sorted(set(lhs.keys()) | set(rhs.keys())):
        left = lhs.get(key)
        right = rhs.get(key)
        if left is None and right is None:
            continue
        if left is None:
            right_cpu = right.detach().cpu().float()
            left_cpu = torch.zeros_like(right_cpu)
        else:
            left_cpu = left.detach().cpu().float()
        if right is None:
            right_cpu = torch.zeros_like(left_cpu)
        else:
            right_cpu = _match_tensor_shape_cpu(right, left_cpu).float()
        lhs_vals.append(left_cpu.reshape(-1))
        rhs_vals.append(right_cpu.reshape(-1))
    if not lhs_vals:
        return None, None
    return torch.cat(lhs_vals), torch.cat(rhs_vals)


def _fisher_pair_metrics(prefix, current, reference, eps=1e-12):
    cur_flat, ref_flat = _flatten_fisher_pair(current, reference)
    if cur_flat is None or ref_flat is None:
        return {
            f"{prefix}_available": 0,
            f"{prefix}_cosine_sim": 0.0,
            f"{prefix}_mismatch": 1.0,
            f"{prefix}_rel_fro_wrt_ref": 0.0,
            f"{prefix}_rel_l1_wrt_ref": 0.0,
            f"{prefix}_trace_ratio_current_to_ref": 0.0,
            f"{prefix}_top1pct_ref_coverage": 0.0,
        }

    cur_fro = float(cur_flat.pow(2).sum().sqrt().item())
    ref_fro = float(ref_flat.pow(2).sum().sqrt().item())
    cur_l1 = float(cur_flat.abs().sum().item())
    ref_l1 = float(ref_flat.abs().sum().item())
    diff = cur_flat - ref_flat
    cosine = float((cur_flat * ref_flat).sum().item() / max(cur_fro * ref_fro, float(eps)))
    cosine = max(-1.0, min(1.0, cosine))
    k = max(1, int(ref_flat.numel() * 0.01))
    ref_top_idx = ref_flat.topk(k, largest=True).indices
    ref_top_sum = float(ref_flat[ref_top_idx].sum().item())
    cur_on_ref_top_sum = float(cur_flat[ref_top_idx].sum().item())
    return {
        f"{prefix}_available": 1,
        f"{prefix}_cosine_sim": cosine,
        f"{prefix}_mismatch": float(1.0 - cosine),
        f"{prefix}_rel_fro_wrt_ref": float(diff.pow(2).sum().sqrt().item() / max(ref_fro, float(eps))),
        f"{prefix}_rel_l1_wrt_ref": float(diff.abs().sum().item() / max(ref_l1, float(eps))),
        f"{prefix}_trace_ratio_current_to_ref": float(cur_l1 / max(ref_l1, float(eps))),
        f"{prefix}_top1pct_ref_coverage": float(cur_on_ref_top_sum / max(ref_top_sum, float(eps))),
    }


def _compute_task0_local_delta_fisher(model, fixed_batches, task_start, task_end, normalize=True, eps=1e-12):
    if not fixed_batches:
        return {}, 0

    net = model._unwrap_network() if hasattr(model, "_unwrap_network") else getattr(model, "_network", model)
    device = getattr(model, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = bool(getattr(net, "training", False))
    fisher = {}
    total = 0

    net.eval()
    use_delta_hooks = all(
        hasattr(model, name)
        for name in ("_set_delta_hook_state", "_clear_delta_hook_grads", "_iter_hook_grads")
    )
    if use_delta_hooks:
        model._set_delta_hook_state(net, True)
    try:
        for inputs_cpu, targets_cpu in fixed_batches:
            inputs = inputs_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)
            local_targets = targets - int(task_start)

            net.zero_grad(set_to_none=True)
            if use_delta_hooks:
                model._clear_delta_hook_grads(net)
            with torch.enable_grad():
                try:
                    outputs = net(inputs, use_new=True)
                except TypeError:
                    outputs = net(inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                local_logits = logits[:, int(task_start): int(task_end)]
                torch.nn.functional.cross_entropy(local_logits, local_targets).backward()

            batch_size = int(inputs.size(0))
            total += batch_size
            if use_delta_hooks:
                grad_iter = model._iter_hook_grads(net)
            elif hasattr(model, "_iter_trainable"):
                grad_iter = (
                    (name, param.grad)
                    for name, param in model._iter_trainable(net)
                    if param.grad is not None
                )
            else:
                grad_iter = (
                    (name, param.grad)
                    for name, param in net.named_parameters()
                    if param.requires_grad and param.grad is not None
                )
            for key, grad in grad_iter:
                grad_sq = (grad.detach().cpu() ** 2) * batch_size
                fisher[key] = grad_sq if key not in fisher else fisher[key] + grad_sq
    finally:
        if use_delta_hooks:
            model._set_delta_hook_state(net, False)
            model._clear_delta_hook_grads(net)
        net.zero_grad(set_to_none=True)
        net.train(was_training)

    if total > 0:
        for key in fisher:
            fisher[key] = fisher[key] / float(total)
    if normalize:
        fisher = _normalize_fisher_cpu(fisher, eps=eps)
    return fisher, int(total)


def _init_task0_trajectory_eval(args, data_manager, model):
    if not _as_bool(args.get("task0_trajectory_eval", False)):
        return None

    task_idx = int(args.get("task0_trajectory_task", 0))
    task_start, task_end = data_manager.get_task_class_range(task_idx)
    batch_size = int(args.get("task0_trajectory_batch_size", args.get("batch_size", 128)))
    num_workers = int(args.get("task0_trajectory_num_workers", args.get("eval_num_workers", 8)))
    max_batches = _parse_max_batches(args.get("task0_trajectory_max_batches", 20), default=20)
    splits = [
        item.strip().lower()
        for item in str(args.get("task0_trajectory_splits", "train,test")).split(",")
        if item.strip()
    ]
    fixed = {
        split: _collect_task_fixed_batches_cpu(
            data_manager,
            task_start,
            task_end,
            split,
            batch_size,
            num_workers,
            max_batches,
        )
        for split in splits
    }
    anchor_metrics = {
        split: _evaluate_fixed_task_batches(model, batches, task_start, task_end)
        for split, batches in fixed.items()
    }
    fisher_delta = _clone_fisher_cpu(_get_accumulated_fisher_for_task0(model))
    task0_fisher_eval = _as_bool(args.get("task0_fisher_recompute_eval", False))
    task0_fisher_split = str(args.get("task0_fisher_split", "train")).strip().lower()
    task0_fisher_batch_size = int(args.get("task0_fisher_batch_size", args.get("batch_size", batch_size)))
    task0_fisher_num_workers = int(args.get("task0_fisher_num_workers", args.get("eval_num_workers", num_workers)))
    task0_fisher_max_batches = _parse_max_batches(
        args.get("task0_fisher_max_batches", args.get("ewc_max_batches", None)),
        default=None,
    )
    task0_fisher_normalize = _as_bool(
        args.get("task0_fisher_normalize", args.get("ewc_normalize_fisher", True))
    )
    task0_fisher_fixed = []
    task0_fisher_anchor = {}
    task0_fisher_anchor_samples = 0
    if task0_fisher_eval:
        task0_fisher_fixed = _collect_task_fixed_batches_cpu(
            data_manager,
            task_start,
            task_end,
            task0_fisher_split,
            task0_fisher_batch_size,
            task0_fisher_num_workers,
            task0_fisher_max_batches,
        )
        task0_fisher_anchor, task0_fisher_anchor_samples = _compute_task0_local_delta_fisher(
            model,
            task0_fisher_fixed,
            task_start,
            task_end,
            normalize=task0_fisher_normalize,
            eps=float(args.get("mechanism_eps", 1e-12)),
        )
    ctx = {
        "task_idx": int(task_idx),
        "task_start": int(task_start),
        "task_end": int(task_end),
        "fixed": fixed,
        "anchor_metrics": anchor_metrics,
        "anchor_delta": _snapshot_delta_terms_cpu(model),
        "fisher_delta": fisher_delta,
        "lambda_ewc": float(args.get("ewc_lambda", args.get("lambda", 0.0))),
        "eps": float(args.get("mechanism_eps", 1e-12)),
        "task0_fisher_eval": bool(task0_fisher_eval),
        "task0_fisher_split": task0_fisher_split,
        "task0_fisher_fixed": task0_fisher_fixed,
        "task0_fisher_anchor": task0_fisher_anchor,
        "task0_fisher_anchor_samples": int(task0_fisher_anchor_samples),
        "task0_fisher_batch_size": int(task0_fisher_batch_size),
        "task0_fisher_max_batches": task0_fisher_max_batches,
        "task0_fisher_normalize": bool(task0_fisher_normalize),
    }
    logging.info(
        "[Task0Trajectory] initialized task=%d classes=%d-%d splits=%s max_batches=%s fisher_tensors=%d task0_fisher_eval=%s task0_fisher_anchor_tensors=%d samples=%d",
        task_idx,
        task_start,
        task_end,
        ",".join(sorted(fixed.keys())),
        "all" if max_batches is None else str(max_batches),
        len(fisher_delta),
        bool(task0_fisher_eval),
        len(task0_fisher_anchor),
        int(task0_fisher_anchor_samples),
    )
    return ctx


def _evaluate_task0_trajectory(model, ctx, step):
    eps = ctx.get("eps", 1e-12)
    current_terms = _snapshot_delta_terms_cpu(model)
    anchor_terms = ctx.get("anchor_delta", {})
    prev_terms = ctx.get("prev_terms", anchor_terms)
    cum_delta_terms = _delta_terms_between(current_terms, anchor_terms)
    step_delta_terms = _delta_terms_between(current_terms, prev_terms)
    accum_fisher = _clone_fisher_cpu(_get_accumulated_fisher_for_task0(model))
    loaded_anchor_fisher = ctx.get("fisher_delta", {})

    quad = _fisher_quadratic_from_delta(
        cum_delta_terms,
        loaded_anchor_fisher,
        eps=eps,
    )
    step_sq = _fisher_quadratic_from_delta(step_delta_terms, {}, eps=eps)
    cum_accum_quad = _fisher_quadratic_from_delta(cum_delta_terms, accum_fisher, eps=eps)
    step_loaded_anchor_quad = _fisher_quadratic_from_delta(
        step_delta_terms,
        loaded_anchor_fisher,
        eps=eps,
    )
    step_accum_quad = _fisher_quadratic_from_delta(step_delta_terms, accum_fisher, eps=eps)
    out = {
        "anchor_task": int(ctx["task_idx"]),
        "anchor_task_start": int(ctx["task_start"]),
        "anchor_task_end": int(ctx["task_end"]),
        "eval_step": int(step),
        "fisher_available": int(bool(quad["available"])),
        "fisher_delta_tensors": int(quad["delta_tensors"]),
        "delta_norm_sq": float(quad["delta_norm_sq"]),
        "delta_norm": float(quad["delta_norm"]),
        "fisher_raw_quadratic": float(quad["fisher_raw_quadratic"]),
        "fisher_quad": float(quad["fisher_quad"]),
        "lambda_fisher_quad": float(ctx.get("lambda_ewc", 0.0) * quad["fisher_quad"]),
        "normalized_fisher_drift": float(quad["normalized_fisher_drift"]),
        "step_delta_sq": float(step_sq["delta_norm_sq"]),
        "cum_delta0_sq": float(quad["delta_norm_sq"]),
        "step_delta_F0_loaded_anchor_delta": float(step_loaded_anchor_quad["fisher_raw_quadratic"]),
        "step_delta_Facc_delta": float(step_accum_quad["fisher_raw_quadratic"]),
        "cum_delta0_F0_loaded_anchor_delta0": float(quad["fisher_raw_quadratic"]),
        "cum_delta0_Facc_delta0": float(cum_accum_quad["fisher_raw_quadratic"]),
        "norm_cum_delta0_F0_loaded_anchor_delta0": float(quad["normalized_fisher_drift"]),
        "norm_cum_delta0_Facc_delta0": float(cum_accum_quad["normalized_fisher_drift"]),
    }
    if ctx.get("task0_fisher_eval", False):
        current_fisher, current_samples = _compute_task0_local_delta_fisher(
            model,
            ctx.get("task0_fisher_fixed", []),
            ctx["task_start"],
            ctx["task_end"],
            normalize=bool(ctx.get("task0_fisher_normalize", True)),
            eps=ctx.get("eps", 1e-12),
        )
        out.update({
            "task0_fisher_eval": 1,
            "task0_fisher_split": ctx.get("task0_fisher_split", "train"),
            "task0_fisher_head_scope": "local_task_head",
            "task0_fisher_batch_size": int(ctx.get("task0_fisher_batch_size", 0)),
            "task0_fisher_max_batches": (
                "all" if ctx.get("task0_fisher_max_batches", None) is None
                else int(ctx.get("task0_fisher_max_batches"))
            ),
            "task0_fisher_normalized": int(bool(ctx.get("task0_fisher_normalize", True))),
            "task0_fisher_current_num_samples": int(current_samples),
            "task0_fisher_anchor_num_samples": int(ctx.get("task0_fisher_anchor_samples", 0)),
        })
        out.update(_fisher_stats("task0_fisher_current", current_fisher))
        task0_anchor_fisher = ctx.get("task0_fisher_anchor", {})
        prev_task0_fisher = ctx.get("prev_task0_fisher", task0_anchor_fisher)
        cum_anchor_quad = _fisher_quadratic_from_delta(
            cum_delta_terms,
            task0_anchor_fisher,
            eps=eps,
        )
        cum_dynamic_quad = _fisher_quadratic_from_delta(
            cum_delta_terms,
            current_fisher,
            eps=eps,
        )
        step_anchor_quad = _fisher_quadratic_from_delta(
            step_delta_terms,
            task0_anchor_fisher,
            eps=eps,
        )
        step_dynamic_prev_quad = _fisher_quadratic_from_delta(
            step_delta_terms,
            prev_task0_fisher,
            eps=eps,
        )
        step_dynamic_current_quad = _fisher_quadratic_from_delta(
            step_delta_terms,
            current_fisher,
            eps=eps,
        )
        out.update({
            "step_delta_F0_anchor_delta": float(step_anchor_quad["fisher_raw_quadratic"]),
            "step_delta_F0_dynamic_prev_delta": float(step_dynamic_prev_quad["fisher_raw_quadratic"]),
            "step_delta_F0_dynamic_current_delta": float(step_dynamic_current_quad["fisher_raw_quadratic"]),
            "cum_delta0_F0_anchor_delta0": float(cum_anchor_quad["fisher_raw_quadratic"]),
            "cum_delta0_F0_dynamic_delta0": float(cum_dynamic_quad["fisher_raw_quadratic"]),
            "norm_cum_delta0_F0_anchor_delta0": float(cum_anchor_quad["normalized_fisher_drift"]),
            "norm_cum_delta0_F0_dynamic_delta0": float(cum_dynamic_quad["normalized_fisher_drift"]),
        })
        out.update(_fisher_pair_metrics(
            "task0_fisher_anchor",
            current_fisher,
            task0_anchor_fisher,
            eps=eps,
        ))
        out.update(_fisher_pair_metrics(
            "task0_fisher_loaded_anchor",
            current_fisher,
            loaded_anchor_fisher,
            eps=eps,
        ))
        out.update(_fisher_pair_metrics(
            "task0_fisher_accum",
            current_fisher,
            accum_fisher,
            eps=eps,
        ))
        ctx["prev_task0_fisher"] = _clone_fisher_cpu(current_fisher)
    else:
        out["task0_fisher_eval"] = 0
    ctx["prev_terms"] = _clone_fisher_cpu(current_terms)

    denom_quad = abs(float(quad["fisher_quad"])) + float(eps)
    for split, batches in ctx["fixed"].items():
        cur = _evaluate_fixed_task_batches(
            model,
            batches,
            ctx["task_start"],
            ctx["task_end"],
        )
        base = ctx["anchor_metrics"][split]
        for key, value in cur.items():
            out[f"{split}_{key}"] = float(value)
        out[f"{split}_base_local_loss"] = float(base["local_loss"])
        out[f"{split}_base_full_loss"] = float(base["full_loss"])
        out[f"{split}_base_local_acc"] = float(base["local_acc"])
        out[f"{split}_base_full_acc"] = float(base["full_acc"])
        local_delta = float(cur["local_loss"] - base["local_loss"])
        full_delta = float(cur["full_loss"] - base["full_loss"])
        out[f"{split}_local_loss_delta"] = local_delta
        out[f"{split}_full_loss_delta"] = full_delta
        out[f"{split}_local_signed_error"] = float(local_delta - quad["fisher_quad"])
        out[f"{split}_local_abs_error"] = float(abs(local_delta - quad["fisher_quad"]))
        out[f"{split}_local_relative_error"] = float(abs(local_delta - quad["fisher_quad"]) / (abs(local_delta) + denom_quad))
        out[f"{split}_full_signed_error"] = float(full_delta - quad["fisher_quad"])
        out[f"{split}_full_abs_error"] = float(abs(full_delta - quad["fisher_quad"]))
        out[f"{split}_full_relative_error"] = float(abs(full_delta - quad["fisher_quad"]) / (abs(full_delta) + denom_quad))
    logging.info("[Task0Trajectory] step=%d metrics=%s", int(step), out)
    return out

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)



def _train(args):

    # ---------------------------
    # 0) I/O & logging
    # ---------------------------

    # Root layout: <outputs_root>/logs_inc[_lora]/<model>/<opt>/<dataset>/<seed>/<prefix>/<mode>/
    # Allow configurable output root; default to outputs_logs for cleaner repo layout.
    outputs_root = args.get("outputs_root", "outputs_logs")

    is_lora = "lora" in str(args["model_name"])
    # optimizer tag
    if args.get("optimizer_tag_override", None):
        opt_tag = str(args["optimizer_tag_override"])
    else:
        schedule_tag = _optimizer_schedule_tag(args.get("optimizer_type_by_task", None))
        if schedule_tag is not None:
            opt_tag = schedule_tag
        elif args["optimizer_type"] == "rwp":
            opt_tag = f"rwp_{args.get('rwp_range')}"
        else:
            opt_tag = str(args["optimizer_type"])
    # mode tag
    if args["optimizer_type"] == "rwp":
        mode = "rwp_full" if args.get("rwp_range") == "full" else ("rwp_lora" if args.get("rwp_range") == "lora" else "rwp")
    else:
        mode = "exp_run"

    logs_root = os.path.join(
        outputs_root,
        "logs_inc_lora" if is_lora else "logs_inc",
        str(args["model_name"]),
        opt_tag,
        str(args["dataset"]),
        str(args["seed"]),
        str(args["prefix"]),
        mode,
    )

    os.makedirs(logs_root, exist_ok=True)

    # Checkpoints and auxiliary artifacts live alongside logs for each run.
    checkpoint_dir = os.path.join(logs_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args["filepath"] = checkpoint_dir if checkpoint_dir.endswith(os.sep) else (checkpoint_dir + os.sep)
    args.setdefault("feature_flat_save_path", checkpoint_dir)

    # Logs for a specific increment are nested under increment id.
    log_dir = os.path.join(logs_root, str(args["increment"]))
    os.makedirs(log_dir, exist_ok=True)

    logfilename = os.path.join(
        log_dir,
        f'{args["prefix"]}_{args["backbone_type"]}'
    )

    metrics_path = metrics_json_path(log_dir, logfilename)
    # ===  ===

    # ———  logging（ handlers，） ———
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # ---------------------------
    # 1) Env & data/model
    # ---------------------------

    set_random(args["seed"])
    set_device(args)
    print_args(args)

    class_shuffle = args.get("class_shuffle", args.get("shuffle", True))

    data_manager = DataManager(
        args["dataset"],
        class_shuffle,
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["task_increments"] = list(data_manager._increments)  # per-task class counts
    try:
        _max_tasks_req = args.get("max_train_tasks", data_manager.nb_tasks)
        _max_tasks_req = int(_max_tasks_req) if _max_tasks_req is not None else data_manager.nb_tasks
    except Exception:
        _max_tasks_req = data_manager.nb_tasks
    nb_tasks = min(data_manager.nb_tasks, max(1, _max_tasks_req))
    if nb_tasks < data_manager.nb_tasks:
        logging.info("Restrict training to %d tasks (of %d total)", nb_tasks, data_manager.nb_tasks)
    args["nb_tasks"] = nb_tasks
    
    model = factory.get_model(args["model_name"], args)
    

    # Ensure the network/backbone is on-device for any pre-training evals
    net_obj = getattr(model, "_network", None)
    if net_obj is None:
        raise AttributeError("Model has no _network attribute")
    bb = getattr(net_obj, "backbone", None)
    if bb is not None and hasattr(bb, "to"):
        bb.to(model._device)
    else:
        net_obj.to(model._device)

    resume_task_arg = args.get("resume_from_task", -1)
    resume_task = -1 if resume_task_arg in {None, ""} else int(resume_task_arg)
    start_task = 0
    if resume_task >= 0:
        resume_ckpt_dir = args.get("resume_from_checkpoint_dir", None)
        if not resume_ckpt_dir:
            raise ValueError("resume_from_task requires resume_from_checkpoint_dir")
        if not hasattr(model, "load_task_checkpoint"):
            raise AttributeError(
                f"Model {args.get('model_name')} does not implement load_task_checkpoint"
            )
        if resume_task >= nb_tasks:
            raise ValueError(
                f"resume_from_task={resume_task} must be smaller than nb_tasks={nb_tasks}"
            )
        model.load_task_checkpoint(resume_ckpt_dir, resume_task, data_manager)
        start_task = resume_task + 1
        net_obj = getattr(model, "_network", None)
        logging.info(
            "[Resume] Continuing training from task %d to task %d",
            start_task,
            nb_tasks - 1,
        )

    # Load pre-trained weights from a final_model.pt (eval-only workflow).
    # Training is then skipped by setting init_epoch=0 epochs=0; after_task()
    # still runs to compute Fisher and all evaluations from scratch.
    _pretrained_path = args.get("pretrained_model_path", None)
    if _pretrained_path:
        if not os.path.exists(_pretrained_path):
            raise FileNotFoundError(f"pretrained_model_path not found: {_pretrained_path}")
        _ckpt = torch.load(_pretrained_path, map_location="cpu", weights_only=False)
        _sd = _ckpt.get("model_state_dict", _ckpt.get("network_state", {}))
        _net = getattr(model, "_network", None)
        if _net is not None:
            if hasattr(_net, "module"):
                _net = _net.module
            _missing, _unexpected = _net.load_state_dict(_sd, strict=False)
            if _missing:
                logging.warning("[Pretrained] Missing keys: %s", _missing[:5])
            logging.info(
                "[Pretrained] Loaded %d params from %s (missing=%d unexpected=%d)",
                len(_sd), _pretrained_path, len(_missing), len(_unexpected),
            )

    for name, _param in net_obj.named_parameters():
        logging.info(" net_obj param name: %s", name)

    class_ranges = [data_manager.get_task_class_range(task_idx) for task_idx in range(nb_tasks)]
    flat_eval_tasks = _resolve_task_indices(args.get("flat_eval_task_indices", None), nb_tasks)
    feature_eval_tasks = _resolve_task_indices(args.get("feature_flat_task_indices", None), nb_tasks)
    task0_trajectory_ctx = None
    if start_task > 0 and resume_task == int(args.get("task0_trajectory_task", 0)):
        try:
            task0_trajectory_ctx = _init_task0_trajectory_eval(args, data_manager, model)
            if task0_trajectory_ctx is not None:
                baseline_metrics = _evaluate_task0_trajectory(model, task0_trajectory_ctx, resume_task)
                write_step_metrics(
                    metrics_path,
                    section="task0_trajectory",
                    step=resume_task,
                    metrics=baseline_metrics,
                    matrix=None,
                    json_safe=json_safe,
                )
        except Exception as _traj_init_exc:
            task0_trajectory_ctx = None
            logging.exception("[Task0Trajectory] Failed to initialize: %s", _traj_init_exc)

    if start_task > 0 and bool(args.get("eval_only_after_resume", False)):
        logging.info("[ResumeEval] Evaluating resumed task %d without additional training", resume_task)
        if not getattr(model, "test_loader", None):
            _batch_size = int(args.get("batch_size", 128))
            _num_workers = int(args.get("eval_num_workers", 8))
            _task_start, _task_end = data_manager.get_task_class_range(resume_task)
            _train_ds = data_manager.get_dataset(
                np.arange(_task_start, _task_end), source="train", mode="train"
            )
            model.train_loader = data_manager.build_dataloader(
                _train_ds, batch_size=_batch_size, shuffle=True, num_workers=_num_workers,
            )
            _test_ds = data_manager.get_dataset(
                np.arange(0, _task_end), source="test", mode="test"
            )
            model.test_loader = data_manager.build_dataloader(
                _test_ds, batch_size=_batch_size, shuffle=False, num_workers=_num_workers,
            )
            logging.info(
                "[ResumeEval] Built data loaders for task %d (classes %d–%d)",
                resume_task, _task_start, _task_end,
            )
        cnn_accy, nme_accy = model.eval_task()
        logging.info("[ResumeEval] CNN: %s", cnn_accy["grouped"])
        if nme_accy is not None:
            logging.info("[ResumeEval] NME: %s", nme_accy["grouped"])
        try:
            if hasattr(model, "evaluate_loaded_checkpoint"):
                mechanism_metrics = model.evaluate_loaded_checkpoint(resume_task)
            elif hasattr(model, "get_mechanism_metrics"):
                mechanism_metrics = model.get_mechanism_metrics()
            else:
                mechanism_metrics = {}
            if mechanism_metrics:
                write_step_metrics(
                    metrics_path,
                    section="mechanism",
                    step=resume_task,
                    metrics=mechanism_metrics,
                    matrix=None,
                    json_safe=json_safe,
                )
        except Exception as _resume_eval_exc:
            logging.exception("[ResumeEval] Failed mechanism evaluation: %s", _resume_eval_exc)
        return
    
    # ---------------------------



    # ---------------------------
    # 2) Switches for new metrics
    # ---------------------------
    # head_eval = args.get("head_eval", False)
    # head_R = np.full((nb_tasks, nb_tasks), np.nan, dtype=float)
    # def _evaluate_head_row(task_idx):
    #     row = np.full(nb_tasks, np.nan, dtype=float)
    #     for j in range(task_idx + 1):
    #         loader = _build_loader(class_ranges[j], source="test", mode="test")
    #         row[j] = model._compute_accuracy(model._network, loader)
    #     return row

    # 4) Main CL loop
    # ---------------------------
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    mechanism_curve = []
    epoch_dynamics_curve = []

    if start_task > 0:
        resume_metrics_json = args.get("resume_from_metrics_json", None)
        if not resume_metrics_json:
            raise ValueError(
                "resume_from_task requires resume_from_metrics_json so final "
                "metrics keep the complete prefix accuracy matrix"
            )
        cnn_matrix = _matrix_rows_from_json(resume_metrics_json, "cnn", resume_task)
        cnn_curve["top1"] = _top1_curve_from_rows(cnn_matrix)
        try:
            nme_matrix = _matrix_rows_from_json(resume_metrics_json, "nme", resume_task)
            nme_curve["top1"] = _top1_curve_from_rows(nme_matrix)
        except Exception:
            nme_matrix = []
            nme_curve["top1"] = []
        logging.info(
            "[Resume] Loaded %d CNN prefix rows from %s",
            len(cnn_matrix),
            resume_metrics_json,
        )

    logging.info("Start traing CL")
    for task in range(start_task, nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()



        # ---- Curves & matrices (restore sorted keys for stable ordering) ----
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        # —— CNN/NME ：（） ——
        T_partial = task + 1
        if len(cnn_matrix) > 0:
            cnn_time_by_task_partial = assemble_eval_matrix(cnn_matrix, T_partial, orientation="time_by_task")
            write_step_metrics(
                metrics_path, "cnn", step=task,
                metrics=None,
                matrix=cnn_time_by_task_partial,
                json_safe=json_safe,
            )

        if nme_accy is not None and len(nme_matrix) > 0:
            nme_time_by_task_partial = assemble_eval_matrix(nme_matrix, T_partial, orientation="time_by_task")
            write_step_metrics(
                metrics_path, "nme", step=task,
                metrics=None,
                matrix=nme_time_by_task_partial,
                json_safe=json_safe,
            )

        if task0_trajectory_ctx is not None:
            try:
                trajectory_metrics = _evaluate_task0_trajectory(model, task0_trajectory_ctx, task)
                write_step_metrics(
                    metrics_path,
                    section="task0_trajectory",
                    step=task,
                    metrics=trajectory_metrics,
                    matrix=None,
                    json_safe=json_safe,
                )
            except Exception as _traj_exc:
                logging.exception("[Task0Trajectory] Failed at task %d: %s", task, _traj_exc)

        # Save the final trained model BEFORE evaluation (FQA/Hessian may take hours;
        # saving first ensures the checkpoint survives even if eval is interrupted).
        try:
            if (task == nb_tasks - 1) and bool(args.get("save_final_model", False)):
                net_to_save = getattr(model, "_network", None)
                if net_to_save is not None:
                    if hasattr(net_to_save, "module"):
                        net_to_save = net_to_save.module
                    final_ckpt = {
                        "method": args.get("model_name", "model_name"),
                        "dataset": args.get("dataset", "dataset"),
                        "init_cls": args.get("init_cls", "init_cls"),
                        "increment": args.get("init_cls", "init_cls"),
                        "seed": args.get("seed", "seed"),
                        "tasks": int(nb_tasks - 1),
                        "model_state_dict": net_to_save.state_dict(),
                    }
                    final_path = os.path.join(checkpoint_dir, "final_model.pt")
                    torch.save(final_ckpt, final_path)
                    logging.info("[FinalSave] Saved final model to %s (before eval)", final_path)
        except Exception as _save_exc:
            logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)

        model.after_task()
        if (bool(args.get("mechanism_eval", False)) or bool(args.get("compute_full_fisher", False))) and hasattr(model, "get_mechanism_metrics"):
            try:
                mechanism_metrics = model.get_mechanism_metrics()
                if mechanism_metrics:
                    mechanism_curve.append(mechanism_metrics)
                    write_step_metrics(
                        metrics_path,
                        section="mechanism",
                        step=task,
                        metrics=mechanism_metrics,
                        matrix=None,
                        json_safe=json_safe,
                    )
            except Exception as _mech_exc:
                logging.exception("[MechanismEval] Failed to write mechanism metrics: %s", _mech_exc)

        if bool(args.get("epoch_dynamics_eval", args.get("train_dynamics_eval", False))) and hasattr(model, "get_epoch_dynamics_metrics"):
            try:
                epoch_dynamics = model.get_epoch_dynamics_metrics()
                if epoch_dynamics:
                    epoch_dynamics_curve.extend(epoch_dynamics)
                    write_step_metrics(
                        metrics_path,
                        section="epoch_dynamics",
                        step=task,
                        metrics={
                            "task": int(task),
                            "epochs": epoch_dynamics,
                        },
                        matrix=None,
                        json_safe=json_safe,
                    )
            except Exception as _epoch_dyn_exc:
                logging.exception("[EpochDynamics] Failed to write epoch dynamics: %s", _epoch_dyn_exc)
                if bool(args.get("epoch_dynamics_required", False)):
                    raise

        # Save per-task checkpoint for staged / shared-stage-1 workflows.
        _staged_ckpt_dir = args.get("save_task_checkpoint_dir", None)
        if _staged_ckpt_dir and hasattr(model, "save_task_checkpoint"):
            try:
                model.save_task_checkpoint(str(_staged_ckpt_dir), task)
            except Exception as _ck_exc:
                logging.exception("[TaskCheckpoint] Failed to save task %d: %s", task, _ck_exc)
                if bool(args.get("save_task_checkpoint_required", False)):
                    raise

        gc.collect();
        torch.cuda.empty_cache();
        torch.cuda.ipc_collect()


        # Optional per-task linear probe at final step (disabled by default)
        try:
            if  (task == nb_tasks - 1) and bool(args.get("linear_probe_softmax_per_task_eval", False)):
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                was_training = net.training
                net.eval()

                set_random(args["seed"])

                #  TUNA， adapter（）
                if str(args.get("model_name", "")).lower() == "tuna":
                    module = net
                    bb = getattr(module, "backbone", None)
                    if bb is not None and hasattr(bb, "adapter_list"):
                        fused_id = (len(bb.adapter_list) + 1) if (len(bb.adapter_list) > 0 and getattr(bb, "merged_adapter", None) is not None) else len(bb.adapter_list)
                        if hasattr(module, "set_active_adapter"):
                            module.set_active_adapter(fused_id)
                        else:
                            try:
                                setattr(bb, "active_adapter", fused_id)
                            except Exception:
                                pass

                probe_log_interval = args.get("probe_log_interval", None)
                probe_eval_interval = max(1, int(args.get("probe_fit_epochs", 50)) // 5)

                # ： j（0..task），/
                acc_final_list = []
                acc_base_list = []
                for j in range(task + 1):
                    start_j, end_j = class_ranges[j][0], class_ranges[j][1]
                    num_classes_j = end_j - start_j
                    if num_classes_j <= 0:
                        acc_final_list.append(float("nan"))
                        acc_base_list.append(float("nan"))
                        continue

                    train_ds_j = data_manager.get_dataset(
                        np.arange(start_j, end_j), source="train", mode=args.get("probe_train_mode", "train")
                    )
                    test_ds_j = data_manager.get_dataset(
                        np.arange(start_j, end_j), source="test", mode=args.get("probe_test_mode", "test")
                    )

                    train_loader_j = DataLoader(
                        train_ds_j,
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        shuffle=True,
                        num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                        persistent_workers=False,
                    )
                    test_loader_j = DataLoader(
                        test_ds_j,
                        batch_size=int(args.get("probe_fit_test_batch_size", 128)),
                        shuffle=False,
                        num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                        persistent_workers=False,
                    )

                    # （）
                    head_final_j = fit_linear_probe_softmax_head(
                        net, train_loader_j,
                        class_offset=start_j, num_classes=num_classes_j, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_j,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix=f"[LP-Softmax][PerTask][Final][t{j:02d}]",
                    )
                    acc_final_j = evaluate_linear_probe_softmax_with_head(
                        head_final_j, net, test_loader_j,
                        class_offset=start_j, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )

                    # （base/backbone， TUNA）
                    if str(args.get("model_name", "")).lower() != "tuna":
                        base_view = _FeatureView(getattr(net, "backbone", net), which="base")
                        base_view.eval()
                        head_base_j = fit_linear_probe_softmax_head(
                            base_view, train_loader_j,
                            class_offset=start_j, num_classes=num_classes_j, device=model._device,
                            epochs=int(args.get("probe_fit_epochs", 50)),
                            lr=float(args.get("probe_fit_lr", 5e-3)),
                            weight_decay=float(args.get("probe_fit_wd", 0.0)),
                            batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                            max_train_batches=args.get("probe_train_max_batches", None),
                            monitor_loader=test_loader_j,
                            monitor_max_batches=args.get("probe_test_max_batches", None),
                            log_interval=probe_log_interval,
                            eval_interval=probe_eval_interval,
                            log_prefix=f"[LP-Softmax][PerTask][Base][t{j:02d}]",
                        )
                        acc_base_j = evaluate_linear_probe_softmax_with_head(
                            head_base_j, base_view, test_loader_j,
                            class_offset=start_j, device=model._device,
                            max_test_batches=args.get("probe_test_max_batches", None)
                        )
                    else:
                        acc_base_j = 0.0

                    logging.info("[LP-PerTask][t%02d] final-model Acc=%.2f | base-model Acc=%.2f", j, acc_final_j, acc_base_j)
                    acc_final_list.append(float(acc_final_j))
                    acc_base_list.append(float(acc_base_j))

                    # ，
                    try:
                        for _m in ["head_final_j", "head_base_j", "base_view"]:
                            if _m in locals() and locals()[_m] is not None:
                                mod = locals()[_m]
                                if isinstance(mod, torch.nn.Module):
                                    with torch.inference_mode():
                                        mod.to("cpu")
                                    for p in mod.parameters():
                                        p.grad = None
                    except Exception:
                        pass
                    head_final_j = None; head_base_j = None; base_view = None
                    train_loader_j = None; test_loader_j = None
                    train_ds_j = None; test_ds_j = None
                    gc.collect();
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

                #  JSON（ + ）
                avg_final = float(np.nanmean(acc_final_list)) if len(acc_final_list) > 0 else float("nan")
                avg_base  = float(np.nanmean(acc_base_list)) if len(acc_base_list) > 0 else float("nan")
                write_step_metrics(
                    metrics_path,
                    section="probe_softmax_per_task",
                    step=task,
                    metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    matrix=None,
                    json_safe=json_safe,
                )
                write_final_metrics(
                    metrics_path,
                    section="probe_softmax_per_task",
                    final_metrics={
                        "final_model_per_task": acc_final_list,
                        "base_model_per_task": acc_base_list,
                        "avg_final": avg_final,
                        "avg_base": avg_base,
                    },
                    final_matrix=None,
                    json_safe=json_safe,
                )
        finally:
            pass

        # linear_probe_softmax_joint_seen_eval
        try:
            if (task == nb_tasks - 1) and args.get("linear_probe_softmax_joint_seen_eval", False):
                net = getattr(model, "_network", model)
                if hasattr(net, "module"):
                    net = net.module
                # ，
                was_training = net.training
                net.eval()


                set_random(args["seed"])

                # 1)～5) （/ head、）
                start_seen = class_ranges[0][0]
                end_seen = class_ranges[task][1]
                num_classes =  end_seen - start_seen 

                train_dataset_all = data_manager.get_dataset(
                    np.arange(start_seen, end_seen), source="train",
                    mode=args.get("probe_train_mode", "train")
                )
                test_dataset_all = data_manager.get_dataset(
                    np.arange(start_seen, end_seen), source="test",
                    mode=args.get("probe_test_mode", "test")
                )

             
                train_loader_all = DataLoader(
                    train_dataset_all,
                    batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                    shuffle=True,
                    num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                    persistent_workers=False,
                )
                test_loader_all = DataLoader(
                    test_dataset_all,
                    batch_size=int(args.get("probe_fit_test_batch_size", 128)),
                    shuffle=False,
                    num_workers=int(args.get("linear_probe_eval_num_workers", 0)),
                    persistent_workers=False,
                    
                )

                probe_log_interval = args.get("probe_log_interval", None)
                
                
                probe_eval_interval = max(1, int(args.get("probe_fit_epochs", 50)) // 5)

                # module = net.module if isinstance(net, torch.nn.DataParallel) else net
                # # ：TUNA “ adapter” = cur_task + 1；
                # #  merge（ adapter）， 0。
                # universal_id = (self._cur_task + 1) if (getattr(module.backbone, "merge", None) and self._cur_task > 0) else 0
                # module.set_active_adapter(universal_id)   # ★ ： adapter
                if str(args.get("model_name", "")).lower() == "tuna":
                    module = net.module if hasattr(net, "module") else net
                    bb = getattr(module, "backbone", None)
                    if bb is not None and hasattr(bb, "adapter_list"):
                        #  fused adapter id（eval  > len(adapter_list)  merged_adapter）
                        fused_id = (len(bb.adapter_list) + 1) if (len(bb.adapter_list) > 0 and getattr(bb, "merged_adapter", None) is not None) else len(bb.adapter_list)
                        if hasattr(module, "set_active_adapter"):
                            module.set_active_adapter(fused_id)
                        else:
                            setattr(bb, "active_adapter_id", int(fused_id))
                    
                    head_final = fit_linear_probe_softmax_head(
                        net, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Final]",
                    )
                    acc_final = evaluate_linear_probe_softmax_with_head(
                        head_final, net, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )
                            
                else:
                    head_final = fit_linear_probe_softmax_head(
                        net, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Final]",
                    )
                    acc_final = evaluate_linear_probe_softmax_with_head(
                        head_final, net, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )

                if str(args.get("model_name", "")).lower() != "tuna":
                    base_view = _FeatureView(getattr(net, "backbone", net), which="base")
                    base_view.eval()

                    head_base = fit_linear_probe_softmax_head(
                        base_view, train_loader_all,
                        class_offset=start_seen, num_classes=num_classes, device=model._device,
                        epochs=int(args.get("probe_fit_epochs", 50)),
                        lr=float(args.get("probe_fit_lr", 5e-3)),
                        weight_decay=float(args.get("probe_fit_wd", 0.0)),
                        batch_size=int(args.get("probe_fit_train_batch_size", 128)),
                        max_train_batches=args.get("probe_train_max_batches", None),
                        monitor_loader=test_loader_all,
                        monitor_max_batches=args.get("probe_test_max_batches", None),
                        log_interval=probe_log_interval,
                        eval_interval=probe_eval_interval,
                        log_prefix="[LP-Softmax][Base]",
                    )
                    acc_base = evaluate_linear_probe_softmax_with_head(
                        head_base, base_view, test_loader_all,
                        class_offset=start_seen, device=model._device,
                        max_test_batches=args.get("probe_test_max_batches", None)
                    )
                else:
                    acc_base = 0.0


                logging.info("[LP-JointSeen@ALL] final-model Acc=%.2f | base-model Acc=%.2f", acc_final, acc_base)
                write_step_metrics(
                    json_path=metrics_path,
                    section="probe_softmax_joint_seen_all",
                    step=task,
                    metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    matrix=None,
                    json_safe=json_safe,
                )
                write_final_metrics(
                    metrics_path, "probe_softmax_joint_seen_all",
                    final_metrics={"final_model": float(acc_final), "base_model": float(acc_base)},
                    final_matrix=None,
                    json_safe=json_safe,
                )

        finally:
            # 1)  Module  CPU ，
            for _m in ["head_final", "head_base", "base_view"]:
                if _m in locals() and locals()[_m] is not None:
                    try:
                        mod = locals()[_m]
                        if isinstance(mod, torch.nn.Module):
                            with torch.inference_mode():
                                mod.to("cpu")
                            for p in mod.parameters():
                                p.grad = None
                    except Exception:
                        pass
            head_final = None
            head_base = None
            base_view = None

            # 2)  DataLoader/Dataset （worker  persistent_workers=False ）
            train_loader_all = None
            test_loader_all = None
            train_dataset_all = None
            test_dataset_all = None

            # 3) /
            if 'net' in locals() and was_training:
                    net.train(True)
            

            # 4)  GC + CUDA 
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
 


        

        do_flat_eval = bool(args.get("flat_eval", False)) and (
            flat_eval_tasks is None or task in flat_eval_tasks
        )
        do_feature_eval = bool(args.get("feature_flat_eval", False)) and (
            feature_eval_tasks is None or task in feature_eval_tasks
        )
        # Evaluate flatness/feature metrics per task (optional)
        if do_flat_eval or do_feature_eval:

            # ---- shared temp handles (visible to finally) ----
            train_loader = getattr(model, "train_loader", None)
            flat_loader = None
            seen_test_loader = None
            dataset = None
            flat_cfg = None
            feature_cfg = None
            flat_metrics = None
            feature_metrics = None
            

            # unwrap to raw nn.Module and remember training flag
            net = getattr(model, "_network", model)
            if hasattr(net, "module"):
                net = net.module
            was_training = net.training

            # paths / tags for saving (as in your original code)
            feature_dir = os.path.join(log_dir, "feature_flatness")
            os.makedirs(feature_dir, exist_ok=True)
            base_stub = os.path.basename(logfilename)
            step_tag = f"t{task:02d}"
            save_prefix = f"{base_stub}_{step_tag}"
            is_main = bool(getattr(model, "_is_main_process", True))

            start_seen = class_ranges[task][0]
            end_seen = class_ranges[task][1]
            num_classes =  end_seen - start_seen 
            data_source = str(args.get("flat_eval_data_source", "train")).lower()
            data_mode = str(args.get("flat_eval_data_mode", "train")).lower()
            flat_loader_shuffle = bool(args.get("flat_eval_loader_shuffle", True))

            try:
                # ----------------- build the shared flat_loader once -----------------
                if train_loader is not None:
                    if data_source not in {"train", "test"}:
                        logging.warning(
                            "[FlatEval] Unknown flat_eval_data_source=%s, fallback to test", data_source
                        )
                        # data_source = "test"
                    if data_mode not in {"train", "test", "flip"}:
                        logging.warning(
                            "[FlatEval] Unknown flat_eval_data_mode=%s, fallback to train", data_mode
                        )
                        data_mode = "train"
                    dataset_seen = data_manager.get_dataset(
                        np.arange(start_seen, end_seen),
                        source=data_source,
                        mode=data_mode,
                    )
                    loader_seen = DataLoader(
                        dataset_seen, 
                        batch_size=args.get("flat_eval_batch_size", 32), 
                        shuffle=flat_loader_shuffle, num_workers=0)


                    flat_loader = fractional_loader(
                        loader=loader_seen,
                        fraction=args.get("flat_eval_dataset_fraction", 0.1),
                        seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
                        balanced=True,
                        batch_size=args.get("flat_eval_batch_size", 32),
                        shuffle=flat_loader_shuffle,
                    )

                # ================= weight-space flatness =================
                if do_flat_eval and (flat_loader is not None):
                    net.eval()  # switch to eval for metric extraction

                    flat_cfg = FlatnessConfig(
                        args=args,
                        save_metrics_path=os.path.join(log_dir, "flatness"),
                        save_prefix=f"{os.path.basename(logfilename)}_t{task:02d}",
                    )

                    # #  requires_grad=True，
                    _saved_requires = [(p, bool(p.requires_grad)) for _, p in net.named_parameters()]
                    # for _, _p in net.named_parameters():
                    #     if not _p.requires_grad:
                    #         _p.requires_grad_(True)
                    try:
                        # Explicitly pass the parameter list we want to probe (selected by substrings in config)
                        flat_params = _select_params_by_name(
                            net,
                            getattr(flat_cfg, "param_name_substrings", None),
                            include_frozen=bool(flat_cfg.include_frozen_params),
                        )
                        flat_metrics = evaluate_flatness_metrics(
                            net,
                            flat_loader,
                            device=model._device,
                            config=flat_cfg,
                            known_classes=start_seen,
                            params_override=flat_params,
                        )
                    finally:
                        #  requires_grad 
                        for _p, _old in _saved_requires:
                            _p.requires_grad_(_old)

                    logging.info("Flatness metrics (task %d): %s", task, flat_metrics)

                    if bool(args.get("flat_eval_task_conditioned", False)):
                        cond_dir = os.path.join(log_dir, "flatness_task_conditioned")
                        os.makedirs(cond_dir, exist_ok=True)
                        cond_tasks = _resolve_conditioned_task_indices(
                            args.get("flat_eval_conditioned_task_indices", "seen"),
                            task,
                            nb_tasks,
                        )
                        cond_loss_mode = str(args.get("flat_eval_conditioned_loss_mode", "global")).lower()
                        for loss_task in cond_tasks:
                            loss_start, loss_end = class_ranges[loss_task]
                            dataset_loss_task = data_manager.get_dataset(
                                np.arange(loss_start, loss_end),
                                source=data_source,
                                mode=data_mode,
                            )
                            loader_loss_task = DataLoader(
                                dataset_loss_task,
                                batch_size=args.get("flat_eval_batch_size", 32),
                                shuffle=flat_loader_shuffle,
                                num_workers=0,
                            )
                            cond_loader = fractional_loader(
                                loader=loader_loss_task,
                                fraction=args.get("flat_eval_dataset_fraction", 0.1),
                                seed=args.get("flat_eval_dataset_fraction_seed", args.get("seed", 42)),
                                balanced=True,
                                batch_size=args.get("flat_eval_batch_size", 32),
                                shuffle=flat_loader_shuffle,
                            )
                            cond_cfg = FlatnessConfig(
                                args=args,
                                save_metrics_path=cond_dir,
                                save_prefix=(
                                    f"{os.path.basename(logfilename)}"
                                    f"_theta_t{task:02d}_loss_task{loss_task:02d}"
                                ),
                            )
                            saved_requires_cond = [(p, bool(p.requires_grad)) for _, p in net.named_parameters()]
                            try:
                                cond_params = _select_params_by_name(
                                    net,
                                    getattr(cond_cfg, "param_name_substrings", None),
                                    include_frozen=bool(cond_cfg.include_frozen_params),
                                )
                                cond_known_classes = None
                                if cond_loss_mode in {"tail", "task_tail", "current_tail"}:
                                    cond_known_classes = loss_start
                                cond_metrics = evaluate_flatness_metrics(
                                    net,
                                    cond_loader,
                                    device=model._device,
                                    config=cond_cfg,
                                    known_classes=cond_known_classes,
                                    params_override=cond_params,
                                )
                                cond_metrics.update(
                                    {
                                        "model_task": int(task),
                                        "loss_task": int(loss_task),
                                        "loss_task_start": int(loss_start),
                                        "loss_task_end": int(loss_end),
                                        "conditioned_loss_mode": cond_loss_mode,
                                        "conditioned_known_classes": cond_known_classes,
                                        "sharpness_index": f"Sh_{{{loss_task}->theta{task}}}^S",
                                    }
                                )
                                cond_file = os.path.join(
                                    cond_dir,
                                    (
                                        f"{os.path.basename(logfilename)}"
                                        f"_theta_t{task:02d}_loss_task{loss_task:02d}_metrics.json"
                                    ),
                                )
                                with open(cond_file, "w", encoding="utf-8") as fh:
                                    json.dump(json_safe(cond_metrics), fh, ensure_ascii=False, indent=2)
                                logging.info(
                                    "Task-conditioned flatness metrics (theta task %d, loss task %d): %s",
                                    task,
                                    loss_task,
                                    cond_metrics,
                                )
                            finally:
                                for _p, _old in saved_requires_cond:
                                    _p.requires_grad_(_old)

                # ================= feature-space flatness =================
                if do_feature_eval and (flat_loader is not None):
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)

                    feature_cfg = FeatureFlatnessConfig(
                        args=args,
                        save_matrix_path=feature_dir,
                        save_prefix=save_prefix,
                        device_override=device_override,
                        max_examples_per_batch=args.get("feature_flat_max_examples_per_batch", None),
                    )

                    feature_metrics = evaluate_feature_metrics(
                        model._network,
                        flat_loader,                 #  flat_loader
                        config=feature_cfg,
                    )
                    logging.info("Feature flatness metrics (task %d): %s", task, feature_metrics)

                    if do_feature_eval and args.get("attention_probe_eval", False):
                        try:
                            run_attention_probe(
                                model=model,
                                net=net,
                                data_manager=data_manager,
                                args=args,
                                log_dir=log_dir,
                                save_prefix=save_prefix,
                                device_override=device_override,
                            )
                        except Exception:
                            logging.exception("[AttentionProbe] Failed to export probe payload")

                    # ---- atomic write of feature metrics ----
                    if do_feature_eval and is_main:
                        final_path = os.path.join(feature_dir, f"{save_prefix}_feature_flatness.json")
                        tmp_path = final_path + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as fh:
                            json.dump(json_safe(feature_metrics), fh, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, final_path)

            finally:
                # ===== single cleanup block (memory + handles) =====
                try:
                    # restore training state
                    if was_training:
                        net.train(True)
                    # drop big tensors/objects
                    X_seen = None; y_seen = None
                    prot = None; counts = None
                    flat_cfg = None; feature_cfg = None
                    flat_metrics = None; feature_metrics = None
                    # release loaders/datasets (do NOT touch model.train_loader)
                    seen_test_loader = None
                    dataset = None
                    flat_loader = None
                    # zero grads to free storage
                    if hasattr(net, "zero_grad"):
                        net.zero_grad(set_to_none=True)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

       
        # ---- Cache per-task anchors/prototypes for later comparisons ----
        if args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False):
            def _parse_task_list(val):
                if val is None:
                    return None
                if isinstance(val, (list, tuple)):
                    try:
                        return [int(x) for x in val]
                    except Exception:
                        return None
                if isinstance(val, int):
                    return [val]
                if isinstance(val, str):
                    s = val.strip()
                    try:
                        if s.startswith("[") and s.endswith("]"):
                            import json as _json
                            return [int(x) for x in _json.loads(s)]
                    except Exception:
                        return None
                    try:
                        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
                    except Exception:
                        return None
                return None

            # ---- unified feature cache + drift evaluation control ----
            drift_mode = str(
                args.get(
                    "feature_drift_mode",
                    "last_only",
                )
            ).lower()
            if bool(args.get("feature_drift_last_only", False)):
                drift_mode = "last_only"
            if drift_mode not in {"per_task", "last_only", "none"}:
                drift_mode = "per_task"

            cache_tasks = _parse_task_list(args.get("feature_cache_tasks", [0]))
            cache_enabled = bool(args.get("feature_cka_eval", False) or args.get("feature_proto_eval", False))

            if cache_enabled and (cache_tasks is None or task in cache_tasks):
                try:
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)
                    base_stub = os.path.basename(logfilename)
                    overwrite = bool(args.get("feature_overwrite_first", False)) if task == 0 else False
                    save_task_feature_cache(
                        model._network,
                        data_manager,
                        class_ranges,
                        task_idx=task,
                        log_dir=log_dir,
                        base_stub=base_stub,
                        args=args,
                        device=device_override or model._device,
                        overwrite=overwrite,
                    )
                except Exception:
                    logging.exception("[FeatureFlat] Saving per-task cache failed")

            if drift_mode == "per_task" and task >= 1 and bool(args.get("feature_drift_eval", False)):
                try:
                    from evaluation_feature.feature_drift import compute_first_vs_current_feature_drift

                    compute_first_vs_current_feature_drift(
                        model=model,
                        data_manager=data_manager,
                        class_ranges=class_ranges,
                        args=args,
                        task=task,
                        log_dir=log_dir,
                        logfilename=logfilename,
                    )
                except Exception:
                    logging.exception("[FIRST-vs-CUR] feature drift failed")

            if drift_mode == "last_only" and task == nb_tasks - 1:
                try:
                    device_override = getattr(model, "_device", None)
                    if isinstance(device_override, str):
                        device_override = torch.device(device_override)
                    base_stub = os.path.basename(logfilename)

                    # Ensure last cache exists
                    save_task_feature_cache(
                        model._network,
                        data_manager,
                        class_ranges,
                        task_idx=task,
                        log_dir=log_dir,
                        base_stub=base_stub,
                        args=args,
                        device=device_override or model._device,
                        overwrite=bool(args.get("feature_overwrite_last", False)),
                    )

                    # Decide which refs to compare
                    ref_tasks = cache_tasks if cache_tasks is not None else [0]
                    for ref in ref_tasks:
                        if ref is None:
                            continue
                        try:
                            ref = int(ref)
                        except Exception:
                            continue
                        if ref < 0 or ref >= nb_tasks:
                            continue
                        if ref == task:
                            continue
                        save_task_feature_cache(
                            model._network,
                            data_manager,
                            class_ranges,
                            task_idx=ref,
                            log_dir=log_dir,
                            base_stub=base_stub,
                            args=args,
                            device=device_override or model._device,
                            overwrite=bool(args.get("feature_overwrite_first", False)) if ref == 0 else False,
                        )
                        compare_task_last_features(
                            model._network,
                            data_manager,
                            class_ranges,
                            ref_task_idx=ref,
                            last_task_idx=task,
                            log_dir=log_dir,
                            base_stub=base_stub,
                            args=args,
                            device=model._device,
                        )
                except Exception:
                    logging.exception("[FIRST-vs-LAST] feature comparison failed")


        # ===================== Last-step feature comparison handled above =====================
    
    # 4.5) OOD evaluation on Tiny-ImageNet-* (all classes)
    #      Placed after CL training loop and before final summaries.
    try:
        if False:
            run_ood_evaluation(model, args, metrics_path=metrics_path)
    except Exception as _ood_exc:
        logging.exception("[OOD] Evaluation failed: %s", _ood_exc)

    # 5) Summaries
    # ---------------------------
    # Save the final trained model once (backbone + head)
    try:
        if bool(args.get("save_final_model", False)):
            net_to_save = getattr(model, "_network", None)
            if net_to_save is not None:
                if hasattr(net_to_save, "module"):
                    net_to_save = net_to_save.module
                final_ckpt = {
                    "tasks": int(nb_tasks - 1),
                    "model_state_dict": net_to_save.state_dict(),
                }
                final_path = os.path.join(checkpoint_dir, "final_model.pt")
                torch.save(final_ckpt, final_path)
                logging.info("Saved final model to %s", final_path)
    except Exception as _save_exc:
        logging.exception("[FinalSave] Failed to save final model: %s", _save_exc)

    #  2) CNN / NME 
    #  CNN / NME 
    if True:
        T = task + 1
        run_dir = log_dir
        run_stub = os.path.basename(logfilename)

        if len(cnn_matrix) > 0:
            cnn_time_by_task = assemble_eval_matrix(cnn_matrix, T, orientation="time_by_task")
            # cnn_task_by_time = cnn_time_by_task.T
            log_eval_matrix(cnn_time_by_task, "CNN Evaluation", orientation="time_by_task")
            # log_matrix(cnn_task_by_time, "CNN Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(cnn_time_by_task, axis=0) - cnn_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (CNN): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_eval_matrix(cnn_time_by_task, run_dir, run_stub, "cnn_R_time_by_task")
                # save_matrix(cnn_task_by_time, run_dir, run_stub, "cnn_R_task_by_time")

            #  JSON（）
            write_final_metrics(
                metrics_path, "cnn",
                final_metrics=compute_sequence_metrics(cnn_time_by_task),
                final_matrix=cnn_time_by_task,
                json_safe=json_safe,
            )

        if len(nme_matrix) > 0:
            nme_time_by_task = assemble_eval_matrix(nme_matrix, T, orientation="time_by_task")
            # nme_task_by_time = nme_time_by_task.T
            log_eval_matrix(nme_time_by_task, "NME Evaluation", orientation="time_by_task")
            # log_matrix(nme_task_by_time, "NME Evaluation", orientation="task_by_time")
            forgetting = np.nanmean((np.nanmax(nme_time_by_task, axis=0) - nme_time_by_task[T-1, :])[:T-1])
            logging.info('Forgetting (NME): %s', forgetting)
            if args.get("save_legacy_artifacts", False):
                save_eval_matrix(nme_time_by_task, run_dir, run_stub, "nme_R_time_by_task")
                # save_matrix(nme_task_by_time, run_dir, run_stub, "nme_R_task_by_time")

            #  JSON（）
            write_final_metrics(
                metrics_path, "nme",
                final_metrics=compute_sequence_metrics(nme_time_by_task),
                final_matrix=nme_time_by_task,
                json_safe=json_safe,
            )

        if mechanism_curve:
            write_final_metrics(
                metrics_path,
                "mechanism",
                final_metrics={
                    "tasks": mechanism_curve,
                    "last": mechanism_curve[-1],
                },
                final_matrix=None,
                json_safe=json_safe,
            )

        if epoch_dynamics_curve:
            write_final_metrics(
                metrics_path,
                "epoch_dynamics",
                final_metrics={
                    "epochs": epoch_dynamics_curve,
                    "last": epoch_dynamics_curve[-1],
                },
                final_matrix=None,
                json_safe=json_safe,
            )
