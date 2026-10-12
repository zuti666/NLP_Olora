#!/usr/bin/env python3
"""Fixed post-training P/C/CD location audit: 18 state-batches, 108 CE gradients.

The three final trajectories provide separate reference states. At EACH state,
all three locations share g0, g+, g-, the augmented batch and forward RNG R0.
This module constructs no optimizer, performs no SGD/HVP and never saves models.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audit_core import rng_equal, rng_get, rng_set, seed_all
from effective_step import EffectiveStep
from response_hvp import backend_context

TASKS = (0, 4, 9)
BRANCHES = ("P", "C", "CD")
RUNS = ("G2-CD01_O_P_natural_s2", "G2-CD02_O_C_natural_s2", "G2-CD03_O_CD_natural_s2")
TASK_SAMPLES = (1259, 997, 993, 596, 426, 260, 357, 317, 398, 378)
EPS = 1e-12


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def vector_sha(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def model_sha(model):
    # Same ordering and serialization as native_lifecycle/base.parameter_hash.
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_sha(state):
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state[:2], protocol=4))
    digest.update(state[2].cpu().numpy().tobytes())
    for value in state[3]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def norm(values):
    squared = sum(float(v.detach().double().square().sum()) for v in values)
    if not math.isfinite(squared) or squared < 0:
        raise FloatingPointError("Nonfinite gradient reduction")
    return math.sqrt(squared)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def normalized(values, eps=EPS):
    return tuple(v / (norm(values) + eps) for v in values)


def cosine(a, b):
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return None
    dot = sum(float((x.detach().double() * y.detach().double()).sum()) for x, y in zip(a, b))
    if not math.isfinite(dot):
        raise FloatingPointError("Nonfinite gradient cosine")
    return dot / denominator


def comparison(a, b):
    delta = sub(a, b)
    na, nb = norm(a), norm(b)
    return {"first_norm": na, "reference_norm": nb, "delta_l2": norm(delta),
            "relative_l2": norm(delta) / nb if nb > 0 else None,
            "cosine": cosine(a, b), "norm_ratio": na / nb if nb > 0 else None,
            "max_abs_delta": max(float(v.abs().max()) for v in delta)}


def _equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _control(name):
    return ("cache" in name.lower() or name.startswith("delta_w_")
            or name.startswith("external_delta_w_")
            or name in ("_register_delta_hook", "backup_momentum", "momentum", "fused_attn"))


class _ObjectState:
    """Restore cache tensors and containers in place, including their identity."""
    def __init__(self, value):
        self.original = value
        if torch.is_tensor(value):
            self.value = value.detach().clone()
        elif isinstance(value, dict):
            self.value = {k: _ObjectState(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            self.value = [_ObjectState(v) for v in value]
        else:
            self.value = copy.deepcopy(value)

    @torch.no_grad()
    def restore(self):
        obj = self.original
        if torch.is_tensor(obj):
            if not torch.equal(obj, self.value):
                obj.copy_(self.value)
        elif isinstance(obj, dict):
            obj.clear()
            obj.update({k: child.restore() for k, child in self.value.items()})
        elif isinstance(obj, list):
            obj[:] = [child.restore() for child in self.value]
        elif isinstance(obj, tuple):
            for child in self.value:
                child.restore()
        return obj

    def exact(self, value):
        if value is not self.original:
            return False
        if torch.is_tensor(value):
            return torch.equal(value, self.value)
        if isinstance(value, dict):
            return value.keys() == self.value.keys() and all(child.exact(value[k]) for k, child in self.value.items())
        if isinstance(value, (list, tuple)):
            return len(value) == len(self.value) and all(child.exact(v) for child, v in zip(self.value, value))
        return _equal(value, self.value)


class FullState:
    """Full parameter/buffer, existing gradient object, mode, cache and RNG txn.

    Unlike the training transaction, this rare offline transaction snapshots
    frozen values as well. Optional optimizer/scheduler state is supported for
    CPU validation; production diagnostics never instantiate either object.
    """
    def __init__(self, model, optimizer=None, scheduler=None):
        self.model = model
        self.parameters = [(p, p.detach().clone(), p.requires_grad, p.grad,
                            None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]
        self.modes = [(module, module.training) for module in model.modules()]
        self.controls = [(module, {k: _ObjectState(v) for k, v in module.__dict__.items() if _control(k)})
                         for module in model.modules()]
        self.rng = rng_get()
        self.optimizer, self.scheduler = optimizer, scheduler
        self.optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        self.scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None

    @torch.no_grad()
    def restore(self):
        for p, value, requires_grad, grad_object, grad_value in self.parameters:
            if not torch.equal(p, value):
                p.copy_(value)
            p.requires_grad_(requires_grad)
            p.grad = grad_object
            if grad_object is not None and not torch.equal(grad_object, grad_value):
                grad_object.copy_(grad_value)
        for buffer, value in self.buffers:
            if not torch.equal(buffer, value):
                buffer.copy_(value)
        for module, mode in self.modes:
            module.training = mode
        for module, attrs in self.controls:
            for key in list(module.__dict__):
                if _control(key) and key not in attrs:
                    delattr(module, key)
            for key, value in attrs.items():
                setattr(module, key, value.restore())
        if self.optimizer is not None and not _equal(self.optimizer.state_dict(), self.optimizer_state):
            self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        if self.scheduler is not None and not _equal(self.scheduler.state_dict(), self.scheduler_state):
            self.scheduler.load_state_dict(copy.deepcopy(self.scheduler_state))
        rng_set(self.rng)

    def check(self):
        checks = {
            "all_parameter_values_exact": all(torch.equal(p, value) for p, value, *_ in self.parameters),
            "parameter_identities_exact": tuple(id(p) for p in self.model.parameters()) == tuple(id(p) for p, *_ in self.parameters),
            "requires_grad_exact": all(p.requires_grad == flag for p, _, flag, *_ in self.parameters),
            "gradient_values_and_identities_exact": all(p.grad is obj and (obj is None or torch.equal(obj, value))
                                                       for p, _, _, obj, value in self.parameters),
            "buffer_values_and_identities_exact": (tuple(id(b) for b in self.model.buffers()) == tuple(id(b) for b, _ in self.buffers)
                                                    and all(torch.equal(b, value) for b, value in self.buffers)),
            "module_modes_exact": all(module.training == mode for module, mode in self.modes),
            "cache_controls_values_and_identities_exact": all(
                {k for k in module.__dict__ if _control(k)} == set(attrs)
                and all(value.exact(getattr(module, k)) for k, value in attrs.items()) for module, attrs in self.controls),
            "optimizer_state_exact": self.optimizer is None or _equal(self.optimizer.state_dict(), self.optimizer_state),
            "scheduler_state_exact": self.scheduler is None or _equal(self.scheduler.state_dict(), self.scheduler_state),
            "rng_exact": rng_equal(self.rng, rng_get()),
        }
        if not all(checks.values()):
            raise AssertionError("Offline diagnostic failed entry restoration: " + repr(checks))
        return checks


@contextmanager
def deterministic_precision():
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:4]
        torch.use_deterministic_algorithms(previous[4], warn_only=previous[5])


def audit_batch(model, inputs, targets, class_start, class_end, geometry=None,
                *, rho=.05, radius=.25, eps=EPS, optimizer=None, scheduler=None,
                closure=None, allow_float64_fixture=False, return_context=False):
    """Six ordinary gradients, sharing g0/g+/g- and restoring entry exactly.

    A custom scalar-loss closure and float64 are allowed ONLY for the isolated
    CPU fixture. CLI always uses the native current-task cross-entropy slice.
    """
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if not params or any(p.dtype != (torch.float64 if allow_float64_fixture else torch.float32) for p in model.parameters()):
        raise ValueError("Expected all model parameters FP32 (or explicit float64 CPU fixture)")
    if allow_float64_fixture and any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Float64 mathematical fixture is CPU only")
    geometry = geometry or EffectiveStep.from_model(model)
    if {id(p) for p in geometry.params} != {id(p) for p in params}:
        raise ValueError("Geometry does not cover every trainable parameter")
    if not all(math.isfinite(v) for v in (rho, radius, eps)) or rho < 0 or radius < 0 or eps <= 0:
        raise ValueError("Invalid probe radii/epsilon")
    for module in model.modules():
        if getattr(module, "_register_delta_hook", False) or getattr(module, "external_delta_w_q", None) is not None or getattr(module, "external_delta_w_v", None) is not None:
            raise ValueError("External deltas/hooks are forbidden")
    live = FullState(model, optimizer, scheduler)
    values = {id(p): value for p, value, *_ in live.parameters}
    base = tuple(values[id(p)] for p in params)
    ordering = {id(p): i for i, p in enumerate(params)}
    def geo(point):
        return tuple(point[ordering[id(p)]] for p in geometry.params)
    if closure is None:
        def closure():
            output = model(inputs)
            logits = output["logits"] if isinstance(output, dict) else output
            if (logits.ndim != 2 or logits.shape[1] != class_end or targets.ndim != 1
                    or logits.shape[0] != targets.numel() or bool(((targets < class_start) | (targets >= class_end)).any())):
                raise ValueError("Current-task CE class slice mismatch")
            return F.cross_entropy(logits[:, class_start:class_end], targets - class_start), logits
    calls = 0
    probe_records = {}
    post_rng = None
    report = {}
    started = time.perf_counter()
    def evaluate(label, point):
        nonlocal calls, post_rng
        live.restore()
        model.train()
        with torch.no_grad():
            for p, value in zip(params, point):
                p.copy_(value)
        if not rng_equal(live.rng, rng_get()):
            raise AssertionError("Probe did not replay exact R0")
        with torch.enable_grad(), backend_context(model, "math"), torch.autocast(device_type=params[0].device.type, enabled=False):
            loss, logits = closure()
            if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite ordinary CE loss/logits")
            gradients = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=False)
            calls += 1
            gradients = tuple(g.detach().clone() for g in gradients)
            if any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError("Nonfinite ordinary CE gradient")
            flags = {"flash": torch.backends.cuda.flash_sdp_enabled(), "math": torch.backends.cuda.math_sdp_enabled(),
                     "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                     "fused_attn_true_modules": sum(bool(m.fused_attn) for m in model.modules() if hasattr(m, "fused_attn"))}
            if flags != {"flash": False, "math": True, "mem_efficient": False, "cudnn": False, "fused_attn_true_modules": 0}:
                raise AssertionError("Ordinary CE probe did not use strict math attention")
        after = rng_get()
        if post_rng is None:
            post_rng = after
        elif not rng_equal(post_rng, after):
            raise AssertionError("Identical forward function consumed different RNG across probe points")
        probe_records[label] = {"loss": float(loss.detach()), "logits_sha256": tensor_sha(logits),
                                "gradient_sha256": vector_sha(gradients), "gradient_norm": norm(gradients),
                                "point_sha256": vector_sha(point), "R0_sha256": rng_sha(live.rng),
                                "R1_sha256": rng_sha(after), "attention_backend_flags": flags}
        return gradients
    try:
        with deterministic_precision():
            g0 = evaluate("g0", base)
            u0 = normalized(g0, eps)
            plus = tuple(p + rho*u for p, u in zip(base, u0))
            minus = tuple(p - rho*u for p, u in zip(base, u0))
            gp = evaluate("g_plus", plus)
            gm = evaluate("g_minus", minus)
            qP = sub(gp, g0)
            # Preserve the existing C operation order: -(g_minus-g0).
            qC = tuple(-value for value in sub(gm, g0))
            s = tuple(.5*(p-m) for p, m in zip(gp, gm))
            e = tuple(.5*(p+m)-g for p, m, g in zip(gp, gm, g0))
            qs = {"P": qP, "C": qC, "CD": s}
            points, g2s, locations = {}, {}, {}
            for branch in BRANCHES:
                q = qs[branch]
                point = tuple(p + radius*u for p, u in zip(base, normalized(q, eps)))
                points[branch] = point
                g2s[branch] = evaluate("g2_" + branch, point)
                ng2, ng0 = norm(g2s[branch]), norm(g0)
                delta = sub(g2s[branch], g0)
                count = sum(value.numel() for value in delta)
                positive = sum(int((value > 0).sum()) for value in delta)
                negative = sum(int((value < 0).sum()) for value in delta)
                locations[branch] = {
                    "q_norm": norm(q), "q_sha256": vector_sha(q), "q_zero": norm(q) == 0,
                    "outer_formula": "theta0 + r * q / (norm(q) + epsilon)",
                    "predicted_parameter_distance": radius*norm(q)/(norm(q)+eps),
                    "actual_outer_displacement": geometry.actual_delta(geo(base), geo(point)),
                    "g2_norm": ng2, "g2_norm_minus_g0_norm": ng2-ng0,
                    "cos_g2_g0": cosine(g2s[branch], g0), "g2_minus_g0": comparison(g2s[branch], g0),
                    "coordinate_delta_signs": {"positive_fraction": positive/count, "negative_fraction": negative/count,
                                               "zero_fraction": (count-positive-negative)/count,
                                               "positive_l1": sum(float(value[value > 0].double().sum()) for value in delta),
                                               "negative_abs_l1": sum(float(-value[value < 0].double().sum()) for value in delta)},
                }
            contrasts = {}
            for a, b in (("P", "C"), ("CD", "P"), ("CD", "C")):
                contrasts[a + "_vs_" + b] = {
                    "q": comparison(qs[a], qs[b]), "g2": comparison(g2s[a], g2s[b]),
                    "outer_point_delta": geometry.actual_delta(geo(points[b]), geo(points[a])),
                }
            threshold = 10*torch.finfo(params[0].dtype).eps*max(norm(g0), norm(gp), norm(gm))
            report = {
                "status": "passed", "ordinary_gradient_calls": calls, "sgd_updates": 0, "optimizer_updates": 0,
                "hvp_calls": 0, "create_graph_gradient_calls": 0, "fisher_calls": 0, "weight_merge_calls": 0, "projection_calls": 0,
                "backend": "math", "model_mode_during_probes": "train", "class_slice": [class_start, class_end],
                "rho": rho, "radius": radius, "eps": eps, "reduction_dtype": "float64",
                "formulas": {"qP": "g_plus-g0", "qC": "g0-g_minus", "qCD": "0.5*(g_plus-g_minus)",
                             "s": "0.5*(g_plus-g_minus)", "e": "0.5*(g_plus+g_minus)-g0"},
                "norms": {"g0": norm(g0), "g_plus": norm(gp), "g_minus": norm(gm), "s": norm(s), "e": norm(e)},
                "e_over_s_plus_eps": norm(e)/(norm(s)+eps), "low_s": norm(s) <= threshold,
                "low_s_threshold": threshold,
                "low_s_interpretation": "predeclared floating-point cancellation reference; does not change any direction",
                "algebra_reconstruction_fp_residual": {
                    "qP_minus_s_plus_e_l2": norm(sub(qP, tuple(a+b for a, b in zip(s, e)))),
                    "qC_minus_s_minus_e_l2": norm(sub(qC, sub(s, e))),
                },
                "q_cosines": {"qP_qC": cosine(qP, qC), "qCD_qP": cosine(s, qP), "qCD_qC": cosine(s, qC)},
                "first_probe_displacements": {"plus": geometry.actual_delta(geo(base), geo(plus)),
                                               "minus": geometry.actual_delta(geo(base), geo(minus))},
                "probes": probe_records, "locations": locations, "same_state_contrasts": contrasts,
                "base_trainable_sha256": vector_sha(base), "R0_sha256": rng_sha(live.rng), "R1_sha256": rng_sha(post_rng),
                "shared_first_three_gradients": True, "all_six_probes_share_R0": True,
                "zero_vector_angle": None, "seconds": time.perf_counter()-started,
            }
            if calls != 6:
                raise AssertionError("Fixed offline state-batch must use exactly six ordinary gradients")
            if return_context:
                report["_context"] = {"base": base, "g0": g0, "gp": gp, "gm": gm, "q": qs, "s": s, "e": e,
                                      "outer": points, "g2": g2s}
        return report
    finally:
        live.restore()
        restored = live.check()
        if report:
            report["entry_state_restore"] = restored
            report["entry_state_restored"] = True


def validate_config(raw, branch):
    expected = {"dataset": "imageneta", "class_shuffle": False, "init_cls": 20, "increment": 20,
                "lora_rank": 10, "probe_branch": branch, "update_direction": "g2", "step_mode": "natural",
                "attention_backend": "math", "gam_grad_rho": .05, "gam_grad_norm_rho": .25,
                "lambda_flat": 1., "ewc_lambda": 0., "ewc_skip_task_fisher": True,
                "in_task_weight_merge": False, "task_endpoint_weight_merge": False,
                "approximate_fisher_in_task": False, "compute_full_fisher": False}
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError("Formal frozen protocol mismatch: " + key + "=" + repr(raw.get(key)))
    if raw.get("seed") not in (2, [2]):
        raise ValueError("Formal protocol requires seed2")


def checkpoint_inputs(run_dir):
    """Pin every input artifact; final checkpoint selection has no score search."""
    pins, configs, manifests = {}, {}, {}
    for run_id, branch in zip(RUNS, BRANCHES):
        reference = run_dir/"outputs"/run_id
        config_path, summary_path = reference/"effective_config.json", reference/"final_summary.json"
        config = json.loads(config_path.read_text())
        validate_config(config, branch)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "completed" or summary.get("smoke") is not False or summary.get("completed_tasks") != 10:
            raise ValueError("Offline diagnostics require all three completed 10-task formal runs")
        if summary.get("completed_steps") != 2550:
            raise ValueError("Formal training update budget differs from approved plan")
        if summary.get("completed_epochs") != 500 or summary.get("effective_config_sha256") != sha(config_path):
            raise AssertionError("Completed formal summary does not cover 500 epochs/current effective config")
        configs[run_id] = config
        files = {"effective_config.json": sha(config_path), "final_summary.json": sha(summary_path)}
        for name in ("task_identities.json", "task_checkpoints.json", "step_trace.jsonl", "task_matrix.json", "task_invariant_receipts.json"):
            path = reference/name
            if not path.is_file():
                raise FileNotFoundError("Formal paired identity artifact is missing: " + str(path))
            files[name] = sha(path)
        for task in TASKS:
            directory = reference/"checkpoints"/("task_%03d" % task)
            manifest = json.loads((directory/"manifest.json").read_text())
            if (manifest.get("task") != task or manifest.get("epoch") != 50 or manifest.get("seed") != 2
                    or manifest.get("class_range") != [0, 20*(task+1)]
                    or manifest.get("current_task_class_range") != [20*task, 20*(task+1)]
                    or not manifest.get("model_sha256") or not manifest.get("files_sha256")):
                raise ValueError("Fixed final native checkpoint manifest mismatch")
            if (manifest.get("probe_branch") != branch or manifest.get("update_direction") != "g2"
                    or manifest.get("step_mode") != "natural" or manifest.get("attention_backend") != "math"
                    or manifest.get("config_sha256") != sha(config_path)):
                raise AssertionError("Checkpoint is not the declared frozen math/g2/natural branch")
            if not (directory/"training_state.pt").is_file():
                raise FileNotFoundError("Final optimizer/scheduler/RNG archive is missing")
            for name, expected in manifest["files_sha256"].items():
                if sha(directory/name) != expected:
                    raise AssertionError("Native checkpoint sidecar hash mismatch")
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    files[str(path.relative_to(reference))] = sha(path)
            manifests[(run_id, task)] = manifest
        pins[run_id] = files
    return pins, configs, manifests


def validate_formal_pairing(run_dir):
    """Match every formal task and augmented training batch before diagnostics.

    Later whole-parameter initializations differ because trajectories fork;
    task0 origin and every new FC row, frozen origin and data identity must match.
    Accuracy values are outcomes and never a criterion for paired identity.
    """
    paths = [run_dir/"outputs"/run_id for run_id in RUNS]
    for path in paths:
        matrix = json.loads((path/"task_matrix.json").read_text()).get("matrix")
        if (not isinstance(matrix, list) or len(matrix) != 10
                or any(not isinstance(row, list) or len(row) != task+1 for task, row in enumerate(matrix))
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100
                       for row in matrix for value in row)):
            raise AssertionError("Formal final result requires the full ten-task accuracy triangle")
    identities = [json.loads((path/"task_identities.json").read_text()) for path in paths]
    if any(len(rows) != 10 for rows in identities):
        raise AssertionError("Each formal trajectory must have ten task identities")
    common = ("task", "seed", "class_slice", "fc_shape", "old_fc_rows_inherited_exact",
              "new_fc_rows_sha256", "frozen_origin_sha256", "data_seed_rule", "forward_rng_rule")
    frozen_origins = set()
    for task in range(10):
        expected = identities[0][task]
        if (expected.get("task") != task or expected.get("seed") != 2
                or expected.get("class_slice") != [20*task, 20*(task+1)]
                or expected.get("fc_shape", [None])[0] != 20*(task+1)
                or expected.get("old_fc_rows_inherited_exact") is not True
                or not expected.get("new_fc_rows_sha256") or not expected.get("frozen_origin_sha256")):
            raise AssertionError("Invalid formal classifier/task initialization identity")
        frozen_origins.add(expected["frozen_origin_sha256"])
        for rows, branch in zip(identities, BRANCHES):
            observed = rows[task]
            if any(key not in observed or observed[key] != expected.get(key) for key in common):
                raise AssertionError("Formal classifier/data identity mismatch at task " + str(task))
            if (observed.get("probe_branch") != branch or observed.get("update_direction") != "g2"
                    or observed.get("attention_backend") != "math"):
                raise AssertionError("Formal task identity method mismatch")
            if task == 0 and observed.get("task_initial_parameter_sha256") != expected.get("task_initial_parameter_sha256"):
                raise AssertionError("Formal task0 full initialization mismatch")
    if len(frozen_origins) != 1:
        raise AssertionError("Frozen pretrained origin changed between tasks")
    batches_per_epoch = [math.ceil(count/128) for count in TASK_SAMPLES]
    expected_coords = ((task, epoch, batch) for task in range(10) for epoch in range(1, 51)
                       for batch in range(batches_per_epoch[task]))
    handles = [open(path/"step_trace.jsonl") for path in paths]
    count = 0
    try:
        for lines, coord in zip_longest(zip_longest(*handles), expected_coords):
            if coord is None or lines is None or any(line is None for line in lines):
                raise AssertionError("Formal paired trace has an incomplete or excessive batch matrix")
            rows = [json.loads(line) for line in lines]
            expected_identity = rows[0].get("identity")
            if not isinstance(expected_identity, dict) or not all(expected_identity.get(k) for k in ("input_sha256", "target_sha256")):
                raise AssertionError("Formal trace is missing augmented input/target identity")
            task, epoch, batch = coord
            for row, branch in zip(rows, BRANCHES):
                if ((row.get("task"), row.get("epoch"), row.get("batch")) != coord
                        or row.get("step_index") != count or row.get("seed") != 2
                        or row.get("class_slice") != [20*task, 20*(task+1)]
                        or row.get("seen_classes") != 20*(task+1)
                        or row.get("identity") != expected_identity
                        or row.get("probe_branch") != branch or row.get("update_direction") != "g2"
                        or row.get("step_mode") != "natural" or row.get("attention_backend") != "math"):
                    raise AssertionError("Formal paired batch/method matrix mismatch at " + repr(coord))
            count += 1
    finally:
        for handle in handles:
            handle.close()
    if count != 2550:
        raise AssertionError("Formal matrix update budget differs from 2550")
    return {"formal_whole_task_matrix_identity": True, "paired_task_count": 10,
            "paired_batch_count": count, "task0_full_initialization_exact": True,
            "all_new_fc_rows_exact": True, "all_frozen_origins_exact": True,
            "all_augmented_batches_exact": True, "batches_per_epoch": batches_per_epoch}


def validate_input_assets(run_dir, source):
    """Recheck the fixed construction assets instead of trusting old pass text."""
    spec_path, gate_path = run_dir/"environment_spec.json", run_dir/"gates"/"INPUT_ASSETS.json"
    spec, gate = json.loads(spec_path.read_text()), json.loads(gate_path.read_text())
    canonical = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if gate.get("status") != "passed" or gate.get("canonical_spec_sha256") != canonical:
        raise AssertionError("Frozen input asset receipt does not cover current environment specification")
    if Path(gate["source"]["root"]).resolve() != source:
        raise AssertionError("Diagnostic source path differs from frozen asset receipt")
    source_manifest_path = Path(gate["source"]["manifest_path"])
    if sha(source_manifest_path) != gate["source"]["manifest_sha256"]:
        raise AssertionError("Frozen source manifest changed")
    source_manifest = json.loads(source_manifest_path.read_text())
    for name, expected in source_manifest["files"].items():
        if sha(source/name) != expected:
            raise AssertionError("Frozen construction source file changed: " + name)
    pretrained = gate["pretrained"]
    pretrained_path = Path(pretrained["path"])
    if (pretrained.get("verified") is not True or pretrained_path.stat().st_size != pretrained["bytes"]
            or sha(pretrained_path) != pretrained["sha256"]):
        raise AssertionError("Pinned pretrained model asset changed")
    if gate["data"].get("all_expected_hashes_exact") is not True or gate["data"]["asset_hashes"] != spec["data"]["asset_hashes"]:
        raise AssertionError("Fixed local dataset manifest mismatch")
    return {"environment_spec_sha256": sha(spec_path), "input_asset_receipt_sha256": sha(gate_path),
            "source_manifest_sha256": sha(source_manifest_path), "pretrained_sha256": pretrained["sha256"],
            "source_file_hashes": source_manifest["files"], "data": spec["data"]}


def reconstruction_config(raw, device, output_dir):
    cfg = dict(raw)
    cfg.update(seed=2, device=[device], filepath=str(output_dir/"unused_native_state") + os.sep,
               train_num_workers=0, eval_num_workers=0, resume_require_experiment_identity=False,
               resume_require_model_sidecars=False, resume_require_fisher_state=False,
               optimizer_type="gam", posterior_state_save=False, mechanism_eval=False, mechanism_grad_stats=False,
               as1_delta_w_penalty=False, memory_size=0, memory_per_class=0, fisher_split=False,
               ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False, ewc_fisher_damping_alpha=0.,
               ewc_dual_ascent=False, compute_nme_class_means=False)
    return cfg


def dataset_identity(dm, fixed_data=None):
    """Reconstruction identity includes local path/label order and source bytes."""
    identity = {"class_order": [int(v) for v in dm._class_order], "task_increments": [int(v) for v in dm._increments]}
    asset_hashes = {}
    for split in ("train", "test"):
        data, labels = getattr(dm, "_" + split + "_data"), getattr(dm, "_" + split + "_targets")
        entries = []
        path_label, path_label_size = hashlib.sha256(), hashlib.sha256()
        for value, label in zip(data, labels):
            path = Path(str(value))
            if not path.is_file():
                raise FileNotFoundError("Local ImageNet-A source sample is missing: " + str(path))
            entries.append({"path": str(path.resolve()), "label": int(label), "sha256": sha(path)})
            if fixed_data is not None:
                relative = str(path.resolve().relative_to(Path(fixed_data["root"])/"imagenet-a"))
                path_label.update((relative+"\t"+str(int(label))+"\n").encode())
                path_label_size.update((relative+"\t"+str(int(label))+"\t"+str(path.stat().st_size)+"\n").encode())
        identity[split] = entries
        asset_hashes[split+"_path_label"] = path_label.hexdigest()
        asset_hashes[split+"_path_label_size"] = path_label_size.hexdigest()
    if (len(identity["train"]) != 5981 or len(identity["test"]) != 1519
            or identity["class_order"] != list(range(200)) or identity["task_increments"] != [20]*10):
        raise ValueError("Local split/class order differs from frozen plan")
    counts = [sum(20*t <= e["label"] < 20*(t+1) for e in identity["train"]) for t in range(10)]
    if counts != list(TASK_SAMPLES):
        raise ValueError("Local per-task sample counts differ from frozen plan")
    identity["train_samples_by_task"] = counts
    if fixed_data is not None:
        if asset_hashes != fixed_data["asset_hashes"]:
            raise AssertionError("Reconstructed dataset differs from training's fixed path/label/size manifest")
        identity["fixed_training_data_asset_hashes"] = asset_hashes
    return identity


def register_batches(dm):
    entry_rng = rng_get()
    batches, registry = {}, {"schema_version": 1, "registered_before_gradient_diagnostics": True,
                            "data_seed_rule": "seed*100000+81000+task",
                            "forward_seed_rule": "seed*100000+82000+task*10+batch", "batch_size": 128, "tasks": {}}
    try:
        for task in TASKS:
            seed = 2*100000 + 81000 + task
            seed_all(seed)
            start, end = dm.get_task_class_range(task)
            dataset = dm.get_dataset(np.arange(start, end), source="train", mode="train")
            if len(dataset) != TASK_SAMPLES[task]:
                raise ValueError("Fixed diagnostic task sample count differs from training protocol")
            loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed), drop_last=False)
            iterator, entries = iter(loader), []
            batches[task] = []
            for batch in range(2):
                ids, x, y = next(iterator)
                identity = {"batch": batch, "samples": int(y.numel()), "indices": ids.tolist(),
                            "input_sha256": tensor_sha(x), "target_sha256": tensor_sha(y),
                            "indices_sha256": tensor_sha(ids), "data_seed": seed,
                            "forward_seed": 2*100000 + 82000 + task*10 + batch}
                batches[task].append((ids.detach().clone(), x.detach().clone(), y.detach().clone(), identity))
                entries.append(identity)
            registry["tasks"][str(task)] = {"class_slice": [start, end], "batches": entries}
    finally:
        rng_set(entry_rng)
    return batches, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "run-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source, run_dir, out = (Path(value).resolve() for value in (args.source, args.run_dir, args.output_dir))
    target = out/"g2_location_report.json"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Offline output directory must be fresh")
    input_pins, configs, manifests = checkpoint_inputs(run_dir)
    paired_identity = validate_formal_pairing(run_dir)
    construction_assets = validate_input_assets(run_dir, source)
    source_pins = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))}
    code_dir = Path(__file__).resolve().parent
    code_pins = {name: sha(code_dir/name) for name in ("g2_location_diagnostics.py", "audit_core.py", "effective_step.py", "response_hvp.py")}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    entry_rng, entry_cwd = rng_get(), Path.cwd()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sys.path.insert(0, str(source))
    os.chdir(source)
    report = {"schema_version": 1, "status": "preparing", "source": str(source), "run_dir": str(run_dir),
              "seed": 2, "tasks": list(TASKS), "runs": list(RUNS), "checkpoint_selection": "fixed task0/4/9 final epoch50",
              "state_results": [], "expected_state_batch_count": 18, "expected_ordinary_gradient_calls": 108,
              "ordinary_gradient_calls": 0, "optimizer_updates": 0, "sgd_updates": 0, "hvp_calls": 0,
              "create_graph_gradient_calls": 0, "fisher_calls": 0, "weight_merge_calls": 0, "projection_calls": 0,
              "input_assets_sha256": input_pins, "source_python_sha256": source_pins, "diagnostic_code_sha256": code_pins,
              "construction_assets": construction_assets, "formal_pairing": paired_identity,
              "formal_whole_task_matrix_identity": paired_identity["formal_whole_task_matrix_identity"],
              "scope": "all three g2 locations compared at each single frozen trajectory checkpoint; no cross-trajectory gradient subtraction",
              "precision": {"dtype": "float32", "autocast": False, "TF32": False, "attention": "math", "strict_deterministic": True},
              "optimizer_scheduler": "never instantiated; archived training_state.pt pinned and kept read-only"}
    write_json(target, report)
    try:
        from utils.data_manager import DataManager
        from utils import factory
        with deterministic_precision():
            seed_all(2)
            cfg = reconstruction_config(configs[RUNS[0]], device, out)
            dm = DataManager(cfg["dataset"], False, 2, 20, 20, cfg)
            cfg.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
            data_manifest = dataset_identity(dm, construction_assets["data"])
            write_json(out/"diagnostic_data_manifest.json", data_manifest)
            batches, registry = register_batches(dm)
            write_json(out/"batch_registry.json", registry)
            report.update(status="running", batch_registry_sha256=sha(out/"batch_registry.json"),
                          diagnostic_data_manifest_sha256=sha(out/"diagnostic_data_manifest.json"))
            write_json(target, report)
            for run_id in RUNS:
                for task in TASKS:
                    seed_all(2)
                    native_cfg = reconstruction_config(configs[run_id], device, out)
                    native_cfg.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
                    learner = factory.get_model(native_cfg["model_name"], native_cfg)
                    learner._unwrap_network().backbone.to(device)
                    checkpoint = run_dir/"outputs"/run_id/"checkpoints"/("task_%03d" % task)
                    learner.load_task_checkpoint(str(checkpoint), task, dm)
                    model = learner._unwrap_network().to(device)
                    expected_hash = manifests[(run_id, task)]["model_sha256"]
                    if model_sha(model) != expected_hash:
                        raise AssertionError("Native final checkpoint reconstruction hash mismatch")
                    geometry = EffectiveStep.from_model(model)
                    state_result = {"run_id": run_id, "task": task, "checkpoint": str(checkpoint),
                                    "checkpoint_model_sha256": expected_hash,
                                    "checkpoint_manifest_sha256": sha(checkpoint/"manifest.json"), "batches": []}
                    report["state_results"].append(state_result)
                    start, end = dm.get_task_class_range(task)
                    for ids, x, y, identity in batches[task]:
                        if any(tensor_sha(value) != identity[key] for value, key in ((ids, "indices_sha256"), (x, "input_sha256"), (y, "target_sha256"))):
                            raise AssertionError("Registered augmented batch changed")
                        batch_rng = rng_get()
                        try:
                            seed_all(identity["forward_seed"])
                            diagnostics = audit_batch(model, x.to(device), y.to(device), start, end, geometry)
                        finally:
                            rng_set(batch_rng)
                        if model_sha(model) != expected_hash:
                            raise AssertionError("Offline probes changed final checkpoint model")
                        state_result["batches"].append({"identity": identity, **diagnostics})
                        report["ordinary_gradient_calls"] += diagnostics["ordinary_gradient_calls"]
                        write_json(target, report)
                        print(json.dumps({"event": "g2_location_batch_complete", "run_id": run_id, "task": task,
                                          "batch": identity["batch"], "ordinary_gradient_calls": 6, "status": "passed"}), flush=True)
                    del learner, model, geometry
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            after_pins, _, _ = checkpoint_inputs(run_dir)
            if after_pins != input_pins:
                raise AssertionError("Reference training artifacts changed during offline audit")
            if {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*.py"))} != source_pins:
                raise AssertionError("Frozen model/data source changed during offline audit")
            if {name: sha(code_dir/name) for name in code_pins} != code_pins:
                raise AssertionError("Diagnostic implementation changed during offline audit")
            if dataset_identity(dm, construction_assets["data"]) != data_manifest:
                raise AssertionError("Local dataset changed during offline audit")
            if validate_input_assets(run_dir, source) != construction_assets:
                raise AssertionError("Pinned pretrained/source/manifest assets changed during offline audit")
            total = sum(len(state["batches"]) for state in report["state_results"])
            if total != 18 or report["ordinary_gradient_calls"] != 108:
                raise AssertionError("Incomplete preregistered 18-state-batch / 108-gradient budget")
            report.update(status="passed", state_batch_count=total, entry_state_restored=True,
                          source_unchanged=True, reference_assets_unchanged=True, diagnostic_code_unchanged=True,
                          local_data_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        rng_set(entry_rng)
        os.chdir(entry_cwd)
        report["process_rng_restored"] = rng_equal(entry_rng, rng_get())
        write_json(target, report)
    print(json.dumps({"status": report["status"], "report": str(target)}), flush=True)


if __name__ == "__main__":
    main()
