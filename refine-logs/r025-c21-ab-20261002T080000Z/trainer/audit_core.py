#!/usr/bin/env python3
"""Read-only, same-checkpoint/batch GAM diagnostics; never trains or saves a model.

Use the frozen source for model/data construction and the real GAM optimizer.
All temporary probes/updates restore parameters, buffers, module modes and RNG.
Old test data is used only to measure counterfactual loss, never to form a gradient.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_get():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_set(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def rng_equal(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0]
            and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
            and torch.equal(a[2], b[2]) and len(a[3]) == len(b[3])
            and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


class Snapshot:
    """Full tensor snapshot, including frozen parameters, and non-tensor RNG/modes."""
    def __init__(self, model):
        self.model = model
        self.params = [(p, p.detach().clone(), None if p.grad is None else p.grad.detach().clone())
                       for p in model.parameters()]
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]
        self.modes = [(m, m.training) for m in model.modules()]
        self.bn = [(m, m.momentum, hasattr(m, "backup_momentum"), getattr(m, "backup_momentum", None))
                   for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.rng = rng_get()

    @torch.no_grad()
    def restore(self):
        for p, value, grad in self.params:
            p.copy_(value)
            p.grad = None if grad is None else grad.clone()
        for b, value in self.buffers:
            b.copy_(value)
        for m, mode in self.modes:
            m.training = mode
        for m, momentum, had_backup, backup in self.bn:
            m.momentum = momentum
            if had_backup:
                m.backup_momentum = backup
            elif hasattr(m, "backup_momentum"):
                delattr(m, "backup_momentum")
        rng_set(self.rng)

    def check(self):
        result = {
            "parameters_exact": all(torch.equal(p, v) for p, v, _ in self.params),
            "buffers_exact": all(torch.equal(b, v) for b, v in self.buffers),
            "modes_exact": all(m.training == mode for m, mode in self.modes),
            "rng_exact": rng_equal(self.rng, rng_get()),
            "gradients_exact": all((p.grad is None and g is None) or
                                   (p.grad is not None and g is not None and torch.equal(p.grad, g))
                                   for p, _, g in self.params),
        }
        if not all(result.values()):
            raise RuntimeError("Restoration failed: " + repr(result))
        return result


def flatten(values):
    return torch.cat([x.reshape(-1) for x in values])


def dot(a, b):
    value = float(torch.sum(a.double() * b.double()).item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite inner product; refusing to mask it in a norm")
    return value


def norm(a):
    return math.sqrt(max(0.0, dot(a, a)))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator > 1e-20 else None


def unit(a):
    n = norm(a)
    return a / n if n > 1e-20 else torch.zeros_like(a)


class Geometry:
    def __init__(self, model):
        self.model = model
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        if not self.params:
            raise RuntimeError("No trainable parameters")
        self.offsets = {}
        offset = 0
        for n, p in self.named:
            self.offsets[id(p)] = (offset, offset + p.numel())
            offset += p.numel()
        self.base = [p.detach().clone() for p in self.params]
        self.base_map = {id(p): v for p, v in zip(self.params, self.base)}
        bb = model.backbone
        if bool(getattr(bb, "learn_alpha", False)):
            raise ValueError("Effective BA coordinate requires frozen identity LoRA scaling")
        self.scaling_classes = [type(m).__name__ for m in getattr(bb, "wrapped_param", [])]
        if self.scaling_classes and any(name != "IdentityScale" for name in self.scaling_classes):
            raise ValueError("Expected IdentityScale, got " + repr(self.scaling_classes))
        self.pairs = list(zip(bb.w_As, bb.w_Bs))
        self.meta = list(getattr(bb, "w_meta", []))
        self.fc = [(n, p) for n, p in self.named if n.startswith("fc.")]
        covered = {id(p) for _, p in self.fc}
        for a, b in self.pairs:
            covered.update((id(a.weight), id(b.weight)))
        if covered != {id(p) for p in self.params}:
            raise RuntimeError("Effective norm does not cover exactly all trainable parameters")
        self.base_ba = [(self.base_map[id(b.weight)] @ self.base_map[id(a.weight)]).detach()
                        for a, b in self.pairs]

    def split(self, vector, p):
        start, end = self.offsets[id(p)]
        return vector[start:end].view_as(p)

    @torch.no_grad()
    def set_displacement(self, vector, scale=1.0):
        for p, value in zip(self.params, self.base):
            p.copy_(value + float(scale) * self.split(vector, p))

    def parameter_delta(self):
        return flatten([p.detach() - v for p, v in zip(self.params, self.base)])

    @torch.no_grad()
    def displacement_metrics(self, with_vector=False):
        parts = [(b.weight @ a.weight - old).detach() for (a, b), old in zip(self.pairs, self.base_ba)]
        fc = [(p - self.base_map[id(p)]).detach() for _, p in self.fc]
        ls = sum(dot(x, x) for x in parts)
        fs = sum(dot(x, x) for x in fc)
        result = {"parameter_norm": norm(self.parameter_delta()),
                  "effective_lora_ba_norm": math.sqrt(ls), "fc_norm": math.sqrt(fs),
                  "effective_joint_norm": math.sqrt(ls + fs),
                  "per_lora_pair_ba_norm": [norm(x) for x in parts]}
        if with_vector:
            return result, flatten(parts + fc)
        return result

    def layer_norms(self, vector):
        result = {"fc": math.sqrt(sum(dot(self.split(vector, p), self.split(vector, p)) for _, p in self.fc))}
        for idx, (a, b) in enumerate(self.pairs):
            meta = self.meta[idx] if len(self.meta) == len(self.pairs) else {}
            key = "block_%02d_%s" % (int(meta.get("layer", idx // 2)), meta.get("channel", "q" if idx % 2 == 0 else "v"))
            result[key] = math.sqrt(dot(self.split(vector, a.weight), self.split(vector, a.weight))
                                    + dot(self.split(vector, b.weight), self.split(vector, b.weight)))
        return result

    @torch.no_grad()
    def effective_polynomial(self, direction):
        # theta' = theta - s*d. BA' - BA = -s*(B*dA+dB*A) + s^2*dB*dA.
        c2 = c3 = c4 = fc2 = 0.0
        for a, b in self.pairs:
            aa, bb = self.base_map[id(a.weight)], self.base_map[id(b.weight)]
            da, db = self.split(direction, a.weight), self.split(direction, b.weight)
            linear = bb @ da + db @ aa
            quadratic = db @ da
            c2 += dot(linear, linear)
            c3 -= 2 * dot(linear, quadratic)
            c4 += dot(quadratic, quadratic)
        for _, p in self.fc:
            dp = self.split(direction, p)
            fc2 += dot(dp, dp)
        return c2 + fc2, c3, c4, fc2


def positive_root(target, fn, initial, rtol=1e-5, max_expand=24, max_iter=48):
    """Bracket positive scalar then bisect. Fail explicitly; no equal-norm fallback."""
    if not math.isfinite(target) or target <= 1e-20:
        return {"ok": False, "reason": "nonpositive_or_nonfinite_target"}
    lo, hi = 0.0, max(float(initial), 1e-12)
    value = float(fn(hi))
    expansions = 0
    while math.isfinite(value) and value < target and expansions < max_expand:
        lo, hi = hi, hi * 2
        value = float(fn(hi))
        expansions += 1
    if not math.isfinite(value) or value < target:
        return {"ok": False, "reason": "unbracketed_or_nonfinite", "upper": hi, "upper_value": value}
    for iteration in range(max_iter):
        mid = (lo + hi) / 2
        value = float(fn(mid))
        if not math.isfinite(value):
            return {"ok": False, "reason": "nonfinite_in_bracket"}
        if abs(value - target) <= rtol * target:
            return {"ok": True, "scale": mid, "value": value, "target": target,
                    "relative_error": abs(value - target) / target, "iterations": iteration + 1}
        if value < target:
            lo = mid
        else:
            hi = mid
    return {"ok": False, "reason": "tolerance_not_met", "scale": mid, "value": value,
            "target": target, "relative_error": abs(value - target) / target}


def kl_radius_root(target, fn, rho):
    if not math.isfinite(target) or target <= 1e-12:
        return {"ok": False, "reason": "KL_target_too_small_or_nonfinite"}
    lo, previous = 0.0, 0.0
    grid = []
    bracket = None
    for index in range(12):
        hi = float(rho) * 0.25 * (2 ** index)
        value = float(fn(hi))
        grid.append([hi, value])
        if not math.isfinite(value):
            return {"ok": False, "reason": "nonfinite_KL", "grid": grid}
        if value < previous - max(1e-10, abs(previous) * 1e-3):
            return {"ok": False, "reason": "nonmonotonic_KL_grid", "grid": grid}
        if value >= target:
            bracket = (lo, hi)
            break
        lo, previous = hi, value
    if bracket is None:
        return {"ok": False, "reason": "no_KL_bracket", "grid": grid}
    lo, hi = bracket
    low_value, high_value = previous, value
    for iteration in range(10):
        mid = (lo + hi) / 2
        value = float(fn(mid))
        relative = abs(value - target) / target
        if not math.isfinite(value):
            return {"ok": False, "reason": "nonfinite_KL_bisection", "grid": grid}
        if value < low_value - max(1e-10, low_value * 1e-3) or value > high_value + max(1e-10, high_value * 1e-3):
            return {"ok": False, "reason": "nonmonotonic_KL_bracket", "grid": grid}
        if relative <= 0.01:
            return {"ok": True, "scale": mid, "value": value, "target": target,
                    "relative_error": relative, "iterations": iteration + 1, "grid": grid}
        if value < target:
            lo, low_value = mid, value
        else:
            hi, high_value = mid, value
    return {"ok": False, "reason": "KL_tolerance_not_met", "scale": mid,
            "value": value, "target": target, "relative_error": relative, "grid": grid}


def logits(model, x):
    result = model(x)
    return result["logits"] if isinstance(result, dict) else result


@torch.no_grad()
def isolated_logits(model, x):
    # Do not allow metric forwards to perturb the RNG stream of the real closures.
    rng = rng_get()
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    try:
        model.eval()
        return logits(model, x).detach()
    finally:
        for b, v in buffers:
            b.copy_(v)
        for m, mode in modes:
            m.training = mode
        rng_set(rng)


def kl_from_logits(reference, changed):
    # KL(p_anchor || p_probe), all seen logits, float64 for small changes.
    logp = reference.double().log_softmax(-1)
    logq = changed.double().log_softmax(-1)
    return float((logp.exp() * (logp - logq)).sum(-1).mean().item())


def gam_probes(model, geometry, anchor, x, y, class_start, class_end, rho, radius, metric_x, reference_logits, GAM):
    anchor.restore()
    model.train()
    params = geometry.params
    base_optimizer = torch.optim.SGD(params, lr=0.0, momentum=0.0, weight_decay=0.0)
    opt = GAM(params, base_optimizer=base_optimizer, model=model, grad_rho=rho,
              grad_norm_rho=radius, adaptive=False, perturb_eps=1e-12,
              args=SimpleNamespace(grad_beta_1=1.0, grad_beta_2=1.0, grad_beta_3=1.0, grad_gamma=0.0))
    points, shifts = [], []
    try:
        # Mirror the learner's extra clean-gradient cache pass, including RNG use.
        clean_logits = logits(model, x)
        clean_loss = F.cross_entropy(clean_logits[:, class_start:class_end], y - class_start)
        clean_grads = torch.autograd.grad(clean_loss, params, allow_unused=True)
        clean = flatten([torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, clean_grads)])
        del clean_logits, clean_loss, clean_grads

        def closure():
            point = geometry.displacement_metrics()
            point["kl_all_seen"] = kl_from_logits(reference_logits, isolated_logits(model, metric_x))
            shifts.append(geometry.parameter_delta().detach().clone())
            opt.zero_grad(set_to_none=True)
            out = logits(model, x)
            loss = F.cross_entropy(out[:, class_start:class_end], y - class_start)
            loss.backward()
            point["probe_batch_ce"] = float(loss.detach())
            points.append(point)
            return out.detach(), loss.detach()

        opt.step(closure)
        grads = {"g%d" % k: flatten([opt.state[p]["g_%d" % k].detach().clone() for p in params]) for k in range(4)}
        if not all(torch.isfinite(g).all().item() for g in grads.values()):
            raise FloatingPointError("Nonfinite GAM gradient")
        stats = {"radius": radius, "rho": rho, "closure_count": len(points), "ce_gradient_passes": 5,
                 "points": points, "clean_cache_vs_g0_maxabs": float((clean - grads["g0"]).abs().max())}
        return grads, stats, shifts
    finally:
        del opt, base_optimizer
        anchor.restore()
        anchor.check()


@torch.no_grad()
def eval_losses(model, batches, class_ranges, device):
    result = {}
    for task, group in batches.items():
        sum_all = sum_local = 0.0
        count = 0
        start, end = class_ranges[task]
        for x, y, _ in group:
            x, y = x.to(device), y.to(device)
            out = isolated_logits(model, x)
            sum_all += float(F.cross_entropy(out, y, reduction="sum"))
            sum_local += float(F.cross_entropy(out[:, start:end], y - start, reduction="sum"))
            count += y.numel()
        result[str(task)] = {"all_seen_ce": sum_all / count, "task_local_ce": sum_local / count, "n": count}
    return result


def loss_deltas(before, after, task):
    per_task = {k: {metric: after[k][metric] - before[k][metric]
                    for metric in ("all_seen_ce", "task_local_ce")} for k in before}
    old = [per_task[str(j)] for j in range(task)]
    return {"per_task": per_task, "current": per_task[str(task)],
            "old_task_macro": {m: sum(v[m] for v in old) / len(old) for m in ("all_seen_ce", "task_local_ce")}
            if old else None, "old_task_status": "available" if old else "NA_no_old_task"}


def counterfactual(model, geo, anchor, direction, label, target, eta, batches, class_ranges, task, device, before, match_rtol):
    c2, c3, c4, fc2 = geo.effective_polynomial(direction)
    def analytic(s):
        return math.sqrt(max(0.0, c2 * s * s + c3 * s ** 3 + c4 * s ** 4))
    match = positive_root(target, analytic, eta, rtol=min(match_rtol / 10, 1e-5))
    result = {"label": label, "direction_norm": norm(direction), "effective_norm_polynomial": [c2, c3, c4],
              "raw_eta": eta, "raw_effective_joint_norm_analytic": analytic(eta), "match": match}
    derivative_roots = np.roots([4 * c4, 3 * c3, 2 * c2]) if c4 != 0 or c3 != 0 else []
    result["positive_stationary_scales"] = sorted(float(z.real) for z in derivative_roots if abs(z.imag) < 1e-10 and z.real > 0)
    result["norm_globally_monotonic"] = not result["positive_stationary_scales"]
    try:
        anchor.restore()
        geo.set_displacement(direction, -eta)
        result["raw_actual_step"] = geo.displacement_metrics()
        if not match["ok"]:
            return result
        anchor.restore()
        geo.set_displacement(direction, -match["scale"])
        metrics = geo.displacement_metrics()
        actual_error = abs(metrics["effective_joint_norm"] - target) / target
        result["matched_actual_step"] = metrics
        match["actual_relative_error"] = actual_error
        match["ok"] = bool(math.isfinite(actual_error) and actual_error <= match_rtol)
        if not match["ok"]:
            match["reason"] = "actual_BA_match_tolerance_not_met"
            return result
        after = eval_losses(model, batches, class_ranges, device)
        result["loss_after"] = after
        result["loss_change"] = loss_deltas(before, after, task)
        return result
    finally:
        anchor.restore()
        anchor.check()


def sam_at_matched_probe(model, geo, anchor, g0, target_distance, x, y, start, end,
                         metric_x, reference_logits, rho, target_kl=None):
    direction = g0 / (torch.linalg.vector_norm(g0) + 1e-12)
    dnorm = norm(direction)
    if dnorm <= 1e-20:
        return {"ok": False, "reason": "zero_g0"}, None
    result = {"matching": "parameter_distance" if target_kl is None else "KL_all_seen"}
    try:
        if target_kl is None:
            scalar = target_distance / dnorm
            result["match"] = {"ok": True, "scale": scalar, "target": target_distance}
        else:
            def measure(s):
                anchor.restore()
                geo.set_displacement(direction, s)
                return kl_from_logits(reference_logits, isolated_logits(model, metric_x))
            match = kl_radius_root(target_kl, measure, rho)
            result["match"] = match
            if not match["ok"]:
                result["ok"] = False
                return result, None
            scalar = match["scale"]
        anchor.restore()
        geo.set_displacement(direction, scalar)
        result["actual_probe"] = geo.displacement_metrics()
        result["actual_probe"]["kl_all_seen"] = kl_from_logits(reference_logits, isolated_logits(model, metric_x))
        if target_kl is None:
            relative = abs(result["actual_probe"]["parameter_norm"] - target_distance) / max(target_distance, 1e-20)
            result["match"]["actual_relative_error"] = relative
            if relative > 1e-4:
                result.update(ok=False, reason="actual_parameter_distance_not_matched")
                return result, None
        model.train()
        loss = F.cross_entropy(logits(model, x)[:, start:end], y - start)
        grads = torch.autograd.grad(loss, geo.params, allow_unused=True)
        vector = flatten([torch.zeros_like(p) if g is None else g.detach() for p, g in zip(geo.params, grads)])
        result.update(ok=bool(torch.isfinite(vector).all()), sam_rho=scalar,
                      probe_batch_ce=float(loss.detach()), direction_norm=norm(vector))
        return result, vector
    finally:
        anchor.restore()
        anchor.check()


def cache_batches(dm, task, source, mode, count, batch_size, seed):
    seed_all(seed)
    start, end = dm.get_task_class_range(task)
    dataset = dm.get_dataset(np.arange(start, end), source=source, mode=mode)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=0, generator=generator, drop_last=False)
    result = []
    for indices, x, y in loader:
        result.append((x.detach().cpu(), y.detach().cpu(), indices.detach().cpu()))
        if len(result) >= count:
            break
    if len(result) < count:
        raise ValueError("Not enough batches: requested %d, got %d" % (count, len(result)))
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def nonfinite_paths(value, path=""):
    if isinstance(value, float):
        return [path] if not math.isfinite(value) else []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_paths(v, path + "/" + str(k))]
    if isinstance(value, (list, tuple)):
        return [p for k, v in enumerate(value) for p in nonfinite_paths(v, path + "/" + str(k))]
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--task", required=True, type=int)
    p.add_argument("--gpu", default="0", help="visible CUDA index, or cpu for tests")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--eval-batches", "--eval-per-task", dest="eval_batches", type=int, default=1, help="held-out batches per current/old task")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260930, help="shared diagnostic data/probe seed, not source training seed")
    p.add_argument("--eta-ref", type=float, default=0.02)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--r-values", default="0,0.1,0.25,0.5,1")
    p.add_argument("--match-rtol", type=float, default=1e-3)
    p.add_argument("--kl-match", action="store_true", help="also solve KL-matched SAM radius; adds metric forwards")
    p.add_argument("--save-vectors", action="store_true")
    args = p.parse_args()
    if min(args.max_batches, args.eval_batches, args.batch_size) < 1:
        p.error("batch counts and batch size must be positive")
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from utils.data_manager import DataManager
    from utils import factory
    from optimer_PerturabtionType.gam import GAM

    config = json.loads(Path(args.base_config).read_text())
    if bool(config.get("gam_adaptive", False)):
        raise ValueError("This diagnostic is pinned to the nonadaptive pure-GAM experiment")
    training_seed = config["seed"][0] if isinstance(config["seed"], list) else config["seed"]
    device = torch.device("cpu" if args.gpu == "cpu" else "cuda:" + args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config.update(seed=int(training_seed), device=[device], filepath=str(Path(args.checkpoint_dir).resolve()) + os.sep,
                  train_num_workers=0, eval_num_workers=0, resume_require_experiment_identity=False,
                  resume_require_model_sidecars=False, resume_require_fisher_state=False)
    seed_all(int(training_seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dm = DataManager(config["dataset"], config.get("class_shuffle", config.get("shuffle", True)),
                     int(training_seed), config["init_cls"], config["increment"], config)
    config.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
    learner = factory.get_model(config["model_name"], config)
    learner._unwrap_network().backbone.to(device)
    learner.load_task_checkpoint(args.checkpoint_dir, args.task, dm)
    model = learner._unwrap_network().to(device)
    stochastic_layers = [name for name, module in model.named_modules()
                         if (isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0)
                         or float(getattr(module, "drop_prob", 0.0) or 0.0) > 0]
    if stochastic_layers:
        raise ValueError("This protocol requires zero dropout/drop-path; cannot silently compare RNG call positions: " + repr(stochastic_layers))
    stochastic_layer_audit = [{"name": name, "class": type(module).__name__,
                              "dropout_p": float(module.p) if isinstance(module, torch.nn.modules.dropout._DropoutNd) else None,
                              "drop_path_probability": float(getattr(module, "drop_prob", 0.0) or 0.0)}
                             for name, module in model.named_modules()
                             if isinstance(module, torch.nn.modules.dropout._DropoutNd) or hasattr(module, "drop_prob")]
    geo = Geometry(model)
    start, end = dm.get_task_class_range(args.task)
    class_ranges = {t: dm.get_task_class_range(t) for t in range(args.task + 1)}
    train_batches = cache_batches(dm, args.task, "train", "train", args.max_batches,
                                  args.batch_size, args.seed + args.task * 1000)
    eval_batches = {t: cache_batches(dm, t, "test", "test", args.eval_batches,
                                     args.batch_size, args.seed + 100000 + t * 1000)
                    for t in range(args.task + 1)}
    seed_all(args.seed + args.task)
    model.train()
    anchor = Snapshot(model)
    metric_x = eval_batches[args.task][0][0].to(device)
    reference_logits = isolated_logits(model, metric_x)
    before = eval_losses(model, eval_batches, class_ranges, device)
    anchor.restore()
    radii = [float(x) for x in args.r_values.split(",")]
    if 1.0 not in radii:
        raise ValueError("r-values must include 1 for the principal paired analysis")
    checkpoint_files = [Path(args.checkpoint_dir) / (prefix + str(args.task) + suffix)
                        for prefix, suffix in (("lora_w_a_", ".pt"), ("lora_w_b_", ".pt"),
                                               ("lora_meta_", ".json"), ("fc_state_", ".pt"))]
    report = {"status": "running", "schema_version": 1, "arguments": vars(args),
              "source": str(source), "source_config_sha256": digest(args.base_config),
              "source_hashes": {f: digest(source / f) for f in ("optimer_PerturabtionType/gam.py", "backbone/lora.py", "models_LoRAbasedCL/baseLoRA.py")},
              "checkpoint_hashes": {f.name: digest(f) for f in checkpoint_files},
              "training_seed": training_seed, "task": args.task, "class_range": [start, end],
              "trainable_parameters": [{"name": n, "shape": list(v.shape), "numel": v.numel()} for n, v in geo.named],
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
              "effective_coordinate": "concatenated unscaled LoRA BA differences and FC weight/bias differences",
              "lora_scaling_modules": geo.scaling_classes, "stochastic_layers": stochastic_layers,
              "stochastic_layer_audit": stochastic_layer_audit,
              "loss_before": before, "eval_batch_hashes": {str(t): [tensor_digest(x) for x, _, _ in bb] for t, bb in eval_batches.items()},
              "notes": ["Task-end offline diagnostics; no training/update is committed.",
                        "eta_ref=.02 is a virtual diagnostic step, not the nearly-zero task-end cosine learning rate.",
                        "Probe gradients use current-task sliced CE; held-out losses report both all-seen and task-local CE.",
                        "Each real GAM run includes the learner's extra clean gradient pass; no momentum/weight decay.",
                        "Positive scale-only and g1 must collapse to the same matched update; this is a sanity check, not an independent direction."],
              "batches": []}
    started = time.time()
    atomic_json(args.output, report)
    try:
        for batch_idx, (inputs, targets, indices) in enumerate(train_batches):
            print(json.dumps({"event": "batch_start", "batch": batch_idx, "task": args.task}), flush=True)
            x, y = inputs.to(device), targets.to(device)
            scans, gradient_scans, shifts = {}, {}, {}
            for radius in radii:
                gradients, stats, delta = gam_probes(model, geo, anchor, x, y, start, end, args.rho, radius, metric_x, reference_logits, GAM)
                scans[str(radius)], gradient_scans[radius], shifts[radius] = stats, gradients, delta
            grads = gradient_scans[1.0]
            g0, g1, g2, g3 = [grads["g%d" % k] for k in range(4)]
            denominator = dot(g1, g1)
            if denominator <= 1e-20:
                raise FloatingPointError("g1 norm is too small for an identified projection coefficient")
            a = dot(g3, g1) / denominator
            residual = g3 - a * g1
            entry = {"batch": batch_idx, "sample_count": y.numel(), "sample_indices": indices.tolist(),
                     "inputs_sha256": tensor_digest(inputs), "targets_sha256": tensor_digest(targets),
                     "geometry": {"a": a, "a_denominator": denominator, "a_clipped": False,
                                  "r_perp_norm": norm(residual), "r_perp_over_g3": norm(residual) / max(norm(g3), 1e-20),
                                  "orthogonality_inner_product": dot(residual, g1),
                                  "orthogonality_relative": abs(dot(residual, g1)) / max(norm(residual) * norm(g1), 1e-20),
                                  "gradient_norms": {k: norm(v) for k, v in grads.items()},
                                  "cosines": {"%s_%s" % (i, j): cosine(grads[i], grads[j])
                                              for i, j in (("g0", "g1"), ("g0", "g3"), ("g1", "g3"), ("g1", "g2"))},
                                  "per_layer_norms": {k: geo.layer_norms(v) for k, v in {**grads, "r_perp": residual}.items()}},
                     "radius_scan": scans, "counterfactuals": [], "radius_matched_sam": {}}
            anchor.restore()
            geo.set_displacement(g1, -args.eta_ref)
            target_metrics, effective_g1 = geo.displacement_metrics(with_vector=True)
            target = target_metrics["effective_joint_norm"]
            entry["reference_update"] = {"direction": "g1", "eta": args.eta_ref, **target_metrics}
            for radius in radii:
                scan_grad = gradient_scans[radius]["g3"]
                anchor.restore()
                geo.set_displacement(scan_grad, -args.eta_ref)
                update_metrics, effective_r = geo.displacement_metrics(with_vector=True)
                scans[str(radius)]["g3_gradient_norm"] = norm(scan_grad)
                scans[str(radius)]["cos_g1_g3"] = cosine(g1, scan_grad)
                scans[str(radius)]["raw_update_at_eta_ref"] = update_metrics
                scans[str(radius)]["effective_update_cos_g1"] = cosine(effective_g1, effective_r)
                scans[str(radius)]["effective_update_direction_deviation"] = (None if cosine(effective_g1, effective_r) is None else 1 - cosine(effective_g1, effective_r))
                scans[str(radius)]["g3_minus_g1_relative_norm"] = norm(scan_grad - g1) / max(norm(g1), 1e-20)
            anchor.restore()
            directions = {"g1": g1, "g3": g3, "sum": g1 + g3, "half_sum": (g1 + g3) / 2,
                          "scale_only": (1 + a) * g1, "g3_scale_only": a * g1}
            for lam in (0, 0.5, 1, 2):
                directions["g1_plus_residual_lambda_%g" % lam] = g1 + lam * residual
                directions["sum_family_lambda_%g" % lam] = (1 + a) * g1 + lam * residual
            for alpha in (0, 0.25, 0.5, 0.75, 1):
                directions["unit_interpolation_alpha_%g" % alpha] = unit((1 - alpha) * unit(g1) + alpha * unit(g3))
            for radius in radii:
                if radius != 1.0:
                    directions["g3_radius_%g" % radius] = gradient_scans[radius]["g3"]
            for kind in (["parameter", "kl"] if args.kl_match else ["parameter"]):
                sam_result, sam_gradient = sam_at_matched_probe(
                    model, geo, anchor, g0, scans["1.0"]["points"][3]["parameter_norm"], x, y, start, end,
                    metric_x, reference_logits, args.rho,
                    target_kl=scans["1.0"]["points"][3]["kl_all_seen"] if kind == "kl" else None)
                entry["radius_matched_sam"][kind] = sam_result
                if sam_gradient is not None and sam_result["ok"]:
                    directions["sam_%s_matched_probe" % kind] = sam_gradient
                    sam_result["cos_g3"] = cosine(sam_gradient, g3)
            for label, direction in directions.items():
                entry["counterfactuals"].append(counterfactual(
                    model, geo, anchor, direction, label, target, args.eta_ref, eval_batches,
                    class_ranges, args.task, device, before, args.match_rtol))
            entry["scale_only_sanity"] = {"expected_same_direction_as_g1": (1 + a) > 0,
                                           "scale_factor": 1 + a, "cosine": cosine(directions["scale_only"], g1)}
            matches = {c["label"]: c["match"] for c in entry["counterfactuals"]}
            if (1 + a) > 0 and matches["g1"]["ok"] and matches["scale_only"]["ok"]:
                relative = abs(matches["scale_only"]["scale"] * (1 + a) - matches["g1"]["scale"]) / max(abs(matches["g1"]["scale"]), 1e-20)
                entry["scale_only_sanity"].update(matched_parameter_step_relative_error=relative,
                                                  passed=relative <= 1e-3)
                if relative > 1e-3:
                    raise AssertionError("Positive scale-only/g1 matched updates differ")
            if 0.0 in gradient_scans:
                r0_relative = norm(gradient_scans[0.0]["g3"] - g1) / max(norm(g1), 1e-20)
                entry["r0_identity_sanity"] = {"relative_error": r0_relative, "tolerance": 5e-4,
                                                "passed": r0_relative <= 5e-4}
                if r0_relative > 5e-4:
                    raise AssertionError("r=0 g3 differs from g1; check stochastic layers/RNG")
            if args.save_vectors:
                vector_path = str(Path(args.output).with_suffix("")) + ".batch%02d.vectors.pt" % batch_idx
                torch.save({"gradients": {k: v.cpu() for k, v in grads.items()}, "residual": residual.cpu(),
                            "trainable_names": [n for n, _ in geo.named],
                            "g3_by_r": {str(r): v["g3"].cpu() for r, v in gradient_scans.items()}}, vector_path)
                entry["vectors_file"] = vector_path
            anchor.restore()
            entry["restore_check"] = anchor.check()
            invalid = nonfinite_paths(entry)
            if invalid:
                raise FloatingPointError("Nonfinite result fields: " + repr(invalid))
            report["batches"].append(entry)
            report["elapsed_seconds"] = time.time() - started
            atomic_json(args.output, report)
            print(json.dumps({"event": "batch_complete", "batch": batch_idx, "seconds": report["elapsed_seconds"]}), flush=True)
        report["finite_check"] = {"ok": not nonfinite_paths(report), "invalid_paths": nonfinite_paths(report)}
        if not report["finite_check"]["ok"]:
            raise FloatingPointError("Nonfinite report")
        report["status"] = "completed"
        report["batch_count"] = len(report["batches"])
        report["all_matches_successful"] = all(c["match"]["ok"] for b in report["batches"] for c in b["counterfactuals"])
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        anchor.restore()
        report["final_restore_check"] = anchor.check()
        report["elapsed_seconds"] = time.time() - started
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
