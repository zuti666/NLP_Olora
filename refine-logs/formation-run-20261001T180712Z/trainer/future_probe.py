#!/usr/bin/env python3
"""One fresh Task-1 susceptibility trajectory from a native Task-0 checkpoint.

No checkpoint is overwritten.  Future S is native GAM g1 (not clean CE): the
inherited batch method retains its extra clean backward and all four probes.
Only SGD's scalar changes for g_match; matching uses actual unscaled BA + FC.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import audit_core as audit
from effective_step import EffectiveStep, solve_positive_scale


def parameter_hash(model, trainable_only=False):
    """Same byte/name/shape convention as native formation checkpoint manifest."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def labels_hash(labels):
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()


def checked_finite(value, label):
    if audit.nonfinite_paths(value):
        raise FloatingPointError(label + ": " + repr(audit.nonfinite_paths(value)))
    return value


def finite_model(model):
    if not all(bool(torch.isfinite(x).all()) for x in list(model.parameters()) + list(model.buffers())):
        raise FloatingPointError("Model contains nonfinite tensors")


@contextlib.contextmanager
def isolated_rng(seed=None):
    state = audit.rng_get()
    try:
        if seed is not None:
            audit.seed_all(int(seed))
        yield
    finally:
        audit.rng_set(state)
        if not audit.rng_equal(state, audit.rng_get()):
            raise AssertionError("RNG restoration failed")


@contextlib.contextmanager
def observational(model):
    """Evaluation has no lasting RNG, buffer, mode, parameter, or grad effects."""
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    versions = [(p, p._version) for p in model.parameters()]
    grads = [(p, p.grad, None if p.grad is None else p.grad.detach().clone())
             for p in model.parameters() if p.requires_grad]
    with isolated_rng():
        try:
            with torch.no_grad():
                model.eval()
                yield
        finally:
            with torch.no_grad():
                for b, value in buffers:
                    b.copy_(value)
            for m, mode in modes:
                m.training = mode
            if any(p._version != version for p, version in versions):
                raise AssertionError("Observational evaluation mutated a parameter")
            for p, original, value in grads:
                if p.grad is not original or (value is not None and not torch.equal(p.grad, value)):
                    raise AssertionError("Observational evaluation mutated gradients")


def validate_split(spec, task, labels, canonical_ids=None):
    """Extra descriptive metadata is allowed; index/hash integrity is mandatory."""
    if spec.get("schema_version") != 1 or not isinstance(spec.get("split_seed"), int):
        raise ValueError("Expected frozen split schema_version=1 and integer split_seed")
    row = spec["task%d" % task]
    labels = np.asarray(labels, dtype=np.int64)
    if row.get("dataset_size") != len(labels) or row.get("labels_sha256") != labels_hash(labels):
        raise ValueError("Frozen split dataset size/labels hash mismatch for task %d" % task)
    if "canonical_ids_sha256" in row:
        if canonical_ids is None:
            raise ValueError("Canonical image IDs are required by this frozen split")
        ids = [str(x) for x in canonical_ids]
        digest = hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        if len(ids)!=len(labels) or len(set(ids))!=len(ids) or digest != row["canonical_ids_sha256"]:
            raise ValueError("Canonical sample IDs/order differs from frozen split")
    subsets = {}
    for name in ("diagnostic", "reporting"):
        ids = row.get(name + "_indices")
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError("Split indices must be lists of integers")
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(labels):
            raise ValueError("Empty, duplicate, or out-of-range split indices")
        subsets[name] = ids
    d, r = map(set, (subsets["diagnostic"], subsets["reporting"]))
    if d & r or d | r != set(range(len(labels))):
        raise ValueError("Diagnostic/reporting split must partition the full dataset")
    for label in np.unique(labels):
        ids = set(np.flatnonzero(labels==label).tolist())
        if len(ids)>=2 and (not ids & d or not ids & r):
            raise ValueError("Each nonsingleton class must occur in both subsets")
        if len(ids)==1 and not ids <= r:
            raise ValueError("Singleton evaluation classes belong to reporting only")
    metadata = {"diagnostic_n":len(d),"reporting_n":len(r),
                "per_class_counts":{str(int(y)):int((labels==y).sum()) for y in np.unique(labels)},
                "diagnostic_per_class_counts":{str(int(y)):int((labels[subsets["diagnostic"]]==y).sum()) for y in np.unique(labels)},
                "singleton_classes_reporting_only":[int(y) for y in np.unique(labels) if int((labels==y).sum())==1]}
    for name,expected in metadata.items():
        if name in row and row[name] != expected:
            raise ValueError("Frozen split count metadata mismatch: " + name)
    return {"full": list(range(len(labels))), **subsets}


class EpochDataStream:
    """Fresh per-epoch sampler and isolated augmentation RNG, never model RNG."""
    def __init__(self, dataset, batch_size, seed, loader_factory=None):
        self.dataset, self.batch_size, self.seed = dataset, int(batch_size), int(seed)
        self.batches_per_epoch = math.ceil(len(dataset) / self.batch_size)
        if self.batches_per_epoch < 1:
            raise ValueError("Empty Task1 training dataset")
        self.loader_factory = loader_factory or DataLoader

    def epoch(self, epoch):
        data_seed = self.seed * 100000 + 10000 + int(epoch)
        generator = torch.Generator().manual_seed(data_seed)
        # Zero workers makes augmentation-state capture complete and reproducible.
        loader = self.loader_factory(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=0, generator=generator, drop_last=False)
        if len(loader) != self.batches_per_epoch:
            raise AssertionError("Loader unexpectedly changed native dataset coverage")
        with isolated_rng(data_seed):
            data_state = audit.rng_get()
        iterator = None
        for batch_index in range(len(loader)):
            training_state = audit.rng_get()
            audit.rng_set(data_state)
            try:
                if iterator is None:
                    iterator = iter(loader)
                ids, inputs, targets = next(iterator)
                data_state = audit.rng_get()
            finally:
                audit.rng_set(training_state)
            identity = {"input_sha256": audit.tensor_digest(inputs),
                        "target_sha256": audit.tensor_digest(targets),
                        "indices_sha256": audit.tensor_digest(ids),
                        "indices": ids.tolist(), "data_seed": data_seed}
            yield batch_index, inputs, targets, identity


def metrics(losses, prediction, targets, start, end):
    losses, prediction, targets = map(np.asarray, (losses, prediction, targets))
    if len(targets) == 0 or not np.isfinite(losses).all():
        raise ValueError("Empty/nonfinite evaluation")
    counts = np.bincount(targets - start, minlength=end-start)
    correct = np.bincount((targets[prediction == targets] - start), minlength=end-start)
    present = counts > 0
    acc = 100. * correct / np.maximum(counts, 1)
    return {"ce": float(losses.mean()), "accuracy": float(100.*np.mean(prediction == targets)),
            "samples": int(len(targets)), "correct": int((prediction == targets).sum()),
            "macro_accuracy": float(acc[present].mean()), "classes_present": int(present.sum()),
            "per_class_counts": counts.tolist(), "per_class_correct": correct.tolist(),
            "per_class_accuracy": acc.tolist()}


def evaluate(model, dataset, start, end, seen, device, subsets=None, batch_size=128):
    """One forward pass per sample; full/diagnostic/reporting derive from same logits."""
    if subsets is None:
        subsets = {"diagnostic": list(range(len(dataset)))}
    rows = {key: [] for key in ("targets", "local_ce", "local_pred", "all_seen_ce", "all_seen_pred")}
    with observational(model):
        for _, x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            x, y = x.to(device), y.to(device)
            out = audit.logits(model, x)
            if out.ndim != 2 or out.shape[1] != seen or not bool(torch.isfinite(out).all()):
                raise ValueError("Unexpected/nonfinite classifier logits")
            if bool(((y < start) | (y >= end)).any()):
                raise ValueError("Evaluation task labels exceed declared class interval")
            rows["targets"].extend(y.cpu().tolist())
            rows["local_ce"].extend(F.cross_entropy(out[:, start:end], y-start, reduction="none").cpu().tolist())
            rows["local_pred"].extend((out[:, start:end].argmax(-1)+start).cpu().tolist())
            rows["all_seen_ce"].extend(F.cross_entropy(out[:, :seen], y, reduction="none").cpu().tolist())
            rows["all_seen_pred"].extend(out[:, :seen].argmax(-1).cpu().tolist())
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    result = {}
    for name, ids in subsets.items():
        result[name] = {space: metrics(arrays[space+"_ce"][ids], arrays[space+"_pred"][ids],
                                        arrays["targets"][ids], start, end)
                        for space in ("local", "all_seen")}
    return checked_finite(result, "evaluation")


def expand_head(model, old_classes, new_classes, seed, device):
    old = {k: v.detach().clone() for k, v in model.fc.state_dict().items()}
    backbone_before = parameter_hash(model.backbone)
    with isolated_rng(seed):
        model.update_fc(new_classes)
        model.to(device)
    for key, value in old.items():
        if not torch.equal(model.fc.state_dict()[key][:old_classes], value):
            raise AssertionError("Head expansion changed old FC rows")
    if parameter_hash(model.backbone) != backbone_before:
        raise AssertionError("Head expansion changed backbone/LoRA tensors")
    return {"head_seed": int(seed), "old_rows_exact": True, "backbone_exact": True,
            "new_rows_sha256": {k: audit.tensor_digest(v[old_classes:])
                                for k, v in model.fc.state_dict().items()},
            "expanded_parameter_sha256": parameter_hash(model)}


class PathIntegral:
    """Every-step diagnostic loss differences, relative to post-expansion K0."""
    def __init__(self, baseline):
        self.baseline = self.previous = float(baseline)
        if not math.isfinite(self.baseline):
            raise FloatingPointError("Nonfinite path baseline")
        self.steps, self.integral, self.positive = 0, 0., 0.

    def append(self, step, loss):
        loss = float(loss)
        if step != self.steps+1 or not math.isfinite(loss):
            raise ValueError("Nonfinite loss or nonconsecutive path step")
        delta = loss-self.previous
        self.integral += delta
        self.positive += max(0., delta)
        self.steps, self.previous = step, loss
        error = self.integral-(loss-self.baseline)
        if abs(error) > 1e-10*max(1., abs(loss), abs(self.baseline)):
            raise AssertionError("Path telescoping identity failed")
        return {"delta_loss": delta, "I": self.integral, "Iplus": self.positive,
                "baseline_k0": self.baseline, "telescoping_error": error, "steps": step}


class StepReference:
    def __init__(self, directory, seed, steps, source_gam_sha256, split_sha256, smoke=False):
        directory = Path(directory)
        meta = json.loads((directory/"identity.json").read_text())
        final = json.loads((directory/"summary.json").read_text())
        if (meta.get("seed") != seed or meta.get("formation_arm") != "s" or
                meta.get("formation_epoch") != (1 if smoke else 50) or meta.get("future_rule") != "s" or
                bool(meta.get("smoke",False)) != bool(smoke) or
                meta.get("source_gam_sha256") != source_gam_sha256 or
                meta.get("selection_split_sha256") != split_sha256):
            raise ValueError("g_match requires same-seed S50→S with identical source and frozen split")
        if final.get("status") != "completed" or final.get("completed_steps", 0) < steps:
            raise ValueError("Future SS reference is incomplete")
        if final.get("step_trace_sha256") != audit.digest(directory/"step_trace.jsonl"):
            raise ValueError("Future SS trace hash differs from its completion receipt")
        if final.get("identity_sha256") != audit.digest(directory/"identity.json"):
            raise ValueError("Future SS identity differs from its completion receipt")
        rows = [json.loads(x) for x in (directory/"step_trace.jsonl").read_text().splitlines() if x.strip()]
        if len(rows) < steps or [r["step_index"] for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError("Future SS reference is missing/reordering steps")
        self.rows, self.directory = rows, str(directory.resolve())
        self.hashes = {name: audit.digest(directory/name) for name in ("identity.json", "summary.json", "step_trace.jsonl")}

    def target(self, step, epoch, batch, seed, lr, identity):
        row = self.rows[step-1]
        if any(row.get(k) != v for k, v in {"step_index":step, "epoch":epoch, "batch":batch,
                                           "seed":seed, "task":1}.items()):
            raise ValueError("SS reference step/seed/epoch identity mismatch")
        if not math.isclose(float(row["lr_nominal"]), lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("SS reference nominal LR mismatch")
        for key in ("input_sha256", "target_sha256", "indices_sha256", "indices", "data_seed"):
            if row["identity"].get(key) != identity[key]:
                raise ValueError("SS reference minibatch/augmentation mismatch: " + key)
        value = float(row["actual_effective_delta"]["joint_norm"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("SS reference has invalid actual effective norm")
        return value


def resolve_config(raw, cli):
    cfg = copy.deepcopy(raw)
    for key, expected in {"dataset":"imageneta", "class_shuffle":False, "init_cls":20,
                          "increment":20, "batch_size":128, "lora_rank":10}.items():
        if cfg.get(key, expected) != expected:
            raise ValueError("Fixed future protocol violation: " + key)
        cfg[key] = expected
    cfg.update(seed=cli.seed, device=[torch.device(cli.device)], optimizer="sgd", optimizer_type="gam",
               gam_base_optimizer="sgd", init_lr=.04, lrate=.02, init_momentum=0., momentum=0.,
               init_weight_decay=0., weight_decay=0., scheduler="cosine", epochs=50, init_epoch=50,
               lambda_flat=1., gam_grad_rho=.05, gam_grad_norm_rho=1., gam_adaptive=False,
               gam_grad_gamma=0., gam_grad_beta_1=1. if cli.future_rule=="s" else 0.,
               gam_grad_beta_2=1., gam_grad_beta_3=0. if cli.future_rule=="s" else 1.,
               ewc_lambda=0., ewc_skip_task_fisher=True, in_task_weight_merge=False,
               task_endpoint_weight_merge=False, approximate_fisher_in_task=False,
               in_task_fisher_omega=0., in_task_weight_merge_omega=0., ewc_fisher_damping_alpha=0.,
               ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False, ewc_dual_ascent=False,
               compute_full_fisher=False, posterior_state_save=False, mechanism_anchor_eval=False,
               mechanism_future_surrogate_eval=False, as1_delta_w_penalty=False, fisher_split=False,
               memory_size=0, memory_per_class=0, mechanism_eval=False, mechanism_grad_stats=False,
               endpoint_eval=False, task0_trajectory_eval=False, task0_fisher_recompute_eval=False,
               gam_use_sgv=False, use_sgv=False, sgv_lambda=0., train_num_workers=0, eval_num_workers=0,
               resume_require_experiment_identity=False, resume_require_model_sidecars=False,
               resume_require_fisher_state=False, filepath=str(Path(cli.output_dir)/"unused_native_checkpoint")+os.sep)
    for key in ("optimizer_type_by_task", "fc_lrate", "init_fc_lrate", "resume_from_task", "resume_from_checkpoint_dir"):
        cfg.pop(key, None)
    return cfg


def checkpoint_files(directory):
    required = {"lora_w_a_0.pt", "lora_w_b_0.pt", "lora_meta_0.json", "fc_state_0.pt",
                "training_state.pt", "manifest.json"}
    available = {p.name for p in directory.iterdir() if p.is_file()}
    if not required <= available:
        raise FileNotFoundError("Incomplete native Task0 checkpoint: " + repr(required-available))
    return {p.name: audit.digest(p) for p in sorted(directory.iterdir()) if p.is_file()}


class NativeStep:
    """Measure/correct the SGD scalar only after native GAM has unperturbed."""
    def __init__(self, learner, model, GAM, future_rule):
        self.learner, self.model, self.future_rule = learner, model, future_rule
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.base = torch.optim.SGD(self.params, lr=.02, momentum=0., weight_decay=0.)
        self.optimizer = GAM(self.params, base_optimizer=self.base, model=model,
                             grad_rho=.05, grad_norm_rho=1., adaptive=False,
                             args=SimpleNamespace(grad_beta_1=1. if future_rule=="s" else 0.,
                                                  grad_beta_2=1., grad_beta_3=0. if future_rule=="s" else 1.,
                                                  grad_gamma=0., use_sgv=False, sgv_lambda=0.))
        self.geometry = EffectiveStep.from_model(model)
        self.original_base_step = self.base.step
        def base_step(_base,*args,**kwargs):
            return self._base_step(_base,*args,**kwargs)
        self.base.step = MethodType(base_step, self.base)
        self.target_norm = None
        self.record = None
        self.nominal_lr = .02
        self.projector_exact = self.decomposition_exact = False
        native_projector = learner._build_direction_projector
        def build_projector(_learner,clean,fisher,delta=None):
            if fisher or delta or float(_learner._lambda_flat) != 1.:
                raise AssertionError("Future probe requires empty Fisher/penalty and lambda_flat=1")
            native = native_projector(clean,fisher,delta)
            def project(groups):
                params = [p for group in groups for p in group["params"]]
                if any(p.grad is None or id(p) not in clean for p in params):
                    raise AssertionError("Missing clean/native gradient")
                expected = [clean[id(p)] + 1.*(p.grad.detach()-clean[id(p)]) for p in params]
                result = native(groups)
                self.projector_exact = all(torch.equal(p.grad,v) for p,v in zip(params,expected))
                if not self.projector_exact:
                    raise AssertionError("Native clean+GAM projector differs from own operation sequence")
                return result
            return project
        learner._build_direction_projector = MethodType(build_projector,learner)

    def _base_step(self, _base, *args, **kwargs):
        if self.record is not None:
            raise AssertionError("Multiple SGD updates in one future step")
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.params):
            raise FloatingPointError("Missing/nonfinite final GAM gradient")
        if not self.projector_exact or not self.decomposition_exact:
            raise AssertionError("Missing decomposition/projector own-spec validation")
        before = self.geometry.capture()
        roundtrip = self.geometry.actual_delta(self.step_origin,before)
        poly = self.geometry.polynomial()
        match = None
        scale = self.nominal_lr
        if self.target_norm is not None:
            match = solve_positive_scale(poly, self.target_norm, initial_scale=self.nominal_lr, max_scale=10.)
            scale = match["scale"]
        for group in self.base.param_groups:
            group["lr"] = scale
        predicted = [p.detach().clone().add_(p.grad,alpha=-scale) for p in self.params]
        try:
            value = self.original_base_step(*args, **kwargs)
        finally:
            for group in self.base.param_groups:
                group["lr"] = self.nominal_lr
        actual = self.geometry.actual_delta(before)
        if not all(torch.equal(p,v) for p,v in zip(self.params,predicted)):
            raise AssertionError("Actual SGD differs from independent zero-momentum update specification")
        actual_match_error = (None if self.target_norm is None else
                              abs(actual["joint_norm"]-self.target_norm)/self.target_norm)
        # Real FP32 update, not only the ideal polynomial, must match.
        if actual_match_error is not None and actual_match_error > 1e-3:
            raise ArithmeticError("Actual FP32 BA+FC norm match exceeds 1e-3: " + str(actual_match_error))
        self.record = {"actual_effective_delta":actual, "step_scale":scale,
                       "effective_step_polynomial":poly.as_dict(), "norm_match":match,
                       "actual_match_relative_error":actual_match_error,
                       "decomposition_spec_exact":self.decomposition_exact,
                       "projector_spec_exact":self.projector_exact,"base_sgd_exact":True,
                       "self_spec_passed":True,"probe_roundtrip_delta":roundtrip,
                       "actual_norm_match_rtol":1e-3,"effective_base":"after_native_GAM_unperturb"}
        return value

    def step(self, inputs, targets, lr, target_norm=None):
        self.record, self.nominal_lr, self.target_norm = None, float(lr), target_norm
        self.projector_exact = self.decomposition_exact = False
        self.step_origin = self.geometry.capture()
        for group in self.base.param_groups:
            group["lr"] = self.nominal_lr
        self.model.train()
        calls = [0]
        closures = [0]
        forwards = [0]
        original = self.optimizer.gradient_decompose
        def decompose(*args, **kwargs):
            calls[0] += 1
            result = original(*args, **kwargs)
            b1,b3 = (1.,0.) if self.future_rule=="s" else (0.,1.)
            expected = [b1*self.optimizer.state[p]["g_1"]+b3*self.optimizer.state[p]["g_3"] for p in self.params]
            self.decomposition_exact = all(torch.equal(p.grad,v) for p,v in zip(self.params,expected))
            if not self.decomposition_exact:
                raise AssertionError("Native gamma0 decomposition differs from own selected-gradient specification")
            return result
        native_step = self.optimizer.step
        def wrapped_step(closure=None):
            if closure is None:
                raise AssertionError("Native GAM closure is required")
            def wrapped_closure():
                closures[0] += 1
                result = closure()
                if not bool(torch.isfinite(result[1])) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.params):
                    raise FloatingPointError("Nonfinite native GAM probe loss/gradient")
                return result
            return native_step(closure=wrapped_closure)
        def count_forward(_module,_inputs,_outputs):
            forwards[0] += 1
        forward_hook = self.model.register_forward_hook(count_forward)
        self.optimizer.gradient_decompose = decompose
        self.optimizer.step = wrapped_step
        try:
            logits, loss, log = self.learner._step_batch_with_log(self.optimizer, inputs, targets, class_offset=20)
        finally:
            self.optimizer.gradient_decompose = original
            self.optimizer.step = native_step
            forward_hook.remove()
        if calls[0] != 1 or closures[0] != 4 or forwards[0] != 5 or self.record is None:
            raise AssertionError("Missing native GAM decomposition/SGD callback")
        if float(log["ewc"]) != 0. or float(log["delta_w_penalty"]) != 0.:
            raise AssertionError("Unexpected EWC/DeltaW penalty")
        if any(not all(k in self.optimizer.state[p] for k in ("g_0", "g_1", "g_2", "g_3")) for p in self.params):
            raise AssertionError("Missing native GAM probe states")
        record = {**self.record, "train_loss":float(loss), "train_log":log,
                  "train_accuracy_local":float(100.*(logits[:,20:40].argmax(-1)==targets-20).float().mean()),
                  "training_ce_gradient_passes":5, "fisher_gradient_passes":0,
                  "weight_merge_calls":0,"closure_count":closures[0],"training_forward_count":forwards[0]}
        return checked_finite(record, "training step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "config", "checkpoint-dir", "output-dir", "selection-split", "formation-arm"):
        parser.add_argument("--"+flag, required=True)
    parser.add_argument("--formation-epoch", type=int, required=True)
    parser.add_argument("--future-rule", choices=("s", "g_match"), default="s")
    parser.add_argument("--reference-dir", "--step-reference-dir", dest="reference_dir")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--milestones", default="1,10,50,100")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke",action="store_true",help="Explicit short validation only; permits epoch1 SS reference")
    cli = parser.parse_args()
    if cli.seed not in (0,1,2) or cli.formation_epoch not in ((0,1,10,20,30,40,50) if cli.smoke else (0,10,20,30,40,50)):
        raise ValueError("Expected registered seed and formation checkpoint grid")
    if cli.formation_arm not in ("s", "s_lr15", "s_lr2", "g_raw", "g_match", "g_r025", "sum", "mean_lr2"):
        raise ValueError("Unknown formation arm")
    if cli.steps < 1 or cli.steps > 100 or (cli.future_rule=="g_match") != bool(cli.reference_dir):
        raise ValueError("Invalid step count or missing/unexpected SS reference")
    milestones = sorted(set(int(x) for x in cli.milestones.split(",") if x))
    if any(k < 1 or k > cli.steps for k in milestones):
        raise ValueError("Milestones must lie inside the single continuous trajectory")
    milestones = sorted(set(milestones + [cli.steps]))
    for key in ("source", "config", "checkpoint_dir", "output_dir", "selection_split"):
        setattr(cli, key, str(Path(getattr(cli,key)).resolve()))
    source, checkpoint, out = map(Path, (cli.source, cli.checkpoint_dir, cli.output_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing future trajectory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.data_manager import DataManager
    from utils import factory
    from utils.random_reproduce import json_safe
    from optimer_PerturabtionType.gam import GAM
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    started = time.time()
    summary = {"schema_version":1, "status":"running", "completed_steps":0, "task":1,
               "seed":cli.seed, "formation_arm":cli.formation_arm, "formation_epoch":cli.formation_epoch,
               "future_rule":cli.future_rule, "requested_steps":cli.steps, "milestones":milestones,
               "smoke":cli.smoke,
               "seed2_scope":"exploratory; this seed's full heldout was previously observed"}
    audit.atomic_json(out/"summary.json", summary)
    audit.atomic_json(out/"final_summary.json", summary)
    try:
        source_hashes = {str(p.relative_to(source)):audit.digest(p) for p in source.rglob("*.py")}
        files_before = checkpoint_files(checkpoint)
        manifest = json.loads((checkpoint/"manifest.json").read_text())
        if manifest.get("task") != 0 or manifest.get("epoch") != cli.formation_epoch or manifest.get("class_range") != [0,20]:
            raise ValueError("Native checkpoint metadata does not match declared Task0 origin")
        if (manifest.get("formation_arm") != cli.formation_arm or manifest.get("seed") != cli.seed or
                manifest.get("source_gam_sha256") != source_hashes["optimer_PerturabtionType/gam.py"]):
            raise ValueError("Formation checkpoint arm/seed/source hash does not match requested origin")
        training_state = torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
        checkpoint_cfg = training_state.get("config",{})
        saved_seed = checkpoint_cfg.get("seed")
        if isinstance(saved_seed,list) and len(saved_seed)==1:
            saved_seed = saved_seed[0]
        if (training_state.get("task") != 0 or training_state.get("epoch") != cli.formation_epoch or
                checkpoint_cfg.get("formation_arm") != cli.formation_arm or saved_seed != cli.seed):
            raise ValueError("Checkpoint training_state is not bound to declared manifest arm/seed/epoch")
        del training_state
        raw = json.loads(Path(cli.config).read_text())
        raw_seed = raw.get("seed")
        if isinstance(raw_seed, list):
            raw_seed = raw_seed[0] if len(raw_seed)==1 else None
        if raw_seed is not None and int(raw_seed) != cli.seed:
            raise ValueError("Input config seed differs from requested checkpoint seed")
        cfg = resolve_config(raw, cli)
        for key in ("dataset","class_shuffle","init_cls","increment","lora_rank","batch_size",
                    "convnet_type","backbone_type","model_name","class_order","class_order_override"):
            if key in cfg or key in checkpoint_cfg:
                if cfg.get(key) != checkpoint_cfg.get(key):
                    raise ValueError("Checkpoint/config model or data setting differs: " + key)
        device = torch.device(cli.device)
        audit.seed_all(cli.seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        dm = DataManager(cfg["dataset"], False, cli.seed, 20, 20, cfg)
        if dm.get_task_class_range(0) != (0,20) or dm.get_task_class_range(1) != (20,40):
            raise ValueError("Unexpected native class/task boundaries")
        cfg.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
        audit.atomic_json(out/"effective_config.json", json_safe(cfg))
        learner = factory.get_model(cfg["model_name"], cfg)
        learner._unwrap_network().backbone.to(device)
        learner.load_task_checkpoint(str(checkpoint), 0, dm)
        model = learner._unwrap_network().to(device)
        finite_model(model)
        loaded_hash = parameter_hash(model)
        if loaded_hash != manifest.get("model_sha256"):
            raise AssertionError("Native-restored parameter hash differs from checkpoint manifest")
        if json.loads(json.dumps(model.backbone.w_meta)) != json.loads((checkpoint/"lora_meta_0.json").read_text()):
            raise AssertionError("Native-restored LoRA metadata differs")
        for name in ("_fisher", "_fisher_matrix", "fisher", "fisher_matrix"):
            value = getattr(learner, name, None)
            if value is not None and (not isinstance(value, dict) or len(value)>0):
                raise AssertionError("Unexpected Fisher state loaded: " + name)
        datasets = {task:dm.get_dataset(np.arange(20*task,20*(task+1)),source="test",mode="test") for task in (0,1)}
        split_spec = json.loads(Path(cli.selection_split).read_text())
        subsets = {task:validate_split(split_spec, task, datasets[task].labels,datasets[task].images) for task in (0,1)}
        train = dm.get_dataset(np.arange(20,40), source="train", mode="train")
        if set(np.asarray(train.labels).tolist()) != set(range(20,40)):
            raise AssertionError("Future training contains non-Task1 data or misses a class")
        stream = EpochDataStream(train, cfg["batch_size"], cli.seed, dm.build_dataloader)
        for task in (0,1):
            train_n = len(train) if task==1 else len(dm.get_dataset(np.arange(20),source="train",mode="train"))
            for key,value in {"train_n":train_n,"train_batches_per_epoch":math.ceil(train_n/cfg["batch_size"])}.items():
                if key in split_spec["task%d"%task] and split_spec["task%d"%task][key] != value:
                    raise ValueError("Native training cardinality differs from frozen split: " + key)
        diagnostic_sets = {task:Subset(datasets[task],subsets[task]["diagnostic"]) for task in (0,1)}
        pre = evaluate(model, datasets[0], 0, 20, 20, device, subsets[0])
        pre["diagnostic"] = evaluate(model,diagnostic_sets[0],0,20,20,device)["diagnostic"]
        expansion = expand_head(model,20,40,cli.seed*100000+20000,device)
        learner._cur_task, learner._known_classes, learner._total_classes = 1,20,40
        learner._network = model
        baseline = {"task%d"%task:evaluate(model,datasets[task],20*task,20*(task+1),40,device,subsets[task]) for task in (0,1)}
        for task in (0,1):
            baseline["task%d"%task]["diagnostic"] = evaluate(model,diagnostic_sets[task],20*task,20*(task+1),40,device)["diagnostic"]
        for split in subsets[0]:
            # GEMM output shape may change reduction numerics. Raw old FC/backbone
            # equality is hard-gated above; old local CE is a measured sanity check.
            if abs(pre[split]["local"]["ce"]-baseline["task0"][split]["local"]["ce"]) > 1e-5:
                raise AssertionError("Head expansion unexpectedly changed old local predictions")
        identity = {"schema_version":1, "seed":cli.seed, "formation_arm":cli.formation_arm,
                    "formation_epoch":cli.formation_epoch, "future_rule":cli.future_rule,
                    "smoke":cli.smoke,
                    "source":str(source), "source_python_sha256":source_hashes,
                    "source_gam_sha256":source_hashes["optimer_PerturabtionType/gam.py"],
                    "trainer_sha256":audit.digest(__file__),
                    "audit_core_sha256":audit.digest(Path(__file__).with_name("audit_core.py")),
                    "effective_step_sha256":audit.digest(Path(__file__).with_name("effective_step.py")),
                    "config_sha256":audit.digest(cli.config), "checkpoint_dir":str(checkpoint),
                    "checkpoint_files_sha256":files_before, "origin_parameter_sha256":loaded_hash,
                    "selection_split_sha256":audit.digest(cli.selection_split), "selection_split":split_spec,
                    "expansion":expansion, "task1_train_samples":len(train),
                    "task1_batches_per_epoch":stream.batches_per_epoch, "batch_size":cfg["batch_size"],
                    "data_seed_rule":"seed*100000+10000+epoch (epoch 1-based)",
                    "training_rng_seed":cli.seed*100000+30000,
                    "class_order":[int(x) for x in dm._class_order],
                    "evaluation_batching":"diagnostic uses identical standalone fixed subset at K0/every step; full and reporting derive from full heldout logits",
                    "lr_schedule":{"start":.02,"eta_min":.002,"T_max":50,"unit":"completed native epochs"},
                    "future_rule_spec":"native GAM g1" if cli.future_rule=="s" else "native GAM g3 r=1, actual BA+FC norm matched to independent S50→S", 
                    "training_old_data":False, "restoration_policy":"native GAM additive unperturb; no saved-copy substitution",
                    "seed2_scope":summary["seed2_scope"]}
        reference = (StepReference(cli.reference_dir,cli.seed,cli.steps,identity["source_gam_sha256"],
                                  identity["selection_split_sha256"],cli.smoke) if cli.reference_dir else None)
        if reference:
            reference_meta = json.loads((Path(reference.directory)/"identity.json").read_text())
            for key in ("source_python_sha256","effective_step_sha256","audit_core_sha256","class_order",
                        "task1_train_samples","task1_batches_per_epoch","batch_size","data_seed_rule","training_rng_seed","lr_schedule"):
                if reference_meta.get(key) != identity[key]:
                    raise ValueError("Future SS reference protocol mismatch: " + key)
            if reference_meta.get("expansion",{}).get("new_rows_sha256") != expansion["new_rows_sha256"]:
                raise ValueError("Future SS reference used a different new-head initialization")
            identity["reference_dir"], identity["reference_files_sha256"] = reference.directory, reference.hashes
        audit.atomic_json(out/"identity.json", identity)
        evaluations = {"pre_expand":{"task0":pre}, "milestones":{"0":baseline}}
        audit.atomic_json(out/"evaluations.json", evaluations)
        paths = {space:PathIntegral(baseline["task0"]["diagnostic"][space]["ce"]) for space in ("local","all_seen")}
        stepper = NativeStep(learner,model,GAM,cli.future_rule)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base,T_max=50,eta_min=.002)
        audit.seed_all(identity["training_rng_seed"])
        step = 0
        max_match_error = 0.
        train_samples = 0
        for epoch in range(1, math.ceil(cli.steps/stream.batches_per_epoch)+1):
            nominal_lr = float(stepper.base.param_groups[0]["lr"])
            for batch, x, y, batch_identity in stream.epoch(epoch):
                step += 1
                target = None if reference is None else reference.target(step,epoch,batch,cli.seed,nominal_lr,batch_identity)
                result = stepper.step(x.to(device),y.to(device),nominal_lr,target)
                diagnostics = {"task%d"%task:evaluate(model,diagnostic_sets[task],20*task,20*(task+1),40,device)["diagnostic"] for task in (0,1)}
                path_row = {space:paths[space].append(step,diagnostics["task0"][space]["ce"]) for space in paths}
                row = {"step_index":step,"epoch":epoch,"batch":batch,"seed":cli.seed,"task":1,
                       "formation_arm":cli.formation_arm,"formation_epoch":cli.formation_epoch,
                       "future_rule":cli.future_rule,"lr_nominal":nominal_lr,"identity":batch_identity,
                       "batch_samples":int(y.numel()),"diagnostic":diagnostics,"old_loss_path":path_row,
                       "trainable_parameter_sha256":parameter_hash(model,trainable_only=True),**result}
                checked_finite(row,"step trace")
                with (out/"step_trace.jsonl").open("a") as handle:
                    handle.write(json.dumps(row,allow_nan=False)+"\n")
                    handle.flush()
                train_samples += int(y.numel())
                max_match_error = max(max_match_error,result["actual_match_relative_error"] or 0.)
                if step in milestones:
                    finite_model(model)
                    evaluations["milestones"][str(step)] = {"task%d"%task:evaluate(model,datasets[task],20*task,20*(task+1),40,device,subsets[task]) for task in (0,1)}
                    for task in (0,1):
                        evaluations["milestones"][str(step)]["task%d"%task]["diagnostic"] = diagnostics["task%d"%task]
                    audit.atomic_json(out/"evaluations.json",evaluations)
                summary.update(completed_steps=step,last_epoch=epoch,last_batch=batch,
                               last_diagnostic=diagnostics,path_integrals=path_row,
                               training_samples=train_samples,max_actual_norm_match_relative_error=max_match_error)
                audit.atomic_json(out/"summary.json",summary)
                audit.atomic_json(out/"final_summary.json",summary)
                print(json.dumps({"event":"future_step", "step":step,"epoch":epoch,"batch":batch,
                                  "lr":nominal_lr,"train_ce":result["train_loss"],
                                  "old_local_ce":diagnostics["task0"]["local"]["ce"],
                                  "new_local_ce":diagnostics["task1"]["local"]["ce"]}),flush=True)
                if batch+1 == stream.batches_per_epoch:
                    scheduler.step()
                if step == cli.steps:
                    break
            if step == cli.steps:
                break
        if checkpoint_files(checkpoint) != files_before:
            raise AssertionError("Original checkpoint files changed during future fork")
        if {p:audit.digest(source/p) for p in source_hashes} != source_hashes:
            raise AssertionError("Frozen source changed during future trajectory")
        if reference and {name:audit.digest(Path(reference.directory)/name) for name in reference.hashes} != reference.hashes:
            raise AssertionError("Reference trajectory changed during matched continuation")
        summary.update(status="completed",elapsed_seconds=time.time()-started,
                       completed_full_epochs=cli.steps//stream.batches_per_epoch,
                       completed_fractional_epochs=cli.steps/stream.batches_per_epoch,
                       checkpoint_unchanged=True,source_unchanged=True,
                       identity_sha256=audit.digest(out/"identity.json"),
                       step_trace_sha256=audit.digest(out/"step_trace.jsonl"),
                       evaluations_sha256=audit.digest(out/"evaluations.json"),
                       final_trainable_parameter_sha256=parameter_hash(model,True),
                       final_evaluation=evaluations["milestones"][str(cli.steps)],
                       milestones_completed=sorted(int(k) for k in evaluations["milestones"]),
                       counters={"training_steps":cli.steps,"ce_gradient_passes":5*cli.steps,
                                 "fisher_estimation_passes":0,"weight_merge_calls":0,
                                 "task_end_hooks":0},
                       match_passed=True if reference else None)
    except Exception as exc:
        summary.update(status="failed",error=repr(exc),elapsed_seconds=time.time()-started)
        raise
    finally:
        audit.atomic_json(out/"summary.json",summary)
        audit.atomic_json(out/"final_summary.json",summary)


if __name__ == "__main__":
    main()
