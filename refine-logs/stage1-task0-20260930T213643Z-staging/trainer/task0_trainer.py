#!/usr/bin/env python3
"""Stage 1 Task-0 wrapper: native frozen learner/data/epoch loop with audit hooks."""
import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from types import MethodType, SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
import audit_core as audit


@contextlib.contextmanager
def observational(model):
    rng = audit.rng_get()
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for b, value in buffers:
                b.copy_(value)
        for m, mode in modes:
            m.training = mode
        audit.rng_set(rng)
        assert audit.rng_equal(rng, audit.rng_get())


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def parameter_hash(model, frozen_only=False):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if frozen_only and param.requires_grad:
            continue
        h.update(name.encode())
        h.update(str(tuple(param.shape)).encode())
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def vector_grads(params):
    return audit.flatten([torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params])


def gradient_geometry(grads):
    g1, g3 = grads["g1"], grads["g3"]
    denominator = audit.dot(g1, g1)
    if denominator <= 1e-20:
        return {"identified": False, "reason": "g1_squared_norm_too_small",
                "gradient_norms": {k: audit.norm(v) for k, v in grads.items()}}
    a = audit.dot(g3, g1) / denominator
    residual = g3 - a * g1
    return {"identified": True, "a": a, "a_clipped": False, "r_perp_norm": audit.norm(residual),
            "r_perp_over_g3": audit.norm(residual) / max(audit.norm(g3), 1e-20),
            "orthogonality_dot": audit.dot(residual, g1),
            "gradient_norms": {k: audit.norm(v) for k, v in grads.items()},
            "cos_g0_g1": audit.cosine(grads["g0"], g1), "cos_g0_g3": audit.cosine(grads["g0"], g3),
            "cos_g1_g3": audit.cosine(g1, g3)}


def isolated_gam(model, geo, x, y, radius, GAM):
    """Read-only first-batch diagnostic for the independent two-gradient SAM arm."""
    snap = audit.Snapshot(model)
    base = torch.optim.SGD(geo.params, lr=0.0)
    opt = GAM(geo.params, base_optimizer=base, model=model, grad_rho=.05, grad_norm_rho=radius,
              adaptive=False, args=SimpleNamespace(grad_beta_1=1., grad_beta_2=1., grad_beta_3=1., grad_gamma=0.))
    points = []
    try:
        model.train()
        clean = F.cross_entropy(audit.logits(model, x), y)
        torch.autograd.grad(clean, geo.params, allow_unused=True)
        del clean
        def closure():
            points.append(geo.displacement_metrics())
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out["logits"], y)
            loss.backward()
            return out, loss.detach()
        opt.step(closure)
        grads = {"g%d" % k: audit.flatten([opt.state[p]["g_%d" % k].detach().clone() for p in geo.params]) for k in range(4)}
        return grads, points
    finally:
        snap.restore()
        snap.check()


def select_final_gradient(opt, direction):
    if direction == "ce":
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.copy_(opt.state[p]["g_0"])


def worker_init(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class EpochLoader:
    """Independent per-epoch sampler/augmentation RNG, including num_workers=0."""
    def __init__(self, original, controller):
        self.original, self.c = original, controller
        self.dataset = original.dataset
        self.epoch = 0

    def __len__(self):
        n = len(self.original)
        return min(n, self.c.cli.max_batches) if self.c.cli.max_batches else n

    def __iter__(self):
        self.epoch += 1
        self.c.begin_epoch(self.epoch)
        seed = self.c.seed * 100000 + self.epoch
        generator = torch.Generator().manual_seed(seed)
        loader = self.c.dm.build_dataloader(self.dataset, batch_size=self.c.config["batch_size"],
            shuffle=True, num_workers=self.c.config.get("train_num_workers", 8),
            generator=generator, worker_init_fn=worker_init, persistent_workers=False)
        training_rng = audit.rng_get()
        audit.seed_all(seed)
        data_rng = audit.rng_get()
        audit.rng_set(training_rng)
        iterator = None
        order = hashlib.sha256()
        for index in range(len(self)):
            training_rng = audit.rng_get()
            audit.rng_set(data_rng)
            try:
                if iterator is None:
                    iterator = iter(loader)
                batch = next(iterator)
                data_rng = audit.rng_get()
            finally:
                audit.rng_set(training_rng)
            ids, inputs, targets = batch
            order.update(ids.contiguous().numpy().tobytes())
            self.c.batch_index = index
            if index == 0:
                self.c.first_batch_hashes = {"input_sha256": audit.tensor_digest(inputs),
                    "target_sha256": audit.tensor_digest(targets), "indices": ids.tolist(), "data_seed": seed}
            yield batch
        self.c.sampler_digest = order.hexdigest()


class Controller:
    def __init__(self, learner, dm, config, cli):
        self.learner, self.dm, self.config, self.cli = learner, dm, config, cli
        self.seed, self.direction = int(config["seed"]), config["stage1_direction"]
        self.out = Path(cli.output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.model = learner._unwrap_network()
        self.optimizer = self.scheduler = self.test_loader = None
        self.epoch = self.batch_index = 0
        self.rows, self.records, self.eval_cache = [], [], {}
        self.counters = {"training_batches": 0, "training_samples": 0, "training_ce_gradient_passes": 0,
                         "isolated_diagnostic_ce_gradient_passes": 0}
        self.started = time.time()

    def evaluate(self, epoch):
        if epoch in self.eval_cache:
            return self.eval_cache[epoch]
        with observational(self.model), torch.no_grad():
            self.model.eval()
            loss = correct = count = 0
            class_counts = torch.zeros(20, dtype=torch.int64)
            class_correct = torch.zeros(20, dtype=torch.int64)
            for _, x, y in self.test_loader:
                x, y = x.to(self.learner._device), y.to(self.learner._device)
                out = audit.logits(self.model, x)
                loss += float(F.cross_entropy(out, y, reduction="sum"))
                correct += int(out.argmax(-1).eq(y).sum())
                class_counts += torch.bincount(y.cpu(), minlength=20)
                class_correct += torch.bincount(y[out.argmax(-1).eq(y)].cpu(), minlength=20)
                count += y.numel()
            present = class_counts > 0
            class_acc = 100.0 * class_correct.double() / class_counts.clamp_min(1).double()
            result = {"ce": loss / count, "accuracy": 100.0 * correct / count, "samples": count,
                      "macro_accuracy": float(class_acc[present].mean()),
                      "per_class_counts": class_counts.tolist(), "per_class_correct": class_correct.tolist(),
                      "per_class_accuracy": class_acc.tolist(), "classes_present": int(present.sum())}
        self.eval_cache[epoch] = result
        return result

    def save_checkpoint(self, epoch):
        directory = self.out / "checkpoints" / ("epoch_%03d" % epoch)
        directory.mkdir(parents=True, exist_ok=True)
        with observational(self.model):
            backbone = self.model.backbone
            # Do not call save_lora_parameters: the native method increments task_id.
            torch.save([copy.deepcopy(m).cpu() for m in backbone.w_As], directory / "lora_w_a_0.pt")
            torch.save([copy.deepcopy(m).cpu() for m in backbone.w_Bs], directory / "lora_w_b_0.pt")
            audit.atomic_json(directory / "lora_meta_0.json", backbone.w_meta)
            state = cpu_tree(self.model.fc.state_dict())
            torch.save(state, directory / "fc_state_0.pt")
            torch.save(state["weight"], directory / "CLs_weight0.pt")
            if "bias" in state:
                torch.save(state["bias"], directory / "CLs_bias0.pt")
            torch.save({"epoch": epoch, "task": 0, "optimizer": cpu_tree(self.optimizer.state_dict()),
                        "base_optimizer": cpu_tree(self.optimizer.base_optimizer.state_dict()),
                        "scheduler": cpu_tree({k: v for k, v in self.scheduler.state_dict().items() if not callable(v)}), "rng": audit.rng_get(),
                        "epoch_data_seed_rule": "seed * 100000 + epoch (1-based)",
                        "config": self.config}, directory / "training_state.pt")
            audit.atomic_json(directory / "manifest.json", {"epoch": epoch, "task": 0,
                "model_sha256": parameter_hash(self.model), "optimizer_class": type(self.optimizer).__name__,
                "backbone_task_id": backbone.task_id, "class_range": [0, 20],
                "note": "Cumulative single current adapter; load using load_task_checkpoint(dir, 0, dm)."})

    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.train_loss_sum = self.correct = self.samples = self.batches = 0
        self.first_batch_hashes, self.sampler_digest = {}, None
        self.lr_used = [float(g["lr"]) for g in self.optimizer.param_groups]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.epoch_start = time.time()

    def finish_epoch(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_seconds = time.time() - self.epoch_start
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        evaluation_start = time.time()
        evaluation = self.evaluate(self.epoch)
        eval_seconds = time.time() - evaluation_start
        row = {"epoch": self.epoch, "task": 0, "train_ce": self.train_loss_sum / self.samples,
               "train_accuracy": 100.0 * self.correct / self.samples, "train_samples": self.samples,
               "batches": self.batches, "eval_ce": evaluation["ce"], "eval_accuracy": evaluation["accuracy"],
               "eval_samples": evaluation["samples"], "lr_used": self.lr_used,
               "eval_macro_accuracy": evaluation["macro_accuracy"],
               "eval_per_class_counts": evaluation["per_class_counts"],
               "eval_per_class_correct": evaluation["per_class_correct"],
               "lr_after_scheduler": [float(g["lr"]) for g in self.optimizer.param_groups],
               "training_seconds_including_sampled_geometry": train_seconds,
               "eval_seconds": eval_seconds, "training_samples_per_second": self.samples / train_seconds,
               "peak_gpu_allocated_bytes": peak, "first_batch": self.first_batch_hashes,
               "sampler_order_sha256": self.sampler_digest}
        if audit.nonfinite_paths(row):
            raise FloatingPointError("Nonfinite epoch metrics")
        self.rows.append(row)
        audit.atomic_json(self.out / "metrics.json", {"epochs": self.rows, "counters": self.counters})
        if self.epoch % 10 == 0 or self.epoch == self.config["init_epoch"]:
            self.save_checkpoint(self.epoch)
        print(json.dumps({"event": "epoch_complete", **row}), flush=True)

    def record_batch(self, out, loss, log, targets):
        n = targets.numel()
        self.train_loss_sum += float(log["ce_clean"]) * n
        self.correct += int(out.argmax(-1).eq(targets).sum())
        self.samples += n
        self.batches += 1
        self.counters["training_batches"] += 1
        self.counters["training_samples"] += n
        self.counters["training_ce_gradient_passes"] += 2 if self.direction == "true_sam" else 5
        if float(log.get("ewc", 0)) != 0 or float(log.get("delta_w_penalty", 0)) != 0:
            raise AssertionError("Unexpected regularization in Stage 1")

    def record_geometry(self, geo, grads, points, final_gradient, closure_count, source):
        data = {"epoch": self.epoch, "batch": self.batch_index, "task": 0, "source": source,
                "geometry": gradient_geometry(grads), "probe_points": points,
                "actual_update": geo.displacement_metrics(), "final_gradient_norm": audit.norm(final_gradient),
                "training_closure_count": closure_count,
                "per_layer_gradient_norms": {k: geo.layer_norms(v) for k, v in grads.items()}}
        expected = {"ce": grads["g0"], "s": grads["g1"], "true_sam": grads["g1"],
                    "g": grads["g3"], "sum": grads["g1"] + grads["g3"],
                    "mean": (grads["g1"] + grads["g3"]) / 2}[self.direction]
        data["selected_gradient_relative_error"] = audit.norm(final_gradient - expected) / max(audit.norm(expected), 1e-20)
        actual_delta = geo.parameter_delta()
        expected_delta = -self.lr_used[0] * expected
        data["actual_step_vs_selected_relative_error"] = audit.norm(actual_delta - expected_delta) / max(audit.norm(expected_delta), 1e-20)
        data["base_sgd_audit"] = self.sample_base_audit
        if data["selected_gradient_relative_error"] > 1e-4 or not self.sample_base_audit["passed"]:
            raise AssertionError("Actual update does not match selected gradient: " + repr(data))
        if audit.nonfinite_paths(data):
            raise FloatingPointError("Nonfinite sampled geometry")
        self.records.append(data)
        audit.atomic_json(self.out / "geometry.json", {"records": self.records})

    def instrument_optimizer(self, opt):
        self.optimizer = opt
        original_base_step = opt.base_optimizer.step
        def base_step(_base, *args, **kwargs):
            sampled = self.batch_index == 0
            if sampled:
                params = [p for group in opt.param_groups for p in group["params"]]
                before = [p.detach().clone() for p in params]
                predicted = []
                for group in opt.param_groups:
                    if float(group.get("momentum", 0)) != 0 or float(group.get("weight_decay", 0)) != 0:
                        raise AssertionError("SGD audit is specific to zero momentum/weight decay")
                    for p in group["params"]:
                        value = p.detach().clone()
                        if p.grad is not None:
                            value.add_(p.grad, alpha=-float(group["lr"]))
                        predicted.append(value)
                self.sample_final_gradient = vector_grads(params)
                roundtrip = self.active_geo.parameter_delta()
                self.sample_base_audit = {"probe_roundtrip_parameter_norm": audit.norm(roundtrip),
                                          "probe_roundtrip_max_abs": float(roundtrip.abs().max())}
            result = original_base_step(*args, **kwargs)
            if sampled:
                max_error = max(float((p.detach() - expected).abs().max()) for p, expected in zip(params, predicted))
                passed = all(torch.allclose(p.detach(), expected, rtol=1e-6, atol=1e-7) for p, expected in zip(params, predicted))
                self.sample_base_audit.update(passed=passed, predicted_after_max_abs_error=max_error,
                    base_step_parameter_norm=audit.norm(audit.flatten([p.detach() - old for p, old in zip(params, before)])),
                    note="Prediction starts after native unperturb and uses actual final gradient in parameter dtype.")
            return result
        opt.base_optimizer.step = MethodType(base_step, opt.base_optimizer)
        if self.direction == "true_sam":
            return opt
        original_decompose = opt.gradient_decompose
        def decompose(_opt, *args, **kwargs):
            result = original_decompose(*args, **kwargs)
            with torch.no_grad():
                select_final_gradient(_opt, self.direction)
            return result
        opt.gradient_decompose = MethodType(decompose, opt)
        original_step = opt.step
        def step(_opt, closure=None):
            sampled = self.batch_index == 0
            geo = audit.Geometry(self.model) if sampled else None
            self.active_geo = geo
            points, count = [], 0
            def wrapped_closure():
                nonlocal count
                count += 1
                if sampled:
                    points.append(geo.displacement_metrics())
                return closure()
            output = original_step(closure=wrapped_closure)
            if count != 4:
                raise AssertionError("Main causal arm must execute four GAM closures")
            if sampled:
                grads = {"g%d" % k: audit.flatten([opt.state[p]["g_%d" % k].detach() for p in geo.params]) for k in range(4)}
                self.record_geometry(geo, grads, points, self.sample_final_gradient, count, "native_training_GAM_state")
            self.active_geo = None
            return output
        opt.step = MethodType(step, opt)
        return opt

    def true_sam_batch(self, optimizer, inputs, targets, class_offset):
        if class_offset != 0:
            raise AssertionError("Stage 1 is Task 0 only")
        from optimer_PerturabtionType.gam import GAM
        from optimer_PerturabtionType.util import enable_running_stats, disable_running_stats
        sampled = self.batch_index == 0
        geo = audit.Geometry(self.model) if sampled else None
        self.active_geo = geo
        if sampled:
            grads, gam_points = isolated_gam(self.model, geo, inputs, targets,
                                            float(self.config["gam_grad_norm_rho"]), GAM)
            self.counters["isolated_diagnostic_ce_gradient_passes"] += 5
        enable_running_stats(self.model)
        optimizer.zero_grad()
        out = self.model(inputs)
        loss = F.cross_entropy(out["logits"], targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        disable_running_stats(self.model)
        perturbed = self.model(inputs)
        perturbed_loss = F.cross_entropy(perturbed["logits"], targets)
        perturbed_loss.backward()
        final = vector_grads(geo.params) if sampled else None
        optimizer.second_step(zero_grad=False)
        enable_running_stats(self.model)
        if sampled:
            self.record_geometry(geo, grads, gam_points, final, 2, "isolated_GAM_diagnostic_plus_native_SAM_training")
        self.active_geo = None
        ce = float(loss.detach())
        return out["logits"].detach(), ce, {"ce_clean": ce, "gam": float(perturbed_loss.detach()),
                                           "ewc": 0., "delta_w_penalty": 0., "lambda_ewc": 0.}

    def install(self):
        learner = self.learner
        original_build = learner._build_optimizer
        def build(_self, params, stage):
            return self.instrument_optimizer(original_build(params, stage))
        learner._build_optimizer = MethodType(build, learner)
        original_batch = learner._step_batch_with_log
        def batch(_self, optimizer, inputs, targets, class_offset):
            result = (self.true_sam_batch(optimizer, inputs, targets, class_offset)
                      if self.direction == "true_sam" else original_batch(optimizer, inputs, targets, class_offset))
            self.record_batch(*result, targets)
            return result
        learner._step_batch_with_log = MethodType(batch, learner)
        original_run = learner._run_regularized_train
        def run(_self, train_loader, test_loader, optimizer, scheduler, *, epochs, class_offset):
            if class_offset != 0 or learner._cur_task != 0:
                raise AssertionError("Stage 1 permits exactly Task 0")
            self.model = learner._unwrap_network()
            self.test_loader, self.scheduler = test_loader, scheduler
            for name, module in self.model.named_modules():
                if (isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0) or float(getattr(module, "drop_prob", 0.) or 0.) > 0:
                    raise AssertionError("Stochastic model layer is outside this paired protocol: " + name)
            self.identity = {"initial_model_sha256": parameter_hash(self.model),
                "frozen_origin_sha256": parameter_hash(self.model, frozen_only=True),
                "trainable": [{"name": n, "shape": list(p.shape)} for n, p in self.model.named_parameters() if p.requires_grad],
                "class_order": list(map(int, self.dm._class_order)), "task_range": [0, 20],
                "sampler_seed_rule": "seed * 100000 + epoch (1-based)",
                "native_scheduler": type(scheduler).__name__, "scheduler_initial": scheduler.state_dict()}
            audit.atomic_json(self.out / "identity.json", self.identity)
            self.save_checkpoint(0)
            audit.atomic_json(self.out / "epoch0_eval.json", self.evaluate(0))
            original_scheduler_step = scheduler.step
            def schedule_step(*args, **kwargs):
                result = original_scheduler_step(*args, **kwargs)
                self.finish_epoch()
                return result
            scheduler.step = schedule_step
            # Native five-epoch accuracy calls reuse our RNG-isolated full test evaluation.
            learner._compute_accuracy = MethodType(lambda _s, model, loader: self.evaluate(self.epoch)["accuracy"], learner)
            learner._compute_task_local_accuracy = MethodType(lambda _s, model, loader, lo, hi: self.evaluate(self.epoch)["accuracy"], learner)
            return original_run(EpochLoader(train_loader, self), test_loader, optimizer, scheduler,
                                epochs=epochs, class_offset=class_offset)
        learner._run_regularized_train = MethodType(run, learner)


def resolve_config(raw, cli):
    cfg = copy.deepcopy(raw)
    direction = cfg.get("stage1_direction", "")
    if direction not in {"ce", "s", "g", "sum", "mean", "true_sam"}:
        raise ValueError("stage1_direction must be ce|s|g|sum|mean|true_sam")
    seed = cfg["seed"][0] if isinstance(cfg.get("seed"), list) else cfg.get("seed", 2)
    if int(seed) != 2:
        raise ValueError("Stage 1 development fixes seed=2")
    for key, expected in {"dataset": "imageneta", "class_shuffle": False, "init_cls": 20,
                          "increment": 20, "lora_rank": 10, "batch_size": 128}.items():
        if cfg.get(key, expected) != expected:
            raise ValueError("Stage 1 fixed protocol violation: " + key)
        cfg[key] = expected
    beta1, beta3 = {"ce": (1., 0.), "s": (1., 0.), "g": (0., 1.),
                    "sum": (1., 1.), "mean": (.5, .5), "true_sam": (1., 0.)}[direction]
    cfg.update(seed=2, device=[torch.device("cuda:0")], init_epoch=50, epochs=50, max_train_tasks=1,
               init_lr=.04, lrate=.02, init_momentum=0., momentum=0., init_weight_decay=0., weight_decay=0.,
               optimizer="sgd", gam_base_optimizer="sgd", scheduler="cosine", lambda_flat=1.,
               optimizer_type="sam" if direction == "true_sam" else "gam", sam_rho=.05, sam_adaptive=False,
               gam_grad_rho=.05, gam_grad_gamma=0., gam_grad_beta_1=beta1, gam_grad_beta_2=1.,
               gam_grad_beta_3=beta3, gam_adaptive=False, ewc_lambda=0., ewc_skip_task_fisher=True,
               in_task_weight_merge=False, task_endpoint_weight_merge=False, approximate_fisher_in_task=False,
               in_task_fisher_omega=0., in_task_weight_merge_omega=0., ewc_fisher_damping_alpha=0.,
               ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False, ewc_dual_ascent=False,
               compute_full_fisher=False, posterior_state_save=False, mechanism_anchor_eval=False,
               mechanism_future_surrogate_eval=False, as1_delta_w_penalty=False, fisher_split=False,
               memory_size=0, memory_per_class=0, mechanism_eval=False, mechanism_grad_stats=False,
               endpoint_eval=False, task0_trajectory_eval=False, task0_fisher_recompute_eval=False)
    cfg.setdefault("gam_grad_norm_rho", 1.)
    for key in ("optimizer_type_by_task", "fc_lrate", "init_fc_lrate", "resume_from_task", "resume_from_checkpoint_dir"):
        cfg.pop(key, None)
    if cli.max_epochs:
        cfg["init_epoch"] = cfg["epochs"] = cli.max_epochs
    cfg["filepath"] = str(Path(cli.output_dir).resolve() / "final_native_checkpoint") + os.sep
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=0, help="smoke only; 0 retains 50")
    parser.add_argument("--max-batches", type=int, default=0, help="smoke only; 0 uses all samples")
    cli = parser.parse_args()
    cli.source, cli.config, cli.output_dir = [str(Path(x).resolve()) for x in (cli.source, cli.config, cli.output_dir)]
    source = Path(cli.source)
    if not (source / "src/trainer.py").is_file() or not (source / "models_LoRAbasedCL/baseLoRA.py").is_file():
        raise FileNotFoundError("Frozen source layout does not match")
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from utils.data_manager import DataManager
    from utils.random_reproduce import set_random, json_safe
    from utils import factory
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    raw = json.loads(Path(cli.config).read_text())
    cfg = resolve_config(raw, cli)
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_random(2)
    torch.cuda.set_device(0)
    dm = DataManager(cfg["dataset"], False, 2, 20, 20, cfg)
    cfg.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
    audit.atomic_json(out / "effective_config.json", json_safe(cfg))
    learner = factory.get_model(cfg["model_name"], cfg)
    controller = Controller(learner, dm, cfg, cli)
    controller.install()
    summary = {"status": "running", "completed_epochs": 0, "task": 0, "trained_tasks": [0],
               "direction": cfg["stage1_direction"], "r": cfg["gam_grad_norm_rho"], "seed": 2,
               "smoke": bool(cli.max_epochs or cli.max_batches),
               "trainer_sha256": audit.digest(__file__), "source_gam_sha256": audit.digest(source / "optimer_PerturabtionType/gam.py")}
    audit.atomic_json(out / "final_summary.json", summary)
    try:
        learner.incremental_train(dm)
        if len(controller.rows) != cfg["init_epoch"]:
            raise AssertionError("Missing epoch callbacks")
        summary.update(status="completed", completed_epochs=len(controller.rows), final=controller.rows[-1],
                       counters=controller.counters, elapsed_seconds=time.time() - controller.started,
                       initial_identity=controller.identity, checkpoint_epochs=[0] + [i for i in range(1, cfg["init_epoch"] + 1) if i % 10 == 0 or i == cfg["init_epoch"]])
        summary["finite_check"] = {"passed": not audit.nonfinite_paths(controller.rows + controller.records)}
        summary["checkpoint_checks"] = {str(e): all((out / "checkpoints" / ("epoch_%03d" % e) / name).is_file()
            for name in ("lora_w_a_0.pt", "lora_w_b_0.pt", "lora_meta_0.json", "fc_state_0.pt", "training_state.pt", "manifest.json"))
            for e in summary["checkpoint_epochs"]}
        summary["task_check"] = {"cur_task": learner._cur_task, "total_classes": learner._total_classes,
                                 "passed": learner._cur_task == 0 and learner._total_classes == 20}
        if not summary["finite_check"]["passed"] or not all(summary["checkpoint_checks"].values()) or not summary["task_check"]["passed"]:
            raise AssertionError("Final acceptance checks failed")
    except Exception as exc:
        summary.update(status="failed", error=repr(exc), completed_epochs=len(controller.rows))
        raise
    finally:
        audit.atomic_json(out / "final_summary.json", summary)


if __name__ == "__main__":
    main()
