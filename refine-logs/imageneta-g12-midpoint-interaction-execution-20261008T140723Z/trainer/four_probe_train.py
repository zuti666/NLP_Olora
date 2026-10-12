#!/usr/bin/env python3
"""Frozen ImageNet-A lifecycle with native three/four-read midpoint directions.

Only the custom ordinary-CE direction executor can perform a training update.
The inherited lifecycle supplies paired data, task expansion, evaluation, and
native checkpoint writing; its legacy five-CE GAM batch method is replaced.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import PROTOCOL, expected_budget
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
from direction_step import FourProbeStep, direction_name, validate_factors
from effective_step import EffectiveStep
from state_transaction import backend_context, State

FACTOR_FIELDS = ("update_direction",)


def deterministic_execution_flags():
    """Fail closed if the paired-run execution contract changes at runtime."""
    flags = dict(deterministic_enabled=torch.are_deterministic_algorithms_enabled(),
                 deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                 cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    if flags != dict(deterministic_enabled=True, deterministic_warn_only=False,
                     cublas_workspace_config=":4096:8"):
        raise RuntimeError("Strict CUDA deterministic execution required: "+repr(flags))
    return flags


def configure_deterministic_execution():
    # The workspace must be configured by the job environment before CUDA starts.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must already equal :4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    return deterministic_execution_flags()


@contextmanager
def observational(model):
    state = State(model)
    try:
        yield
    finally:
        state.restore()
        if not all(state.check().values()):
            raise AssertionError("Evaluation did not restore model/cache/RNG state")


def resolve_config(raw, cli):
    if raw.get("cuda_deterministic_algorithms") is not True or raw.get("cuda_deterministic_warn_only") is not False:
        raise ValueError("Explicit cuda_deterministic_algorithms=true and cuda_deterministic_warn_only=false required")
    direction = raw["update_direction"]
    mode = raw["step_mode"]
    backend = raw.get("attention_backend", "native")
    if validate_factors(dict(raw, direction=direction)) != direction:
        raise ValueError("Direction alias mismatch")
    branch = "P"
    if raw.get("component_rule") != direction:
        raise ValueError("component_rule must explicitly match update_direction")
    if mode != "natural" or backend != "native":
        raise ValueError("Natural SGD and native attention required")
    radius = float(raw["gam_grad_norm_rho"])
    rho = float(raw.get("gam_grad_rho", .05))
    seed = raw.get("seed", [2])
    seed = seed[0] if isinstance(seed, list) and len(seed) == 1 else seed
    if seed != 2 or rho != .05 or radius != 1. or raw.get("lora_rank") != 10:
        raise ValueError("Protocol requires seed=2, rank10, rho=.05, r=1")
    if raw.get("protocol") != PROTOCOL or raw.get("extra_diagnostic_gradient_calls") != 0:
        raise ValueError("Explicit midpoint protocol and zero extra diagnostic gradients required")
    surrogate = copy.deepcopy(raw)
    surrogate.update(component_rule="u0", component_scale="natural")
    cfg = native.resolve_config(surrogate, cli)
    cfg.update(probe_branch=branch, update_direction=direction, step_mode=mode,
               attention_backend=backend, component_rule=direction,
               component_scale=mode, optimizer_tag_override="midpoint_interaction_strict_v1",
               gam_grad_rho=rho, gam_grad_norm_rho=radius, lora_rank=10,
               compute_nme_class_means=False,
               protocol=PROTOCOL,
               cuda_deterministic_algorithms=True, cuda_deterministic_warn_only=False,
               deterministic_enabled=True, deterministic_warn_only=False,
               cublas_workspace_config=":4096:8",
               compute_g3_probe=direction == "sum13", extra_diagnostic_gradient_calls=0,
               second_probe_side_diagnostics=False,
               in_task_path_normalize_snapshots=False)
    for key in ("first_probe_sign", "second_probe_sign", "second_difference_sign", "second_difference_weight"):
        cfg.pop(key, None)
    cfg["expected_budget"] = expected_budget(cfg)
    return cfg


class Controller(native.Controller):
    def __init__(self, learner, cfg, cli):
        super().__init__(learner, cfg, cli)
        self.direction = cfg["update_direction"]
        self.branch = cfg["probe_branch"]
        self.backend = cfg["attention_backend"]
        self.factor_config = dict(direction=self.direction)
        self.budget = expected_budget(cfg)
        self.counters.update(training_hvp_passes=0, training_hvp_reference_gradient_passes=0,
                             training_create_graph_passes=0, training_forward_count=0,
                             training_sgd_updates=0, training_g3_gradient_passes=0,
                             training_opposite_g1_gradient_passes=0,
                             training_global_projection_calls=0,
                             training_g1_positive_gradient_passes=0, training_g1_negative_gradient_passes=0,
                             training_first_positive_gradient_passes=0, training_first_negative_gradient_passes=0,
                             training_local_positive_gradient_passes=0, training_local_negative_gradient_passes=0,
                             training_g3_positive_gradient_passes=0, training_g3_negative_gradient_passes=0,
                             diagnostic_gradient_calls=0, diagnostic_forward_count=0,
                             diagnostic_sgd_updates=0, diagnostic_snapshots=0,
                             diagnostic_g3_positive_gradient_calls=0, diagnostic_g3_negative_gradient_calls=0)
        self.counters.update({"training_"+key+"_gradient_passes": 0 for key in ("g0", "g1", "g2", "gm")})
        self.last_record = None
        self.observations = []
        self.diagnostic_trace = self.out/"side_diagnostics.jsonl"
        self.diagnostic_trace.touch(exist_ok=False)
        self.context_trace = self.out/"context_diagnostics.jsonl"
        self.context_handle = self.context_trace.open("x")
        self.diagnostic_tasks = set()
        self.pairing_snapshot = None

    def install(self):
        l = self.learner
        native.install_forbidden_work_guards(l, self.counters)

        def build(_self, params, stage):
            model = l._unwrap_network()
            self.geometry = EffectiveStep.from_model(model)
            params = list(params)
            if {id(p) for p in params} != {id(p) for p in self.geometry.params}:
                raise AssertionError("Optimizer omitted trainable LoRA/FC parameters")
            self.optimizer = torch.optim.SGD(params, lr=.04 if stage == "init" else .02,
                                             momentum=0, weight_decay=0)
            # Native checkpoint writer expects a base_optimizer; this is the same SGD.
            self.optimizer.base_optimizer = self.optimizer
            scfg = dict(self.factor_config, branch=self.branch, direction=self.direction,
                        rho=float(self.cfg["gam_grad_rho"]),
                        radius=float(self.cfg["gam_grad_norm_rho"]),
                        eps=1e-12, backend=self.backend)
            self.stepper = FourProbeStep(model, params, self.geometry, scfg)
            old_count = 20*self.task
            fc = model.fc.state_dict()
            inherited = all(torch.equal(fc[k][:old_count], v.to(fc[k])) for k,v in self.old_fc.items())
            if not inherited:
                raise AssertionError("Classifier expansion changed existing rows")
            identity = dict(task=self.task, seed=self.seed, probe_branch=self.branch,
                            update_direction=self.direction, attention_backend=self.backend,
                            rho=self.stepper.rho, radius=self.stepper.radius,
                            class_slice=[old_count,old_count+20], fc_shape=list(model.fc.weight.shape),
                            old_fc_rows_inherited_exact=True,
                            new_fc_rows_sha256={k:audit.tensor_digest(v[old_count:]) for k,v in fc.items()},
                            task_initial_parameter_sha256=base.parameter_hash(model),
                            frozen_origin_sha256=base.parameter_hash(model,True),
                            data_seed_rule="seed*100000+task*1000+epoch",
                            forward_rng_rule="replay R0 at each point; advance to R1 from g0 once",
                            **self.factor_config)
            self.task_identities.append(identity)
            audit.atomic_json(self.out/"task_identities.json", self.task_identities)
            return self.optimizer
        l._build_optimizer = MethodType(build, l)

        def batch(_self, optimizer, inputs, targets, class_offset):
            execution_flags = deterministic_execution_flags()
            if class_offset != 20*self.task or bool(((targets<class_offset)|(targets>=class_offset+20)).any()):
                raise AssertionError("Incorrect current-task CE range")
            model = l._unwrap_network()
            def closure(x,y):
                output = model(x)["logits"]
                loss, _, _ = l._task_loss(output,y,class_offset)
                return loss,output
            lr = float(optimizer.param_groups[0]["lr"])
            forwards = [0]
            def count_forward(*_args):
                forwards[0] += 1
            hook = model.register_forward_hook(count_forward)
            try:
                with backend_context(model,self.backend):
                    result = self.stepper.compute(inputs,targets,closure,lr)
            finally:
                hook.remove()
            if forwards[0] != self.budget["train_ce_per_update"]:
                raise AssertionError("Training CE forwards differ from explicit estimator budget")
            # Native read budgets omit unused gradients. Context diagnostics
            # serialize only quantities already obtained by the training step.
            diagnostic_count = 0
            with backend_context(model,self.backend):
                update = self.stepper.apply_update(lr,optimizer=optimizer)
            record = dict(result["record"])
            record.update(update)
            if (forwards[0] != self.budget["train_ce_per_update"] or
                    record["ordinary_gradient_calls"] != self.budget["train_ce_per_update"] or
                    record["sgd_updates"] != 1):
                raise AssertionError("Expected dynamic CE budget and one SGD update")
            expected_sides = {
                "g1_positive_gradient_calls": self.budget["first_plus_ce_per_update"],
                "g1_negative_gradient_calls": self.budget["first_minus_ce_per_update"],
                "g3_positive_gradient_calls": self.budget["g3_plus_ce_per_update"],
                "g3_negative_gradient_calls": self.budget["g3_minus_ce_per_update"],
                "g3_gradient_calls": self.budget["g3_plus_ce_per_update"]+self.budget["g3_minus_ce_per_update"],
                "opposite_g1_gradient_calls": 0,
                **{key+"_gradient_calls": self.budget[key+"_ce_per_update"] for key in ("g0", "g1", "g2", "gm", "g3")},
            }
            if any(record[key] != value for key, value in expected_sides.items()):
                raise AssertionError("Actual local gradient sides differ from explicit estimators")
            if any(record[k] for k in ("fisher_calls", "weight_merge_calls", "gradient_projection_calls",
                                       "hvp_calls", "create_graph_calls")):
                raise AssertionError("Unexpected auxiliary work")
            row = dict(record,step_index=self.steps,task=self.task,epoch=self.epoch,batch=self.batch,
                       protocol=self.cfg["protocol"], **execution_flags,
                       task_step_index=self.steps-self.task_start_step,seed=self.seed,
                       identity=copy.deepcopy(self.batch_identity),class_slice=[class_offset,class_offset+20],
                       seen_classes=l._total_classes,fc_shape=list(model.fc.weight.shape),
                       probe_branch=self.branch,update_direction=self.direction,step_mode=self.mode,
                       attention_backend=self.backend,training_ce_gradient_passes=self.budget["train_ce_per_update"],
                       training_forward_count=forwards[0],
                       isolated_diagnostic_gradient_calls=diagnostic_count,
                       training_create_graph_passes=0,training_hvp_passes=0,
                       training_hvp_reference_gradient_passes=0,
                       batch_samples=int(targets.numel()),train_ce=result["loss0"])
            if self.steps == self.task_start_step:
                # Once per task: exact ordered trainable-vector fingerprints
                # support same-state pairing without per-step CPU transfers.
                row["g012_fingerprints"] = {
                    label: [audit.tensor_digest(value) for value in result["context"]["grads"][label]]
                    for label in ("g0", "g1", "g2", "gm", "g3") if label in result["context"]["grads"]}
                row["point012_fingerprints"] = {
                    label: [audit.tensor_digest(value) for value in result["context"]["points"][label]]
                    for label in ("phi0", "phi1", "phi2", "phim", "phi3") if label in result["context"]["points"]}
                if self.task == 0 and bool(self.cli.max_tasks or self.cli.max_epochs or self.cli.max_batches):
                    snapshot_path = self.out/"smoke_g012_snapshot.pt"
                    names = {id(p): name for name, p in model.named_parameters()}
                    with observational(model):
                        torch.save(dict(schema_version=1, task=0, step_index=self.steps,
                            protocol=self.cfg["protocol"], factors=self.factor_config,
                            parameter_names=[names[id(p)] for p in self.stepper.params],
                            identity=copy.deepcopy(self.batch_identity),
                            grads={label:tuple(value.detach().cpu().clone() for value in result["context"]["grads"][label])
                                   for label in result["context"]["grads"]},
                            points={label:tuple(value.detach().cpu().clone() for value in result["context"]["points"][label])
                                    for label in result["context"]["points"]},
                            rng0=result["context"]["rng0"],rng1=result["context"]["rng1"],
                            g012_fingerprints=row["g012_fingerprints"],
                            point012_fingerprints=row["point012_fingerprints"]), snapshot_path)
                    self.pairing_snapshot = dict(schema_version=1, path=str(snapshot_path),
                        sha256=audit.digest(snapshot_path), task=0, step_index=self.steps)
                    row["same_state_snapshot"] = self.pairing_snapshot
            if audit.nonfinite_paths(row):
                raise FloatingPointError("Nonfinite step audit")
            self.last_record = row
            self.handle.write(json.dumps(row,allow_nan=False)+"\n")
            self.handle.flush()
            context_record = dict(step_index=self.steps, task=self.task, epoch=self.epoch, batch=self.batch,
                                  update_direction=self.direction, protocol=self.cfg["protocol"],
                                  scope="training_reads_only_no_extra_gradient", gradient_labels=list(result["context"]["grads"]),
                                  ordinary_gradient_calls=record["ordinary_gradient_calls"], diagnostic_gradient_calls=0,
                                  metrics=record.get("context_metrics", {}))
            self.context_handle.write(json.dumps(context_record,allow_nan=False)+"\n")
            self.context_handle.flush()
            self.steps += 1
            self.counters["training_batches"] += 1
            self.counters["training_samples"] += targets.numel()
            self.counters["training_ce_gradient_passes"] += record["ordinary_gradient_calls"]
            self.counters["training_forward_count"] += forwards[0]
            self.counters["training_sgd_updates"] += record["sgd_updates"]
            self.counters["training_global_projection_calls"] += record["gradient_projection_calls"]
            self.counters["training_g3_gradient_passes"] += record["g3_gradient_calls"]
            for key in ("g0", "g1", "g2", "gm"):
                self.counters["training_"+key+"_gradient_passes"] += record[key+"_gradient_calls"]
            self.counters["training_g3_positive_gradient_passes"] += record["g3_positive_gradient_calls"]
            self.counters["training_g3_negative_gradient_passes"] += record["g3_negative_gradient_calls"]
            self.counters["training_opposite_g1_gradient_passes"] += record["opposite_g1_gradient_calls"]
            self.counters["training_g1_positive_gradient_passes"] += record["g1_positive_gradient_calls"]
            self.counters["training_g1_negative_gradient_passes"] += record["g1_negative_gradient_calls"]
            self.counters["training_first_positive_gradient_passes"] += record["g1_positive_gradient_calls"]
            self.counters["training_first_negative_gradient_passes"] += record["g1_negative_gradient_calls"]
            self.counters["training_local_positive_gradient_passes"] += record["g1_positive_gradient_calls"]+record["g3_positive_gradient_calls"]
            self.counters["training_local_negative_gradient_passes"] += record["g1_negative_gradient_calls"]+record["g3_negative_gradient_calls"]
            self.samples += targets.numel()
            self.loss_sum += result["loss0"]*targets.numel()
            self.correct += int(result["logits0"][:,class_offset:class_offset+20].argmax(-1).eq(targets-class_offset).sum())
            self.epoch_batches += 1
            log = dict(ce_clean=result["loss0"],gam=result["loss0"],ewc=0.,delta_w_penalty=0.,
                       lambda_ewc=0.,ewc_gam_ratio=0.,delta_w_penalty_gam_ratio=0.)
            return result["logits0"],result["loss0"],log
        l._step_batch_with_log = MethodType(batch,l)

        original_run = l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            deterministic_execution_flags()
            if self.task != l._cur_task or class_offset != 20*self.task:
                raise AssertionError("Incorrect native task lifecycle")
            if scheduler.T_max != 50 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1):
                raise AssertionError("Scheduler protocol mismatch")
            self.scheduler = scheduler
            self.task_dataset_size = len(train_loader.dataset)
            self.task_batch_count = min(len(train_loader),self.cli.max_batches) if self.cli.max_batches else len(train_loader)
            self.task_start_step = self.steps
            self.save_observation("entry")
            old_rows = len(self.rows)
            original_schedule = scheduler.step
            def schedule(*a,**kw):
                result = original_schedule(*a,**kw)
                self.finish_epoch()
                if self.epoch in (1,5,25,50):
                    self.save_observation("epoch"+str(self.epoch))
                return result
            scheduler.step = schedule
            result = original_run(native.EpochLoader(train_loader,self),test_loader,optimizer,scheduler,
                                  epochs=self.cli.max_epochs or epochs,class_offset=class_offset)
            if len(self.rows)-old_rows != (self.cli.max_epochs or 50):
                raise AssertionError("Incomplete epochs")
            return result
        l._run_regularized_train = MethodType(run,l)

        original_increment = l.incremental_train
        def incremental(_self,dm):
            self.dm = dm
            self.task = l._cur_task+1
            fc = getattr(l._unwrap_network(),"fc",None)
            self.old_fc = {} if fc is None else {k:v.detach().clone() for k,v in fc.state_dict().items()}
            native.tensor_state_empty(l)
            result = original_increment(dm)
            native.tensor_state_empty(l)
            self.save_task_checkpoint()
            return result
        l.incremental_train = MethodType(incremental,l)

        original_eval = l.eval_task
        def evaluate(_self,*args,**kwargs):
            with observational(l._unwrap_network()):
                result = original_eval(*args,**kwargs)
            cnn,_ = result
            self.matrix.append(native.seen_accuracy_row(cnn,self.task,l._total_classes,len(self.matrix)))
            self.sample_curve.append(float(cnn["top1"]))
            self.write_matrix()
            return result
        l.eval_task = MethodType(evaluate,l)
        for name in ("_compute_accuracy","_compute_task_local_accuracy"):
            if hasattr(l,name):
                original = getattr(l,name)
                def observed(_self,*args,_original=original,**kwargs):
                    with observational(l._unwrap_network()):
                        return _original(*args,**kwargs)
                setattr(l,name,MethodType(observed,l))
        original_after = l.after_task
        def after(_self):
            if not l._ewc_skip_task_fisher or float(l._lambda_ewc) != 0.:
                raise AssertionError("Fisher/EWC enabled")
            before = native.trainable_snapshot(l._unwrap_network())
            result = original_after()
            if not native.trainable_unchanged(l._unwrap_network(),before):
                raise AssertionError("after_task modified pure-run weights")
            native.tensor_state_empty(l)
            self.after_task_checks.append(dict(task=self.task,parameters_exact=True,
                                               parameter_references_exact=True,fisher_empty=True,
                                               known_classes=l._known_classes))
            audit.atomic_json(self.out/"after_task_checks.json",self.after_task_checks)
            return result
        l.after_task = MethodType(after,l)

    def save_observation(self, point):
        model = self.learner._unwrap_network()
        directory = self.out/"observations"
        directory.mkdir(exist_ok=True)
        path = directory/("task_%03d_%s.pt"%(self.task,point))
        if path.exists():
            raise FileExistsError("Observation must never overwrite an existing snapshot")
        names = {id(p): name for name,p in model.named_parameters()}
        with observational(model):
            state = dict(schema_version=2,task=self.task,point=point,epoch=self.epoch if point != "entry" else 0,
                            step_index=self.steps,config=self.cfg,
                            trainable_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                            buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()},
                            module_training={n:module.training for n,module in model.named_modules()},
                            named_parameter_identity=[dict(name=names[id(p)],shape=list(p.shape),dtype=str(p.dtype),
                                requires_grad=p.requires_grad) for p in self.stepper.params],
                            trainable_grad_state={n:None if p.grad is None else p.grad.detach().cpu().clone()
                                                 for n,p in model.named_parameters() if p.requires_grad},
                            optimizer=base.cpu_tree(self.optimizer.state_dict()),
                            scheduler=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),
                            lr_next=[float(g["lr"]) for g in self.optimizer.param_groups],
                            lr_used=None if point == "entry" else copy.deepcopy(self.lr_used),
                            rng=audit.rng_get(), model_sha256=base.parameter_hash(model),
                            frozen_origin_sha256=base.parameter_hash(model,True),
                            source=str(self.cli.source), protocol=self.cfg["protocol"])
            torch.save(state,path)
        self.observations.append(dict(task=self.task,point=point,path=str(path),sha256=audit.digest(path),
                                      schema_version=2,epoch=state["epoch"],step_index=self.steps,
                                      model_sha256=state["model_sha256"]))
        audit.atomic_json(self.out/"observations.json",self.observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","config","output-dir"):
        parser.add_argument("--"+name,required=True)
    for name in ("max-tasks","max-epochs","max-batches"):
        parser.add_argument("--"+name,type=int,default=0)
    parser.add_argument("--dry-run",action="store_true")
    cli = parser.parse_args()
    for name in ("source","config","output_dir"):
        setattr(cli,name,str(Path(getattr(cli,name)).resolve()))
    cfg = resolve_config(json.loads(Path(cli.config).read_text()),cli)
    if cli.dry_run:
        print(json.dumps(dict(status="valid",source=cli.source,config=cli.config,output_dir=cli.output_dir,
                              branch=cfg["probe_branch"],direction=cfg["update_direction"],
                              mode=cfg["step_mode"],backend=cfg["attention_backend"],
                              rho=cfg["gam_grad_rho"],r=cfg["gam_grad_norm_rho"],
                              gradients_per_step=expected_budget(cfg)["train_ce_per_update"],updates_per_step=1,
                              **{key:cfg[key] for key in FACTOR_FIELDS},
                              expected_budget=expected_budget(cfg),extra_diagnostic_gradient_calls=0,
                              tasks=cli.max_tasks or 10,epochs=cli.max_epochs or 50,seed=2,
                              protocol=cfg["protocol"], cuda_deterministic_algorithms=True,
                              cuda_deterministic_warn_only=False)))
        return
    execution_flags = configure_deterministic_execution()
    cfg.update(execution_flags)
    out = Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    sys.path.insert(0,cli.source)
    os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import json_safe
    from src import trainer
    original_factory = factory.get_model
    holder = {}
    def create(name,args):
        args.update(deterministic_execution_flags())
        if holder:
            raise AssertionError("Exactly one model per independent run")
        learner = original_factory(name,args)
        controller = Controller(learner,args,cli)
        holder["controller"] = controller
        controller.install()
        audit.atomic_json(out/"effective_config.json",json_safe(args))
        return learner
    factory.get_model = create
    summary = dict(status="running",schema_version=1,seed=2,protocol=cfg["protocol"],
                   **execution_flags, execution_flags=execution_flags,
                   probe_branch=cfg["probe_branch"],update_direction=cfg["update_direction"],
                   step_mode=cfg["step_mode"],attention_backend=cfg["attention_backend"],
                   r=cfg["gam_grad_norm_rho"],rho=cfg["gam_grad_rho"],
                   expected_budget=expected_budget(cfg),extra_diagnostic_gradient_calls=0,
                   gradients_per_step=expected_budget(cfg)["train_ce_per_update"],updates_per_step=1,
                   smoke=bool(cli.max_tasks or cli.max_epochs or cli.max_batches),
                   requested_tasks=cli.max_tasks or 10,epochs_per_task=cli.max_epochs or 50,
                   scheduler_T_max=50,completed_tasks=0,input_config_sha256=audit.digest(cli.config),
                   trainer_sha256=audit.digest(__file__),source=cli.source)
    audit.atomic_json(out/"final_summary.json",summary)
    try:
        trainer.train(cfg)
        deterministic_execution_flags()
        c = holder["controller"]
        n = cli.max_tasks or 10
        if len(c.matrix)!=n or len(c.task_rows)!=n or len(c.after_task_checks)!=n:
            raise AssertionError("Incomplete task lifecycle")
        if len(c.rows)!=n*(cli.max_epochs or 50) or c.steps!=sum(t["steps"] for t in c.task_rows):
            raise AssertionError("Incomplete update accounting")
        if c.counters["forbidden_work_attempts"] or c.counters["weight_merge_updates"] or c.counters["fisher_estimation_passes"]:
            raise AssertionError("Forbidden module executed")
        budget = expected_budget(cfg)
        if (c.counters["training_ce_gradient_passes"] != budget["train_ce_per_update"]*c.steps or
                c.counters["training_forward_count"] != budget["train_ce_per_update"]*c.steps):
            raise AssertionError("Incomplete explicit estimator CE accounting")
        if c.counters["training_sgd_updates"] != c.steps:
            raise AssertionError("Incomplete one-SGD accounting")
        if any(c.counters["training_"+key+"_gradient_passes"] != budget[key+"_ce_per_update"]*c.steps
               for key in ("g0", "g1", "g2", "gm", "g3")):
            raise AssertionError("Native gradient labels disagree with estimator read budget")
        side_counts = {
            "training_g1_positive_gradient_passes": "first_plus_ce_per_update",
            "training_g1_negative_gradient_passes": "first_minus_ce_per_update",
            "training_first_positive_gradient_passes": "first_plus_ce_per_update",
            "training_first_negative_gradient_passes": "first_minus_ce_per_update",
            "training_g3_positive_gradient_passes": "g3_plus_ce_per_update",
            "training_g3_negative_gradient_passes": "g3_minus_ce_per_update",
            "training_local_positive_gradient_passes": "local_plus_ce_per_update",
            "training_local_negative_gradient_passes": "local_minus_ce_per_update",
        }
        if any(c.counters[counter] != budget[field]*c.steps for counter, field in side_counts.items()):
            raise AssertionError("Local gradient side accounting differs from explicit estimators")
        if c.counters["training_g3_gradient_passes"] != (budget["g3_plus_ce_per_update"]+budget["g3_minus_ce_per_update"])*c.steps:
            raise AssertionError("Incomplete final local gradient accounting")
        diagnostic_keys = ("diagnostic_snapshots", "diagnostic_gradient_calls", "diagnostic_forward_count",
                           "diagnostic_sgd_updates", "diagnostic_g3_positive_gradient_calls", "diagnostic_g3_negative_gradient_calls")
        if any(c.counters[key] for key in diagnostic_keys) or c.diagnostic_trace.stat().st_size:
            raise AssertionError("An unrequested extra diagnostic probe executed")
        if any(c.counters[k] for k in ("training_hvp_passes", "training_create_graph_passes",
                                      "training_global_projection_calls", "training_opposite_g1_gradient_passes")):
            raise AssertionError("Unrequested gradient work executed")
        summary.update(status="completed",completed_tasks=n,completed_epochs=len(c.rows),completed_steps=c.steps,
                       counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
                       initial_identity=c.task_identities[0],elapsed_seconds=time.time()-c.started,
                       checks=dict(finite=True,all_steps_self_spec=True,exact_snapshot_restore=True,
                                   dynamic_probe_budget=True,outer_point_geometry=True,unrequested_probe_omitted=True,
                                   no_extra_diagnostic_gradients=True,
                                   natural_sgd_exact=True,fisher_empty_all_tasks=True,actual_weight_merges_zero=True),
                       factor_config={key:cfg[key] for key in FACTOR_FIELDS},
                       formula=direction_name(cfg["update_direction"]),
                       side_diagnostics_sha256=audit.digest(c.diagnostic_trace),
                       context_diagnostics_sha256=audit.digest(c.context_trace),
                       observations_sha256=audit.digest(out/"observations.json"),
                       same_state_snapshot=c.pairing_snapshot,
                       step_trace_sha256=audit.digest(c.trace),
                       task_matrix_sha256=audit.digest(out/"task_matrix.json"),
                       final_metrics=json.loads((out/"task_matrix.json").read_text()))
        expected_points = ["entry"]+["epoch"+str(epoch) for epoch in (1,5,25,50) if epoch <= (cli.max_epochs or 50)]
        if sorted((o["task"],o["point"]) for o in c.observations) != sorted((task,point) for task in range(n) for point in expected_points):
            raise AssertionError("Missing native observation snapshot")
        summary["checks"].update(first_location_geometry=True,midpoint_geometry=True,selected_formula=True,
                                 all_tasks_observation_snapshots=True,
                                 deterministic_execution_verified=True)
    except Exception as exc:
        summary.update(status="failed",error=repr(exc),completed_tasks=len(holder["controller"].matrix) if holder else 0)
        if holder and holder["controller"].last_record:
            audit.atomic_json(out/"failed_step_audit.json",holder["controller"].last_record)
        raise
    finally:
        summary["execution_flags_at_exit"] = dict(
            deterministic_enabled=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        factory.get_model = original_factory
        if holder:
            holder["controller"].handle.close()
            holder["controller"].context_handle.close()
        audit.atomic_json(out/"final_summary.json",summary)


if __name__ == "__main__":
    main()
