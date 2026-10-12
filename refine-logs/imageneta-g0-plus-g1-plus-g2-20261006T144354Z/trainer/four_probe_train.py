#!/usr/bin/env python3
"""Frozen ImageNet-A lifecycle with four probes and one natural SGD update."""
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
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
from direction_step import FourProbeStep, DIRECTIONS
from effective_step import EffectiveStep
from state_transaction import backend_context, State


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
    branch = raw["probe_branch"]
    direction = raw["update_direction"]
    mode = raw["step_mode"]
    backend = raw.get("attention_backend", "native")
    if branch != "P" or direction not in DIRECTIONS:
        raise ValueError("Only positive P geometry and g0_plus_g1_plus_g2 are supported")
    if mode != "natural" or backend not in ("native", "math"):
        raise ValueError("Natural step mode and native|math backend required")
    radius = float(raw["gam_grad_norm_rho"])
    rho = float(raw.get("gam_grad_rho", .05))
    seed = raw.get("seed", [2])
    seed = seed[0] if isinstance(seed, list) and len(seed) == 1 else seed
    if seed != 2 or rho != .05 or radius not in (.25, 1.):
        raise ValueError("Protocol requires seed=2, rho=.05, r=.25|1")
    surrogate = copy.deepcopy(raw)
    surrogate.update(component_rule="u0", component_scale="natural")
    cfg = native.resolve_config(surrogate, cli)
    cfg.update(probe_branch=branch, update_direction=direction, step_mode=mode,
               attention_backend=backend, component_rule=branch+"_"+direction,
               component_scale=mode, optimizer_tag_override=direction+"_original_positive_four_probe_v1",
               gam_grad_rho=rho, gam_grad_norm_rho=radius, lora_rank=10,
               compute_nme_class_means=False,
               protocol="original_positive_four_probe_g0_plus_g1_plus_g2_exact_restore_rng_replay_v1",
               in_task_path_normalize_snapshots=False)
    return cfg


class Controller(native.Controller):
    def __init__(self, learner, cfg, cli):
        super().__init__(learner, cfg, cli)
        self.direction = cfg["update_direction"]
        self.branch = cfg["probe_branch"]
        self.backend = cfg["attention_backend"]
        self.counters.update(training_hvp_passes=0, training_hvp_reference_gradient_passes=0,
                             training_create_graph_passes=0, training_forward_count=0,
                             training_sgd_updates=0, training_g3_gradient_passes=0,
                             training_opposite_g1_gradient_passes=0,
                             training_global_projection_calls=0)
        self.last_record = None
        self.observations = []

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
            scfg = dict(branch=self.branch, direction=self.direction,
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
                            forward_rng_rule="replay R0 at each point; advance to R1 from g0 once")
            self.task_identities.append(identity)
            audit.atomic_json(self.out/"task_identities.json", self.task_identities)
            self.save_observation("entry")
            return self.optimizer
        l._build_optimizer = MethodType(build, l)

        def batch(_self, optimizer, inputs, targets, class_offset):
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
                    update = self.stepper.apply_update(lr,optimizer=optimizer)
            finally:
                hook.remove()
            record = dict(result["record"])
            record.update(update)
            if forwards[0] != 4 or record["ordinary_gradient_calls"] != 4 or record["sgd_updates"] != 1:
                raise AssertionError("Expected exactly four CE forwards/gradients and one SGD update")
            if record["g3_gradient_calls"] != 1 or record["opposite_g1_gradient_calls"] != 0:
                raise AssertionError("Expected one g3 gradient and no opposite g1")
            expected_projections = 0
            if record["gradient_projection_calls"] != expected_projections:
                raise AssertionError("Pure sum must perform zero projections")
            if record["g3_used_in_final"] or not record["original_positive_probe"]:
                raise AssertionError("Final update must use positive g012 only")
            if any(record[k] for k in ("fisher_calls", "weight_merge_calls", "hvp_calls", "create_graph_calls")):
                raise AssertionError("Unexpected auxiliary work")
            row = dict(record,step_index=self.steps,task=self.task,epoch=self.epoch,batch=self.batch,
                       task_step_index=self.steps-self.task_start_step,seed=self.seed,
                       identity=copy.deepcopy(self.batch_identity),class_slice=[class_offset,class_offset+20],
                       seen_classes=l._total_classes,fc_shape=list(model.fc.weight.shape),
                       probe_branch=self.branch,update_direction=self.direction,step_mode=self.mode,
                       attention_backend=self.backend,training_ce_gradient_passes=4,
                       training_forward_count=forwards[0],
                       training_create_graph_passes=0,training_hvp_passes=0,
                       training_hvp_reference_gradient_passes=0,
                       batch_samples=int(targets.numel()),train_ce=result["loss0"])
            if audit.nonfinite_paths(row):
                raise FloatingPointError("Nonfinite step audit")
            self.last_record = row
            self.handle.write(json.dumps(row,allow_nan=False)+"\n")
            self.handle.flush()
            self.steps += 1
            self.counters["training_batches"] += 1
            self.counters["training_samples"] += targets.numel()
            self.counters["training_ce_gradient_passes"] += record["ordinary_gradient_calls"]
            self.counters["training_forward_count"] += forwards[0]
            self.counters["training_sgd_updates"] += record["sgd_updates"]
            self.counters["training_global_projection_calls"] += record["gradient_projection_calls"]
            self.counters["training_g3_gradient_passes"] += record["g3_gradient_calls"]
            self.counters["training_opposite_g1_gradient_passes"] += record["opposite_g1_gradient_calls"]
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
            if self.task != l._cur_task or class_offset != 20*self.task:
                raise AssertionError("Incorrect native task lifecycle")
            if scheduler.T_max != 50 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1):
                raise AssertionError("Scheduler protocol mismatch")
            self.scheduler = scheduler
            self.task_dataset_size = len(train_loader.dataset)
            self.task_batch_count = min(len(train_loader),self.cli.max_batches) if self.cli.max_batches else len(train_loader)
            self.task_start_step = self.steps
            old_rows = len(self.rows)
            original_schedule = scheduler.step
            def schedule(*a,**kw):
                result = original_schedule(*a,**kw)
                self.finish_epoch()
                if self.epoch == 1:
                    self.save_observation("epoch1")
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
            self.save_observation("end")
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
        if self.task not in (0,1,4,9):
            return
        model = self.learner._unwrap_network()
        directory = self.out/"observations"
        directory.mkdir(exist_ok=True)
        path = directory/("task_%03d_%s.pt"%(self.task,point))
        with observational(model):
            torch.save(dict(task=self.task,point=point,config=self.cfg,
                            trainable_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                            rng=audit.rng_get()),path)
        self.observations.append(dict(task=self.task,point=point,path=str(path),sha256=audit.digest(path)))
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
                              gradients_per_step=4,updates_per_step=1,g3_used_in_final=False,
                              projection_calls_per_step=0,
                              tasks=cli.max_tasks or 10,epochs=cli.max_epochs or 50,seed=2)))
        return
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
                   probe_branch=cfg["probe_branch"],update_direction=cfg["update_direction"],
                   step_mode=cfg["step_mode"],attention_backend=cfg["attention_backend"],
                   r=cfg["gam_grad_norm_rho"],rho=cfg["gam_grad_rho"],
                   gradients_per_step=4,updates_per_step=1,g3_used_in_final=False,
                   projection_calls_per_step=0,
                   smoke=bool(cli.max_tasks or cli.max_epochs or cli.max_batches),
                   requested_tasks=cli.max_tasks or 10,epochs_per_task=cli.max_epochs or 50,
                   scheduler_T_max=50,completed_tasks=0,input_config_sha256=audit.digest(cli.config),
                   trainer_sha256=audit.digest(__file__),source=cli.source)
    audit.atomic_json(out/"final_summary.json",summary)
    try:
        trainer.train(cfg)
        c = holder["controller"]
        n = cli.max_tasks or 10
        if len(c.matrix)!=n or len(c.task_rows)!=n or len(c.after_task_checks)!=n:
            raise AssertionError("Incomplete task lifecycle")
        if len(c.rows)!=n*(cli.max_epochs or 50) or c.steps!=sum(t["steps"] for t in c.task_rows):
            raise AssertionError("Incomplete update accounting")
        if c.counters["forbidden_work_attempts"] or c.counters["weight_merge_updates"] or c.counters["fisher_estimation_passes"]:
            raise AssertionError("Forbidden module executed")
        if c.counters["training_ce_gradient_passes"] != 4*c.steps or c.counters["training_forward_count"] != 4*c.steps:
            raise AssertionError("Incomplete four-probe accounting")
        if c.counters["training_sgd_updates"] != c.steps:
            raise AssertionError("Incomplete one-SGD accounting")
        if c.counters["training_g3_gradient_passes"] != c.steps:
            raise AssertionError("Incomplete g3 accounting")
        expected_projections = 0
        if c.counters["training_global_projection_calls"] != expected_projections:
            raise AssertionError("Pure sum performed unexpected projection work")
        if any(c.counters[k] for k in ("training_hvp_passes", "training_create_graph_passes",
                                      "training_opposite_g1_gradient_passes")):
            raise AssertionError("Unrequested gradient work executed")
        summary.update(status="completed",completed_tasks=n,completed_epochs=len(c.rows),completed_steps=c.steps,
                       counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
                       initial_identity=c.task_identities[0],elapsed_seconds=time.time()-c.started,
                       checks=dict(finite=True,all_steps_self_spec=True,exact_snapshot_restore=True,
                                   four_probes=True,g3_outer_probe=True,g3_computed_unused=True,
                                   original_positive_probe=True,requested_projection_only=True,opposite_probe_omitted=True,
                                   natural_sgd_exact=True,fisher_empty_all_tasks=True,actual_weight_merges_zero=True),
                       step_trace_sha256=audit.digest(c.trace),
                       task_matrix_sha256=audit.digest(out/"task_matrix.json"),
                       final_metrics=json.loads((out/"task_matrix.json").read_text()))
    except Exception as exc:
        summary.update(status="failed",error=repr(exc),completed_tasks=len(holder["controller"].matrix) if holder else 0)
        if holder and holder["controller"].last_record:
            audit.atomic_json(out/"failed_step_audit.json",holder["controller"].last_record)
        raise
    finally:
        factory.get_model = original_factory
        if holder:
            holder["controller"].handle.close()
        audit.atomic_json(out/"final_summary.json",summary)


if __name__ == "__main__":
    main()
