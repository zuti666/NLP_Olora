#!/usr/bin/env python3
"""Frozen ImageNet-A lifecycle with a new four-probe, exact-restoration update."""
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
from direction_step import FourProbeStep
from effective_step import EffectiveStep
from direction_step import FullState
from response_hvp import backend_context


@contextmanager
def observational(model):
    state = FullState(model)
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
    backend = raw.get("attention_backend", "math")
    if branch not in ("P", "C", "CD") or direction != "g2" or mode != "natural" or backend != "math":
        raise ValueError("This experiment freezes P/C/CD, g2, natural SGD and math attention")
    seed = raw.get("seed", [2])
    if (isinstance(seed, list) and seed != [2]) or (not isinstance(seed, list) and (type(seed) is not int or seed != 2)):
        raise ValueError("Exactly seed 2 is registered")
    fixed = dict(dataset="imageneta",class_shuffle=False,init_cls=20,increment=20,
        lora_rank=10,batch_size=128,nb_tasks=10,total_sessions=10,max_train_tasks=10,
        backbone_type="vit_base_patch16_224",init_epoch=50,epochs=50,init_lr=.04,lrate=.02,
        init_momentum=0.,momentum=0.,init_weight_decay=0.,weight_decay=0.,
        optimizer="sgd",gam_base_optimizer="sgd",optimizer_type="gam",scheduler="cosine",
        lambda_flat=1.,gam_grad_rho=.05,gam_grad_norm_rho=.25,eps=1e-12,
        normalization_eps=1e-12,cosine_min_lr_ratio=.1,
        gam_grad_beta_1=1.,gam_grad_beta_2=1.,gam_grad_beta_3=0.,gam_grad_gamma=0.,
        gam_adaptive=False,use_sgv=False,gam_use_sgv=False,sgv_lambda=0.,gam_sgv_lambda=0.,
        ewc_lambda=0.,ewc_skip_task_fisher=True,approximate_fisher_in_task=False,
        in_task_weight_merge=False,task_endpoint_weight_merge=False,in_task_fisher_omega=0.,
        in_task_weight_merge_omega=0.,ewc_fisher_damping_alpha=0.,ewc_normalize_fisher=False,
        ewc_normalize_accumulated_fisher=False,ewc_dual_ascent=False,compute_full_fisher=False,
        posterior_state_save=False,fisher_split=False,as1_delta_w_penalty=False,
        memory_size=0,memory_per_class=0,in_task_path_normalize_snapshots=False,
        compute_nme_class_means=False,amp=False,use_amp=False,tf32=False,allow_tf32=False,
        deterministic_algorithms=True,precision="float32",dtype="float32",
        normalize_final_gradient=False,gradient_projection=False,hvp=False,create_graph=False)
    for key, expected in fixed.items():
        if key in raw and raw[key] != expected:
            raise ValueError("Frozen protocol conflict: " + key)
    for key in raw:
        if (key.endswith("_eval") or key.endswith("_probe")) and raw[key] is True:
            raise ValueError("Unregistered auxiliary evaluation/probe: " + key)
    for key in ("optimizer_type_by_task","fc_lrate","init_fc_lrate","resume_from_task",
                "resume_from_checkpoint_dir","resume_from_metrics_json","eval_only_after_resume",
                "step_reference_dir","cosine_eta_min","min_lr","min_lr_ratio"):
        if key in raw:
            raise ValueError("Unsupported override: " + key)
    if raw.get("component_rule", "u0") not in ("u0",branch+"_g2") or raw.get("component_scale","natural") != "natural":
        raise ValueError("Incompatible legacy direction settings")
    if not 0 <= cli.max_tasks <= 10 or not 0 <= cli.max_epochs <= 50 or cli.max_batches < 0:
        raise ValueError("Invalid smoke budget")
    surrogate = copy.deepcopy(raw)
    surrogate.update(component_rule="u0",component_scale="natural")
    cfg = native.resolve_config(surrogate, cli)
    cfg.update({k:v for k,v in fixed.items() if k not in ("precision","dtype")})
    cfg.update(probe_branch=branch,update_direction="g2",step_mode="natural",attention_backend="math",
               component_rule=branch+"_g2",component_scale="natural",optimizer_tag_override="g2_central_location_v1",
               protocol="g2_central_location_exact_restore_v1",precision="float32",dtype="float32",
               seed=[2],eps=1e-12,max_train_tasks=cli.max_tasks or 10,
               actual_update_method="theta0-lr*g2",actual_optimizer="torch.optim.SGD",
               actual_optimizer_momentum=0.,actual_optimizer_weight_decay=0.,
               actual_model_gradient_dtype="float32",actual_amp_enabled=False)
    return cfg


class Controller(native.Controller):
    def __init__(self, learner, cfg, cli):
        super().__init__(learner, cfg, cli)
        self.direction = cfg["update_direction"]
        self.branch = cfg["probe_branch"]
        self.backend = cfg["attention_backend"]
        self.counters.update(training_hvp_passes=0, training_hvp_reference_gradient_passes=0,
                             training_create_graph_passes=0, training_gradient_projection_passes=0,
                             training_forward_calls=0, sgd_updates=0,
                             diagnostic_replay_ce_gradient_passes=0,
                             isolated_diagnostic_ce_gradient_passes=0, diagnostic_replay_forward_calls=0)
        self.last_record = None
        self.observations = []
        self.replay_receipts = []
        self.task_invariant_receipts = []

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
                                             momentum=0, weight_decay=0, foreach=False)
            # Native checkpoint writer expects a base_optimizer; this is the same SGD.
            self.optimizer.base_optimizer = self.optimizer
            scfg = dict(branch=self.branch,
                        direction=self.direction,
                        rho=.05, radius=.25, eps=1e-12, backend=self.backend)
            self.stepper = FourProbeStep(model, params, self.geometry, scfg)
            old_count = 20*self.task
            fc = model.fc.state_dict()
            inherited = all(torch.equal(fc[k][:old_count], v.to(fc[k])) for k,v in self.old_fc.items())
            if not inherited:
                raise AssertionError("Classifier expansion changed existing rows")
            identity = dict(task=self.task, seed=self.seed, probe_branch=self.branch,
                            update_direction=self.direction, attention_backend=self.backend,
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
            forward_calls = [0]
            def count_forward(_module, _args, _output):
                forward_calls[0] += 1
            hook = model.register_forward_hook(count_forward)
            try:
                if self.task not in {r["task"] for r in self.replay_receipts}:
                    before_ce = self.stepper.total_ordinary_gradient_calls
                    before_forward = forward_calls[0]
                    try:
                        receipt = self.stepper.verify_zero_update_replay(inputs,targets,closure,lr)
                        receipt.update(task=self.task,seed=self.seed,branch=self.branch,
                                       batch_identity=copy.deepcopy(self.batch_identity),
                                       class_slice=[class_offset,class_offset+20],
                                       forward_calls=forward_calls[0]-before_forward)
                        if receipt["forward_calls"] != 8:
                            raise AssertionError("Replay must execute exactly eight model forwards")
                        self.replay_receipts.append(receipt)
                        audit.atomic_json(self.out/"replay_receipts.json",self.replay_receipts)
                    except BaseException as exc:
                        failed = getattr(exc,"replay_receipt",dict(passed=False,error=repr(exc)))
                        failed.update(task=self.task,branch=self.branch,seed=self.seed,
                                      batch_identity=copy.deepcopy(self.batch_identity))
                        audit.atomic_json(self.out/"failed_replay_receipt.json",failed)
                        raise
                    finally:
                        count = self.stepper.total_ordinary_gradient_calls-before_ce
                        self.counters["diagnostic_replay_ce_gradient_passes"] += count
                        self.counters["isolated_diagnostic_ce_gradient_passes"] += count
                        self.counters["diagnostic_replay_forward_calls"] += forward_calls[0]-before_forward
                before_forward = forward_calls[0]
                result = self.stepper.compute(inputs,targets,closure,lr)
                training_forward_calls = forward_calls[0]-before_forward
                if training_forward_calls != 4:
                    self.stepper.discard()
                    raise AssertionError("Training must execute exactly four model forwards")
                update = self.stepper.apply_update(lr,matched=False,optimizer=optimizer)
            finally:
                hook.remove()
            record = dict(result["record"])
            record.update(update)
            hvp = reference = 0
            ce_passes = result["record"]["ordinary_gradient_calls"]
            row = dict(record,step_index=self.steps,task=self.task,epoch=self.epoch,batch=self.batch,
                       task_step_index=self.steps-self.task_start_step,seed=self.seed,
                       identity=copy.deepcopy(self.batch_identity),class_slice=[class_offset,class_offset+20],
                       seen_classes=l._total_classes,fc_shape=list(model.fc.weight.shape),
                       probe_branch=self.branch,update_direction=self.direction,step_mode=self.mode,
                       attention_backend=self.backend,training_ce_gradient_passes=ce_passes,
                       training_forward_calls=training_forward_calls,
                       training_create_graph_passes=0,training_hvp_passes=0,
                       training_hvp_reference_gradient_passes=0,training_gradient_projection_passes=0,
                       batch_samples=int(targets.numel()),train_ce=result["loss0"])
            if audit.nonfinite_paths(row):
                raise FloatingPointError("Nonfinite step audit")
            self.last_record = row
            self.handle.write(json.dumps(row,allow_nan=False)+"\n")
            self.handle.flush()
            self.steps += 1
            self.counters["training_batches"] += 1
            self.counters["training_samples"] += targets.numel()
            self.counters["training_ce_gradient_passes"] += ce_passes
            self.counters["training_forward_calls"] += training_forward_calls
            self.counters["sgd_updates"] += update["sgd_updates"]
            self.counters["training_hvp_passes"] += hvp
            self.counters["training_create_graph_passes"] += hvp
            self.counters["training_hvp_reference_gradient_passes"] += reference
            self.samples += targets.numel()
            self.loss_sum += result["loss0"]*targets.numel()
            self.correct += int(result["logits0"][:,class_offset:class_offset+20].argmax(-1).eq(targets-class_offset).sum())
            self.epoch_batches += 1
            if self.mode == "matched":
                if not update["matched"]:
                    raise AssertionError("Unmatched update")
                self.match_count += 1
                self.max_match_error = max(self.max_match_error,update["actual_match_relative_error"])
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
            self.task_start_counters = dict(self.counters)
            expected_sizes = [1259,997,993,596,426,260,357,317,398,378]
            if self.task_dataset_size != expected_sizes[self.task]:
                raise AssertionError("Task dataset count differs from registered local split")
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
            self.write_task_invariants()
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

    def write_task_invariants(self):
        steps = self.steps-self.task_start_step
        observed = {k:self.counters[k]-self.task_start_counters.get(k,0) for k in self.counters}
        forbidden_keys = ("training_hvp_passes","training_create_graph_passes",
                          "training_hvp_reference_gradient_passes","training_gradient_projection_passes",
                          "fisher_estimation_passes","weight_merge_updates","forbidden_work_attempts")
        replay = next(r for r in self.replay_receipts if r["task"]==self.task)
        checks = dict(four_ce_per_update=observed["training_ce_gradient_passes"]==4*steps,
                      four_forward_per_update=observed["training_forward_calls"]==4*steps,
                      one_sgd_per_update=observed["sgd_updates"]==steps,
                      eight_isolated_replay_ce=observed["diagnostic_replay_ce_gradient_passes"]==8,
                      eight_isolated_replay_forwards=observed["diagnostic_replay_forward_calls"]==8,
                      forbidden_modules_zero=all(observed[k]==0 for k in forbidden_keys),
                      strict_replay_passed=replay["passed"],
                      old_fc_rows_inherited_exact=self.task_identities[-1]["old_fc_rows_inherited_exact"])
        receipt = dict(task=self.task,seed=self.seed,branch=self.branch,steps=steps,counters=observed,status="passed" if all(checks.values()) else "failed",
                       checks=checks,passed=all(checks.values()))
        if not receipt["passed"]:
            audit.atomic_json(self.out/"failed_task_invariant_receipt.json",receipt)
            raise AssertionError("Task invariant gate failed")
        self.task_invariant_receipts.append(receipt)
        audit.atomic_json(self.out/"task_invariant_receipts.json",self.task_invariant_receipts)

    def save_task_checkpoint(self):
        super().save_task_checkpoint()
        directory = Path(self.checkpoints[-1])
        path = directory/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(probe_branch=self.branch,update_direction="g2",step_mode="natural",
                        attention_backend="math",protocol=self.cfg["protocol"],
                        source=self.cli.source,config_sha256=audit.digest(self.out/"effective_config.json"),
                        task_initial_identity=self.task_identities[-1])
        audit.atomic_json(path,manifest)


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
                              tasks=cli.max_tasks or 10,epochs=cli.max_epochs or 50,seed=2)))
        return
    out = Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
                   step_mode=cfg["step_mode"],attention_backend=cfg["attention_backend"],r=.25,rho=.05,
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
        if len(c.replay_receipts)!=n or len(c.task_invariant_receipts)!=n:
            raise AssertionError("Missing task replay/invariant receipts")
        if c.counters["training_ce_gradient_passes"]!=4*c.steps or c.counters["sgd_updates"]!=c.steps or c.counters["diagnostic_replay_ce_gradient_passes"]!=8*n:
            raise AssertionError("Observed training/diagnostic compute budget mismatch")
        if c.mode=="matched" and c.match_count!=c.steps:
            raise AssertionError("Unmatched steps")
        summary.update(status="completed",completed_tasks=n,completed_epochs=len(c.rows),completed_steps=c.steps,
                       step_count=c.steps,task_count=n,counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
                       after_task_checks=c.after_task_checks,replay_receipts=c.replay_receipts,
                       replay_receipts_sha256=audit.digest(out/"replay_receipts.json"),
                       task_invariant_receipts=c.task_invariant_receipts,
                       task_invariant_receipts_sha256=audit.digest(out/"task_invariant_receipts.json"),
                       task_identities_sha256=audit.digest(out/"task_identities.json"),
                       effective_config_sha256=audit.digest(out/"effective_config.json"),
                       initial_identity=c.task_identities[0],elapsed_seconds=time.time()-c.started,
                       checks=dict(finite=True,all_steps_self_spec=True,exact_snapshot_restore=True,
                                   four_base_probes=True,fisher_empty_all_tasks=True,actual_weight_merges_zero=True,
                                   strict_bitwise_first_batch_replay=True,all_task_invariants_passed=True,
                                   natural_g2_only=True,complete_task_lifecycle=True),
                       max_match_relative_error=c.max_match_error if c.mode=="matched" else None,
                       matched_steps=c.match_count,step_trace_sha256=audit.digest(c.trace),
                       task_matrix_sha256=audit.digest(out/"task_matrix.json"),
                       final_metrics=json.loads((out/"task_matrix.json").read_text()))
    except Exception as exc:
        summary.update(status="failed",error=repr(exc),completed_tasks=len(holder["controller"].matrix) if holder else 0)
        if holder:
            c=holder["controller"]
            summary.update(counters=c.counters,completed_steps=c.steps,step_count=c.steps,
                           replay_receipts=c.replay_receipts,task_invariant_receipts=c.task_invariant_receipts)
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
