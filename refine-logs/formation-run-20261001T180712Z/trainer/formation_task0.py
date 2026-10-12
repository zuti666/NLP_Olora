#!/usr/bin/env python3
"""Isolated Task0 formation study; native GAM + per-step actual BA/FC audits."""
import argparse
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
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
from effective_step import EffectiveStep, solve_positive_scale

ARMS = {"s": ("s",1.,1.), "s_lr15": ("s",1.5,1.), "s_lr2": ("s",2.,1.),
        "g_raw": ("g",1.,1.), "g_r025": ("g",1.,.25), "g_match": ("g",1.,1.),
        "sum": ("sum",1.,1.), "mean_lr2": ("mean",2.,1.)}
TRACE_SCHEMA = 1
MATCH_RTOL = 1e-3


def resolve_config(raw, cli):
    arm = raw.get("formation_arm")
    if arm not in ARMS:
        raise ValueError("formation_arm must be one of " + ",".join(ARMS))
    seed_value = raw.get("seed",2)
    if isinstance(seed_value,list):
        if len(seed_value) != 1:
            raise ValueError("One independent seed per invocation")
        seed_value = seed_value[0]
    seed = int(seed_value)
    if seed not in (0,1,2) or seed != seed_value:
        raise ValueError("Formation permits seeds 0,1,2")
    if not 0 <= cli.max_epochs <= 50 or cli.max_batches < 0:
        raise ValueError("Invalid smoke budget")
    direction,multiplier,radius = ARMS[arm]
    surrogate = copy.deepcopy(raw)
    surrogate.update(seed=2,stage1_direction=direction)
    cfg = base.resolve_config(surrogate,cli)
    # Native scheduler reads epochs; native Task0 loop reads init_epoch.
    cfg.update(seed=seed,formation_arm=arm,init_epoch=cli.max_epochs or 50,epochs=50,
               init_lr=.04*multiplier,lrate=.02*multiplier,cosine_min_lr_ratio=.1,
               formation_lr_multiplier=multiplier,gam_grad_norm_rho=radius,
               use_sgv=False,gam_use_sgv=False,sgv_lambda=0.,gam_sgv_lambda=0.,
               max_train_tasks=1)
    for key in ("cosine_eta_min","min_lr","min_lr_ratio"):
        cfg.pop(key,None)
    if arm == "g_match" and not cfg.get("step_reference_dir"):
        raise ValueError("g_match requires completed independent S step_reference_dir")
    return cfg


class StepReference:
    """Immutable independent-S trace, keyed by the same epoch and minibatch."""
    def __init__(self, directory, seed, source_gam_sha256=None, protocol=None):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "step_trace.jsonl"
        self.sha256 = audit.digest(self.path)
        self.summary = json.loads((self.directory/"final_summary.json").read_text())
        manifest = json.loads((self.directory/"step_trace_manifest.json").read_text())
        if (self.summary.get("status") != "completed" or self.summary.get("formation_arm") != "s"
                or self.summary.get("seed") != seed):
            raise ValueError("Reference must be completed independent S of the same seed")
        if manifest.get("trace_sha256") != self.sha256 or self.summary.get("step_trace_sha256") != self.sha256:
            raise ValueError("S reference trace hash mismatch")
        if manifest.get("schema_version") != TRACE_SCHEMA or manifest.get("seed") != seed:
            raise ValueError("S reference schema/seed mismatch")
        if source_gam_sha256 and self.summary.get("source_gam_sha256") != source_gam_sha256:
            raise ValueError("S reference frozen GAM source mismatch")
        self.rows = {}
        for line in self.path.read_text().splitlines():
            row = json.loads(line)
            key = (row["epoch"],row["batch"])
            if key in self.rows or row.get("seed") != seed or row.get("task") != 0:
                raise ValueError("Duplicate or wrong-seed/task reference step")
            if row.get("step_index") != len(self.rows) or row.get("formation_arm") != "s":
                raise ValueError("S trace step sequence/arm mismatch")
            if not row.get("self_spec_passed") or row.get("closure_count") != 4:
                raise ValueError("Unaudited S reference step")
            norm = row["actual_effective_delta"]["joint_norm"]
            if not math.isfinite(norm) or norm <= 0 or audit.nonfinite_paths(row):
                raise ValueError("Invalid S reference effective norm")
            self.rows[key] = row
        if len(self.rows) != manifest.get("steps") or len(self.rows) != self.summary.get("counters",{}).get("training_batches"):
            raise ValueError("Incomplete S reference trace")
        if protocol is not None:
            epochs,batches = protocol["epochs"],protocol["batches_per_epoch"]
            expected_keys = [(epoch,batch) for epoch in range(1,epochs+1) for batch in range(batches)]
            snapshots = [0]+[epoch for epoch in range(1,epochs+1) if epoch%10==0 or epoch==epochs]
            if (self.summary.get("smoke") is not protocol["smoke"]
                    or self.summary.get("completed_epochs") != epochs
                    or list(self.rows) != expected_keys
                    or self.summary.get("checkpoint_epochs") != snapshots
                    or not self.summary.get("finite_check",{}).get("passed")
                    or not self.summary.get("task_check",{}).get("passed")
                    or not self.summary.get("step_checks",{}).get("passed")
                    or set(self.summary.get("checkpoint_checks",{})) != set(map(str,snapshots))
                    or not all(self.summary["checkpoint_checks"].values())):
                raise ValueError("S reference smoke/formal budget, complete sequence, checkpoint or acceptance mismatch")
            for key,value in protocol["code_identity"].items():
                if self.summary.get(key) != value:
                    raise ValueError("S reference code identity mismatch: " + key)
            reference_config = json.loads((self.directory/"effective_config.json").read_text())
            for key in ("seed","dataset","class_shuffle","init_cls","increment","lora_rank","batch_size",
                        "init_epoch","epochs","init_lr","cosine_min_lr_ratio","gam_grad_rho",
                        "gam_grad_norm_rho","gam_grad_gamma","lambda_flat","ewc_lambda","model_name"):
                if reference_config.get(key) != protocol["config"].get(key):
                    raise ValueError("S reference effective config mismatch: " + key)

    def get(self, epoch, batch, identity):
        row = self.rows.get((epoch,batch))
        if row is None:
            raise ValueError("Missing independent S reference step")
        if row["identity"] != identity:
            raise ValueError("S reference minibatch/order/augmentation pairing mismatch")
        return row


class EpochLoader(base.EpochLoader):
    def __iter__(self):
        self.epoch += 1
        self.c.begin_epoch(self.epoch)
        seed = self.c.seed*100000+self.epoch
        generator = torch.Generator().manual_seed(seed)
        loader = self.c.dm.build_dataloader(self.dataset,batch_size=self.c.config["batch_size"],
            shuffle=True,num_workers=self.c.config.get("train_num_workers",8),
            generator=generator,worker_init_fn=base.worker_init,persistent_workers=False)
        training_rng = audit.rng_get()
        audit.seed_all(seed)
        data_rng = audit.rng_get()
        audit.rng_set(training_rng)
        iterator,order = None,hashlib.sha256()
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
            ids,inputs,targets = batch
            order.update(ids.contiguous().numpy().tobytes())
            self.c.batch_index = index
            self.c.current_batch_identity = {"input_sha256":audit.tensor_digest(inputs),
                "target_sha256":audit.tensor_digest(targets),"indices_sha256":audit.tensor_digest(ids),
                "indices":ids.tolist(),"data_seed":seed}
            if index == 0:
                self.c.first_batch_hashes = copy.deepcopy(self.c.current_batch_identity)
            yield batch
        self.c.sampler_digest = order.hexdigest()


class FormationController(base.Controller):
    def __init__(self,learner,dm,config,cli):
        super().__init__(learner,dm,config,cli)
        self.arm = config["formation_arm"]
        self.source_sha = audit.digest(Path(cli.source)/"optimer_PerturabtionType/gam.py")
        protocol = {"epochs":config["init_epoch"],"batches_per_epoch":min(cli.max_batches,10) if cli.max_batches else 10,
                    "smoke":bool(cli.max_epochs or cli.max_batches),"config":config,
                    "code_identity":{"trainer_sha256":audit.digest(__file__),
                        "effective_step_sha256":audit.digest(Path(__file__).with_name("effective_step.py")),
                        "base_trainer_sha256":audit.digest(Path(base.__file__))}}
        self.reference = (StepReference(config["step_reference_dir"],self.seed,self.source_sha,protocol)
                          if self.arm == "g_match" else None)
        self.trace_path = self.out/"step_trace.jsonl"
        self.trace_file = self.trace_path.open("x")
        self.trace_rows = []
        self.effective = None
        self.pending_step = None

    def install(self):
        # The native clean+GAM projector is audited, never replaced.
        original_projector = self.learner._build_direction_projector
        def build_projector(_learner,clean,fisher,delta=None):
            if fisher or delta or float(_learner._lambda_flat) != 1.:
                raise AssertionError("Formation requires empty Fisher/penalty and lambda_flat=1")
            native = original_projector(clean,fisher,delta)
            def project(groups):
                params = [p for group in groups for p in group["params"]]
                expected = []
                for p in params:
                    if p.grad is None or id(p) not in clean:
                        raise AssertionError("Missing clean/native gradient")
                    expected.append(clean[id(p)] + 1.0*(p.grad.detach()-clean[id(p)]))
                result = native(groups)
                self.projector_exact = all(torch.equal(p.grad,e) for p,e in zip(params,expected))
                if not self.projector_exact:
                    raise AssertionError("Native lambda_flat=1 projector disagrees with own operation sequence")
                return result
            return project
        self.learner._build_direction_projector = MethodType(build_projector,self.learner)
        base.EpochLoader = EpochLoader
        super().install()

    def save_checkpoint(self,epoch):
        if epoch == 0:
            initial = self.scheduler.state_dict()
            expected_floor = self.config["init_lr"]*.1
            if initial.get("T_max") != 50 or not math.isclose(initial.get("eta_min",-1),expected_floor,rel_tol=1e-12):
                raise AssertionError("Cosine budget/floor changed, including smoke")
            if self.reference:
                other = self.reference.summary["initial_identity"]
                for key in ("initial_model_sha256","frozen_origin_sha256","class_order","task_range"):
                    if self.identity[key] != other[key]:
                        raise ValueError("Independent S origin identity mismatch: " + key)
        super().save_checkpoint(epoch)
        directory = self.out/"checkpoints"/("epoch_%03d" % epoch)
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest.update(formation_arm=self.arm,seed=self.seed,formation_schema_version=1,
                        source_gam_sha256=self.source_sha,completed_steps=len(self.trace_rows))
        audit.atomic_json(directory/"manifest.json",manifest)

    def finish_epoch(self):
        self.trace_file.flush()
        os.fsync(self.trace_file.fileno())
        audit.atomic_json(self.out/"step_trace_manifest.json",{
            "schema_version":TRACE_SCHEMA,"steps":len(self.trace_rows),"completed_epochs":self.epoch,
            "seed":self.seed,"formation_arm":self.arm,"trace_sha256":audit.digest(self.trace_path),
            "reference_trace_sha256":self.reference.sha256 if self.reference else None,
            "base_point":"native GAM unperturb complete; immediately before SGD",
            "effective_map":"concat(unscaled LoRA B@A, FC weight, FC bias)",
            "measurement":"actual updated factor values, float64 low-rank Gram identity",
            "matching_tolerance":MATCH_RTOL,"source_gam_sha256":self.source_sha})
        super().finish_epoch()

    def instrument_optimizer(self,opt):
        self.optimizer = opt
        params = [p for group in opt.param_groups for p in group["params"]]
        if opt.param_groups is not opt.base_optimizer.param_groups:
            raise AssertionError("Expected shared native GAM/SGD parameter groups")
        self.effective = EffectiveStep.from_model(self.model)
        if {id(p) for p in params} != {id(p) for p in self.effective.params}:
            raise AssertionError("Optimizer/effective map scope mismatch")
        original_decompose = opt.gradient_decompose
        def decompose(_opt,*args,**kwargs):
            result = original_decompose(*args,**kwargs)
            b1,b3 = float(self.config["gam_grad_beta_1"]),float(self.config["gam_grad_beta_3"])
            expected = [b1*opt.state[p]["g_1"]+b3*opt.state[p]["g_3"] for p in params]
            self.decomposition_exact = all(torch.equal(p.grad,e) for p,e in zip(params,expected))
            if not self.decomposition_exact:
                raise AssertionError("Native gamma0 direction differs from beta1*g1+beta3*g3")
            base.require_finite_gradients(params,"GAM selected direction")
            return result
        opt.gradient_decompose = MethodType(decompose,opt)

        original_base = opt.base_optimizer.step
        def base_step(_base,*args,**kwargs):
            geom = self.effective
            before = geom.capture()  # AFTER native unperturb, never saved-copy restoration.
            roundtrip = geom.actual_delta(self.step_origin,before)
            nominal = [float(group["lr"]) for group in opt.param_groups]
            if not nominal or any(not math.isfinite(x) or x<=0 for x in nominal) or len(set(nominal)) != 1:
                raise AssertionError("One finite positive SGD scale required for all LoRA/FC parameters")
            for group in opt.param_groups:
                if any(float(group.get(k,0)) != 0 for k in ("momentum","weight_decay","dampening")) or group.get("nesterov",False) or group.get("maximize",False):
                    raise AssertionError("Effective-step audit requires ordinary zero-momentum/decay SGD")
            base.require_finite_gradients(params,"Actual SGD gradient")
            target,matched,reference_row,root = None,None,None,None
            scale = nominal[0]
            if self.reference:
                reference_row = self.reference.get(self.epoch,self.batch_index,self.current_batch_identity)
                target = reference_row["actual_effective_delta"]["joint_norm"]
                root = solve_positive_scale(geom.polynomial(),target,initial_scale=scale,max_scale=10.)
                scale = root["scale"]
            predicted = [p.detach().clone().add_(p.grad,alpha=-scale) for p in params]
            if self.batch_index == 0:
                self.sample_final_gradient = base.vector_grads(params)
            try:
                for group in opt.param_groups:
                    group["lr"] = scale
                result = original_base(*args,**kwargs)
            finally:
                for group,lr in zip(opt.param_groups,nominal):
                    group["lr"] = lr
            actual = geom.actual_delta(before)
            exact = all(torch.equal(p,e) for p,e in zip(params,predicted))
            maximum = max(float((p.detach()-e).abs().max()) for p,e in zip(params,predicted))
            error = abs(actual["joint_norm"]-target)/target if target is not None else None
            matched = error <= MATCH_RTOL if error is not None else None
            row = {"schema_version":TRACE_SCHEMA,"formation_arm":self.arm,"seed":self.seed,
                "task":0,"epoch":self.epoch,"batch":self.batch_index,"step_index":len(self.trace_rows),
                "identity":copy.deepcopy(self.current_batch_identity),"lr_nominal":nominal[0],
                "step_scale":scale,"scale_over_nominal_lr":scale/nominal[0],
                "actual_effective_delta":actual,"probe_roundtrip_delta":roundtrip,
                "decomposition_spec_exact":self.decomposition_exact,"projector_spec_exact":self.projector_exact,
                "base_sgd_exact":exact,"base_sgd_max_abs_error":maximum,
                "reference_trace_sha256":self.reference.sha256 if self.reference else None,
                "reference_step_index":reference_row["step_index"] if reference_row else None,
                "reference_target_norm":target,"actual_match_relative_error":error,"matched":matched,
                "root":root,"self_spec_passed":bool(exact and self.decomposition_exact and self.projector_exact)}
            self.pending_step = row
            if not exact or not row["self_spec_passed"] or matched is False or audit.nonfinite_paths(row):
                audit.atomic_json(self.out/"failed_step_audit.json",row)
                raise AssertionError("Actual SGD/self-spec/effective-step matching failed")
            self.sample_base_audit = {"passed":exact,"predicted_after_max_abs_error":maximum,
                "probe_roundtrip_parameter_norm":roundtrip["parameter_norm"],
                "base_step_parameter_norm":actual["parameter_norm"],"step_scale":scale,
                "note":"Actual update starts after native unperturb; finite-roundtrip drift is separate."}
            return result
        opt.base_optimizer.step = MethodType(base_step,opt.base_optimizer)
        native_step = opt.step
        def step(_opt,closure=None):
            sampled = self.batch_index == 0
            self.active_geo = audit.Geometry(self.model) if sampled else None
            self.step_origin = self.effective.capture()
            self.decomposition_exact = self.projector_exact = False
            points,count = [],0
            def wrapped():
                nonlocal count
                count += 1
                if sampled:
                    points.append(self.active_geo.displacement_metrics())
                result = closure()
                base.require_finite_gradients(params,"GAM probe %d" % (count-1))
                if not bool(torch.isfinite(result[1])):
                    raise FloatingPointError("Nonfinite GAM probe CE")
                return result
            result = native_step(closure=wrapped)
            if count != 4 or self.pending_step is None:
                raise AssertionError("Formation requires four native GAM closures and one SGD step")
            self.pending_step.update(closure_count=count,training_ce_gradient_passes=5)
            self.trace_rows.append(self.pending_step)
            self.trace_file.write(json.dumps(self.pending_step,allow_nan=False)+"\n")
            self.trace_file.flush()
            self.pending_step = None
            if sampled:
                grads = {"g%d" % k:audit.flatten([opt.state[p]["g_%d" % k].detach() for p in self.active_geo.params]) for k in range(4)}
                self.record_geometry(self.active_geo,grads,points,self.sample_final_gradient,count,"native_training_GAM_state")
            self.active_geo = None
            self.step_origin = None
            return result
        opt.step = MethodType(step,opt)
        return opt

    def record_geometry(self,geo,grads,points,final_gradient,closure_count,source):
        selected = {"s":grads["g1"],"g":grads["g3"],"sum":grads["g1"]+grads["g3"],
                    "mean":.5*grads["g1"]+.5*grads["g3"]}[self.direction]
        row = {"epoch":self.epoch,"batch":self.batch_index,"task":0,"source":source,
               "formation_arm":self.arm,"geometry":base.gradient_geometry(grads),"probe_points":points,
               "actual_update_including_roundtrip":geo.displacement_metrics(),
               "final_gradient_norm":audit.norm(final_gradient),"training_closure_count":closure_count,
               "per_layer_gradient_norms":{k:geo.layer_norms(v) for k,v in grads.items()},
               "projector_vs_raw_direction_relative_error":audit.norm(final_gradient-selected)/max(audit.norm(selected),1e-20),
               "base_sgd_audit":self.sample_base_audit,"step_audit":self.trace_rows[-1],
               "note":"Direction/decomposition/projector/SGD checked against their own exact operations every step."}
        if audit.nonfinite_paths(row):
            raise FloatingPointError("Nonfinite sampled formation geometry")
        self.records.append(row)
        audit.atomic_json(self.out/"geometry.json",{"records":self.records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source","config","output-dir"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--max-epochs",type=int,default=0)
    parser.add_argument("--max-batches",type=int,default=0)
    cli = parser.parse_args()
    cli.source,cli.config,cli.output_dir = [str(Path(x).resolve()) for x in (cli.source,cli.config,cli.output_dir)]
    source,out = Path(cli.source),Path(cli.output_dir)
    if not (source/"src/trainer.py").is_file() or not (source/"models_LoRAbasedCL/baseLoRA.py").is_file():
        raise FileNotFoundError("Frozen source layout mismatch")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be fresh and isolated")
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    os.chdir(source)
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    from utils.data_manager import DataManager
    from utils.random_reproduce import set_random,json_safe
    from utils import factory
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",force=True)
    cfg = resolve_config(json.loads(Path(cli.config).read_text()),cli)
    set_random(cfg["seed"])
    torch.cuda.set_device(0)
    dm = DataManager(cfg["dataset"],False,cfg["seed"],20,20,cfg)
    cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
    if dm.nb_tasks != 10:
        raise AssertionError("Task partition must remain native ten-task/20-class increments")
    audit.atomic_json(out/"effective_config.json",json_safe(cfg))
    learner = factory.get_model(cfg["model_name"],cfg)
    controller = FormationController(learner,dm,cfg,cli)
    controller.install()
    summary = {"status":"running","completed_epochs":0,"task":0,"trained_tasks":[0],
        "formation_schema_version":1,"formation_arm":cfg["formation_arm"],"seed":cfg["seed"],
        "direction":cfg["stage1_direction"],"r":cfg["gam_grad_norm_rho"],
        "smoke":bool(cli.max_epochs or cli.max_batches),"scheduler_T_max":50,
        "trainer_sha256":audit.digest(__file__),"effective_step_sha256":audit.digest(Path(__file__).with_name("effective_step.py")),
        "base_trainer_sha256":audit.digest(Path(base.__file__)),"source_gam_sha256":controller.source_sha,
        "reference_trace_sha256":controller.reference.sha256 if controller.reference else None}
    audit.atomic_json(out/"final_summary.json",summary)
    try:
        learner.incremental_train(dm)
        if len(controller.rows) != cfg["init_epoch"] or len(controller.trace_rows) != controller.counters["training_batches"]:
            raise AssertionError("Missing epoch/step trace callbacks")
        snapshots = [0]+[e for e in range(1,cfg["init_epoch"]+1) if e%10 == 0 or e == cfg["init_epoch"]]
        summary.update(status="completed",completed_epochs=len(controller.rows),final=controller.rows[-1],
            counters=controller.counters,elapsed_seconds=time.time()-controller.started,
            initial_identity=controller.identity,checkpoint_epochs=snapshots,
            step_trace_sha256=audit.digest(controller.trace_path),step_trace_steps=len(controller.trace_rows))
        summary["finite_check"] = {"passed":not audit.nonfinite_paths(controller.rows+controller.records+controller.trace_rows)}
        summary["checkpoint_checks"] = {str(e):all((out/"checkpoints"/("epoch_%03d"%e)/name).is_file()
            for name in ("lora_w_a_0.pt","lora_w_b_0.pt","lora_meta_0.json","fc_state_0.pt","training_state.pt","manifest.json")) for e in snapshots}
        summary["task_check"] = {"cur_task":learner._cur_task,"total_classes":learner._total_classes,
                                  "passed":learner._cur_task == 0 and learner._total_classes == 20}
        summary["step_checks"] = {"passed":all(r["self_spec_passed"] and r["closure_count"]==4 and r["matched"] is not False for r in controller.trace_rows),
            "steps":len(controller.trace_rows),"matched_steps":sum(r["matched"] is True for r in controller.trace_rows),
            "max_match_relative_error":max((r["actual_match_relative_error"] for r in controller.trace_rows if r["matched"] is not None),default=None)}
        if not all((summary["finite_check"]["passed"],all(summary["checkpoint_checks"].values()),summary["task_check"]["passed"],summary["step_checks"]["passed"])):
            raise AssertionError("Final formation acceptance failed")
    except Exception as exc:
        summary.update(status="failed",error=repr(exc),completed_epochs=len(controller.rows))
        raise
    finally:
        controller.trace_file.close()
        audit.atomic_json(out/"final_summary.json",summary)


if __name__ == "__main__":
    main()
