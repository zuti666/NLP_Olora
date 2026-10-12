#!/usr/bin/env python3
"""Read-only same-state probe-selection and response-reading diagnostics.

Loads each task's native end checkpoint, then overlays its exact entry/epoch1/
end trainable observation.  Batches are registered before gradient diagnostics.
No optimizer object is created and no actual update is submitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from direction_step import FourProbeStep, vector_cosine, vector_norm
from effective_step import EffectiveStep
from hvp_gate import model_sha, pure_config, seed_all, sha, tensor_sha, write_json
from response_hvp import State, compare, response_observer, rng_get, rng_set

TASKS = (0,1,4,9)
POINTS = ("entry","epoch1","end")
BRANCHES = ("P","D","C","R")


def trainable_sha(model):
    digest = hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(str(tuple(p.shape)).encode())
            digest.update(str(p.dtype).encode())
            digest.update(p.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def overlay_observation(model, payload, task, point):
    """Validate every value before changing any parameter; preserve identities."""
    if not isinstance(payload,dict) or payload.get("task") != task or payload.get("point") != point:
        raise ValueError("Observation task/point mismatch")
    if not isinstance(payload.get("config"),dict) or "rng" not in payload:
        raise ValueError("Observation must contain config and archived RNG")
    values = payload.get("trainable_state")
    current = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if not isinstance(values,dict) or set(values) != set(current):
        provided = set(values) if isinstance(values,dict) else set()
        missing = sorted(set(current)-provided)
        extra = sorted(provided-set(current))
        raise ValueError("Trainable name mismatch; missing="+repr(missing)+" extra="+repr(extra))
    for name,p in current.items():
        value = values[name]
        if not torch.is_tensor(value) or value.shape != p.shape or value.dtype != p.dtype:
            raise ValueError("Trainable shape/dtype mismatch: "+name)
        if value.requires_grad or value.grad_fn is not None or not bool(torch.isfinite(value).all()):
            raise ValueError("Trainable observation must be detached and finite: "+name)
    identities = {name:id(p) for name,p in current.items()}
    with torch.no_grad():
        for name,p in current.items():
            p.copy_(values[name].to(device=p.device))
    if not all(id(current[name])==identities[name] and torch.equal(current[name],values[name].to(current[name].device))
               for name in current):
        raise AssertionError("Observation overlay was not exact")
    return {"trainable_state_sha256":trainable_sha(model),"tensor_count":len(current),
            "parameter_count":sum(p.numel() for p in current.values()),
            "names_shapes_dtypes_exact":True,"parameter_identities_preserved":True}


def _direction_comparison(candidate, reference):
    stats = compare(candidate,reference)
    # E/C/R definitions are explicitly relative to the named reference.
    return {"relative_error":stats["relative_l2"],"cosine":stats["cosine"],
            "norm_ratio":stats["norm_ratio"],"candidate_norm":stats["first_norm"],
            "reference_norm":stats["reference_norm"],"near_zero":stats["degenerate"]}


def audit_same_state(model, inputs, targets, closure, geometry, backend="native",
                     responses=False, current_lr=.02):
    """Strict transaction: all candidates share parameters, batch and RNG R0."""
    live = State(model)
    initial_trainable = trainable_sha(model)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    report = {}
    retained = {}
    engines = []
    started = time.perf_counter()
    try:
        selected = ("P",) if responses else BRANCHES
        for branch in selected:
            live.restore()
            engine = FourProbeStep(model,params,geometry,
                                  {"branch":branch,"direction":"A","backend":backend,
                                   "rho":.05,"radius":.25,"eps":1e-12})
            engines.append(engine)
            result = engine.compute(inputs,targets,closure,current_lr)
            context = result["context"]
            engine.discard()
            if result["record"]["sgd_updates"] != 0 or result["record"]["fisher_calls"] != 0 or result["record"]["weight_merge_calls"] != 0:
                raise AssertionError("Read-only diagnostic submitted forbidden work")
            if responses:
                sec = context["c21"]
                U_sec = context["U"]
                candidates = {"Sec":{"direction_norm":vector_norm(U_sec),
                                     "response_norm":vector_norm(sec),
                                     "response_vs_Sec":{"relative_error":0.,"cosine":1. if vector_norm(sec)>1e-30 else None,
                                                        "norm_ratio":1. if vector_norm(sec)>1e-30 else None,
                                                        "near_zero":vector_norm(sec)<=1e-30},
                                     "cosine_U_g0":vector_cosine(U_sec,context["grads"]["g0"]),
                                     "cosine_U_g1_plus":vector_cosine(U_sec,context["grads"]["g1_plus"])}}
                for label,response in (("Hstart","hstart"),("Hmid","hmid")):
                    U, observer_report = response_observer(context,response)
                    raw = observer_report["hvp_vs_sec"]
                    candidates[label] = {"direction_norm":vector_norm(U),
                                         "response_norm":raw["first_norm"],
                                         "response_vs_Sec":{"relative_error":raw["relative_l2"],
                                                            "cosine":raw["cosine"],"norm_ratio":raw["norm_ratio"],
                                                            "candidate_norm":raw["first_norm"],
                                                            "reference_norm":raw["reference_norm"],"near_zero":raw["degenerate"]},
                                         "direction_vs_Sec":_direction_comparison(U,U_sec),
                                         "cosine_U_g0":vector_cosine(U,context["grads"]["g0"]),
                                         "cosine_U_g1_plus":vector_cosine(U,context["grads"]["g1_plus"]),
                                         "observer":observer_report}
                report = {"status":"passed","backend":backend,"mode":"same_state_response_reading",
                          "ordinary_probes":result["record"],"responses":candidates,
                          "d12_source":"actual detached phi2 minus phi1plus",
                          "d12_norm":vector_norm(context["d12"]),
                          "same_endpoint_rule":True,"training_trajectories_compared":False}
            else:
                retained[branch] = context
                report.setdefault("branches",{})[branch] = result["record"]
            live.restore()
            live.check()
        if not responses:
            reference = retained["P"]
            identities = {}
            for branch,ctx in retained.items():
                identities[branch] = {label:all(torch.equal(a,b) for a,b in zip(
                    ctx["grads"][label],reference["grads"][label]))
                    for label in ("g0","g1_plus","g1_minus")}
                if not all(identities[branch].values()):
                    raise AssertionError("Four cells did not share exact g0/g1+/g1-: "+repr(identities))
            contrasts = {}
            for left,right in (("P","C"),("P","D"),("P","R"),("C","D"),("D","R"),("C","R")):
                contrasts[left+"_vs_"+right] = {
                    "O_g2":_direction_comparison(retained[left]["grads"]["g2"],retained[right]["grads"]["g2"]),
                    "A":_direction_comparison(retained[left]["U"],retained[right]["U"]),
                    "c21":_direction_comparison(retained[left]["c21"],retained[right]["c21"]),
                    "outer_q":_direction_comparison(retained[left]["q"],retained[right]["q"])}
            report.update(status="passed",backend=backend,mode="same_state_four_cell_selection",
                          base_gradient_identities=identities,cross_cell_contrasts=contrasts,
                          exact_same_parameter_batch_rng=True)
        report.update(seconds=time.perf_counter()-started,optimizer_updates=0,fisher_calls=0,
                      weight_merge_calls=0,hvp_calls=2 if responses else 0,
                      create_graph_gradient_calls=2 if responses else 0,
                      additional_ordinary_reference_gradient_calls=1 if responses else 0,
                      base_ordinary_gradient_calls=4 if responses else 16)
        return report
    finally:
        for engine in engines:
            engine.discard()
        live.restore()
        restored = live.check()
        if trainable_sha(model) != initial_trainable:
            raise AssertionError("Read-only mechanism diagnostic changed trainable values")
        if report:
            report["entry_state_restore"] = restored
            report["trainable_values_unchanged"] = True


def _validate_config(raw):
    required = {"dataset":"imageneta","init_cls":20,"increment":20,"lora_rank":10,
                "gam_grad_rho":.05,"gam_grad_norm_rho":.25,"ewc_lambda":0.,
                "ewc_skip_task_fisher":True,"in_task_weight_merge":False,
                "task_endpoint_weight_merge":False,"approximate_fisher_in_task":False,
                "compute_full_fisher":False,"gam_grad_gamma":0.}
    for key,value in required.items():
        if raw.get(key) != value:
            raise ValueError("Reference protocol mismatch: "+key+"="+repr(raw.get(key)))
    seed = raw.get("seed")
    if seed not in (2,[2]):
        raise ValueError("This frozen protocol requires seed2")


def _pinned_assets(reference, tasks):
    pins = {"observations":{},"checkpoints":{}}
    manifest = {}
    for task in tasks:
        directory = reference/"checkpoints"/("task_%03d"%task)
        mpath = directory/"manifest.json"
        data = json.loads(mpath.read_text())
        if data.get("task") != task or data.get("seed") != 2:
            raise ValueError("Native checkpoint seed/task mismatch")
        files = {p.name:sha(p) for p in directory.iterdir() if p.is_file()}
        if not data.get("files_sha256") or any(files.get(name)!=h for name,h in data["files_sha256"].items()):
            raise AssertionError("Native checkpoint file hash mismatch")
        pins["checkpoints"][str(task)] = files
        manifest[task] = data
        for point in POINTS:
            path = reference/"observations"/("task_%03d_%s.pt"%(task,point))
            pins["observations"][str(path.relative_to(reference))] = sha(path)
    return pins,manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--reference-output",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--backend",choices=("native","math"),default="native")
    parser.add_argument("--response-diagnostics",action="store_true")
    parser.add_argument("--tasks",nargs="+",type=int,default=list(TASKS))
    parser.add_argument("--max-batches",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    if len(set(args.tasks))!=len(args.tasks) or any(t not in TASKS for t in args.tasks):
        raise ValueError("Invalid or duplicate task budget")
    if args.max_batches not in (1,2) or args.batch_size!=128:
        raise ValueError("Frozen budget is at most two batches of 128 per task")
    source = Path(args.source).resolve()
    reference = Path(args.reference_output).resolve()
    out = Path(args.output_dir).resolve()
    target = out/("response_mechanism_report.json" if args.response_diagnostics else "stage1_mechanism_report.json")
    registry_path = out/"batch_registry.json"
    if target.exists() or registry_path.exists():
        raise FileExistsError("Refusing to overwrite mechanism diagnostic artifacts")
    raw_path = reference/"effective_config.json"
    raw = json.loads(raw_path.read_text())
    _validate_config(raw)
    assets,checkpoint_manifests = _pinned_assets(reference,args.tasks)
    source_pins = {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*.py"))}
    code_dir = Path(__file__).parent
    code_pins = {name:sha(code_dir/name) for name in ("mechanism_diagnostics.py","direction_step.py",
                 "effective_step.py","response_hvp.py","audit_core.py","hvp_gate.py")}
    entry_rng = rng_get()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type=="cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    sys.path.insert(0,str(source))
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.chdir(source)
    from utils.data_manager import DataManager
    from utils import factory
    cfg = pure_config(raw,device,out)
    cfg["attention_backend"] = args.backend
    report = {"schema_version":1,"status":"preparing","mode":"response" if args.response_diagnostics else "four_cell",
              "source":str(source),"reference_output":str(reference),"reference_training_backend":raw.get("attention_backend","native"),
              "diagnostic_backend":args.backend,"seed":2,"tasks":args.tasks,"points":list(POINTS),
              "source_python_sha256":source_pins,"diagnostic_code_sha256":code_pins,
              "config_sha256":sha(raw_path),"reference_assets_sha256":assets,
              "optimizer_updates":0,"fisher_calls":0,"weight_merge_calls":0,
              "scope":"common reference checkpoints; same-state candidates, not training trajectory comparisons",
              "observation_rng":"archived and pinned; diagnostics use separate predeclared RNG stream",
              "state_results":[],"expected_state_batch_count":0}
    write_json(target,report)
    try:
        seed_all(2)
        dm = DataManager(cfg["dataset"],cfg.get("class_shuffle",False),2,cfg["init_cls"],cfg["increment"],cfg)
        cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
        if dm.nb_tasks!=10 or dm.nb_classes!=200 or any(i!=20 for i in dm._increments):
            raise ValueError("Dataset is not ImageNet-A 10x20")
        batches = {}
        registry = {"schema_version":1,"registered_before_direction_diagnostics":True,
                    "data_seed_rule":"seed*100000+81000+task",
                    "forward_seed_rule":"seed*100000+82000+task*10+batch",
                    "batch_size":128,"max_batches":args.max_batches,"tasks":{}}
        # Complete input registration across all selected tasks before computing
        # any scientific gradient or looking at candidate-direction statistics.
        for task in args.tasks:
            start,end = dm.get_task_class_range(task)
            if (start,end)!=(20*task,20*(task+1)):
                raise ValueError("Class range mismatch")
            data_seed = 2*100000+81000+task
            seed_all(data_seed)
            dataset = dm.get_dataset(np.arange(start,end),source="train",mode="train")
            loader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0,
                                generator=torch.Generator().manual_seed(data_seed),drop_last=False)
            entries = []
            batches[task] = []
            for index,(ids,x,y) in enumerate(loader):
                if index >= args.max_batches:
                    break
                if bool(((y<start)|(y>=end)).any()):
                    raise ValueError("Registration labels outside current task")
                identity = {"batch":index,"samples":int(y.numel()),"indices":ids.tolist(),
                            "input_sha256":tensor_sha(x),"target_sha256":tensor_sha(y),
                            "indices_sha256":tensor_sha(ids),"data_seed":data_seed,
                            "forward_seed":2*100000+82000+task*10+index}
                entries.append(identity)
                batches[task].append((ids.detach().clone(),x.detach().clone(),y.detach().clone(),identity))
            if not entries:
                raise ValueError("No real training batches available")
            registry["tasks"][str(task)] = {"class_range":[start,end],"dataset_samples":len(dataset),"batches":entries}
            del loader,dataset
        write_json(registry_path,registry)
        report.update(status="running",batch_registry_sha256=sha(registry_path),
                      expected_state_batch_count=3*sum(len(v) for v in batches.values()))
        write_json(target,report)
        for task in args.tasks:
            seed_all(2)
            learner = factory.get_model(cfg["model_name"],cfg)
            learner._unwrap_network().backbone.to(device)
            checkpoint = reference/"checkpoints"/("task_%03d"%task)
            learner.load_task_checkpoint(str(checkpoint),task,dm)
            model = learner._unwrap_network().to(device)
            model.train()
            if model_sha(model) != checkpoint_manifests[task]["model_sha256"]:
                raise AssertionError("Native restored model hash mismatch")
            for module in model.modules():
                if bool(getattr(module,"_register_delta_hook",False)) or getattr(module,"external_delta_w_q",None) is not None or getattr(module,"external_delta_w_v",None) is not None:
                    raise AssertionError("Native reference contains external delta/hooks")
            origin = State(model)
            geometry = EffectiveStep.from_model(model)
            start,end = dm.get_task_class_range(task)
            for point in POINTS:
                origin.restore()
                path = reference/"observations"/("task_%03d_%s.pt"%(task,point))
                payload = torch.load(path,map_location="cpu",weights_only=False)
                _validate_config(payload.get("config",{}))
                overlay = overlay_observation(model,payload,task,point)
                observed_hash = trainable_sha(model)
                if point=="end" and model_sha(model)!=checkpoint_manifests[task]["model_sha256"]:
                    raise AssertionError("End observation does not equal native end checkpoint")
                result = {"task":task,"point":point,"observation_path":str(path),
                          "observation_sha256":sha(path),"overlay":overlay,
                          "class_slice":[start,end],"batches":[]}
                report["state_results"].append(result)
                for ids,x,y,identity in batches[task]:
                    if tensor_sha(x)!=identity["input_sha256"] or tensor_sha(y)!=identity["target_sha256"] or tensor_sha(ids)!=identity["indices_sha256"]:
                        raise AssertionError("Preregistered data batch changed")
                    batch_rng = rng_get()
                    seed_all(identity["forward_seed"])
                    def closure(inputs,targets):
                        output = model(inputs)
                        logits = output["logits"] if isinstance(output,dict) else output
                        if logits.shape[1]!=end:
                            raise AssertionError("Seen classifier shape differs from current-task observation")
                        return F.cross_entropy(logits[:,start:end],targets-start),logits
                    try:
                        diagnostics = audit_same_state(model,x.to(device),y.to(device),closure,geometry,
                                                        args.backend,args.response_diagnostics,
                                                        .04 if task==0 else .02)
                    finally:
                        rng_set(batch_rng)
                    if trainable_sha(model)!=observed_hash:
                        raise AssertionError("Diagnostic changed observation parameters")
                    result["batches"].append({"identity":identity,**diagnostics})
                    write_json(target,report)
                    print(json.dumps({"event":"mechanism_batch_complete","task":task,"point":point,
                                      "batch":identity["batch"],"mode":report["mode"],"status":"passed"}),flush=True)
                if sha(path)!=assets["observations"][str(path.relative_to(reference))]:
                    raise AssertionError("Observation archive changed")
            origin.restore()
            origin.check()
            if model_sha(model)!=checkpoint_manifests[task]["model_sha256"]:
                raise AssertionError("Read-only diagnostic changed native origin")
            del learner,model,geometry,origin
            if device.type=="cuda":
                torch.cuda.empty_cache()
        after_assets,_ = _pinned_assets(reference,args.tasks)
        if after_assets!=assets or sha(raw_path)!=report["config_sha256"]:
            raise AssertionError("Reference files changed during diagnostics")
        if {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*.py"))}!=source_pins:
            raise AssertionError("Frozen source changed during diagnostics")
        if {name:sha(code_dir/name) for name in code_pins}!=code_pins:
            raise AssertionError("Diagnostic implementation changed during execution")
        total = sum(len(row["batches"]) for row in report["state_results"])
        if total!=report["expected_state_batch_count"]:
            raise AssertionError("Incomplete state/batch budget")
        report.update(status="passed",state_batch_count=total,
                      candidate_count=3*total if args.response_diagnostics else 4*total,
                      source_unchanged=True,reference_assets_unchanged=True,
                      diagnostic_code_unchanged=True,
                      native_origin_restored=True)
    except BaseException as error:
        report.update(status="failed",error=repr(error))
        raise
    finally:
        rng_set(entry_rng)
        write_json(target,report)
    print(json.dumps({"status":report["status"],"report":str(target),"batch_registry":str(registry_path)}))


if __name__=="__main__":
    main()
