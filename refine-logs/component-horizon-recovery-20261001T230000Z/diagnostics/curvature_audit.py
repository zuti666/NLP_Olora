#!/usr/bin/env python3
"""Read-only finite-response/HVP audit at actual native GAM closure states.

No optimizer update is committed.  Low curvature alignment is a valid finding,
not a failed job.  This audits trainable-factor coordinates, not BA Hessians.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def tensor_sha(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def parameter_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def pin_tree(directory,pattern="**/*"):
    directory=Path(directory)
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.glob(pattern)) if p.is_file()}


def verify_tree(directory,pins,pattern="**/*"):
    if pin_tree(directory,pattern)!=pins:
        raise AssertionError("Read-only input tree changed: "+str(directory))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_get():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
            [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_set(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
    if s[3]:torch.cuda.set_rng_state_all(s[3])


def rng_equal(a,b):
    return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
            and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and len(a[3])==len(b[3])
            and all(torch.equal(x,y) for x,y in zip(a[3],b[3])))


class State:
    """Trainable tensors, grads, buffers, modes, BN controls, and RNG snapshot.

    Frozen tensors cannot be altered by the audit; version counters hard-gate
    any such mutation. Full checkpoint files are separately hash-pinned.
    """
    def __init__(self,model):
        self.model=model
        self.params=[(p,p.detach().clone(),None if p.grad is None else p.grad.detach().clone())
                     for p in model.parameters() if p.requires_grad]
        self.frozen=[(p,p._version) for p in model.parameters() if not p.requires_grad]
        self.buffers=[(b,b.detach().clone()) for b in model.buffers()]
        self.modes=[(m,m.training) for m in model.modules()]
        self.bn=[(m,m.momentum,hasattr(m,"backup_momentum"),getattr(m,"backup_momentum",None))
                 for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        self.rng=rng_get()

    @torch.no_grad()
    def restore(self):
        if any(p._version!=v for p,v in self.frozen):
            raise AssertionError("Audit mutated a frozen parameter")
        for p,value,grad in self.params:
            p.copy_(value);p.grad=None if grad is None else grad.clone()
        for b,value in self.buffers:b.copy_(value)
        for m,training in self.modes:m.training=training
        for m,momentum,had_backup,backup in self.bn:
            m.momentum=momentum
            if had_backup:m.backup_momentum=backup
            elif hasattr(m,"backup_momentum"):delattr(m,"backup_momentum")
        rng_set(self.rng)

    def check(self):
        checks={"parameters_exact":all(torch.equal(p,v) for p,v,_ in self.params),
                "frozen_versions_exact":all(p._version==v for p,v in self.frozen),
                "buffers_exact":all(torch.equal(b,v) for b,v in self.buffers),
                "modes_exact":all(m.training==t for m,t in self.modes),
                "gradients_exact":all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad,g)) for p,_,g in self.params),
                "rng_exact":rng_equal(self.rng,rng_get())}
        if not all(checks.values()):raise AssertionError("Audit restore failed: "+repr(checks))
        return checks


def finite(values,label):
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(label+" has nonfinite values")


def dot(a,b):
    result=float(sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b)))
    if not math.isfinite(result):raise FloatingPointError("Nonfinite inner product")
    return result


def norm(v):return math.sqrt(max(0.,dot(v,v)))


def difference(a,b,scale=1.):return [(x-y)/scale for x,y in zip(a,b)]


def compare(a,b):
    finite(a+b,"vector comparison")
    na,nb=norm(a),norm(b)
    return {"first_norm":na,"reference_norm":nb,
            "cosine":dot(a,b)/(na*nb) if na>1e-30 and nb>1e-30 else None,
            "relative_l2":norm(difference(a,b))/nb if nb>1e-30 else None,
            "norm_ratio":na/nb if nb>1e-30 else None,
            "max_abs":max(float((x-y).abs().max()) for x,y in zip(a,b)),
            "degenerate":na<=1e-30 or nb<=1e-30}


def native_unit(grads):
    """Exact source GAM's sequential FP32 sum(g**2), with its epsilon."""
    n=None
    for g in grads:
        term=torch.sum(g**2)
        n=term if n is None else n+term
    n=torch.sqrt(n)
    if not bool(torch.isfinite(n)) or float(n)<=0:
        raise ValueError("Undefined zero/nonfinite probe direction")
    return [(g.detach()/(n+1e-12)).detach() for g in grads]


@contextlib.contextmanager
def backend_context(backend):
    if backend=="native":
        yield
    elif backend=="math":
        # Local dispatch context, never modifies model/frozen source or training.
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,
                                           enable_mem_efficient=False,enable_cudnn=False):
            yield
    else:raise ValueError("Unknown HVP backend")


def logits(model,x):
    out=model(x)
    return out["logits"] if isinstance(out,dict) else out


def task_loss(model,x,y,start,end):
    out=logits(model,x)
    if out.ndim!=2 or out.shape[1]!=end or bool(((y<start)|(y>=end)).any()):
        raise ValueError("Wrong current-task CE slice or classifier width")
    return F.cross_entropy(out[:,start:end],y-start)


def gradient_at(state,x,y,start,end,displacement=None,backend="native",need_hvp=False,direction=None):
    state.restore()
    params=[p for p,_,_ in state.params]
    try:
        if displacement is not None:
            with torch.no_grad():
                for p,delta in zip(params,displacement):p.add_(delta)
        with backend_context(backend),torch.enable_grad():
            loss=task_loss(state.model,x,y,start,end)
            g=torch.autograd.grad(loss,params,create_graph=need_hvp,allow_unused=False)
            finite(list(g),"same-point CE gradient")
            values=[v.detach().clone() for v in g]
            h=None
            if need_hvp:
                if direction is None or len(direction)!=len(params):raise ValueError("HVP direction missing")
                # Direction is fixed. Differentiating its normalization is forbidden.
                fixed=[v.detach() for v in direction]
                inner=sum((v*u).sum() for v,u in zip(g,fixed))
                if inner.requires_grad:
                    hv=torch.autograd.grad(inner,params,allow_unused=True)
                    h=[torch.zeros_like(p) if v is None else v.detach().clone() for p,v in zip(params,hv)]
                else:h=[torch.zeros_like(p) for p in params]
                finite(h,"autograd HVP")
        return float(loss.detach()),values,h
    finally:
        state.restore();state.check()


def assert_same_gradient(observed,reference,tolerance=2e-5):
    stats=compare(observed,reference)
    if stats["relative_l2"] is None:
        if stats["max_abs"]>1e-10:raise AssertionError("Degenerate same-point gradient mismatch")
    elif stats["relative_l2"]>tolerance:
        raise AssertionError("Same-point CE gradient mismatch: "+repr(stats))
    return stats


def unsupported_double_backward(exc):
    text=str(exc).lower()
    return ("derivative" in text or "double backward" in text or "backward" in text) and (
        "not implemented" in text or "not supported" in text or "does not support" in text)


def audited_hvp(state,x,y,start,end,direction,native_gradient,requested_backend):
    attempts=[]
    backends=("native","math") if requested_backend=="auto" else (requested_backend,)
    for backend in backends:
        try:
            loss,g,h=gradient_at(state,x,y,start,end,backend=backend,need_hvp=True,direction=direction)
            agreement=assert_same_gradient(g,native_gradient)
            return h,{"status":"available","backend":backend,"same_point_gradient":agreement,
                      "same_point_ce":loss,"attempts":attempts,"direction_detached":True}
        except RuntimeError as exc:
            if not unsupported_double_backward(exc):raise
            attempts.append({"backend":backend,"reason":str(exc),"unsupported_double_backward":True})
    return None,{"status":"unsupported","backend":None,"attempts":attempts,
                 "direction_detached":True,"interpretation":"Finite differences only; no autograd-HVP verification claim"}


def capture_native(learner,GAM,x,y,start,end,rho,radius):
    model=learner._unwrap_network()
    anchor=State(model)
    params=[p for p in model.parameters() if p.requires_grad]
    base=torch.optim.SGD(params,lr=0.,momentum=0.,weight_decay=0.)
    opt=GAM(params,base_optimizer=base,model=model,grad_rho=rho,grad_norm_rho=radius,
            adaptive=False,args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.,use_sgv=False))
    frames=[];grads=[];losses=[];forward_count=[0]
    hook=model.register_forward_hook(lambda *_:forward_count.__setitem__(0,forward_count[0]+1))
    native_step=opt.step
    def tracked_step(closure=None):
        if closure is None:raise AssertionError("Missing native closure")
        def tracked():
            frames.append(State(model)) # true state immediately before this closure
            result=closure()
            if any(p.grad is None for p in params):raise AssertionError("Missing native probe gradient")
            gg=[p.grad.detach().clone() for p in params]
            finite(gg,"native GAM gradient")
            grads.append(gg);losses.append(float(result[1]))
            if not math.isfinite(losses[-1]):raise FloatingPointError("Nonfinite native probe CE")
            return result
        return native_step(closure=tracked)
    opt.step=tracked_step
    try:
        model.train()
        _,_,log=learner._step_batch_with_log(opt,x,y,class_offset=start)
        if len(frames)!=4 or forward_count[0]!=5 or float(log["ewc"])!=0. or float(log["delta_w_penalty"])!=0.:
            raise AssertionError("Expected pure native clean+four probes and zero penalties")
        # Actual frozen operations, including native multiply-by-scale rounding.
        def scale_native(gradient,r):
            n=None
            for g in gradient:
                term=torch.sum(g**2);n=term if n is None else n+term
            scale=r/(torch.sqrt(n)+1e-12)
            return [g*scale.to(g) for g in gradient]
        e0=scale_native(grads[0],rho)
        d10=[g1.clone().sub_(g0) for g0,g1 in zip(grads[0],grads[1])]
        outer=scale_native(d10,radius)
        e2=scale_native(grads[2],rho)
        values=[[v for _,v,_ in frame.params] for frame in frames]
        expected1=[p.clone().add_(e) for p,e in zip(values[0],e0)]
        expected2=[p.clone().sub_(a).add_(b) for p,a,b in zip(values[1],e0,outer)]
        expected3=[p.clone().add_(e) for p,e in zip(values[2],e2)]
        point_checks={"g1_native_perturb_exact":all(torch.equal(a,b) for a,b in zip(expected1,values[1])),
                      "g2_native_roundtrip_outer_exact":all(torch.equal(a,b) for a,b in zip(expected2,values[2])),
                      "g3_native_perturb_exact":all(torch.equal(a,b) for a,b in zip(expected3,values[3]))}
        if not all(point_checks.values()):raise AssertionError("Native closure point specification mismatch")
        mathematical2=[p+b for p,b in zip(values[0],outer)]
        return frames,grads,{"closure_count":4,"training_ce_gradient_passes":5,"training_forward_count":5,
                            "native_point_checks":point_checks,"probe_ce":losses,
                            "outer_point_vs_ideal_without_roundtrip":compare(values[2],mathematical2),
                            "native_step_lr":0.,"source_loss":"frozen learner current-task sliced CE"}
    finally:
        hook.remove();anchor.restore();anchor.check()


def audit_batch(learner,GAM,x,y,start,end,rho=.05,radius=.25,epsilons=(.05,.025,.0125),hvp_backend="auto"):
    model=learner._unwrap_network()
    stochastic=[n for n,m in model.named_modules() if
                (isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p>0) or float(getattr(m,"drop_prob",0.) or 0.)>0]
    if stochastic:raise ValueError("This audit requires disabled dropout/drop-path: "+repr(stochastic))
    anchor=State(model)
    try:
        frames,grads,native=capture_native(learner,GAM,x,y,start,end,rho,radius)
        result={"native":native,"rho":rho,"r":radius,"epsilons":list(epsilons),"responses":{}}
        delta_s=difference(grads[1],grads[0]);delta_g=difference(grads[3],grads[2])
        result["deltaG_vs_deltaS"]=compare(delta_g,delta_s)
        for name,index,delta in (("original",0,delta_s),("outer",2,delta_g)):
            frame=frames[index]
            u=native_unit(grads[index])
            baseline_ce,baseline,_=gradient_at(frame,x,y,start,end)
            same_point=assert_same_gradient(baseline,grads[index])
            if abs(baseline_ce-native["probe_ce"][index])>1e-5*max(1.,abs(native["probe_ce"][index])):
                raise AssertionError("Same-point CE differs from native closure loss")
            hvp,hvp_info=audited_hvp(frame,x,y,start,end,u,grads[index],hvp_backend)
            if hvp_info["status"]=="available" and abs(hvp_info["same_point_ce"]-baseline_ce)>1e-5*max(1.,abs(baseline_ce)):
                raise AssertionError("Autograd-HVP backend CE differs from replayed native CE")
            native_response=[d/rho for d in delta]
            scan=[]
            for eps in epsilons:
                if not math.isfinite(eps) or eps<=0:raise ValueError("FD epsilon must be positive")
                _,plus,_=gradient_at(frame,x,y,start,end,[eps*v for v in u])
                _,minus,_=gradient_at(frame,x,y,start,end,[-eps*v for v in u])
                fd=difference(plus,baseline,eps)
                central=difference(plus,minus,2*eps)
                scan.append({"epsilon":eps,"forward_vs_hvp":compare(fd,hvp) if hvp is not None else None,
                             "central_vs_hvp":compare(central,hvp) if hvp is not None else None,
                             "forward_vs_central":compare(fd,central),
                             "forward_vs_native_response_at_training_rho":compare(fd,native_response) if eps==rho else None})
            result["responses"][name]={"native_closure_index":index,"native_baseline_ce":native["probe_ce"][index],
                "replayed_baseline_ce":baseline_ce,"same_point_native_gradient":same_point,
                "native_normalization_direction_norm":norm(u),"hvp":hvp_info,
                "native_finite_response_vs_hvp":compare(native_response,hvp) if hvp is not None else None,
                "finite_difference_scan":scan}
        return result
    finally:
        anchor.restore();result_restore=anchor.check()
        if "result" in locals():result["restore"]=result_restore


def pure_config(raw,device,run_path):
    cfg=dict(raw)
    seed=cfg.get("seed",2)
    if isinstance(seed,list):
        if len(seed)!=1:raise ValueError("One seed per diagnostic")
        seed=seed[0]
    cfg.update(seed=int(seed),device=[device],filepath=str(Path(run_path)/"unused_native_state")+os.sep,
        train_num_workers=0,eval_num_workers=0,resume_require_experiment_identity=False,
        resume_require_model_sidecars=False,resume_require_fisher_state=False,
        optimizer_type="gam",lambda_flat=1.,ewc_lambda=0.,ewc_skip_task_fisher=True,
        in_task_weight_merge=False,task_endpoint_weight_merge=False,approximate_fisher_in_task=False,
        in_task_fisher_omega=0.,in_task_weight_merge_omega=0.,ewc_fisher_damping_alpha=0.,
        ewc_normalize_fisher=False,ewc_normalize_accumulated_fisher=False,ewc_dual_ascent=False,
        compute_full_fisher=False,posterior_state_save=False,mechanism_anchor_eval=False,
        mechanism_future_surrogate_eval=False,as1_delta_w_penalty=False,mechanism_eval=False,
        mechanism_grad_stats=False,memory_size=0,memory_per_class=0,fisher_split=False,
        gam_adaptive=False,gam_grad_gamma=0.,gam_use_sgv=False,use_sgv=False,sgv_lambda=0.)
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","config","checkpoint-dir","output"):p.add_argument("--"+name,required=True)
    p.add_argument("--task",type=int,choices=(0,4,9),required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--rho",type=float,default=.05)
    p.add_argument("--radius",type=float,default=.25)
    p.add_argument("--epsilons",default="0.05,0.025,0.0125")
    p.add_argument("--max-batches",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hvp-backend",choices=("auto","native","math"),default="auto")
    args=p.parse_args()
    args.source,args.config,args.checkpoint_dir,args.output=map(lambda x:str(Path(x).resolve()),
            (args.source,args.config,args.checkpoint_dir,args.output))
    out=Path(args.output);checkpoint=Path(args.checkpoint_dir);source=Path(args.source)
    if out.exists():raise FileExistsError("Refusing to overwrite curvature report")
    if args.max_batches<1 or args.batch_size<1 or args.rho<=0 or args.radius<0:raise ValueError("Invalid audit budget/radius")
    files=pin_tree(checkpoint)
    if not files:raise ValueError("Empty checkpoint directory")
    source_pins=pin_tree(source,"**/*.py")
    sys.path.insert(0,str(source));os.chdir(source);os.environ.setdefault("HF_HUB_OFFLINE","1")
    from utils.data_manager import DataManager
    from utils import factory
    from optimer_PerturabtionType.gam import GAM
    device=torch.device(args.device)
    cfg=pure_config(json.loads(Path(args.config).read_text()),device,out.parent)
    manifest=json.loads((checkpoint/"manifest.json").read_text())
    if (manifest.get("task")!=args.task or manifest.get("seed")!=cfg["seed"] or
            manifest.get("source_gam_sha256")!=source_pins.get("optimer_PerturabtionType/gam.py")):
        raise ValueError("Checkpoint manifest task/seed/source binding failed")
    if not manifest.get("files_sha256") or any(files.get(name)!=value for name,value in manifest["files_sha256"].items()):
        raise ValueError("Native checkpoint files differ from manifest hashes")
    for key in ("component_rule","component_scale"):
        if manifest.get(key)!=cfg.get(key):raise ValueError("Checkpoint/config component mismatch: "+key)
    seed_all(cfg["seed"])
    if device.type=="cuda":
        torch.cuda.set_device(device);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    dm=DataManager(cfg["dataset"],cfg.get("class_shuffle",False),cfg["seed"],cfg["init_cls"],cfg["increment"],cfg)
    cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
    learner=factory.get_model(cfg["model_name"],cfg)
    learner._unwrap_network().backbone.to(device)
    learner.load_task_checkpoint(str(checkpoint),args.task,dm)
    model=learner._unwrap_network().to(device)
    start,end=dm.get_task_class_range(args.task)
    if manifest.get("class_range")!=[0,end] or manifest.get("current_task_class_range")!=[start,end]:
        raise ValueError("Checkpoint manifest class range does not match native task")
    restored_sha=parameter_sha(model)
    if restored_sha!=manifest.get("model_sha256"):
        raise AssertionError("Native-restored model differs from saved parameter hash")
    if json.loads(json.dumps(model.backbone.w_meta))!=json.loads((checkpoint/("lora_meta_%d.json"%args.task)).read_text()):
        raise AssertionError("Restored LoRA metadata differs from native checkpoint")
    learner._cur_task,learner._known_classes,learner._total_classes=args.task,start,end
    finite(list(model.parameters())+list(model.buffers()),"restored model")
    dataset=dm.get_dataset(np.arange(start,end),source="train",mode="train")
    data_seed=cfg["seed"]*100000+70000+args.task
    seed_all(data_seed)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(data_seed),drop_last=False)
    if args.max_batches>len(loader):raise ValueError("Requested more batches than the native task contains")
    report={"schema_version":1,"status":"running","task":args.task,"seed":cfg["seed"],
            "checkpoint_dir":str(checkpoint),"checkpoint_files_sha256":files,"source_python_sha256":source_pins,
            "script_sha256":sha(__file__),"config_sha256":sha(args.config),"class_range":[start,end],
            "restored_model_sha256":restored_sha,"checkpoint_manifest":manifest,
            "coordinate_system":"trainable LoRA factors plus FC; not the effective-BA Hessian",
            "selection":"predeclared fixed batches, no high-alignment selection",
            "hvp_backend_requested":args.hvp_backend,"batches":[]}
    write_json(out,report)
    try:
        iterator=iter(loader)
        for index in range(args.max_batches):
            ids,x,y=next(iterator)
            result=audit_batch(learner,GAM,x.to(device),y.to(device),start,end,args.rho,args.radius,
                               tuple(float(v) for v in args.epsilons.split(",")),args.hvp_backend)
            report["batches"].append({"batch":index,"samples":len(y),"data_seed":data_seed,
                                      "input_sha256":tensor_sha(x),"target_sha256":tensor_sha(y),
                                      "indices":ids.tolist(),**result})
            write_json(out,report)
        verify_tree(checkpoint,files)
        verify_tree(source,source_pins,"**/*.py")
        if parameter_sha(model)!=restored_sha:raise AssertionError("Audit did not restore original model")
        report.update(status="completed",checkpoint_unchanged=True,source_unchanged=True,
                      hvp_all_available=all(r["hvp"]["status"]=="available" for b in report["batches"] for r in b["responses"].values()),
                      interpretation="Local finite-response/HVP diagnostics only; not evidence of long-horizon causal mediation")
    except Exception as exc:
        report.update(status="failed",error=repr(exc));raise
    finally:write_json(out,report)


if __name__=="__main__":main()
