#!/usr/bin/env python3
"""CPU-only contract/negative tests for future susceptibility forks."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace, MethodType

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import future_probe as future
import audit_core as audit


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_As = nn.ModuleList([nn.Linear(4,2,bias=False)])
        self.w_Bs = nn.ModuleList([nn.Linear(2,4,bias=False)])
        self.learn_alpha = False
    def forward(self,x):
        return torch.tanh(x + self.w_Bs[0](self.w_As[0](x)))


class Toy(nn.Module):
    def __init__(self,classes=40):
        super().__init__()
        self.backbone = ToyBackbone()
        self.fc = nn.Linear(4,classes)
        self.register_buffer("observed",torch.tensor(0.))
    def forward(self,x):
        # Deliberately impure buffer/RNG use exercises observational isolation.
        self.observed.add_(1.)
        if not self.training:
            random.random(); np.random.rand(); torch.rand(1)
        return {"logits":self.fc(self.backbone(x))}
    def update_fc(self,classes):
        old = self.fc
        self.fc = nn.Linear(4,classes)
        with torch.no_grad():
            self.fc.weight[:old.out_features].copy_(old.weight)
            self.fc.bias[:old.out_features].copy_(old.bias)


class ToyData(Dataset):
    def __init__(self,n=997,augment=True,offset=20,classes=20):
        self.n,self.augment,self.offset,self.classes=n,augment,offset,classes
        self.labels = np.asarray([offset+i%classes for i in range(n)])
    def __len__(self):
        return self.n
    def __getitem__(self,i):
        noise = (random.random()+np.random.rand()+float(torch.rand(1))) if self.augment else 0.
        return i,torch.tensor([i*.001,noise,.2,-.1],dtype=torch.float32),int(self.labels[i])


def expect_rejected(fn):
    try:
        fn()
    except (ValueError,AssertionError,FloatingPointError,ArithmeticError):
        return
    raise AssertionError("Negative test was incorrectly accepted")


def test_data_rng():
    data = ToyData()
    def trajectory(external_seed):
        audit.seed_all(external_seed)
        stream = future.EpochDataStream(data,128,2)
        rows=[]
        for epoch in (1,2):
            for batch,x,y,identity in stream.epoch(epoch):
                rows.append(identity)
                torch.rand(external_seed%5+1); random.random();np.random.rand(7)
        return rows
    a,b = trajectory(17),trajectory(29)
    assert a==b
    assert sorted(sum((r["indices"] for r in a[:8]),[])) == list(range(997))
    assert [len(r["indices"]) for r in a[:8]] == [128]*7+[101]
    assert math.ceil(997/128)==8 and 100/8==12.5


def test_head_rng():
    audit.seed_all(7)
    original = Toy(20)
    old = copy.deepcopy(original.state_dict())
    a,b=copy.deepcopy(original),copy.deepcopy(original)
    rng=audit.rng_get()
    one=future.expand_head(a,20,40,220000,"cpu")
    assert audit.rng_equal(rng,audit.rng_get())
    audit.seed_all(120)
    two=future.expand_head(b,20,40,220000,"cpu")
    assert one==two and torch.equal(a.fc.weight[:20],old["fc.weight"])
    assert torch.equal(a.fc.bias[:20],old["fc.bias"])


def test_observational():
    audit.seed_all(3)
    model=Toy(40).train()
    model.fc.eval()
    for p in model.parameters():
        p.grad=torch.randn_like(p)
    old=copy.deepcopy(model.state_dict())
    gradients=[p.grad.clone() for p in model.parameters()]
    modes=[m.training for m in model.modules()]
    rng=audit.rng_get()
    result=future.evaluate(model,ToyData(40,False),20,40,40,"cpu")
    assert result["diagnostic"]["local"]["samples"]==40
    assert audit.rng_equal(rng,audit.rng_get())
    assert modes==[m.training for m in model.modules()]
    assert all(torch.equal(old[k],v) for k,v in model.state_dict().items())
    assert all(torch.equal(a,p.grad) for a,p in zip(gradients,model.parameters()))


def split_fixture():
    labels=np.repeat(np.arange(20),3)
    entry={"dataset_size":len(labels),"labels_sha256":future.labels_hash(labels),
           "diagnostic_indices":list(range(0,len(labels),3)),
           "reporting_indices":[i for i in range(len(labels)) if i%3]}
    spec={"schema_version":1,"split_seed":1729,"task0":entry,"extra":"allowed"}
    return labels,spec


def test_split():
    labels,spec=split_fixture()
    valid=future.validate_split(spec,0,labels)
    assert len(valid["diagnostic"])==20 and len(valid["reporting"])==40
    for change in ("duplicate","overlap","hash","missing_class"):
        bad=copy.deepcopy(spec)
        if change=="duplicate":bad["task0"]["diagnostic_indices"].append(0)
        if change=="overlap":bad["task0"]["reporting_indices"].append(0)
        if change=="hash":bad["task0"]["labels_sha256"]="wrong"
        if change=="missing_class":bad["task0"]["diagnostic_indices"].remove(0)
        expect_rejected(lambda:future.validate_split(bad,0,labels))
    singleton_labels=np.append(labels,20)
    singleton=copy.deepcopy(spec)
    singleton["task0"].update(dataset_size=61,labels_sha256=future.labels_hash(singleton_labels))
    singleton["task0"]["reporting_indices"].append(60)
    singleton["task0"]["singleton_classes_reporting_only"]=[20]
    assert len(future.validate_split(singleton,0,singleton_labels)["reporting"])==41
    bad=copy.deepcopy(singleton)
    bad["task0"]["reporting_indices"].remove(60)
    bad["task0"]["diagnostic_indices"].append(60)
    expect_rejected(lambda:future.validate_split(bad,0,singleton_labels))
    ids=["image_%d"%i for i in range(len(labels))]
    spec["task0"]["canonical_ids_sha256"]=hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    future.validate_split(spec,0,labels,ids)
    expect_rejected(lambda:future.validate_split(spec,0,labels,ids[::-1]))


def test_path():
    path=future.PathIntegral(1.)
    for step,loss in enumerate((1.3,1.1,1.4,.9),1):
        row=path.append(step,loss)
    assert math.isclose(row["I"],-.1,abs_tol=1e-15)
    assert math.isclose(row["Iplus"],.6,abs_tol=1e-15)
    assert row["telescoping_error"]==0.
    expect_rejected(lambda:path.append(7,1.))
    expect_rejected(lambda:path.append(5,float("nan")))


def test_reference():
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)
        ident={k:k for k in ("input_sha256","target_sha256","indices_sha256")}
        ident.update(indices=[2,1],data_seed=210001)
        meta={"seed":2,"formation_arm":"s","formation_epoch":50,"future_rule":"s",
              "source_gam_sha256":"gam","selection_split_sha256":"split","smoke":False}
        row={"step_index":1,"epoch":1,"batch":0,"seed":2,"task":1,
             "identity":ident,"lr_nominal":.02,"actual_effective_delta":{"joint_norm":.3}}
        (path/"identity.json").write_text(json.dumps(meta))
        (path/"step_trace.jsonl").write_text(json.dumps(row)+"\n")
        receipt={"status":"completed","completed_steps":1,"step_trace_sha256":audit.digest(path/"step_trace.jsonl"),
                 "identity_sha256":audit.digest(path/"identity.json")}
        (path/"summary.json").write_text(json.dumps(receipt))
        ref=future.StepReference(path,2,1,"gam","split")
        assert ref.target(1,1,0,2,.02,ident)==.3
        expect_rejected(lambda:ref.target(1,2,0,2,.02,ident))
        expect_rejected(lambda:ref.target(1,1,0,2,.01,ident))
        expect_rejected(lambda:ref.target(1,1,0,2,.02,{**ident,"input_sha256":"other"}))
        expect_rejected(lambda:future.StepReference(path,1,1,"gam","split"))
        expect_rejected(lambda:future.StepReference(path,2,1,"gam","split",smoke=True))
        (path/"step_trace.jsonl").write_text(json.dumps({**row,"lr_nominal":.01})+"\n")
        expect_rejected(lambda:future.StepReference(path,2,1,"gam","split"))


def make_learner(model,source):
    sys.path.insert(0,str(source))
    from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner
    learner=SimpleNamespace(_optimizer_type="gam",_network=model,_lambda_flat=1.,
        _lambda_ewc=0.,_mechanism_eps=1e-12,_mechanism_grad_stats=False)
    def task_loss(logits,targets,offset):
        return torch.nn.functional.cross_entropy(logits[:,offset:],targets-offset),None,None
    learner._task_loss=task_loss
    learner._flatten_optimizer_params=lambda opt:[p for group in opt.param_groups for p in group["params"]]
    def clean_cache(params,inputs,targets,class_offset):
        loss=task_loss(model(inputs)["logits"],targets,class_offset)[0]
        grads=torch.autograd.grad(loss,params)
        return float(loss.detach()),{id(p):g.detach() for p,g in zip(params,grads)}
    learner._compute_clean_task_grad_cache=clean_cache
    learner._compute_fisher_grad_cache=lambda params:(0.,{})
    learner._compute_delta_w_penalty_grad_cache=lambda params:(0.,{},{})
    learner._record_mechanism_loss_stats=lambda *args:None
    learner._build_direction_projector=MethodType(Learner._build_direction_projector,learner)
    learner._step_batch_with_log=MethodType(Learner._step_batch_with_log,learner)
    return learner


def test_native_steps(source):
    sys.path.insert(0,str(source))
    from optimer_PerturabtionType.gam import GAM
    audit.seed_all(12)
    initial=Toy()
    x=torch.randn(6,4);y=torch.arange(20,26)
    targets=[]
    for rule in ("s","g_match"):
        model=copy.deepcopy(initial)
        learner=make_learner(model,source)
        stepper=future.NativeStep(learner,model,GAM,rule)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base,T_max=50,eta_min=.002)
        before=model.fc.weight[:20].detach().clone()
        for i in range(3):
            result=stepper.step(x,y,float(stepper.base.param_groups[0]["lr"]),
                               None if rule=="s" else targets[i])
            if rule=="s":targets.append(result["actual_effective_delta"]["joint_norm"])
            else:assert result["actual_match_relative_error"]<=1e-3
            assert result["closure_count"]==4 and result["training_forward_count"]==5
            assert result["self_spec_passed"] and result["base_sgd_exact"]
            assert torch.equal(model.fc.weight[:20],before)
            scheduler.step()
            expected=.002+.5*(.02-.002)*(1+math.cos(math.pi*(i+1)/50))
            assert math.isclose(stepper.base.param_groups[0]["lr"],expected,rel_tol=1e-12)
        # Corrupt projector after its own spec calculation: hard gate must reject.
        original=learner._build_direction_projector
        def corrupt(*args):
            proj=original(*args)
            def bad(groups):
                proj(groups)
                with torch.no_grad():groups[0]["params"][0].grad.fill_(float("nan"))
            return bad
        learner._build_direction_projector=corrupt
        expect_rejected(lambda:stepper.step(x,y,.02,None if rule=="s" else targets[0]))
    return {"s_steps":3,"g_match_steps":3,"max_match_rtol":1e-3}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Run this test with CUDA_VISIBLE_DEVICES='' to guarantee CPU-only")
    torch.set_num_threads(1)
    checks={}
    for fn in (test_data_rng,test_head_rng,test_observational,test_split,test_path,test_reference):
        fn();checks[fn.__name__]="PASS"
    checks["test_native_steps"]=test_native_steps(Path(args.source))
    report={"status":"PASS","device":"cpu","checks":checks,
            "future_probe_sha256":audit.digest(Path(__file__).with_name("future_probe.py")),
            "tests_sha256":audit.digest(__file__)}
    audit.atomic_json(Path(args.output),report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
