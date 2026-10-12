#!/usr/bin/env python3
"""CPU positive/negative tests for local HVP and native closure-state audit."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from types import MethodType

import torch
from torch import nn

import curvature_audit as ca


def reject(fn):
    try:fn()
    except (AssertionError,ValueError,FloatingPointError):return
    raise AssertionError("Negative control did not fail")


class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.w=nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float64))
        self.register_buffer("counter",torch.tensor(0.))
    def forward(self,x):
        self.counter.add_(1.)
        z=x@self.w
        return torch.stack((torch.zeros_like(z),z),dim=1)


def analytic_hvp():
    model=Logistic()
    state=ca.State(model)
    x=torch.tensor([[1.,2.],[-.7,.4],[.2,-.5]],dtype=torch.float64)
    y=torch.tensor([0,1,0])
    # This direction deliberately depends on live parameters. The audited HVP
    # must detach it, otherwise a derivative-of-direction term contaminates H*u.
    direction=[model.w*2+torch.tensor([.1,.6],dtype=torch.float64)]
    _,g,h=ca.gradient_at(state,x,y,0,2,need_hvp=True,direction=direction)
    probability=(x@model.w).sigmoid().detach()
    hessian=x.T@(x*(probability*(1-probability))[:,None])/len(y)
    expected=[hessian@direction[0].detach()]
    assert ca.compare(h,expected)["relative_l2"]<1e-12
    eps_grid=(.05,.025,.0125)
    forward=[];central=[]
    for eps in eps_grid:
        _,plus,_=ca.gradient_at(state,x,y,0,2,[eps*u for u in direction])
        _,minus,_=ca.gradient_at(state,x,y,0,2,[-eps*u for u in direction])
        forward.append(ca.compare(ca.difference(plus,g,eps),h)["relative_l2"])
        central.append(ca.compare(ca.difference(plus,minus,2*eps),h)["relative_l2"])
    assert forward[-1]<forward[0] and central[-1]<central[0]
    assert central[-1]<forward[-1]
    state.restore();assert all(state.check().values())
    reject(lambda:ca.assert_same_gradient([g[0]*1.01],g))
    reject(lambda:ca.compare([torch.full_like(g[0],float("nan"))],g))
    return {"analytic_relative_error":ca.compare(h,expected)["relative_l2"],
            "forward_relative_errors":forward,"central_relative_errors":central,
            "direction_detached":True,"restore":state.check()}


def fallback_contract():
    def flags():
        return [torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled()]
    before=flags()
    with ca.backend_context("math"):
        assert flags()==[False,True,False,False]
    assert flags()==before
    original=ca.gradient_at
    calls=[]
    native=[torch.tensor([1.,2.])]
    def fake(*args,backend="native",**kwargs):
        calls.append(backend)
        if backend=="native":raise RuntimeError("derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented")
        return .4,native,[torch.tensor([3.,4.])]
    try:
        ca.gradient_at=fake
        hvp,info=ca.audited_hvp(None,None,None,0,2,native,native,"auto")
        assert calls==["native","math"] and info["backend"]=="math" and info["status"]=="available"
        hvp,info=ca.audited_hvp(None,None,None,0,2,native,native,"native")
        assert hvp is None and info["status"]=="unsupported"
        def oom(*args,**kwargs):raise RuntimeError("CUDA out of memory")
        ca.gradient_at=oom
        try:ca.audited_hvp(None,None,None,0,2,native,native,"auto")
        except RuntimeError as exc:assert "out of memory" in str(exc)
        else:raise AssertionError("OOM was incorrectly treated as unsupported HVP")
    finally:ca.gradient_at=original
    return {"explicit_math_fallback":True,"math_only_including_cudnn_disabled":True,
            "backend_flags_restored":True,"unsupported_reported":True,"oom_not_masked":True}


def file_pin_contract():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        (path/"native.pt").write_bytes(b"immutable native checkpoint")
        (path/"source.py").write_text("native_source = True\n")
        pins=ca.pin_tree(path)
        ca.verify_tree(path,pins)
        (path/"native.pt").write_bytes(b"changed")
        reject(lambda:ca.verify_tree(path,pins))
        (path/"native.pt").write_bytes(b"immutable native checkpoint")
        (path/"extra.py").write_text("unexpected = True\n")
        reject(lambda:ca.verify_tree(path,pins))
    return {"unchanged_passed":True,"changed_checkpoint_rejected":True,"added_source_file_rejected":True}


def native_gam(source,formation_trainer):
    sys.path[:0]=[str(source),str(formation_trainer)]
    from optimer_PerturabtionType.gam import GAM
    from test_future_probe_cpu import Toy,make_learner
    ca.seed_all(17)
    model=Toy()
    learner=make_learner(model,source)
    learner._unwrap_network=lambda:model
    x=torch.randn(5,4);y=torch.tensor([20,21,22,23,24])
    original=ca.State(model)
    result=ca.audit_batch(learner,GAM,x,y,20,40,.05,.25,(.05,.025,.0125),"native")
    assert result["native"]["closure_count"]==4 and result["native"]["training_forward_count"]==5
    assert all(result["native"]["native_point_checks"].values())
    assert all(result["restore"].values()) and all(original.check().values())
    for response in result["responses"].values():
        assert response["hvp"]["status"]=="available"
        assert response["same_point_native_gradient"]["relative_l2"]<2e-5
    class WrongPointGAM(GAM):
        @torch.no_grad()
        def grad_norm_ascent(self):
            super().grad_norm_ascent()
            self.param_groups[0]["params"][0].add_(.01)
    reject(lambda:ca.capture_native(learner,WrongPointGAM,x,y,20,40,.05,.25))
    assert all(original.check().values())
    # Checkpoint/probe labels must be sliced to current 20 classes.
    reject(lambda:ca.gradient_at(original,x,y,0,20))
    model.dropout=nn.Dropout(.2)
    reject(lambda:ca.audit_batch(learner,GAM,x,y,20,40))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--formation-trainer",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if torch.cuda.is_available():raise RuntimeError("Run CPU tests with CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    result={"status":"PASS","device":"cpu","analytic_hvp":analytic_hvp(),
            "fallback_contract":fallback_contract(),
            "file_pin_contract":file_pin_contract(),
            "native_gam":native_gam(Path(args.source),Path(args.formation_trainer)),
            "script_sha256":ca.sha(Path(__file__).with_name("curvature_audit.py")),
            "tests_sha256":ca.sha(__file__)}
    ca.write_json(args.output,result)
    print(json.dumps({"status":"PASS","script_sha256":result["script_sha256"],
                      "analytic_error":result["analytic_hvp"]["analytic_relative_error"],
                      "native_point_checks":result["native_gam"]["native"]["native_point_checks"]},indent=2))


if __name__=="__main__":main()
