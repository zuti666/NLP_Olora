#!/usr/bin/env python3
"""Analytic CPU float64 verification of arbitrary detached H(x)d."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch
from response_hvp import (State, arbitrary_Hd, compare, gradient_at,
                          rng_get, rng_equal, vector_norm, center_difference_gate)


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([.7,-1.2,.3], dtype=torch.float64))
        self.register_buffer('Q', torch.tensor([[2.,.4,-.2],[.4,3.,.6],[-.2,.6,1.5]], dtype=torch.float64))
        self.register_buffer('linear', torch.tensor([.1,-.2,.8], dtype=torch.float64))
        self.forward_cache = {'value': torch.tensor(13.)}
        self.delta_w_q_new_grad = torch.tensor([4.,5.], dtype=torch.float64)
        self.delta_w_v_new_grad = None
        self.external_delta_w_q = torch.tensor([8.,9.], dtype=torch.float64)
        self.external_delta_w_v = None
        self._register_delta_hook = False
    def forward(self):
        self.forward_cache['value'] = self.x.detach().clone()
        self.delta_w_q_new_grad = self.x.detach().clone()
        self.delta_w_v_new_grad = torch.ones_like(self.x)
        self.external_delta_w_q = None
        self._register_delta_hook = True
        return .5 * self.x @ self.Q @ self.x + self.linear @ self.x


def run():
    model = Quadratic(); params = (model.x,)
    model.x.grad = torch.tensor([9.,8.,7.], dtype=torch.float64)
    entry = State(model); R0 = rng_get()
    x1 = (torch.tensor([.1,.2,-.4], dtype=torch.float64),)
    x2 = (torch.tensor([.3,-.9,.8], dtype=torch.float64),)
    d = tuple((b-a).detach() for a,b in zip(x1,x2))
    h, info = arbitrary_Hd(model,params,x1,d,model,R0,backend='native')
    expected = (model.Q @ d[0],)
    stats = compare(h,expected)
    assert stats['relative_l2'] < 1e-13 and vector_norm(d) != 1.
    scaled,_ = arbitrary_Hd(model,params,x1,(3.7*d[0],),model,R0,backend='native')
    assert torch.allclose(scaled[0],3.7*h[0],rtol=1e-13,atol=1e-13)
    _,g1,_,_ = gradient_at(model,params,x1,model,R0,'native')
    _,g2,_,_ = gradient_at(model,params,x2,model,R0,'native')
    mid = tuple(((a+b)/2).detach() for a,b in zip(x1,x2))
    hm,_ = arbitrary_Hd(model,params,mid,d,model,R0,'native')
    sec = tuple(b-a for a,b in zip(g1,g2))
    assert torch.allclose(sec[0],h[0],rtol=1e-13,atol=1e-13)
    assert torch.allclose(hm[0],h[0],rtol=1e-13,atol=1e-13)
    linear = lambda: model.linear @ model.x
    hz,linfo = arbitrary_Hd(model,params,x1,d,linear,R0,'native')
    assert torch.equal(hz[0],torch.zeros_like(model.x))
    # Correct forward/backward first-difference identity in the quadratic case.
    base=(model.x.detach().clone(),)
    _,g0,_,_ = gradient_at(model,params,base,model,R0,'native')
    u0=g0[0]/(g0[0].norm()+1e-12);rho=.05;r=.25
    plus=(base[0]+rho*u0,);minus=(base[0]-rho*u0,)
    _,gp,_,_=gradient_at(model,params,plus,model,R0,'native')
    _,gm,_,_=gradient_at(model,params,minus,model,R0,'native')
    cplus=gp[0]-g0[0];cminus=gm[0]-g0[0]
    assert torch.allclose(cplus,-cminus,rtol=1e-12,atol=1e-14)
    P=base[0]+r*cplus/(cplus.norm()+1e-12)
    C=base[0]-r*cminus/(cminus.norm()+1e-12)
    D=base[0]+r*cminus/(cminus.norm()+1e-12)
    R=base[0]-r*cplus/(cplus.norm()+1e-12)
    assert torch.allclose(P,C,rtol=1e-12,atol=1e-14)
    assert torch.allclose(D,R,rtol=1e-12,atol=1e-14)
    gate = center_difference_gate(model,params,x1,d,model,R0,'native')
    assert gate['status']=='passed'
    # A requires_grad vector must be refused, not silently differentiated.
    refused=False
    try: arbitrary_Hd(model,params,x1,(d[0].requires_grad_(),),model,R0,'native')
    except ValueError: refused=True
    assert refused
    entry.restore(); restore=entry.check(); assert rng_equal(R0,rng_get())
    return {'schema_version':1,'status':'passed','device':'cpu','dtype':'float64',
            'arbitrary_nonunit_direction':stats,'direction_norm':vector_norm(d),
            'direction_length_scaled_linearly':True,'quadratic_sec_start_mid_equal':True,
            'linear_hvp_exact_zero':True,'P_C_and_D_R_quadratic_identity':True,
            'attached_direction_refused':True,'state_restore':restore,
            'same_point_gradient':info['same_point_gradient'],'finite_difference_gate':gate}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args()
    result=run()
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'analytic_relative_l2':result['arbitrary_nonunit_direction']['relative_l2'],
                      'direction_norm':result['direction_norm'],'restored':all(result['state_restore'].values())},allow_nan=False))


if __name__=='__main__': main()
