#!/usr/bin/env python3
"""Numerical gates for the isolated diagnostic, independent of model lifecycle."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
from offline_diagnostics import combine,position,virtual_sgd,unit,norm,interaction,precision_verdict,six_reads
from state_transaction import State
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import atomic_json,sha256,now


def tests():
    checks={};torch.manual_seed(31);dtype=torch.float64
    H=torch.tensor([[2.,.3],[.3,1.]],dtype=dtype);theta=torch.tensor([.4,-.7],dtype=dtype)
    grad=lambda x: H@x
    loss=lambda x:.5*x@H@x
    g0=grad(theta);p1=theta+.05*g0/g0.norm();g1=grad(p1)
    p2=theta+1.*(g1-g0)/(g1-g0).norm();g2=grad(p2);pm=(p1+p2)/2;gm=grad(pm)
    p3=p2+.05*g2/g2.norm();g3=grad(p3)
    torch.testing.assert_close((g1+g2)/2,gm,atol=2e-15,rtol=0)
    checks['quadratic_midpoint_e12_zero']=True
    eta=.02;c=g3-g2
    ls={k:torch.tensor([loss(theta-eta*v)],dtype=dtype) for k,v in dict(g2=g2,g3=g3,sum12=g1+g2,sum13=g1+g3).items()}
    I=interaction(ls)['I'];expected=float(eta**2*g1@H@c)
    assert abs(I-expected)<2e-15
    checks['quadratic_interaction_identity']=True
    for factor in (.5,.25):
        scaled=interaction({k:torch.tensor([loss(theta-eta*factor*v)],dtype=dtype) for k,v in dict(g2=g2,g3=g3,sum12=g1+g2,sum13=g1+g3).items()})['I']
        assert abs(scaled-I*factor**2)<2e-15
    checks['quadratic_eta_squared_scaling']=True
    def factor_grad(z):
        z=z.detach().clone().requires_grad_(True);f=.5*(z[0]*z[1]-1.)**2
        return torch.autograd.grad(f,z)[0]
    f1=torch.tensor([.2,.8],dtype=dtype);f2=torch.tensor([1.,.4],dtype=dtype)
    e=(factor_grad(f1)+factor_grad(f2))/2-factor_grad((f1+f2)/2)
    assert e.norm()>1e-3
    checks['factor_BA_midpoint_may_have_e12']=True
    a=(torch.tensor([1e6,1.1,.3],dtype=torch.float32),)
    u=(torch.tensor([.02,.4,-.2],dtype=torch.float32),)
    stored=virtual_sgd(a,u,.04)
    exact_double=a[0].double()-.04*u[0].double()
    assert not torch.equal(stored[0].double(),exact_double)
    checks['same_fp32_virtual_coordinates_upcast_not_recomputed']=True
    verdict=precision_verdict(dict(I=1e-7),dict(I=1e-7),dict(I=2e-7))
    assert verdict['status']=='numerically_unresolved' and verdict['epsilon_num']==1e-7
    assert precision_verdict(dict(I=1e-3),dict(I=1e-3),dict(I=1e-3+1e-10))['status']=='resolved'
    checks['zero_repeat_noise_not_precision_certificate']=True

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(2,3,dtype=dtype))
            self.register_buffer('counter',torch.zeros(1,dtype=dtype));self.dropout=torch.nn.Dropout(.1)
        def forward(self,x):
            self.counter.add_(1)
            logits=self.dropout(x)@self.weight.T
            return {'logits':logits}
    model=Toy();x=torch.randn(5,3,dtype=dtype);y=torch.tensor([0,1,0,1,0]);model.train()
    anchor=State(model)
    read=six_reads(model,tuple(model.parameters()),(x,y),0)
    assert read['ordinary_gradient_calls']==6 and set(read['grads'])=={'g0','g1','g2','gm','g3','galign'}
    assert all(anchor.check().values())
    torch.testing.assert_close(read['points']['phim'][0],(read['points']['phi1'][0]+read['points']['phi2'][0])/2,atol=0,rtol=0)
    checks['six_reads_same_rng_exact_transaction']=True
    checks['endpoint_factor_coordinate_midpoint_exact']=True
    class Bad(Toy):
        def forward(self,x):
            super().forward(x);raise RuntimeError('injected gradient failure')
    bad=Bad();before=State(bad)
    try:six_reads(bad,tuple(bad.parameters()),(x,y),0)
    except RuntimeError:pass
    else:raise AssertionError('failure not raised')
    assert all(before.check().values())
    checks['exception_restoration']=True
    return checks

if __name__=='__main__':
    checks=tests();root=Path(__file__).resolve().parent.parent
    report=dict(status='passed',accepted=True,checks=checks,passed_checks=sum(checks.values()),
                total_checks=len(checks),at=now(),test_file_sha256=sha256(__file__),
                diagnostic_file_sha256=sha256(Path(__file__).with_name('offline_diagnostics.py')))
    atomic_json(root/'DIAGNOSTIC_CPU_GATE.json',report)
    print(json.dumps(report,indent=2))
