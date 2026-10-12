import sys, json, hashlib, math, random, copy
from pathlib import Path
import numpy as np
import torch
NEW=Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z')
sys.path.insert(0,str(NEW/'trainer'))
from independent_response_step import IndependentResponseStep
from response_step import ResponseStep
from state_transaction import rng_get,rng_set,rng_equal

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
H=torch.tensor([[3.,.2,0.,.1,0.],[.2,1.5,.15,0.,.1],[0.,.15,2.,.3,0.],[.1,0.,.3,4.,.2],[0.,.1,0.,.2,.75]])
b=torch.tensor([.2,-.1,.3,-.2,.1])
start=torch.tensor([.4,-.3,.2,-.1,.35])
checks={};details=[]
def norm(v): return float(torch.linalg.vector_norm(v.double()))
def assertallclose(a,b,label):
    if not torch.allclose(a,b,rtol=2e-6,atol=3e-7):
        raise AssertionError(label+': '+str((a-b).abs().max()))
class Model(torch.nn.Module):
    def __init__(self,point=start):
        super().__init__();self.p=torch.nn.Parameter(point[:3].clone());self.q=torch.nn.Parameter(point[3:].clone());self.frozen=torch.nn.Parameter(torch.tensor([9.]),requires_grad=False);self.register_buffer('counter',torch.tensor(5.));self.cache_test={'value':torch.tensor(7.)};self.dropout=torch.nn.Dropout(.3)
    def vector(self):return torch.cat((self.p,self.q))

def cfg(im,om):
    return dict(method='current_g2',history_weight=0.,radius_mode='independent_inner_outer_v1',radius_scale=1.,eps=1e-12,backend='native',inner_probe_radius_mode=im,outer_probe_radius_mode=om,fixed_inner_probe_radius=.05 if im=='fixed_l2' else None,fixed_outer_probe_radius=.05 if om=='fixed_l2' else None)

def run(im,om,eta,point=start,kind='quadratic',fail_call=None):
    torch.manual_seed(113);random.seed(113);np.random.seed(113)
    m=Model(point);m.train();m.dropout.eval();params=[m.p,m.q]
    m.p.grad=torch.ones_like(m.p)*2;m.q.grad=torch.ones_like(m.q)*3
    step=IndependentResponseStep(m,params,None,cfg(im,om));observed=[];draws=[]
    prior=rng_get();oldgrads=[p.grad.clone() for p in params]
    def closure(_x,_y):
        observed.append(m.vector().detach().clone());draws.append((float(torch.rand(())),random.random(),float(np.random.rand())))
        with torch.no_grad():m.counter.add_(1.)
        m.cache_test={'value':torch.tensor(8.)};m.dropout.train()
        v=m.vector()
        if fail_call is not None and len(observed)==fail_call:raise RuntimeError('deliberate oracle probe failure')
        if kind=='quadratic':loss=.5*torch.dot(v,H@v)+torch.dot(b,v)
        elif kind=='linear':loss=torch.dot(b,v)
        elif kind=='zero':loss=(v*0).sum()
        else:raise AssertionError('kind')
        return loss,v.unsqueeze(0)
    try:out=step.compute(None,None,closure,eta)
    except RuntimeError:
        assert fail_call is not None
        assert torch.equal(m.vector(),point)
        assert m.counter.item()==5 and m.cache_test['value'].item()==7 and not m.dropout.training
        assert all(torch.equal(p.grad,g) for p,g in zip(params,oldgrads));assert rng_equal(rng_get(),prior)
        assert step._pending is None
        return None
    assert torch.equal(m.vector(),point),'base restore'
    assert m.counter.item()==5 and m.cache_test['value'].item()==7 and not m.dropout.training
    assert all(torch.equal(p.grad,g) for p,g in zip(params,oldgrads))
    assert all(v==draws[0] for v in draws),'R0 replay'
    actualR1=rng_get();rng_set(prior);_=(float(torch.rand(())),random.random(),float(np.random.rand()));expectedR1=rng_get();assert rng_equal(actualR1,expectedR1),'single R1 advance';rng_set(actualR1)
    opt=torch.optim.SGD(params,lr=eta,momentum=0,weight_decay=0)
    U=torch.cat(out['U']);expected_commit=torch.cat([p.detach().clone().add_(u,alpha=-eta) for p,u in zip(params,out['U'])])
    proof=step.apply_update(eta,opt)
    assert torch.equal(m.vector(),expected_commit),'natural ordinary SGD'
    assert proof['sgd_updates']==1 and proof['natural_sgd_exact']
    assert step._history is None
    s=step.state_dict();assert s['history'] is None
    step.load_state_dict(s)
    for k in ['fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls']:assert out['record'][k]==0,k
    return {'out':out,'points':observed,'U':U,'after':m.vector().detach().clone(),'proof':proof,'model':m,'state':s}

for label,im,om in [('FixedInner','fixed_l2','sgd_step'),('FixedOuter','sgd_step','fixed_l2'),('FixedBoth','fixed_l2','fixed_l2')]:
    for eta in [.04,.004]:
        got=run(im,om,eta)
        g0=H@start+b;h=eta*norm(g0)
        inner=start.clone().add_(g0,alpha=-eta if im=='sgd_step' else -.05/norm(g0))
        g1=H@inner+b;a=g0-g1
        outer=start.clone().add_(a,alpha=(h if om=='sgd_step' else .05)/norm(a))
        g2=H@outer+b
        for observed,expected,what in [(got['points'][0],start,'g0base'),(got['points'][1],inner,'descending inner'),(got['points'][2],outer,'outer FROM BASE'),(got['U'],g2,'g2 only')]:assertallclose(observed,expected,label+what)
        assert got['out']['record']['ordinary_gradient_calls']==3 and len(got['points'])==3
        assert got['out']['record']['history_used'] is False
        assert got['out']['record']['radius_mode']=='independent_inner_outer_v1'
        checks[label+'_eta_'+str(eta)+'_closed_form_global_multitensor']=True
        details.append({'variant':label,'eta':eta,'inner_actual':norm(got['points'][1]-start),'outer_actual':norm(got['points'][2]-start),'h':h,'sgd_step_actual':got['proof']['actual_parameter_step_norm']})
# FixedBoth geometry/U invariant, current_lr affects only the commit.
a=run('fixed_l2','fixed_l2',.04);z=run('fixed_l2','fixed_l2',.004);zero=run('fixed_l2','fixed_l2',0.)
assert all(torch.equal(x,y) for x,y in zip(a['points'],z['points'])) and torch.equal(a['U'],z['U'])
assert all(torch.equal(x,y) for x,y in zip(a['points'],zero['points'])) and torch.equal(a['U'],zero['U'])
assert not torch.equal(a['after'],z['after']) and torch.equal(zero['after'],start)
checks['FixedBoth_eta_changes_leave_probes_and_U_bitwise_identical_including_eta_zero']=True
# FixedInner changes outer actual distance with eta; FixedOuter preserves its distance.
fi1=run('fixed_l2','sgd_step',.04);fi2=run('fixed_l2','sgd_step',.004)
fo1=run('sgd_step','fixed_l2',.04);fo2=run('sgd_step','fixed_l2',.004)
assert math.isclose(norm(fi1['points'][2]-start)/norm(fi2['points'][2]-start),10,rel_tol=2e-5)
assert math.isclose(norm(fo1['points'][2]-start),.05,rel_tol=2e-6) and math.isclose(norm(fo2['points'][2]-start),.05,rel_tol=2e-6)
checks['two_single_fixed_variants_change_only_the_registered_distance']=True
# Numeric fallbacks and exception restoration, without history or hidden work.
for im,om in [('fixed_l2','sgd_step'),('sgd_step','fixed_l2'),('fixed_l2','fixed_l2')]:
    got=run(im,om,.04,kind='linear');assert got['out']['record']['ordinary_gradient_calls']==2 and got['out']['record']['fallback_reason']=='current_response_length_at_or_below_eps';assertallclose(got['U'],b,'linear fallback')
    zero=run(im,om,.04,kind='zero');assert zero['out']['record']['ordinary_gradient_calls']==1 and zero['out']['record']['fallback_reason']==('base_gradient_length_at_or_below_eps' if im=='fixed_l2' else 'inner_probe_length_at_or_below_eps')
    run(im,om,.04,fail_call=2);run(im,om,.04,fail_call=3)
checks['zero_response_zero_gradient_exception_state_RNG_guards']=True
# Large FP32 base: fixed .05 may be unresolvable; explicit first-probe fallback.
big=run('fixed_l2','fixed_l2',.04,point=torch.full((5,),1e8))
assert big['out']['record']['ordinary_gradient_calls']==1 and big['out']['record']['fallback_reason']=='predicted_point_bitwise_indistinguishable_in_FP32'
outer_big=run('sgd_step','fixed_l2',.04,point=torch.full((5,),1e8))
assert outer_big['out']['record']['ordinary_gradient_calls']==2 and outer_big['out']['record']['fallback_reason']=='relocated_point_bitwise_indistinguishable_in_FP32'
checks['actual_FP32_indistinguishable_probe_accounting']=True
# Same implementation must reject cross-geometry checkpoint restoration.
try:
    bad=IndependentResponseStep(a['model'],[a['model'].p,a['model'].q],None,cfg('fixed_l2','sgd_step'));bad.load_state_dict(a['state']);raise AssertionError('accepted different geometry state')
except ValueError:pass
checks['checkpoint_geometry_rejection']=True
report={'status':'passed','checks':checks,'details':details,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256((NEW/'trainer/independent_response_step.py').read_bytes()).hexdigest(),'test_kind':'independently authored analytic multi-tensor quadratic and transaction CPU oracle; no author tests used'}
Path('/tmp/radius45_independent_oracle.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
