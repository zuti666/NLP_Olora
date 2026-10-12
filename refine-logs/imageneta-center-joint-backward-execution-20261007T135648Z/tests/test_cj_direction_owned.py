"""CPU formula/state regressions; performance evidence comes only from training."""
from pathlib import Path
import importlib.util
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'trainer'))
from direction_step import FourProbeStep, direction_name
from state_transaction import capture_state, rng_get, rng_equal, rng_set


class Model(torch.nn.Module):
    def __init__(self, p=(.31, -.27, .43)):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(p, dtype=torch.float64))
        self.register_buffer('observations', torch.tensor(0.))
        self.forward_cache = {'untouched': torch.tensor(7.)}


class Geometry:
    def __init__(self, model): self.params = tuple(model.parameters())
    def actual_delta(self, before, after=None):
        after = tuple(p.detach() for p in self.params) if after is None else after
        return {'norm': float(torch.linalg.vector_norm(torch.cat([(a-b).flatten() for a,b in zip(after,before)])))}


def config(first, second, coef=1, rho=.05):
    return dict(first_location_estimator=first, second_residual_estimator=second,
                residual_coefficient=coef, update_base='g0', rho=rho, radius=1., eps=1e-12,
                direction=direction_name(first, second, coef), backend='native')


def make_closure(model, points, quadratic=False, raise_at=None):
    def closure(inputs, targets):
        points.append(model.p.detach().clone())
        model.observations.add_(1.)
        model.forward_cache = {'modified': model.p.detach().clone()}
        torch.rand(2)
        if raise_at == len(points): raise RuntimeError('deliberate closure failure')
        loss = .5*(model.p*model.p).sum()
        if not quadratic: loss = loss + .25*(model.p**4).sum()
        return loss, model.p.reshape(1,-1)
    return closure


def analytical(first, second, coefficient, base, rho=.05, quadratic=False):
    grad = lambda v: v if quadratic else v+v**3
    unit = lambda v: v/(float(torch.linalg.vector_norm(v))+1e-12)
    g0=grad(base); p1=base+(.05 if first=='forward' else -.05)*unit(g0)
    g1=grad(p1); q=g1-g0 if first=='forward' else g0-g1
    p2=base+unit(q); g2=grad(p2)
    pp=p2+rho*unit(g2); pm=p2-rho*unit(g2)
    if second=='forward': residual=grad(pp)-g2
    elif second=='backward_corrected': residual=g2-grad(pm)
    else: residual=.5*(grad(pp)-grad(pm))
    return g0+coefficient*residual


def test_nonlinear_and_counts():
    cases=[('forward','forward',1),('forward','backward_corrected',1),
           ('forward','centered',1),('forward','centered',-1),
           ('backward_corrected','forward',1),('backward_corrected','backward_corrected',1)]
    for first,second,coef in cases:
        model=Model(); base=model.p.detach().clone(); points=[]
        step=FourProbeStep(model,tuple(model.parameters()),Geometry(model),config(first,second,coef))
        result=step.compute(None,None,make_closure(model,points),.02)
        expected=analytical(first,second,coef,base)
        torch.testing.assert_close(result['U'][0],expected,atol=1e-12,rtol=1e-12)
        count=5 if second=='centered' else 4
        assert len(points)==result['record']['ordinary_gradient_calls']==count
        assert model.observations.item()==0 and 'untouched' in model.forward_cache
        assert torch.equal(model.p,base)
        assert result['record']['first_positive_gradient_calls']==int(first=='forward')
        assert result['record']['g3_positive_gradient_calls']==int(second!='backward_corrected')
        assert 'g1_plus' in result['context']['grads'] if first=='forward' else 'g1_plus' not in result['context']['grads']
        if second=='centered':
            assert result['record']['centered_response']['e2_norm']>1e-8
            assert result['record']['centered_response']['additional_gradient_calls']==0
        before=len(points); diag=step.diagnose_sides(None,None,None,result,.02)
        assert len(points)==before and diag['diagnostic_gradient_calls']==0
        step.apply_update()
        assert torch.equal(model.p,base.add(result['U'][0],alpha=-.02))


def test_quadratic():
    outputs=[]
    for first,second in [('forward','forward'),('forward','backward_corrected'),
                         ('backward_corrected','forward'),('backward_corrected','backward_corrected'),
                         ('forward','centered')]:
        model=Model();step=FourProbeStep(model,tuple(model.parameters()),Geometry(model),config(first,second))
        result=step.compute(None,None,make_closure(model,[],quadratic=True),.02)
        outputs.append(result['U'][0])
        if second=='centered':assert result['record']['centered_response']['e2_norm']<1e-14
    for value in outputs: torch.testing.assert_close(value,outputs[0],atol=1e-10,rtol=0.)


def test_zero_and_compute_rollback():
    for first,second,coef in [('forward','centered',-1),('backward_corrected','backward_corrected',1)]:
        model=Model((0.,0.,0.));step=FourProbeStep(model,tuple(model.parameters()),Geometry(model),config(first,second,coef,rho=0.))
        result=step.compute(None,None,make_closure(model,[]),.02)
        assert torch.equal(result['U'][0],model.p) and torch.isfinite(result['U'][0]).all()
    model=Model();model.p.grad=torch.tensor([1.,2.,3.],dtype=torch.float64)
    state=capture_state(model)
    step=FourProbeStep(model,tuple(model.parameters()),Geometry(model),config('forward','centered'))
    try:step.compute(None,None,make_closure(model,[],raise_at=5),.02)
    except RuntimeError as error:assert 'deliberate' in str(error)
    else:raise AssertionError('Missing injected failure')
    assert all(state.check().values())


def test_apply_rollback():
    model=Model();model.p.grad=torch.tensor([1.,2.,3.],dtype=torch.float64);state=capture_state(model)
    step=FourProbeStep(model,tuple(model.parameters()),Geometry(model),config('backward_corrected','backward_corrected'))
    step.compute(None,None,make_closure(model,[]),.02)
    optimizer=torch.optim.SGD(model.parameters(),lr=.02)
    def fail():
        with torch.no_grad():model.p.add_(10);model.observations.add_(11)
        model.eval();model.forward_cache={'bad':0};torch.rand(5)
        raise RuntimeError('deliberate optimizer failure')
    optimizer.step=fail
    try:step.apply_update(optimizer=optimizer)
    except RuntimeError as error:assert 'deliberate' in str(error)
    else:raise AssertionError('Missing injected optimizer failure')
    assert all(state.check().values())


def test_old_exact_regression():
    oldfile=ROOT.parent/'imageneta-second-probe-sign-deterministic-20261007T071400Z'/'trainer'/'direction_step.py'
    spec=importlib.util.spec_from_file_location('old_frozen_direction',oldfile)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for second,sign in [('forward',1),('backward_corrected',-1)]:
        initial=rng_get(); results=[]; models=[]; after_rng=[]
        for cls,cfg in [(module.FourProbeStep,dict(branch='P',direction=module.direction_name('g0',sign,sign),first_probe_sign=1,second_probe_sign=sign,second_difference_sign=sign,update_base='g0',second_difference_weight=1.,rho=.05,radius=1.,eps=1e-12,backend='native')),
                        (FourProbeStep,config('forward',second))]:
            rng_set(initial);model=Model();step=cls(model,tuple(model.parameters()),Geometry(model),cfg)
            result=step.compute(None,None,make_closure(model,[]),.02)
            step.apply_update();results.append(result);models.append(model);after_rng.append(rng_get())
        for key in ('g0','g1','g2','g3'):
            assert all(torch.equal(a,b) for a,b in zip(results[0]['context']['grads'][key],results[1]['context']['grads'][key]))
        for key in ('phi0','phi1','phi2','phi3'):
            assert all(torch.equal(a,b) for a,b in zip(results[0]['context']['points'][key],results[1]['context']['points'][key]))
        assert torch.equal(results[0]['U'][0],results[1]['U'][0])
        assert torch.equal(models[0].p,models[1].p) and rng_equal(after_rng[0],after_rng[1])


if __name__=='__main__':
    for test in (test_nonlinear_and_counts,test_quadratic,test_zero_and_compute_rollback,test_apply_rollback,test_old_exact_regression):
        test();print(test.__name__+': PASS')
