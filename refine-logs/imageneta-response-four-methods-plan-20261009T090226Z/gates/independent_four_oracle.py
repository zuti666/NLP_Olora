"""Independent CPU analytic and old-HRF equivalence checks.

Does not call the implementation's vector-combination helpers. The quadratic
oracle uses NumPy float64 closed-form gradients evaluated at actual FP32 points;
the separate reuse gate requires bitwise equality with the old HRF core.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'imageneta-history-response-execution-20261008T175403Z'
sys.path.insert(0, str(ROOT / 'trainer'))
import numpy as np
import torch
from response_step import ResponseStep

spec = importlib.util.spec_from_file_location('independent_prior_hrf', OLD / 'trainer/history_step.py')
oldmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldmod)

class Quad(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))

H = np.array([[2.4, .3, -.1], [.3, 1.2, .2], [-.1, .2, .8]], dtype=np.float64)
INITIAL = [.7, -.45, .3]
BS = [np.array([.11, -.22, .07]), np.array([-.16, .05, .09]), np.array([.04, .12, -.1])]
LRS = [.04, .025, .001, .055, .012, .003, .04, .02, .033, .001, .01, .018]
ht = torch.tensor(H, dtype=torch.float32)

def close(actual, expected, label):
    av = actual.detach().double().numpy()
    err = float(np.max(np.abs(av - expected)))
    if not np.allclose(av, expected, rtol=4e-6, atol=5e-7):
        raise AssertionError((label, err, av.tolist(), expected.tolist()))
    return err

def model_closure(model, b):
    bt = torch.tensor(b, dtype=torch.float32)
    def closure(_x, _y):
        theta = model.theta
        loss = .5 * torch.dot(theta, ht @ theta) + torch.dot(bt, theta)
        return loss, theta
    return closure

checks = {}
max_errors = {}
for method in ('sgd', 'current_g2', 'history_fusion', 'history_g2'):
    model = Quad(INITIAL)
    driver = ResponseStep(model, list(model.parameters()), None,
                          dict(method=method, history_weight=.5 if method.startswith('history') else 0.))
    optimizer = torch.optim.SGD(model.parameters(), lr=LRS[0])
    prior_vector = prior_delta = None
    worst = 0.
    for step, eta in enumerate(LRS):
        if step == 7:
            driver.reset_history('independent_task_boundary')
            prior_vector = prior_delta = None
        optimizer.param_groups[0]['lr'] = eta
        base = model.theta.detach().clone()
        b = BS[step % len(BS)]
        g0 = H @ base.double().numpy() + b
        h = eta * np.linalg.norm(g0)
        pred = (base.double().numpy() - eta * g0).astype(np.float32).astype(np.float64)
        g1 = H @ pred + b
        response = g0 - g1
        if method.startswith('history') and prior_vector is not None:
            response = .5 * response + .5 * h / np.linalg.norm(prior_delta) * (prior_vector - g0)
        if method == 'sgd':
            final = g0
        elif method == 'history_fusion':
            final = g0 + response
        else:
            relocated = (base.double().numpy() + h * response / np.linalg.norm(response)).astype(np.float32).astype(np.float64)
            final = H @ relocated + b
        result = driver.compute(None, None, model_closure(model, b), eta)
        worst = max(worst, close(result['context']['g0'][0], g0, method+' g0'))
        worst = max(worst, close(result['U'][0], final, method+' final'))
        if method in ('current_g2','history_g2'):
            worst = max(worst, close(result['context']['relocation'][0], relocated, method+' relocation from base'))
        gradient_reads = {'sgd':1,'current_g2':3,'history_fusion':2,'history_g2':3}[method]
        assert result['record']['ordinary_gradient_calls'] == gradient_reads
        assert torch.equal(model.theta, base)
        actual_final = result['U'][0].detach().clone()
        update = driver.apply_update(eta, optimizer)
        expected_committed = base.clone().add_(actual_final, alpha=-eta)
        assert torch.equal(model.theta, expected_committed)
        actual_delta = model.theta.detach() - base
        cache = driver.state_dict()['history']
        if method.startswith('history'):
            assert cache is not None
            vector = result['context']['g0'][0] if method == 'history_fusion' else actual_final
            assert torch.equal(cache['vector'][0], vector)
            assert torch.equal(cache['displacement'][0], actual_delta)
            assert cache['cache_source'] == ('base' if method == 'history_fusion' else 'final')
            if method == 'history_g2':
                assert not torch.equal(cache['vector'][0], vector / torch.linalg.vector_norm(vector))
                assert not torch.equal(cache['vector'][0], actual_delta)
            prior_vector = vector.double().numpy()
            prior_delta = actual_delta.double().numpy()
        else:
            assert cache is None
        assert update['sgd_updates'] == 1 and update['natural_sgd_exact']
    checks[method+'_closed_form_12steps_variable_lr_cache_reset'] = True
    max_errors[method] = worst

# No-history history_g2 must equal current_g2, including all actual FP32 bits.
m1, m2 = Quad(INITIAL), Quad(INITIAL)
d1 = ResponseStep(m1, list(m1.parameters()), None, dict(method='current_g2', history_weight=0.))
d2 = ResponseStep(m2, list(m2.parameters()), None, dict(method='history_g2', history_weight=.5))
r1 = d1.compute(None,None,model_closure(m1,BS[0]),.04)
r2 = d2.compute(None,None,model_closure(m2,BS[0]),.04)
assert all(torch.equal(a,b) for a,b in zip(r1['U'],r2['U']))
assert torch.equal(r1['context']['relocation'][0], r2['context']['relocation'][0])
checks['history_g2_first_step_bitwise_current_g2'] = True

# Prior HRF equivalence on changing minibatch losses, changing LR and stochastic
# dropout, checking g0/g1/final, actual committed displacement and RNG after step.
support_equal = {name: (ROOT/'trainer'/name).read_bytes()==(OLD/'trainer'/name).read_bytes()
                 for name in ('state_transaction.py','effective_step.py','native_lifecycle.py','task0_trainer_base_v2.py','audit_core.py')}
assert all(support_equal.values())
equivalence_runs = []
for stochastic in (False, True):
    new_model, old_model = Quad(INITIAL), Quad(INITIAL)
    new = ResponseStep(new_model,list(new_model.parameters()),None,dict(method='history_fusion',history_weight=.5))
    old = oldmod.HistoryResponseStep(old_model,list(old_model.parameters()),None,dict(direction='history_fused',history_weight=.5))
    no = torch.optim.SGD(new_model.parameters(),lr=LRS[0])
    oo = torch.optim.SGD(old_model.parameters(),lr=LRS[0])
    torch.manual_seed(2234)
    for step, eta in enumerate(LRS):
        if step==7:
            new.reset_history('independent_task_boundary');old.reset_history('independent_task_boundary')
        no.param_groups[0]['lr']=oo.param_groups[0]['lr']=eta
        state0=torch.get_rng_state().clone()
        b=BS[step%len(BS)]
        def closure_for(model):
            bt=torch.tensor(b,dtype=torch.float32)
            def closure(_x,_y):
                theta=model.theta
                z=torch.nn.functional.dropout(theta,p=.25,training=True) if stochastic else theta
                loss=.5*torch.dot(z,ht@z)+torch.dot(bt,z)
                return loss,z
            return closure
        rn=new.compute(None,None,closure_for(new_model),eta);un=new.apply_update(eta,no)
        rngn=torch.get_rng_state().clone()
        torch.set_rng_state(state0)
        ro=old.compute(None,None,closure_for(old_model),eta);uo=old.apply_update(eta,oo)
        assert torch.equal(torch.get_rng_state(),rngn)
        for key in ('g','gf','U'):
            assert all(torch.equal(a,b) for a,b in zip(rn['context'][key],ro['context'][key])), (stochastic,step,key)
        assert torch.equal(new_model.theta,old_model.theta)
        assert un['actual_step_displacement_sha256']==uo['actual_step_displacement_sha256']
        assert rn['record']['history_used']==ro['record']['history_used']
        assert rn['record']['history_coefficient']==ro['record']['history_coefficient']
        hn,ho=new.state_dict()['history'],old.state_dict()['history']
        assert (hn is None)==(ho is None)
        if hn is not None:
            assert all(torch.equal(a,b) for a,b in zip(hn['vector'],ho['raw_gradient']))
            assert all(torch.equal(a,b) for a,b in zip(hn['displacement'],ho['displacement']))
    equivalence_runs.append(dict(stochastic=stochastic,steps=len(LRS),bitwise_gradients_updates_cache_rng=True))
checks['old_hrf_equivalence_24steps_deterministic_and_dropout'] = True

report = dict(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),
              cpu_only=True,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              checks=checks,analytic_max_absolute_error=max_errors,
              prior_support_files_byte_equal=support_equal,
              prior_hrf_equivalence=equivalence_runs,
              new_core_sha256=hashlib.sha256((ROOT/'trainer/response_step.py').read_bytes()).hexdigest(),
              old_core_sha256=hashlib.sha256((OLD/'trainer/history_step.py').read_bytes()).hexdigest(),
              limitation='CPU mathematical/core equivalence does not by itself establish full trainer/GPU trajectory identity.')
(ROOT/'gates/INDEPENDENT_FOUR_ORACLE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
