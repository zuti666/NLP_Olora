#!/usr/bin/env python3
"""C21 selector and actual native GAM/projector/SGD tests; no GPU performance claim."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import audit_core as audit
import component_horizon as component
import test_component_cpu as existing


def states(g0,g1,g2,g3=None,dtype=torch.float64):
    if g3 is None:g3=g2
    return [{f'g_{k}':torch.tensor(v,dtype=dtype) for k,v in enumerate(values)}
            for values in zip(g0,g1,g2,g3)]


def independent(values,rule):
    # Deliberately use one concatenated vector, not production per-tensor reduction.
    g1=torch.cat([s['g_1'].double().flatten() for s in values])
    d=torch.cat([(s['g_2']-s['g_1']).double().flatten() for s in values])
    denominator=float(g1@g1);a=float(g1@d)/denominator if denominator else 0.
    out=[]
    for state in values:
        correction=state['g_2']-state['g_1']
        out.append(state['g_0']+correction if rule=='c21_base'
                   else state['g_1']+(correction-a*state['g_1']))
    return out,a


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(7612)
    checks=[];negatives=[]
    examples={
        'global_not_per_tensor':states([[3.],[4.]],[[1.],[2.]],[[3.],[1.]]),
        'parallel':states([[2.,-1.],[.5]],[[1.,2.],[3.]],[[4.,8.],[12.]]),
        'perpendicular':states([[2.,-1.]],[[1.,0.]],[[1.,2.]]),
        'g1_zero':states([[2.,-1.]],[[0.,0.]],[[3.,4.]]),
        'c21_zero':states([[2.,-1.]],[[1.,2.]],[[1.,2.]]),
        'all_zero':states([[0.,0.]],[[0.,0.]],[[0.,0.]]),
        'tiny_nonzero_fp32':states([[2.,-1.]],[[1e-20,2e-20]],[[3e-20,6e-20]],dtype=torch.float32),
    }
    for name,value in examples.items():
        for rule in component.C21_RULES:
            rng=audit.rng_get();before=copy.deepcopy(value)
            actual,record=component.selected_c21_direction(value,rule)
            expected,a=independent(value,rule)
            assert all(torch.allclose(x,y,rtol=1e-6,atol=1e-30) for x,y in zip(actual,expected))
            assert math.isclose(record['coefficient_a'],a,rel_tol=1e-12,abs_tol=1e-15)
            assert audit.rng_equal(rng,audit.rng_get())
            assert all(torch.equal(s[k],v[k]) for s,v in zip(value,before) for k in s)
            assert record['orthogonality_passed'] and record['parameter_count']==sum(s['g_1'].numel() for s in value)
            c10=torch.cat([(s['g_1']-s['g_0']).double().flatten() for s in value])
            c21=torch.cat([(s['g_2']-s['g_1']).double().flatten() for s in value])
            denominator=float(c10.norm()*c21.norm())
            expected_cos=float(c10@c21)/denominator if denominator else 0.
            assert math.isclose(record['cos_c10_c21'],expected_cos,rel_tol=1e-12,abs_tol=1e-15)
            assert record['cos_c10_c21_defined']==(denominator>0.)
            assert math.isclose(record['norm_c21_parallel'],abs(a)*record['norms']['g1'],rel_tol=1e-12,abs_tol=1e-30)
            if name=='global_not_per_tensor':
                assert record['coefficient_a']==0. and math.isclose(record['perp_over_c21'],1.)
                if rule=='c21_perp':assert [float(v) for v in actual]==[3.,1.]
            if name=='parallel':assert record['norms']['perp']==0.
            if name=='perpendicular':assert record['coefficient_a']==0.
            if name in ('g1_zero','all_zero'):assert record['g1_zero'] and record['coefficient_a']==0.
            if name=='tiny_nonzero_fp32':assert not record['g1_zero']
            checks.append(name+'/'+rule)
    for special in (float('nan'),float('inf')):
        value=states([[0.]],[[1.]],[[special]])
        negatives.append(existing.rejected(lambda:component.selected_c21_direction(value,'c21_base'),FloatingPointError,'nonfinite_'+str(special)))
    negatives.append(existing.rejected(lambda:component.selected_c21_direction([], 'c21_perp'),ValueError,'empty_states'))
    negatives.append(existing.rejected(lambda:component.selected_c21_direction(examples['parallel'], 'ug'),ValueError,'invalid_c21_rule'))
    negatives.append(existing.rejected(lambda:component.ComponentStepAudit(None,None,'c21_base','matched'),ValueError,'matched_constructor_forbidden'))
    cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/c21_cpu_unused')
    negatives.append(existing.rejected(lambda:component.resolve_config({'component_rule':'c21_perp','component_scale':'matched'},cli),ValueError,'matched_config_forbidden'))
    # Both A/B actual native four-probe updates at 20- and 40-class heads.
    initial=existing.Toy().state_dict();x=torch.randn(9,7);y=torch.arange(9)
    native=[];call_counts=[]
    for rule in component.C21_RULES:
        model=existing.Toy();model.load_state_dict(initial)
        for task in (0,1):
            if task:
                rng=audit.rng_get();torch.manual_seed(129);model.expand(40);audit.rng_set(rng)
            parts=existing.make_step(model,rule,'natural')
            learner,_,opt,stepper=parts
            real_step=opt.base_optimizer.step
            validations=[]
            def independently_checked_step(*a,**kw):
                values=[opt.state[p] for p in stepper.params]
                expected,coefficient=independent(values,rule)
                # Toy is deterministic: clean cached g equals native g0.
                expected=[s['g_0']+1.*(e-s['g_0']) for s,e in zip(values,expected)]
                assert all(torch.allclose(p.grad,e,rtol=2e-6,atol=1e-7) for p,e in zip(stepper.params,expected))
                before=[p.detach().clone() for p in stepper.params]
                result=real_step(*a,**kw)
                eta=.04 if task==0 else .02
                # Actual final parameter state independently uses the selected formula.
                assert all(torch.allclose(p,b.add(e,alpha=-eta),rtol=1e-6,atol=1e-7)
                           for p,b,e in zip(stepper.params,before,expected))
                validations.append({'coefficient':coefficient,'actual_selected_and_SGD_verified':True})
                return result
            opt.base_optimizer.step=independently_checked_step
            for _ in range(3):
                row=existing.run_step(parts,x,y+20*task)
                assert row['step_scale']==row['lr_nominal'] and row['scale_over_nominal_lr']==1.
                assert row['c21_audit']['selected_spec_exact'] and row['c21_audit']['orthogonality_passed']
                assert row['c21_audit']['parameter_count']==sum(p.numel() for p in model.parameters())
                native.append({'rule':rule,'task':task,'class_count':model.fc.out_features,
                    'c21_audit':row['c21_audit'],'base_sgd_exact':row['base_sgd_exact']})
            assert len(validations)==3
            call_counts.append({'rule':rule,'task':task,'independent_verified_steps':len(validations)})
    # Detect illegal actual optimizer changes, not merely selector scalar mistakes.
    model=existing.Toy();parts=existing.make_step(model,'c21_perp','natural',corrupt=True)
    negatives.append(existing.rejected(lambda:existing.run_step(parts,x,y),AssertionError,'corrupt_actual_SGD'))
    report={'status':'passed','scope':'CPU synthetic geometry and actual frozen GAM/projector/SGD; not ViT training results',
        'selector_cases':checks,'actual_native_steps':len(native),'heads':[20,40],
        'five_CE_forwards_four_GAM_closures_per_step':True,'native_records':native,
        'independent_actual_SGD_checks':call_counts,'negative_cases':negatives,
        'trainer_sha256':audit.digest(component.__file__),'test_sha256':audit.digest(__file__),
        'source_gam_sha256':audit.digest(existing.SOURCE/'optimer_PerturabtionType/gam.py')}
    if args.output:audit.atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='native_records'},indent=2))


if __name__=='__main__':main()
