#!/usr/bin/env python3
"""Real frozen-GAM/CE/HVP CPU tests, not evidence of full ViT GPU execution."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import component_horizon as trainer
import hvp_helpers as helper
import test_component_cpu as old


def make_step(model,rule,mode,first='fd',second='fd',coefficient=1.,backend='math',corrupt_sgd=False):
    cfg={'component_first_response':first,'component_second_response':second,
         'component_lambda_corr':coefficient,'component_attention_backend':backend}
    learner=SimpleNamespace(args=cfg,_unwrap_network=lambda:model,_lambda_flat=1.,_mechanism_eps=1e-12,_mechanism_grad_stats=False)
    native=MethodType(old.native_projector(),learner)
    sgd=torch.optim.SGD(model.parameters(),lr=.04 if model.fc.out_features==20 else .02,momentum=0.,weight_decay=0.)
    if corrupt_sgd:
        original=sgd.step
        def damaged(*args,**kwargs):
            result=original(*args,**kwargs)
            with torch.no_grad():next(model.parameters()).add_(.1)
            return result
        sgd.step=damaged
    opt=old.GAM(model.parameters(),sgd,model,grad_rho=.05,grad_norm_rho=1.,adaptive=False,
        args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.,use_sgv=False))
    stepper=trainer.ComponentStepAudit(learner,opt,rule,mode)
    return learner,native,opt,stepper


def run_step(parts,x,y):
    learner,projector,opt,s=parts;model=learner._unwrap_network();lo=model.fc.out_features-20
    s.set_batch(x,y,lo,lo+20);calls=[0]
    hook=model.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
    original_report=s.response_report;captured={}
    def report():
        captured.update(points=copy.deepcopy(s.points),grads=copy.deepcopy(s.native_grads),
            first_correction=copy.deepcopy(s.first_correction),second_correction=copy.deepcopy(s.second_correction))
        return original_report()
    s.response_report=report
    try:
        with helper.backend_context(s.backend):
            loss=F.cross_entropy(model(x)['logits'][:,lo:lo+20],y-lo)
            grads=torch.autograd.grad(loss,list(model.parameters()))
            cache={id(p):g.detach().clone() for p,g in zip(model.parameters(),grads)}
            opt.set_direction_projector(s.projector(projector(cache,{},{}),cache,{},{}))
            def closure():
                opt.zero_grad();outputs=model(x)
                value=F.cross_entropy(outputs['logits'][:,lo:lo+20],y-lo);value.backward()
                return outputs,value.detach()
            opt.step(closure)
    finally:hook.remove();s.response_report=original_report
    row=copy.deepcopy(s.record);extra=int(s.first=='hvp')+int(s.second=='hvp')
    assert calls[0]==5+extra and row['closure_count']==4
    assert row['self_spec_passed'] and row['response_audit']['hvp_forward_count']==extra
    assert all(row['response_audit']['native_point_checks'].values())
    flow=row['gradient_flow'];assert flow['selected_expected_sha256']==flow['selected_actual_sha256']
    assert flow['projected_expected_sha256']==flow['projected_actual_sha256']==flow['actual_sgd_gradient_sha256']
    for item in row['response_audit']['hvp_reports'].values():
        assert all(item['entry_state_restore'].values()) and item['direction_detached'] is True and item['fallback_used'] is False
    if s.mode=='matched':assert row['matched'] and row['actual_match_relative_error']<=.001
    return row,captured


def independent_hvp_at(model,values,x,y,lo,u):
    clone=copy.deepcopy(model)
    with torch.no_grad():
        for p,v in zip(clone.parameters(),values):p.copy_(v)
    with helper.backend_context('math'):
        loss=F.cross_entropy(clone(x)['logits'][:,lo:lo+20],y-lo)
        g=torch.autograd.grad(loss,tuple(clone.parameters()),create_graph=True)
        h=torch.autograd.grad(sum((a*b.detach()).sum() for a,b in zip(g,u)),tuple(clone.parameters()))
    return [v.detach() for v in h]


def analytical_hvp():
    class LogitModel(torch.nn.Module):
        def __init__(self):super().__init__();self.z=torch.nn.Parameter(torch.linspace(-1,1,20,dtype=torch.float64))
        def forward(self,x):return {'logits':self.z.expand(len(x),-1)}
    m=LogitModel();x=torch.zeros(7,1,dtype=torch.float64);y=torch.arange(7)
    frame=helper.State(m)
    loss=F.cross_entropy(m(x)['logits'],y);g=list(torch.autograd.grad(loss,tuple(m.parameters())))
    u=helper.fixed_unit(g);entry=helper.State(m)
    corr,r=helper.same_point_hvp(frame,g,u,x,y,0,20,.05)
    p=m.z.detach().softmax(0);exact=(torch.diag(p)-p[:,None]*p[None,:])@u[0]
    error=float((corr[0]-.05*exact).abs().max())
    assert error<1e-15 and all(entry.check().values())
    assert not torch.allclose(corr[0],exact,rtol=1e-5,atol=1e-12), 'Missing rho must be detectable'
    return {'closed_form_softmax_hessian_max_abs_error':error,'rho_20x_omission_distinguished':True,'entry_restored':True}


def reject(label,fn,types=(ValueError,AssertionError,RuntimeError,FloatingPointError)):
    try:fn()
    except types as exc:return {'case':label,'error':repr(exc)}
    raise AssertionError('Invalid case accepted '+label)


def lifecycle(first,second):
    native=trainer.resolve_config
    def resolve(raw,cli):
        return native({**raw,'component_first_response':first,'component_second_response':second,
            'component_attention_backend':'math','gam_grad_norm_rho':1.},cli)
    trainer.resolve_config=resolve
    try:result=old.controller_lifecycle()
    finally:trainer.resolve_config=native
    assert result['tasks']==2 and result['steps']==2 and result['copied_native_format_checkpoint_roundtrip']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(944)
    initial=old.Toy().state_dict();x=torch.randn(11,7);y=torch.arange(11)%20
    cases={};first_points={};max_hvp_error=0.
    for first in ('fd','hvp'):
        for second in ('fd','hvp'):
            for mode in ('natural','matched'):
                model=old.Toy();model.load_state_dict(initial);rows=[]
                for task in (0,1):
                    if task:model.expand(40)
                    parts=make_step(model,'ug',mode,first,second)
                    for step in range(2):
                        row,captured=run_step(parts,x,y+20*task);rows.append(row)
                        if task==0 and step==0:
                            first_points[first,second,mode]=captured
                        if second=='hvp':
                            u=helper.fixed_unit(captured['grads'][2])
                            h=independent_hvp_at(model,captured['points'][2],x,y+20*task,20*task,u)
                            err=max(float((c-.05*v).abs().max()) for c,v in zip(captured['second_correction'],h))
                            assert err<2e-6;max_hvp_error=max(max_hvp_error,err)
                cases[first+'/'+second+'/'+mode]={'steps':len(rows),'max_effective_match_error':max(r['actual_match_relative_error'] for r in rows) if mode=='matched' else None,
                    'all_selected_projector_sgd_exact':all(r['self_spec_passed'] for r in rows),'hvp_calls_per_step':int(first=='hvp')+int(second=='hvp')}
    for first in ('fd','hvp'):
        a=first_points[first,'fd','natural'];b=first_points[first,'hvp','natural']
        assert all(torch.equal(v,w) for point in range(3) for v,w in zip(a['points'][point],b['points'][point]))
        assert all(torch.equal(v,w) for point in range(3) for v,w in zip(a['grads'][point],b['grads'][point]))
    lambdas={}
    for rule in ('us','ug'):
        for coefficient in (.5,1.,2.):
            for mode in ('natural','matched'):
                model=old.Toy();model.load_state_dict(initial)
                row,c=run_step(make_step(model,rule,mode,coefficient=coefficient),x,y)
                if rule=='us' and coefficient==1.:assert row['us_nominal_reference_after_tensors_exact'] is True
                else:assert row['us_nominal_reference_after_tensors_exact'] is None
                lambdas[rule+'/'+str(coefficient)+'/'+mode]={'step_scale':row['step_scale'],'identity_control':row['response_audit']['us_lambda1_identity_control']}
    def new_model():
        m=old.Toy();m.load_state_dict(initial);return m
    negatives=[reject('HVP_native_backend_forbidden',lambda:make_step(new_model(),'ug','natural','fd','hvp',backend='native')),
        reject('HVP_nonUG_forbidden',lambda:make_step(new_model(),'us','natural','fd','hvp')),
        reject('nonfinite_lambda',lambda:make_step(new_model(),'ug','natural',coefficient=float('nan'))),
        reject('coefficient_for_unrelated_rule',lambda:make_step(new_model(),'uo','natural',coefficient=.5))]
    original_hvp=helper.same_point_hvp
    def unsupported(*a,**kw):raise RuntimeError('derivative for synthetic operation not implemented')
    helper.same_point_hvp=unsupported
    try:negatives.append(reject('unsupported_HVP_no_silent_FD_fallback',lambda:run_step(make_step(new_model(),'ug','natural','fd','hvp'),x,y)))
    finally:helper.same_point_hvp=original_hvp
    def missing_rho(*a,**kw):
        c,r=original_hvp(*a,**kw);return [v/.05 for v in c],r
    helper.same_point_hvp=missing_rho
    try:negatives.append(reject('rho_omission_in_real_selected_update',lambda:run_step(make_step(new_model(),'ug','natural','fd','hvp'),x,y)))
    finally:helper.same_point_hvp=original_hvp
    m=new_model();frame=helper.State(m)
    loss=F.cross_entropy(m(x)['logits'],y);g=list(torch.autograd.grad(loss,tuple(m.parameters())));u=helper.fixed_unit(g)
    negatives.append(reject('undetached_HVP_direction',lambda:original_hvp(frame,g,[v.clone().requires_grad_() for v in u],x,y,0,20,.05)))
    wrong_g=[v+.1 for v in g]
    negatives.append(reject('wrong_native_reference_gradient',lambda:original_hvp(frame,wrong_g,helper.fixed_unit(wrong_g),x,y,0,20,.05)))
    negatives.append(reject('wrong_fixed_HVP_direction',lambda:original_hvp(frame,g,[-v for v in u],x,y,0,20,.05)))
    negatives.append(reject('nonfinite_HVP_direction',lambda:original_hvp(frame,g,[torch.full_like(v,float('nan')) for v in u],x,y,0,20,.05)))
    negatives.append(reject('damaged_actual_HVP_SGD',lambda:run_step(make_step(new_model(),'ug','natural','fd','hvp',corrupt_sgd=True),x,y)))
    model=new_model();parts=make_step(model,'ug','natural','fd','hvp')
    project=parts[1]
    def corrupt_projector(clean,fisher,delta):
        native=project(clean,fisher,delta)
        def call(groups):native(groups);next(model.parameters()).grad.add_(.1)
        return call
    parts=(parts[0],corrupt_projector,parts[2],parts[3])
    negatives.append(reject('actual_projector_gradient_corruption',lambda:run_step(parts,x,y)))
    result={'status':'passed','device':'cpu','scope':'actual frozen GAM and projector with tiny CE/LoRA models; no full ViT or GPU training',
        'analytical_hvp':analytical_hvp(),'native_integration_cases':cases,'coefficient_cases':lambdas,
        'same_first_three_points_and_gradients_for_second_only_replacement':True,
        'independent_second_HVP_max_abs_error':max_hvp_error,
        'controller_two_task_HVP_lifecycle':lifecycle('fd','hvp'),
        'controller_two_task_both_HVP_lifecycle':lifecycle('hvp','hvp'),
        'negative_cases':negatives,'trainer_sha256':old.audit.digest(Path(trainer.__file__)),
        'hvp_helpers_sha256':old.audit.digest(Path(helper.__file__)),'test_sha256':old.audit.digest(__file__)}
    if args.output:old.audit.atomic_json(args.output,result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
