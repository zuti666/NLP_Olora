#!/usr/bin/env python3
"""Actual frozen GAM + AST-exact native clean/projector/batch CPU regression."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
import legacy_direction as extension
import residual_selection as residual
from effective_step import EffectiveStep
from effective_step import StepPolynomial,solve_positive_scale
import step_reference

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(SOURCE))
from optimer_PerturabtionType.gam import GAM


def native_method(name,file='as1_normfisher_gam_intask.py'):
    path=SOURCE/'models_LoRAbasedCL'/file;tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name));node.decorator_list=[]
    env={'torch':torch};exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path)+':'+name,'exec'),env)
    return env[name]


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_As=torch.nn.ModuleList([torch.nn.Linear(7,3,bias=False)])
        self.w_Bs=torch.nn.ModuleList([torch.nn.Linear(3,7,bias=False)])


class Model(torch.nn.Module):
    def __init__(self,classes=20):
        super().__init__();self.backbone=Backbone()
        self.fc=torch.nn.Linear(7,classes);self.calls=0
    @property
    def A(self):return self.backbone.w_As[0]
    @property
    def B(self):return self.backbone.w_Bs[0]
    def forward(self,x):
        # Deliberate all-family RNG consumption tests that audits do not reseed,
        # draw, or rewind. Native ViT also consumes RNG in temporary Linear init.
        random.random();np.random.random();torch.rand(())
        self.calls+=1;return {'logits':self.fc(x+self.B(self.A(x)))}


class Learner:
    _build_direction_projector=native_method('_build_direction_projector')
    _step_batch_with_log=native_method('_step_batch_with_log')
    _compute_clean_task_grad_cache=native_method('_compute_clean_task_grad_cache','ewclora_youyue_fitArchitechture_gam.py')
    _flatten_optimizer_params=staticmethod(native_method('_flatten_optimizer_params','ewclora_youyue_fitArchitechture_gam.py'))
    def __init__(self,state):
        classes=state['fc.weight'].shape[0]
        self._network=Model(classes);self._network.load_state_dict(state)
        self._lambda_flat=1.;self._lambda_ewc=0.;self._mechanism_eps=1e-12;self._mechanism_grad_stats=True
        self._optimizer_type='gam';self._cur_task=classes//20-1;self._total_classes=classes;self.mechanism=[];self.loss_stats=[]
    def _task_loss(self,logits,y,offset):return F.cross_entropy(logits[:,offset:offset+20],y-offset),None,None
    def _compute_fisher_grad_cache(self,params):return 0.,{}
    def _compute_delta_w_penalty_grad_cache(self,params):return 0.,{},{}
    def _record_mechanism_loss_stats(self,*values):self.loss_stats.append(values)
    def _accumulate_mechanism_grad_stats(self,values):self.mechanism.append(values)
    def _cosine_from_sums(self,dot,a,b,eps):return dot/(math.sqrt(a*b)+eps)


def optimizer(learner):
    model=learner._network;sgd=torch.optim.SGD(model.parameters(),lr=.04 if learner._cur_task==0 else .02,momentum=0.,weight_decay=0.)
    return GAM(model.parameters(),sgd,model,grad_rho=.05,grad_norm_rho=1.,adaptive=False,
        args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.))


def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)


def oracle_selector(opt,selector):
    if selector in ('g1_control','native_control','native_beta'):return
    original=opt.gradient_decompose
    def changed(_opt,*args,**kwargs):
        result=original(*args,**kwargs)
        with torch.no_grad():
            parameters=[p for group in opt.param_groups for p in group['params']]
            if selector in residual.C21_SELECTORS:
                # Independent oracle: concatenate every coordinate rather than
                # using the production per-tensor FP64 reduction function.
                g1=torch.cat([opt.state[p]['g_1'].double().flatten() for p in parameters])
                delta=torch.cat([(opt.state[p]['g_2']-opt.state[p]['g_1']).double().flatten() for p in parameters])
                coefficient=float(g1@delta)/float(g1@g1) if bool((g1!=0.).any()) else 0.
            for p in parameters:
                v=opt.state[p]
                if selector=='c21_base':expected=v['g_0']+(v['g_2']-v['g_1'])
                elif selector=='c21_perp':expected=v['g_1']+((v['g_2']-v['g_1'])-coefficient*v['g_1'])
                elif selector=='g3':expected=v['g_3']
                elif selector=='sum':expected=v['g_1']+v['g_3']
                elif selector=='mean':expected=.5*v['g_1']+.5*v['g_3']
                else:expected=v['g_0'] if selector=='g0' else v['g_2'] if selector=='g2' else v['g_0']+(v['g_3']-v['g_2'])
                p.grad.copy_(expected)
        return result
    opt.gradient_decompose=MethodType(changed,opt)


def run(state,selector,patched,path):
    learner=Learner(state);opt=optimizer(learner);factory=SimpleNamespace(get_model=lambda *_:learner)
    initial_get=factory.get_model;traces=[];source_class=GAM.gradient_decompose
    if patched:
        context=extension.install(factory,selector,path,SOURCE);context.__enter__();factory.get_model('mock',{})
    else:oracle_selector(opt,selector)
    seed(4122);offset=20*learner._cur_task
    try:
        for step in range(4):
            # Global-RNG data order/augmentation proxy, deliberately outside hook.
            indices=torch.randperm(13)[:9];x=torch.randn(9,7);y=indices%20
            output=learner._step_batch_with_log(opt,x,y+offset,offset)
            traces.append({'parameters':extension.tensors_sha(list(learner._network.parameters())),
                'gradients':extension.tensors_sha([p.grad for p in learner._network.parameters()]),
                'RNG':extension.rng_sha(),'input':extension.tensors_sha([x,y]),'loss':output[1],
                'native_four_gradient_hashes':{f'g{i}':extension.tensors_sha([opt.state[p][f'g_{i}'] for p in learner._network.parameters()]) for i in range(4)}})
    finally:
        if patched:context.__exit__(*sys.exc_info())
    assert learner._network.calls==20 and len(learner.mechanism)==len(learner.loss_stats)==4
    assert GAM.gradient_decompose is source_class and factory.get_model is initial_get
    if patched:
        assert '_step_batch_with_log' not in learner.__dict__
        assert all(name not in opt.__dict__ for name in ('step','gradient_decompose'))
        assert 'step' not in opt.base_optimizer.__dict__
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==4 and all((r['forward_count'],r['closure_count'],r['sgd_count'])==(5,4,1) for r in rows)
        assert all(r['selected_exact'] and r['projector_exact'] and r['sgd_exact'] for r in rows)
        assert all(r['selection_rng_unchanged'] and r['projector_rng_unchanged'] and r['sgd_rng_unchanged'] for r in rows)
        assert all(r['decomposition_method_wrapped']==(selector!='native_control') for r in rows)
        assert all(r['selected_expected']==r['selected_actual'] and r['projected_expected']==r['projected_actual']==r['actual_sgd_gradient'] for r in rows)
        assert all(r['effective_step_measurement_only'] and r['reference_counterfactual_state_unchanged'] for r in rows)
        assert all(r['lr_nominal']==r['step_scale'] and r['scale_over_nominal_lr']==1.
                   and not r['final_norm_matching'] for r in rows)
        if selector in residual.C21_SELECTORS:
            assert all(r['c21_audit']['orthogonality_passed'] and r['c21_audit']['selected_spec_exact'] for r in rows)
            assert all(r['c21_audit']['parameter_count']==sum(p.numel() for p in learner._network.parameters()) for r in rows)
        summary=json.loads(context.summary_path.read_text());assert summary['status']=='completed' and summary['steps']==4
    return traces,learner.mechanism


def negative_case(root,state,name,mutation,selector='g0'):
    learner=Learner(state);opt=optimizer(learner);factory=SimpleNamespace(get_model=lambda *_:learner)
    mutation(learner,opt)
    try:
        with extension.install(factory,selector,root/(name+'.jsonl'),SOURCE):
            factory.get_model('mock',{});learner._step_batch_with_log(opt,torch.ones(4,7),torch.arange(4),0)
    except (AssertionError,ValueError,FloatingPointError) as exc:
        assert GAM.gradient_decompose is ORIGINAL_DECOMPOSE
        assert '_step_batch_with_log' not in learner.__dict__
        return {'case':name,'error':repr(exc)}
    raise AssertionError('Invalid update accepted '+name)


ORIGINAL_DECOMPOSE=GAM.gradient_decompose


def rejected(function,error,label):
    try:function()
    except error as exc:return {'case':label,'error':repr(exc)}
    raise AssertionError('Invalid input accepted '+label)


def numeric_checks(root,state):
    def gradients(g0,g1,g2,dtype=torch.float64):
        return [{key:torch.tensor(value,dtype=dtype) for key,value in
                 zip(('g_0','g_1','g_2','g_3'),values)} for values in zip(g0,g1,g2,g2)]
    examples={
        'global_not_per_tensor':gradients([[3.],[4.]],[[1.],[2.]],[[3.],[1.]]),
        'parallel':gradients([[2.,-1.],[.5]],[[1.,2.],[3.]],[[4.,8.],[12.]]),
        'perpendicular_no_norm_matching':gradients([[2.,-1.]],[[1.,0.]],[[1.,2.]]),
        'g1_zero':gradients([[2.,-1.]],[[0.,0.]],[[3.,4.]]),
        'delta_zero':gradients([[2.,-1.]],[[1.,2.]],[[1.,2.]]),
        'all_zero':gradients([[0.,0.]],[[0.,0.]],[[0.,0.]]),
        'tiny_nonzero_fp32':gradients([[2.,-1.]],[[1e-20,2e-20]],[[3e-20,6e-20]],torch.float32)}
    checks=[];negatives=[]
    for name,values in examples.items():
        for rule in residual.C21_SELECTORS:
            before=copy.deepcopy(values);rng=extension.rng_sha()
            selected,record=residual.selected_c21_direction(values,rule)
            g1=torch.cat([v['g_1'].double().flatten() for v in values])
            delta=torch.cat([(v['g_2']-v['g_1']).double().flatten() for v in values])
            coefficient=float(g1@delta)/float(g1@g1) if bool((g1!=0.).any()) else 0.
            expected=[v['g_0']+(v['g_2']-v['g_1']) if rule=='c21_base'
                      else v['g_1']+((v['g_2']-v['g_1'])-coefficient*v['g_1']) for v in values]
            assert all(torch.equal(a,b) for a,b in zip(selected,expected))
            assert math.isclose(coefficient,record['coefficient_a'],rel_tol=1e-12,abs_tol=1e-15)
            assert all(torch.equal(v[k],b[k]) for v,b in zip(values,before) for k in v)
            assert extension.rng_sha()==rng and record['orthogonality_passed']
            assert not record['final_norm_matching']
            if name=='global_not_per_tensor' and rule=='c21_perp':
                assert coefficient==0. and [float(x) for x in selected]==[3.,1.]
                # Each tensor has one coordinate: per-tensor projection would
                # erase each residual and return [1,2], proving the distinction.
                assert [float(x) for x in selected]!=[1.,2.]
            if name=='parallel':assert record['norms']['perp']==0.
            if name=='perpendicular_no_norm_matching' and rule=='c21_perp':
                assert math.isclose(record['norms']['selected'],math.sqrt(5.))
                assert record['norms']['selected']!=record['norms']['g1']
            if name=='tiny_nonzero_fp32':assert not record['g1_zero']
            checks.append(name+'/'+rule)
    for value in (float('nan'),float('inf')):
        negatives.append(rejected(lambda:residual.selected_c21_direction(
            gradients([[0.]],[[1.]],[[value]]),'c21_perp'),FloatingPointError,'nonfinite_'+str(value)))
    negatives.append(rejected(lambda:residual.selected_c21_direction([], 'c21_base'),ValueError,'empty_states'))
    negatives.append(rejected(lambda:residual.selected_c21_direction(examples['parallel'],'matched'),ValueError,'matching_selector_forbidden'))
    # AST-exact native FP32 projector must retain its cancellation arithmetic.
    learner=Learner(state);p=next(learner._network.parameters())
    p.grad=torch.ones_like(p);clean={id(p):torch.full_like(p,1e10)}
    expected=clean[id(p)]+1.*(p.grad-clean[id(p)])
    assert bool((expected==0.).all()) and not torch.equal(expected,p.grad)
    rng=extension.rng_sha()
    learner._build_direction_projector(clean,{},{})([{'params':[p]}])
    assert torch.equal(p.grad,expected) and extension.rng_sha()==rng
    checks.append('AST_native_FP32_clean_projector_cancellation')
    # Dense small-matrix oracle independently checks actual factor values.
    model=Model();geometry=EffectiveStep.from_model(model);before=geometry.capture()
    after=[v.clone().add_(torch.linspace(-.03,.04,v.numel()).reshape(v.shape)) for v in before]
    actual=geometry.actual_delta(before,after)
    a0,b0,a1,b1=(v.double() for v in (before[0],before[1],after[0],after[1]))
    dense_lora=float(((b1@a1)-(b0@a0)).square().sum())
    dense_fc=sum(float((b.double()-a.double()).square().sum()) for a,b in zip(before[2:],after[2:]))
    assert math.isclose(actual['lora_norm'],math.sqrt(dense_lora),rel_tol=1e-12,abs_tol=1e-15)
    assert math.isclose(actual['fc_norm'],math.sqrt(dense_fc),rel_tol=1e-12,abs_tol=1e-15)
    assert math.isclose(actual['joint_norm'],math.sqrt(dense_lora+dense_fc),rel_tol=1e-12,abs_tol=1e-15)
    checks.append('effective_BA_plus_full_FC_dense_float64_oracle')
    # Also validate both logged effective norms against dense calculations at
    # the actual native SGD call, after its restoration, not a predicted point.
    for rule in residual.C21_SELECTORS:
        learner=Learner(state);opt=optimizer(learner);factory=SimpleNamespace(get_model=lambda *_:learner)
        geom=EffectiveStep.from_model(learner._network);native_sgd=opt.base_optimizer.step;measurements=[]
        native_clean=learner._compute_clean_task_grad_cache;clean_cache={}
        def observed_clean(_l,*a,**kw):
            result=native_clean(*a,**kw);clean_cache.update(result[-1]);return result
        learner._compute_clean_task_grad_cache=MethodType(observed_clean,learner)
        def dense_delta(old,new):
            aa,bb,an,bn=(v.double() for v in (old[0],old[1],new[0],new[1]))
            return math.sqrt(float(((bn@an)-(bb@aa)).square().sum())+
                             sum(float((b.double()-a.double()).square().sum()) for a,b in zip(old[2:],new[2:])))
        def observed_sgd(*a,**kw):
            old=geom.capture();eta=opt.base_optimizer.param_groups[0]['lr']
            ref=[v.clone().add_(clean_cache[id(p)]+1.*(opt.state[p]['g_1']-clean_cache[id(p)]),alpha=-eta)
                 for p,v in zip(geom.params,old)]
            result=native_sgd(*a,**kw)
            measurements.append((dense_delta(old,geom.capture()),dense_delta(old,ref)))
            return result
        opt.base_optimizer.step=observed_sgd
        path=root/(rule+'_dense.jsonl')
        with extension.install(factory,rule,path,SOURCE):
            factory.get_model('mock',{})
            for _ in range(3):learner._step_batch_with_log(opt,torch.randn(9,7),torch.arange(9),0)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row,(actual,reference) in zip(rows,measurements):
            assert math.isclose(row['actual_effective_delta']['joint_norm'],actual,rel_tol=1e-11,abs_tol=1e-14)
            assert math.isclose(row['same_state_g1_effective_delta']['joint_norm'],reference,rel_tol=1e-11,abs_tol=1e-14)
        checks.append(rule+'/actual_native_SGD_effective_metrics_dense_oracle_3_steps')
    # Inject prohibited selection side effects and ensure execution fails.
    original=extension.selected_c21_direction
    for label,mutate in [('selection_RNG_draw',lambda values:torch.rand(())),
                         ('selection_native_cache_mutation',lambda values:values[0]['g_0'].add_(.1))]:
        def corrupted(values,rule,_mutate=mutate):
            result=original(values,rule);_mutate(values);return result
        extension.selected_c21_direction=corrupted
        try:negatives.append(negative_case(root,state,label,lambda l,o:None,'c21_perp'))
        finally:extension.selected_c21_direction=original
    negatives.append(negative_case(root,state,'unsupported_effective_module_scaling',
        lambda l,o:setattr(l._network.backbone,'learn_alpha',True),'c21_perp'))
    def matching_sgd(l,o):
        real=o.base_optimizer.step
        def changed(*a,**kw):
            for group in o.base_optimizer.param_groups:group['lr']*=.5
            return real(*a,**kw)
        o.base_optimizer.step=changed
    negatives.append(negative_case(root,state,'SGD_norm_matching_or_LR_change',matching_sgd,'c21_perp'))
    return {'checks':checks,'negative_cases':negatives}


def matching_regression(root,initial,matching_seed=2):
    checks=[];negatives=[]
    paired={}
    for selector,eta in [('sum',.04),('mean',.08)]:
        learner=Learner(initial);opt=optimizer(learner)
        for group in opt.param_groups:group['lr']=eta
        factory=SimpleNamespace(get_model=lambda *_:learner)
        path=root/(selector+'_magnitude.jsonl');seed(987)
        inputs=torch.randn(128,7);labels=torch.arange(128)%20
        with extension.install(factory,selector,path,SOURCE):
            factory.get_model('mock',{});learner._step_batch_with_log(opt,inputs,labels,0)
        row=json.loads(path.read_text())
        paired[selector]={'row':row,'parameters':[p.detach().clone() for p in learner._network.parameters()]}
    assert paired['sum']['row']['probe_parameter_sha256']==paired['mean']['row']['probe_parameter_sha256']
    assert paired['sum']['row']['gradients']==paired['mean']['row']['gradients']
    snorm=paired['sum']['row']['actual_effective_delta']['joint_norm']
    mnorm=paired['mean']['row']['actual_effective_delta']['joint_norm']
    relative=abs(snorm-mnorm)/snorm
    assert relative<=1e-3
    max_error=max(float((a-b).abs().max()) for a,b in zip(paired['sum']['parameters'],paired['mean']['parameters']))
    checks.append('same_state_native_Sum_eta_vs_Mean2_2eta_actual_effective_magnitude')
    cancellation=[]
    for clean,value in [(1.,1e-8),(1e10,1024.),(-1e10,-1024.),(1e10,512.),(-1e10,-512.)]:
        c=torch.tensor([clean],dtype=torch.float32);g1=torch.tensor([value],dtype=torch.float32)
        g3=torch.tensor([value],dtype=torch.float32)
        raw_sum=g1+g3;raw_mean=.5*g1+.5*g3
        projected_sum=c+1.*(raw_sum-c);projected_mean=c+1.*(raw_mean-c)
        start=torch.tensor([.1234567],dtype=torch.float32)
        after_sum=start.clone().add_(projected_sum,alpha=-.04)
        after_mean=start.clone().add_(projected_mean,alpha=-.08)
        cancellation.append({'clean':clean,'value':value,'projected_sum':float(projected_sum),
                             'projected_mean':float(projected_mean),
                             'actual_parameter_difference':float((after_sum-after_mean).abs()),
                             'bitwise_identical':torch.equal(after_sum,after_mean)})
    # The cancellation witnesses are observations, not a false universal
    # bitwise-equivalence gate on long trajectories or arbitrary gradients.
    assert any(not case['bitwise_identical'] for case in cancellation)
    checks.append('FP32_projector_Sum_Mean2_cancellation_observations')
    # Immutable receipt fixtures are generated from actual native CPU training,
    # with the complete 18-step/2256-sample smoke budget at 20/40-class heads.
    reference_dir=root/'independent_s';reference_dir.mkdir()
    config={'seed':[matching_seed],'dataset':'imageneta','class_shuffle':False,'init_cls':20,
            'increment':20,'nb_tasks':10,'batch_size':128,'gam_grad_norm_rho':1.,
            'gam_grad_rho':.05,'gam_grad_beta_1':1.,'gam_grad_beta_2':1.,
            'gam_grad_beta_3':0.,'gam_grad_gamma':0.,'init_lr':.04,'lrate':.02,
            'init_epoch':1,'epochs':1,'prefix':'CPU_reference_fixture'}
    config_path=reference_dir/'config.json';config_path.write_text(json.dumps(config))
    job={'id':'CPU_independent_s_fixture','phase':'smoke','direction_selector':'g1_control',
         'seed':matching_seed,'expected_steps':18,'expected_samples':2256,
         'expected_task_steps':{'0':10,'1':8},'config_sha256':extension.sha_file(config_path)}
    spec={'job':job,'run_dir':str(reference_dir),'candidate_config':{**config,'prefix':'CPU_gmatch_fixture'},
          'candidate_phase':'smoke'}

    def run_native(selector,directory,reference_spec=None):
        learner=Learner(initial);factory=SimpleNamespace(get_model=lambda *_:learner)
        seed(918245);path=directory/'direction_audit.jsonl'
        with extension.install(factory,selector,path,SOURCE,reference_spec=reference_spec) as session:
            factory.get_model('mock',{})
            for task,batches,total in ((0,10,1259),(1,8,997)):
                if task:
                    old=learner._network.fc;new=torch.nn.Linear(7,40)
                    with torch.no_grad():
                        new.weight[:20].copy_(old.weight);new.bias[:20].copy_(old.bias)
                    learner._network.fc=new;learner._cur_task=1;learner._total_classes=40
                opt=optimizer(learner)
                for batch in range(batches):
                    size=128 if batch<batches-1 else total-128*(batches-1)
                    inputs=torch.randn(size,7);labels=torch.arange(size)%20+20*task
                    learner._step_batch_with_log(opt,inputs,labels,20*task)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert session.steps==18 and session.samples==2256
        assert all(row['nominal_learning_rates_restored'] for row in rows)
        return rows,session

    reference_rows,session=run_native('g1_control',reference_dir)
    receipt={'id':job['id'],'status':'completed','config_sha256':job['config_sha256'],
             'direction_proof':{'audit_sha256':extension.sha_file(session.path),
                                'audit_summary_sha256':extension.sha_file(session.summary_path),'steps':18}}
    (reference_dir/'completed.json').write_text(json.dumps(receipt))
    target_reader=step_reference.StepReference(spec,SOURCE,extension.__file__)
    directory=root/'gmatch';directory.mkdir();actual_rows,_=run_native('gmatch',directory,spec)
    differs_from_same_candidate=False
    for candidate,reference in zip(actual_rows,reference_rows):
        target=reference['actual_effective_delta']['joint_norm']
        assert candidate['reference_target_norm']==target
        assert candidate['reference_effective_delta']==reference['actual_effective_delta']
        assert candidate['reference_step_index']==reference['step_index']
        assert candidate['reference_batch_identity_exact'] and candidate['matched']
        assert candidate['actual_match_relative_error']<=1e-3
        assert candidate['matching_solver']['fp32_predicted_relative_error']<=1e-3
        assert candidate['matching_solver']['no_live_trial_updates']
        assert candidate['reference_counterfactual_state_unchanged']
        assert candidate['sgd_count']==1 and candidate['closure_count']==4 and candidate['forward_count']==5
        same=candidate['same_state_g1_effective_delta']['joint_norm']
        differs_from_same_candidate |= abs(same-target)>1e-6*target
    assert differs_from_same_candidate,'Fixture must distinguish independent S from same-candidate g1 target'
    checks.append('native_S_to_Gmatch_18_steps_2256_samples_heads20_40_global_RNG_and_actual_match')
    checks.append('independent_S_target_differs_from_same_candidate_projected_g1')
    # Reference mismatch failures occur before any actual optimizer update.
    current=copy.deepcopy(reference_rows[0]);nominal=current['lr_nominal']
    for field,value in [('input_sha256','wrong'),('rng_before_batch_sha256','wrong'),
                        ('seen_classes',40),('parameter_shapes',[[999]])]:
        wrong=copy.deepcopy(current);wrong[field]=value
        negatives.append(rejected(lambda wrong=wrong:target_reader.get(wrong,nominal),ValueError,'S_identity_'+field))
    negatives.append(rejected(lambda:target_reader.get({**current,'task_step_index':999},nominal),ValueError,'missing_S_step'))
    negatives.append(rejected(lambda:target_reader.get(current,.08),ValueError,'S_nominal_LR_mismatch'))
    for label,mutation in [('S_config_mismatch',lambda s:s['candidate_config'].update(batch_size=64)),
                           ('S_seed_mismatch',lambda s:s['job'].update(seed=(matching_seed+1)%3)),
                           ('formal_rejects_smoke_S',lambda s:s.update(candidate_phase='formal')),
                           ('incomplete_S_budget',lambda s:s['job'].update(expected_steps=17))]:
        wrong=copy.deepcopy(spec);mutation(wrong)
        negatives.append(rejected(lambda wrong=wrong:step_reference.StepReference(wrong,SOURCE,extension.__file__),ValueError,label))
    saved=session.path.read_bytes();session.path.write_bytes(saved+b' ')
    try:
        negatives.append(rejected(lambda:step_reference.StepReference(spec,SOURCE,extension.__file__),ValueError,'S_trace_hash_corruption'))
        negatives.append(rejected(target_reader.verify_unchanged,ValueError,'S_mutation_after_reader_initialization'))
    finally:session.path.write_bytes(saved)
    negatives.append(rejected(lambda:extension.install(SimpleNamespace(),'gmatch',root/'unused',SOURCE),ValueError,'Gmatch_requires_independent_S'))
    negatives.append(rejected(lambda:extension.install(SimpleNamespace(),'g3',root/'unused',SOURCE,reference_spec=spec),ValueError,'natural_arm_rejects_matching_reference'))
    # Independent low-rank Gram polynomial versus dense BA+all-FC algebra.
    model=Model();geometry=EffectiveStep.from_model(model)
    for p in geometry.params:p.grad=torch.linspace(-.15,.2,p.numel()).reshape(p.shape)
    before=geometry.capture();scale=.037
    target=geometry.actual_delta(before,[v.clone().add_(p.grad,alpha=-scale) for p,v in zip(geometry.params,before)])['joint_norm']
    saved_rng=extension.rng_sha();saved_grad=[p.grad.clone() for p in geometry.params]
    solved,after,record=step_reference.solve_fp32_scale(geometry,before,target,.04)
    a0,b0,a1,b1=(v.double() for v in (before[0],before[1],after[0],after[1]))
    dense=math.sqrt(float((b1@a1-b0@a0).square().sum())+
                    sum(float((b.double()-a.double()).square().sum()) for a,b in zip(before[2:],after[2:])))
    assert abs(dense-target)/target<=1e-3
    assert record['fp32_predicted_relative_error']<=1e-3 and solved>0.
    assert all(torch.equal(p,v) and torch.equal(p.grad,g) for p,v,g in zip(geometry.params,before,saved_grad))
    assert extension.rng_sha()==saved_rng
    checks.append('FP64_Gram_root_FP32_clone_dense_BA_and_all_FC_oracle_pure')
    poly=StepPolynomial(4.,-4.,1.)
    root_result=solve_positive_scale(poly,.75,max_scale=4.)
    assert abs(root_result['scale']-.5)<1e-8 and len(root_result['stationary_points'])==2
    checks.append('nonmonotonic_BA_polynomial_smallest_positive_root')
    for label,value in [('zero_target',0.),('nonfinite_target',float('nan'))]:
        negatives.append(rejected(lambda value=value:step_reference.solve_fp32_scale(geometry,before,value,.04),
                                  (ValueError,FloatingPointError),label))
    negatives.append(rejected(lambda:step_reference.solve_fp32_scale(geometry,before,1e10,.04,max_scale=.001),ValueError,'unreachable_scale_bound'))
    for p in geometry.params:p.grad.zero_()
    negatives.append(rejected(lambda:step_reference.solve_fp32_scale(geometry,before,1.,.04),ValueError,'zero_effective_direction'))
    # G025 declares its radius explicitly; all nominal updates remain natural.
    learner=Learner(initial);opt=optimizer(learner);opt.grad_norm_rho=.25
    factory=SimpleNamespace(get_model=lambda *_:learner)
    with extension.install(factory,'g3',root/'radius025.jsonl',SOURCE,expected_radius=.25):
        factory.get_model('mock',{});learner._step_batch_with_log(opt,torch.ones(4,7),torch.arange(4),0)
    checks.append('G025_declared_radius_native_path')
    return {'checks':checks,'negative_cases':negatives,'native_matched_steps':18,
            'same_candidate_g1_would_be_a_different_target':differs_from_same_candidate,
            'same_state_Sum_Mean2_relative_effective_norm_difference':relative,
            'same_state_Sum_Mean2_max_parameter_difference':max_error,
            'FP32_projector_cancellation_cases':cancellation,
            'no_full_trajectory_equivalence_claim':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);seed(823);state=Model().state_dict();cases={};first={}
    with tempfile.TemporaryDirectory(prefix='legacy_direction_cpu_') as temporary:
        root=Path(temporary)
        numeric=numeric_checks(root,state)
        matching=matching_regression(root,state)
        for classes in (20,40):
            head_state=state if classes==20 else Model(classes).state_dict()
            paired_first={}
            for selector in (v for v in extension.SELECTORS if v not in ('gmatch','native_beta')):
                name=selector if classes==20 else selector+'_head40'
                reference,ref_metrics=run(head_state,selector,False,root/'unused')
                actual,metrics=run(head_state,selector,True,root/(name+'.jsonl'))
                assert reference==actual and ref_metrics==metrics,name
                paired_first[selector]=actual[0]['native_four_gradient_hashes']
                cases[name]={'steps':4,'classes':classes,'oracle_parameters_gradients_losses_mechanism_RNG_and_next_inputs_exact':True,
                    'native_forward_count':20,'native_gradient_closures':16,
                    'reference_is_completely_unpatched':selector in ('g1_control','native_control')}
            assert all(v==paired_first['g1_control'] for v in paired_first.values()),'Selection changed same-start probe gradients'
            first[str(classes)]=paired_first
        # Factory binding only affects the first training learner. A different
        # diagnostic GAM instance on that same learner is not globally patched.
        learner=Learner(state);other=Learner(state);opt=optimizer(learner);diagnostic=optimizer(learner)
        pending=[learner,other];factory=SimpleNamespace(get_model=lambda *_:pending.pop(0))
        with extension.install(factory,'g0',root/'scope.jsonl',SOURCE) as session:
            assert factory.get_model('first',{}) is learner
            assert factory.get_model('diagnostic',{}) is other
            assert '_step_batch_with_log' not in other.__dict__
            learner._step_batch_with_log(opt,torch.randn(4,7),torch.arange(4),0)
            assert session.steps==1 and 'gradient_decompose' not in diagnostic.__dict__
            other._step_batch_with_log(optimizer(other),torch.randn(4,7),torch.arange(4),0)
            assert session.steps==1 and GAM.gradient_decompose is ORIGINAL_DECOMPOSE
        def bad_projector(l,o):
            original=l._build_direction_projector
            def factory(_l,*a,**kw):
                callback=original(*a,**kw)
                def wrapped(groups):
                    result=callback(groups)
                    with torch.no_grad():next(l._network.parameters()).grad.add_(.01)
                    return result
                return wrapped
            l._build_direction_projector=MethodType(factory,l)
        def bad_sgd(l,o):
            original=o.base_optimizer.step
            def wrapped(*a,**kw):
                result=original(*a,**kw)
                with torch.no_grad():next(l._network.parameters()).add_(.01)
                return result
            o.base_optimizer.step=wrapped
        def bad_cache(l,o):
            original=o.gradient_decompose
            def wrapped(_o,*a,**kw):
                result=original(*a,**kw);o.state[next(l._network.parameters())]['g_0'].add_(.1);return result
            o.gradient_decompose=MethodType(wrapped,o)
        def missing_sgd(l,o):
            def step(_o,closure=None):
                for _ in range(4):result=closure()
                return result
            o.step=MethodType(step,o)
        negatives=[negative_case(root,state,'gamma_changed',lambda l,o:setattr(o.args,'grad_gamma',.1)),
            negative_case(root,state,'radius_changed',lambda l,o:setattr(o,'grad_norm_rho',.25)),
            negative_case(root,state,'lambda_flat_changed',lambda l,o:setattr(l,'_lambda_flat',.5)),
            negative_case(root,state,'projector_corrupted',bad_projector),
            negative_case(root,state,'actual_SGD_corrupted',bad_sgd),
            negative_case(root,state,'native_gradient_cache_corrupted',bad_cache),
            negative_case(root,state,'missing_actual_SGD',missing_sgd)]
        learner=Learner(state);opt=optimizer(learner);factory=SimpleNamespace(get_model=lambda *_:learner)
        original_set=extension.set_temporary;original_factory=factory.get_model
        def partial_install(obj,name,value,restores):
            original_set(obj,name,value,restores)
            if obj is opt and name=='step':raise RuntimeError('intentional partial optimizer-hook installation failure')
        extension.set_temporary=partial_install
        try:
            try:
                with extension.install(factory,'g0',root/'partial_install.jsonl',SOURCE):
                    factory.get_model('mock',{});learner._step_batch_with_log(opt,torch.ones(4,7),torch.arange(4),0)
            except RuntimeError as exc:
                assert 'intentional partial' in str(exc)
                negatives.append({'case':'partial_install_failure_restores_methods_and_forward_hook','error':repr(exc)})
            else:raise AssertionError('Injected installation failure was swallowed')
        finally:extension.set_temporary=original_set
        assert 'step' not in opt.__dict__ and '_step_batch_with_log' not in learner.__dict__
        assert not learner._network._forward_hooks and factory.get_model is original_factory
    report={'status':'PASS','passed':True,'device':'cpu','cases':cases,'negative_cases':negatives,
        'c21_numeric_and_native_effective_step':numeric,
        'full_task_native_independent_S_matching':matching,
        'same_start_four_probe_gradients_equal_across_all_selectors':True,
        'factory_first_training_learner_only_and_other_GAM_instances_untouched':True,
        'full_ViT_or_GPU_not_exercised':True,'legacy_direction_sha256':extension.sha_file(extension.__file__),
        'residual_selection_sha256':extension.sha_file(residual.__file__),
        'effective_step_sha256':extension.sha_file(Path(__file__).with_name('effective_step.py')),
        'step_reference_sha256':extension.sha_file(step_reference.__file__),
        'test_sha256':extension.sha_file(__file__),'source_gam_sha256':extension.sha_file(SOURCE/'optimer_PerturabtionType/gam.py'),
        'source_files_sha256':{name:extension.sha_file(SOURCE/name) for name in
            ['optimer_PerturabtionType/gam.py','models_LoRAbasedCL/as1_normfisher_gam_intask.py','models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py']},
        'native_AST_methods':['as1_normfisher_gam_intask._step_batch_with_log','as1_normfisher_gam_intask._build_direction_projector',
            'ewclora_youyue_fitArchitechture_gam._compute_clean_task_grad_cache','ewclora_youyue_fitArchitechture_gam._flatten_optimizer_params']}
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
