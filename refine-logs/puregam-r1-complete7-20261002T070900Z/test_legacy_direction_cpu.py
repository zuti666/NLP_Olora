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

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(SOURCE))
from optimer_PerturabtionType.gam import GAM


def native_method(name,file='as1_normfisher_gam_intask.py'):
    path=SOURCE/'models_LoRAbasedCL'/file;tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name));node.decorator_list=[]
    env={'torch':torch};exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path)+':'+name,'exec'),env)
    return env[name]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.A=torch.nn.Linear(7,3,bias=False);self.B=torch.nn.Linear(3,7,bias=False)
        self.fc=torch.nn.Linear(7,20);self.calls=0
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
        self._network=Model();self._network.load_state_dict(state)
        self._lambda_flat=1.;self._lambda_ewc=0.;self._mechanism_eps=1e-12;self._mechanism_grad_stats=True
        self._optimizer_type='gam';self._cur_task=0;self._total_classes=20;self.mechanism=[];self.loss_stats=[]
    def _task_loss(self,logits,y,offset):return F.cross_entropy(logits[:,offset:offset+20],y-offset),None,None
    def _compute_fisher_grad_cache(self,params):return 0.,{}
    def _compute_delta_w_penalty_grad_cache(self,params):return 0.,{},{}
    def _record_mechanism_loss_stats(self,*values):self.loss_stats.append(values)
    def _accumulate_mechanism_grad_stats(self,values):self.mechanism.append(values)
    def _cosine_from_sums(self,dot,a,b,eps):return dot/(math.sqrt(a*b)+eps)


def optimizer(learner):
    model=learner._network;sgd=torch.optim.SGD(model.parameters(),lr=.04,momentum=0.,weight_decay=0.)
    return GAM(model.parameters(),sgd,model,grad_rho=.05,grad_norm_rho=1.,adaptive=False,
        args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.))


def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)


def oracle_selector(opt,selector):
    if selector in ('g1_control','native_control'):return
    original=opt.gradient_decompose
    def changed(_opt,*args,**kwargs):
        result=original(*args,**kwargs)
        with torch.no_grad():
            for p in (p for group in opt.param_groups for p in group['params']):
                v=opt.state[p]
                expected=v['g_0'] if selector=='g0' else v['g_2'] if selector=='g2' else v['g_0']+(v['g_3']-v['g_2'])
                p.grad.copy_(expected)
        return result
    opt.gradient_decompose=MethodType(changed,opt)


def run(state,selector,patched,path):
    learner=Learner(state);opt=optimizer(learner);factory=SimpleNamespace(get_model=lambda *_:learner)
    initial_get=factory.get_model;traces=[];source_class=GAM.gradient_decompose
    if patched:
        context=extension.install(factory,selector,path,SOURCE);context.__enter__();factory.get_model('mock',{})
    else:oracle_selector(opt,selector)
    seed(4122)
    try:
        for step in range(4):
            # Global-RNG data order/augmentation proxy, deliberately outside hook.
            indices=torch.randperm(13)[:9];x=torch.randn(9,7);y=indices%20
            output=learner._step_batch_with_log(opt,x,y,0)
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);seed(823);state=Model().state_dict();cases={};first={}
    with tempfile.TemporaryDirectory(prefix='legacy_direction_cpu_') as temporary:
        root=Path(temporary)
        for selector in extension.SELECTORS:
            reference,ref_metrics=run(state,selector,False,root/'unused')
            actual,metrics=run(state,selector,True,root/(selector+'.jsonl'))
            assert reference==actual and ref_metrics==metrics,selector
            first[selector]=actual[0]['native_four_gradient_hashes']
            cases[selector]={'steps':4,'oracle_parameters_gradients_losses_mechanism_RNG_and_next_inputs_exact':True,
                'native_forward_count':20,'native_gradient_closures':16,
                'reference_is_completely_unpatched':selector in ('g1_control','native_control')}
        assert all(v==first['g1_control'] for v in first.values()),'Selection changed same-start probe gradients'
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
        'same_start_four_probe_gradients_equal_across_all_selectors':True,
        'factory_first_training_learner_only_and_other_GAM_instances_untouched':True,
        'full_ViT_or_GPU_not_exercised':True,'legacy_direction_sha256':extension.sha_file(extension.__file__),
        'test_sha256':extension.sha_file(__file__),'source_gam_sha256':extension.sha_file(SOURCE/'optimer_PerturabtionType/gam.py'),
        'source_files_sha256':{name:extension.sha_file(SOURCE/name) for name in
            ['optimer_PerturabtionType/gam.py','models_LoRAbasedCL/as1_normfisher_gam_intask.py','models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py']},
        'native_AST_methods':['as1_normfisher_gam_intask._step_batch_with_log','as1_normfisher_gam_intask._build_direction_projector',
            'ewclora_youyue_fitArchitechture_gam._compute_clean_task_grad_cache','ewclora_youyue_fitArchitechture_gam._flatten_optimizer_params']}
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
