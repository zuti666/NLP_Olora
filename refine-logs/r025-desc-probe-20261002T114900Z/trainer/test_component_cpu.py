#!/usr/bin/env python3
"""CPU actual frozen-GAM tests for five components, two scales and two heads."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import audit_core as audit
import component_horizon as component

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(SOURCE))
from optimer_PerturabtionType.gam import GAM


def native_projector():
    tree=ast.parse((SOURCE/'models_LoRAbasedCL/as1_normfisher_gam_intask.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_direction_projector')
    env={'torch':torch};exec(compile('from __future__ import annotations\n'+ast.unparse(fun),'<frozen-projector>','exec'),env)
    return env['_build_direction_projector']


class Toy(torch.nn.Module):
    def __init__(self,classes=20):
        super().__init__();self.backbone=torch.nn.Module()
        self.backbone.w_As=torch.nn.ModuleList([torch.nn.Linear(7,3,bias=False),torch.nn.Linear(7,3,bias=False)])
        self.backbone.w_Bs=torch.nn.ModuleList([torch.nn.Linear(3,7,bias=False),torch.nn.Linear(3,7,bias=False)])
        self.backbone.w_meta=[{'layer':0,'channel':'q'},{'layer':0,'channel':'v'}]
        self.backbone.task_id=0;self.fc=torch.nn.Linear(7,classes)
    def forward(self,x):
        for a,b in zip(self.backbone.w_As,self.backbone.w_Bs):x=x+b(a(x))
        return {'logits':self.fc(x)}
    def expand(self,classes):
        old=self.fc;new=torch.nn.Linear(7,classes)
        with torch.no_grad():new.weight[:old.out_features].copy_(old.weight);new.bias[:old.out_features].copy_(old.bias)
        self.fc=new


def explicit_delta(geom,before,after):
    n=len(geom.pairs)
    vals=[(after[2*i+1].double()@after[2*i].double()-before[2*i+1].double()@before[2*i].double()).flatten() for i in range(n)]
    vals += [(b.double()-a.double()).flatten() for a,b in zip(before[2*n:],after[2*n:])]
    return float(torch.cat(vals).norm())


def make_step(model,rule,mode,corrupt=False):
    learner=SimpleNamespace(_unwrap_network=lambda:model,_lambda_flat=1.,_mechanism_eps=1e-12,_mechanism_grad_stats=False)
    native=MethodType(native_projector(),learner)
    base=torch.optim.SGD(model.parameters(),lr=.04 if model.fc.out_features==20 else .02,momentum=0.,weight_decay=0.)
    if corrupt:
        actual=base.step
        def corrupted(*args,**kwargs):
            result=actual(*args,**kwargs)
            with torch.no_grad():next(model.parameters()).add_(.01)
            return result
        base.step=corrupted
    opt=GAM(model.parameters(),base,model,grad_rho=.05,grad_norm_rho=.25,adaptive=False,
            args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.,use_sgv=False))
    audit_step=component.ComponentStepAudit(learner,opt,rule,mode)
    return learner,native,opt,audit_step


def run_step(parts,x,y,zero=False,nan=False):
    learner,projector,opt,stepper=parts;model=learner._unwrap_network();lo=model.fc.out_features-20
    count=[0]
    hook=model.register_forward_hook(lambda *_:count.__setitem__(0,count[0]+1))
    try:
        clean_loss=F.cross_entropy(model(x)['logits'][:,lo:lo+20],y-lo)
        if zero:clean_loss=clean_loss*0
        grads=torch.autograd.grad(clean_loss,list(model.parameters()))
        cache={id(p):g.detach().clone() for p,g in zip(model.parameters(),grads)}
        opt.set_direction_projector(stepper.projector(projector(cache,{},{}),cache,{},{}))
        def closure():
            opt.zero_grad();out=model(x);loss=F.cross_entropy(out['logits'][:,lo:lo+20],y-lo)
            if zero:loss=loss*0
            loss.backward()
            if nan:next(model.parameters()).grad.fill_(float('nan'))
            return out,loss.detach()
        opt.step(closure)
    finally:hook.remove()
    assert count[0]==5 and stepper.record['closure_count']==4
    assert stepper.record['self_spec_passed'] and stepper.record['reference_counterfactual_state_unchanged']
    return copy.deepcopy(stepper.record)


def rejected(fn,exception,label):
    try:fn()
    except exception:return label
    raise AssertionError('Negative case accepted: '+label)


def verify_native_after_task():
    # Actual source method, with a minimal parent and forbidden estimator.
    tree=ast.parse((SOURCE/'models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='after_task')
    class Parent:
        def after_task(self):self._known_classes=self._total_classes
    module=ast.Module(body=[ast.ClassDef(name='ActualEWC',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=[fun],decorator_list=[])],type_ignores=[])
    env={'Parent':Parent};exec(compile(ast.fix_missing_locations(module),'<actual-ewc-after-task>','exec'),env)
    obj=env['ActualEWC']();obj._ewc_skip_task_fisher=True;obj._is_main_process=False
    obj._last_full_fisher_diff={'should_clear':1};obj._delta_reference={'should_clear':1};obj._known_classes=20;obj._total_classes=40
    def forbidden(*a,**kw):raise AssertionError('Fisher estimator was called')
    obj._compute_delta_fisher=forbidden;obj.after_task()
    assert obj._known_classes==40 and obj._last_full_fisher_diff=={} and obj._delta_reference=={}
    return {'actual_frozen_EWC_after_task':True,'skip_prevents_estimation':True,'reference_cache_cleared':True}


def controller_lifecycle():
    """Synthetic learner lifecycle; optimizer and projector are actual source.

    This checks wrapper interfaces/artifacts, not full ViT/native data loading.
    """
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,task):
            gen=torch.Generator().manual_seed(200+task);self.x=torch.randn(9,7,generator=gen);self.y=torch.arange(9)+20*task
        def __len__(self):return len(self.y)
        def __getitem__(self,i):return i,self.x[i],self.y[i]
    class DM:
        def build_dataloader(self,dataset,**kwargs):return torch.utils.data.DataLoader(dataset,**kwargs)
    class Learner:
        def __init__(self,cfg):
            self.args=cfg;self._network=Toy();self._network.fc=None;self._cur_task=-1;self._known_classes=0;self._total_classes=0
            self._lambda_flat=1.;self._lambda_ewc=0.;self._mechanism_eps=1e-12;self._mechanism_grad_stats=False
            self._ewc_skip_task_fisher=True;self._fisher_past_delta={};self._in_task_weight_merge=False;self._task_endpoint_weight_merge=False
            self._build_direction_projector=MethodType(native_projector(),self)
        def _unwrap_network(self):return self._network
        def _build_optimizer(self,params,stage):
            lr=.04 if stage=='init' else .02
            base=torch.optim.SGD(params,lr=lr,momentum=0.,weight_decay=0.)
            return GAM(params,base,self._network,grad_rho=.05,grad_norm_rho=.25,adaptive=False,
                args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.,use_sgv=False))
        def _step_batch_with_log(self,optimizer,inputs,targets,class_offset):
            values=self._network(inputs);loss=F.cross_entropy(values['logits'][:,class_offset:class_offset+20],targets-class_offset)
            grads=torch.autograd.grad(loss,list(self._network.parameters()))
            cache={id(p):g.detach().clone() for p,g in zip(self._network.parameters(),grads)}
            optimizer.set_direction_projector(self._build_direction_projector(cache,{},{}))
            def closure():
                optimizer.zero_grad();out=self._network(inputs)
                value=F.cross_entropy(out['logits'][:,class_offset:class_offset+20],targets-class_offset);value.backward()
                return out,value.detach()
            optimizer.step(closure)
            return values['logits'].detach(),float(loss.detach()),{'ce_clean':float(loss.detach()),'ewc':0.,'delta_w_penalty':0.}
        def _run_regularized_train(self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            for _ in range(epochs):
                self._network.train()
                for _,x,y in train_loader:self._step_batch_with_log(optimizer,x,y,class_offset)
                scheduler.step()
            self._apply_in_task_weight_merge(self._network);self._apply_endpoint_weight_merge(self._network)
        def _apply_in_task_weight_merge(self,model):return {}
        def _apply_endpoint_weight_merge(self,model):return {}
        def incremental_train(self,dm):
            self._cur_task+=1;self._total_classes=self._known_classes+20
            if self._network.fc is None:self._network.fc=torch.nn.Linear(7,self._total_classes)
            elif self._total_classes!=self._network.fc.out_features:self._network.expand(self._total_classes)
            loader=dm.build_dataloader(Dataset(self._cur_task),batch_size=128,shuffle=True,num_workers=0)
            opt=self._build_optimizer(list(self._network.parameters()),'init' if self._cur_task==0 else 'update')
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.004 if self._cur_task==0 else .002)
            self._run_regularized_train(loader,loader,opt,scheduler,epochs=50,class_offset=self._known_classes)
            directory=Path(self.args['filepath']);directory.mkdir(exist_ok=True)
            torch.save(list(self._network.backbone.w_As),directory/('lora_w_a_%d.pt'%self._cur_task))
            torch.save(list(self._network.backbone.w_Bs),directory/('lora_w_b_%d.pt'%self._cur_task))
            audit.atomic_json(directory/('lora_meta_%d.json'%self._cur_task),self._network.backbone.w_meta)
            torch.save(self._network.fc.state_dict(),directory/('fc_state_%d.pt'%self._cur_task))
            self._network.backbone.task_id+=1
        def eval_task(self):
            # Actual native all-ten-boundary output, including structural future NaNs.
            import numpy as np
            from utils.toolkit import accuracy
            labels=np.repeat(np.arange(self._total_classes),2);predictions=labels.copy()
            predictions[::2]=(predictions[::2]+1)%self._total_classes
            grouped=accuracy(predictions,labels,self._known_classes,20,20,
                task_boundaries=[(20*t,20*(t+1)) for t in range(10)])
            return {'grouped':grouped,'top1':grouped['total']},None
        def after_task(self):self._known_classes=self._total_classes
    with tempfile.TemporaryDirectory(prefix='component_controller_cpu_') as tmp:
        out=Path(tmp)/'output';out.mkdir()
        cli=SimpleNamespace(output_dir=str(out),source=str(SOURCE),max_tasks=2,max_epochs=1,max_batches=1)
        cfg=component.resolve_config({'component_rule':'ug','component_scale':'matched','seed':2,'train_num_workers':0},cli)
        cfg.update(seed=2,filepath=str(Path(tmp)/'native_checkpoint'))
        learner=Learner(cfg);controller=component.Controller(learner,cfg,cli);controller.install()
        try:
            for task in range(2):learner.incremental_train(DM());learner.eval_task();learner.after_task()
        finally:controller.handle.close()
        assert len(controller.matrix)==len(controller.rows)==len(controller.after_task_checks)==2
        assert controller.counters['training_batches']==2 and controller.counters['training_ce_gradient_passes']==10
        assert controller.counters['disabled_merge_hook_calls']==4 and controller.counters['forbidden_work_attempts']==0
        mapping=json.loads((out/'task_checkpoints.json').read_text())
        assert set(mapping['all_tasks'])=={'0','1'} and set(mapping['representative_tasks'])=={'0'}
        for task in range(2):
            cp=Path(mapping['all_tasks'][str(task)]);manifest=json.loads((cp/'manifest.json').read_text())
            restored=Toy(20*(task+1))
            for key,modules in (('a',restored.backbone.w_As),('b',restored.backbone.w_Bs)):
                saved=torch.load(cp/('lora_w_%s_%d.pt'%(key,task)),map_location='cpu',weights_only=False)
                with torch.no_grad():
                    for layer,value in zip(modules,saved):layer.weight.copy_(value.weight)
            restored.fc.load_state_dict(torch.load(cp/('fc_state_%d.pt'%task),map_location='cpu',weights_only=False))
            assert component.base.parameter_hash(restored)==manifest['model_sha256']
        return {'tasks':2,'epochs':2,'steps':2,'five_CE_each':True,'full_seen_FC':[20,40],
                'disabled_merge_hooks_preserve_parameters':True,'copied_native_format_checkpoint_roundtrip':True,
                'native_ViT_loading_not_exercised':True,'matrix':controller.matrix}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(728)
    initial=Toy().state_dict();x=torch.randn(9,7);y=torch.arange(9)%20
    records={};us_states={};dense_errors=[]
    for rule in component.LEGACY_RULES:
        for mode in ('natural','matched'):
            model=Toy();model.load_state_dict(initial);all_rows=[]
            for task in (0,1):
                if task:
                    rng=audit.rng_get();torch.manual_seed(999);model.expand(40);audit.rng_set(rng)
                parts=make_step(model,rule,mode)
                for _ in range(2):
                    row=run_step(parts,x,y+20*task);all_rows.append(row)
                    if mode=='matched':assert row['matched'] and row['actual_match_relative_error']<=1e-3
                    if rule=='us':assert row['step_scale']==row['lr_nominal'] and row['us_nominal_reference_after_tensors_exact'] and row['actual_match_relative_error']==0
                    # Independent FP64 dense BA comparison of an actual FP32
                    # shadow step at the present state, including all seen FC.
                    geom=parts[-1].geometry;before=geom.capture()
                    after=[v.clone().add_(p.grad,alpha=-.017) for p,v in zip(geom.params,before)]
                    fast=geom.actual_delta(before,after)['joint_norm'];dense=explicit_delta(geom,before,after)
                    assert math.isclose(fast,dense,rel_tol=1e-10,abs_tol=1e-12)
                    dense_errors.append(abs(fast-dense))
                    assert geom.fc_parameters[0].shape[0]==20*(task+1)
            records[rule+'_'+mode]={'steps':len(all_rows),'head_sizes':[20,40],'all_self_spec':True,
                'max_actual_match_error':max(r['actual_match_relative_error'] for r in all_rows) if mode=='matched' else None,
                'reference_is_current_candidate_projected_g1':all('same_candidate_state' in r['reference_definition'] for r in all_rows)}
            if rule=='us':us_states[mode]=copy.deepcopy(model.state_dict())
    assert all(torch.equal(us_states['natural'][k],us_states['matched'][k]) for k in us_states['natural'])
    negatives=[]
    negatives.append(rejected(lambda:component.selected_direction({},'bad'),ValueError,'invalid_rule'))
    model=Toy();parts=make_step(model,'ug','matched')
    negatives.append(rejected(lambda:run_step(parts,x,y,zero=True),ValueError,'zero_reference_target'))
    model=Toy();parts=make_step(model,'uo','matched')
    negatives.append(rejected(lambda:run_step(parts,x,y,nan=True),FloatingPointError,'nonfinite_probe_gradient'))
    model=Toy();parts=make_step(model,'ufull','matched',corrupt=True)
    negatives.append(rejected(lambda:run_step(parts,x,y),AssertionError,'corrupt_actual_SGD'))
    learner=SimpleNamespace(_fisher_past_delta={'bad':torch.ones(1)})
    negatives.append(rejected(lambda:component.tensor_state_empty(learner),AssertionError,'nonempty_fisher_state'))
    for kind in ('value','reference'):
        model=Toy();counters={'forbidden_work_attempts':0,'disabled_merge_hook_calls':0}
        def bad_merge(m):
            if kind=='value':
                with torch.no_grad():m.fc.weight.add_(1)
            else:m.fc.weight=torch.nn.Parameter(m.fc.weight.detach().clone())
        learner=SimpleNamespace(_in_task_weight_merge=False,_apply_in_task_weight_merge=bad_merge,_compute_delta_fisher=lambda:None)
        component.install_forbidden_work_guards(learner,counters)
        negatives.append(rejected(lambda:learner._apply_in_task_weight_merge(model),AssertionError,'disabled_merge_mutated_'+kind))
        negatives.append(rejected(lambda:learner._compute_delta_fisher(),AssertionError,'forbidden_fisher_'+kind))
    schedule=[]
    for rule in component.LEGACY_RULES:
        for mode in ('natural','matched'):
            cfg=component.resolve_config({'component_rule':rule,'component_scale':mode,'seed':[2]},SimpleNamespace(output_dir='/tmp/component_config_only',max_tasks=2,max_epochs=1,max_batches=1))
            assert cfg['epochs']==cfg['init_epoch']==50 and cfg['max_train_tasks']==2 and cfg['init_lr']==.04 and cfg['lrate']==.02 and cfg['cosine_min_lr_ratio']==.1
            schedule.append(rule+'_'+mode)
    report={'status':'passed','device':'cpu','native_frozen_GAM':True,'native_frozen_projector':True,
        'rules':records,'native_after_task':verify_native_after_task(),'negative_cases_rejected':negatives,
        'synthetic_controller_lifecycle':controller_lifecycle(),
        'us_natural_matched_two_task_parameters_bitwise_equal':True,'dense_effective_norm_max_abs_error':max(dense_errors),
        'schedule_smoke_cases':schedule,'trainer_sha256':audit.digest(Path(component.__file__)),
        'helper_sha256':audit.digest(Path(component.__file__).with_name('effective_step.py'))}
    print(json.dumps(report,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
