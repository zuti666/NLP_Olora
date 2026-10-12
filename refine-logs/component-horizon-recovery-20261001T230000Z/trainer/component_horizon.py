#!/usr/bin/env python3
"""Native ten-task GAM component screen, natural or same-state g1 step matched.

Training/task expansion/checkpointing/evaluation remains in the frozen source.
Only the post-decomposition direction and, for matched runs, SGD scalar change.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import MethodType,SimpleNamespace

sys.dont_write_bytecode=True
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
from effective_step import EffectiveStep,solve_positive_scale

RULES=('u0','us','ug','uo','ufull')
MATCH_RTOL=1e-3


def selected_direction(state,rule):
    """Exact requested parentheses: UG = g0 + (g3 - g2)."""
    if rule=='u0':return state['g_0'].detach().clone()
    if rule=='us':return state['g_1'].detach().clone()
    if rule=='ug':return state['g_0']+(state['g_3']-state['g_2'])
    if rule=='uo':return state['g_2'].detach().clone()
    if rule=='ufull':return state['g_3'].detach().clone()
    raise ValueError('Unknown component rule '+str(rule))


def seen_accuracy_row(cnn,task,seen_classes,history_rows):
    """Read only canonical seen-task groups from native all-ten-boundary output.

    Native accuracy emits NaN placeholders for future tasks with no examples.
    Those canonical future keys are not observations in the current matrix row.
    Missing/invalid seen groups and unexpected numeric ranges remain errors.
    """
    if type(task) is not int or not 0<=task<10 or history_rows!=task or seen_classes!=20*(task+1):
        raise AssertionError('Native evaluation task/history/seen-class mismatch')
    grouped=cnn['grouped']
    keys=['%02d-%02d'%(20*t,20*t+19) for t in range(10)]
    numeric={key for key in grouped if re.fullmatch(r'\d+-\d+',key)}
    if numeric-set(keys):raise AssertionError('Unexpected native evaluation class range')
    expected=keys[:task+1]
    if any(key not in grouped for key in expected):raise AssertionError('Missing seen native evaluation class range')
    if any(key in grouped and not math.isnan(float(grouped[key])) for key in keys[task+1:]):
        raise AssertionError('Future native evaluation range contains observed/non-NaN accuracy')
    row=[float(grouped[key]) for key in expected]
    values=row+[float(cnn['top1'])]
    if any(not math.isfinite(value) or not 0.<=value<=100. for value in values):
        raise AssertionError('Nonfinite or invalid seen native evaluation accuracy')
    return row

def resolve_config(raw,cli):
    rule=str(raw.get('component_rule','')).lower()
    mode=str(raw.get('component_scale','')).lower()
    if rule not in RULES or mode not in ('natural','matched'):
        raise ValueError('component_rule=u0|us|ug|uo|ufull; component_scale=natural|matched required')
    seed=raw.get('seed',2)
    if isinstance(seed,list):
        if len(seed)!=1:raise ValueError('One independent seed per invocation')
        seed=seed[0]
    if int(seed)!=seed or not 0<=int(seed)<=4:raise ValueError('Expected integer seed 0..4')
    if not 0<=cli.max_tasks<=10 or not 0<=cli.max_epochs<=50 or cli.max_batches<0:
        raise ValueError('Invalid smoke budget')
    radius=float(raw.get('gam_grad_norm_rho',.25))
    if radius not in (.25,1.):raise ValueError('Only main r=.25 or robustness r=1 supported')
    surrogate=copy.deepcopy(raw);surrogate.update(seed=2,stage1_direction='s')
    cfg=base.resolve_config(surrogate,SimpleNamespace(max_epochs=0,output_dir=cli.output_dir))
    cfg.update(component_rule=rule,component_scale=mode,seed=[int(seed)],device=[0],
        init_epoch=50,epochs=50,max_train_tasks=cli.max_tasks or 10,init_lr=.04,lrate=.02,
        cosine_min_lr_ratio=.1,gam_grad_norm_rho=radius,gam_grad_beta_1=1.,gam_grad_beta_2=1.,
        gam_grad_beta_3=0.,gam_grad_gamma=0.,use_sgv=False,gam_use_sgv=False,sgv_lambda=0.,
        gam_sgv_lambda=0.,outputs_root=str(Path(cli.output_dir).resolve()/'native'),
        save_final_model=False,epoch_dynamics_eval=False,train_dynamics_eval=False)
    cfg.setdefault('prefix','component_'+rule+'_'+mode+'_s'+str(seed))
    # All auxiliary evaluation is isolated in separate, explicitly scheduled jobs.
    for key in list(cfg):
        if key.endswith('_eval') or key.endswith('_probe'):
            if isinstance(cfg[key],bool):cfg[key]=False
    for key in ('cosine_eta_min','min_lr','min_lr_ratio','step_reference_dir','pretrained_model_path',
                'resume_from_metrics_json','eval_only_after_resume','optimizer_type_by_task'):
        cfg.pop(key,None)
    return cfg


class ComponentStepAudit:
    """Instrument the real GAM instance; no extra CE forward/backward is added."""
    def __init__(self,learner,optimizer,rule,mode):
        if rule not in RULES or mode not in ('natural','matched'):raise ValueError('Invalid rule/mode')
        self.learner,self.optimizer,self.rule,self.mode=learner,optimizer,rule,mode
        self.model=learner._unwrap_network()
        self.geometry=EffectiveStep.from_model(self.model)
        self.params=[p for g in optimizer.param_groups for p in g['params']]
        if {id(p) for p in self.params}!={id(p) for p in self.geometry.params}:raise AssertionError('Optimizer scope mismatch')
        if optimizer.param_groups is not optimizer.base_optimizer.param_groups:raise AssertionError('Unshared native SGD groups')
        self.record=None
        self.install_optimizer()

    def projector(self,native,clean,fisher,delta):
        if fisher or delta or float(self.learner._lambda_flat)!=1.:
            raise AssertionError('Component screen requires empty Fisher/penalty and lambda_flat=1')
        def wrapped(groups):
            params=[p for group in groups for p in group['params']]
            if any(p.grad is None or id(p) not in clean for p in params):raise AssertionError('Missing clean/component gradient')
            self.projected_g1={id(p):clean[id(p)]+1.*(self.optimizer.state[p]['g_1']-clean[id(p)]) for p in params}
            expected=[clean[id(p)]+1.*(p.grad.detach()-clean[id(p)]) for p in params]
            result=native(groups)
            self.projector_exact=all(torch.equal(p.grad,e) for p,e in zip(params,expected))
            if not self.projector_exact:raise AssertionError('Native clean projector own-spec mismatch')
            return result
        return wrapped

    def install_optimizer(self):
        opt=self.optimizer
        native_decompose=opt.gradient_decompose
        def decompose(_opt,*args,**kwargs):
            result=native_decompose(*args,**kwargs)
            # Fixed native beta=(1,0), gamma=0 for every candidate before replacement.
            self.native_decomposition_exact=all(torch.equal(p.grad,1.*opt.state[p]['g_1']+0.*opt.state[p]['g_3']) for p in self.params)
            if not self.native_decomposition_exact:raise AssertionError('Native beta=(1,0), gamma0 decomposition mismatch')
            with torch.no_grad():
                expected=[selected_direction(opt.state[p],self.rule) for p in self.params]
                for p,value in zip(self.params,expected):p.grad.copy_(value)
            self.component_spec_exact=all(torch.equal(p.grad,e) for p,e in zip(self.params,expected))
            if not self.component_spec_exact:raise AssertionError('Component selection own-spec mismatch')
            base.require_finite_gradients(self.params,'Component gradient')
            return result
        opt.gradient_decompose=MethodType(decompose,opt)
        native_base_step=opt.base_optimizer.step
        def base_step(_base,*args,**kwargs):
            if self.record is not None:raise AssertionError('Multiple SGD steps per GAM batch')
            if not self.projector_exact or not self.component_spec_exact:raise AssertionError('Missing selected-gradient audits')
            base.require_finite_gradients(self.params,'Actual SGD gradient')
            lrs=[float(g['lr']) for g in opt.param_groups]
            if not lrs or len(set(lrs))!=1 or not math.isfinite(lrs[0]) or lrs[0]<=0:raise AssertionError('One positive nominal SGD LR required')
            for g in opt.param_groups:
                if any(float(g.get(k,0))!=0 for k in ('momentum','dampening','weight_decay')) or g.get('nesterov',False) or g.get('maximize',False):
                    raise AssertionError('Component matching requires ordinary zero-momentum/decay SGD')
            geom=self.geometry;before=geom.capture();eta=lrs[0]
            roundtrip=geom.actual_delta(self.origin,before)
            old_grads=[p.grad.detach().clone() for p in geom.params]
            buffers=[(b,b.detach().clone()) for b in self.model.buffers()]
            rng=audit.rng_get()
            # Actual same-dtype counterfactual. No copy_ into the live model and
            # no optimizer call: all parameter/buffer/RNG/gradient state stays put.
            reference_after=[value.clone().add_(self.projected_g1[id(p)],alpha=-eta)
                             for p,value in zip(geom.params,before)]
            reference=geom.actual_delta(before,reference_after)
            raw_after=[value.clone().add_(opt.state[p]['g_1'],alpha=-eta) for p,value in zip(geom.params,before)]
            raw_reference=geom.actual_delta(before,raw_after)
            raw_projected_diff=math.sqrt(sum(float((self.projected_g1[id(p)].double()-opt.state[p]['g_1'].double()).square().sum()) for p in geom.params))
            reference_no_side_effects=(all(torch.equal(p,v) for p,v in zip(geom.params,before))
                and all(torch.equal(p.grad,v) for p,v in zip(geom.params,old_grads))
                and all(torch.equal(b,v) for b,v in buffers) and audit.rng_equal(rng,audit.rng_get()))
            if not reference_no_side_effects:raise AssertionError('Counterfactual reference changed live state')
            target=reference['joint_norm']
            if not math.isfinite(target) or target<=0:raise ValueError('Invalid same-state g1 reference target')
            poly=geom.polynomial();scale=eta;root=None
            if self.mode=='matched' and self.rule!='us':
                root=solve_positive_scale(poly,target,initial_scale=eta,max_scale=10.)
                scale=root['scale']
            elif self.mode=='matched':
                root={'scale':eta,'method':'US nominal scalar identity control','matched':True}
            predicted=[p.detach().clone().add_(p.grad,alpha=-scale) for p in self.params]
            try:
                for group in opt.param_groups:group['lr']=scale
                result=native_base_step(*args,**kwargs)
            finally:
                for group,lr in zip(opt.param_groups,lrs):group['lr']=lr
            actual=geom.actual_delta(before)
            exact=all(torch.equal(p,e) for p,e in zip(self.params,predicted))
            us_identity=(all(torch.equal(p,v) for p,v in zip(geom.params,reference_after)) if self.rule=='us' else None)
            if self.rule=='us' and not us_identity:raise AssertionError('US update differs from projected-g1 nominal reference tensors')
            error=abs(actual['joint_norm']-target)/target
            matched=error<=MATCH_RTOL if self.mode=='matched' else None
            self.record={'component_rule':self.rule,'component_scale':self.mode,'lr_nominal':eta,
                'step_scale':scale,'scale_over_nominal_lr':scale/eta,
                'reference_definition':'same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD',
                'reference_effective_delta':reference,'reference_target_norm':target,
                'raw_g1_reference_effective_delta':raw_reference,'raw_vs_projected_g1_parameter_l2':raw_projected_diff,
                'us_nominal_reference_after_tensors_exact':us_identity,
                'reference_counterfactual_state_unchanged':reference_no_side_effects,
                'actual_effective_delta':actual,'actual_to_reference_norm_ratio':actual['joint_norm']/target,
                'actual_match_relative_error':error,'matched':matched,'matching_tolerance':MATCH_RTOL,
                'root':root,'effective_step_polynomial':poly.as_dict(),'probe_roundtrip_delta':roundtrip,
                'native_decomposition_exact':self.native_decomposition_exact,'component_spec_exact':self.component_spec_exact,
                'projector_spec_exact':self.projector_exact,'base_sgd_exact':exact,
                'base_sgd_max_abs_error':max(float((p.detach()-e).abs().max()) for p,e in zip(self.params,predicted)),
                'self_spec_passed':bool(exact and self.native_decomposition_exact and self.component_spec_exact and self.projector_exact)}
            if not exact or matched is False or audit.nonfinite_paths(self.record):
                raise AssertionError('Actual SGD/effective norm verification failed: '+repr(self.record))
            return result
        opt.base_optimizer.step=MethodType(base_step,opt.base_optimizer)
        native_step=opt.step
        def step(_opt,closure=None):
            self.record=None;self.origin=self.geometry.capture()
            self.native_decomposition_exact=self.component_spec_exact=self.projector_exact=False
            calls=0
            def wrapped():
                nonlocal calls
                calls+=1;result=closure()
                base.require_finite_gradients(self.params,'GAM probe %d'%(calls-1))
                if not bool(torch.isfinite(result[1])):raise FloatingPointError('Nonfinite probe CE')
                return result
            result=native_step(closure=wrapped)
            if calls!=4 or self.record is None:raise AssertionError('Expected four GAM closures/one SGD')
            self.record['closure_count']=calls
            self.origin=None
            return result
        opt.step=MethodType(step,opt)


class EpochLoader:
    def __init__(self,original,controller):
        self.original,self.c,self.dataset,self.epoch=original,controller,original.dataset,0
    def __len__(self):return min(len(self.original),self.c.cli.max_batches) if self.c.cli.max_batches else len(self.original)
    def __iter__(self):
        self.epoch+=1;c=self.c;c.begin_epoch(self.epoch)
        seed=c.seed*100000+c.task*1000+self.epoch
        loader=c.dm.build_dataloader(self.dataset,batch_size=128,shuffle=True,
            num_workers=c.cfg.get('train_num_workers',8),generator=torch.Generator().manual_seed(seed),
            worker_init_fn=base.worker_init,persistent_workers=False)
        train_rng=audit.rng_get();audit.seed_all(seed);data_rng=audit.rng_get();audit.rng_set(train_rng)
        iterator=None;order=hashlib.sha256()
        for batch in range(len(self)):
            train_rng=audit.rng_get();audit.rng_set(data_rng)
            try:
                if iterator is None:iterator=iter(loader)
                item=next(iterator);data_rng=audit.rng_get()
            finally:audit.rng_set(train_rng)
            ids,x,y=item;order.update(ids.contiguous().numpy().tobytes())
            c.batch=batch;c.batch_identity={'input_sha256':audit.tensor_digest(x),'target_sha256':audit.tensor_digest(y),
                'indices_sha256':audit.tensor_digest(ids),'indices':ids.tolist(),'data_seed':seed}
            yield item
        c.sampler_digest=order.hexdigest()


def tensor_state_empty(learner):
    names=('_fisher_past_delta','_posterior_fisher_endpoint','_posterior_fisher_merge',
           '_fisher','_fisher_matrix','fisher','fisher_matrix')
    for name in names:
        value=getattr(learner,name,None)
        if value is not None and (not isinstance(value,dict) or value):raise AssertionError('Unexpected Fisher state: '+name)
    return True


def trainable_snapshot(model):
    return {name:(id(p),p.detach().clone()) for name,p in model.named_parameters() if p.requires_grad}


def trainable_unchanged(model,snapshot):
    current={name:p for name,p in model.named_parameters() if p.requires_grad}
    return (set(current)==set(snapshot) and all(id(current[name])==old_id and torch.equal(current[name],old)
            for name,(old_id,old) in snapshot.items()))


def install_forbidden_work_guards(learner,counters):
    for name in ('_compute_delta_fisher','_compute_raw_delta_fisher','_compute_fisher_all_seen','_accumulate_fisher',
                 '_capture_in_task_fisher_snapshot','_capture_in_task_weight_snapshot'):
        if hasattr(learner,name):
            def forbidden(_self,*args,_name=name,**kwargs):
                counters['forbidden_work_attempts']+=1
                raise AssertionError('Forbidden Fisher/weight snapshot work: '+_name)
            setattr(learner,name,MethodType(forbidden,learner))
    for name,flag in (('_apply_in_task_weight_merge','_in_task_weight_merge'),('_apply_endpoint_weight_merge','_task_endpoint_weight_merge')):
        if hasattr(learner,name):
            original=getattr(learner,name)
            def disabled_merge(_self,model,*args,_name=name,_flag=flag,_original=original,**kwargs):
                if bool(getattr(_self,_flag,True)):raise AssertionError('Weight merge enabled: '+_name)
                before=trainable_snapshot(model)
                result=_original(model,*args,**kwargs)
                if not trainable_unchanged(model,before):raise AssertionError('Disabled merge hook changed weights or parameter references')
                counters['disabled_merge_hook_calls']+=1
                tensor_state_empty(_self)
                return result
            setattr(learner,name,MethodType(disabled_merge,learner))


class Controller:
    def __init__(self,learner,cfg,cli):
        self.learner,self.cfg,self.cli=learner,cfg,cli
        self.seed=int(cfg['seed']);self.rule=cfg['component_rule'];self.mode=cfg['component_scale']
        self.out=Path(cli.output_dir);self.trace=self.out/'step_trace.jsonl';self.handle=self.trace.open('x')
        self.task=-1;self.epoch=self.batch=0;self.rows=[];self.task_rows=[];self.matrix=[];self.sample_curve=[];self.steps=0
        self.counters={'training_batches':0,'training_samples':0,'training_ce_gradient_passes':0,
            'fisher_estimation_passes':0,'weight_merge_updates':0,'forbidden_work_attempts':0,'disabled_merge_hook_calls':0}
        self.max_match_error=0.;self.match_count=0;self.stepper=None;self.checkpoints=[];self.started=time.time()
        self.after_task_checks=[]
        self.task_identities=[]
    def install(self):
        l=self.learner;install_forbidden_work_guards(l,self.counters)
        original_projector=l._build_direction_projector
        def projector(_self,clean,fisher,delta=None):
            return self.stepper.projector(original_projector(clean,fisher,delta),clean,fisher,delta)
        l._build_direction_projector=MethodType(projector,l)
        original_build=l._build_optimizer
        def build(_self,params,stage):
            self.optimizer=original_build(params,stage)
            self.stepper=ComponentStepAudit(l,self.optimizer,self.rule,self.mode)
            model=l._unwrap_network();old_count=20*self.task
            fc=model.fc.state_dict()
            old_exact=all(torch.equal(fc[k][:old_count],value.to(fc[k])) for k,value in self.old_fc.items())
            if not old_exact:raise AssertionError('Native classifier expansion changed old rows')
            identity={'task':self.task,'seed':self.seed,'component_rule':self.rule,'component_scale':self.mode,
                'class_slice':[old_count,old_count+20],'fc_shape':list(model.fc.weight.shape),'old_fc_rows_inherited_exact':old_exact,
                'new_fc_rows_sha256':{k:audit.tensor_digest(value[old_count:]) for k,value in fc.items()},
                'task_initial_parameter_sha256':base.parameter_hash(model),'frozen_origin_sha256':base.parameter_hash(model,True),
                'data_seed_rule':'seed*100000+task*1000+epoch (epoch 1-based)',
                'native_global_rng_protocol_preserved':False}
            self.task_identities.append(identity);audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            return self.optimizer
        l._build_optimizer=MethodType(build,l)
        original_batch=l._step_batch_with_log
        def batch(_self,optimizer,inputs,targets,class_offset):
            if class_offset!=20*self.task or bool(((targets<class_offset)|(targets>=class_offset+20)).any()):
                raise AssertionError('Current-task CE class slice mismatch')
            calls=[0]
            def count(*_args):calls[0]+=1
            hook=l._unwrap_network().register_forward_hook(count)
            try:result=original_batch(optimizer,inputs,targets,class_offset)
            finally:hook.remove()
            out,loss,log=result
            if calls[0]!=5 or float(log.get('ewc',0))!=0 or float(log.get('delta_w_penalty',0))!=0:
                raise AssertionError('Expected clean+four CE only; no EWC/penalty')
            row={**self.stepper.record,'step_index':self.steps,'task':self.task,'epoch':self.epoch,'batch':self.batch,
                'task_step_index':self.steps-self.task_start_step,'seed':self.seed,'identity':copy.deepcopy(self.batch_identity),
                'class_slice':[class_offset,class_offset+20],'seen_classes':l._total_classes,
                'fc_shape':list(l._unwrap_network().fc.weight.shape),'training_ce_gradient_passes':5,
                'training_forward_count':calls[0],'batch_samples':int(targets.numel()),'train_ce':float(log['ce_clean'])}
            if audit.nonfinite_paths(row):raise FloatingPointError('Nonfinite step trace')
            self.handle.write(json.dumps(row,allow_nan=False)+'\n');self.handle.flush()
            self.steps+=1;self.counters['training_batches']+=1;self.counters['training_samples']+=targets.numel();self.counters['training_ce_gradient_passes']+=5
            self.samples+=targets.numel();self.loss_sum+=float(log['ce_clean'])*targets.numel()
            self.correct+=int(out[:,class_offset:class_offset+20].argmax(-1).eq(targets-class_offset).sum())
            self.epoch_batches+=1
            if self.mode=='matched':self.match_count+=1;self.max_match_error=max(self.max_match_error,row['actual_match_relative_error'])
            return result
        l._step_batch_with_log=MethodType(batch,l)
        original_run=l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            if self.task!=l._cur_task or class_offset!=20*self.task:raise AssertionError('Native task/offset mismatch')
            if scheduler.T_max!=50 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1,rel_tol=1e-12):raise AssertionError('Native scheduler budget/floor mismatch')
            self.scheduler=scheduler;self.task_dataset_size=len(train_loader.dataset);self.task_batch_count=min(len(train_loader),self.cli.max_batches) if self.cli.max_batches else len(train_loader)
            if self.stepper.geometry.fc_parameters[0].shape[0]!=(self.task+1)*20:raise AssertionError('Full seen FC omitted')
            self.task_start_step=self.steps;before_count=len(self.rows)
            original_schedule=scheduler.step
            def schedule(*a,**kw):
                result=original_schedule(*a,**kw);self.finish_epoch();return result
            scheduler.step=schedule
            result=original_run(EpochLoader(train_loader,self),test_loader,optimizer,scheduler,
                epochs=self.cli.max_epochs or epochs,class_offset=class_offset)
            if len(self.rows)-before_count!=(self.cli.max_epochs or 50):raise AssertionError('Missing task epoch callbacks')
            return result
        l._run_regularized_train=MethodType(run,l)
        original_increment=l.incremental_train
        def incremental(_self,dm):
            self.dm=dm;self.task=l._cur_task+1
            fc=getattr(l._unwrap_network(),'fc',None)
            self.old_fc={} if fc is None else {k:v.detach().clone() for k,v in fc.state_dict().items()}
            tensor_state_empty(l)
            result=original_increment(dm)
            tensor_state_empty(l);self.save_task_checkpoint()
            return result
        l.incremental_train=MethodType(incremental,l)
        original_eval=l.eval_task
        def evaluate(_self,*args,**kwargs):
            with base.observational(l._unwrap_network()):result=original_eval(*args,**kwargs)
            cnn,_=result
            row=seen_accuracy_row(cnn,self.task,l._total_classes,len(self.matrix))
            self.matrix.append(row);self.sample_curve.append(float(cnn['top1']))
            self.write_matrix();return result
        l.eval_task=MethodType(evaluate,l)
        # Native periodic accuracy observations cannot perturb subsequent training RNG.
        for name in ('_compute_accuracy','_compute_task_local_accuracy'):
            if hasattr(l,name):
                original=getattr(l,name)
                def observed(_self,*args,_original=original,**kwargs):
                    with base.observational(l._unwrap_network()):return _original(*args,**kwargs)
                setattr(l,name,MethodType(observed,l))
        original_after=l.after_task
        def after(_self):
            if not l._ewc_skip_task_fisher or float(l._lambda_ewc)!=0.:raise AssertionError('Fisher/EWC enabled at task end')
            params=trainable_snapshot(l._unwrap_network())
            result=original_after()
            exact=trainable_unchanged(l._unwrap_network(),params);tensor_state_empty(l)
            if not exact:raise AssertionError('after_task changed parameters in pure component run')
            self.after_task_checks.append({'task':self.task,'parameters_exact':True,'parameter_references_exact':True,'fisher_empty':True,'known_classes':l._known_classes})
            audit.atomic_json(self.out/'after_task_checks.json',self.after_task_checks)
            return result
        l.after_task=MethodType(after,l)

    def begin_epoch(self,epoch):
        self.epoch=epoch;self.samples=self.correct=self.loss_sum=self.epoch_batches=0
        self.sampler_digest=None;self.lr_used=[float(g['lr']) for g in self.optimizer.param_groups]
        self.epoch_start=time.time()
        if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    def finish_epoch(self):
        if torch.cuda.is_available():torch.cuda.synchronize()
        seconds=time.time()-self.epoch_start
        row={'task':self.task,'epoch':self.epoch,'samples':self.samples,'batches':self.epoch_batches,
             'train_ce':self.loss_sum/self.samples,'train_accuracy_local':100.*self.correct/self.samples,
             'lr_used':self.lr_used,'lr_after_scheduler':[float(g['lr']) for g in self.optimizer.param_groups],
             'sampler_order_sha256':self.sampler_digest,'seconds':seconds,'samples_per_second':self.samples/seconds,
             'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
        if self.epoch_batches!=self.task_batch_count or audit.nonfinite_paths(row):raise AssertionError('Incomplete/nonfinite epoch')
        self.rows.append(row);audit.atomic_json(self.out/'metrics.json',{'epochs':self.rows,'counters':self.counters})
        self.handle.flush();os.fsync(self.handle.fileno())
        print(json.dumps({'event':'component_epoch_complete',**row}),flush=True)
    def save_task_checkpoint(self):
        l=self.learner;model=l._unwrap_network();task=self.task
        directory=self.out/'checkpoints'/('task_%03d'%task);directory.mkdir(parents=True,exist_ok=False)
        native=Path(self.cfg['filepath'])
        names=['lora_w_a_%d.pt'%task,'lora_w_b_%d.pt'%task,'lora_meta_%d.json'%task,'fc_state_%d.pt'%task]
        for name in names:
            if not (native/name).is_file():raise FileNotFoundError('Missing native checkpoint '+str(native/name))
            shutil.copyfile(native/name,directory/name)
        with base.observational(model):
            state={'task':task,'epoch':self.cli.max_epochs or 50,'config':self.cfg,'optimizer':base.cpu_tree(self.optimizer.state_dict()),
                'base_optimizer':base.cpu_tree(self.optimizer.base_optimizer.state_dict()),'scheduler':base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),'rng':audit.rng_get()}
            torch.save(state,directory/'training_state.pt')
            manifest={'task':task,'epoch':self.cli.max_epochs or 50,'seed':self.seed,'component_rule':self.rule,'component_scale':self.mode,
                'class_range':[0,l._total_classes],'current_task_class_range':[20*task,20*(task+1)],'model_sha256':base.parameter_hash(model),
                'source_gam_sha256':audit.digest(Path(self.cli.source)/'optimer_PerturabtionType/gam.py'),
                'fc_shape':list(model.fc.weight.shape),'backbone_task_id':model.backbone.task_id,'native_checkpoint_dir':str(native),
                'files_sha256':{name:audit.digest(directory/name) for name in names},
                'loader':'learner.load_task_checkpoint(checkpoint_dir, task, data_manager); cumulative current single adapter'}
            audit.atomic_json(directory/'manifest.json',manifest)
        self.checkpoints.append(str(directory))
        audit.atomic_json(self.out/'task_checkpoints.json',{'loader':'load_task_checkpoint(directory, task, data_manager)',
            'all_tasks':{str(i):p for i,p in enumerate(self.checkpoints)},
            'representative_tasks':{str(i):self.checkpoints[i] for i in (0,4,9) if i<len(self.checkpoints)}})
        self.task_rows.append({'task':task,'train_dataset_samples':self.task_dataset_size,'batches_per_epoch':self.task_batch_count,
            'epochs':self.cli.max_epochs or 50,'steps':self.steps-self.task_start_step,'checkpoint':str(directory)})
        audit.atomic_json(self.out/'tasks.json',self.task_rows)
    def write_matrix(self):
        n=len(self.matrix);faa=sum(self.matrix[-1])/n;aaa=sum(sum(row)/len(row) for row in self.matrix)/n
        acquisition=sum(self.matrix[t][t] for t in range(n))/n
        bwt=sum(self.matrix[-1][j]-self.matrix[j][j] for j in range(n-1))/(n-1) if n>1 else None
        forgetting=sum(max(self.matrix[t][j] for t in range(j,n))-self.matrix[-1][j] for j in range(n-1))/(n-1) if n>1 else None
        result={'matrix':self.matrix,'matrix_scope':'native CNN argmax over all seen classes; rows=time, columns=task',
            'AAA_task_macro':aaa,'FAA_task_macro':faa,'LA':acquisition,'BWT':bwt,'forgetting':forgetting,
            'sample_accuracy_curve':self.sample_curve,'FAA_sample_weighted':self.sample_curve[-1],'AAA_sample_weighted':sum(self.sample_curve)/n}
        if audit.nonfinite_paths(result):raise FloatingPointError('Nonfinite observed matrix metrics')
        audit.atomic_json(self.out/'task_matrix.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','config','output-dir'):parser.add_argument('--'+key,required=True)
    for key in ('max-tasks','max-epochs','max-batches'):parser.add_argument('--'+key,type=int,default=0)
    cli=parser.parse_args()
    cli.source,cli.config,cli.output_dir=[str(Path(x).resolve()) for x in (cli.source,cli.config,cli.output_dir)]
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli)
    os.environ.setdefault('HF_HUB_OFFLINE','1');sys.path.insert(0,cli.source);os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import json_safe
    from src import trainer
    original_factory=factory.get_model;holder={}
    def create(name,args):
        if holder:raise AssertionError('Exactly one independent seed/model per run')
        learner=original_factory(name,args)
        controller=Controller(learner,args,cli);holder['controller']=controller;controller.install()
        audit.atomic_json(out/'effective_config.json',json_safe(args))
        return learner
    factory.get_model=create
    summary={'status':'running','schema_version':1,'seed':cfg['seed'][0],'component_rule':cfg['component_rule'],
        'component_scale':cfg['component_scale'],'r':cfg['gam_grad_norm_rho'],'rho':.05,
        'smoke':bool(cli.max_tasks or cli.max_epochs or cli.max_batches),'requested_tasks':cli.max_tasks or 10,
        'epochs_per_task':cli.max_epochs or 50,'scheduler_T_max':50,'completed_tasks':0,
        'reference_definition':'same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD',
        'trainer_sha256':audit.digest(__file__),'effective_step_sha256':audit.digest(Path(__file__).with_name('effective_step.py')),
        'source_gam_sha256':audit.digest(Path(cli.source)/'optimer_PerturabtionType/gam.py'),
        'input_config_sha256':audit.digest(cli.config),'source':cli.source,'input_config':cli.config}
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        trainer.train(cfg)
        c=holder['controller'];n=cli.max_tasks or 10
        if len(c.matrix)!=n or len(c.task_rows)!=n or len(c.after_task_checks)!=n:raise AssertionError('Incomplete native task lifecycle')
        if len(c.rows)!=n*(cli.max_epochs or 50) or c.steps!=sum(t['steps'] for t in c.task_rows):raise AssertionError('Incomplete epoch/step accounting')
        if c.counters['forbidden_work_attempts'] or c.counters['fisher_estimation_passes'] or c.counters['weight_merge_updates']:raise AssertionError('Forbidden work occurred')
        if c.mode=='matched' and c.match_count!=c.steps:raise AssertionError('Unmatched training step')
        summary.update(status='completed',completed_tasks=n,task_count=n,completed_epochs=len(c.rows),completed_steps=c.steps,
            counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,after_task_checks=c.after_task_checks,
            initial_identity=c.task_identities[0],effective_config_sha256=audit.digest(out/'effective_config.json'),
            checks={'finite':True,'all_steps_self_spec':True,'all_matched_steps':c.match_count==c.steps if c.mode=='matched' else None,
                    'fisher_empty_all_tasks':True,'actual_weight_merges_zero':True,'complete_task_lifecycle':True},
            task_identities_sha256=audit.digest(out/'task_identities.json'),task_checkpoints_sha256=audit.digest(out/'task_checkpoints.json'),
            max_match_relative_error=c.max_match_error if c.mode=='matched' else None,matched_steps=c.match_count,
            step_trace_sha256=audit.digest(c.trace),task_matrix_sha256=audit.digest(out/'task_matrix.json'),
            final_metrics=json.loads((out/'task_matrix.json').read_text()),finite_check={'passed':True},elapsed_seconds=time.time()-c.started)
    except Exception as exc:
        summary.update(status='failed',error=repr(exc),completed_tasks=len(holder['controller'].matrix) if holder else 0)
        if holder and holder['controller'].stepper and holder['controller'].stepper.record:
            audit.atomic_json(out/'failed_step_audit.json',holder['controller'].stepper.record)
        raise
    finally:
        factory.get_model=original_factory
        if holder:holder['controller'].handle.close()
        audit.atomic_json(out/'final_summary.json',summary)


if __name__=='__main__':main()
