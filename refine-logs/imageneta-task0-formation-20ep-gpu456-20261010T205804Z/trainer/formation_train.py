#!/usr/bin/env python3
"""Six Task0 formation schedules, each followed by identical 20-epoch SGD Task1.

This is an independent20epoch protocol; historical50epoch endpoints are never
loaded, pooled or reused. Complete native lifecycle and probe kernels are
privately copied and pinned from the audited preceding campaign.
"""
import argparse
import copy
import json
import hashlib
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import MethodType, SimpleNamespace
sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
import response_train as inherited
from response_train import observational, HISTORY_SOURCE, INDEPENDENT_MODE
from layout_lifecycle import TaskLayout, seen_accuracy_row
from effective_step import EffectiveStep
from state_transaction import State
from probe_study_data import build_data_manager, build_dev_loader, split_fingerprint
from formation_step import FormationStep, CONDITIONS, READS, FORMULAS, active_method, cosine_lr
PROTOCOL='task0_formation_20ep_task1_sgd_v1'

def rng_hash(state):
    h=hashlib.sha256();h.update(repr(state[0]).encode());h.update(str(state[1][0]).encode())
    h.update(state[1][1].tobytes());h.update(repr(state[1][2:]).encode());h.update(state[2].cpu().numpy().tobytes())
    for value in state[3]:h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def radius_metadata(cfg):
    task=cfg.get('_formation_task',0);epoch=cfg.get('_formation_epoch',1)
    method=active_method(cfg['formation_condition_id'],task,epoch)
    return dict(formation_condition_id=cfg['formation_condition_id'],active_method=method,
        gradient_formula=FORMULAS[method],normal_gradient_calls=READS[method],
        gam_radii_active=method in ('sam','gam_residual'),history_source=None,history_weight=0.,
        formation_switch_epoch=11,task1_method='sgd',native_probe_skipping=True,
        probe_radius_coordinates='global trainable LoRA plus full current classifier parameter L2',
        inner_rho=.05 if method in ('sam','gam_residual') else None,
        outer_R=1. if method=='gam_residual' else .25 if method=='sf_g2' else None,
        inner_probe_radius_mode='sgd_step' if method=='sf_g2' else 'fixed_l2' if method in ('sam','gam_residual') else None,
        outer_probe_radius_mode='fixed_l2' if method in ('sf_g2','gam_residual') else None,
        dev_split_sha256=audit.digest(cfg['study_dev_split_path']))

def resolve_config(raw,cli):
    condition=raw.get('formation_condition_id');rawseed=raw.get('seed')
    seed=rawseed[0] if isinstance(rawseed,list) and len(rawseed)==1 else rawseed
    if condition not in CONDITIONS or seed not in (0,1,2) or isinstance(seed,bool):
        raise ValueError('Registered six conditions and seed0/1/2 required')
    if raw.get('init_epoch')!=20 or raw.get('epochs')!=20:
        raise ValueError('Independent exactly20epoch protocol required for both tasks')
    if raw.get('formation_task1_method')!='sgd' or raw.get('formation_switch_epoch')!=11:
        raise ValueError('Task1 must use SGD; Task0 switch boundary is epoch11')
    expected=[active_method(condition,0,e) for e in (1,11)]
    if raw.get('formation_task0_schedule')!=expected:raise ValueError('Task0 schedule disagrees with condition')
    if raw.get('study_outer_radius')!=.25 or raw.get('gam_grad_rho')!=.05 or raw.get('gam_grad_norm_rho')!=1.:
        raise ValueError('Frozen S_F outer.25 and GAM rho.05/r1 definitions required')
    for key,value in dict(lambda_flat=1.,gam_grad_gamma=0.,use_sgv=False,gam_use_sgv=False,sgv_lambda=0.,
                           cosine_min_lr_ratio=.1,optimizer='sgd',scheduler='cosine',study_inner_mode='sgd_step',
                           study_outer_mode='fixed').items():
        if raw.get(key)!=value:raise ValueError('Frozen formation setting mismatch: '+key)
    if cli.max_batches not in (0,1):raise ValueError('Only full datasets or onebatch EACH20epochs smoke permitted')
    if bool(cli.max_batches)!=bool(raw.get('formation_smoke',False)):
        raise ValueError('Smoke registration and CLI maxbatches disagree')
    split=Path(raw.get('study_dev_split_path','')).resolve()
    if not split.is_file():raise FileNotFoundError('Pinned train-only dev split required')
    clean=copy.deepcopy(raw)
    for key in inherited.PROBE_CONFIG_KEYS+('response_probe_variant','radius_variant'):
        clean.pop(key,None)
    clean.update(seed=2,init_epoch=50,epochs=50,response_method='sgd',response_layout='cl10',
        response_radius_mode='sgd_step',radius_scale=1.,history_weight=0.,history_source=None,
        history_reset='task',post_update_loss=True,strict_deterministic=True)
    surrogate=SimpleNamespace(max_epochs=0,max_batches=0,max_tasks=0,output_dir=cli.output_dir)
    cfg=inherited.resolve_config(clean,surrogate)
    cfg.update(seed=seed,device=[0],init_epoch=20,epochs=20,max_train_tasks=2,
        protocol=PROTOCOL,formation_protocol=PROTOCOL,formation_condition_id=condition,
        study_role='formation',study_condition_id=condition,study_dev_split_path=str(split),
        formation_task0_schedule=expected,formation_task1_method='sgd',formation_switch_epoch=11,
        formation_smoke=bool(cli.max_batches),optimizer_tag_override='formation20_native_'+condition,
        cosine_min_lr_ratio=.1,probe_gradient_count=None,compute_g3_probe=condition=='gam_residual',
        gam_radii_active=condition in ('sam','gam_residual'),actual_optimizer='torch.optim.SGD',
        gam_grad_beta_3=1. if condition=='gam_residual' else 0.,g3_used_in_final=condition=='gam_residual',
        formation_schedule_authoritative=True,active_algorithm_definition={method:FORMULAS[method] for method in ('sgd','sam','gam_residual','sf_g2')},
        configured_gamma_projection_active=False,configured_beta_formula_active=False,
        update_direction='formation_schedule',attention_backend='native')
    return cfg

class Controller(inherited.Controller):
    def __init__(self,learner,cfg,cli):
        super().__init__(learner,cfg,cli)
        self.task0_endpoint=None;self.post_expand_baseline=None;self.retention_curve=[];self.dev_curve=[]
        self.switch_receipts=[];self.epoch10_boundary=None;self.epoch_task0_curve=[]
        self.native_compute_seconds=0.;self.native_commit_seconds=0.;self.passive_eval_seconds=0.;self.training_peak_bytes=0
    def install(self):
        l=self.learner
        native.install_forbidden_work_guards(l,self.counters)
        def build(_self,params,stage):
            model=l._unwrap_network()
            self.geometry=EffectiveStep.from_model(model)
            params=list(params)
            if {id(p) for p in params}!={id(p) for p in self.geometry.params}:
                raise AssertionError('Optimizer must cover every trainable LoRA/FC tensor')
            self.optimizer=torch.optim.SGD(params,lr=.04 if stage=='init' else .02,momentum=0,weight_decay=0)
            self.optimizer.base_optimizer=self.optimizer
            self.cfg['_formation_task']=self.task
            self.stepper=FormationStep(model,params,self.geometry,self.cfg)
            self.previous_identity=None
            old_count,end_count=self.layout.class_range(self.task)
            fc=model.fc.state_dict()
            if not all(torch.equal(fc[k][:old_count],v.to(fc[k])) for k,v in self.old_fc.items()):
                raise AssertionError('Expanded classifier modified existing rows')
            identity=dict(task=self.task,seed=self.seed,update_direction=self.direction,
                          class_slice=[old_count,end_count],fc_shape=list(model.fc.weight.shape),
                          response_layout=self.layout.name,history_source=HISTORY_SOURCE[self.direction],
                          old_fc_rows_inherited_exact=True,history_reset=True,
                          new_fc_rows_sha256={k:audit.tensor_digest(v[old_count:]) for k,v in fc.items()},
                          task_initial_parameter_sha256=base.parameter_hash(model),
                          frozen_origin_sha256=base.parameter_hash(model,True),
                          data_seed_rule='seed*100000+task*1000+epoch',
                          forward_rng_rule='R0 replay; one g0/R1 advance; passive diagnostic restored')
            self.task_identities.append(identity)
            identity.update(radius_metadata(self.cfg))
            audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            return self.optimizer
        l._build_optimizer=MethodType(build,l)

        def batch(_self,optimizer,inputs,targets,class_offset):
            lo,hi=self.layout.validate_batch(self.task,class_offset,targets,l._total_classes,
                                            l._unwrap_network().fc.weight.shape[0])
            model=l._unwrap_network()
            def closure(x,y):
                logits=model(x)['logits']
                loss,task_logits,task_targets=l._task_loss(logits,y,class_offset)
                if task_logits.shape[-1]!=hi-lo or not torch.equal(task_targets,y-lo):
                    raise AssertionError('Native task loss did not use the configured CE partition')
                return loss,logits
            lr=float(optimizer.param_groups[0]['lr'])
            forward_count=[0]
            hook=model.register_forward_hook(lambda *_:forward_count.__setitem__(0,forward_count[0]+1))
            try:
                result=self.stepper.compute(inputs,targets,closure,lr)
                update=self.stepper.apply_update(lr,optimizer=optimizer)
                # One same-current-batch/R0 diagnostic at the actual committed point.
                committed=State(model)
                try:
                    audit.rng_set(result['context']['rng0'])
                    with torch.no_grad():
                        after_loss,_=closure(inputs,targets)
                    loss_after=float(after_loss.item())
                    if not math.isfinite(loss_after):
                        raise FloatingPointError('Nonfinite actual post-update diagnostic loss')
                finally:
                    committed.restore()
                    diagnostic_restore=committed.check()
            finally:
                hook.remove()
            row=dict(result['record'],**update)
            passes=row['ordinary_gradient_calls']
            expected_passes=READS[self.stepper.active]
            if forward_count[0]!=passes+1 or not 1<=passes<=expected_passes or row['sgd_updates']!=1:
                raise AssertionError('Incorrect method-specific gradient / diagnostic / one-SGD budget')
            if passes!=expected_passes and not row.get('fallback_reason'):
                raise AssertionError('Missing explicit numerical fallback reason')
            if self.previous_identity is None and row['history_used']:
                raise AssertionError('History cannot cross a task expansion')
            if self.direction in ('sgd','current_g2') and row['history_used']:
                raise AssertionError('No-history baseline used history')
            self.history_used_steps+=int(row['history_used'])
            prev=self.previous_identity
            current_ids=set(self.batch_identity['indices'])
            previous_ids=set(prev['indices']) if prev else set()
            row.update(step_index=self.steps,task=self.task,epoch=self.epoch,batch=self.batch,
                       task_step_index=self.steps-self.task_start_step,seed=self.seed,
                       identity=copy.deepcopy(self.batch_identity),previous_batch_identity=prev,
                       batch_overlap_count=len(current_ids&previous_ids) if prev else None,
                       class_slice=[lo,hi],seen_classes=l._total_classes,
                       fc_shape=list(model.fc.weight.shape),update_direction=self.stepper.active,
                       formation_condition_id=self.cfg['formation_condition_id'],active_method=self.stepper.active,
                       optimizer_id=id(self.optimizer),scheduler_id=id(self.scheduler),scheduler_last_epoch=self.scheduler.last_epoch,
                       training_ce_gradient_passes=passes,training_forward_count=passes,
                       diagnostic_loss_forward_count=1,actual_model_forward_count=forward_count[0],
                       diagnostic_state_restoration=diagnostic_restore,
                       train_ce=result['loss0'],post_update_ce=loss_after,
                       post_update_delta_ce=loss_after-result['loss0'],nominal_lr=lr,
                       strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                       history_source=HISTORY_SOURCE[self.direction],response_layout=self.layout.name,
                       normal_gradient_calls=expected_passes,
                       history_cache_carries_full_gradient_magnitude=HISTORY_SOURCE[self.direction] is not None,
                       gam_radii_active=self.stepper.active in ('sam','gam_residual'),reference_outer_radius=1.,reference_inner_radius=.05)
            row.update(radius_metadata(self.cfg))
            if self.cfg['response_radius_mode']==INDEPENDENT_MODE:
                if row['radius_mode']!=INDEPENDENT_MODE:
                    raise AssertionError('Independent radius mode did not reach the active stepper')
                for prefix in ('inner','outer'):
                    theory=row[prefix+'_probe_radius_theoretical']
                    expected=self.cfg[prefix+'_probe_radius'] if self.cfg[prefix+'_probe_radius_mode']=='fixed_l2' else row['h_ref']
                    if not math.isclose(theory,expected,rel_tol=1e-12,abs_tol=1e-15):
                        raise AssertionError('Actual probe construction differs from its registered radius')
            if audit.nonfinite_paths(row):
                raise FloatingPointError('Nonfinite step record')
            self.previous_identity=copy.deepcopy(self.batch_identity)
            self.last_record=row
            self.native_compute_seconds+=row['native_probe_compute_seconds']
            self.native_commit_seconds+=row['native_sgd_commit_seconds']
            self.training_peak_bytes=max(self.training_peak_bytes,row['native_training_peak_allocated_bytes'])
            self.handle.write(json.dumps(row,allow_nan=False)+'\n');self.handle.flush()
            self.steps+=1
            self.counters['training_batches']+=1
            self.counters['training_samples']+=int(targets.numel())
            self.counters['training_ce_gradient_passes']+=passes
            self.counters['training_forward_count']+=passes
            self.counters['diagnostic_loss_forward_count']+=1
            self.counters['training_sgd_updates']+=1
            self.samples+=int(targets.numel());self.loss_sum+=result['loss0']*targets.numel()
            self.correct+=int(result['logits0'][:,lo:hi].argmax(-1).eq(targets-lo).sum())
            self.epoch_batches+=1
            return result['logits0'],result['loss0'],dict(ce_clean=result['loss0'],gam=result['loss0'],ewc=0.,
                delta_w_penalty=0.,lambda_ewc=0.,ewc_gam_ratio=0.,delta_w_penalty_gam_ratio=0.)
        l._step_batch_with_log=MethodType(batch,l)

        original_run=l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            if self.task!=l._cur_task or class_offset!=self.layout.class_range(self.task)[0]:
                raise AssertionError('Native lifecycle task mismatch')
            if scheduler.T_max!=20 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1):
                raise AssertionError('Base epoch scheduler changed')
            self.layout.validate_data_manager(self.dm)
            self.scheduler=scheduler;self.test_loader=test_loader;self.task_dataset_size=len(train_loader.dataset)
            if self.task==1:
                self.post_expand_baseline=self.evaluate_task1_curve(0)
                self.retention_curve=[self.post_expand_baseline]
                audit.atomic_json(self.out/'task1_baselines.json',dict(task0_end20=self.task0_endpoint,post_expand=self.post_expand_baseline))
                self.write_retention_curve()
            self.task_batch_count=min(len(train_loader),self.cli.max_batches) if self.cli.max_batches else len(train_loader)
            self.task_start_step=self.steps
            initial_epochs=len(self.rows)
            original_schedule=scheduler.step
            def schedule(*a,**kw):
                result=original_schedule(*a,**kw);self.finish_epoch();return result
            scheduler.step=schedule
            result=original_run(native.EpochLoader(train_loader,self),test_loader,optimizer,scheduler,
                                epochs=20,class_offset=class_offset)
            if len(self.rows)-initial_epochs!=20:
                raise AssertionError('Incomplete epoch lifecycle')
            return result
        l._run_regularized_train=MethodType(run,l)

        original_increment=l.incremental_train
        def incremental(_self,dm):
            self.layout.validate_data_manager(dm)
            if self.dataset_receipt is None:
                train_count=len(dm._train_targets);test_count=len(dm._test_targets)
                if train_count!=5981 or test_count!=1519:
                    raise AssertionError('Frozen ImageNet-A full train/test sizes must be 5981/1519')
                self.dataset_receipt=dict(response_layout=self.layout.name,nb_classes=dm.nb_classes,
                                          full_train_samples=train_count,full_test_samples=test_count,
                                          full_task_increments=list(dm._increments),
                                          full_task_boundaries=[list(x) for x in self.layout.boundaries],
                                          class_order=list(map(int,dm._class_order)))
                audit.atomic_json(self.out/'dataset_partition_receipt.json',self.dataset_receipt)
            self.dm=dm;self.task=l._cur_task+1
            fc=getattr(l._unwrap_network(),'fc',None)
            self.old_fc={} if fc is None else {k:v.detach().clone() for k,v in fc.state_dict().items()}
            native.tensor_state_empty(l)
            result=original_increment(dm)
            native.tensor_state_empty(l);self.save_task_checkpoint()
            if self.task==0:
                self.task0_endpoint=self.evaluate_test_group(0,20,old_local=True)
                self.task0_endpoint.update(task=0,epoch=20,classification_classes=20)
                audit.atomic_json(self.out/'task0_endpoint.json',self.task0_endpoint)
            return result
        l.incremental_train=MethodType(incremental,l)
        original_eval=l.eval_task
        def evaluate(_self,*a,**kw):
            with observational(l._unwrap_network()): result=original_eval(*a,**kw)
            cnn,_=result
            self.matrix.append(seen_accuracy_row(cnn,self.task,l._total_classes,len(self.matrix),self.layout))
            self.sample_curve.append(float(cnn['top1']));self.write_matrix();return result
        l.eval_task=MethodType(evaluate,l)
        for name in ('_compute_accuracy','_compute_task_local_accuracy'):
            if hasattr(l,name):
                original=getattr(l,name)
                def observed(_self,*a,_original=original,**kw):
                    with observational(l._unwrap_network()): return _original(*a,**kw)
                setattr(l,name,MethodType(observed,l))
        original_after=l.after_task
        def after(_self):
            if not l._ewc_skip_task_fisher or float(l._lambda_ewc)!=0.:
                raise AssertionError('Task Fisher/EWC enabled')
            before=native.trainable_snapshot(l._unwrap_network());result=original_after()
            if not native.trainable_unchanged(l._unwrap_network(),before):
                raise AssertionError('after_task changed pure-run parameters')
            native.tensor_state_empty(l)
            self.after_task_checks.append(dict(task=self.task,parameters_exact=True,parameter_references_exact=True,
                                               fisher_empty=True,known_classes=l._known_classes))
            audit.atomic_json(self.out/'after_task_checks.json',self.after_task_checks)
            return result
        l.after_task=MethodType(after,l)

    def evaluate_dev(self):
        model=self.learner._unwrap_network();lo,hi=self.layout.class_range(self.task)
        loader=build_dev_loader(self.dm,self.task,self.cfg)
        loss=0.;correct=count=0
        with inherited.observational(model):
            model.eval()
            with torch.no_grad():
                for _,inputs,targets in loader:
                    inputs=inputs.to(self.learner._device);targets=targets.to(self.learner._device)
                    logits=model(inputs)['logits'][:,lo:hi]
                    loss+=float(F.cross_entropy(logits,targets-lo,reduction='sum'))
                    correct+=int(logits.argmax(-1).eq(targets-lo).sum());count+=int(targets.numel())
        if count!=len(loader.dataset) or not math.isfinite(loss):raise AssertionError('Incomplete/nonfinite passive dev evaluation')
        return dict(task=self.task,epoch=self.epoch,dev_current_task_ce=loss/count,
                    dev_current_task_accuracy=100.*correct/count,dev_samples=count,
                    dev_raw_train_indices=list(loader.dataset.study_full_train_indices),
                    dev_split_sha256=self.dm.study_split_sha256,selection_rule='fixed endpoint epoch20 CE; no best-epoch selection',
                    test_used_for_selection=False,passive_state_and_rng_restored=True)
    def evaluate_test_group(self,task,seen_classes=None,old_local=False):
        model=self.learner._unwrap_network();seen_classes=seen_classes or model.fc.weight.shape[0]
        dataset=self.dm.get_dataset(range(20*task,20*(task+1)),source='test',mode='test')
        loader=self.dm.build_dataloader(dataset,batch_size=128,shuffle=False,num_workers=self.cfg.get('eval_num_workers',0),
            generator=torch.Generator().manual_seed(20261010+task),persistent_workers=False)
        loss=0.;correct=count=0;local_correct=0
        with inherited.observational(model):
            model.eval()
            with torch.no_grad():
                for _,inputs,targets in loader:
                    inputs=inputs.to(self.learner._device);targets=targets.to(self.learner._device)
                    logits=model(inputs)['logits'][:,:seen_classes]
                    loss+=float(F.cross_entropy(logits,targets,reduction='sum'))
                    correct+=int(logits.argmax(-1).eq(targets).sum());count+=int(targets.numel())
                    if old_local:local_correct+=int(logits[:,:20].argmax(-1).eq(targets).sum())
        if count!=len(dataset) or not math.isfinite(loss):raise AssertionError('Incomplete/nonfinite passive test group')
        result=dict(accuracy=100.*correct/count,ce=loss/count,samples=count,classification_classes=seen_classes)
        if old_local:result['old20_only_accuracy']=100.*local_correct/count
        return result

    def evaluate_task1_curve(self,epoch):
        started=time.perf_counter();old=self.evaluate_test_group(0,40,old_local=True)
        current=self.evaluate_test_group(1,40)
        row=dict(epoch=epoch,task=1,old_test_allseen40=old,current_test_allseen40=current,
            allseen40_task_macro_accuracy=(old['accuracy']+current['accuracy'])/2,
            allseen40_sample_accuracy=(old['accuracy']*old['samples']+current['accuracy']*current['samples'])/(old['samples']+current['samples']),
            evaluation_state_rng_restored=True,test_used_for_selection=False)
        self.passive_eval_seconds+=time.perf_counter()-started
        return row


    def begin_epoch(self,epoch):
        native.Controller.begin_epoch(self,epoch)
        lr=float(self.optimizer.param_groups[0]['lr'])
        if not math.isclose(lr,cosine_lr(.04 if self.task==0 else .02,epoch),rel_tol=1e-12,abs_tol=1e-15):
            raise AssertionError('Common20epoch cosine schedule was restarted or changed')
        if self.scheduler.last_epoch!=epoch-1:raise AssertionError('Scheduler epoch progress is discontinuous')
        before_method=self.stepper.active
        self.cfg['_formation_epoch']=epoch
        self.stepper.set_epoch(epoch)
        if self.task==0 and epoch==11:
            before=self.epoch10_boundary
            same_model=base.parameter_hash(self.learner._unwrap_network())==before['model_sha256']
            same_rng=rng_hash(audit.rng_get())==before['rng_identity']
            receipt=dict(task=0,from_epoch=10,to_epoch=11,method_before=before_method,method_after=self.stepper.active,
                optimizer_id_before=before['optimizer_id'],optimizer_id_after=id(self.optimizer),
                scheduler_id_before=before['scheduler_id'],scheduler_id_after=id(self.scheduler),
                optimizer_same=before['optimizer_id']==id(self.optimizer),scheduler_same=before['scheduler_id']==id(self.scheduler),
                model_parameters_unchanged_at_switch=same_model,parameter_references_unchanged_at_switch=before['parameter_ids']==[id(p) for p in self.learner._unwrap_network().parameters()],
                rng_unchanged_at_switch=same_rng,lr_used_epoch10=before['lr_used'],lr_used_epoch11=lr,
                scheduler_last_epoch_at_epoch11=self.scheduler.last_epoch,scheduler_T_max=self.scheduler.T_max,
                reset_model=False,reset_optimizer=False,reset_scheduler=False,reset_rng=False)
            if not all(receipt[k] for k in ('optimizer_same','scheduler_same','model_parameters_unchanged_at_switch','parameter_references_unchanged_at_switch','rng_unchanged_at_switch')):
                raise AssertionError('Method boundary reset live state')
            self.switch_receipts.append(receipt);audit.atomic_json(self.out/'switch_receipts.json',self.switch_receipts)

    def write_retention_curve(self):
        trained=[r for r in self.retention_curve if r['epoch']>0]
        audit.atomic_json(self.out/'retention_curve.json',dict(epochs=self.retention_curve,
            definition='epoch0 after20to40 head expansion, before Task1 learning; then20 SGD epochs',
            old_drop_definition='epoch0 old_allseen40 accuracy minus epoch20 old_allseen40 accuracy',
            retention_auc=sum(r['old_test_allseen40']['accuracy'] for r in trained)/len(trained) if trained else None,
            retention_auc_definition='mean old_allseen40 over trained epochs1..20; epoch0 excluded'))

    def finish_epoch(self):
        native.Controller.finish_epoch(self)
        self.rows[-1].update(active_method=self.stepper.active,formation_condition_id=self.cfg['formation_condition_id'],
            optimizer_id=id(self.optimizer),scheduler_id=id(self.scheduler),scheduler_last_epoch=self.scheduler.last_epoch,
            expected_native_gradient_reads=READS[self.stepper.active]*self.epoch_batches)
        started=time.perf_counter();dev=self.evaluate_dev();self.passive_eval_seconds+=time.perf_counter()-started
        self.dev_curve.append(dev);audit.atomic_json(self.out/'dev_curve.json',dict(epochs=self.dev_curve,selection_rule='fixed epoch20 endpoint; no selection'))
        if self.task==0:
            started=time.perf_counter();test=self.evaluate_test_group(0,20,old_local=True);self.passive_eval_seconds+=time.perf_counter()-started
            self.epoch_task0_curve.append(dict(epoch=self.epoch,active_method=self.stepper.active,test=test,dev=dev))
            audit.atomic_json(self.out/'task0_learning_curve.json',dict(epochs=self.epoch_task0_curve))
            if self.epoch==10:
                self.epoch10_boundary=dict(model_sha256=base.parameter_hash(self.learner._unwrap_network()),
                    rng_identity=rng_hash(audit.rng_get()),
                    parameter_ids=[id(p) for p in self.learner._unwrap_network().parameters()],optimizer_id=id(self.optimizer),
                    scheduler_id=id(self.scheduler),lr_used=self.lr_used[0])
        else:
            curve=self.evaluate_task1_curve(self.epoch);curve.update(current_dev_ce=dev['dev_current_task_ce'],current_dev_accuracy=dev['dev_current_task_accuracy'])
            self.retention_curve.append(curve);self.write_retention_curve()
        audit.atomic_json(self.out/'metrics.json',dict(epochs=self.rows,counters=self.counters))

    def save_task_checkpoint(self):
        # Native LoRA/FC exports; complete optimizer/scheduler/RNG sidecar, no
        # 686MB full frozen-backbone copies or unrequested offline diagnostics.
        l=self.learner;model=l._unwrap_network();task=self.task
        directory=self.out/'checkpoints'/('task_%03d'%task);directory.mkdir(parents=True,exist_ok=False)
        native_dir=Path(self.cfg['filepath'])
        names=['lora_w_a_%d.pt'%task,'lora_w_b_%d.pt'%task,'lora_meta_%d.json'%task,'fc_state_%d.pt'%task]
        for name in names:
            if not (native_dir/name).is_file():raise FileNotFoundError('Missing native export '+name)
            shutil.copyfile(native_dir/name,directory/name)
        with observational(model):
            state=dict(task=task,epoch=20,config=copy.deepcopy(self.cfg),optimizer=base.cpu_tree(self.optimizer.state_dict()),
                base_optimizer=base.cpu_tree(self.optimizer.state_dict()),
                scheduler=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),
                rng=audit.rng_get(),algorithm_state=base.cpu_tree(self.stepper.state_dict()),history=None,
                formation_condition_id=self.cfg['formation_condition_id'])
            torch.save(state,directory/'training_state.pt')
            manifest=dict(task=task,epoch=20,seed=self.seed,protocol=PROTOCOL,formation_condition_id=self.cfg['formation_condition_id'],
                response_layout='cl10',class_range=[0,l._total_classes],current_task_class_range=[20*task,20*(task+1)],
                model_sha256=base.parameter_hash(model),frozen_origin_sha256=base.parameter_hash(model,True),
                fc_shape=list(model.fc.weight.shape),backbone_task_id=model.backbone.task_id,
                source_gam_sha256=audit.digest(Path(self.cli.source)/'optimer_PerturabtionType/gam.py'),
                native_checkpoint_dir=str(native_dir),files_sha256={name:audit.digest(directory/name) for name in names+['training_state.pt']},
                loader='learner.load_task_checkpoint(directory, task, data_manager); no fullbackbone copy required')
            audit.atomic_json(directory/'manifest.json',manifest)
        self.checkpoints.append(str(directory))
        audit.atomic_json(self.out/'task_checkpoints.json',dict(all_tasks={str(i):p for i,p in enumerate(self.checkpoints)}))
        self.task_rows.append(dict(task=task,epochs=20,steps=self.steps-self.task_start_step,batches_per_epoch=self.task_batch_count,
            train_dataset_samples=self.task_dataset_size,checkpoint=str(directory),formation_condition_id=self.cfg['formation_condition_id']))
        audit.atomic_json(self.out/'tasks.json',self.task_rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','config','output-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--dry-run',action='store_true');cli=parser.parse_args()
    cli.max_tasks=2;cli.max_epochs=20
    for key in ('source','config','output_dir'):setattr(cli,key,str(Path(getattr(cli,key)).resolve()))
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli);cfg['study_source']=cli.source
    if cli.dry_run:
        counts=[READS[active_method(cfg['formation_condition_id'],task,epoch)] for task in (0,1) for epoch in range(1,21)]
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,seed=cfg['seed'],formation_condition_id=cfg['formation_condition_id'],
            training_tasks_this_job=2,epochs_per_task=20,smoke=bool(cli.max_batches),max_batches=cli.max_batches,
            task0_schedule=cfg['formation_task0_schedule'],task1_method='sgd',switch_epoch=11,
            common_cosine_T_max=20,stage_lr=[.04,.02],gradient_reads_per_epoch_per_batch=counts,
            full_partition_tasks=10,trained_task_ids=[0,1],r_sf=.25,gam_r=.1*10,gam_rho=.05,
            selection_rule='fixed endpoint epoch20; no test/dev selection')));return
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Formation output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Strict CUBLAS workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    os.environ.setdefault('HF_HUB_OFFLINE','1');sys.path.insert(0,cli.source);os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import set_device,set_random,json_safe
    logging.basicConfig(level=logging.INFO,stream=sys.stdout,format='%(asctime)s %(message)s')
    cfg['filepath']=str(out/'native'/'checkpoints')+os.sep;Path(cfg['filepath']).mkdir(parents=True)
    cfg['nb_classes']=200;cfg['nb_tasks']=10;cfg['task_increments']=[20]*10
    set_random(cfg['seed']);set_device(cfg)
    dm=build_data_manager(cli.source,cfg,cfg['study_dev_split_path'])
    learner=factory.get_model(cfg['model_name'],cfg);learner._network.backbone.to(learner._device)
    c=Controller(learner,cfg,cli);c.dm=dm;c.install()
    audit.atomic_json(out/'effective_config.json',json_safe(cfg));audit.atomic_json(out/'study_split_receipt.json',split_fingerprint(dm))
    summary=dict(status='running',protocol=PROTOCOL,formation_protocol=PROTOCOL,seed=cfg['seed'],formation_condition_id=cfg['formation_condition_id'],
        response_method='formation_schedule',response_layout='cl10',study_role='formation',study_condition_id=cfg['formation_condition_id'],
        smoke=bool(cli.max_batches),training_tasks_this_job=2,full_partition_tasks=10,total_dataset_classes=200,
        task_boundaries=[[20*i,20*(i+1)] for i in range(10)],epochs_per_task=20,max_batches=cli.max_batches,
        input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
        algorithm_sha256=audit.digest(Path(__file__).with_name('formation_step.py')),
        source=cli.source,history_source=None,history_weight=0.,completed_tasks=0,trained_task_ids=[],
        dev_split_sha256=audit.digest(cfg['study_dev_split_path']),formation_task0_schedule=cfg['formation_task0_schedule'],
        formation_task1_method='sgd',formation_switch_epoch=11,sf_outer_radius=.25,gam_rho=.05,gam_outer_radius=1.)
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        for task in (0,1):
            learner.incremental_train(dm);learner.eval_task();learner.after_task()
            if learner._cur_task!=task:raise AssertionError('Incorrect two-task lifecycle')
        if len(c.task_rows)!=2 or len(c.after_task_checks)!=2 or len(c.rows)!=40 or len(c.matrix)!=2 or len(c.retention_curve)!=21:
            raise AssertionError('Incomplete two-task20epoch protocol')
        if any(c.counters[k] for k in ('forbidden_work_attempts','weight_merge_updates','fisher_estimation_passes')):
            raise AssertionError('Forbidden auxiliary work occurred')
        c.dataset_receipt.update(study_split=split_fingerprint(dm),actual_training_tasks=[0,1])
        audit.atomic_json(out/'dataset_partition_receipt.json',c.dataset_receipt)
        endpoint=c.retention_curve[-1];start=c.post_expand_baseline
        old_start=start['old_test_allseen40']['accuracy'];old_final=endpoint['old_test_allseen40']['accuracy']
        current_final=endpoint['current_test_allseen40']['accuracy'];task0_end=c.task0_endpoint['accuracy']
        faa=(old_final+current_final)/2
        formation_metrics=dict(task0_end20=task0_end,old_start40=old_start,old_final40=old_final,current_final40=current_final,
            old_drop_since_task1_start=old_start-old_final,training_BWT=old_final-old_start,
            expansion_damage=task0_end-old_start,FAA=faa,traditional_BWT=old_final-task0_end,
            stage_AAA=(task0_end+faa)/2,retention_auc=sum(r['old_test_allseen40']['accuracy'] for r in c.retention_curve[1:])/20,
            accuracy_unit='percent',delta_unit='percentage_points',old_drop_definition='old_start_after_head_expansion40 minus old_final40')
        final_metrics=json.loads((out/'task_matrix.json').read_text())
        summary.update(status='completed',completed_tasks=2,trained_task_ids=[0,1],completed_epochs=40,completed_steps=c.steps,
            counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,task_identities=c.task_identities,
            dataset_partition_receipt=c.dataset_receipt,initial_identity=c.task_identities[0],history_used_steps=0,
            task0_endpoint=c.task0_endpoint,task1_baselines=dict(task0_end20=c.task0_endpoint,post_expand=c.post_expand_baseline),
            retention_curve=c.retention_curve,formation_metrics=formation_metrics,switch_receipts=c.switch_receipts,
            elapsed_seconds=time.time()-c.started,step_trace_sha256=audit.digest(c.trace),task_matrix_sha256=audit.digest(out/'task_matrix.json'),
            final_metrics=final_metrics,endpoint_dev=c.dev_curve[-1],
            native_probe_compute_seconds=c.native_compute_seconds,native_sgd_commit_seconds=c.native_commit_seconds,
            native_training_compute_seconds=c.native_compute_seconds+c.native_commit_seconds,passive_extra_eval_seconds=c.passive_eval_seconds,
            native_training_peak_allocated_bytes=c.training_peak_bytes,final_model_parameter_sha256=base.parameter_hash(learner._unwrap_network()),
            checks=dict(finite=True,exact_snapshot_restore=True,one_sgd_per_batch=True,no_history=True,fisher_empty=True,
                actual_weight_merges_zero=True,train_dev_disjoint=True,test_not_used_for_selection=True,
                task1_only_sgd=True,switch_without_reset=True,old_drop_after_head_expansion=True))
        if audit.nonfinite_paths(summary):raise FloatingPointError('Nonfinite completed summary')
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_steps=c.steps);raise
    finally:
        c.handle.close();audit.atomic_json(out/'final_summary.json',summary)

if __name__=='__main__':main()
