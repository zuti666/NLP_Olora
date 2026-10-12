#!/usr/bin/env python3
"""Four response methods on the frozen ImageNet-A one-task / CL lifecycle."""
import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import MethodType, SimpleNamespace

sys.dont_write_bytecode = True
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
from effective_step import EffectiveStep
from state_transaction import State
from response_step import ResponseStep
from layout_lifecycle import TaskLayout, seen_accuracy_row, matrix_metrics

PROTOCOL = 'four_response_methods_sgd_prediction_exact_restore_rng_replay_v1'
INDEPENDENT_MODE = 'independent_inner_outer_v1'
INDEPENDENT_PROTOCOL = 'current_g2_independent_inner_outer_exact_restore_rng_replay_v1'
DIRECTIONS = ('sgd', 'current_g2', 'history_fusion', 'history_g2')
NORMAL_GRADIENTS = {'sgd': 1, 'current_g2': 3, 'history_fusion': 2, 'history_g2': 3}
HISTORY_SOURCE = {'sgd': None, 'current_g2': None, 'history_fusion': 'base', 'history_g2': 'final'}
PROBE_VARIANTS = {
    'CurrentG2-FixedInner': ('fixed_l2', 'sgd_step', 'fixed_inner'),
    'CurrentG2-FixedOuter': ('sgd_step', 'fixed_l2', 'fixed_outer'),
    'CurrentG2-FixedBoth': ('fixed_l2', 'fixed_l2', 'fixed_both'),
}
PROBE_CONFIG_KEYS = ('inner_probe_radius_mode', 'inner_probe_radius',
                     'outer_probe_radius_mode', 'outer_probe_radius')

def radius_metadata(cfg):
    """Configured radii are null for the dynamic eta*||g0|| branches.

    Legacy GAM radii remain inactive reference metadata in every response run;
    the independent .05 radii below belong only to this new current_g2 variant.
    Step records additionally report their measured/theoretical probe distances.
    """
    independent = cfg['response_radius_mode'] == INDEPENDENT_MODE
    return dict(
        response_probe_variant=cfg.get('response_probe_variant',
            'CurrentG2-Original' if cfg['response_method']=='current_g2' else cfg['response_method']),
        radius_variant=cfg.get('radius_variant', 'coupled_reference'),
        probe_mode=cfg['response_radius_mode'],
        inner_probe_radius_mode=(cfg['inner_probe_radius_mode'] if independent else
                                 'sgd_step' if cfg['response_method']!='sgd' else None),
        outer_probe_radius_mode=(cfg['outer_probe_radius_mode'] if independent else
                                 'sgd_step' if cfg['response_method'] in ('current_g2','history_g2') else None),
        inner_rho=cfg['inner_probe_radius'] if independent else None,
        outer_R=cfg['outer_probe_radius'] if independent else None,
        probe_radius_coordinates='global L2 over all trainable LoRA and current classifier tensors',
        normal_gradient_calls=NORMAL_GRADIENTS[cfg['response_method']],
        gam_radii_active=False,
    )

def make_stepper(model, params, geometry, cfg):
    if cfg['response_radius_mode'] == INDEPENDENT_MODE:
        # The registered original four-method route retains ResponseStep.
        from independent_response_step import IndependentResponseStep
        return IndependentResponseStep(model, params, geometry,
            dict(method=cfg['response_method'],history_weight=0.,eps=1e-12,backend='native',
                 radius_mode=INDEPENDENT_MODE,radius_scale=1.,
                 radius_variant=cfg['radius_variant'],
                 response_probe_variant=cfg['response_probe_variant'],
                 inner_probe_radius_mode=cfg['inner_probe_radius_mode'],
                 outer_probe_radius_mode=cfg['outer_probe_radius_mode'],
                 fixed_inner_probe_radius=cfg['inner_probe_radius'],
                 fixed_outer_probe_radius=cfg['outer_probe_radius']))
    return ResponseStep(model, params, geometry,
        dict(method=cfg['response_method'],history_weight=cfg['history_weight'],eps=1e-12,backend='native',
             radius_mode='sgd_step',radius_scale=1.))

@contextmanager
def observational(model):
    state = State(model)
    try:
        yield
    finally:
        state.restore()
        state.check()

def resolve_config(raw, cli):
    direction = raw.get('response_method')
    if direction not in DIRECTIONS:
        raise ValueError('Expected sgd, current_g2, history_fusion, or history_g2')
    layout = TaskLayout.from_name(raw.get('response_layout'))
    expected_w = .5 if direction in ('history_fusion','history_g2') else 0.
    if raw.get('history_weight') != expected_w:
        raise ValueError('History weight is frozen to .5 for history methods and 0 otherwise')
    if raw.get('history_source') != HISTORY_SOURCE[direction]:
        raise ValueError('History cache source must follow the selected method')
    mode = raw.get('response_radius_mode')
    if mode not in ('sgd_step', INDEPENDENT_MODE) or raw.get('radius_scale') != 1:
        raise ValueError('Explicit supported response_radius_mode and radius_scale=1 required')
    for key in ('fixed_inner_probe_radius','fixed_outer_probe_radius',
                'inner_radius_mode','outer_radius_mode','inner_radius','outer_radius'):
        if key in raw:
            raise ValueError('Unsupported raw radius alias would be ignored: '+key)
    if mode == 'sgd_step':
        if any(key in raw for key in PROBE_CONFIG_KEYS):
            raise ValueError('Independent probe radius fields cannot be silently ignored by the original method')
        if 'response_probe_variant' in raw and not (
                direction=='current_g2' and raw['response_probe_variant']=='CurrentG2-Original'):
            raise ValueError('The original current_g2 accepts only CurrentG2-Original as its optional variant label')
        if raw.get('radius_variant','coupled_reference') != 'coupled_reference':
            raise ValueError('The original method cannot carry an independent radius variant label')
    else:
        if direction != 'current_g2':
            raise ValueError('Independent inner/outer radii are implemented only for current_g2 without history')
        variant = raw.get('response_probe_variant')
        if variant not in PROBE_VARIANTS:
            raise ValueError('An explicit FixedInner, FixedOuter, or FixedBoth variant is required')
        inner_mode, outer_mode, short_variant = PROBE_VARIANTS[variant]
        if raw.get('radius_variant', short_variant) != short_variant:
            raise ValueError('radius_variant conflicts with the independent probe variant')
        for prefix, expected_mode in (('inner',inner_mode),('outer',outer_mode)):
            key=prefix+'_probe_radius'
            if prefix+'_probe_radius_mode' not in raw or key not in raw:
                raise ValueError('Both explicit probe modes and radii (null for dynamic branches) are required')
            if raw[prefix+'_probe_radius_mode'] != expected_mode:
                raise ValueError('Probe mode conflicts with '+variant)
            value=raw[key]
            if expected_mode=='fixed_l2':
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value != .05:
                    raise ValueError('This registered contrast fixes '+key+' to .05')
            elif value is not None:
                raise ValueError('Dynamic SGD-distance probe radii must be null in the configuration')
        if raw.get('response_algorithm_family','current_g2') != 'current_g2':
            raise ValueError('Independent probes retain the current_g2 algorithm family')
    if raw.get('history_reset') != 'task':
        raise ValueError('Task/head changes must clear cached history')
    if raw.get('post_update_loss') is not True or raw.get('strict_deterministic') is not True:
        raise ValueError('Passive loss diagnostic and strict deterministic kernels required')
    seed = raw.get('seed')
    if seed not in (2, [2]):
        raise ValueError('First round is the fixed seed2 comparison')
    fixed = {'dataset':'imageneta','class_shuffle':False,
             'init_cls':layout.increments[0],'increment':layout.increments[0],
             'lora_rank':10,'gam_grad_rho':.05,'gam_grad_norm_rho':1.,'init_epoch':50,'epochs':50,
             'batch_size':128,'init_lr':.04,'lrate':.02,'init_momentum':0,'momentum':0,
             'init_weight_decay':0.,'weight_decay':0.,'ewc_lambda':0.,'ewc_skip_task_fisher':True,
             'approximate_fisher_in_task':False,'in_task_weight_merge':False,'task_endpoint_weight_merge':False}
    for k,v in fixed.items():
        if raw.get(k) != v:
            raise ValueError(f'Frozen setting mismatch: {k}')
    if not 0<=cli.max_tasks<=layout.tasks or not 0<=cli.max_epochs<=50 or cli.max_batches<0:
        raise ValueError('Smoke budget exceeds the selected full class partition')
    for key in ('task_splits','eval_task_splits','task_increments'):
        if key in raw and tuple(raw[key]) != layout.increments:
            raise ValueError('Explicit task partition differs from response_layout: '+key)
    surrogate = copy.deepcopy(raw)
    # The legacy config normalizer is 10x20-specific. Its surrogate only
    # normalizes the shared pure-LoRA settings; the real DataManager below
    # receives the selected complete partition, including init_cls=200.
    surrogate.update(component_rule='u0',component_scale='natural',init_cls=20,increment=20)
    cfg = native.resolve_config(surrogate, cli)
    cfg.update(response_method=direction,update_direction=direction,response_layout=layout.name,
               init_cls=layout.increments[0],increment=layout.increments[0],
               task_increments=list(layout.increments),eval_task_splits=list(layout.increments),
               max_train_tasks=cli.max_tasks or layout.tasks,
               component_rule=direction,component_scale='natural',
               optimizer_tag_override=direction+'_response_v1',history_weight=expected_w,
               history_source=HISTORY_SOURCE[direction],response_radius_mode=mode,radius_scale=1.,
               history_reset='task',
               history_min_distance=1e-12, post_update_loss=True, strict_deterministic=True,
               protocol=PROTOCOL,compute_nme_class_means=False,probe_gradient_count=NORMAL_GRADIENTS[direction],
               compute_g3_probe=False, gam_radii_active=False, reference_outer_radius=1.,
               reference_inner_radius=.05, prediction_distance='current_lr_times_global_gradient_norm',
               in_task_path_normalize_snapshots=False, attention_backend='native')
    if mode == INDEPENDENT_MODE:
        cfg.update(response_probe_variant=variant,radius_variant=short_variant,
                   inner_probe_radius_mode=inner_mode,outer_probe_radius_mode=outer_mode,
                   inner_probe_radius=raw['inner_probe_radius'],outer_probe_radius=raw['outer_probe_radius'],
                   response_algorithm_family='current_g2',protocol=INDEPENDENT_PROTOCOL,
                   optimizer_tag_override='current_g2_'+short_variant+'_response_v1',
                   response_lambda_mode=('fixed_global_l2_outer_radius' if outer_mode=='fixed_l2'
                                         else 'sgd_prediction_distance'),
                   prediction_distance=('fixed_global_L2_inner_probe_radius' if inner_mode=='fixed_l2'
                                        else 'current_lr_times_global_gradient_norm'))
    return cfg

class Controller(native.Controller):
    def __init__(self, learner, cfg, cli):
        super().__init__(learner,cfg,cli)
        self.direction=cfg['response_method']
        self.layout=TaskLayout.from_name(cfg['response_layout'])
        self.epoch_test_curve=[]
        self.dataset_receipt=None
        self.last_record=None
        self.previous_identity=None
        self.history_used_steps=0
        self.counters.update(training_forward_count=0,training_sgd_updates=0,
                             diagnostic_loss_forward_count=0,training_hvp_passes=0,
                             training_create_graph_passes=0,training_global_projection_calls=0)

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
            self.stepper=make_stepper(model,params,self.geometry,self.cfg)
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
            expected_passes=NORMAL_GRADIENTS[self.direction]
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
                       fc_shape=list(model.fc.weight.shape),update_direction=self.direction,
                       training_ce_gradient_passes=passes,training_forward_count=passes,
                       diagnostic_loss_forward_count=1,actual_model_forward_count=forward_count[0],
                       diagnostic_state_restoration=diagnostic_restore,
                       train_ce=result['loss0'],post_update_ce=loss_after,
                       post_update_delta_ce=loss_after-result['loss0'],nominal_lr=lr,
                       strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                       history_source=HISTORY_SOURCE[self.direction],response_layout=self.layout.name,
                       normal_gradient_calls=expected_passes,
                       history_cache_carries_full_gradient_magnitude=HISTORY_SOURCE[self.direction] is not None,
                       gam_radii_active=False,reference_outer_radius=1.,reference_inner_radius=.05)
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
            if scheduler.T_max!=50 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1):
                raise AssertionError('Base epoch scheduler changed')
            self.layout.validate_data_manager(self.dm)
            self.scheduler=scheduler;self.test_loader=test_loader;self.task_dataset_size=len(train_loader.dataset)
            self.task_batch_count=min(len(train_loader),self.cli.max_batches) if self.cli.max_batches else len(train_loader)
            self.task_start_step=self.steps
            initial_epochs=len(self.rows)
            original_schedule=scheduler.step
            def schedule(*a,**kw):
                result=original_schedule(*a,**kw);self.finish_epoch();return result
            scheduler.step=schedule
            result=original_run(native.EpochLoader(train_loader,self),test_loader,optimizer,scheduler,
                                epochs=self.cli.max_epochs or epochs,class_offset=class_offset)
            if len(self.rows)-initial_epochs!=(self.cli.max_epochs or 50):
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
            native.tensor_state_empty(l);self.save_task_checkpoint();return result
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

    def finish_epoch(self):
        # The single-task condition needs an epoch curve, not a single CL-stage
        # observation mislabelled as an epoch AAA. This evaluation is passive
        # and the full transaction prevents it from changing subsequent RNG,
        # gradients, parameter references, buffers, or mutable model caches.
        accuracy=None
        if self.layout.name=='single200':
            model=self.learner._unwrap_network()
            with observational(model):
                accuracy=float(self.learner._compute_accuracy(self.learner._network,self.test_loader))
            if not math.isfinite(accuracy) or not 0<=accuracy<=100:
                raise FloatingPointError('Invalid single-task epoch test accuracy')
        super().finish_epoch()
        if accuracy is not None:
            self.rows[-1]['test_accuracy_all_200_classes']=accuracy
            self.epoch_test_curve.append(dict(epoch=self.epoch,test_accuracy=accuracy,
                                             train_ce=self.rows[-1]['train_ce']))
            audit.atomic_json(self.out/'metrics.json',dict(epochs=self.rows,counters=self.counters))
            audit.atomic_json(self.out/'epoch_test_curve.json',dict(response_layout=self.layout.name,
                              evaluation_scope='all 200 classes; observational per-epoch test evaluation',
                              selection_rule='report epoch50, no best-epoch checkpoint selection',
                              epochs=self.epoch_test_curve))

    def write_matrix(self):
        result=matrix_metrics(self.matrix,self.sample_curve,self.layout)
        result['epoch_test_curve']=copy.deepcopy(self.epoch_test_curve)
        audit.atomic_json(self.out/'task_matrix.json',result)

    def _save_layout_checkpoint(self):
        """Same native LoRA/FC artifacts with layout-driven checkpoint metadata."""
        l=self.learner;model=l._unwrap_network();task=self.task
        directory=self.out/'checkpoints'/('task_%03d'%task);directory.mkdir(parents=True,exist_ok=False)
        native_dir=Path(self.cfg['filepath'])
        names=['lora_w_a_%d.pt'%task,'lora_w_b_%d.pt'%task,'lora_meta_%d.json'%task,'fc_state_%d.pt'%task]
        for name in names:
            if not (native_dir/name).is_file():
                raise FileNotFoundError('Missing native checkpoint '+str(native_dir/name))
            shutil.copyfile(native_dir/name,directory/name)
        with observational(model):
            state=dict(task=task,epoch=self.cli.max_epochs or 50,config=self.cfg,
                       optimizer=base.cpu_tree(self.optimizer.state_dict()),
                       base_optimizer=base.cpu_tree(self.optimizer.base_optimizer.state_dict()),
                       scheduler=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),
                       rng=audit.rng_get())
            torch.save(state,directory/'training_state.pt')
            manifest=dict(task=task,epoch=self.cli.max_epochs or 50,seed=self.seed,
                          response_method=self.direction,response_layout=self.layout.name,
                          history_source=HISTORY_SOURCE[self.direction],
                          class_range=[0,l._total_classes],
                          current_task_class_range=list(self.layout.class_range(task)),
                          model_sha256=base.parameter_hash(model),
                          source_gam_sha256=audit.digest(Path(self.cli.source)/'optimer_PerturabtionType/gam.py'),
                          fc_shape=list(model.fc.weight.shape),backbone_task_id=model.backbone.task_id,
                          native_checkpoint_dir=str(native_dir),
                          files_sha256={name:audit.digest(directory/name) for name in names},
                          loader='learner.load_task_checkpoint(checkpoint_dir, task, data_manager); cumulative current single adapter')
            manifest.update(radius_metadata(self.cfg))
            audit.atomic_json(directory/'manifest.json',manifest)
        self.checkpoints.append(str(directory))
        reps=sorted({0,self.layout.tasks//2,self.layout.tasks-1})
        audit.atomic_json(self.out/'task_checkpoints.json',dict(
            loader='load_task_checkpoint(directory, task, data_manager)',
            all_tasks={str(i):p for i,p in enumerate(self.checkpoints)},
            representative_tasks={str(i):self.checkpoints[i] for i in reps if i<len(self.checkpoints)}))
        self.task_rows.append(dict(task=task,response_layout=self.layout.name,
                                  current_task_class_range=list(self.layout.class_range(task)),
                                  train_dataset_samples=self.task_dataset_size,batches_per_epoch=self.task_batch_count,
                                  epochs=self.cli.max_epochs or 50,steps=self.steps-self.task_start_step,
                                  checkpoint=str(directory),**radius_metadata(self.cfg)))
        audit.atomic_json(self.out/'tasks.json',self.task_rows)

    def save_task_checkpoint(self):
        self._save_layout_checkpoint()
        directory=Path(self.checkpoints[-1])
        state_path=directory/'training_state.pt'
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        state['algorithm_state']=base.cpu_tree(self.stepper.state_dict())
        state['previous_batch_identity']=copy.deepcopy(self.previous_identity)
        state['algorithm']=self.direction
        state['history_source']=HISTORY_SOURCE[self.direction]
        state['response_radius_metadata']=radius_metadata(self.cfg)
        torch.save(state,state_path)
        manifest=json.loads((directory/'manifest.json').read_text())
        manifest.update(algorithm=self.direction,algorithm_cache_in_training_state=True)
        manifest['files_sha256']['training_state.pt']=audit.digest(state_path)
        audit.atomic_json(directory/'manifest.json',manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','config','output-dir'):parser.add_argument('--'+name,required=True)
    for name in ('max-tasks','max-epochs','max-batches'):parser.add_argument('--'+name,type=int,default=0)
    parser.add_argument('--dry-run',action='store_true');cli=parser.parse_args()
    for name in ('source','config','output_dir'):setattr(cli,name,str(Path(getattr(cli,name)).resolve()))
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli)
    layout=TaskLayout.from_name(cfg['response_layout'])
    direction=cfg['response_method'];normal_calls=NORMAL_GRADIENTS[direction]
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=cfg['protocol'],response_method=direction,response_layout=layout.name,seed=2,
                             gradients_per_normal_step=normal_calls,updates_per_step=1,diagnostic_forwards_per_step=1,
                             history_weight=cfg['history_weight'],history_source=cfg['history_source'],
                             prediction_distance=cfg['prediction_distance'],response_radius_mode=cfg['response_radius_mode'],
                             gam_radii_active=False,init_cls=cfg['init_cls'],increment=cfg['increment'],
                             full_partition_tasks=layout.tasks,classes=layout.classes,
                             tasks=cli.max_tasks or layout.tasks,epochs=cli.max_epochs or 50,
                             **{key:value for key,value in radius_metadata(cfg).items() if key!='gam_radii_active'})))
        return
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('Deterministic CUBLAS workspace must be set before CUDA initialization')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    os.environ.setdefault('HF_HUB_OFFLINE','1');sys.path.insert(0,cli.source);os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import json_safe
    from src import trainer
    original_factory=factory.get_model;holder={}
    def create(name,args):
        if holder:raise AssertionError('Exactly one model per independent run')
        learner=original_factory(name,args);c=Controller(learner,args,cli);holder['controller']=c;c.install()
        audit.atomic_json(out/'effective_config.json',json_safe(args));return learner
    factory.get_model=create
    algorithm_file=Path(__file__).with_name('independent_response_step.py' if cfg['response_radius_mode']==INDEPENDENT_MODE else 'response_step.py')
    summary=dict(status='running',protocol=cfg['protocol'],seed=2,update_direction=cfg['update_direction'],
                 response_method=direction,response_layout=layout.name,history_source=cfg['history_source'],
                 history_weight=cfg['history_weight'],response_radius_mode=cfg['response_radius_mode'],radius_scale=1.,
                 prediction_distance=cfg['prediction_distance'],gam_radii_active=False,r=None,rho=None,
                 reference_outer_radius=1.,reference_inner_radius=.05,
                 gradients_per_normal_step=normal_calls,diagnostic_forwards_per_step=1,updates_per_step=1,
                 smoke=bool(cli.max_tasks or cli.max_epochs or cli.max_batches),
                 requested_tasks=cli.max_tasks or layout.tasks,full_partition_tasks=layout.tasks,
                 task_boundaries=[list(x) for x in layout.boundaries],
                 total_dataset_classes=layout.classes,init_cls=cfg['init_cls'],increment=cfg['increment'],
                 epochs_per_task=cli.max_epochs or 50,
                 scheduler_cadence='epoch',scheduler_T_max=50,completed_tasks=0,
                 input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
                 algorithm_sha256=audit.digest(algorithm_file),
                 algorithm_file=algorithm_file.name,
                 inherited_algorithm_sha256=audit.digest(Path(__file__).with_name('response_step.py')),
                 layout_sha256=audit.digest(Path(__file__).with_name('layout_lifecycle.py')),source=cli.source)
    summary.update(radius_metadata(cfg))
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        trainer.train(cfg);c=holder['controller'];n=cli.max_tasks or layout.tasks
        if len(c.matrix)!=n or len(c.task_rows)!=n or len(c.after_task_checks)!=n:
            raise AssertionError('Incomplete native task lifecycle')
        if len(c.rows)!=n*(cli.max_epochs or 50) or c.steps!=sum(r['steps'] for r in c.task_rows):
            raise AssertionError('Incomplete epoch/update budget')
        if any(c.counters[k] for k in ('forbidden_work_attempts','weight_merge_updates','fisher_estimation_passes')):
            raise AssertionError('Forbidden work occurred')
        if c.counters['training_sgd_updates']!=c.steps or c.counters['diagnostic_loss_forward_count']!=c.steps:
            raise AssertionError('Incomplete SGD/diagnostic accounting')
        summary.update(status='completed',completed_tasks=n,completed_epochs=len(c.rows),completed_steps=c.steps,
                       counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
                       dataset_partition_receipt=c.dataset_receipt,
                       initial_identity=c.task_identities[0],history_used_steps=c.history_used_steps,
                       elapsed_seconds=time.time()-c.started,
                       checks=dict(finite=True,all_steps_self_spec=True,exact_snapshot_restore=True,
                                   exact_actual_displacement_cache=True,method_specific_gradient_cache=True,
                                   task_history_reset=True,one_sgd_per_batch=True,
                                   passive_diagnostics_restore=True,epoch_scheduler_preserved=True,
                                   fisher_empty_all_tasks=True,actual_weight_merges_zero=True),
                       step_trace_sha256=audit.digest(c.trace),task_matrix_sha256=audit.digest(out/'task_matrix.json'),
                       final_metrics=json.loads((out/'task_matrix.json').read_text()))
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_tasks=len(holder['controller'].matrix) if holder else 0)
        if holder and holder['controller'].last_record:
            audit.atomic_json(out/'failed_step_audit.json',holder['controller'].last_record)
        raise
    finally:
        factory.get_model=original_factory
        if holder:holder['controller'].handle.close()
        audit.atomic_json(out/'final_summary.json',summary)

if __name__=='__main__':main()
