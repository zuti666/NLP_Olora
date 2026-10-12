#!/usr/bin/env python3
"""Two-gradient cached-history prototype on the frozen ImageNet-A lifecycle."""
import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
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
from history_step import HistoryResponseStep

PROTOCOL = 'cached_history_sgd_prediction_direct_response_exact_restore_rng_replay_v1'
DIRECTIONS = ('history_fused', 'future_only')

@contextmanager
def observational(model):
    state = State(model)
    try:
        yield
    finally:
        state.restore()
        state.check()

def resolve_config(raw, cli):
    direction = raw.get('update_direction')
    if direction not in DIRECTIONS:
        raise ValueError('Expected history_fused or future_only')
    expected_w = .5 if direction == 'history_fused' else 0.
    if raw.get('history_weight') != expected_w:
        raise ValueError('History weights are frozen to .5 and 0')
    if raw.get('response_lambda_mode') != 'sgd_prediction_distance' or raw.get('history_reset') != 'task':
        raise ValueError('Predictor-distance response and per-task history reset required')
    if raw.get('post_update_loss') is not True or raw.get('strict_deterministic') is not True:
        raise ValueError('Passive loss diagnostic and strict deterministic kernels required')
    seed = raw.get('seed')
    if seed not in (2, [2]):
        raise ValueError('First round is the fixed seed2 comparison')
    fixed = {'lora_rank':10,'gam_grad_rho':.05,'gam_grad_norm_rho':1.,'init_epoch':50,'epochs':50,
             'batch_size':128,'init_lr':.04,'lrate':.02,'init_momentum':0,'momentum':0,
             'init_weight_decay':0.,'weight_decay':0.,'ewc_lambda':0.,'ewc_skip_task_fisher':True,
             'approximate_fisher_in_task':False,'in_task_weight_merge':False,'task_endpoint_weight_merge':False}
    for k,v in fixed.items():
        if raw.get(k) != v:
            raise ValueError(f'Frozen setting mismatch: {k}')
    surrogate = copy.deepcopy(raw)
    surrogate.update(component_rule='u0', component_scale='natural')
    cfg = native.resolve_config(surrogate, cli)
    cfg.update(update_direction=direction, component_rule=direction, component_scale='natural',
               optimizer_tag_override=direction+'_two_gradient_v1', history_weight=expected_w,
               response_lambda_mode='sgd_prediction_distance', history_reset='task',
               history_min_distance=1e-12, post_update_loss=True, strict_deterministic=True,
               protocol=PROTOCOL, compute_nme_class_means=False, probe_gradient_count=2,
               compute_g3_probe=False, gam_radii_active=False, reference_outer_radius=1.,
               reference_inner_radius=.05, prediction_distance='current_lr_times_global_gradient_norm',
               in_task_path_normalize_snapshots=False, attention_backend='native')
    return cfg

class Controller(native.Controller):
    def __init__(self, learner, cfg, cli):
        super().__init__(learner,cfg,cli)
        self.direction=cfg['update_direction']
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
            self.stepper=HistoryResponseStep(model,params,self.geometry,
                dict(direction=self.direction,history_weight=self.cfg['history_weight'],eps=1e-12,backend='native'))
            self.previous_identity=None
            old_count=20*self.task
            fc=model.fc.state_dict()
            if not all(torch.equal(fc[k][:old_count],v.to(fc[k])) for k,v in self.old_fc.items()):
                raise AssertionError('Expanded classifier modified existing rows')
            identity=dict(task=self.task,seed=self.seed,update_direction=self.direction,
                          class_slice=[old_count,old_count+20],fc_shape=list(model.fc.weight.shape),
                          old_fc_rows_inherited_exact=True,history_reset=True,
                          new_fc_rows_sha256={k:audit.tensor_digest(v[old_count:]) for k,v in fc.items()},
                          task_initial_parameter_sha256=base.parameter_hash(model),
                          frozen_origin_sha256=base.parameter_hash(model,True),
                          data_seed_rule='seed*100000+task*1000+epoch',
                          forward_rng_rule='R0 replay; one g0/R1 advance; passive diagnostic restored')
            self.task_identities.append(identity)
            audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            return self.optimizer
        l._build_optimizer=MethodType(build,l)

        def batch(_self,optimizer,inputs,targets,class_offset):
            if class_offset!=20*self.task or bool(((targets<class_offset)|(targets>=class_offset+20)).any()):
                raise AssertionError('Incorrect current-task CE class slice')
            model=l._unwrap_network()
            def closure(x,y):
                logits=model(x)['logits']
                loss,_,_=l._task_loss(logits,y,class_offset)
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
            if forward_count[0]!=passes+1 or passes not in (1,2) or row['sgd_updates']!=1:
                raise AssertionError('Incorrect two-probe / diagnostic / one-SGD budget')
            if passes!=2 and not row.get('fallback_reason'):
                raise AssertionError('Missing explicit degenerate-predictor fallback reason')
            if self.previous_identity is None and row['history_used']:
                raise AssertionError('History cannot cross a task expansion')
            if self.direction=='future_only' and row['history_used']:
                raise AssertionError('No-history baseline used history')
            self.history_used_steps+=int(row['history_used'])
            prev=self.previous_identity
            current_ids=set(self.batch_identity['indices'])
            previous_ids=set(prev['indices']) if prev else set()
            row.update(step_index=self.steps,task=self.task,epoch=self.epoch,batch=self.batch,
                       task_step_index=self.steps-self.task_start_step,seed=self.seed,
                       identity=copy.deepcopy(self.batch_identity),previous_batch_identity=prev,
                       batch_overlap_count=len(current_ids&previous_ids) if prev else None,
                       class_slice=[class_offset,class_offset+20],seen_classes=l._total_classes,
                       fc_shape=list(model.fc.weight.shape),update_direction=self.direction,
                       training_ce_gradient_passes=passes,training_forward_count=passes,
                       diagnostic_loss_forward_count=1,actual_model_forward_count=forward_count[0],
                       diagnostic_state_restoration=diagnostic_restore,
                       train_ce=result['loss0'],post_update_ce=loss_after,
                       post_update_delta_ce=loss_after-result['loss0'],nominal_lr=lr,
                       strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                       history_source='previous_minibatch_raw_gradient',
                       gam_radii_active=False,reference_outer_radius=1.,reference_inner_radius=.05)
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
            self.correct+=int(result['logits0'][:,class_offset:class_offset+20].argmax(-1).eq(targets-class_offset).sum())
            self.epoch_batches+=1
            return result['logits0'],result['loss0'],dict(ce_clean=result['loss0'],gam=result['loss0'],ewc=0.,
                delta_w_penalty=0.,lambda_ewc=0.,ewc_gam_ratio=0.,delta_w_penalty_gam_ratio=0.)
        l._step_batch_with_log=MethodType(batch,l)

        original_run=l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            if self.task!=l._cur_task or class_offset!=20*self.task:
                raise AssertionError('Native lifecycle task mismatch')
            if scheduler.T_max!=50 or not math.isclose(scheduler.eta_min,(.04 if self.task==0 else .02)*.1):
                raise AssertionError('Base epoch scheduler changed')
            self.scheduler=scheduler;self.task_dataset_size=len(train_loader.dataset)
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
            self.matrix.append(native.seen_accuracy_row(cnn,self.task,l._total_classes,len(self.matrix)))
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

    def save_task_checkpoint(self):
        super().save_task_checkpoint()
        directory=Path(self.checkpoints[-1])
        state_path=directory/'training_state.pt'
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if hasattr(self.stepper,'state_dict'):
            cache=self.stepper.state_dict()
        else:
            cache={'previous_gradient':getattr(self.stepper,'previous_gradient',None),
                   'previous_displacement':getattr(self.stepper,'previous_displacement',None),
                   'history_valid':getattr(self.stepper,'history_valid',False)}
        state['algorithm_state']=base.cpu_tree(cache)
        state['previous_batch_identity']=copy.deepcopy(self.previous_identity)
        state['algorithm']=self.direction
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
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,direction=cfg['update_direction'],seed=2,
                             gradients_per_normal_step=2,updates_per_step=1,diagnostic_forwards_per_step=1,
                             history_weight=cfg['history_weight'],prediction_distance=cfg['prediction_distance'],
                             gam_radii_active=False,tasks=cli.max_tasks or 10,epochs=cli.max_epochs or 50)))
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
    summary=dict(status='running',protocol=PROTOCOL,seed=2,update_direction=cfg['update_direction'],
                 history_weight=cfg['history_weight'],response_lambda_mode=cfg['response_lambda_mode'],
                 prediction_distance=cfg['prediction_distance'],gam_radii_active=False,r=None,rho=None,
                 reference_outer_radius=1.,reference_inner_radius=.05,
                 gradients_per_normal_step=2,diagnostic_forwards_per_step=1,updates_per_step=1,
                 smoke=bool(cli.max_tasks or cli.max_epochs or cli.max_batches),
                 requested_tasks=cli.max_tasks or 10,epochs_per_task=cli.max_epochs or 50,
                 scheduler_cadence='epoch',scheduler_T_max=50,completed_tasks=0,
                 input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
                 algorithm_sha256=audit.digest(Path(__file__).with_name('history_step.py')),source=cli.source)
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        trainer.train(cfg);c=holder['controller'];n=cli.max_tasks or 10
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
                       initial_identity=c.task_identities[0],history_used_steps=c.history_used_steps,
                       elapsed_seconds=time.time()-c.started,
                       checks=dict(finite=True,all_steps_self_spec=True,exact_snapshot_restore=True,
                                   exact_actual_displacement_cache=True,raw_gradient_cache=True,
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
