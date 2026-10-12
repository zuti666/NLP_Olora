#!/usr/bin/env python3
"""Shared fresh SGD Task0 parent and exactly paired Task1 G2 study forks."""
import argparse
import copy
import json
import hashlib
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import audit_core as audit
import task0_trainer_base_v2 as base
import response_train as inherited
from response_step import ResponseStep, _norm, _dot
from independent_response_step import IndependentResponseStep
from state_transaction import State, _cache_name
from probe_study_data import build_data_manager, build_dev_loader, split_fingerprint

def _rng_hash(state):
    h=hashlib.sha256();h.update(repr(state[0]).encode());h.update(str(state[1][0]).encode());h.update(state[1][1].tobytes());h.update(repr(state[1][2:]).encode());h.update(state[2].cpu().numpy().tobytes())
    for v in state[3]:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

PROTOCOL='g2_shared_sgd_parent_train_dev_holdout_native_task1_fork_v1'
CONDITIONS={
    'parent_sgd':('sgd',None,None,None,None),
    'B0':('sgd',None,None,None,None),
    'R25':('current_g2','fixed_l2',.05,'fixed_l2',.25),
    'S_F':('current_g2','sgd_step',None,'fixed_l2',.25),
    'F_S':('current_g2','fixed_l2',.05,'sgd_step',None),
    'S_S':('current_g2','sgd_step',None,'sgd_step',None),
    'R100':('current_g2','fixed_l2',.05,'fixed_l2',1.),
}

def study_metadata(cfg):
    _,im,ir,om,orr=CONDITIONS[cfg['study_condition_id']]
    return dict(study_condition_id=cfg['study_condition_id'],study_role=cfg['study_role'],probe_variant=cfg['study_condition_id'],fork_method=cfg['study_condition_id'],
        response_probe_variant=cfg['study_condition_id'],radius_variant=cfg['radius_variant'],
        probe_mode=cfg['response_radius_mode'],inner_probe_radius_mode=im,outer_probe_radius_mode=om,
        inner_rho=ir,outer_R=orr,normal_gradient_calls=1 if cfg['response_method']=='sgd' else 3,
        gam_radii_active=False,probe_radius_coordinates='global trainable LoRA+FC parameter L2',
        dev_split_sha256=audit.digest(cfg['study_dev_split_path']))

def make_study_stepper(model,params,geometry,cfg):
    if cfg['response_method']=='sgd':
        stepper=ResponseStep(model,params,geometry,dict(method='sgd',history_weight=0.,eps=1e-12,backend='native',radius_mode='sgd_step',radius_scale=1.))
    else:
        stepper=IndependentResponseStep(model,params,geometry,dict(method='current_g2',history_weight=0.,eps=1e-12,
            backend='native',radius_mode=inherited.INDEPENDENT_MODE,radius_scale=1.,radius_variant=cfg['radius_variant'],
            response_probe_variant=cfg['study_condition_id'],inner_probe_radius_mode=cfg['inner_probe_radius_mode'],
            outer_probe_radius_mode=cfg['outer_probe_radius_mode'],fixed_inner_probe_radius=cfg['inner_probe_radius'],
            fixed_outer_probe_radius=cfg['outer_probe_radius']))
    name_by_id={id(p):n for n,p in model.named_parameters()}
    original=stepper.compute
    def compute(*args,**kwargs):
        result=original(*args,**kwargs);r=result['record'];c=result['context'];g0=c['g0'];g2=c['g2']
        fc=[g for p,g in zip(stepper.params,g0) if name_by_id[id(p)].startswith('fc.')]
        lora=[g for p,g in zip(stepper.params,g0) if not name_by_id[id(p)].startswith('fc.')]
        fc_norm=_norm(fc) if fc else 0.;lora_norm=_norm(lora) if lora else 0.;g0norm=_norm(g0)
        h=r['h'];inner=r.get('inner_probe_radius_theoretical',h if cfg['response_method']!='sgd' else None)
        outer=r.get('outer_probe_radius_theoretical',h if cfg['response_method']!='sgd' else None)
        r.update(g0_fc_parameter_l2=fc_norm,g0_lora_parameter_l2=lora_norm,
            g0_fc_squared_norm_share=fc_norm**2/g0norm**2 if g0norm else None,
            g0_lora_squared_norm_share=lora_norm**2/g0norm**2 if g0norm else None,
            study_inner_over_h=inner/h if inner is not None and h>0 else None,
            study_outer_over_h=outer/h if outer is not None and h>0 else None,
            study_outer_over_inner=outer/inner if inner is not None and inner>0 else None,
            study_g0_dot_g2=_dot(g0,g2) if g2 is not None else None,
            study_g0_g2_cosine=_dot(g0,g2)/(g0norm*_norm(g2)) if g2 is not None and g0norm*_norm(g2)>0 else None)
        return result
    stepper.compute=compute
    return stepper

# The copied inherited module is private to this process. Its exact restore,
# native CE/head/data loop, natural SGD commit and forbidden-work guards remain.
inherited.make_stepper=make_study_stepper
inherited.radius_metadata=study_metadata

def resolve_config(raw,cli):
    condition=raw.get('study_condition_id');role=raw.get('study_role')
    if condition not in CONDITIONS or role not in ('parent','branch'):
        raise ValueError('Explicit registered study condition and parent/branch role required')
    if (condition=='parent_sgd')!=(role=='parent'):
        raise ValueError('Only parent_sgd can train the shared parent')
    method,im,ir,om,orr=CONDITIONS[condition]
    expected=dict(response_method=method,study_inner_mode=im,study_inner_radius=ir,study_outer_mode=om,study_outer_radius=orr)
    for key,value in expected.items():
        if key not in raw or raw[key]!=value:raise ValueError('Study field mismatch: '+key)
    split=Path(raw.get('study_dev_split_path','')).resolve()
    if not split.is_file():raise FileNotFoundError('Pre-registered train-only development split required')
    if cli.max_epochs not in (0,1) or cli.max_batches not in (0,2):
        raise ValueError('Only full 50-epoch or bounded 1-epoch /2-batch smoke budget supported')
    if bool(cli.max_epochs)!=bool(cli.max_batches):raise ValueError('Smoke epoch/batch caps must be supplied together')
    clean=copy.deepcopy(raw)
    for key in inherited.PROBE_CONFIG_KEYS+('response_probe_variant','radius_variant'):
        clean.pop(key,None)
    clean.update(response_radius_mode='sgd_step',radius_scale=1.,history_weight=0.,history_source=None,
                 response_method=method,response_layout='cl10',history_reset='task',post_update_loss=True,strict_deterministic=True)
    surrogate=copy.copy(cli);surrogate.max_tasks=0
    cfg=inherited.resolve_config(clean,surrogate)
    cfg.update(study_condition_id=condition,study_role=role,study_dev_split_path=str(split),
               study_parent_output_dir=raw.get('study_parent_output_dir'),study_parent_checkpoint=raw.get('study_parent_checkpoint'),
               study_parent_receipt=raw.get('study_parent_receipt'),protocol=PROTOCOL,max_train_tasks=1 if role=='parent' else 2,
               optimizer_tag_override='study_'+condition,seed=2,device=[0],
               resume_require_experiment_identity=False,resume_require_model_sidecars=False,
               resume_require_producer_receipt=False)
    if method=='current_g2':
        mask=(im=='fixed_l2',om=='fixed_l2')
        short={(False,False):'coupled_reference',(True,False):'fixed_inner',(False,True):'fixed_outer',(True,True):'fixed_both'}[mask]
        cfg.update(response_radius_mode=inherited.INDEPENDENT_MODE,radius_variant=short,
                   inner_probe_radius_mode=im,inner_probe_radius=ir,outer_probe_radius_mode=om,outer_probe_radius=orr,
                   response_probe_variant=condition)
    else:cfg.update(radius_variant='sgd_direction')
    cfg.update(study_inner_mode=im,study_inner_radius=ir,study_outer_mode=om,study_outer_radius=orr)
    if role=='branch' and not all(cfg.get(k) for k in ('study_parent_output_dir','study_parent_checkpoint','study_parent_receipt')):
        raise ValueError('Branch requires explicit parent output, checkpoint, accepted completion receipt')
    return cfg

def _runtime_device(value,device):
    if isinstance(value,torch.nn.Parameter):return value
    if torch.is_tensor(value):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:_runtime_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [_runtime_device(v,device) for v in value]
    if isinstance(value,tuple):return tuple(_runtime_device(v,device) for v in value)
    return value

def _move_caches(model,device):
    for module in model.modules():
        for name,value in list(vars(module).items()):
            if _cache_name(name):setattr(module,name,_runtime_device(value,device))

def _clone_complete_model(model):
    result=copy.deepcopy(model).cpu()
    # Parameter deepcopy does not copy .grad. Preserve even nontrainable gradients.
    for source,target in zip(model.parameters(),result.parameters()):
        target.grad=None if source.grad is None else source.grad.detach().cpu().clone()
    _move_caches(result,'cpu')
    return result

def _complete_snapshot(model,cfg,task,epoch,current_lr,extra=None):
    entry=State(model)
    try:
        payload=dict(schema='G2StudySnapshot_v1',model=_clone_complete_model(model),config=copy.deepcopy(cfg),
                     rng=base.cpu_tree(audit.rng_get()),
                     parameter_gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()},
                     requires_grad_by_name={n:p.requires_grad for n,p in model.named_parameters()},
                     module_modes={n:m.training for n,m in model.named_modules()},task=task,class_slice=[20*task,20*(task+1)],epoch=epoch,
                     epoch_completed=epoch,current_optimizer_lr=current_lr,
                     snapshot_location='after epoch scheduler; before next epoch' if epoch else 'expanded head; before first Task1 update',
                     full_model_parameter_sha256=base.parameter_hash(model),frozen_origin_sha256=base.parameter_hash(model,True),
                     dev_split_sha256=audit.digest(cfg['study_dev_split_path']),source=cfg['study_source'],
                     trainer_sha256=audit.digest(__file__),data_seed_rule='seed*100000+task*1000+epoch')
        if extra:payload.update(copy.deepcopy(extra))
        return payload
    finally:entry.restore();entry.check()

def load_snapshot(path,device='cuda:0',restore_rng=True):
    # Callers insert the frozen source path before unpickling native modules.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='G2StudySnapshot_v1':raise ValueError('Unsupported complete snapshot schema')
    model=payload['model'].to(device);_move_caches(model,device)
    if set(payload['parameter_gradients'])!={n for n,p in model.named_parameters()}:raise AssertionError('Snapshot gradient inventory changed')
    for name,p in model.named_parameters():
        if p.requires_grad!=payload['requires_grad_by_name'][name]:raise AssertionError('Snapshot trainable scope changed')
        grad=payload['parameter_gradients'][name];p.grad=None if grad is None else grad.to(p).clone()
    for name,module in model.named_modules():module.training=payload['module_modes'][name]
    if base.parameter_hash(model)!=payload['full_model_parameter_sha256']:
        raise AssertionError('Complete model snapshot parameter identity mismatch')
    if base.parameter_hash(model,True)!=payload['frozen_origin_sha256']:
        raise AssertionError('Complete frozen origin snapshot identity mismatch')
    if restore_rng:audit.rng_set(payload['rng'])
    return model,payload

class Controller(inherited.Controller):
    def __init__(self,learner,cfg,cli):
        super().__init__(learner,cfg,cli)
        self.dev_curve=[];self.snapshot_rows=[];self.parent_receipt=None
    def install(self):
        super().install();l=self.learner
        old_build=l._build_optimizer
        def build(_self,params,stage):
            result=old_build(params,stage)
            if self.cfg['study_condition_id']=='B0' and self.task==1:
                self.save_snapshot(0)
            self.task_identities[-1].update(parent_receipt=copy.deepcopy(self.parent_receipt))
            audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            return result
        l._build_optimizer=MethodType(build,l)
        # Passive train-only development evaluation at every completed epoch.
        # Its deterministic native test transform never influences the update RNG.
        original_batch=l._step_batch_with_log
        def batch(_self,*a,**kw):
            result=original_batch(*a,**kw)
            return result
        l._step_batch_with_log=MethodType(batch,l)
    def save_snapshot(self,epoch):
        model=self.learner._unwrap_network()
        payload=_complete_snapshot(model,self.cfg,self.task,epoch,float(self.optimizer.param_groups[0]['lr']),
             dict(parent_receipt=self.parent_receipt,initial_identity=self.task_identities[-1],split_receipt=split_fingerprint(self.dm)))
        directory=self.out/'snapshots';directory.mkdir(exist_ok=True)
        path=directory/('epoch_%03d.pt'%epoch)
        if path.exists():raise FileExistsError('Snapshot is immutable')
        torch.save(payload,path)
        row={k:v for k,v in payload.items() if k not in ('model','rng','config','parameter_gradients','requires_grad_by_name','module_modes')}
        row.update(path=str(path),sha256=audit.digest(path))
        self.snapshot_rows.append(row);audit.atomic_json(self.out/'snapshots.json',dict(schema='G2StudySnapshotManifest_v1',snapshots=self.snapshot_rows))
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
                    dev_split_sha256=self.dm.study_split_sha256,selection_rule='fixed endpoint epoch50 CE; no best-epoch selection',
                    test_used_for_selection=False,passive_state_and_rng_restored=True)
    def finish_epoch(self):
        super().finish_epoch()
        row=self.evaluate_dev();self.dev_curve.append(row)
        audit.atomic_json(self.out/'dev_curve.json',dict(epochs=self.dev_curve,selection_rule='fixed epoch50 endpoint current-task held-out-train CE'))
        if self.cfg['study_condition_id']=='B0' and self.task==1 and self.epoch in ({1} if self.cli.max_epochs else {1,10,25,49}):
            self.save_snapshot(self.epoch)
    def save_task_checkpoint(self):
        super().save_task_checkpoint()
        all_tasks={str(json.loads((Path(p)/'manifest.json').read_text())['task']):p for p in self.checkpoints}
        audit.atomic_json(self.out/'task_checkpoints.json',dict(loader='load_task_checkpoint(directory, task, data_manager)',all_tasks=all_tasks))

    def restore_parent(self,dm):
        cfg=self.cfg;parent=Path(cfg['study_parent_output_dir']).resolve();checkpoint=Path(cfg['study_parent_checkpoint']).resolve()
        receipt_path=Path(cfg['study_parent_receipt']).resolve()
        receipt=json.loads(receipt_path.read_text())
        summary=json.loads((parent/'final_summary.json').read_text())
        expected_summary=audit.digest(parent/'final_summary.json')
        pinned=receipt.get('summary_sha256') or receipt.get('validation',{}).get('summary_sha256')
        if pinned!=expected_summary:raise AssertionError('Parent completion receipt does not pin the completed summary')
        if summary['status']!='completed' or summary['study_role']!='parent' or summary['study_condition_id']!='parent_sgd':
            raise AssertionError('Branch parent is not the accepted fresh SGD Task0')
        if bool(summary['smoke'])!=bool(self.cli.max_epochs):raise AssertionError('Smoke/formal parent mixing is forbidden')
        if summary['dev_split_sha256']!=dm.study_split_sha256:raise AssertionError('Parent/branch dev split mismatch')
        if str(checkpoint) not in summary.get('checkpoint_dirs',[]):raise AssertionError('Accepted parent summary does not bind this checkpoint directory')
        if receipt.get('exit',{}).get('returncode')!=0:raise AssertionError('Parent did not receive a successful worker completion')
        manifest=json.loads((checkpoint/'manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            if audit.digest(checkpoint/name)!=digest:raise AssertionError('Parent checkpoint file changed: '+name)
        if 'shared_parent.pt' not in manifest['files_sha256']:raise AssertionError('Complete shared parent absent')
        l=self.learner;l.load_task_checkpoint(str(checkpoint),0,dm)
        model,payload=load_snapshot(checkpoint/'shared_parent.pt',str(l._device),restore_rng=False)
        if payload['task']!=0 or model.fc.weight.shape[0]!=20:raise AssertionError('Shared parent task/head differs')
        pcfg=payload['config']
        if (pcfg.get('protocol')!=PROTOCOL or pcfg.get('seed')!=2 or pcfg.get('lora_rank')!=10 or
            pcfg.get('study_condition_id')!='parent_sgd' or pcfg.get('study_role')!='parent' or
            payload['dev_split_sha256']!=dm.study_split_sha256):raise AssertionError('Shared parent protocol/seed/rank/split differs')
        if payload['full_model_parameter_sha256']!=manifest['model_sha256'] or payload['full_model_parameter_sha256']!=summary['final_model_parameter_sha256']:
            raise AssertionError('Shared parent does not match accepted final model identity')
        if payload['frozen_origin_sha256']!=summary['initial_identity']['frozen_origin_sha256']:
            raise AssertionError('Shared parent frozen origin changed')
        l._network=model;l._cur_task=0;l._known_classes=20;l._total_classes=20
        inherited.native.tensor_state_empty(l)
        # Parent stage metrics are inherited only as a verified evaluation prefix.
        if audit.digest(parent/'task_matrix.json')!=summary['task_matrix_sha256']:raise AssertionError('Accepted parent evaluation prefix changed')
        matrix=json.loads((parent/'task_matrix.json').read_text())
        if len(matrix['matrix'])!=1:raise AssertionError('Shared parent contains unexpected trained tasks')
        self.matrix=copy.deepcopy(matrix['matrix']);self.sample_curve=copy.deepcopy(matrix['sample_accuracy_curve'])
        self.dm=dm
        self.parent_receipt=dict(parent_output_dir=str(parent),parent_completion_receipt=str(receipt_path),
            parent_completion_receipt_sha256=audit.digest(receipt_path),parent_summary_sha256=expected_summary,
            parent_checkpoint_manifest_sha256=audit.digest(checkpoint/'manifest.json'),
            shared_parent_sha256=audit.digest(checkpoint/'shared_parent.pt'),
            parent_model_sha256=payload['full_model_parameter_sha256'],frozen_origin_sha256=payload['frozen_origin_sha256'],
            parent_rng_sha256=_rng_hash(payload['rng']),complete_network_restored=True,
            native_checkpoint_loader_used=True,parameter_identity_verified=True,parent_test_prefix_inherited_exact=True)
        audit.atomic_json(self.out/'parent_restore_receipt.json',self.parent_receipt)
        audit.rng_set(payload['rng'])
        if not audit.rng_equal(audit.rng_get(),payload['rng']):raise AssertionError('Parent RNG restore failed')
        return self.parent_receipt

    def save_complete_parent(self):
        l=self.learner;checkpoint=Path(self.checkpoints[-1])
        payload=_complete_snapshot(l._unwrap_network(),self.cfg,0,self.cli.max_epochs or 50,
                  float(self.optimizer.param_groups[0]['lr']),dict(parent_after_task_completed=True,known_classes=20,
                       initial_identity=self.task_identities[0],split_receipt=split_fingerprint(self.dm)))
        torch.save(payload,checkpoint/'shared_parent.pt')
        manifest=json.loads((checkpoint/'manifest.json').read_text())
        manifest['files_sha256']['shared_parent.pt']=audit.digest(checkpoint/'shared_parent.pt')
        manifest['shared_parent_after_task_complete']=True
        manifest['shared_parent_rng_sha256']=_rng_hash(payload['rng'])
        audit.atomic_json(checkpoint/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','config','output-dir'):parser.add_argument('--'+key,required=True)
    for key in ('max-epochs','max-batches'):parser.add_argument('--'+key,type=int,default=0)
    parser.add_argument('--dry-run',action='store_true');cli=parser.parse_args();cli.max_tasks=0
    for key in ('source','config','output_dir'):setattr(cli,key,str(Path(getattr(cli,key)).resolve()))
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli);cfg['study_source']=cli.source
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,task=0 if cfg['study_role']=='parent' else 1,
            training_tasks_this_job=1,epochs=cli.max_epochs or 50,smoke=bool(cli.max_epochs),
            parent_retraining_per_branch=False,selection_rule='fixed epoch50 currenttask train-only dev CE',
            **study_metadata(cfg))));return
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Study output must be fresh')
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
    set_random(2);set_device(cfg)
    dm=build_data_manager(cli.source,cfg,cfg['study_dev_split_path'])
    learner=factory.get_model(cfg['model_name'],cfg)
    learner._network.backbone.to(learner._device)
    c=Controller(learner,cfg,cli);c.dm=dm;c.install()
    audit.atomic_json(out/'effective_config.json',json_safe(cfg));audit.atomic_json(out/'study_split_receipt.json',split_fingerprint(dm))
    summary=dict(status='running',protocol=PROTOCOL,seed=2,response_method=cfg['response_method'],response_layout='cl10',
        smoke=bool(cli.max_epochs),training_tasks_this_job=1,full_partition_tasks=10,total_dataset_classes=200,
        task_boundaries=[[20*i,20*(i+1)] for i in range(10)],requested_task=0 if cfg['study_role']=='parent' else 1,
        epochs_per_task=cli.max_epochs or 50,input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
        algorithm_sha256=audit.digest(Path(__file__).with_name('response_step.py' if cfg['response_method']=='sgd' else 'independent_response_step.py')),
        inherited_algorithm_sha256=audit.digest(Path(__file__).with_name('response_step.py')),
        source=cli.source,history_source=None,history_weight=0.,completed_tasks=0,
        r=None,rho=None,response_radius_mode=cfg['response_radius_mode'],radius_scale=1.,trained_task_ids=[],**study_metadata(cfg))
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        if cfg['study_role']=='branch':c.restore_parent(dm)
        learner.incremental_train(dm);learner.eval_task();learner.after_task()
        if cfg['study_role']=='parent':c.save_complete_parent()
        if len(c.task_rows)!=1 or len(c.after_task_checks)!=1 or len(c.rows)!=(cli.max_epochs or 50):
            raise AssertionError('Only one task of the correct epoch budget must be trained in this job')
        if len(c.matrix)!=(1 if cfg['study_role']=='parent' else 2):raise AssertionError('Incorrect shared-parent evaluation prefix')
        if any(c.counters[k] for k in ('forbidden_work_attempts','weight_merge_updates','fisher_estimation_passes')):
            raise AssertionError('Forbidden auxiliary training work detected')
        c.dataset_receipt.update(study_split=split_fingerprint(dm),actual_training_task=c.task)
        audit.atomic_json(out/'dataset_partition_receipt.json',c.dataset_receipt)
        final_metrics=json.loads((out/'task_matrix.json').read_text())
        summary.update(status='completed',completed_tasks=len(c.matrix),trained_task_ids=[c.task],completed_epochs=len(c.rows),completed_steps=c.steps,
            counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
            dataset_partition_receipt=c.dataset_receipt,initial_identity=c.task_identities[0],
            parent_restore_receipt=c.parent_receipt,parent_provenance=c.parent_receipt,history_used_steps=0,
            elapsed_seconds=time.time()-c.started,step_trace_sha256=audit.digest(c.trace),task_matrix_sha256=audit.digest(out/'task_matrix.json'),
            final_metrics=final_metrics,endpoint_dev=c.dev_curve[-1],
            final_model_parameter_sha256=base.parameter_hash(learner._unwrap_network()),
            checks=dict(finite=True,exact_snapshot_restore=True,one_sgd_per_batch=True,no_history=True,
                fisher_empty=True,actual_weight_merges_zero=True,train_dev_disjoint=True,test_not_used_for_selection=True,
                shared_parent_exact=cfg['study_role']=='parent' or c.parent_receipt['parameter_identity_verified']))
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_steps=c.steps);raise
    finally:
        c.handle.close();audit.atomic_json(out/'final_summary.json',summary)

if __name__=='__main__':main()
