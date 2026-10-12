#!/usr/bin/env python3
"""Shared complete SGD Task0 parent and eight upward-side Task1 forks."""
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
from upward_residual_step import UpwardResidualStep, DIRECTIONS, READS, FORMULAS
from state_transaction import State, _cache_name
from probe_study_data import build_data_manager, build_dev_loader, split_fingerprint

def _rng_hash(state):
    h=hashlib.sha256();h.update(repr(state[0]).encode());h.update(str(state[1][0]).encode());h.update(state[1][1].tobytes());h.update(repr(state[1][2:]).encode());h.update(state[2].cpu().numpy().tobytes())
    for v in state[3]:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

PROTOCOL='upward_eight_shared_sgd_parent_exact_restore_native_skip_v1'
PARENT_PROTOCOLS={PROTOCOL,'g2_shared_sgd_parent_train_dev_holdout_native_task1_fork_v1'}
SNAPSHOT_EPOCHS={0,1,2,5,10,20,30,40,50}

def study_metadata(cfg):
    direction=cfg['upward_direction']
    return dict(study_condition_id=cfg['study_condition_id'],study_role=cfg['study_role'],
        upward_direction=direction,probe_variant='upward_'+direction,fork_method=direction,
        response_probe_variant='upward_'+direction,radius_variant='upward_fixed_l2',
        probe_mode='upward_fixed_l2',inner_probe_radius_mode='fixed_l2',outer_probe_radius_mode='fixed_l2',
        inner_rho=cfg['gam_grad_rho'],outer_R=cfg['gam_grad_norm_rho'],
        normal_gradient_calls=READS[direction],gradient_formula=FORMULAS[direction],
        native_probe_skipping=True,gam_radii_active=True,
        probe_radius_coordinates='global trainable LoRA+FC parameter L2',
        dev_split_sha256=audit.digest(cfg['study_dev_split_path']))

def make_study_stepper(model,params,geometry,cfg):
    stepper=UpwardResidualStep(model,params,geometry,dict(direction=cfg['upward_direction'],
        rho=cfg['gam_grad_rho'],radius=cfg['gam_grad_norm_rho'],eps=1e-12,backend='native'))
    original_compute=stepper.compute
    original_apply=stepper.apply_update
    def compute(*a,**kw):
        if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();result=original_compute(*a,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        result['record']['native_probe_compute_seconds']=time.perf_counter()-started
        result['record']['native_training_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        return result
    def apply(*a,**kw):
        if torch.cuda.is_available():torch.cuda.synchronize()
        started=time.perf_counter();result=original_apply(*a,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        result['native_sgd_commit_seconds']=time.perf_counter()-started
        if torch.cuda.is_available():
            result['native_training_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated())
        return result
    stepper.compute=compute;stepper.apply_update=apply
    return stepper

inherited.make_stepper=make_study_stepper
inherited.radius_metadata=study_metadata

def resolve_config(raw,cli):
    role=raw.get('study_role');direction=raw.get('upward_direction')
    if role not in ('parent','branch') or direction not in DIRECTIONS:
        raise ValueError('Registered upward direction and parent/branch role required')
    if role=='parent' and direction!='g0':raise ValueError('Shared parent must use SGD')
    rawseed=raw.get('seed');seed=rawseed[0] if isinstance(rawseed,list) and len(rawseed)==1 else rawseed
    if seed not in (0,1,2) or isinstance(seed,bool):raise ValueError('Registered seeds are 0,1,2')
    if raw.get('gam_grad_rho')!=.05 or raw.get('gam_grad_norm_rho') not in (.25,1.):
        raise ValueError('Registered upward radii are rho=.05 and r=.25/1')
    if raw.get('gam_grad_norm_rho')==.25 and direction!='g2':
        raise ValueError('r=.25 is the direct-g2 bridge only')
    split=Path(raw.get('study_dev_split_path','')).resolve()
    if not split.is_file():raise FileNotFoundError('Registered train-only dev split is required')
    if cli.max_epochs not in (0,1) or cli.max_batches not in (0,2) or bool(cli.max_epochs)!=bool(cli.max_batches):
        raise ValueError('Only full budget or 1 epoch / 2 batch smoke is registered')
    method='sgd' if direction=='g0' else 'current_g2'
    clean=copy.deepcopy(raw)
    for key in inherited.PROBE_CONFIG_KEYS+('response_probe_variant','radius_variant'):
        clean.pop(key,None)
    clean.update(seed=2,gam_grad_norm_rho=1.,response_radius_mode='sgd_step',radius_scale=1.,
        history_weight=0.,history_source=None,response_method=method,response_layout='cl10',
        history_reset='task',post_update_loss=True,strict_deterministic=True)
    surrogate=copy.copy(cli);surrogate.max_tasks=0
    cfg=inherited.resolve_config(clean,surrogate)
    cfg.update(study_condition_id='parent_sgd' if role=='parent' else 'upward_'+direction,
        study_role=role,upward_direction=direction,study_dev_split_path=str(split),
        study_parent_output_dir=raw.get('study_parent_output_dir'),study_parent_checkpoint=raw.get('study_parent_checkpoint'),
        study_parent_receipt=raw.get('study_parent_receipt'),protocol=PROTOCOL,max_train_tasks=1 if role=='parent' else 2,
        optimizer_tag_override='upward_'+direction,seed=seed,device=[0],
        gam_grad_rho=.05,gam_grad_norm_rho=raw['gam_grad_norm_rho'],probe_gradient_count=READS[direction],
        compute_g3_probe=READS[direction]==4,gam_radii_active=True,
        resume_require_experiment_identity=False,resume_require_model_sidecars=False,resume_require_producer_receipt=False)
    inherited.NORMAL_GRADIENTS[method]=READS[direction]
    if role=='branch' and not all(cfg.get(k) for k in ('study_parent_output_dir','study_parent_checkpoint','study_parent_receipt')):
        raise ValueError('Branch requires accepted complete parent paths')
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
        self.retention_curve=[];self.pre_expand_baseline=None;self.post_expand_baseline=None
        self.native_compute_seconds=0.;self.native_commit_seconds=0.;self.passive_eval_seconds=0.;self.snapshot_seconds=0.;self.training_peak_bytes=0
    def install(self):
        super().install();l=self.learner
        old_build=l._build_optimizer
        def build(_self,params,stage):
            result=old_build(params,stage)
            if self.cfg['study_role']=='branch' and self.task==1:
                self.post_expand_baseline=self.evaluate_task1_curve(0)
                audit.atomic_json(self.out/'fork_baselines.json',dict(pre_expand=self.pre_expand_baseline,post_expand=self.post_expand_baseline))
            self.task_identities[-1].update(parent_receipt=copy.deepcopy(self.parent_receipt))
            audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            return result
        l._build_optimizer=MethodType(build,l)
        original_run=l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            # Epoch0 is saved after the optimizer AND scheduler exist, before
            # any minibatch iterator/probe has advanced the training state.
            self.scheduler=scheduler
            if self.cfg['study_role']=='branch' and self.task==1:
                self.save_snapshot(0)
            return original_run(train_loader,test_loader,optimizer,scheduler,epochs=epochs,class_offset=class_offset)
        l._run_regularized_train=MethodType(run,l)
        # Passive train-only development evaluation at every completed epoch.
        # Its deterministic native test transform never influences the update RNG.
        original_batch=l._step_batch_with_log
        def batch(_self,*a,**kw):
            result=original_batch(*a,**kw)
            self.native_compute_seconds+=self.last_record['native_probe_compute_seconds']
            self.native_commit_seconds+=self.last_record['native_sgd_commit_seconds']
            self.training_peak_bytes=max(self.training_peak_bytes,self.last_record['native_training_peak_allocated_bytes'])
            return result
        l._step_batch_with_log=MethodType(batch,l)
    def save_snapshot(self,epoch):
        started=time.perf_counter()
        model=self.learner._unwrap_network()
        payload=_complete_snapshot(model,self.cfg,self.task,epoch,float(self.optimizer.param_groups[0]['lr']),
             dict(parent_receipt=self.parent_receipt,initial_identity=self.task_identities[-1],split_receipt=split_fingerprint(self.dm),optimizer_state=base.cpu_tree(self.optimizer.state_dict()),scheduler_state=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}) if hasattr(self,'scheduler') else None,pre_expand_baseline=self.pre_expand_baseline,post_expand_baseline=self.post_expand_baseline))
        directory=self.out/'snapshots';directory.mkdir(exist_ok=True)
        path=directory/('epoch_%03d.pt'%epoch)
        if path.exists():raise FileExistsError('Snapshot is immutable')
        torch.save(payload,path)
        row={k:v for k,v in payload.items() if k not in ('model','rng','config','parameter_gradients','requires_grad_by_name','module_modes')}
        row.update(path=str(path),sha256=audit.digest(path))
        self.snapshot_seconds+=time.perf_counter()-started
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

    def finish_epoch(self):
        inherited.Controller.finish_epoch(self)
        started=time.perf_counter();row=self.evaluate_dev();self.passive_eval_seconds+=time.perf_counter()-started
        self.dev_curve.append(row)
        audit.atomic_json(self.out/'dev_curve.json',dict(epochs=self.dev_curve,selection_rule='fixed epoch50 endpoint current-task held-out train CE'))
        if self.cfg['study_role']=='branch' and self.task==1:
            curve=self.evaluate_task1_curve(self.epoch);curve.update(current_dev_ce=row['dev_current_task_ce'],current_dev_accuracy=row['dev_current_task_accuracy'])
            self.retention_curve.append(curve)
            audit.atomic_json(self.out/'retention_curve.json',dict(epochs=self.retention_curve,
                definition='epoch-wise old/current test classified over all seen40; not stage AAA',
                retention_auc=sum(x['old_test_allseen40']['accuracy'] for x in self.retention_curve)/len(self.retention_curve),
                retention_auc_definition='uniform arithmetic average of all completed epoch old-allseen40 accuracies'))
            if self.epoch in SNAPSHOT_EPOCHS or self.cli.max_epochs:
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
        if (pcfg.get('protocol') not in PARENT_PROTOCOLS or pcfg.get('seed')!=cfg['seed'] or pcfg.get('lora_rank')!=10 or
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
        self.pre_expand_baseline=self.evaluate_test_group(0,20,old_local=True)
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
    set_random(cfg['seed']);set_device(cfg)
    dm=build_data_manager(cli.source,cfg,cfg['study_dev_split_path'])
    learner=factory.get_model(cfg['model_name'],cfg)
    learner._network.backbone.to(learner._device)
    c=Controller(learner,cfg,cli);c.dm=dm;c.install()
    audit.atomic_json(out/'effective_config.json',json_safe(cfg));audit.atomic_json(out/'study_split_receipt.json',split_fingerprint(dm))
    summary=dict(status='running',protocol=PROTOCOL,seed=cfg['seed'],response_method=cfg['response_method'],response_layout='cl10',
        smoke=bool(cli.max_epochs),training_tasks_this_job=1,full_partition_tasks=10,total_dataset_classes=200,
        task_boundaries=[[20*i,20*(i+1)] for i in range(10)],requested_task=0 if cfg['study_role']=='parent' else 1,
        epochs_per_task=cli.max_epochs or 50,input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
        algorithm_sha256=audit.digest(Path(__file__).with_name('upward_residual_step.py')),
        inherited_algorithm_sha256=audit.digest(Path(__file__).with_name('response_step.py')),
        source=cli.source,history_source=None,history_weight=0.,completed_tasks=0,
        r=cfg['gam_grad_norm_rho'],rho=cfg['gam_grad_rho'],response_radius_mode=cfg['response_radius_mode'],radius_scale=1.,trained_task_ids=[],**study_metadata(cfg))
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
            fork_baselines=dict(pre_expand=c.pre_expand_baseline,post_expand=c.post_expand_baseline),
            retention_curve=copy.deepcopy(c.retention_curve),
            old_training_delta=(c.retention_curve[-1]['old_test_allseen40']['accuracy']-c.post_expand_baseline['old_test_allseen40']['accuracy']) if c.retention_curve else None,
            retention_auc=sum(x['old_test_allseen40']['accuracy'] for x in c.retention_curve)/len(c.retention_curve) if c.retention_curve else None,
            native_probe_compute_seconds=c.native_compute_seconds,native_sgd_commit_seconds=c.native_commit_seconds,
            native_training_compute_seconds=c.native_compute_seconds+c.native_commit_seconds,
            passive_extra_eval_seconds=c.passive_eval_seconds,snapshot_save_seconds=c.snapshot_seconds,
            native_training_peak_allocated_bytes=c.training_peak_bytes,
            compute_time_definition='CUDA synchronized probe compute plus one real SGD commit; passive diagnostics/evaluations/snapshot saves excluded',
            final_model_parameter_sha256=base.parameter_hash(learner._unwrap_network()),
            checks=dict(finite=True,exact_snapshot_restore=True,one_sgd_per_batch=True,no_history=True,
                fisher_empty=True,actual_weight_merges_zero=True,train_dev_disjoint=True,test_not_used_for_selection=True,
                shared_parent_exact=cfg['study_role']=='parent' or c.parent_receipt['parameter_identity_verified']))
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_steps=c.steps);raise
    finally:
        c.handle.close();audit.atomic_json(out/'final_summary.json',summary)

if __name__=='__main__':main()
