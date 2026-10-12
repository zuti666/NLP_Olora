#!/usr/bin/env python3
"""Shared expanded Task1 state, paired native SAM/g2 then common SGD trajectories."""
import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType,SimpleNamespace
sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import audit_core as audit
import task0_trainer_base_v2 as base
import native_lifecycle as native
import response_train as inherited
import upward_fork_train as old
from state_transaction import State,_cache_name
from effective_step import EffectiveStep
from probe_study_data import build_data_manager,split_fingerprint
from chain_step import ChainStep,READS,FORMULAS,active_method,cosine_lr
from chain_statistics import factor_epoch,update_decomposition
PROTOCOL='response_state_chain_shared_task0_sam_or_sf25_then_sgd50_v1'
FULL_STATE_EPOCHS={0,1,10,25,50}

class PreparedStart(Exception):pass

def metadata(cfg):
    method=cfg.get('_active_chain_method','sam' if cfg['chain_branch']=='S' else 'sf_g2')
    return dict(chain_role=cfg['chain_role'],chain_branch=cfg['chain_branch'],active_method=method,
        chain_switch_after_epoch=cfg['chain_switch_after_epoch'],gradient_formula=FORMULAS[method],normal_gradient_calls=READS[method],
        history_source=None,history_weight=0.,gam_radii_active=method=='sam',native_probe_skipping=True,
        sam_rho=.05,sf_outer_radius=.25,sf_inner_mode='sgd_step',sf_inner_displacement='-eta*g0',
        probe_radius_coordinates='global trainable Q/V LoRA and full FC parameter L2',dev_split_sha256=audit.digest(cfg['study_dev_split_path']))

inherited.make_stepper=lambda model,params,geometry,cfg:ChainStep(model,params,geometry,cfg)
inherited.radius_metadata=metadata

def resolve_config(raw,cli):
    role=raw.get('chain_role');branch=raw.get('chain_branch');rawseed=raw.get('seed')
    seed=rawseed[0] if isinstance(rawseed,list) and len(rawseed)==1 else rawseed
    smoke=bool(raw.get('chain_smoke',False))
    if role not in ('start','branch') or branch not in ('S','R') or seed not in (0,1,2) or isinstance(seed,bool):raise ValueError('Registered rolesstart/branch, branchesS/R, seeds0/1/2 required')
    if (cli.max_epochs,cli.max_batches) not in ((0,0),(2,2)):raise ValueError('Formal50epochsfullbatches or smoke2epochs2batches only')
    if smoke!=bool(cli.max_epochs):raise ValueError('Configsmoke and CLIbudget must agree')
    switch=raw.get('chain_switch_after_epoch',25)
    if switch!=(1 if smoke else 25):raise ValueError('Switchafter25 formal or1smoke required')
    for key,value in dict(init_epoch=50,epochs=50,init_lr=.04,lrate=.02,batch_size=128,lora_rank=10,optimizer='sgd',scheduler='cosine',cosine_min_lr_ratio=.1,init_momentum=0,momentum=0,init_weight_decay=0.,weight_decay=0.,gam_grad_rho=.05,gam_grad_norm_rho=1.,ewc_lambda=0.,ewc_skip_task_fisher=True,approximate_fisher_in_task=False,in_task_weight_merge=False,task_endpoint_weight_merge=False).items():
        if raw.get(key)!=value:raise ValueError('Frozen settingmismatch:'+key)
    split=Path(raw.get('study_dev_split_path','')).resolve()
    if not split.is_file():raise FileNotFoundError('Pinned splitrequired')
    clean=copy.deepcopy(raw)
    for key in inherited.PROBE_CONFIG_KEYS+('response_probe_variant','radius_variant'):clean.pop(key,None)
    clean.update(seed=2,response_method='sgd',response_layout='cl10',response_radius_mode='sgd_step',radius_scale=1.,history_weight=0.,history_source=None,history_reset='task',post_update_loss=True,strict_deterministic=True)
    cfg=inherited.resolve_config(clean,SimpleNamespace(max_epochs=0,max_batches=0,max_tasks=0,output_dir=cli.output_dir))
    cfg.update(seed=seed,device=[0],protocol=PROTOCOL,max_train_tasks=2,study_role='branch',study_condition_id='response_chain_'+branch,
        study_dev_split_path=str(split),study_parent_output_dir=raw.get('study_parent_output_dir'),study_parent_checkpoint=raw.get('study_parent_checkpoint'),study_parent_receipt=raw.get('study_parent_receipt'),
        chain_role=role,chain_branch=branch,chain_smoke=smoke,chain_switch_after_epoch=switch,chain_common_start_snapshot=raw.get('chain_common_start_snapshot'),
        optimizer_tag_override='response_chain_'+branch,chain_full_state_epochs=sorted(FULL_STATE_EPOCHS),probe_gradient_count=None,compute_g3_probe=False,
        resume_require_experiment_identity=False,resume_require_model_sidecars=False,resume_require_producer_receipt=False)
    if role=='start' and not all(cfg.get(k) for k in ('study_parent_output_dir','study_parent_checkpoint','study_parent_receipt')):raise ValueError('Startmustbindacceptedparent')
    if role=='branch' and not cfg['chain_common_start_snapshot']:raise ValueError('Branchmustbindcommonstart')
    return cfg

@torch.no_grad()
def runtime_payload(model):
    return dict(trainable_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()},
        parameter_gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()},requires_grad_by_name={n:p.requires_grad for n,p in model.named_parameters()},
        module_modes={n:m.training for n,m in model.named_modules()},controls={n:base.cpu_tree({k:v for k,v in vars(m).items() if _cache_name(k) or k in ('momentum','fused_attn')}) for n,m in model.named_modules()},rng=base.cpu_tree(audit.rng_get()))

@torch.no_grad()
def apply_runtime(model,payload,restore_rng=True):
    params=dict(model.named_parameters());buffers=dict(model.named_buffers());modules=dict(model.named_modules())
    for n,value in payload['trainable_state'].items():params[n].copy_(value.to(params[n]))
    for n,value in payload['buffers'].items():buffers[n].copy_(value.to(buffers[n]))
    for n,p in params.items():
        p.requires_grad_(payload['requires_grad_by_name'][n]);v=payload['parameter_gradients'][n];p.grad=None if v is None else v.to(p).clone()
    for n,m in modules.items():
        m.training=payload['module_modes'][n]
        attrs=payload.get('controls',{}).get(n,{})
        if 'controls' in payload:
            for key in list(vars(m)):
                if (_cache_name(key) or key in ('momentum','fused_attn')) and key not in attrs:delattr(m,key)
            for key,value in attrs.items():setattr(m,key,old._runtime_device(value,next(model.parameters()).device))
    if restore_rng:audit.rng_set(payload['rng'])

def load_snapshot(path,device='cuda:0',restore_rng=True):
    path=Path(path).resolve();payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')=='G2StudySnapshot_v1':return old.load_snapshot(path,device,restore_rng)
    if payload.get('schema')!='ResponseChainCheckpoint_v1':raise ValueError('Unknownresponsechaincheckpoint')
    common=Path(payload['common_start_snapshot']);
    if audit.digest(common)!=payload['common_start_sha256']:raise AssertionError('Commonstart referencechanged')
    model,_=old.load_snapshot(common,device,restore_rng=False)
    apply_runtime(model,payload,restore_rng)
    if base.parameter_hash(model)!=payload['full_model_parameter_sha256'] or base.parameter_hash(model,True)!=payload['frozen_origin_sha256']:raise AssertionError('Reference-backed fullmodel identitymismatch')
    return model,payload

def load_diagnostic_context(run_root,checkpoint,seed):
    p=torch.load(checkpoint,map_location='cpu',weights_only=False);source=p['config']['study_source']
    if source not in sys.path:sys.path.insert(0,source)
    model,p=load_snapshot(checkpoint,'cuda:0',restore_rng=True)
    if p['config']['seed']!=seed:raise AssertionError('Diagnosticseed mismatch')
    dm=build_data_manager(source,p['config'],p['config']['study_dev_split_path'])
    dataset=dm.get_dataset(range(20,40),source='train',mode='train');ids=dataset.study_full_train_indices
    batches=[]
    with inherited.observational(model):
        for b in range(2):
            audit.seed_all(20261011+seed*100+b)
            items=[dataset[i] for i in range(b*128,(b+1)*128)]
            x=torch.stack([z[1] for z in items]).to('cuda:0');y=torch.tensor([int(z[2]) for z in items],device='cuda:0')
            identity=dict(raw_train_indices=ids[b*128:(b+1)*128],inputs_sha256=audit.tensor_digest(x),targets_sha256=audit.tensor_digest(y),fixed_aug_seed=20261011+seed*100+b)
            batches.append(dict(inputs=x,targets=y,identity=identity))
    if 'optimizer_state' in p and any(not math.isclose(float(g['lr']),float(p['next_lr']),rel_tol=1e-12) for g in p['optimizer_state']['param_groups']):raise AssertionError('Saved next-update optimizerLR mismatch')
    geometry=EffectiveStep.from_model(model)
    def loss_fn(m,x,y):return F.cross_entropy(m(x)['logits'][:,20:40],y-20)
    return dict(model=model,loss_fn=loss_fn,batches=batches,lora_pairs=[(a,b,1.) for a,b in geometry.pairs],head_params=geometry.fc_parameters,lr=p['next_lr'],payload=p,geometry=geometry)

class Controller(old.Controller):
    def __init__(self,*args):
        super().__init__(*args);self.common_payload=None;self.common_path=None;self.common_sha=None;self.factor_curve=[];self.update_rows=[];self.clean_curve=[];self.epoch25_boundary=None;self.switch_receipts=[];self.start_geometry=None;self.factor_seconds=0.;self.snapshot_seconds=0.;self.baseline_clean=None
    def install(self):
        inherited.Controller.install(self);l=self.learner
        original_build=l._build_optimizer
        def build(_self,params,stage):
            if self.task==1:self.old_fc={k:v[:20] for k,v in self.old_fc.items()}
            return original_build(params,stage)
        l._build_optimizer=MethodType(build,l)
        original_run=l._run_regularized_train
        def run(_self,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
            self.scheduler=scheduler;self.task=1
            if self.cfg['chain_role']=='start':
                self.post_expand_baseline=self.evaluate_task1_curve(0)
                self.pre_expand_baseline=self.pre_expand_baseline or self.evaluate_test_group(0,20,old_local=True)
                self.save_common_start();raise PreparedStart()
            payload=self.common_payload
            if base.parameter_hash(l._unwrap_network())!=payload['full_model_parameter_sha256']:raise AssertionError('Nativebranchsetupchangedcommonparameters')
            apply_runtime(l._unwrap_network(),payload,restore_rng=True)
            self.optimizer.load_state_dict(payload['optimizer_state']);scheduler.load_state_dict(payload['scheduler_state'])
            if base.parameter_hash(l._unwrap_network())!=payload['full_model_parameter_sha256']:raise AssertionError('Branchcommonstartmodelnotexact')
            self.task_identities[-1].update(common_start_sha256=self.common_sha,common_start_model_sha256=payload['full_model_parameter_sha256'],common_start_rng_sha256=old._rng_hash(payload['rng']),head_expansion_skipped=True,expanded_once_in_start_job=True,parent_receipt=copy.deepcopy(payload['parent_receipt']))
            audit.atomic_json(self.out/'task_identities.json',self.task_identities)
            self.post_expand_baseline=self.evaluate_task1_curve(0);self.retention_curve=[self.post_expand_baseline]
            self.start_geometry=self.geometry.capture();self.baseline_clean=self.evaluate_clean_train()
            self.save_epoch_checkpoint(0)
            return original_run(train_loader,test_loader,optimizer,scheduler,epochs=epochs,class_offset=class_offset)
        l._run_regularized_train=MethodType(run,l)
        original_batch=l._step_batch_with_log
        def batch(_self,*args,**kw):
            selected=self.batch in {0,self.task_batch_count//2}
            before=self.geometry.capture() if selected else None
            r=original_batch(*args,**kw)
            self.native_compute_seconds+=self.last_record['native_probe_compute_seconds'];self.native_commit_seconds+=self.last_record['native_sgd_commit_seconds']
            self.training_peak_bytes=max(self.training_peak_bytes,self.last_record['native_training_peak_allocated_bytes'])
            if selected:
                t=time.perf_counter();row=update_decomposition(self.geometry,before);self.factor_seconds+=time.perf_counter()-t
                row.update(epoch=self.epoch,batch=self.batch,step=self.steps-1,active_method=self.stepper.active,batch_selection='fixed first and midpoint minibatch',identity=copy.deepcopy(self.batch_identity))
                self.update_rows.append(row);audit.atomic_json(self.out/'factor_update_decomposition.json',dict(steps=self.update_rows))
            return r
        l._step_batch_with_log=MethodType(batch,l)
    def save_common_start(self):
        l=self.learner;model=l._unwrap_network()
        payload=old._complete_snapshot(model,self.cfg,1,0,.02,dict(parent_receipt=self.parent_receipt,initial_identity=self.task_identities[-1],split_receipt=split_fingerprint(self.dm),optimizer_state=base.cpu_tree(self.optimizer.state_dict()),scheduler_state=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),pre_expand_baseline=self.pre_expand_baseline,post_expand_baseline=self.post_expand_baseline,chain_common_start=True,head_expansion_count=1,source=self.cli.source,common_start_prepared_once=True))
        payload.update(runtime_payload(model))
        self.common_path=self.out/'common_start.pt';torch.save(payload,self.common_path);self.common_sha=audit.digest(self.common_path)
        audit.atomic_json(self.out/'common_start_receipt.json',dict(snapshot=str(self.common_path),sha256=audit.digest(self.common_path),model_sha256=payload['full_model_parameter_sha256'],rng_sha256=old._rng_hash(payload['rng']),seed=self.seed,head_shape=list(model.fc.weight.shape),head_expansion_count=1,parent_receipt=self.parent_receipt,initial_identity=self.task_identities[-1],baseline=self.post_expand_baseline))
    def restore_common(self,dm):
        path=Path(self.cfg['chain_common_start_snapshot']).resolve();model,payload=old.load_snapshot(path,str(self.learner._device),restore_rng=False)
        if not payload.get('chain_common_start') or payload['config']['seed']!=self.seed or payload['dev_split_sha256']!=dm.study_split_sha256 or model.fc.weight.shape[0]!=40:raise AssertionError('InvalidsharedexpandedTask1start')
        l=self.learner;l._network=model;l._cur_task=0;l._known_classes=20;l._total_classes=20;l._lora_initialized=True
        native.tensor_state_empty(l);self.common_path=path;self.common_sha=audit.digest(path);self.common_payload=payload;self.parent_receipt=payload['parent_receipt'];self.pre_expand_baseline=payload['pre_expand_baseline'];self.dm=dm
        parent=Path(self.parent_receipt['parent_output_dir']);matrix=json.loads((parent/'task_matrix.json').read_text());self.matrix=copy.deepcopy(matrix['matrix']);self.sample_curve=copy.deepcopy(matrix['sample_accuracy_curve'])
        if audit.digest(parent/'task_matrix.json')!=json.loads((parent/'final_summary.json').read_text())['task_matrix_sha256']:raise AssertionError('Acceptedparentevalprefixchanged')
        # update_fc is called by the native lifecycle but the shared expansion has
        # already happened once. It must neither sample nor allocate another head.
        def already_expanded(_self,count):
            if count!=40 or _self.fc.weight.shape[0]!=40:raise AssertionError('Unexpectedbranchheadrequest')
            return None
        model.update_fc=MethodType(already_expanded,model)
        audit.rng_set(payload['rng']);audit.atomic_json(self.out/'parent_restore_receipt.json',self.parent_receipt)
    def evaluate_clean_train(self):
        model=self.learner._unwrap_network();ds=self.dm.get_dataset(range(20,40),source='train',mode='test')
        loader=self.dm.build_dataloader(ds,batch_size=128,shuffle=False,num_workers=self.cfg.get('eval_num_workers',0),generator=torch.Generator().manual_seed(20261011),persistent_workers=False)
        loss=0.;correct=count=0
        with inherited.observational(model):
            model.eval()
            with torch.no_grad():
                for _,x,y in loader:
                    x=x.to(self.learner._device);y=y.to(self.learner._device);logits=model(x)['logits'][:,20:40]
                    loss+=float(F.cross_entropy(logits,y-20,reduction='sum'));correct+=int(logits.argmax(-1).eq(y-20).sum());count+=len(y)
        if count!=906:raise AssertionError('Cleanfitmustuseall906trainingimageswithoutdev')
        return dict(ce=loss/count,accuracy=100*correct/count,samples=count,transform='native deterministic test transform',classification_classes=20,train_indices=list(ds.study_full_train_indices),dev_excluded=True,state_rng_restored=True)
    def evaluate_task1_curve(self,epoch):
        row=super().evaluate_task1_curve(epoch)
        model=self.learner._unwrap_network();ds=self.dm.get_dataset(range(20,40),source='test',mode='test')
        loader=self.dm.build_dataloader(ds,batch_size=128,shuffle=False,num_workers=self.cfg.get('eval_num_workers',0),generator=torch.Generator().manual_seed(20261012),persistent_workers=False)
        loss=0.;correct=count=0
        with inherited.observational(model):
            model.eval()
            with torch.no_grad():
                for _,x,y in loader:
                    x=x.to(self.learner._device);y=y.to(self.learner._device);z=model(x)['logits'][:,20:40]
                    loss+=float(F.cross_entropy(z,y-20,reduction='sum'));correct+=int(z.argmax(-1).eq(y-20).sum());count+=len(y)
        row['current_test_conditional20']=dict(ce=loss/count,accuracy=100*correct/count,samples=count,diagnostic_only=True)
        if self.post_expand_baseline:
            row['old_drop_since_start']=self.post_expand_baseline['old_test_allseen40']['accuracy']-row['old_test_allseen40']['accuracy']
            row['old_ce_growth_since_start']=row['old_test_allseen40']['ce']-self.post_expand_baseline['old_test_allseen40']['ce']
        return row
    def begin_epoch(self,epoch):
        native.Controller.begin_epoch(self,epoch)
        lr=float(self.optimizer.param_groups[0]['lr'])
        if not math.isclose(lr,cosine_lr(.02,epoch),rel_tol=1e-12,abs_tol=1e-15) or self.scheduler.last_epoch!=epoch-1:raise AssertionError('Continuous50epochcosinechanged')
        before_method=self.stepper.active;self.stepper.set_epoch(epoch);self.cfg['_active_chain_method']=self.stepper.active;inherited.NORMAL_GRADIENTS['sgd']=READS[self.stepper.active]
        if epoch==self.cfg['chain_switch_after_epoch']+1:
            b=self.epoch25_boundary;m=self.learner._unwrap_network()
            r=dict(epoch=epoch,method_before=before_method,method_after=self.stepper.active,optimizer_same=b['optimizer_id']==id(self.optimizer),scheduler_same=b['scheduler_id']==id(self.scheduler),parameters_exact=b['model_sha256']==base.parameter_hash(m),parameter_references_exact=b['parameter_ids']==[id(p) for p in m.parameters()],rng_exact=b['rng_sha256']==old._rng_hash(audit.rng_get()),scheduler_T_max=self.scheduler.T_max,scheduler_last_epoch=self.scheduler.last_epoch,lr=lr,reset_model=False,reset_optimizer=False,reset_scheduler=False,reset_rng=False)
            if not all(r[k] for k in ('optimizer_same','scheduler_same','parameters_exact','parameter_references_exact','rng_exact')):raise AssertionError('Switchresetstate')
            self.switch_receipts.append(r);audit.atomic_json(self.out/'switch_receipts.json',self.switch_receipts)
    def finish_epoch(self):
        inherited.Controller.finish_epoch(self)
        self.rows[-1].update(active_method=self.stepper.active,chain_branch=self.cfg['chain_branch'],expected_native_gradient_reads=READS[self.stepper.active]*self.epoch_batches)
        t=time.perf_counter();clean=self.evaluate_clean_train();curve=self.evaluate_task1_curve(self.epoch);self.passive_eval_seconds+=time.perf_counter()-t
        curve.update(clean_training=clean,active_method=self.stepper.active,lr_used=self.lr_used[0]);self.retention_curve.append(curve);self.clean_curve.append(dict(epoch=self.epoch,**clean))
        audit.atomic_json(self.out/'retention_curve.json',dict(epochs=self.retention_curve,old_drop_definition='post-expansionepoch0 minus current oldjoint40',test_used_for_selection=False));audit.atomic_json(self.out/'clean_training_curve.json',dict(epochs=self.clean_curve))
        t=time.perf_counter();f=factor_epoch(self.geometry,self.start_geometry);f.update(epoch=self.epoch,active_method=self.stepper.active,clean_training_ce=clean['ce']);self.factor_curve.append(f);self.factor_seconds+=time.perf_counter()-t
        audit.atomic_json(self.out/'factor_state_curve.json',dict(epochs=self.factor_curve,aggregation='all24Q/V layers medianandIQR preregistered'))
        self.save_epoch_checkpoint(self.epoch)
        if self.epoch==self.cfg['chain_switch_after_epoch']:
            m=self.learner._unwrap_network();self.epoch25_boundary=dict(model_sha256=base.parameter_hash(m),rng_sha256=old._rng_hash(audit.rng_get()),parameter_ids=[id(p) for p in m.parameters()],optimizer_id=id(self.optimizer),scheduler_id=id(self.scheduler))
        audit.atomic_json(self.out/'metrics.json',dict(epochs=self.rows,counters=self.counters))
    def save_epoch_checkpoint(self,epoch):
        t=time.perf_counter();m=self.learner._unwrap_network();directory=self.out/'checkpoints';directory.mkdir(exist_ok=True)
        payload=dict(schema='ResponseChainCheckpoint_v1',protocol=PROTOCOL,config=copy.deepcopy(self.cfg),common_start_snapshot=str(self.common_path),common_start_sha256=self.common_sha,epoch=epoch,task=1,seed=self.seed,branch=self.cfg['chain_branch'],epoch_lr_used=self.lr_used[0] if epoch else .02,next_lr=float(self.optimizer.param_groups[0]['lr']),full_model_parameter_sha256=base.parameter_hash(m),frozen_origin_sha256=self.common_payload['frozen_origin_sha256'],source=self.cli.source,epoch_end_location='after scheduler and passive diagnostics; before next epoch',reference_backbone=True)
        full=epoch in FULL_STATE_EPOCHS or bool(self.cli.max_epochs)
        if full:
            payload.update(runtime_payload(m));payload.update(optimizer_state=base.cpu_tree(self.optimizer.state_dict()),scheduler_state=base.cpu_tree({k:v for k,v in self.scheduler.state_dict().items() if not callable(v)}),algorithm_state=self.stepper.state_dict(),full_runtime_state=True)
        else:
            payload['schema']='ResponseChainTrainableEpoch_v1';payload['trainable_state']={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad};payload['full_runtime_state']=False
        path=directory/('epoch_%03d.pt'%epoch)
        if path.exists():raise FileExistsError('Immutableepochcheckpointexists')
        torch.save(payload,path);self.snapshot_seconds+=time.perf_counter()-t
        self.snapshot_rows.append(dict(epoch=epoch,path=str(path),sha256=audit.digest(path),full_runtime_state=full,model_sha256=payload['full_model_parameter_sha256'],common_start_sha256=payload['common_start_sha256']))
        audit.atomic_json(self.out/'snapshots.json',dict(snapshots=self.snapshot_rows))
    def save_task_checkpoint(self):
        inherited.Controller.save_task_checkpoint(self)
        audit.atomic_json(self.out/'task_checkpoints.json',dict(all_tasks={'1':self.checkpoints[-1]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','config','output-dir'):p.add_argument('--'+key,required=True)
    for key in ('max-epochs','max-batches'):p.add_argument('--'+key,type=int,default=0)
    p.add_argument('--dry-run',action='store_true');cli=p.parse_args();cli.max_tasks=0
    for key in ('source','config','output_dir'):setattr(cli,key,str(Path(getattr(cli,key)).resolve()))
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli);cfg['study_source']=cli.source
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,seed=cfg['seed'],role=cfg['chain_role'],branch=cfg['chain_branch'],epochs=cli.max_epochs or 50,switch_after_epoch=cfg['chain_switch_after_epoch'],smoke=bool(cli.max_epochs),**metadata(cfg))));return
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Outputmustbefresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('StrictCUBLASrequired')
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,cli.source);os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import set_device,set_random,json_safe
    logging.basicConfig(level=logging.INFO,stream=sys.stdout,format='%(asctime)s %(message)s')
    cfg['filepath']=str(out/'native'/'checkpoints')+os.sep;Path(cfg['filepath']).mkdir(parents=True);cfg.update(nb_classes=200,nb_tasks=10,task_increments=[20]*10)
    set_random(cfg['seed']);set_device(cfg);dm=build_data_manager(cli.source,cfg,cfg['study_dev_split_path']);learner=factory.get_model(cfg['model_name'],cfg);learner._network.backbone.to(learner._device)
    c=Controller(learner,cfg,cli);c.dm=dm;c.install();audit.atomic_json(out/'effective_config.json',json_safe(cfg));audit.atomic_json(out/'study_split_receipt.json',split_fingerprint(dm))
    summary=dict(status='running',protocol=PROTOCOL,role=cfg['chain_role'],branch=cfg['chain_branch'],seed=cfg['seed'],smoke=bool(cli.max_epochs),input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),algorithm_sha256=audit.digest(Path(__file__).with_name('chain_step.py')),source=cli.source,dev_split_sha256=dm.study_split_sha256,epochs_per_task=cli.max_epochs or 50,switch_after_epoch=cfg['chain_switch_after_epoch'],completed_steps=0)
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        if cfg['chain_role']=='start':
            c.restore_parent(dm)
            try:learner.incremental_train(dm)
            except PreparedStart:pass
            else:raise AssertionError('Commonstartunexpectedlytrained')
            summary.update(status='completed',completed_steps=0,completed_epochs=0,common_start_snapshot=str(c.common_path),common_start_sha256=c.common_sha,parent_restore_receipt=c.parent_receipt,checks=dict(expanded_once=True,no_training_updates=True,shared_parent_exact=True))
        else:
            c.restore_common(dm);learner.incremental_train(dm);learner.eval_task();learner.after_task()
            epochs=cli.max_epochs or 50
            if len(c.rows)!=epochs or len(c.retention_curve)!=epochs+1 or len(c.matrix)!=2 or len(c.switch_receipts)!=1:raise AssertionError('Incompletepairedbranchprotocol')
            if any(c.counters[k] for k in ('forbidden_work_attempts','weight_merge_updates','fisher_estimation_passes')):raise AssertionError('Forbiddenauxiliarywork')
            summary.update(status='completed',completed_steps=c.steps,completed_epochs=len(c.rows),completed_tasks=2,training_tasks_this_job=1,trained_task_ids=[1],counters=c.counters,parent_restore_receipt=c.parent_receipt,initial_identity=c.task_identities[0],task_identities=c.task_identities,common_start_snapshot=str(c.common_path),common_start_sha256=c.common_sha,snapshots=c.snapshot_rows,checkpoint_dirs=c.checkpoints,switch_receipts=c.switch_receipts,retention_curve=c.retention_curve,endpoint=c.retention_curve[-1],clean_training_start=c.baseline_clean,final_model_parameter_sha256=base.parameter_hash(learner._unwrap_network()),step_trace_sha256=audit.digest(c.trace),task_matrix_sha256=audit.digest(out/'task_matrix.json'),native_training_compute_seconds=c.native_compute_seconds+c.native_commit_seconds,passive_eval_seconds=c.passive_eval_seconds,passive_factor_diagnostic_seconds=c.factor_seconds,snapshot_seconds=c.snapshot_seconds,native_training_peak_allocated_bytes=c.training_peak_bytes,elapsed_seconds=time.time()-c.started,checks=dict(finite=True,one_sgd_per_batch=True,no_history=True,fisher_empty=True,weight_merges_zero=True,switch_without_reset=True,common_start_exact=True,head_expanded_once=True,train_dev_disjoint=True,test_not_used_for_selection=True))
        if audit.nonfinite_paths(summary):raise FloatingPointError('Nonfinitecompletedsummary')
    except BaseException as exc:summary.update(status='failed',error=repr(exc),completed_steps=c.steps);raise
    finally:c.handle.close();audit.atomic_json(out/'final_summary.json',summary)
if __name__=='__main__':main()
