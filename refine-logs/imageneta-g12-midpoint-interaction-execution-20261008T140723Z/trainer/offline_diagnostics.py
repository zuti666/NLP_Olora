#!/usr/bin/env python3
"""Isolated midpoint/terminal diagnostics from historical named snapshots.

Old held-out data only enters torch.no_grad evaluation. No optimizer step is
called. All virtual SGD coordinates are materialized in FP32 before any FP64
precision audit; deterministic repetition is not a precision certificate.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from common import atomic_json, sha256, now, snapshot_rng_fingerprint
import audit_core as audit
import native_lifecycle as native
from effective_step import EffectiveStep
from state_transaction import State, rng_set, rng_get, rng_equal, backend_context

CALIBRATION = {(0, "entry"), (1, "epoch1"), (4, "entry"), (4, "end"), (9, "entry"), (9, "end")}
INTERACTION_KEYS = ("g2", "g3", "sum12", "sum13")
CANDIDATE_KEYS = ("sum12", "base02", "two_midpoint", "sum13", "g2", "g3", "two_g2", "align")


def clone(v): return tuple(x.detach().clone() for x in v)
def combine(a, b, sa=1., sb=1.): return tuple(sa*x+sb*y for x, y in zip(a, b))
def dot(a, b): return float(sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b)).item())
def norm(a): return math.sqrt(max(0., dot(a,a)))
def cosine(a,b):
    d=norm(a)*norm(b)
    return dot(a,b)/d if d > 1e-24 else None

def unit(v):
    denominator=norm(v)+1e-12
    return tuple(x/denominator for x in v)

@torch.no_grad()
def position(base, direction, scale):
    return tuple(x+float(scale)*y for x,y in zip(base,direction))

@torch.no_grad()
def virtual_sgd(base, direction, eta):
    result=[]
    for x,u in zip(base,direction):
        value=x.clone(); value.add_(u, alpha=-float(eta)); result.append(value)
    return tuple(result)

@torch.no_grad()
def set_values(params, values):
    if len(params)!=len(values): raise ValueError("named parameter count mismatch")
    for p,v in zip(params,values):
        if p.shape != v.shape: raise ValueError("parameter shape mismatch")
        p.copy_(v.to(p))


def tensor_fingerprints(values): return [audit.tensor_digest(x) for x in values]

def tree_fingerprint(tree):
    digest=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for k in sorted(value,key=str): digest.update(str(k).encode()); visit(value[k])
        elif isinstance(value,(list,tuple)):
            for item in value: visit(item)
        else: digest.update(repr(value).encode())
    visit(tree); return digest.hexdigest()


def interaction(losses):
    """Differences accumulate from aligned per-example values, never log means."""
    vectors={k:v.detach().cpu().double() for k,v in losses.items()}
    d0=vectors['g3']-vectors['g2']
    d1=vectors['sum13']-vectors['sum12']
    response=d1-d0
    return dict(D0=float(d0.mean()),D1=float(d1.mean()),I=float(response.mean()),
                samples=int(response.numel()),loss_means={k:float(v.mean()) for k,v in vectors.items()},
                per_sample_sha256=audit.tensor_digest(response))


def precision_verdict(fp32, repeat, fp64):
    noise=abs(fp32['I']-repeat['I'])
    precision_error=abs(fp32['I']-fp64['I'])
    epsilon=max(noise,precision_error)
    sign_consistent=(fp32['I']*fp64['I'] > 0)
    resolved=sign_consistent and abs(fp64['I']) > 5*epsilon
    return dict(status='resolved' if resolved else 'numerically_unresolved',epsilon_num=epsilon,
                repeated_fp32_I_difference=noise,fp32_fp64_I_difference=precision_error,
                fp32_I=fp32['I'],fp64_I=fp64['I'],same_sign=sign_consistent,
                threshold_multiple=5,statistical_significance=False)


def point_metrics(points, grads, geometry, reorder):
    p0,p1,p2,pm=(points[k] for k in ('phi0','phi1','phi2','phim'))
    d1,d2=combine(p1,p0,1,-1),combine(p2,p0,1,-1)
    midpoint=combine(pm,p0,1,-1); half=combine(p2,p1,.5,-.5)
    g0,g1,g2,gm=(grads[k] for k in ('g0','g1','g2','gm'))
    e=combine(combine(g1,g2,.5,.5),gm,1,-1)
    signal_floor=max(1e-10,1e-8*max(norm(g0),norm(g1),norm(g2)))
    low=norm(gm)<=signal_floor
    a=dot(gm,e)/dot(gm,gm) if not low else None
    perp=combine(e,gm,1,-a) if not low else None
    def ordered(values): return tuple(values[i] for i in reorder)
    endpoint_midpoint=combine(p1,p2,.5,.5)
    return dict(delta1_norm=norm(d1),delta2_norm=norm(d2),midpoint_displacement_norm=norm(midpoint),
        endpoint_half_separation_norm=norm(half),cos_delta1_delta2=cosine(d1,d2),
        midpoint_exact_endpoint_coordinate=True,
        midpoint_coordinate_error=norm(combine(pm,endpoint_midpoint,1,-1)),
        theoretical_midpoint_rounding_error=norm(combine(pm,combine(p0,combine(d1,d2,.5,.5)),1,-1)),
        q_norm=norm(combine(g1,g0,1,-1)),q_degenerate=norm(combine(g1,g0,1,-1))<=1e-12,
        normalization_r_over_q_plus_eps=1./(norm(combine(g1,g0,1,-1))+1e-12),
        g_norms={k:norm(v) for k,v in grads.items()},e12_norm=norm(e),
        low_signal=low,signal_floor=signal_floor,e12_over_gm=None if low else norm(e)/norm(gm),
        a12=a,a12_at_most_minus_one=None if low else a<=-1,
        perpendicular_e12_norm=None if low else norm(perp),
        perpendicular_e12_over_gm=None if low else norm(perp)/norm(gm),
        cosine_sum12_two_midpoint=cosine(combine(g1,g2),tuple(2*x for x in gm)),
        probe_effective_deltas={k:geometry.actual_delta(ordered(p0),ordered(v)) for k,v in points.items() if k!='phi0'})


def six_reads(model, params, batch, task, rho=.05, radius=1.):
    """Six detached ordinary gradients, all reading one R0 and restoring exactly."""
    state=State(model); base=clone(params); grads={}; points={'phi0':base}; losses={}
    x,y=batch
    def read(label, values):
        state.restore(); set_values(params,values); model.train()
        with torch.enable_grad(),backend_context(model,'native'):
            logits=model(x)['logits']
            loss=F.cross_entropy(logits[:,20*task:20*(task+1)],y-20*task)
            gradient=torch.autograd.grad(loss,params,create_graph=False,retain_graph=False)
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(g).all()) for g in gradient):
            raise FloatingPointError('Nonfinite diagnostic '+label)
        grads[label]=clone(gradient); losses[label]=float(loss.detach()); return grads[label]
    try:
        g0=read('g0',base)
        points['phi1']=position(base,unit(g0),rho); g1=read('g1',points['phi1'])
        points['phi2']=position(base,unit(combine(g1,g0,1,-1)),radius); g2=read('g2',points['phi2'])
        points['phim']=tuple((a+b)*.5 for a,b in zip(points['phi1'],points['phi2'])); gm=read('gm',points['phim'])
        points['phi3']=position(points['phi2'],unit(g2),rho); g3=read('g3',points['phi3'])
        length=norm(combine(points['phim'],base,1,-1))
        points['phialign']=position(base,unit(g0),length); ga=read('galign',points['phialign'])
        candidates=dict(sum12=combine(g1,g2),base02=combine(g0,g2),two_midpoint=tuple(2*v for v in gm),
            sum13=combine(g1,g3),g2=g2,g3=g3,two_g2=tuple(2*v for v in g2),align=tuple(2*v for v in ga))
        return dict(points=points,grads=grads,candidates=candidates,train_ce=losses,ordinary_gradient_calls=6)
    finally:
        state.restore(); state.check()


def fixed_batches(dm,task,point,count,batch_size=128):
    """Independent diagnostic sampling; IDs/augmented tensors persisted verbatim."""
    start,end=dm.get_task_class_range(task)
    dataset=dm.get_dataset(np.arange(start,end),'train','train')
    epoch={'entry':1,'epoch1':2,'end':50}[point]
    seed=2*100000+task*1000+epoch
    outer_rng=rng_get()
    try:
        audit.seed_all(seed)
        loader=dm.build_dataloader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,
                                   generator=torch.Generator().manual_seed(seed),persistent_workers=False)
        batches=[]
        for ids,x,y in loader:
            batches.append((ids.clone(),x.clone(),y.clone(),seed,epoch))
            if len(batches)==count:break
        if len(batches)!=count: raise ValueError('fewer than required fixed training batches')
        return batches,dataset
    finally:rng_set(outer_rng)


def evaluation_samples(dm,task,max_samples):
    """Round-robin by class and in-source sample order, shared for every U."""
    start,end=dm.get_task_class_range(task)
    dataset=dm.get_dataset(np.arange(start,end),'test','test')
    by_class={k:np.flatnonzero(dataset.labels==k).tolist() for k in range(start,end)}
    indices=[]; offset=0
    while len(indices)<min(len(dataset),max_samples):
        added=False
        for k in range(start,end):
            if offset<len(by_class[k]):
                indices.append(by_class[k][offset]);added=True
                if len(indices)==max_samples:break
        if not added:break
        offset+=1
    samples=[dataset[i] for i in indices]
    ids=torch.tensor([int(s[0]) for s in samples]);x=torch.stack([s[1] for s in samples])
    y=torch.tensor([int(s[2]) for s in samples])
    return (ids,x,y),dict(task=task,local_dataset_indices=indices,labels=y.tolist(),
         paths=[str(dataset.images[i]) for i in indices],input_sha256=audit.tensor_digest(x),
         target_sha256=audit.tensor_digest(y),selection='source_order_class_round_robin_max32')


@torch.no_grad()
def evaluate_point(model,params,anchor,values,eval_batches,device,dtype):
    anchor.restore();set_values(params,values);model.eval();seen=int(model.fc.weight.shape[0]);result={}
    with backend_context(model,'math'):
        for task,(ids,x,y) in eval_batches.items():
            logits=model(x.to(device=device,dtype=dtype))['logits'][:,:seen]
            losses=F.cross_entropy(logits,y.to(device),reduction='none')
            local=F.cross_entropy(logits[:,20*task:20*(task+1)],y.to(device)-20*task,reduction='none')
            if not bool(torch.isfinite(losses).all()):raise FloatingPointError('Nonfinite evaluation')
            result[task]=dict(seen_ce=losses.cpu(),task_local_ce=local.cpu())
    return result


def evaluation_bundle(model,params,anchor,coords,eval_batches,device,dtype):
    return {k:evaluate_point(model,params,anchor,value,eval_batches,device,dtype) for k,value in coords.items()}


def interaction_by_task(evals):
    tasks=next(iter(evals.values())).keys()
    return {str(t):interaction({k:evals[k][t]['seen_ce'] for k in INTERACTION_KEYS}) for t in tasks}


def task_equal(interactions,task):
    selected=[v for k,v in interactions.items() if int(k)<task]
    if not selected:return None
    return {k:sum(v[k] for v in selected)/len(selected) for k in ('D0','D1','I')}


def snapshot_lr(metrics,task,point):
    epoch=50 if point=='end' else 1
    row=next(x for x in metrics['epochs'] if x['task']==task and x['epoch']==epoch)
    used=row['lr_used']
    if len(used)!=1:raise ValueError('multiple optimizer learning rates cannot use additive diagnostic')
    return float(used[0]),dict(file='metrics.json',task=task,epoch=epoch,field='lr_used[0]',
                              point_rule='end=epoch50 used;entry=epoch1;epoch1=epoch1 used')


def construct_learner(code,reference,task,raw_config):
    from utils.data_manager import DataManager
    from utils import factory
    cfg=json.loads((reference/'effective_config.json').read_text())
    cfg.update(device=[torch.device('cuda:0')],seed=2,train_num_workers=0,eval_num_workers=0,
               resume_require_experiment_identity=False,resume_require_model_sidecars=False,
               resume_require_fisher_state=False,ewc_skip_task_fisher=True)
    for key in ('ewc_lambda','approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge',
                'ewc_normalize_fisher','ewc_normalize_accumulated_fisher','ewc_fisher_damping_alpha'):
        cfg[key]=raw_config.get(key,0. if key in ('ewc_lambda','ewc_fisher_damping_alpha') else False)
    audit.seed_all(2)
    dm=DataManager(cfg['dataset'],False,2,20,20,cfg)
    cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
    learner=factory.get_model(cfg['model_name'],cfg)
    learner._unwrap_network().backbone.to('cuda:0')
    checkpoint=json.loads((reference/'task_checkpoints.json').read_text())['all_tasks'][str(task)]
    learner.load_task_checkpoint(checkpoint,task,dm)
    model=learner._unwrap_network().to('cuda:0')
    counters=dict(forbidden_work_attempts=0)
    native.install_forbidden_work_guards(learner,counters)
    native.tensor_state_empty(learner)
    return learner,model,dm,Path(checkpoint),counters


def run(args):
    out=Path(args.output_dir).resolve();source=Path(args.source).resolve();reference=Path(args.reference).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('diagnostic output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    raw=json.loads(Path(args.config).read_text())
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Pinned deterministic cuBLAS workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(0)
    if float(raw['gam_grad_rho'])!=.05 or float(raw['gam_grad_norm_rho'])!=1. or raw['lora_rank']!=10:
        raise ValueError('r1/rho.05/rank10 diagnostic required')
    if raw.get('ewc_lambda')!=0 or not raw.get('ewc_skip_task_fisher') or raw.get('in_task_weight_merge') or raw.get('task_endpoint_weight_merge'):
        raise ValueError('Pure GAM no Fisher/merge config required')
    os.environ.setdefault('HF_HUB_OFFLINE','1');sys.path.insert(0,str(source));os.chdir(source)
    observations=json.loads((reference/'observations.json').read_text())
    expected={(t,p) for t in (0,1,4,9) for p in ('entry','epoch1','end')}
    if {(x['task'],x['point']) for x in observations}!=expected or len(observations)!=12:
        raise ValueError('Expected exact 12 historical observation states')
    source_hashes={str(reference/name):sha256(reference/name) for name in ('observations.json','metrics.json','effective_config.json','task_checkpoints.json')}
    for item in observations:
        if sha256(item['path'])!=item['sha256']:raise AssertionError('Historical observation hash mismatch')
        source_hashes[item['path']]=item['sha256']
    source_manifest=Path(__file__).resolve().parent.parent/'source_manifest.json'
    frozen_source=json.loads(source_manifest.read_text())
    source_hashes[str(source_manifest)]=sha256(source_manifest)
    for relative,expected_sha in frozen_source['files'].items():
        actual=source/relative
        if sha256(actual)!=expected_sha:raise AssertionError('Frozen user source hash mismatch: '+relative)
        source_hashes[str(actual)]=expected_sha
    metrics=json.loads((reference/'metrics.json').read_text()); rows=[]
    selected=observations[:1] if args.smoke else observations
    summary=dict(schema_version=1,status='running',accepted=False,smoke=args.smoke,protocol='isolated_g12_midpoint_terminal_same_fp32_coordinate_precision_v1',
        started_at=now(),reference_output=str(reference),source=str(source),input_config_sha256=sha256(args.config),
        trainer_sha256=sha256(__file__),source_hashes=source_hashes,combination_count=0,
        ordinary_gradient_calls=0,sgd_updates=0,old_data_gradient_calls=0,fp64_calibrated_count=0,
        precision_uncalibrated_count=0,saved_virtual_fp32_coordinates=True,diagnostic_rows='diagnostics.jsonl')
    atomic_json(out/'diagnostics_summary.json',summary)
    start_time=time.monotonic()
    try:
        for item in selected:
            task,point=item['task'],item['point'];learner,model,dm,checkpoint,guards=construct_learner(source,reference,task,raw)
            snapshot=torch.load(item['path'],map_location='cpu',weights_only=False)
            trainables={n:p for n,p in model.named_parameters() if p.requires_grad}
            if set(trainables)!=set(snapshot['trainable_state']):raise AssertionError('Named snapshot/trainable parameter identity mismatch')
            with torch.no_grad():
                for name,p in trainables.items():
                    saved=snapshot['trainable_state'][name]
                    if p.shape!=saved.shape:raise AssertionError('Snapshot classifier/factor shape mismatch')
                    p.copy_(saved.to(p))
            model.train();rng_set(snapshot['rng']);names=tuple(trainables);params=tuple(trainables.values())
            if not all(torch.equal(p.cpu(),snapshot['trainable_state'][name]) for name,p in trainables.items()):
                raise AssertionError('Observation named tensor load not exact')
            geometry=EffectiveStep.from_model(model);reorder=[next(i for i,p in enumerate(params) if p is q) for q in geometry.params]
            state=State(model);anchor_parameter_hash=audit.tensor_digest(torch.cat([p.detach().flatten() for p in params]))
            saved_training_state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            saved_training_hash=tree_fingerprint(saved_training_state)
            for checkpoint_artifact in checkpoint.iterdir():
                if checkpoint_artifact.is_file():source_hashes[str(checkpoint_artifact)]=sha256(checkpoint_artifact)
            batches,_=fixed_batches(dm,task,point,1 if args.smoke else 2)
            eval_batches={};eval_manifest={}
            for j in range(task+1):
                eval_batches[j],eval_manifest[str(j)]=evaluation_samples(dm,j,4 if args.smoke else 32)
            state.restore();state.check()
            eta,lr_source=snapshot_lr(metrics,task,point)
            for batch_index,(ids,x,y,data_seed,data_epoch) in enumerate(batches):
                state.restore();state.check()
                calibration=(task,point) in CALIBRATION and batch_index==0
                device=next(model.parameters()).device
                read=six_reads(model,params,(x.to(device),y.to(device)),task)
                points,grads,candidates=read['points'],read['grads'],read['candidates']
                coords={k:virtual_sgd(points['phi0'],u,eta) for k,u in candidates.items()}
                coordinate_path=out/f'task_{task:03d}_{point}_batch_{batch_index}_coordinates.pt'
                torch.save(dict(parameter_names=names,base=tuple(v.cpu() for v in points['phi0']),
                    points={k:tuple(v.cpu() for v in values) for k,values in points.items()},
                    gradients={k:tuple(v.cpu() for v in values) for k,values in grads.items()},
                    virtual_fp32_parameters={k:tuple(v.cpu() for v in values) for k,values in coords.items()},
                    current_train_batch=dict(indices=ids,input=x,target=y,data_seed=data_seed),
                    eval_manifest=eval_manifest,eta=eta),coordinate_path)
                before_eval=evaluate_point(model,params,state,points['phi0'],eval_batches,device,torch.float32)
                evals=evaluation_bundle(model,params,state,coords,eval_batches,device,torch.float32)
                responses=interaction_by_task(evals)
                repeated=evaluation_bundle(model,params,state,{k:coords[k] for k in INTERACTION_KEYS},eval_batches,device,torch.float32)
                repeats=interaction_by_task(repeated)
                precision={str(j):dict(status='precision_uncalibrated',statistical_significance=False) for j in range(task+1)}
                if calibration:
                    state.restore();double_model=copy.deepcopy(model).double();double_model.eval()
                    double_named={n:p for n,p in double_model.named_parameters() if p.requires_grad}
                    double_params=tuple(double_named[n] for n in names);double_state=State(double_model)
                    double_coords={k:tuple(v.double() for v in coords[k]) for k in INTERACTION_KEYS}
                    double_evals=evaluation_bundle(double_model,double_params,double_state,double_coords,eval_batches,device,torch.float64)
                    double_responses=interaction_by_task(double_evals)
                    precision={k:dict(precision_verdict(responses[k],repeats[k],double_responses[k]),
                        fp64_same_coordinate=True,fp64_interaction=double_responses[k],
                        max_per_example_forward_loss_difference=max(float((evals[c][int(k)]['seen_ce'].double()-double_evals[c][int(k)]['seen_ce']).abs().max()) for c in INTERACTION_KEYS)) for k in responses}
                    double_state.restore();double_state.check();del double_state,double_model,double_params,double_coords,double_evals,double_named;gc.collect()
                    summary['fp64_calibrated_count']+=1
                else:summary['precision_uncalibrated_count']+=1
                scales={}
                for factor in (.5,.25):
                    sc={k:virtual_sgd(points['phi0'],candidates[k],eta*factor) for k in INTERACTION_KEYS}
                    scales[str(factor)]=interaction_by_task(evaluation_bundle(model,params,state,sc,eval_batches,device,torch.float32))
                def ordered(values):return tuple(values[i] for i in reorder)
                updates={}
                for key,values in candidates.items():
                    before_ordered,after_ordered=ordered(points['phi0']),ordered(coords[key])
                    pair_layers={}
                    for layer,(a,b) in enumerate(geometry.pairs):
                        pair_geo=EffectiveStep([(a,b)],geometry.fc_parameters)
                        pair_before=before_ordered[2*layer:2*layer+2]+before_ordered[2*len(geometry.pairs):]
                        pair_after=after_ordered[2*layer:2*layer+2]+after_ordered[2*len(geometry.pairs):]
                        pair_layers[names[reorder[2*layer]]]=pair_geo.actual_delta(pair_before,pair_after)['lora_norm']
                    updates[key]=dict(direction_norm=norm(values),eta_direction_norm=eta*norm(values),
                        actual_lora_layer_delta_norms=pair_layers,
                        g0_dot_direction=dot(grads['g0'],values),
                        actual_effective_delta=geometry.actual_delta(ordered(points['phi0']),ordered(coords[key])),
                        parameter_coordinates_sha256=tensor_fingerprints(coords[key]),
                        task_losses={str(j):dict(seen_ce_before=float(before_eval[j]['seen_ce'].double().mean()),
                            seen_ce_after=float(evals[key][j]['seen_ce'].double().mean()),
                            seen_ce_change=float((evals[key][j]['seen_ce'].double()-before_eval[j]['seen_ce'].double()).mean()),
                            task_local_ce_after=float(evals[key][j]['task_local_ce'].double().mean())) for j in eval_batches})
                state.restore();restoration=state.check()
                if tree_fingerprint(saved_training_state)!=saved_training_hash:raise AssertionError('Optimizer/scheduler saved state mutated')
                row=dict(task=task,point=point,batch_index=batch_index,eta=eta,lr_source=lr_source,
                    snapshot_sha256=item['sha256'],snapshot_named_load_exact=True,
                    base_parameter_sha256=anchor_parameter_hash,parameter_names=list(names),
                    rng0_sha256=snapshot_rng_fingerprint(state.rng),current_batch_identity=dict(indices=ids.tolist(),
                        input_sha256=audit.tensor_digest(x),target_sha256=audit.tensor_digest(y),data_seed=data_seed,data_epoch=data_epoch),
                    eval_manifest=eval_manifest,coordinate_file=str(coordinate_path),coordinate_file_sha256=sha256(coordinate_path),
                    gradients=read['ordinary_gradient_calls'],train_ce=read['train_ce'],
                    midpoint=point_metrics(points,grads,geometry,reorder),updates=updates,
                    terminal_interactions=responses,current_interaction=responses[str(task)],
                    old_task_equal_interaction=task_equal(responses,task),precision=precision,
                    eta_scale_fp32_uncalibrated_interactions=scales,
                    state_restoration=restoration,optimizer_scheduler_checkpoint_unmodified=True,
                    optimization_state_policy='historical checkpoint state only; no live optimizer or scheduler; no step',
                    frozen_candidate_updates=True,virtual_update_coordinate_dtype='torch.float32',fp64_calibrated=calibration,
                    old_data_used_only_no_grad_eval=True,sgd_updates=0)
                if audit.nonfinite_paths(row):raise FloatingPointError('Nonfinite diagnostic report')
                with (out/'diagnostics.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row);summary['combination_count']+=1;summary['ordinary_gradient_calls']+=6
                summary['elapsed_seconds']=time.monotonic()-start_time;atomic_json(out/'diagnostics_summary.json',summary)
                print(json.dumps(dict(completed_combination=summary['combination_count'],task=task,point=point,batch=batch_index,
                       calibrated=calibration,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
                del read,points,grads,candidates,coords,evals,repeated,repeats,scales;gc.collect()
            if guards['forbidden_work_attempts']:raise AssertionError('Forbidden work attempted')
            state.restore();state.check();del state,model,learner,geometry,params,trainables;gc.collect();torch.cuda.empty_cache()
        unchanged=all(sha256(path)==fingerprint for path,fingerprint in source_hashes.items())
        if not unchanged:raise AssertionError('Historical source artifact changed')
        expected_count=1 if args.smoke else 24
        if len(rows)!=expected_count or summary['fp64_calibrated_count']!=(1 if args.smoke else 6):
            raise AssertionError('Unexpected diagnostic count or calibration coverage')
        summary.update(status='completed',accepted=True,finished_at=now(),elapsed_seconds=time.monotonic()-start_time,
            restoration_all_exact=all(all(row['state_restoration'].values()) for row in rows),
            source_hashes_unchanged=unchanged,source_hashes=source_hashes,
            fp64_same_fp32_virtual_coordinates=True,task0_old_metrics_are_null=True,
            optimizer_scheduler_states_unmodified=True,forbidden_work_attempts=0,
            diagnostic_sha256=sha256(out/'diagnostics.jsonl'),
            fp64_expected_states=[dict(task=t,point=p,batch=0) for t,p in sorted(CALIBRATION)],
            interpretation='one-step observational diagnostics; no formal AAA/FAA training result',
            main_candidate_names=list(CANDIDATE_KEYS),per_combination_gradient_calls=6,
            full_formal_training_runs=0)
        atomic_json(out/'diagnostics_summary.json',summary)
        return summary
    except Exception as exc:
        summary.update(status='failed',accepted=False,error=repr(exc),finished_at=now())
        atomic_json(out/'diagnostics_summary.json',summary);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference','config','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args)

if __name__=='__main__':main()
