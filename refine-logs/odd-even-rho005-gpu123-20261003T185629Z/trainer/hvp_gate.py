#!/usr/bin/env python3
"""Real ImageNet-A ViT arbitrary-HVP gate; no optimiser update or training.

An initial Task0 feasibility mode accepts --config.  The full gate accepts
--reference-output, loads native end checkpoints for tasks 0/4/9, and audits
two predeclared real training batches per task.  All scientific HVPs and their
ordinary CE references use the same backend; there is no numerical fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from response_hvp import (State, attention_backend, center_difference_gate,
                          gradient_at, rng_get, vector_norm, compare)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def model_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path,report):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def normalized(gradients):
    # Matches this campaign's mathematical/global float64 reduction exactly.
    n=vector_norm(gradients)
    return tuple(g/(n+1e-12) for g in gradients)


def audit_batch(model,x,y,start,end,backend):
    params=tuple(p for p in model.parameters() if p.requires_grad)
    live=State(model);R0=rng_get();point0=tuple(p.detach().clone() for p in params)
    def closure():
        out=model(x);logits=out['logits'] if isinstance(out,dict) else out
        if logits.ndim!=2 or logits.shape[1]!=end or bool(((y<start)|(y>=end)).any()):
            raise ValueError('Current-task CE class slice mismatch')
        return F.cross_entropy(logits[:,start:end],y-start)
    report={}
    started=time.perf_counter()
    if x.device.type=='cuda':torch.cuda.reset_peak_memory_stats(x.device)
    try:
        _,g0,_,r0=gradient_at(model,params,point0,closure,R0,backend)
        phi1=tuple((p+.05*u).detach() for p,u in zip(point0,normalized(g0)))
        _,g1,_,r1=gradient_at(model,params,phi1,closure,R0,backend)
        c10=tuple(a-b for a,b in zip(g1,g0))
        phi2=tuple((p+.25*u).detach() for p,u in zip(point0,normalized(c10)))
        _,g2,_,r2=gradient_at(model,params,phi2,closure,R0,backend)
        d12=tuple((b-a).detach() for a,b in zip(phi1,phi2))
        middle=tuple(((a+b)*.5).detach() for a,b in zip(phi1,phi2))
        sec=tuple(b-a for a,b in zip(g1,g2))
        checks={}
        for label,point in [('hstart',phi1),('hmid',middle)]:
            gate=center_difference_gate(model,params,point,d12,closure,R0,backend)
            # Not an acceptance criterion: finite-path response may differ.
            _,_,hv,_=gradient_at(model,params,point,closure,R0,backend,True,d12)
            gate['hvp_vs_finite_sec']=compare(hv,sec)
            checks[label]=gate
        report={'status':'passed' if all(v['status']=='passed' for v in checks.values()) else 'failed',
                'backend':backend,'ordinary_probe_reports':[r0,r1,r2],
                'd12_source':'actual_detached_phi2_minus_phi1_snapshots',
                'd12_norm':vector_norm(d12),'first_probe_rho':.05,'outer_radius':.25,
                'coordinate_system':'all trainable LoRA factors plus seen FC',
                'fixed_current_task_ce':[start,end],'responses':checks,
                'seconds':time.perf_counter()-started,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(x.device) if x.device.type=='cuda' else 0,
                'optimizer_updates':0,'fisher_calls':0,'weight_merge_calls':0}
        return report
    finally:
        live.restore();restored=live.check()
        if report:report['entry_state_restore']=restored


def pure_config(raw,device,output_dir):
    cfg=dict(raw);seed=cfg.get('seed',2)
    if isinstance(seed,list):
        if len(seed)!=1:raise ValueError('One seed required')
        seed=seed[0]
    cfg.update(seed=int(seed),device=[device],filepath=str(Path(output_dir)/'unused_native_state')+os.sep,
               train_num_workers=0,eval_num_workers=0,resume_require_experiment_identity=False,
               resume_require_model_sidecars=False,resume_require_fisher_state=False,
               optimizer_type='gam',lambda_flat=1.,ewc_lambda=0.,ewc_skip_task_fisher=True,
               in_task_weight_merge=False,task_endpoint_weight_merge=False,approximate_fisher_in_task=False,
               compute_full_fisher=False,posterior_state_save=False,mechanism_eval=False,
               mechanism_grad_stats=False,as1_delta_w_penalty=False,memory_size=0,memory_per_class=0,
               fisher_split=False,ewc_normalize_fisher=False,ewc_normalize_accumulated_fisher=False,
               ewc_fisher_damping_alpha=0.,ewc_dual_ascent=False,compute_nme_class_means=False)
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--config')
    parser.add_argument('--reference-output')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--backend',choices=('native','math'),default='math')
    parser.add_argument('--tasks',nargs='+',type=int,default=[0,4,9])
    parser.add_argument('--max-batches',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    source=Path(args.source).resolve();out=Path(args.output_dir).resolve()
    target=out/'hvp_gate_report.json'
    if target.exists():raise FileExistsError('Refusing to overwrite HVP gate report')
    if args.max_batches<1 or args.batch_size<1 or any(t not in (0,4,9) for t in args.tasks):
        raise ValueError('Invalid frozen gate budget/tasks')
    reference=Path(args.reference_output).resolve() if args.reference_output else None
    config_path=Path(args.config).resolve() if args.config else reference/'effective_config.json' if reference else None
    if config_path is None or not config_path.is_file():raise FileNotFoundError('Configuration missing')
    if reference is None and args.tasks!=[0]:raise ValueError('Initial no-checkpoint gate requires --tasks 0')
    script_dir=str(Path(__file__).parent)
    sys.path.insert(0,str(source));os.environ.setdefault('HF_HUB_OFFLINE','1');os.chdir(source)
    from utils.data_manager import DataManager
    from utils import factory
    device=torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=pure_config(json.loads(config_path.read_text()),device,out)
    source_pins={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}
    report={'schema_version':1,'status':'running','backend':args.backend,'seed':cfg['seed'],
            'source':str(source),'source_python_sha256':source_pins,'reference_output':str(reference) if reference else None,
            'config_sha256':sha(config_path),'hvp_code_sha256':sha(Path(script_dir)/'response_hvp.py'),
            'gate_code_sha256':sha(__file__),'tasks':{},'initial_feasibility_only':reference is None,
            'h_grid':[.1,.03,.01,.003,.001],'selection':'fixed deterministic real training batches; no alignment selection',
            'same_point_gradient_relative_max':2e-5,'fd_relative_max':5e-3,'fd_cosine_min':.999,
            'adjacent_accepted_h_required':2,'optimizer_updates':0,
            'precision':{'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                         'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'autocast':False}}
    write_json(target,report)
    try:
        for task in args.tasks:
            seed_all(cfg['seed'])
            dm=DataManager(cfg['dataset'],cfg.get('class_shuffle',False),cfg['seed'],cfg['init_cls'],cfg['increment'],cfg)
            cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
            learner=factory.get_model(cfg['model_name'],cfg)
            learner._unwrap_network().backbone.to(device)
            start,end=dm.get_task_class_range(task)
            checkpoint=reference/'checkpoints'/('task_%03d'%task) if reference else None
            checkpoint_pins={}
            if checkpoint:
                manifest=json.loads((checkpoint/'manifest.json').read_text())
                if manifest.get('task')!=task or manifest.get('seed')!=cfg['seed']:
                    raise ValueError('Checkpoint seed/task mismatch')
                checkpoint_pins={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
                if not manifest.get('files_sha256') or any(checkpoint_pins.get(n)!=h for n,h in manifest['files_sha256'].items()):
                    raise AssertionError('Native checkpoint pin mismatch')
                learner.load_task_checkpoint(str(checkpoint),task,dm)
            else:
                learner._unwrap_network().update_fc(end)
                learner._cur_task,learner._known_classes,learner._total_classes=task,start,end
            model=learner._unwrap_network().to(device);model.train()
            initial_hash=model_sha(model)
            if checkpoint and initial_hash!=manifest.get('model_sha256'):raise AssertionError('Native restored model hash mismatch')
            nonzero=[name for name,m in model.named_modules() if
                     (isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p>0) or float(getattr(m,'drop_prob',0.) or 0.)>0]
            if nonzero:raise ValueError('Gate requires disabled dropout/drop-path: '+repr(nonzero))
            dataset=dm.get_dataset(np.arange(start,end),source='train',mode='train')
            data_seed=cfg['seed']*100000+70000+task;seed_all(data_seed)
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                              generator=torch.Generator().manual_seed(data_seed),drop_last=False)
            if args.max_batches>len(loader):raise ValueError('More requested gate batches than data')
            task_report={'class_range':[start,end],'model_sha256':initial_hash,'checkpoint_sha256':checkpoint_pins,
                         'data_seed':data_seed,'batches':[]}
            report['tasks'][str(task)]=task_report
            iterator=iter(loader)
            for index in range(args.max_batches):
                ids,x,y=next(iterator)
                result=audit_batch(model,x.to(device),y.to(device),start,end,args.backend)
                task_report['batches'].append({'batch':index,'samples':len(y),'input_sha256':tensor_sha(x),
                                              'target_sha256':tensor_sha(y),'indices':ids.tolist(),**result})
                write_json(target,report)
            if model_sha(model)!=initial_hash:raise AssertionError('Gate changed model parameters')
            if checkpoint and {p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}!=checkpoint_pins:
                raise AssertionError('Gate changed checkpoint')
            del learner,model,loader,dataset
            if device.type=='cuda':torch.cuda.empty_cache()
        if {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}!=source_pins:
            raise AssertionError('Gate modified frozen source')
        all_pass=all(batch['status']=='passed' for t in report['tasks'].values() for batch in t['batches'])
        report.update(status='passed' if all_pass else 'failed',source_unchanged=True,checkpoint_unchanged=True,
                      full_stage2_allowed=all_pass and reference is not None and set(args.tasks)=={0,4,9} and args.max_batches>=2)
    except Exception as exc:
        report.update(status='failed',error=repr(exc),full_stage2_allowed=False)
        raise
    finally:write_json(target,report)
    print(json.dumps({'status':report['status'],'full_stage2_allowed':report['full_stage2_allowed'],'report':str(target)}))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
