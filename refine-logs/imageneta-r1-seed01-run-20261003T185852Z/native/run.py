#!/usr/bin/env python3
"""Run one immutable native-family job; resource admission belongs to root worker."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, shutil, sys, traceback
from pathlib import Path
from types import MethodType
from native_support import HERE,MODULE,read_json,sha256,now,exclusive_json,verify_config,inspect_metrics

def source_check(suite):
    source=Path(suite['source_root']);manifest=read_json(suite['source_manifest'])
    for name,digest in manifest['files'].items():
        if sha256(source/name)!=digest:raise RuntimeError('Frozen source mismatch: '+name)
    return source

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',required=True);ap.add_argument('--job',required=True);ap.add_argument('--dry-run',action='store_true');cli=ap.parse_args()
    suite=read_json(cli.suite);job=next(j for j in suite['jobs'] if j['id']==cli.job)
    cfgpath,cfg,output=verify_config(job);source=source_check(suite)
    if cli.dry_run:
        print(json.dumps({'passed':True,'job':job['id'],'seed':job['seed'],'selector':job['direction_selector'],'scheduler_T_max':50,'epochs_per_task':1 if job['phase']=='smoke' else 50}));return 0
    run=HERE/'runtime/jobs'/job['id'];run.mkdir(parents=True,exist_ok=True)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refuse nonempty scientific output: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    exclusive_json(run/'config.json',cfg)
    if sha256(run/'config.json')!=job['config_sha256']:
        # JSON indentation is not scientific identity: retain original immutable bytes.
        (run/'config.json').unlink();shutil.copyfile(cfgpath,run/'config.json')
    record={'id':job['id'],'seed':job['seed'],'phase':job['phase'],'status':'started','started_at':now(),
            'config_sha256':sha256(cfgpath),'suite_sha256':sha256(cli.suite),'run_pins_sha256':sha256(HERE/'RUN_PINS.json')}
    try:
        os.chdir(source);sys.path.insert(0,str(source));sys.dont_write_bytecode=True
        import torch
        torch.set_num_threads(8)
        versions={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':importlib.metadata.version('torchvision'),
            'timm':importlib.metadata.version('timm'),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        expected={'python':'3.11.15','torch':'2.4.1+cu121','torchvision':'0.19.1','timm':'1.0.9','cuda':'12.1','cudnn':90100}
        if versions!=expected or torch.cuda.device_count()!=1 or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError('Native runtime/FP32/CUDA contract differs '+repr(versions))
        from src.main import main as train
        from utils import factory
        from legacy_direction import install,set_temporary
        reference_spec=None
        if job['direction_selector']=='gmatch':
            reference=next(j for j in suite['jobs'] if j['id']==job['reference_job'])
            if reference['seed']!=job['seed'] or reference['phase']!=job['phase'] or reference['condition']!='N03':raise ValueError('Incorrect S reference identity')
            from audit_validation import validate_job_audit
            validate_job_audit(suite,reference,require_receipt=True)
            reference_spec={'job':reference,'run_dir':str(HERE/'runtime/jobs'/reference['id']),
                            'candidate_config':cfg,'candidate_phase':job['phase']}
        exclusive_json(run/'native_started.json',{**record,'versions':versions,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})
        print(json.dumps({'native_started':job['id'],'seed':job['seed'],'selector':job['direction_selector'],'phase':job['phase']}),flush=True)
        session=install(factory,job['direction_selector'],run/'direction_audit.jsonl',source,expected_radius=1.,reference_spec=reference_spec)
        original_bind=session.bind
        def bind(learner):
            original_bind(learner)
            native_run=learner._run_regularized_train
            def limited(_learner,train_loader,test_loader,optimizer,scheduler,*,epochs,class_offset):
                if epochs!=50 or scheduler.T_max!=50:raise AssertionError('Native scheduler/full epoch configuration differs')
                rate=(.04 if learner._cur_task==0 else .02)*job['lr_multiplier']
                if not __import__('math').isclose(scheduler.eta_min,rate*.1,rel_tol=1e-12):raise AssertionError('Native LR floor differs')
                return native_run(train_loader,test_loader,optimizer,scheduler,epochs=1 if job['phase']=='smoke' else epochs,class_offset=class_offset)
            set_temporary(learner,'_run_regularized_train',MethodType(limited,learner),session.restores)
        session.bind=bind
        with session: train(['--config',str(run/'config.json')])
        sys.stdout.flush();sys.stderr.flush();source_check(suite)
        from audit_validation import validate_job_audit
        result=inspect_metrics(suite,job);proof=validate_job_audit(suite,job,require_receipt=False)
        final={**record,'status':'completed','ended_at':now(),'family':'native_global_rng_5CE','condition':job['condition'],
               'direction_selector':job['direction_selector'],'scheduler_T_max':50,'epochs_per_task':1 if job['phase']=='smoke' else 50,
               'completed_tasks':job['expected_tasks'],'completed_steps':job['expected_steps'],'result':result,'direction_proof':proof}
        exclusive_json(output/'final_summary.json',final);exclusive_json(run/'native_completed.json',final)
        print(json.dumps({'native_completed':job['id'],'result':result},allow_nan=False),flush=True);return 0
    except BaseException as exc:
        exclusive_json(run/'native_failed.json',{**record,'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'ended_at':now()})
        traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
