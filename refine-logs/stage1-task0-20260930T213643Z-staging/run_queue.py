#!/usr/bin/env python3
"""Versioned Task0 queue; only idle GPUs, inherited leases, no blind retry."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from common import atomic_json,now,proc_identity,query_gpus,sha256,verify_source,assert_host

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',required=True);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    path=Path(args.suite).resolve();suite=json.loads(path.read_text());suite_hash=sha256(path);runtime=Path(suite['runtime']);runtime.mkdir(parents=True,exist_ok=True)
    def validate():
        assert_host(suite)
        if sha256(path)!=suite_hash:raise RuntimeError('Changed suite')
        verify_source(suite)
        for p,h in suite['file_pins'].items():
            if sha256(p)!=h:raise RuntimeError('Changed pinned file: '+p)
        if not set(suite['allowed_gpus'])<=set(range(7)):raise RuntimeError('Invalid GPU pool')
        if len({j['id'] for j in suite['jobs']})!=len(suite['jobs']):raise RuntimeError('Duplicate job ids')
        for j in suite['jobs']:
            if not Path(j['output']).is_relative_to(Path(suite['campaign'])):raise RuntimeError('Invalid output root')
            cfg=json.loads(Path(j['config']).read_text())
            if cfg['seed']!=[2] or cfg['max_train_tasks']!=1 or cfg['init_cls']!=20 or cfg['increment']!=20:raise RuntimeError('Changed task0 protocol')
    validate()
    if args.validate_only:print(json.dumps({'status':'passed','jobs':len(suite['jobs']),'suite_sha256':suite_hash}));return
    lock=open(runtime/'controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (runtime/'controller.json').exists():raise RuntimeError('Existing runtime, use an explicitly reconciled recovery suite')
    atomic_json(runtime/'controller.json',dict(proc_identity(os.getpid()),time=now(),suite_sha256=suite_hash))
    pending=list(suite['jobs']);active={};completed=[];failed=[];halt=False;leases=Path('/tmp/olora_gamfs_gpu_locks');leases.mkdir(exist_ok=True)
    def state():
        atomic_json(runtime/'state.json',{'time':now(),'status':'failed' if halt and not active else 'completed' if not pending and not active else 'running','pending':[j['id'] for j in pending],'running':[{k:v for k,v in a.items() if k not in ('process','log')} for a in active.values()],'completed':completed,'failed':failed,'paused':(runtime/'PAUSE').exists(),'suite_sha256':suite_hash})
    def inspect(job,rc):
        errors=[];result={}
        if rc:errors.append('exit_code='+str(rc))
        try:
            result=json.loads((Path(job['output'])/'final_summary.json').read_text())
            if result.get('status')!='completed':errors.append('status')
            if result.get('completed_epochs')!=job['expected_epochs']:errors.append('epoch_count')
            if result.get('task')!=0 or result.get('trained_tasks')!=[0]:errors.append('unexpected_task')
            cfg=json.loads(Path(job['config']).read_text())
            if result.get('seed')!=2 or result.get('direction')!=cfg['stage1_direction']:errors.append('configuration_identity')
            if float(result.get('r',-1))!=float(cfg['gam_grad_norm_rho']):errors.append('radius')
            is_smoke=bool(job.get('extra_args'))
            if bool(result.get('smoke'))!=is_smoke:errors.append('smoke_identity')
            import math
            for key in ('eval_ce','eval_accuracy','train_ce','train_accuracy'):
                value=result.get('final',{}).get(key)
                if value is None or not math.isfinite(value):errors.append('nonfinite_or_missing:'+key)
            count=result.get('counters',{})
            if count.get('training_ce_gradient_passes')!=count.get('training_batches',-1)*(2 if cfg['stage1_direction']=='true_sam' else 5):errors.append('gradient_pass_count')
            expected_ckpt=[0,job['expected_epochs']] if is_smoke else [0,10,20,30,40,50]
            if result.get('checkpoint_epochs')!=expected_ckpt:errors.append('checkpoint_epochs')
            for epoch in expected_ckpt:
                for name in ('lora_w_a_0.pt','lora_w_b_0.pt','fc_state_0.pt','lora_meta_0.json','manifest.json','training_state.pt'):
                    if not (Path(job['output'])/'checkpoints'/('epoch_%03d'%epoch)/name).is_file():errors.append('missing_checkpoint:'+str(epoch)+':'+name)
            if not is_smoke:
                if count.get('training_samples')!=1259*50 or count.get('training_batches')!=500:errors.append('training_sample_count')
                epochs=json.loads((Path(job['output'])/'metrics.json').read_text())['epochs']
                if len(epochs)!=50 or any(e.get('train_samples')!=1259 or e.get('eval_samples')!=307 for e in epochs):errors.append('epoch_sample_counts')
        except Exception as exc:errors.append(repr(exc))
        return errors,result
    try:
        while pending or active:
            for key,a in list(active.items()):
                rc=a['process'].poll()
                if rc is None:continue
                a['log'].close();errors,res=inspect(a['job'],rc);validate()
                rec={'id':key,'time':now(),'returncode':rc,'errors':errors,'output':a['job']['output'],'summary':res}
                atomic_json(runtime/'jobs'/key/('failed.json' if errors else 'completed.json'),rec)
                (failed if errors else completed).append(rec);del active[key]
                if errors:halt=True
                print(json.dumps({'event':'finish','id':key,'returncode':rc,'errors':errors}),flush=True)
            if halt and not active:state();break
            if not halt and not (runtime/'PAUSE').exists():
                for gpu,info in query_gpus().items():
                    if not pending or len(active)>=suite['max_parallel']:break
                    if gpu not in suite['allowed_gpus'] or info['memory_used_mb']>=500 or info['utilization_percent']>=5:continue
                    lease=open(leases/f'gpu{gpu}.lock','a+')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    check=query_gpus().get(gpu)
                    if not check or check['uuid']!=info['uuid'] or check['memory_used_mb']>=500 or check['utilization_percent']>=5:lease.close();continue
                    try:
                        validate();j=pending.pop(0);jd=runtime/'jobs'/j['id'];jd.mkdir(parents=True,exist_ok=False)
                        if Path(j['output']).exists():raise RuntimeError('Existing output '+j['output'])
                        env=dict(os.environ,**suite['environment']);env['CUDA_VISIBLE_DEVICES']=info['uuid'];cmd=[suite['python'],suite['script'],'--source',suite['source_root'],'--config',j['config'],'--output-dir',j['output']]+j.get('extra_args',[])
                        log=open(jd/'console.log','x');proc=subprocess.Popen(cmd,cwd=suite['source_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                        rec=dict(proc_identity(proc.pid),id=j['id'],physical_gpu=gpu,gpu_uuid=info['uuid'],time=now(),command=cmd,job=j,script_sha256=sha256(suite['script']),config_sha256=sha256(j['config']))
                        atomic_json(jd/'started.json',rec);active[j['id']]={**rec,'process':proc,'log':log};print(json.dumps({'event':'started','id':j['id'],'pid':proc.pid,'gpu':gpu}),flush=True)
                    finally:lease.close()
            state()
            if pending or active:time.sleep(10)
        state()
        if failed:raise SystemExit(1)
        if suite.get('final_command'):subprocess.run(suite['final_command'],check=True,env=dict(os.environ,**suite['environment']))
    except BaseException as exc:
        atomic_json(runtime/'exception.json',{'time':now(),'error':repr(exc),'active':list(active),'pending':[j['id'] for j in pending]});raise

if __name__=='__main__':main()
