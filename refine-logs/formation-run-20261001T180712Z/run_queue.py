#!/usr/bin/env python3
"""Host-bound, UUID-bound idle-GPU queue; success-only dependencies; no preemption."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_json,assert_host,load_suite,now,process_matches,proc_identity,query_gpus,sha256,verify_source
from validate_job import validate_job,receipt_matches

HERE=Path(__file__).resolve().parent

def verify_pins(suite):
    pins=json.loads((HERE/'RUN_PINS.json').read_text())
    for name,expected in pins['files'].items():
        if sha256(HERE/name)!=expected: raise ValueError('run pin mismatch '+name)
    if sha256(suite['source_manifest'])!=suite['source_manifest_sha256']: raise ValueError('source manifest changed')
    verify_source(suite)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',default=str(HERE/'suite.json'));ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args();suite=load_suite(args.suite);assert_host(suite);verify_pins(suite)
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    control=open(runtime/'controller.lock','a+')
    fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=runtime/'queue_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'run_id':suite['run_id'],'created_at':now(),'jobs':{j['id']:{'status':'pending'} for j in suite['jobs']}}
    if state['run_id']!=suite['run_id'] or set(state['jobs'])!={j['id'] for j in suite['jobs']}:
        raise ValueError('loaded queue state belongs to a different manifest')
    for job in suite['jobs']:
        record=state['jobs'][job['id']]
        if record['status'] not in ('pending','running','completed','failed'): raise ValueError('unknown persisted status')
        if record['status']=='completed':
            receipt=json.loads((runtime/'jobs'/job['id']/'completed.json').read_text())
            actual=validate_job(job)
            if receipt['job_id']!=job['id'] or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,actual) or record['summary_sha256']!=actual['summary_sha256']:
                raise ValueError('persisted completion receipt is not valid '+job['id'])
    state.update(controller=proc_identity(os.getpid()),smoke_only=args.smoke_only,status='running')
    atomic_json(runtime/'controller.json',{'identity':proc_identity(os.getpid()),'started_at':now(),'suite_sha256':sha256(args.suite),'command':sys.argv})
    jobs={j['id']:j for j in suite['jobs']};active={};leases={}
    lockdir=Path(suite['gpu_lock_dir']);lockdir.mkdir(exist_ok=True)
    for value in suite['environment'].values():
        if value.startswith('/tmp/formation-'): Path(value).mkdir(exist_ok=True)
    # After restart, live workers keep inherited leases. Missing exit receipts fail closed.
    for jid,record in state['jobs'].items():
        if record['status']=='running': active[jid]=None
    while True:
        for jid in list(active):
            record=state['jobs'][jid];worker=active[jid]
            if worker is not None:
                if worker.poll() is None: continue
            elif process_matches(record['process']): continue
            d=runtime/'jobs'/jid
            try:
                exit_record=json.loads((d/'exit.json').read_text())
                if exit_record['job_id']!=jid or exit_record['returncode']!=0: raise ValueError('runner failed '+repr(exit_record))
                verify_pins(suite)
                result=validate_job(jobs[jid])
                receipt={'job_id':jid,'completed_at':now(),'exit':exit_record,**result}
                atomic_json(d/'completed.json',receipt)
                record.update(status='completed',completed_at=now(),summary_sha256=result['summary_sha256'])
                print(json.dumps({'event':'completed','job':jid}),flush=True)
                subprocess.run([suite['python'],str(HERE/'collect_results.py')],cwd=HERE,check=True)
            except Exception as exc:
                record.update(status='failed',failed_at=now(),error=repr(exc))
                atomic_json(d/'failed.json',record)
                print(json.dumps({'event':'failed','job':jid,'error':repr(exc)}),flush=True)
            finally:
                active.pop(jid)
                if jid in leases: leases.pop(jid).close()
                atomic_json(state_path,state)
        failed=any(r['status']=='failed' for r in state['jobs'].values())
        candidates=[j for j in suite['jobs'] if (not args.smoke_only or j['smoke'])]
        pending=[j for j in candidates if state['jobs'][j['id']]['status']=='pending']
        if failed and not active:
            state['status']='halted_failure';atomic_json(state_path,state);return 1
        if not pending and not active:
            state['counts']=dict(Counter(r['status'] for r in state['jobs'].values()))
            state['status']='smokes_completed' if args.smoke_only else 'completed';atomic_json(state_path,state);return 0
        paused=(runtime/'PAUSE').exists()
        state['paused']=paused
        if not paused and not failed:
            gpus=query_gpus();used={state['jobs'][jid]['gpu'] for jid in active}
            for job in pending:
                if len(active)>=suite['max_parallel']: break
                if any(state['jobs'][dep]['status']!='completed' for dep in job['depends_on']): continue
                for dep in job['depends_on']:
                    prior=json.loads((runtime/'jobs'/dep/'completed.json').read_text())
                    checked=validate_job(jobs[dep])
                    if not receipt_matches(prior,checked):
                        raise ValueError('dependency output changed '+dep)
                verify_pins(suite)
                if sha256(job['config'])!=job['config_sha256']: raise ValueError('config pin changed')
                chosen=None
                for index in suite['allowed_gpus']:
                    gpu=gpus.get(index)
                    if not gpu or index in used or gpu['memory_used_mb']>=500 or gpu['utilization_percent']>=5: continue
                    lease=open(lockdir/f'gpu{index}.lock','a+')
                    try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: lease.close();continue
                    current=query_gpus().get(index)
                    if not current or current['uuid']!=gpu['uuid'] or current['memory_used_mb']>=500 or current['utilization_percent']>=5:
                        lease.close();continue
                    chosen=(index,current,lease);break
                if chosen is None: break
                index,gpu,lease=chosen
                jid=job['id'];d=runtime/'jobs'/jid;d.mkdir(parents=True,exist_ok=False)
                output=Path(job['output_dir'])
                if output.exists() and any(output.iterdir()): lease.close();raise ValueError('refusing existing output '+jid)
                atomic_json(d/'job.json',{**job,'cwd':suite['source_root']})
                env=os.environ.copy();env.update(suite['environment']);env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
                log=open(d/'console.log','xb')
                try:
                    worker=subprocess.Popen([suite['python'],str(HERE/'job_worker.py'),'--job',str(d/'job.json'),'--lease-fd',str(lease.fileno())],
                        env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                finally: log.close()
                state['jobs'][jid].update(status='running',started_at=now(),gpu=index,gpu_uuid=gpu['uuid'],
                    process=proc_identity(worker.pid),command=job['command'])
                active[jid]=worker;leases[jid]=lease;used.add(index)
                atomic_json(d/'launched.json',state['jobs'][jid]);atomic_json(state_path,state)
                print(json.dumps({'event':'launched','job':jid,'gpu':index,'pid':worker.pid}),flush=True)
        state.update(updated_at=now(),counts=dict(Counter(r['status'] for r in state['jobs'].values())))
        atomic_json(state_path,state)
        time.sleep(suite['poll_seconds'])

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        atomic_json(HERE/'runtime/controller_error.json',{'at':now(),'error':repr(exc)})
        raise
