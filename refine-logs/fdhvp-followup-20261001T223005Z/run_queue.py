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
from conditional_gate import evaluate,verify_skipped

HERE=Path(__file__).resolve().parent

def verify_parent_release():
    guard=json.loads((HERE/'handoff_guard.json').read_text())
    if os.readlink('/proc/self/ns/pid')!=guard['execution_pid_namespace']:raise ValueError('host PID namespace mismatch')
    proof=json.loads((HERE/'runtime/PARENT_VALIDATED.json').read_text())
    if proof['status']!='validated' or proof['guard_sha256']!=sha256(HERE/'handoff_guard.json') or proof['parent_jobs_validated']!=guard['expected_parent_jobs']:
        raise ValueError('parent release not validated')
    parent=Path(guard['parent_dir'])
    for name,key in (('suite.json','parent_suite_sha256'),('RUN_PINS.json','parent_run_pins_sha256')):
        if proof[key]!=guard[key] or sha256(parent/name)!=guard[key]:raise ValueError('parent release pin changed')
    parent_pins=json.loads((parent/'RUN_PINS.json').read_text())
    for path,expected in parent_pins['files'].items():
        if sha256(parent/path)!=expected:raise ValueError('parent pinned file changed '+path)
    state=json.loads((parent/'runtime/queue_state.json').read_text())
    parent_suite=json.loads((parent/'suite.json').read_text())
    handoff=json.loads((parent/'runtime/handoff_state.json').read_text())
    if state['status']!='completed' or state['run_id']!=parent_suite['run_id'] or set(state['jobs'])!={j['id'] for j in parent_suite['jobs']} or len(state['jobs'])!=guard['expected_parent_jobs'] or any(r['status']!='completed' for r in state['jobs'].values()) or process_matches(state['controller']):
        raise ValueError('parent cohort not successfully finished')
    if handoff['status']!='completed' or process_matches(handoff['process']):raise ValueError('parent handoff not finished')
    if set(proof['parent_receipts_sha256'])!=set(state['jobs']) or any(sha256(parent/'runtime/jobs'/jid/'completed.json')!=expected for jid,expected in proof['parent_receipts_sha256'].items()):
        raise ValueError('validated parent receipt changed')

def verify_pins(suite):
    pins=json.loads((HERE/'RUN_PINS.json').read_text())
    for name,expected in pins['files'].items():
        if sha256(HERE/name)!=expected: raise ValueError('run pin mismatch '+name)
    if sha256(suite['source_manifest'])!=suite['source_manifest_sha256']: raise ValueError('source manifest changed')
    verify_source(suite)

def verify_gpu_smoke_release(suite,group):
    path=HERE/'runtime'/f'GPU_SMOKES_{group}_VERIFIED.json'
    if not path.exists():
        subprocess.run([suite['python'],str(HERE/'check_gpu_smokes.py'),'--group',group],cwd=HERE,check=True)
    proof=json.loads(path.read_text())
    ids=set(suite['smoke_groups'][group])
    if proof['status']!='passed' or proof['group']!=group or proof['verified_smokes']!=len(ids) or proof['suite_sha256']!=sha256(HERE/'suite.json') or proof['frozen_gates_sha256']!=sha256(HERE/'FROZEN_GATES.json') or set(proof['receipts_sha256'])!=ids:
        raise ValueError('current GPU smoke release not validated')
    if any(sha256(HERE/'runtime/jobs'/jid/'completed.json')!=value for jid,value in proof['receipts_sha256'].items()):
        raise ValueError('GPU smoke receipt changed')

def skipped_receipt(job,result):
    return {'status':'skipped_by_gate','job_id':job['id'],'skipped_at':now(),
        'config_sha256':sha256(job['config']),'suite_sha256':sha256(HERE/'suite.json'),
        'frozen_gates_sha256':sha256(HERE/'FROZEN_GATES.json'),'gate_result':result}

def verify_skip(job,record):
    receipt=json.loads((HERE/'runtime/jobs'/job['id']/'skipped.json').read_text())
    if receipt['status']!='skipped_by_gate' or receipt['job_id']!=job['id'] or receipt['gate_result']['decision']!='skip':raise ValueError('invalid conditional skip')
    if receipt['config_sha256']!=sha256(job['config']) or receipt['suite_sha256']!=sha256(HERE/'suite.json') or receipt['frozen_gates_sha256']!=sha256(HERE/'FROZEN_GATES.json'):
        raise ValueError('skip proof pins changed')
    if record['skip_sha256']!=sha256(HERE/'runtime/jobs'/job['id']/'skipped.json'):raise ValueError('skip proof changed')
    verify_skipped(job,receipt,root=HERE)

def apply_skips(suite,state):
    jobs={j['id']:j for j in suite['jobs']};changed=True;total=0
    while changed:
        changed=False
        for jid,record in state['jobs'].items():
            if record['status']!='pending':continue
            job=jobs[jid];deps=job['depends_on']
            skipped=[d for d in deps if state['jobs'][d]['status']=='skipped']
            if skipped:
                for d in skipped:verify_skip(jobs[d],state['jobs'][d])
                result=evaluate(job,suite=suite,root=HERE)
                if result['decision']!='skip':raise ValueError('skipped dependency contradicts current frozen gate '+jid)
            elif any(state['jobs'][d]['status']!='completed' for d in deps):continue
            else:result=evaluate(job,suite=suite,root=HERE)
            if result['decision']!='skip':continue
            d=HERE/'runtime/jobs'/jid;d.mkdir(parents=True,exist_ok=False)
            atomic_json(d/'skipped.json',skipped_receipt(job,result))
            record.update(status='skipped',skipped_at=now(),reason=result.get('reason'),skip_sha256=sha256(d/'skipped.json'))
            total+=1
            print(json.dumps({'event':'skipped','job':jid,'reason':result.get('reason')}),flush=True)
            changed=True
    atomic_json(HERE/'runtime/queue_state.json',state)
    return total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',default=str(HERE/'suite.json'));ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args();suite=load_suite(args.suite);assert_host(suite);verify_pins(suite);verify_parent_release()
    if args.smoke_only:raise ValueError('conditional queue must run as one dependency graph')
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    control=open(runtime/'controller.lock','a+')
    fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=runtime/'queue_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'run_id':suite['run_id'],'created_at':now(),'jobs':{j['id']:{'status':'pending'} for j in suite['jobs']}}
    if state['run_id']!=suite['run_id'] or set(state['jobs'])!={j['id'] for j in suite['jobs']}:
        raise ValueError('loaded queue state belongs to a different manifest')
    for job in suite['jobs']:
        record=state['jobs'][job['id']]
        if record['status'] not in ('pending','running','completed','failed','skipped'): raise ValueError('unknown persisted status')
        if record['status']=='skipped':verify_skip(job,record)
        if record['status']=='completed':
            receipt=json.loads((runtime/'jobs'/job['id']/'completed.json').read_text())
            actual=validate_job(job)
            if receipt['job_id']!=job['id'] or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,actual) or record['summary_sha256']!=actual['summary_sha256']:
                raise ValueError('persisted completion receipt is not valid '+job['id'])
    state.update(controller=proc_identity(os.getpid()),smoke_only=args.smoke_only,status='running')
    atomic_json(runtime/'controller.json',{'identity':proc_identity(os.getpid()),'started_at':now(),'suite_sha256':sha256(args.suite),'command':sys.argv})
    jobs={j['id']:j for j in suite['jobs']};active={};leases={}
    lockdir=Path(suite['gpu_lock_dir']);lockdir.mkdir(exist_ok=True)
    for key in ('TMPDIR','TMP','TEMP'):
        Path(suite['environment'][key]).mkdir(exist_ok=True)
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
        if not failed:
            if apply_skips(suite,state):subprocess.run([suite['python'],str(HERE/'collect_results.py')],cwd=HERE,check=True)
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
                gate=evaluate(job,suite=suite,root=HERE)
                if gate['decision']!='ready':continue
                verify_parent_release()
                if not job['smoke'] and job.get('smoke_group'):verify_gpu_smoke_release(suite,job['smoke_group'])
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
                atomic_json(d/'gate_release.json',{'job_id':jid,'at':now(),'result':gate,'suite_sha256':sha256(HERE/'suite.json')})
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
