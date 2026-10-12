#!/usr/bin/env python3
"""Wait for ALL validated parent jobs, run short tests, then release this cohort."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_json,now,process_matches,proc_identity,sha256,verify_source

HERE=Path(__file__).resolve().parent
def parent_ready(guard):
    if os.readlink('/proc/self/ns/pid')!=guard['execution_pid_namespace']:raise RuntimeError('host PID namespace mismatch')
    parent=Path(guard['parent_dir'])
    for name,key in (('suite.json','parent_suite_sha256'),('RUN_PINS.json','parent_run_pins_sha256')):
        if sha256(parent/name)!=guard[key]:raise RuntimeError('parent manifest changed '+name)
    suite=json.loads((parent/'suite.json').read_text());state=json.loads((parent/'runtime/queue_state.json').read_text())
    if len(suite['jobs'])!=guard['expected_parent_jobs'] or state['run_id']!=suite['run_id'] or set(state['jobs'])!={j['id'] for j in suite['jobs']}:
        raise RuntimeError('parent identity/count mismatch')
    if any(v['status']=='failed' for v in state['jobs'].values()) or state['status']=='halted_failure':
        raise RuntimeError('parent contains failed jobs; downstream not released')
    complete=state['status']=='completed' and all(v['status']=='completed' for v in state['jobs'].values())
    if not complete:
        if not process_matches(state['controller']):raise RuntimeError('parent controller exited before successful completion')
        return False,state
    if process_matches(state['controller']):return False,state
    pins=json.loads((parent/'RUN_PINS.json').read_text())
    for path,expected in pins['files'].items():
        if sha256(parent/path)!=expected:raise RuntimeError('parent source/config pin changed '+path)
    verify_source(suite)
    # Parent imports must not reuse this campaign's cached common.HERE.
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    completed=subprocess.run([suite['python'],'-I','-B',str(HERE/'validate_parent.py'),str(parent)],
        cwd=parent,env=env,capture_output=True,text=True,check=True)
    proof=json.loads(completed.stdout.strip().splitlines()[-1])
    if proof['status']!='validated' or proof['parent_dir']!=str(parent.resolve()) or proof['parent_jobs_validated']!=guard['expected_parent_jobs']:
        raise RuntimeError('isolated parent validation incomplete')
    if set(proof['receipts_sha256'])!=set(state['jobs']) or any(sha256(parent/'runtime/jobs'/jid/'completed.json')!=expected for jid,expected in proof['receipts_sha256'].items()):
        raise RuntimeError('isolated parent receipt proof changed')
    return True,state

def main():
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    control=open(runtime/'handoff.lock','a+')
    fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard=json.loads((HERE/'handoff_guard.json').read_text())
    suite=json.loads((HERE/'suite.json').read_text())
    status={'created_at':now(),'process':proc_identity(os.getpid()),'status':'waiting_parent','parent_dir':guard['parent_dir'],
            'no_GPU_allocated_while_waiting':True,'jobs_registered':len(suite['jobs'])}
    atomic_json(runtime/'handoff_state.json',status)
    while True:
        if (runtime/'PAUSE').exists():
            status.update(status='paused_handoff',updated_at=now());atomic_json(runtime/'handoff_state.json',status);time.sleep(30);continue
        ready,parent=parent_ready(guard)
        status.update(status='waiting_parent' if not ready else 'parent_validated',updated_at=now(),
            parent_counts={key:sum(v['status']==key for v in parent['jobs'].values()) for key in ('completed','running','pending','failed')})
        atomic_json(runtime/'handoff_state.json',status)
        if ready:break
        time.sleep(30)
    # No GPU-using process is created above this boundary.
    atomic_json(runtime/'PARENT_VALIDATED.json',{'status':'validated','at':now(),
        'guard_sha256':sha256(HERE/'handoff_guard.json'),'parent_jobs_validated':guard['expected_parent_jobs'],
        'parent_suite_sha256':guard['parent_suite_sha256'],'parent_run_pins_sha256':guard['parent_run_pins_sha256'],
        'parent_receipts_sha256':{jid:sha256(Path(guard['parent_dir'])/'runtime/jobs'/jid/'completed.json') for jid in parent['jobs']}})
    for stage,args in (('smoke',['--smoke-only']),('formal',[])):
        status.update(status='running_'+stage,updated_at=now());atomic_json(runtime/'handoff_state.json',status)
        with (runtime/f'controller_{stage}.log').open('xb') as log:
            subprocess.run([suite['python'],str(HERE/'run_queue.py'),'--suite',str(HERE/'suite.json'),*args],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        if stage=='smoke':
            subprocess.run([suite['python'],str(HERE/'check_gpu_smokes.py')],cwd=HERE,check=True)
    status.update(status='completed',updated_at=now());atomic_json(runtime/'handoff_state.json',status)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
        record=json.loads((runtime/'handoff_state.json').read_text()) if (runtime/'handoff_state.json').exists() else {}
        record.update(status='blocked_failure',updated_at=now(),error=repr(exc));atomic_json(runtime/'handoff_state.json',record)
        raise
