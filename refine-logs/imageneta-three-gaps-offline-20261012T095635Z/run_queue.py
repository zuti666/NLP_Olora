"""Durable, max-three, GPU4/5/6-only read-only diagnostic queue."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.dont_write_bytecode=True
from queue_utils import atomic_json,alive,identity,idle,now,query_gpus,read,sha,validate_output
HERE=Path(__file__).resolve().parent

def verify(suite):
    manifest=read(HERE/'implementation_manifest.json')
    for path,expected in manifest['files'].items():
        if sha(path)!=expected:raise RuntimeError('Pinned implementation/input changed: '+path)
    for relative,expected in read(suite['source_manifest'])['files'].items():
        p=(Path(suite['source_root'])/relative).resolve()
        if not p.is_relative_to(Path(suite['source_root'])) or sha(p)!=expected:raise RuntimeError('Frozen historical source changed: '+relative)
    if suite['allowed_gpus']!=[4,5,6] or suite['max_parallel']!=3:raise ValueError('Only authorized GPU4/5/6 max3')
    identifiers={j['id'] for j in suite['jobs']}
    if len(identifiers)!=len(suite['jobs']):raise ValueError('Duplicate jobs')
    for j in suite['jobs']:
        if set(j['depends_on'])-identifiers or j['id'] in j['depends_on']:raise ValueError('Invalid dependency')
        if Path(j['output_dir']).resolve()!=Path(suite['bulk_output_root']).resolve()/j['id']:raise ValueError('Nonexclusive output path')
        if j['command'][0]!=suite['python'] or j['command'][1]!=str(HERE/(j['kind_script'])):raise ValueError('Unregistered command')
    unresolved=set(identifiers)
    while unresolved:
        ready={j['id'] for j in suite['jobs'] if j['id'] in unresolved and not set(j['depends_on'])&unresolved}
        if not ready:raise ValueError('Dependency cycle')
        unresolved-=ready
    if 'CUDA_VISIBLE_DEVICES' in suite['environment']:raise ValueError('Device binding is worker-exclusive')

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',default=str(HERE/'suite.json'));p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    suite=read(a.suite);verify(suite)
    if a.dry_run:
        print(__import__('json').dumps(dict(status='accepted_dry_run',jobs=len(suite['jobs']),gpus=suite['allowed_gpus'],pins=len(read(HERE/'implementation_manifest.json')['files']),training_updates=0)));return 0
    if socket.gethostname()!=suite['hostname'] or os.readlink('/proc/self/ns/pid')!=suite['pid_namespace']:raise RuntimeError('Use registered host PID namespace')
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    controller=open(runtime/'controller.lock','a+');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=runtime/'queue_state.json'
    state=read(path) if path.exists() else dict(status='new',jobs={j['id']:dict(status='pending') for j in suite['jobs']},suite_sha256=sha(a.suite))
    if state['suite_sha256']!=sha(a.suite):raise RuntimeError('Registered suite changed')
    byid={j['id']:j for j in suite['jobs']};active={};leases={}
    for jid,row in state['jobs'].items():
        if row['status']=='completed':
            if validate_output(byid[jid])['report_sha256']!=row['report_sha256']:raise RuntimeError('Completed evidence changed')
        elif row['status']=='running':active[jid]=None
        elif row['status']=='launching':raise RuntimeError('Incomplete launch requires audit: '+jid)
    atomic_json(runtime/'controller.json',dict(process=identity(os.getpid()),started_at=now(),hostname=socket.gethostname(),pid_namespace=os.readlink('/proc/self/ns/pid'),command=sys.argv))
    lockdir=Path(suite['gpu_lock_dir']);lockdir.mkdir(exist_ok=True)
    for key in ('TMPDIR','TMP','TEMP','MPLCONFIGDIR'):Path(suite['environment'][key]).mkdir(parents=True,exist_ok=True)
    while True:
        for jid in list(active):
            worker=active[jid];row=state['jobs'][jid]
            if worker is not None and worker.poll() is None:continue
            if worker is None and alive(row['process']):continue
            try:
                d=runtime/'jobs'/jid;exited=read(d/'exit.json')
                if exited['returncode']!=0 or exited['job_id']!=jid:raise RuntimeError('Offline worker failed: '+repr(exited))
                verify(suite);checked=validate_output(byid[jid]);witness=read(d/'cuda_witness.json')
                if witness['status']!='passed' or witness['cuda_visible_devices']!=row['gpu_uuid'] or witness['tf32_matmul'] or witness['tf32_cudnn']:raise RuntimeError('CUDA dispatch witness mismatch')
                row.update(status='completed',completed_at=now(),**checked);atomic_json(d/'completed.json',row)
                print(__import__('json').dumps(dict(event='completed',job=jid,items=checked['items'],scientific_outcome=checked['scientific_outcome'])),flush=True)
            except Exception as exc:
                row.update(status='failed',error=repr(exc),failed_at=now());atomic_json(runtime/'jobs'/jid/'failed.json',row)
                print(__import__('json').dumps(dict(event='failed',job=jid,error=repr(exc))),flush=True)
            finally:
                active.pop(jid)
                for lock in leases.pop(jid,[]):lock.close()
                atomic_json(path,state)
        pending=[j for j in suite['jobs'] if state['jobs'][j['id']]['status']=='pending']
        failed=any(x['status']=='failed' for x in state['jobs'].values())
        paused=(runtime/'PAUSE').exists()
        if not active and (failed or not pending):
            state.update(status='halted_failure' if failed else 'completed',updated_at=now(),counts=dict(Counter(x['status'] for x in state['jobs'].values())))
            atomic_json(path,state);return 1 if failed else 0
        if not failed and not paused:
            gpus=query_gpus();used={state['jobs'][jid]['gpu'] for jid in active}
            for j in pending:
                if len(active)>=suite['max_parallel']:break
                if any(state['jobs'][dep]['status']!='completed' for dep in j['depends_on']):continue
                for dep in j['depends_on']:
                    if validate_output(byid[dep])['report_sha256']!=state['jobs'][dep]['report_sha256']:raise RuntimeError('Dependency receipt changed')
                verify(suite);choice=None
                for index in suite['allowed_gpus']:
                    gpu=gpus.get(index)
                    if index in used or not idle(gpu) or gpu['uuid']!=suite['gpu_uuids'][str(index)]:continue
                    held=[]
                    for name in ('gpu%d.lock'%index,gpu['uuid']+'.lock'):
                        lock=open(lockdir/name,'a+')
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:lock.close();break
                        held.append(lock)
                    fresh=query_gpus().get(index) if len(held)==2 else None
                    if len(held)!=2 or not idle(fresh) or fresh['uuid']!=gpu['uuid']:
                        for lock in held:lock.close()
                        continue
                    choice=(index,fresh,held);break
                if choice is None:break
                index,gpu,held=choice;jid=j['id'];d=runtime/'jobs'/jid;d.mkdir(parents=True,exist_ok=False)
                output=Path(j['output_dir'])
                if output.exists() and any(output.iterdir()):
                    for lock in held:lock.close()
                    raise RuntimeError('Refusing nonempty output: '+jid)
                atomic_json(d/'job.json',dict(j,cwd=str(HERE)))
                row=state['jobs'][jid];row.update(status='launching',started_at=now(),gpu=index,gpu_uuid=gpu['uuid'],command=j['command']);atomic_json(path,state)
                env=os.environ.copy();env.update(suite['environment']);env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
                with open(d/'console.log','xb') as log:
                    worker=subprocess.Popen([suite['python'],str(HERE/'job_worker.py'),'--job',str(d/'job.json'),'--lease-fd',str(held[0].fileno()),'--uuid-lease-fd',str(held[1].fileno())],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(x.fileno() for x in held))
                row.update(status='running',process=identity(worker.pid));atomic_json(d/'launched.json',row);atomic_json(path,state)
                active[jid]=worker;leases[jid]=held;used.add(index)
                print(__import__('json').dumps(dict(event='launched',job=jid,gpu=index,pid=worker.pid)),flush=True)
        state.update(updated_at=now(),controller=identity(os.getpid()),status='halted_failure_draining' if failed else 'paused' if paused else 'running' if active else 'waiting_for_idle_gpu',counts=dict(Counter(x['status'] for x in state['jobs'].values())))
        atomic_json(path,state);time.sleep(suite['poll_seconds'])
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        atomic_json(HERE/'runtime/controller_error.json',dict(at=now(),error=repr(exc)));raise
