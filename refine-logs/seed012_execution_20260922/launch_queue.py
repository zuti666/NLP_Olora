"""Four independent shared-GPU lanes, success dependencies, durable receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from run_job import save,sha

ROOT=Path(__file__).resolve().parent
def available(suite,active):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=20)
    busy=set(processes.split())|{v['gpu']['uuid']for v in active.values()}
    result=[]
    for line in text.splitlines():
        idx,uid,mem,util=[x.strip()for x in line.split(',')]
        if int(idx)in suite['allowed_gpus']and uid not in busy and int(mem)<500 and int(util)<5:
            result.append(dict(index=int(idx),uuid=uid))
    return result
def pid_running(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    suite=json.loads((ROOT/'suite.json').read_text());jobs=suite['jobs'];ids={j['id']for j in jobs}
    assert suite['allowed_gpus']==[2,3,4,5]and suite['max_parallel']==4
    assert len(ids)==len(jobs)
    resolved=set()
    while len(resolved)<len(ids):
        ready={j['id']for j in jobs if set(j['depends_on'])<=resolved}-resolved
        assert ready, 'Cyclic or missing success dependency'
        resolved.update(ready)
    for j in jobs:
        assert sha(j['config'])==j['config_sha256'],j['id']
        assert sha(j['source_manifest'])==j['source_manifest_sha256'],j['id']
        assert sha(j['data_manifest'])==j['data_manifest_sha256'],j['id']
        assert set(j['depends_on'])<=ids
        assert Path(j['source_root']).is_dir()
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    assert json.loads((ROOT/'env_witness.json').read_text())['passed']
    for relative,digest in json.loads((ROOT/'runtime_frozen.json').read_text())['files'].items():assert sha(ROOT/relative)==digest,relative
    if args.dry_run:
        print(json.dumps(dict(dry_run='PASS',jobs=len(jobs),formal=sum(j['stage']=='formal'for j in jobs),gpus=suite['allowed_gpus'])));return
    lock=(ROOT/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'state.json'
    if path.exists():
        assert args.resume,'Use --resume only after checking previous controller and children'
        state=json.loads(path.read_text());assert state['suite_sha256']==sha(ROOT/'suite.json')
        for name,row in state['jobs'].items():
            if row['status']=='running':
                assert not pid_running(row['pid']),f'Child {name} still running; refuse duplicate controller'
                receipt=ROOT/'receipts'/f'{name}.json';r=json.loads(receipt.read_text())if receipt.exists()else{}
                row['status']='completed'if r.get('status')=='completed'and sha(r['metric_path'])==r['metric_sha256']else'failed'
        state.update(pid=os.getpid(),status='running',resumed_unix=time.time())
    else:
        state=dict(pid=os.getpid(),status='running',started_unix=time.time(),suite_sha256=sha(ROOT/'suite.json'),
            jobs={j['id']:dict(status='pending',stage=j['stage'],method=j['method'],dataset=j['dataset'],seed=j['seed'])for j in jobs})
    active={};save(path,state)
    while True:
        for name,item in list(active.items()):
            code=item['proc'].poll()
            if code is None:continue
            item['log'].close();item['gpu_lock'].close()
            receipt=ROOT/'receipts'/f'{name}.json';r=json.loads(receipt.read_text())if receipt.exists()else{}
            okay=code==0 and r.get('status')=='completed'and sha(r['metric_path'])==r['metric_sha256']
            state['jobs'][name].update(status='completed'if okay else'failed',exit_code=code,finished_unix=time.time())
            if not okay:state['jobs'][name]['error']=r.get('error','missing successful completion receipt')
            del active[name];print('FINISH',name,code,flush=True)
        for job in jobs:
            row=state['jobs'][job['id']]
            if row['status']!='pending':continue
            deps=[state['jobs'][k]['status']for k in job['depends_on']]
            if any(x in ['failed','held_dependency_failure']for x in deps):row['status']='held_dependency_failure';continue
            if not all(x=='completed'for x in deps):continue
            if len(active)>=4:break
            free=available(suite,active)
            if not free:break
            gpu=None;gpu_lock=None
            for candidate in free:
                handle=Path('/tmp',f"codex_cv_gpu_{candidate['uuid']}.lock").open('a+')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                if candidate not in available(suite,active):handle.close();continue
                gpu=candidate;gpu_lock=handle;break
            if gpu is None:break
            env=os.environ.copy();env.update(suite['environment']);env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['PYTHONHASHSEED']=str(job['seed']);env['PYTHONDONTWRITEBYTECODE']='1'
            log=(ROOT/'logs'/f"{job['id']}.log").open('x')
            command=[suite['python'],str(ROOT/'run_job.py'),'--job',job['id']]
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[job['id']]=dict(proc=proc,log=log,gpu=gpu,gpu_lock=gpu_lock)
            row.update(status='running',pid=proc.pid,gpu=gpu,started_unix=time.time(),command=command)
            save(path,state);print('START',job['id'],gpu,proc.pid,flush=True)
        state['updated_unix']=time.time();save(path,state)
        collected=subprocess.run([suite['python'],str(ROOT/'collect.py')],cwd=ROOT,stdout=subprocess.DEVNULL)
        if collected.returncode:print('COLLECTOR_ERROR',collected.returncode,flush=True)
        if not active and not any(r['status']=='pending'for r in state['jobs'].values()):break
        time.sleep(15)
    state.update(status='completed'if all(r['status']=='completed'for r in state['jobs'].values())else'partial_failure',finished_unix=time.time());save(path,state)
    subprocess.run([suite['python'],str(ROOT/'collect.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
