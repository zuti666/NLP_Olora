import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
p=Path('/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution')
run=p/'campaign_seed0_after_gpu03'
if (run/'launcher.json').exists():
    raise RuntimeError('Launch receipt already exists; inspect the running queue instead of duplicating it')
command=['/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python','-u',str(p/'wait_then_run.py'),'--plan',str(p/'campaign_plan.json'),'--execute']
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (run/'supervisor.log').open('xb') as log:
    child=subprocess.Popen(command,cwd='/data/140-1/users/liying/O-LoRA',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
identity_fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
receipt={'pid':child.pid,'start_ticks':int(identity_fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'detached':True,'log':str(run/'supervisor.log'),'plan_sha256':hashlib.sha256((p/'campaign_plan.json').read_bytes()).hexdigest()}
(run/'launcher.json').write_text(json.dumps(receipt,indent=2)+'\n')
time.sleep(2)
assert child.poll() is None, (child.returncode,(run/'supervisor.log').read_text())
state=json.loads((run/'queue_state.json').read_text())
assert state['status']=='waiting',state
assert state['queue_identity']['pid']==child.pid
assert not state.get('pipeline_started_at'),state
assert not (p/'results_seed0').exists()
print(json.dumps({'launcher':receipt,'queue_status':state['status'],'predecessor_status':state['gate']['predecessor_status'],'pipeline_started':False,'gpu_experiments_started':False},indent=2))
