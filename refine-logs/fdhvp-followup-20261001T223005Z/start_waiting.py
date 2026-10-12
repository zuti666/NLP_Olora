#!/usr/bin/env python3
"""Start only the host waiting process; parent barrier owns GPU release."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from common import assert_host,atomic_json,now,proc_identity,process_matches,sha256,verify_source

HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text())
guard=json.loads((HERE/'handoff_guard.json').read_text())
assert_host(suite)
if os.readlink('/proc/self/ns/pid')!=guard['execution_pid_namespace']:raise RuntimeError('host PID namespace mismatch')
pins=json.loads((HERE/'RUN_PINS.json').read_text())
for path,expected in pins['files'].items():
    if sha256(HERE/path)!=expected:raise RuntimeError('run pin changed '+path)
verify_source(suite)
runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
control=open(runtime/'launcher.lock','a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=runtime/'launch_waiting.json'
if previous.exists() and process_matches(json.loads(previous.read_text())['identity']):raise RuntimeError('waiting process already running')
if (runtime/'queue_state.json').exists():raise RuntimeError('fresh waiting launch only; existing state preserved')
atomic_json(runtime/'queue_state.json',{'run_id':suite['run_id'],'created_at':now(),'status':'waiting_parent','counts':{'pending':len(suite['jobs'])},'jobs':{j['id']:{'status':'pending'} for j in suite['jobs']}})
env=os.environ.copy();env.update(suite['environment'])
for key in ('TMPDIR','TMP','TEMP'):Path(env[key]).mkdir(exist_ok=True)
command=[suite['python'],str(HERE/'wait_then_launch.py')]
with (runtime/'waiting.log').open('xb') as log:
    child=subprocess.Popen(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'identity':proc_identity(child.pid),'at':now(),'command':command,'log':str(runtime/'waiting.log'),'PID_namespace':guard['execution_pid_namespace'],'jobs_registered':len(suite['jobs']),'GPU_jobs_started':0}
atomic_json(previous,receipt)
control.close()
print(json.dumps(receipt))
