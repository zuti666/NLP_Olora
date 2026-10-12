"""Start this new isolated pipeline once, in the physical GPU host namespace."""
import json,os,subprocess,time
from pathlib import Path
from common import atomic_json,assert_host,load_suite,now,proc_identity,process_matches
H=Path(__file__).resolve().parent
s=load_suite();assert_host(s)
g=json.loads((H/'handoff_guard.json').read_text())
if os.readlink('/proc/self/ns/pid')!=g['execution_pid_namespace']:raise RuntimeError('host PID namespace required')
(H/'runtime').mkdir(exist_ok=True)
receipt=H/'runtime/pipeline_launch.json'
if receipt.exists():raise RuntimeError('already launched: inspect prior process/state; do not duplicate')
log=open(H/'runtime/pipeline.log','xb');env=os.environ.copy();env.update(s['environment']);env.pop('CUDA_VISIBLE_DEVICES',None)
for k in ['TMPDIR','TMP','TEMP']:Path(env[k]).mkdir(exist_ok=True)
p=subprocess.Popen([s['python'],'-u',str(H/'launch_pipeline.py')],cwd=H,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close();r=dict(started_at=now(),identity=proc_identity(p.pid),command=[s['python'],'-u',str(H/'launch_pipeline.py')]);atomic_json(receipt,r)
time.sleep(1)
if p.poll() is not None:raise RuntimeError('pipeline exited immediately, inspect pipeline.log')
print(json.dumps(r))
