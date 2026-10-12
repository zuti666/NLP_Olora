"""Detach the authorized four-lane queue without inheriting terminal pipes."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'launch.json').exists(), 'Already launched; inspect state and children before recovery'
suite=json.loads((ROOT/'suite.json').read_text())
subprocess.run([suite['python'],str(ROOT/'launch_queue.py'),'--dry-run'],check=True)
env=os.environ.copy();env.update(suite['environment'])
with (ROOT/'queue.log').open('x') as log:
    p=subprocess.Popen([suite['python'],str(ROOT/'launch_queue.py')],cwd=ROOT,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,started_unix=time.time(),command=p.args,gpus=suite['allowed_gpus'])
    (ROOT/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
