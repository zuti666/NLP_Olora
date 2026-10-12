#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
from common import atomic_json,now,proc_identity,process_matches

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
suite=json.loads((HERE/'suite.json').read_text())
runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
record=runtime/'controller.json'
if record.exists() and process_matches(json.loads(record.read_text())['identity']): raise RuntimeError('controller already running')
if not a.smoke_only:
    verified=json.loads((runtime/'GPU_SMOKES_VERIFIED.json').read_text())
    if verified['status']!='passed' or verified['verified_smokes']!=12: raise RuntimeError('GPU cross-arm smoke gate not passed')
command=[suite['python'],str(HERE/'run_queue.py'),'--suite',str(HERE/'suite.json')]
if a.smoke_only:command+=['--smoke-only']
log=runtime/('controller_smoke.log' if a.smoke_only else 'controller_formal.log')
env=os.environ.copy();env.update(suite['environment'])
with log.open('xb') as handle:
    child=subprocess.Popen(command,cwd=HERE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':child.pid,'identity':proc_identity(child.pid),'at':now(),'command':command,'log':str(log)}
atomic_json(runtime/('launch_smoke.json' if a.smoke_only else 'launch_formal.json'),receipt)
print(json.dumps(receipt))
