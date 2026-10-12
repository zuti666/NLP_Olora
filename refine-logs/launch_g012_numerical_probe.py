"""Run an isolated, read-only numerical probe on idle physical GPU3."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--script', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('extra', nargs=argparse.REMAINDER)
args = parser.parse_args()
out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=False)
suite = json.loads(Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/suite.json').read_text())
def gpu_state():
    value = subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid, memory, util = [v.strip() for v in value.strip().split(',')]
    return {'physical_gpu':3,'uuid':uuid,'memory_used_mib':int(memory),'utilization_percent':int(util)}
before = gpu_state()
if before['memory_used_mib']>=500 or before['utilization_percent']>=5:
    raise RuntimeError('GPU3 is occupied; diagnostic not started')
locks=[]
for name in ('gpu3.lock',before['uuid']+'.lock'):
    handle=open(Path(suite['gpu_lock_dir'])/name,'a+')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append(handle)
current=gpu_state()
if current['uuid']!=before['uuid'] or current['memory_used_mib']>=500 or current['utilization_percent']>=5:
    raise RuntimeError('GPU3 changed before launch')
env=dict(os.environ)
env.update(suite['environment'])
env['CUDA_VISIBLE_DEVICES']='3'
extra=args.extra[1:] if args.extra[:1]==['--'] else args.extra
command=[suite['python'],str(Path(args.script).resolve()),'--output-dir',str(out/'artifacts'),'--device','cuda:0',*extra]
(out/'launch.json').write_text(json.dumps({'command':command,'gpu_before':before,'zero_training_updates':True,'queue_unchanged':True},indent=2)+'\n')
print(json.dumps({'event':'read_only_probe_launch','output':str(out),'physical_gpu':3}),flush=True)
with (out/'console.log').open('w') as log:
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        log.write(line);log.flush()
        print(line,end='',flush=True)
    code=process.wait()
(out/'exit.json').write_text(json.dumps({'returncode':code,'gpu_after':gpu_state()},indent=2)+'\n')
for handle in locks:handle.close()
sys.exit(code)
