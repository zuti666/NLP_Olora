"""Follow the pinned documented invocation; short kernel witness only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from common import atomic_json,now,query_gpus

H=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu",type=int,choices=(2,3),required=True)
    a=ap.parse_args()
    suite=json.loads((H/"suite.json").read_text())
    spec=json.loads((H/"environment_spec.json").read_text())
    env=os.environ.copy();env.update(suite["environment"])
    dry=subprocess.run([suite["python"],str(H/"run_queue.py"),"--dry-run"],env=env,cwd=H,
                       capture_output=True,text=True,check=True)
    print(dry.stdout,end="",flush=True)
    gpu=query_gpus()[a.gpu]
    if gpu["memory_used_mb"]>=500 or gpu["utilization_percent"]>=5:
        raise RuntimeError("Requested GPU is busy; no preemption")
    handles=[]
    try:
        for name in (f"gpu{a.gpu}.lock",gpu["uuid"]+".lock"):
            handle=open(Path(suite["gpu_lock_dir"])/name,"a+")
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
        current=query_gpus()[a.gpu]
        if current["uuid"]!=gpu["uuid"] or current["memory_used_mb"]>=500 or current["utilization_percent"]>=5:
            raise RuntimeError("GPU admission changed")
        env["CUDA_VISIBLE_DEVICES"]=gpu["uuid"]
        test=spec["smoke"]["gpu_tests"][0]
        proc=subprocess.run(shlex.split(test["cmd"]),env=env,cwd=H,capture_output=True,text=True,check=True)
        print(proc.stdout,end="",flush=True)
        if not re.search(test["expect"],proc.stdout,re.MULTILINE):
            raise AssertionError("Canonical kernel sentinel absent")
        report=dict(status="passed",at=now(),physical_gpu=a.gpu,gpu_uuid=gpu["uuid"],
                    dry_run=json.loads(dry.stdout),canonical_witness_stdout=proc.stdout,
                    gpu0_gpu1_excluded=True,optimizer_updates=0)
        atomic_json(H/"gates/DOCUMENTED_INVOCATION_WITNESS.json",report)
    finally:
        for handle in handles:handle.close()

if __name__=="__main__":main()
