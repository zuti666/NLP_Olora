"""One scoped, idle-GPU initial feasibility gate under the shared lease."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
from common import query_gpus

H=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu",type=int,required=True,choices=(2,3,4,5,6))
    ap.add_argument("--backend",choices=("native","math"),required=True)
    a=ap.parse_args()
    suite=json.loads((H/"suite.json").read_text())
    gpu=query_gpus()[a.gpu]
    if gpu["memory_used_mb"]>=500 or gpu["utilization_percent"]>=5:
        raise RuntimeError("Requested GPU is occupied; no preemption")
    locks=[]
    lockdir=Path(suite["gpu_lock_dir"]);lockdir.mkdir(exist_ok=True)
    for name in (f"gpu{a.gpu}.lock",gpu["uuid"]+".lock"):
        lock=open(lockdir/name,"a+")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    new=query_gpus()[a.gpu]
    if new["uuid"]!=gpu["uuid"] or new["memory_used_mb"]>=500 or new["utilization_percent"]>=5:
        raise RuntimeError("GPU changed or became occupied")
    env=os.environ.copy();env.update(suite["environment"])
    env["CUDA_VISIBLE_DEVICES"]=gpu["uuid"]
    output=H/"gates"/("initial_"+a.backend)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Gate output already exists")
    output.mkdir(parents=True)
    (H/"logs").mkdir(exist_ok=True)
    with (H/"logs"/("initial_"+a.backend+".log")).open("xb") as log:
        command=[suite["python"],str(H/"trainer/hvp_gate.py"),"--source",suite["source_root"],
                 "--config",str(H/"configs/S1-07_A_P_natural_s2.json"),"--output-dir",str(output),
                 "--backend",a.backend,"--tasks","0","--max-batches","1"]
        subprocess.run(command,env=env,cwd=H,stdout=log,stderr=subprocess.STDOUT,
                       pass_fds=tuple(lock.fileno() for lock in locks),check=True)
    print("INITIAL_GATE_PASSED",a.backend,a.gpu,flush=True)

if __name__=="__main__":
    main()
