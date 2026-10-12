"""Keep index and UUID leases alive through CUDA witness and offline child."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
from queue_utils import atomic_json,identity,now,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--lease-fd',type=int,required=True);p.add_argument('--uuid-lease-fd',type=int,required=True)
    a=p.parse_args();job=read(a.job);folder=Path(a.job).parent;fds=(a.lease_fd,a.uuid_lease_fd)
    for fd in fds:os.fstat(fd)
    try:
        subprocess.run([sys.executable,str(Path(__file__).parent/'cuda_witness.py'),'--output',str(folder/'cuda_witness.json')],pass_fds=fds,check=True)
        child=subprocess.Popen(job['command'],cwd=job['cwd'],pass_fds=fds)
        atomic_json(folder/'runner.json',dict(process=identity(child.pid),started_at=now(),command=job['command'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
        rc=child.wait();atomic_json(folder/'exit.json',dict(job_id=job['id'],returncode=rc,runner_pid=child.pid,exited_at=now()));return rc
    except BaseException as exc:
        atomic_json(folder/'exit.json',dict(job_id=job['id'],returncode=-1,error=repr(exc),exited_at=now()));raise
if __name__=='__main__':raise SystemExit(main())
