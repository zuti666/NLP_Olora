#!/usr/bin/env python3
"""Detached child holds GPU lease through runner exit and writes an exit receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import atomic_json,now,proc_identity

p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--lease-fd',type=int)
p.add_argument('--uuid-lease-fd',type=int)
a=p.parse_args();job=json.loads(Path(a.job).read_text())
receipt=Path(a.job).parent/'exit.json'
atomic_json(Path(a.job).parent/'worker_identity.json',{'job_id':job['id'],'process':proc_identity(os.getpid()),'started_at':now()})
try:
    held=() if not job['gpu_required'] else (a.lease_fd,a.uuid_lease_fd)
    if job['gpu_required']:
        if any(fd is None for fd in held):
            raise RuntimeError('GPU worker requires both shared GPU leases')
        witness=Path(__file__).parent/'trainer/cuda_witness.py'
        subprocess.run([sys.executable,str(witness),'--output',job['cuda_witness_output']],
                       pass_fds=held,check=True)
    elif os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('CPU pairing worker must hide all GPUs')
    child=subprocess.Popen(job['command'],pass_fds=held,cwd=job['cwd'])
    runner_process=proc_identity(child.pid)
    atomic_json(Path(a.job).parent/'runner_identity.json',{'job_id':job['id'],'process':runner_process,'started_at':now()})
    rc=child.wait()
    atomic_json(receipt,{'job_id':job['id'],'returncode':rc,'exited_at':now(),'runner_pid':child.pid,'runner_process':runner_process})
except BaseException as exc:
    atomic_json(receipt,{'job_id':job['id'],'returncode':-1,'exited_at':now(),'error':repr(exc)})
    raise
sys.exit(rc)
