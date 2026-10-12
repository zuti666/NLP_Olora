#!/usr/bin/env python3
"""Detached child holds GPU lease through runner exit and writes an exit receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import atomic_json,now

p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--lease-fd',type=int,required=True)
a=p.parse_args();job=json.loads(Path(a.job).read_text())
receipt=Path(a.job).parent/'exit.json'
try:
    child=subprocess.Popen(job['command'],pass_fds=(a.lease_fd,),cwd=job['cwd'])
    rc=child.wait()
    atomic_json(receipt,{'job_id':job['id'],'returncode':rc,'exited_at':now(),'runner_pid':child.pid})
except BaseException as exc:
    atomic_json(receipt,{'job_id':job['id'],'returncode':-1,'exited_at':now(),'error':repr(exc)})
    raise
sys.exit(rc)
