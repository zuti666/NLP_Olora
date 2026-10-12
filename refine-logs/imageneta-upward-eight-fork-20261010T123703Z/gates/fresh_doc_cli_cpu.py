#!/usr/bin/env python3
"""Run the five literal documented CPU commands without GPU admission."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

H = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    run = (H/'RUN.md').read_text()
    commands = [line for line in run.splitlines()
                if line.startswith('/home-local/liying/.conda/envs/Pilot_new_local/bin/python ')
                and ('trainer/test_upward_' in line or 'trainer/upward_' in line)]
    if len(commands) != 5 or sum('--dry-run' in c for c in commands) != 3:
        raise AssertionError('RUN.md must contain the two tests and three literal trainer dry-runs')
    if socket.gethostname() != 'cudahpc40.cvc.uab.es' or os.readlink('/proc/self/ns/pid') != 'pid:[4026531836]':
        raise AssertionError('Literal CPU checks require the documented actual host namespace')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def run_one(item):
        index, command = item
        before = time.monotonic()
        p = subprocess.run(shlex.split(command), cwd=H, env=env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = H/'gates'/('FRESH_LITERAL_CPU_%d.log'%index)
        path.write_text(p.stdout)
        row = dict(command=command, argv=shlex.split(command), cwd=str(H),
                   returncode=p.returncode, elapsed_seconds=time.monotonic()-before,
                   log=str(path), log_sha256=sha(path), gpu_visible=False)
        if '--dry-run' in command:
            payloads = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
            if not payloads or payloads[-1].get('status') != 'valid':
                row['payload_error']='missing valid CLI payload'
            else:
                row['payload']=payloads[-1]
        else:
            row['unittest_passed']='\nOK\n' in '\n'+p.stdout+'\n'
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        rows = list(executor.map(run_one, enumerate(commands)))
    passed = all(r['returncode'] == 0 and not r.get('payload_error') and
                 (r.get('unittest_passed') is not False) for r in rows)
    evidence = dict(status='passed' if passed else 'failed',
                    checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    hostname=socket.gethostname(), pid_namespace=os.readlink('/proc/self/ns/pid'),
                    literal_run_md_sha256=sha(H/'RUN.md'), environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                    commands=rows, gpu_training_or_admission_performed=False,
                    queue_dry_run_status='Deferred until root seals RUN_PINS and CPU_IMPLEMENTATION_ACCEPTED; no queue command executed by this reviewer')
    target=H/'gates/FRESH_LITERAL_CPU_EVIDENCE.json'
    target.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(dict(status=evidence['status'], commands=len(rows),
                         failures=[r['command'] for r in rows if r['returncode']!=0 or r.get('payload_error') or r.get('unittest_passed') is False],
                         evidence=str(target))))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
