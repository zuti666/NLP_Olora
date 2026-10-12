#!/usr/bin/env python3
"""Launch one detached controller after validated local preflight."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'


def main():
    if (HERE / 'LAUNCH.json').exists() or (HERE / 'campaign/pipeline_state.json').exists():
        raise RuntimeError('Existing launch/state found: inspect it before any restart.')
    check = json.loads((HERE / 'PRELAUNCH_CHECKS.json').read_text())
    if check['status'] != 'passed':
        raise RuntimeError('Prelaunch checks not passed.')
    for path, expected in check['frozen_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Prelaunch input changed: ' + path)
    command = [PYTHON, '-u', str(HERE / 'campaign_runner.py'), '--plan', str(HERE / 'plan.json'), '--execute']
    env = dict(os.environ, TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with (HERE / 'controller.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    fields = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()
    result = dict(status='launched', pid=proc.pid, start_ticks=int(fields[19]),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  command=command, time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  policy='all idle GPUs 0-6; prefer arms 0/1/2/5; no all-GPU barrier; poll900s',
                  prelaunch_sha256=hashlib.sha256((HERE / 'PRELAUNCH_CHECKS.json').read_bytes()).hexdigest())
    (HERE / 'LAUNCH.json').write_text(json.dumps(result, indent=2) + '\n')
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Controller exited immediately; inspect controller.log.')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
