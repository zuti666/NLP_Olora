"""Detach the user-authorized GPU0/3 radius extension once, with durable logs."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / 'campaign'
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


with (CAMPAIGN / 'launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = CAMPAIGN / 'launcher.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if identity(prior['pid']) == prior['process_identity']:
            raise RuntimeError('This radius-extension controller is already running')
        raise RuntimeError('Prior launch exists; inspect state and explicitly resume instead')
    command = [PYTHON, '-u', str(HERE / 'queue_radius.py'), '--execute']
    env = dict(os.environ, TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (CAMPAIGN / 'supervisor.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=HERE.parents[3], env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    info = dict(pid=proc.pid, process_identity=identity(proc.pid), command=command,
                started_utc=datetime.now(timezone.utc).isoformat(),
                plan_sha256=hashlib.sha256((CAMPAIGN / 'plan.json').read_bytes()).hexdigest(),
                detached=True, log=str(CAMPAIGN / 'supervisor.log'))
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
