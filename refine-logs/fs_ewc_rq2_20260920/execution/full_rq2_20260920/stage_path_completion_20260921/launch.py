"""Detach the authorized formal diagnostic extension once; no training."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from campaign_runner import process_identity


def main():
    campaign = HERE / 'campaign'
    with (campaign / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = campaign / 'launcher.json'
        if receipt.exists():
            raise RuntimeError('Launch receipt exists: inspect queue state and explicitly resume, never duplicate launch')
        python = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
        command = [python, '-u', str(HERE/'queue_extension.py'), '--execute']
        env = dict(os.environ, TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONDONTWRITEBYTECODE='1')
        with (campaign/'supervisor.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = dict(pid=proc.pid, process_identity=process_identity(proc.pid), command=command,
                    started_utc=datetime.now(timezone.utc).isoformat(), detached=True,
                    plan_sha256=hashlib.sha256((campaign/'plan.json').read_bytes()).hexdigest(),
                    log=str(campaign/'supervisor.log'))
        receipt.write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info,indent=2))


if __name__ == '__main__':
    main()
