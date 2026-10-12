"""Detach the GPU0/1 training queue once; preserve all process identities."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
from campaign_runner import process_identity, validate_plan


def main():
    campaign = HERE / 'training_campaign'
    plan_path = campaign / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['gpus'] == [0, 1] and plan['max_parallel'] == 2
    assert len(validate_plan(plan)) == 6
    with (campaign / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = campaign / 'launcher.json'
        if receipt.exists() or (campaign / 'pipeline_state.json').exists():
            raise RuntimeError('Existing launch/state; do not duplicate training.')
        command = ['/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python', '-u',
                   str(HERE.parent / 'campaign_runner.py'), '--plan', str(plan_path), '--execute']
        env = dict(os.environ, TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONDONTWRITEBYTECODE='1')
        with (campaign / 'supervisor.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(pid=proc.pid, process_identity=process_identity(proc.pid), command=command,
                      started_utc=datetime.now(timezone.utc).isoformat(), detached=True,
                      authorized_gpus=[0, 1],
                      arms=[dict(probe=.05, norm=1.), dict(probe=1., norm=1.)],
                      plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                      log=str(campaign / 'supervisor.log'))
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
