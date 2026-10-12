"""Wait for all six new Task0 anchors, then run diagnostics only on GPU0/1."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
from campaign_runner import execute_plan, fingerprint, process_identity, validate_plan, write_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    campaign = HERE / 'diagnostic_campaign'
    plan_path = campaign / 'plan.json'
    plan = json.loads(plan_path.read_text())
    jobs = validate_plan(plan)
    assert plan['gpus'] == [0, 1] and plan['max_parallel'] == 2 and len(jobs) == 13
    frozen = fingerprint(plan['code_files'] + [str(plan_path)])
    if not args.execute:
        print(json.dumps(dict(dry_run=True, gpu_jobs=12, cpu_jobs=1, states=132,
                              authorized_gpus=[0, 1], fingerprints=len(frozen))))
        return
    with (campaign / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = campaign / 'queue_state.json'
        if state_path.exists():
            raise RuntimeError('Existing diagnostic queue; inspect instead of duplicating.')
        state = dict(status='waiting_for_training', pid=os.getpid(),
                     process_identity=process_identity(os.getpid()), created_utc=now(),
                     fingerprints=frozen, required_anchors=6, authorized_gpus=[0, 1])
        try:
            while True:
                if fingerprint(frozen) != frozen:
                    raise RuntimeError('Frozen diagnostic code or protocol changed while waiting.')
                parent = json.loads((HERE / 'training_campaign/pipeline_state.json').read_text())
                if parent['status'] in ('failed', 'draining_after_failure'):
                    raise RuntimeError('Training failed; refuse to measure incomplete anchors.')
                complete = sum(v['status'] == 'complete' for v in parent['jobs'].values())
                state.update(completed_anchors=complete, updated_utc=now())
                write_json(state_path, state)
                if parent['status'] == 'complete':
                    assert complete == len(parent['jobs']) == 6
                    assert all(v.get('receipt_valid') for v in parent['jobs'].values())
                    break
                time.sleep(30)
            state.update(status='diagnostics_running', updated_utc=now())
            write_json(state_path, state)
            os.environ['RQ2_QUEUE_LOCK_FD'] = str(lock.fileno())
            execute_plan(plan, plan_path)
            state.update(status='complete', updated_utc=now())
            write_json(state_path, state)
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), updated_utc=now())
            write_json(state_path, state)
            raise


if __name__ == '__main__':
    main()
