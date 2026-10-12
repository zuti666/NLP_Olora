"""Read-only predecessor gate; only this new campaign's receipt/status are written."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'radius50_extension_20260921'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    state_path = PREVIOUS / 'campaign/pipeline_state.json'
    receipt_path = PREVIOUS / 'report/radius50_complete.json'
    while True:
        state = json.loads(state_path.read_text())
        status = state.get('status')
        now = datetime.now(timezone.utc).isoformat()
        wait_status = dict(status='waiting_for_random_radius50', checked_utc=now,
                           predecessor_status=status, predecessor_state=str(state_path),
                           poll_seconds=30, gpu_work_started=False)
        write(HERE / 'campaign/predecessor_wait.json', wait_status)
        if status in ('failed', 'draining_after_failure'):
            raise RuntimeError('Random-radius predecessor failed; preserve its evidence before launching successor')
        if status == 'complete':
            receipt = json.loads(receipt_path.read_text())
            assert receipt['status'] == 'complete'
            assert receipt['expected_jobs'] == receipt['validated_jobs'] == 12
            assert receipt['automatic_numeric_and_layout_checks'] == 'passed'
            assert all(j['status'] == 'complete' and j.get('receipt_valid') for j in state['jobs'].values())
            for item in receipt['output_files']:
                assert digest(item['path']) == item['sha256'], ('Predecessor output changed', item['path'])
            result = dict(status='complete', completed_utc=now, predecessor_state=str(state_path),
                          predecessor_state_sha256=digest(state_path), predecessor_receipt=str(receipt_path),
                          predecessor_receipt_sha256=digest(receipt_path), predecessor_gpu_jobs=12,
                          all_predecessor_jobs_valid=True, predecessor_outputs_verified=True)
            write(HERE / 'campaign/predecessor_complete.json', result)
            write(HERE / 'campaign/predecessor_wait.json', result)
            print(json.dumps(result), flush=True)
            return
        print(json.dumps(wait_status), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
