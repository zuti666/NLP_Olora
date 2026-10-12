"""Durable CPU finalizer for the separately locked historical GPU queue."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / 'campaign'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def main():
    manifest = json.loads((CAMPAIGN / 'finalizer_manifest.json').read_text())
    status_path = CAMPAIGN / 'finalizer_state.json'
    with (CAMPAIGN / 'finalizer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(value, **extra):
            write(status_path, dict(status=value, pid=os.getpid(),
                                    utc=datetime.now(timezone.utc).isoformat(), **extra))
        while True:
            state = json.loads((CAMPAIGN / 'pipeline_state.json').read_text())
            jobs = state['jobs']
            completed = sum(j['status'] == 'complete' for j in jobs.values())
            if state['status'] == 'complete':
                if completed != 16 or len(jobs) != 16:
                    raise RuntimeError('Unexpected queue coverage')
                break
            if state['status'] == 'failed':
                status('queue_failed', new_jobs_complete=completed, error=state.get('error'))
                return 1
            live = identity(state['pid'])
            if live is None or live != state['process_identity']:
                status('queue_stopped_needs_review', new_jobs_complete=completed)
                return 1
            status('waiting_for_measurements', new_jobs_complete=completed, total_new_jobs=16)
            time.sleep(30)
        for path, expected in manifest['fingerprints'].items():
            if sha(path) != expected:
                raise RuntimeError(f'Finalizer input changed: {path}')
        status('collecting_and_plotting', new_jobs_complete=16)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONDONTWRITEBYTECODE='1')
        with (CAMPAIGN / 'postprocess.log').open('ab', buffering=0) as log:
            subprocess.run(manifest['command'], cwd=HERE, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = HERE / 'report/completeness.json'
        result = json.loads(receipt.read_text())
        if result['status'] != 'complete' or not result.get('figure_generated_this_run'):
            raise RuntimeError('Full report/figure did not pass validation')
        spec_path = HERE / 'extension_spec.json'
        spec = json.loads(spec_path.read_text())
        spec.update(status='complete', report=str(HERE / 'report'),
                    completed_utc=datetime.now(timezone.utc).isoformat())
        for job in spec['jobs']:
            job['status'] = 'complete'
        write(spec_path, spec)
        status('complete', new_jobs_complete=16, reused_legacy_paths=2,
               report=str(HERE / 'report'), completeness_sha256=sha(receipt))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        write(CAMPAIGN / 'finalizer_state.json',
              dict(status='failed', pid=os.getpid(), error=repr(exc),
                   utc=datetime.now(timezone.utc).isoformat()))
        raise
