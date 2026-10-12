"""Wait for the already launched smoke queue; launch formal jobs only after verification."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from common import (HERE, atomic_json, exclusive_json, load_suite, now,
                    process_matches, proc_identity, read_json, sha256, validate)


def verify_pinned_files():
    manifest = read_json(HERE / 'gate_manifest.json')
    for rel, expected in manifest['files'].items():
        if sha256(HERE / rel) != expected:
            raise RuntimeError('Pipeline file changed: ' + rel)


def smoke_decision(expected_ids, rows):
    if len(rows) != len(expected_ids) or {r['id'] for r in rows} != set(expected_ids):
        raise RuntimeError('Smoke result job set differs from its reviewed suite')
    if any(r['status'] in {'failed', 'invalid'} for r in rows):
        raise RuntimeError('At least one integration smoke failed; formal launch prohibited')
    return all(row['status'] == 'completed' for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    verify_pinned_files()
    suite = load_suite()
    validate(suite)
    smoke = HERE / suite.get('integration_campaign', 'preflight/smoke_campaign')
    smoke_suite = load_suite(smoke / 'suite.json')
    expected = [j['id'] for j in smoke_suite['jobs']]
    if len(expected) != 20 or len(suite['jobs']) != 18:
        raise RuntimeError('Reviewed campaign requires 20 integration smokes and 18 full runs')
    if args.validate_only:
        print(json.dumps({'passed': True, 'formal_jobs': len(suite['jobs']), 'smoke_jobs': len(expected)}))
        return 0
    lock = open(HERE / 'pipeline.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = read_json(smoke / 'controller_launch.json')
    try:
        while True:
            verify_pinned_files()
            result_file = smoke / 'results.json'
            rows = read_json(result_file)['jobs'] if result_file.exists() else []
            completed = sum(r['status'] == 'completed' for r in rows)
            running = sum(r['status'] == 'running' for r in rows)
            if rows and smoke_decision(expected, rows):
                break
            if not process_matches(controller):
                raise RuntimeError('Smoke controller exited before all smoke runs completed')
            atomic_json(HERE / 'PIPELINE_STATUS.json', {
                'updated_at': now(), 'phase': 'integration_running' if running else 'waiting_for_gpus_for_integration',
                'smoke_completed': completed, 'smoke_total': len(expected), 'smoke_running': running,
                'formal_started': False, 'allowed_gpus': suite['allowed_gpus'],
                'controller': proc_identity(os.getpid()), 'smoke_controller': controller,
            })
            time.sleep(15)
        # The verifier recomputes receipt hashes, every arm's effect, and exact
        # observer-on/off A/B/FC equality. Failure is fatal to formal launch.
        with (HERE / 'preflight/smoke_verification.log').open('xb') as log:
            subprocess.run([suite['python'], str(HERE / 'preflight/verify_smoke.py')],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        review = read_json(HERE / 'preflight/smoke_review.json')
        if not review.get('passed'):
            raise RuntimeError('Smoke verifier did not issue a passed receipt')
        verify_pinned_files()
        with (HERE / 'formal_queue.log').open('xb') as log:
            process = subprocess.Popen([suite['python'], '-u', str(HERE / 'launch_queue.py'),
                                        '--execute', '--poll-seconds', '30'], cwd=HERE,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        identity = proc_identity(process.pid)
        if not identity:
            raise RuntimeError('Formal controller exited at launch')
        receipt = {'launched_at': now(), **identity, 'suite_sha256': sha256(HERE / 'suite.json'),
                   'smoke_review_sha256': sha256(HERE / 'preflight/smoke_review.json')}
        exclusive_json(HERE / 'formal_controller_launch.json', receipt)
        while process.poll() is None:
            atomic_json(HERE / 'PIPELINE_STATUS.json', {
                'updated_at': now(), 'phase': 'formal_queue_active', 'smoke_completed': len(expected),
                'smoke_total': len(expected), 'formal_started': True,
                'formal_controller': receipt, 'allowed_gpus': suite['allowed_gpus'],
            })
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f'Formal queue or final analysis exited {process.returncode}; inspect receipts')
        atomic_json(HERE / 'PIPELINE_STATUS.json', {
            'updated_at': now(), 'phase': 'completed', 'smoke_completed': len(expected),
            'formal_started': True, 'formal_completed': len(suite['jobs']),
        })
        return 0
    except BaseException as exc:
        atomic_json(HERE / 'PIPELINE_STATUS.json', {
            'updated_at': now(), 'phase': 'stopped_on_error', 'error': repr(exc),
            'formal_started': (HERE / 'formal_controller_launch.json').exists(),
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
