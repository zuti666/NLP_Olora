"""Apply an authorized GPU-pool change to an entirely pending campaign."""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone

STAGE = Path(__file__).resolve().parent
CAM = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920')
SMOKE = CAM / 'preflight/smoke_attempt2'
PYTHON = '/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
GPUS = list(range(7))
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else {'pid': pid, 'proc_starttime': fields[19]}
    except FileNotFoundError:
        return None


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.gpu_pool_update.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def main():
    expected = read(STAGE / 'expected_before.json')
    for rel, h in expected.items():
        assert sha(CAM / rel) == h, rel
    for rel, h in read(CAM / 'gate_manifest.json')['files'].items():
        assert sha(CAM / rel) == h, rel
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], text=True)
    assert sorted(int(x) for x in actual.split()) == GPUS
    for rel in ('suite.json', 'preflight/smoke_attempt2/suite.json'):
        old = read(CAM / rel)
        new = read(STAGE / 'files' / rel)
        assert {k: v for k, v in old.items() if k not in ('allowed_gpus', 'max_parallel')} == {
            k: v for k, v in new.items() if k not in ('allowed_gpus', 'max_parallel')}
    old_controllers = [
        {'pid': 1708363, 'proc_starttime': '39126127', 'script': CAM / 'pipeline_gate.py'},
        {'pid': 1699290, 'proc_starttime': '39082749', 'script': SMOKE / 'launch_queue.py'},
    ]
    # Freeze only the two precisely identified controllers; abort/resume if a
    # job started since inspection, so no live run loses its suite provenance.
    frozen = []
    try:
        for rec in old_controllers:
            assert identity(rec['pid']) == {k: rec[k] for k in ('pid', 'proc_starttime')}
            assert str(rec['script']).encode() in Path(f"/proc/{rec['pid']}/cmdline").read_bytes().split(b'\0')
            os.kill(rec['pid'], signal.SIGSTOP)
            frozen.append(rec['pid'])
        assert not (CAM / 'formal_controller_launch.json').exists()
        assert not (CAM / 'preflight/smoke_review.json').exists()
        assert not (CAM / 'preflight/smoke_verification.log').exists()
        assert not (CAM / 'runtime/state.json').exists()
        assert not list((SMOKE / 'runtime/jobs').iterdir())
        state = read(SMOKE / 'runtime/state.json')
        assert len(state['jobs']) == 20 and all(r['status'] == 'pending' for r in state['jobs'].values())
    except BaseException:
        for pid in frozen:
            if identity(pid):
                os.kill(pid, signal.SIGCONT)
        raise
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = CAM / 'pipeline_history' / ('gpu_pool_before_all_' + timestamp)
    backup.mkdir(parents=True, exist_ok=False)
    snapshot = list(expected) + [
        'PIPELINE_STATUS.json', 'pipeline_controller_launch.json', 'pipeline_gate.log',
        'preflight/smoke_attempt2/controller_launch.json', 'preflight/smoke_attempt2/queue.log',
        'preflight/smoke_attempt2/results.json', 'preflight/smoke_attempt2/results.csv',
        'preflight/smoke_attempt2/EXPERIMENT_TRACKER.md',
    ]
    for rel in snapshot:
        src = CAM / rel
        if src.exists():
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for pid in frozen:
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 5
    while any(identity(pid) for pid in frozen) and time.monotonic() < deadline:
        time.sleep(.1)
    assert not any(identity(pid) for pid in frozen), 'Controllers did not terminate'
    # Only empty pending runtime is moved; old logs and receipts remain archived.
    (SMOKE / 'runtime').rename(backup / 'preflight/smoke_attempt2/runtime')
    for rel in expected:
        if rel not in ('runtime_manifest.json', 'gate_manifest.json'):
            shutil.copy2(STAGE / 'files' / rel, CAM / rel)
    runtime = read(CAM / 'runtime_manifest.json')
    runtime['files'] = {rel: sha(CAM / rel) for rel in runtime['files']}
    runtime.update(suite_sha256=sha(CAM / 'suite.json'), smoke_suite_sha256=sha(SMOKE / 'suite.json'),
                   assigned_gpus=GPUS, max_parallel=7)
    write(CAM / 'runtime_manifest.json', runtime)
    gate = read(CAM / 'gate_manifest.json')
    gate['files'] = {rel: sha(CAM / rel) for rel in gate['files']}
    gate['assigned_gpus'] = GPUS
    write(CAM / 'gate_manifest.json', gate)
    checks = []
    for script in (SMOKE / 'launch_queue.py', CAM / 'launch_queue.py', CAM / 'pipeline_gate.py'):
        result = subprocess.run([PYTHON, str(script), '--validate-only'], cwd=script.parent,
                                env=ENV, capture_output=True, text=True, timeout=60)
        checks.append({'script': str(script), 'returncode': result.returncode,
                       'stdout': result.stdout, 'stderr': result.stderr})
        assert result.returncode == 0, result.stderr
    write(backup / 'new_configuration_validation.json', checks)
    for rel in ('preflight/smoke_attempt2/queue.log', 'pipeline_gate.log'):
        # Exact old content was copied above; preserve original file by renaming too.
        src = CAM / rel
        src.rename(backup / (rel.replace('/', '__') + '.original'))
    with (SMOKE / 'queue.log').open('xb') as log:
        smoke = subprocess.Popen([PYTHON, '-u', str(SMOKE / 'launch_queue.py'), '--execute', '--poll-seconds', '10'],
                                 cwd=SMOKE, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(.5)
    smoke_id = identity(smoke.pid)
    assert smoke_id and smoke.poll() is None
    smoke_receipt = dict(smoke_id, attempt=2, allowed_gpus=GPUS, max_parallel=7,
                         suite_sha256=sha(SMOKE / 'suite.json'), scheduling_revision='all_local_gpus',
                         launched_at=datetime.now(timezone.utc).isoformat())
    write(SMOKE / 'controller_launch.json', smoke_receipt)
    with (CAM / 'pipeline_gate.log').open('xb') as log:
        gate_process = subprocess.Popen([PYTHON, '-u', str(CAM / 'pipeline_gate.py')], cwd=CAM, env=ENV,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(.5)
    gate_id = identity(gate_process.pid)
    assert gate_id and gate_process.poll() is None
    gate_receipt = dict(gate_id, allowed_gpus=GPUS, max_parallel=7,
                        gate_manifest_sha256=sha(CAM / 'gate_manifest.json'),
                        launched_at=datetime.now(timezone.utc).isoformat())
    write(CAM / 'pipeline_controller_launch.json', gate_receipt)
    receipt = {
        'changed_at': datetime.now(timezone.utc).isoformat(), 'user_authorization': '等待所有GPU',
        'allowed_gpus': GPUS, 'max_parallel': 7, 'backup_directory': str(backup),
        'changed_files': {rel: {'before': h, 'after': sha(CAM / rel)} for rel, h in expected.items()},
        'source_manifest_sha256': sha(CAM / 'source_manifest.json'), 'scientific_job_configs_changed': False,
        'old_pending_smoke_jobs': 20, 'live_training_jobs_interrupted': 0,
        'validation_passed': True, 'smoke_controller': smoke_receipt, 'gate_controller': gate_receipt,
        'readiness_rule': 'memory_used_mb < 500 and utilization_percent < 5; recheck GPU UUID immediately before launch',
    }
    write(CAM / 'GPU_POOL_UPDATE.json', receipt)
    shutil.copy2(STAGE / 'changes.diff', backup / 'changes.diff')
    shutil.copy2(Path(__file__), backup / 'apply_gpu_pool.py')
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
