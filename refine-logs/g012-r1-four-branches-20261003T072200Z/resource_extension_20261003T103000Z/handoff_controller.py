"""Transfer admission control only; keep independently leased training workers alive."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

E = Path(__file__).resolve().parent
H = E.parent
sys.path.insert(0, str(H))
import common


def main():
    suite = common.load_suite()
    common.assert_host(suite)
    policy = common.read_json(E / 'RESOURCE_MANIFEST.json')
    witness = common.read_json(E / 'DOCUMENTED_WITNESS.json')
    if (witness['status'] != 'passed'
            or witness['resource_manifest_sha256'] != common.sha256(E / 'RESOURCE_MANIFEST.json')
            or common.sha256(E / 'resource_queue.py') != policy['wrapper_sha256']
            or common.sha256(E / 'RUN.md') != policy['document_sha256']
            or common.sha256(H / 'RUN_PINS.json') != policy['frozen_run_pins_sha256']):
        raise RuntimeError('Documented resource witness or pins changed')
    for path, expected in common.read_json(H / 'RUN_PINS.json')['files'].items():
        if common.sha256(H / path) != expected:
            raise RuntimeError('Frozen scientific file changed: ' + path)
    old = {'pid': 3985116, 'proc_starttime': '146832628'}
    pause = H / 'runtime/PAUSE'
    if common.read_json(pause)['owner'] != E.name:
        raise RuntimeError('Admission pause belongs to a different operation')
    deadline = time.monotonic() + 20
    while True:
        state = common.read_json(H / 'runtime/queue_state.json')
        if state.get('paused') and not any(r['status'] == 'launching' for r in state['jobs'].values()):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Old controller has not acknowledged admission pause')
        time.sleep(.5)
    if (state['controller'] != old
            or common.read_json(H / 'runtime/controller.json')['process'] != old
            or not common.process_matches(old)):
        raise RuntimeError('Old controller identity changed; no signal sent')
    command = Path(f'/proc/{old["pid"]}/cmdline').read_bytes().split(b'\0')
    expected_old = str(H / 'resource_extension_20261003T074000Z/resource_queue.py').encode()
    if expected_old not in command or b'controller' not in command:
        raise RuntimeError('Old controller command differs; no signal sent')
    preserved = {j: r for j, r in state['jobs'].items() if r['status'] == 'running'}
    if len(preserved) != 3 or {r['gpu'] for r in preserved.values()} != {1, 2, 6}:
        raise RuntimeError('Expected active worker set changed; no signal sent')
    if not all(common.process_matches(r['process']) for r in preserved.values()):
        raise RuntimeError('A preserved worker is not live; no signal sent')
    common.atomic_json(E / 'BEFORE_HANDOFF.json', {
        'at': common.now(), 'controller': old, 'workers': preserved,
        'counts': state['counts'], 'resource_manifest_sha256': common.sha256(E / 'RESOURCE_MANIFEST.json')})
    # Deliberately never signal a process group or worker.
    os.kill(old['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while common.process_matches(old):
        if time.monotonic() > deadline:
            raise RuntimeError('Old controller did not exit; do not start a duplicate')
        time.sleep(.1)
    with open(H / 'runtime/controller.lock', 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
    if not all(common.process_matches(r['process']) for r in preserved.values()):
        raise RuntimeError('Preserved worker exited during handoff; inspect before proceeding')
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [suite['python'], str(E / 'resource_queue.py'), 'controller']
    with open(E / 'controller.log', 'xb') as log:
        child = subprocess.Popen(command, cwd=H, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    new_identity = common.proc_identity(child.pid)
    if not new_identity:
        raise RuntimeError('New controller did not remain alive')
    common.atomic_json(E / 'LAUNCH.json', {
        'at': common.now(), 'process': new_identity, 'command': command,
        'resource_manifest_sha256': common.sha256(E / 'RESOURCE_MANIFEST.json'),
        'handoff_script_sha256': common.sha256(__file__)})
    deadline = time.monotonic() + 20
    while True:
        after = common.read_json(H / 'runtime/queue_state.json')
        if after['controller'] == new_identity and after.get('paused'):
            break
        if child.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('New controller failed to adopt paused queue; inspect controller.log')
        time.sleep(.5)
    for j, r in preserved.items():
        if (after['jobs'][j]['process'] != r['process']
                or not common.process_matches(r['process'])):
            raise RuntimeError('Active worker identity was not preserved')
    if common.read_json(pause)['owner'] != E.name:
        raise RuntimeError('Admission pause changed ownership')
    pause.unlink()
    receipt = {
        'status': 'passed', 'at': common.now(), 'previous_controller': old,
        'new_controller': new_identity, 'preserved_worker_processes': {
            j: dict(process=r['process'], gpu=r['gpu'], gpu_uuid=r['gpu_uuid'])
            for j, r in preserved.items()},
        'training_workers_signaled': [], 'temporary_admission_pause_removed': True,
        'allowed_gpus': policy['allowed_gpus'], 'max_parallel': policy['max_parallel'],
        'resource_manifest_sha256': common.sha256(E / 'RESOURCE_MANIFEST.json'),
        'frozen_run_pins_sha256': policy['frozen_run_pins_sha256']}
    common.atomic_json(E / 'HANDOFF.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
