"""Start this authorized campaign once in the host PID/GPU namespace."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from common import atomic_json, assert_host, load_suite, now, proc_identity, process_matches, sha256

H = Path(__file__).resolve().parent


def main():
    suite = load_suite(H / 'suite.json')
    assert_host(suite)
    doc = json.loads((H / 'gates/DOC_CHECK.json').read_text())
    if doc.get('status') != 'passed' or doc.get('suite_sha256') != sha256(H / 'suite.json') or doc.get('run_pins_sha256') != sha256(H / 'RUN_PINS.json'):
        raise RuntimeError('Final independent documented invocation must pass against the frozen suite and pins')
    runtime = H / 'runtime'
    runtime.mkdir(exist_ok=True)
    launch_lock = open(runtime / 'launch.lock', 'a+')
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in (runtime / 'controller.json', H / 'LAUNCH.json'):
        if path.exists():
            old = json.loads(path.read_text())
            ident = old.get('process')
            if ident and process_matches(ident):
                print(json.dumps({'status': 'already_running', 'process': ident}))
                return 0
    subprocess.run([suite['python'], str(H / 'run_queue.py'), '--suite', str(H / 'suite.json'), '--dry-run'],
                   cwd=H, check=True)
    env = os.environ.copy()
    env.update(suite['environment'])
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = [suite['python'], str(H / 'run_queue.py'), '--suite', str(H / 'suite.json')]
    with (runtime / 'controller.log').open('ab', buffering=0) as log:
        controller = subprocess.Popen(command, cwd=H, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ident = proc_identity(controller.pid)
    record = {'status': 'controller_started_pending_verification', 'at': now(),
              'process': ident, 'command': command, 'allowed_gpus': suite['allowed_gpus'],
              'max_parallel': suite['max_parallel'], 'formal_training_claimed': False}
    atomic_json(H / 'LAUNCH.json', record)
    print(json.dumps(record))
    return 0


if __name__ == '__main__':
    sys.exit(main())
