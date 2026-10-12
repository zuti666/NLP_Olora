"""Resource-only extension; execute the unchanged pinned experiment entrypoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

CAMPAIGN = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z')
EXTENSION = Path(__file__).resolve().parent
SUITE_HASH = 'bf5f4b387bb67cc5569ac67a01cae0e1baff1538ceb5331da72b57ad6915e7e6'
PINS_HASH = 'bb31af2e01cea8c1c81e5a657e72a7f8b2bd3a652c41139cddabc53da1e5688b'
PENDING_JOB = 'formal_Gmatch_r1_k10_s2'
ADOPTED_JOB = 'formal_Mean2_r1_k10_s2'
NEW_GPUS = (2, 3)
GPU_UUIDS = {2: 'GPU-fe4ca05a-b7fd-efa7-8192-201e24573825',
             3: 'GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validate', 'controller', 'job'])
    parser.add_argument('native_entry', nargs='?')
    args, remainder = parser.parse_known_args()
    manifest = json.loads((EXTENSION / 'RESOURCE_MANIFEST.json').read_text())
    manifest_hash = digest(EXTENSION / 'RESOURCE_MANIFEST.json')
    if digest(__file__) != manifest['wrapper_sha256']:
        raise RuntimeError('Resource wrapper changed')
    if digest(CAMPAIGN / 'suite.json') != SUITE_HASH or digest(CAMPAIGN / 'RUN_PINS.json') != PINS_HASH:
        raise RuntimeError('Frozen campaign identity changed')
    if (manifest['campaign'] != str(CAMPAIGN) or manifest['new_launch_gpus'] != list(NEW_GPUS)
            or manifest['pending_job'] != PENDING_JOB or manifest['max_parallel'] != 2
            or manifest['frozen_suite_sha256'] != SUITE_HASH
            or manifest['frozen_run_pins_sha256'] != PINS_HASH
            or manifest['gpu_uuids'] != {str(k): v for k, v in GPU_UUIDS.items()}
            or manifest['adopt_only_gpu'] != 1 or manifest['preserved_job'] != ADOPTED_JOB
            or manifest['scientific_changes'] != []):
        raise RuntimeError('Resource extension policy changed')
    sys.path.insert(0, str(CAMPAIGN))
    import common
    original_load = common.load_suite
    original_validate = common.validate
    original_suite = original_load()
    common.assert_host(original_suite)
    if common.verify_run_pins(original_suite) != PINS_HASH:
        raise RuntimeError('Pinned numerical implementation changed')
    expected = copy.deepcopy(original_suite)
    expected['allowed_gpus'] = [2, 3, 1]  # GPU1 is present only to adopt the live Mean2.
    expected['max_parallel'] = 2

    def extended_load(path=CAMPAIGN / 'suite.json'):
        original = original_load(path)
        if original != original_suite:
            raise RuntimeError('Scientific manifest changed')
        return copy.deepcopy(expected)

    def resource_validate(suite):
        if suite != expected:
            raise RuntimeError('Only the declared resource fields may differ')
        return original_validate(original_suite)

    common.load_suite = extended_load
    common.validate = resource_validate
    if args.mode == 'validate':
        proof = resource_validate(extended_load())
        from validate_smokes import validate_job_audit
        reference = next(j for j in original_suite['jobs'] if j['id'] == 'formal_S_r1_k10_s2')
        proof = validate_job_audit(original_suite, reference)
        print(json.dumps({'passed': True, 'reference_steps': proof['steps'],
                          'new_launch_gpus': NEW_GPUS, 'wrapper_sha256': digest(__file__)}))
        return 0

    if args.mode == 'job':
        if args.native_entry != str(CAMPAIGN / 'run_job.py'):
            raise RuntimeError('Unexpected native job entrypoint')
        options = argparse.ArgumentParser()
        options.add_argument('--job', required=True)
        options.add_argument('--gpu', type=int, required=True)
        options.add_argument('--uuid', required=True)
        options.add_argument('--run-pins-sha256', required=True)
        native = options.parse_args(remainder)
        if native.job != PENDING_JOB or native.gpu not in NEW_GPUS or native.run_pins_sha256 != PINS_HASH:
            raise RuntimeError('Only the pending Gmatch may newly launch on GPU2/3')
        if native.uuid != GPU_UUIDS[native.gpu]:
            raise RuntimeError('Physical GPU UUID differs from resource authorization')
        run = CAMPAIGN / 'runtime/jobs' / native.job
        receipt = {'time': common.now(), 'campaign': str(CAMPAIGN), 'job': native.job,
                   'gpu': native.gpu, 'gpu_uuid': native.uuid,
                   'resource_manifest_sha256': digest(EXTENSION / 'RESOURCE_MANIFEST.json'),
                   'resource_wrapper_sha256': digest(__file__), 'frozen_suite_sha256': SUITE_HASH,
                   'frozen_run_pins_sha256': PINS_HASH, 'scientific_changes': [],
                   **common.proc_identity(os.getpid())}
        common.exclusive_json(run / 'resource_extension.json', receipt)
        import run_job
        sys.argv = [str(CAMPAIGN / 'run_job.py'), *remainder]
        code = run_job.main()
        final = run / ('completed.json' if code == 0 else 'failed.json')
        common.exclusive_json(run / 'resource_extension_finished.json',
                              {**receipt, 'ended_at': common.now(), 'exit_code': code,
                               'native_receipt_sha256': digest(final)})
        return code

    if remainder or args.native_entry:
        raise RuntimeError('Unexpected controller arguments')
    import launch_queue
    original_query = launch_queue.query_gpus

    def pending_gpu_query():
        # Existing GPU1 process remains adopted; only GPU2/3 are eligible for new launches.
        if digest(__file__) != manifest['wrapper_sha256'] or digest(EXTENSION / 'RESOURCE_MANIFEST.json') != manifest_hash:
            raise RuntimeError('Resource policy changed while running')
        values = {k: v for k, v in original_query().items() if k in NEW_GPUS}
        if any(v['uuid'] != GPU_UUIDS[k] for k, v in values.items()):
            raise RuntimeError('Authorized physical GPU identity changed')
        return values

    launch_queue.query_gpus = pending_gpu_query
    original_popen = launch_queue.subprocess.Popen

    def resource_popen(command, *positional, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 3
                and command[2] == str(CAMPAIGN / 'run_job.py')):
            command = list(command)
            job = command[command.index('--job') + 1]
            gpu = int(command[command.index('--gpu') + 1])
            if job != PENDING_JOB or gpu not in NEW_GPUS:
                raise RuntimeError('Controller attempted an undeclared new job/GPU')
            command = [command[0], '-u', str(EXTENSION / 'resource_wrapper.py'),
                       'job', str(CAMPAIGN / 'run_job.py'), *command[3:]]
        return original_popen(command, *positional, **kwargs)

    launch_queue.subprocess.Popen = resource_popen
    # Reuse the original GPU smoke gate on the original resource declaration;
    # do not mislabel the already-completed GPU1 smokes as GPU2/3 evidence.
    import validate_smokes
    original_smoke = validate_smokes.validate_smokes
    validate_smokes.validate_smokes = lambda suite: original_smoke(original_suite)
    sys.argv = [str(CAMPAIGN / 'launch_queue.py'), '--execute', '--job', PENDING_JOB,
                '--poll-seconds', '15']
    return launch_queue.main()


if __name__ == '__main__':
    raise SystemExit(main())
