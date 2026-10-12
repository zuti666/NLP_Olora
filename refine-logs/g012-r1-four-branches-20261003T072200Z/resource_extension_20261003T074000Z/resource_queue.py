"""Apply only the renewed idle-GPU resource policy to the frozen r1 study."""
import argparse
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout

E = Path(__file__).resolve().parent
H = E.parent
PINS = '7ca822c744d31818a0587bde4d9cccd15b83cefd28bb20d7d6c6b21b3846cf03'
GPUS = [1, 4, 5, 6, 3, 2]
MAX_PARALLEL = 3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'controller'])
    parser.add_argument('--gpu', type=int, choices=GPUS)
    args = parser.parse_args()
    policy = json.loads((E / 'RESOURCE_MANIFEST.json').read_text())
    policy_hash = digest(E / 'RESOURCE_MANIFEST.json')
    if (policy['allowed_gpus'] != GPUS or policy['max_parallel'] != MAX_PARALLEL
            or policy['frozen_run_pins_sha256'] != PINS
            or policy['wrapper_sha256'] != digest(__file__)
            or policy['document_sha256'] != digest(E / 'RUN.md')
            or policy['scientific_changes'] != []):
        raise RuntimeError('Resource policy differs from the authorized addendum')
    sys.path.insert(0, str(H))
    import common
    import experiment_contract
    original_load = common.load_suite
    original_verify = experiment_contract.verify_contract
    original = original_load()
    if digest(H / 'RUN_PINS.json') != PINS or digest(H / 'suite.json') != policy['frozen_suite_sha256']:
        raise RuntimeError('Frozen experiment identity changed')
    expected = copy.deepcopy(original)
    expected['allowed_gpus'] = GPUS
    expected['max_parallel'] = MAX_PARALLEL

    def extended_load(path=H / 'suite.json'):
        if Path(path).resolve() != H / 'suite.json' or original_load(path) != original:
            raise RuntimeError('Scientific manifest changed')
        return copy.deepcopy(expected)

    def verify_resources(suite):
        if suite != expected:
            raise RuntimeError('Only GPU list and parallel count may change')
        if digest(__file__) != policy['wrapper_sha256'] or digest(E / 'RESOURCE_MANIFEST.json') != policy_hash:
            raise RuntimeError('Resource addendum changed while executing')
        return original_verify(original)

    common.load_suite = extended_load
    experiment_contract.verify_contract = verify_resources
    import run_queue
    original_query = common.query_gpus

    def bound_query():
        gpus = original_query()
        if any(str(i) not in policy['gpu_uuids'] or gpus[i]['uuid'] != policy['gpu_uuids'][str(i)]
               for i in GPUS if i in gpus):
            raise RuntimeError('Authorized physical GPU identity changed')
        return gpus

    run_queue.query_gpus = bound_query
    if args.mode == 'controller':
        if args.gpu is not None:
            raise RuntimeError('Controller uses the declared idle-GPU pool')
        witness = json.loads((E / 'DOCUMENTED_WITNESS.json').read_text())
        if (witness['status'] != 'passed' or witness['resource_manifest_sha256'] != policy_hash
                or witness['frozen_run_pins_sha256'] != PINS):
            raise RuntimeError('The resource documented witness did not pass')
        sys.argv = [str(H / 'run_queue.py')]
        return run_queue.main()

    if args.gpu is None:
        raise RuntimeError('Check requires one idle physical GPU')
    output = io.StringIO()
    sys.argv = [str(H / 'run_queue.py'), '--dry-run']
    with redirect_stdout(output):
        code = run_queue.main()
    if code:
        raise RuntimeError('Frozen science/resource dry-run failed')
    dry = json.loads(output.getvalue())
    if (dry['jobs'], dry['formal'], dry['smokes'], dry['allowed_gpus'], dry['max_parallel']) != (28, 12, 16, GPUS, MAX_PARALLEL):
        raise RuntimeError('Documented resource design differs')
    print(json.dumps(dry), flush=True)
    suite = extended_load()
    gpu = bound_query()[args.gpu]
    if gpu['memory_used_mb'] >= 500 or gpu['utilization_percent'] >= 5:
        raise RuntimeError('Selected witness GPU is busy; no preemption')
    handles = []
    try:
        for name in (f'gpu{args.gpu}.lock', gpu['uuid'] + '.lock'):
            handle = open(Path(suite['gpu_lock_dir']) / name, 'a+')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            handles.append(handle)
        current = bound_query()[args.gpu]
        if current['uuid'] != gpu['uuid'] or current['memory_used_mb'] >= 500 or current['utilization_percent'] >= 5:
            raise RuntimeError('Witness admission changed')
        env = os.environ.copy()
        env.update(suite['environment'])
        for key in ('TMPDIR', 'TMP', 'TEMP'):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        env['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        spec = json.loads((H / 'environment_spec.json').read_text())
        test = spec['smoke']['gpu_tests'][0]
        child = subprocess.run(shlex.split(test['cmd']), env=env, cwd=H,
                               capture_output=True, text=True, check=True)
        print(child.stdout, end='', flush=True)
        if not re.search(test['expect'], child.stdout, re.MULTILINE):
            raise RuntimeError('Canonical CUDA sentinel missing')
        common.atomic_json(E / 'DOCUMENTED_WITNESS.json', {
            'status': 'passed', 'at': common.now(), 'dry_run': dry,
            'physical_gpu': args.gpu, 'gpu_uuid': gpu['uuid'],
            'canonical_witness_stdout': child.stdout, 'optimizer_updates': 0,
            'resource_manifest_sha256': policy_hash, 'frozen_run_pins_sha256': PINS,
            'resource_wrapper_sha256': digest(__file__)})
    finally:
        for handle in handles:
            handle.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
