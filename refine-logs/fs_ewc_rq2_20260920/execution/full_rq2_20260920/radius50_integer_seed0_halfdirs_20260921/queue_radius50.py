"""Isolated seed0 queue: 50 loss directions, 16 Fisher directions, integer radii."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
GPUS = [0, 2, 3, 5]
UUIDS = {'0': 'GPU-fd7e2955-3733-e30a-dff1-5da992423839',
         '2': 'GPU-fe4ca05a-b7fd-efa7-8192-201e24573825',
         '3': 'GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26',
         '5': 'GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e'}
MISSING = [r for r in range(25, 51) if r not in [25, 30, 35, 40, 45, 50]]
sys.path.insert(0, str(BASE))
from campaign_runner import execute_plan, fingerprint, validate_plan


def make_plan():
    campaign = HERE / 'campaign'
    for name in ('diagnostic_radius50.py', 'publish_radius50.py', 'worker_cpu_check.json'):
        if not (HERE / name).is_file():
            raise RuntimeError(f'Missing validated queue prerequisite: {name}')
    if json.loads((HERE / 'worker_cpu_check.json').read_text())['status'] != 'passed':
        raise RuntimeError('Worker CPU validation must pass before building the queue')
    prior = BASE / 'radius50_extension_20260921'
    prior_receipt = json.loads((prior / 'report/radius50_complete.json').read_text())
    if prior_receipt['status'] != 'complete' or prior_receipt['validated_jobs'] != 12:
        raise RuntimeError('The prior coarse-radius cohort must be complete')
    jobs = []
    for job in ('dense', 'sparse_train'):
        for method in ('sgd', 'gam'):
            jid = f'{job}_seed0_{method}'
            jobs.append(dict(
                id=jid, gpu=GPUS[len(jobs)], gpu_pool=GPUS,
                command=[PYTHON, '-u', str(HERE / 'diagnostic_radius50.py'), '--config',
                         str(HERE / f'configs/{job}_seed0.json'),
                         '--method', method, '--job', job, '--device', 'cuda:0'],
                cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                receipt=str(HERE / f'results/seed0/{method}/{job}/extension_receipt.json'),
                receipt_require={'status': 'complete', 'seed': 0, 'method': method, 'job': job},
                preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                   '--output', str(campaign / f'{jid}_env.json')]))
    old = json.loads((prior / 'campaign/plan.json').read_text())
    files = set(old['code_files'])
    files.update(str(p) for p in HERE.glob('*.py'))
    files.update(str(p) for p in (HERE / 'configs').glob('*.json'))
    files.update(str(prior / name) for name in ('report/radius50_complete.json', 'publish_radius50.py'))
    files.add(str(HERE / 'worker_cpu_check.json'))
    return dict(schema='rq2_seed0_dense50_fisher16_integer_v1',
                purpose='Fill missing integer radii 25–50 with seed0 and shared public direction prefixes',
                training_seeds=[0], dense_directions=50, fisher_directions=16,
                fisher_samples=256, missing_radii=MISSING,
                gpus=GPUS, max_parallel=4, run_dir=str(campaign),
                gpu_lock_dir='/data/140-1/users/liying/O-LoRA/.cache/gpu_locks',
                start_after={'gpu_uuids': UUIDS}, code_files=sorted(files),
                stages=[dict(id='measure', depends_on=[], jobs=jobs),
                        dict(id='publish', depends_on=['measure'], jobs=[dict(
                            id='publish_radius50', gpu=None,
                            command=[PYTHON, '-u', str(HERE / 'publish_radius50.py'), '--execute'],
                            cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                            receipt=str(HERE / 'report/radius50_complete.json'),
                            receipt_require={'status': 'complete', 'expected_jobs': 4, 'validated_jobs': 4,
                                             'automatic_numeric_and_layout_checks': 'passed'})])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Create the plan once; never overwrite it')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = HERE / 'campaign/plan.json'
    if args.build:
        if path.exists():
            raise RuntimeError('Plan already exists; inspect rather than replacing it')
        plan = make_plan()
        validate_plan(plan)
        fingerprint(plan['code_files'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2) + '\n')
    else:
        plan = json.loads(path.read_text())
    jobs = validate_plan(plan)
    if args.execute:
        execute_plan(plan, path, resume=args.resume)
    else:
        print(json.dumps(dict(status='passed', dry_run=True, gpu_jobs=4, total_jobs=len(jobs),
            training_seeds=[0], dense_directions=50, fisher_directions=16, fisher_samples=256,
            gpus=plan['gpus'], max_parallel=4, new_radii=MISSING,
            dense_states_per_job=1001, fisher_states_per_job=321,
            fingerprints=len(fingerprint(plan['code_files'] + [str(path)]))), indent=2))


if __name__ == '__main__':
    main()
