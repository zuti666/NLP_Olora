"""Queue fixed Task0 positive-gradient diagnostics after the radius50 campaign."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
sys.path.insert(0, str(BASE))
from campaign_runner import execute_plan, fingerprint, validate_plan


def build_plan():
    previous_path = BASE / 'radius50_extension_20260921/campaign/plan.json'
    previous = json.loads(previous_path.read_text())
    for name in ('diagnostic_fixed_gradient.py', 'publish_fixed_gradient.py', 'worker_cpu_check.json'):
        if not (HERE / name).is_file():
            raise RuntimeError(f'Missing queue prerequisite: {name}')
    if json.loads((HERE / 'worker_cpu_check.json').read_text())['status'] != 'passed':
        raise RuntimeError('Positive-gradient worker CPU validation must pass before queue creation')
    gpus = previous['gpus']
    assert gpus == [1, 4, 6]
    campaign = HERE / 'campaign'
    common = dict(cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'})
    jobs = []
    for seed in range(3):
        for method in ('sgd', 'gam'):
            jid = f'gradient_seed{seed}_{method}'
            jobs.append(dict(**common, id=jid, gpu=gpus[len(jobs) % len(gpus)], gpu_pool=gpus,
                command=[PYTHON, '-u', str(HERE / 'diagnostic_fixed_gradient.py'), '--config',
                         str(HERE / f'configs/seed{seed}.json'), '--method', method,
                         '--job', 'fixed_gradient', '--device', 'cuda:0'],
                receipt=str(HERE / f'results/seed{seed}/{method}/fixed_gradient/extension_receipt.json'),
                receipt_require={'status': 'complete', 'seed': seed, 'method': method,
                                 'job': 'fixed_gradient', 'gradient_task': 0, 'direction_sign': 1},
                preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                   '--output', str(campaign / f'{jid}_env.json')]))
    files = set(previous['code_files'])
    files.add(str(previous_path))
    files.update(str(p) for p in HERE.glob('*.py'))
    files.update(str(p) for p in (HERE / 'configs').glob('*.json'))
    files.add(str(HERE / 'worker_cpu_check.json'))
    return dict(schema='rq2_fixed_task0_positive_gradient_queue_v1',
        purpose='One fixed Task0 positive CE gradient per anchor, radius0–50, no training',
        gpus=gpus, max_parallel=3, run_dir=str(campaign), gpu_lock_dir=previous['gpu_lock_dir'],
        start_after=previous['start_after'], code_files=sorted(files),
        stages=[dict(id='wait_radius50', depends_on=[], jobs=[dict(**common,
                    id='wait_random_radius50', gpu=None,
                    command=[PYTHON, '-u', str(HERE / 'wait_radius50.py')],
                    receipt=str(campaign / 'predecessor_complete.json'),
                    receipt_require={'status': 'complete', 'predecessor_gpu_jobs': 12,
                                     'all_predecessor_jobs_valid': True, 'predecessor_outputs_verified': True})]),
                dict(id='fixed_gradient', depends_on=['wait_radius50'], jobs=jobs),
                dict(id='publish', depends_on=['fixed_gradient'], jobs=[dict(**common,
                    id='publish_fixed_gradient', gpu=None,
                    command=[PYTHON, '-u', str(HERE / 'publish_fixed_gradient.py'), '--execute'],
                    receipt=str(HERE / 'report/fixed_gradient_complete.json'),
                    receipt_require={'status': 'complete', 'expected_jobs': 6, 'validated_jobs': 6,
                                     'automatic_numeric_and_layout_checks': 'passed'})])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = HERE / 'campaign/plan.json'
    if args.build:
        if path.exists():
            raise RuntimeError('Plan exists; do not overwrite it')
        plan = build_plan()
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
        print(json.dumps(dict(status='passed', dry_run=True, gpu_jobs=6, total_jobs=len(jobs),
            gpus=plan['gpus'], max_parallel=3, gradient_task=0, direction_sign=1,
            gradient_samples=1259, fisher_samples=256, fixed_directions_per_anchor=1,
            states_per_job=93, radius_max=50,
            predecessor='radius50_extension_20260921 must complete before GPU jobs',
            fingerprints=len(fingerprint(plan['code_files'] + [str(path)]))), indent=2))


if __name__ == '__main__':
    main()
