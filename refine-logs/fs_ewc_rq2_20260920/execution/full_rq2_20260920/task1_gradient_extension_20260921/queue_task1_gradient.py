"""Queue six Task1 negative-gradient scans using the existing local GPU locks."""
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
    previous = BASE / 'fixed_gradient_extension_20260921'
    previous_plan_path = previous / 'campaign/plan.json'
    previous_plan = json.loads(previous_plan_path.read_text())
    previous_receipt = previous / 'report/fixed_gradient_complete.json'
    complete = json.loads(previous_receipt.read_text())
    assert complete['status'] == 'complete' and complete['validated_jobs'] == 6
    for name in ('diagnostic_task1_gradient.py', 'publish_task1_gradient.py', 'worker_cpu_check.json'):
        assert (HERE / name).is_file(), name
    assert json.loads((HERE / 'worker_cpu_check.json').read_text())['status'] == 'passed'
    gpus = previous_plan['gpus']
    assert gpus == [1, 4, 6]
    campaign = HERE / 'campaign'
    common = dict(cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'})
    jobs = []
    for seed in range(3):
        for method in ('sgd', 'gam'):
            jid = f'task1_gradient_seed{seed}_{method}'
            jobs.append(dict(**common, id=jid, gpu=gpus[len(jobs) % len(gpus)], gpu_pool=gpus,
                command=[PYTHON, '-u', str(HERE / 'diagnostic_task1_gradient.py'), '--config',
                         str(HERE / f'configs/seed{seed}.json'), '--method', method,
                         '--job', 'task1_gradient', '--gradient-task', '1', '--device', 'cuda:0'],
                receipt=str(HERE / f'results/seed{seed}/{method}/task1_gradient/extension_receipt.json'),
                receipt_require={'status': 'complete', 'seed': seed, 'method': method,
                                 'job': 'task1_gradient', 'gradient_task': 1,
                                 'evaluation_task': 0, 'direction_sign': -1},
                preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                   '--output', str(campaign / f'{jid}_env.json')]))
    files = set(previous_plan['code_files'])
    files.update((str(previous_plan_path), str(previous_receipt)))
    files.update(str(p) for p in HERE.glob('*.py'))
    files.update(str(p) for p in (HERE / 'configs').glob('*.json'))
    files.add(str(HERE / 'worker_cpu_check.json'))
    return dict(schema='rq2_fixed_task1_negative_gradient_queue_v1',
        purpose='Fixed initial Task1 CE descent direction, old Task0 measurements, no training',
        gpus=gpus, max_parallel=3, run_dir=str(campaign),
        gpu_lock_dir=previous_plan['gpu_lock_dir'], start_after=previous_plan['start_after'],
        code_files=sorted(files), stages=[dict(id='task1_gradient', depends_on=[], jobs=jobs),
            dict(id='publish', depends_on=['task1_gradient'], jobs=[dict(**common,
                id='publish_task1_gradient', gpu=None,
                command=[PYTHON, '-u', str(HERE / 'publish_task1_gradient.py'), '--execute'],
                receipt=str(HERE / 'report/task1_gradient_complete.json'),
                receipt_require={'status': 'complete', 'expected_jobs': 6, 'validated_jobs': 6,
                                 'gradient_task': 1, 'direction_sign': -1,
                                 'automatic_numeric_and_layout_checks': 'passed'})])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = HERE / 'campaign/plan.json'
    if args.build:
        assert not path.exists(), 'Plan already exists; inspect rather than overwrite'
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
            gpus=plan['gpus'], max_parallel=3, gradient_task=1, evaluation_task=0, direction_sign=-1,
            gradient_samples=997, fisher_samples=256, fixed_directions_per_anchor=1,
            states_per_job=93, radius_max=50, task1_head='reconstructed formal untrained new20 rows',
            fingerprints=len(fingerprint(plan['code_files'] + [str(path)]))), indent=2))


if __name__ == '__main__':
    main()
