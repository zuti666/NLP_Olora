"""Build/check the isolated radius-50 queue or execute its frozen plan."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
GPUS = [1, 4, 6]
UUIDS = {'1': 'GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a',
         '4': 'GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5',
         '6': 'GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a'}
sys.path.insert(0, str(BASE))
from campaign_runner import execute_plan, fingerprint, validate_plan


def make_plan():
    campaign = HERE / 'campaign'
    for name in ('diagnostic_radius50.py', 'publish_radius50.py', 'worker_cpu_check.json'):
        if not (HERE / name).is_file():
            raise RuntimeError(f'Missing validated queue prerequisite: {name}')
    if json.loads((HERE / 'worker_cpu_check.json').read_text())['status'] != 'passed':
        raise RuntimeError('Worker CPU validation must pass before building the queue')
    jobs = []
    for job in ('sparse_train', 'dense'):
        for seed in range(3):
            for method in ('sgd', 'gam'):
                jid = f'{job}_seed{seed}_{method}'
                jobs.append(dict(
                    id=jid, gpu=GPUS[len(jobs) % len(GPUS)], gpu_pool=GPUS,
                    command=[PYTHON, '-u', str(HERE / 'diagnostic_radius50.py'), '--config',
                             str(HERE / f'configs/{job}_seed{seed}.json'),
                             '--method', method, '--job', job, '--device', 'cuda:0'],
                    cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                    receipt=str(HERE / f'results/seed{seed}/{method}/{job}/extension_receipt.json'),
                    receipt_require={'status': 'complete', 'seed': seed, 'method': method, 'job': job},
                    preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                       '--output', str(campaign / f'{jid}_env.json')]))
    old = json.loads((BASE / 'stage_path_completion_20260921/campaign_idle_gpu146/plan.json').read_text())
    # Reuse the audited source closure and add all new executables/configuration.
    files = set(old['code_files'])
    files.update(str(p) for p in HERE.glob('*.py'))
    files.update(str(p) for p in (HERE / 'configs').glob('*.json'))
    files.update(str(p) for p in (BASE / 'figures_formal_20260921/B1_merged_latest_20260921').iterdir()
                 if p.is_file())
    files.add(str(BASE / 'stage_path_completion_20260921/report/radius_complete.json'))
    files.add(str(HERE / 'worker_cpu_check.json'))
    return dict(schema='rq2_radius50_v1', purpose='Same-protocol random-neighborhood radius extension',
                gpus=GPUS, max_parallel=3, run_dir=str(campaign),
                gpu_lock_dir=str(Path('/data/140-1/users/liying/O-LoRA/.cache/gpu_locks')),
                start_after={'gpu_uuids': UUIDS}, code_files=sorted(files),
                stages=[dict(id='measure', depends_on=[], jobs=jobs),
                        dict(id='publish', depends_on=['measure'], jobs=[dict(
                            id='publish_radius50', gpu=None,
                            command=[PYTHON, '-u', str(HERE / 'publish_radius50.py'), '--execute'],
                            cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                            receipt=str(HERE / 'report/radius50_complete.json'),
                            receipt_require={'status': 'complete', 'expected_jobs': 12, 'validated_jobs': 12,
                                             'automatic_numeric_and_layout_checks': 'passed'})])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Create new plan once; never overwrite a plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = HERE / 'campaign/plan.json'
    if args.build:
        if path.exists():
            raise RuntimeError('Plan already exists; inspect it rather than replacing it')
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
        print(json.dumps(dict(status='passed', dry_run=True, gpu_jobs=12, total_jobs=len(jobs),
            gpus=plan['gpus'], max_parallel=3, stages=[s['id'] for s in plan['stages']],
            radius_grid=[0,25,30,35,40,45,50], dense_states_per_job=601,
            fisher_states_per_job=97, fingerprints=len(fingerprint(plan['code_files'] + [str(path)]))), indent=2))


if __name__ == '__main__':
    main()
