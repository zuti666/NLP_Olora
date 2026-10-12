#!/usr/bin/env python3
"""Build the fixed-lambda four-arm DAG; never launches training."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
GPUS = [0, 1, 2, 5, 4, 6, 3]
UUIDS = dict(enumerate([
    'GPU-fd7e2955-3733-e30a-dff1-5da992423839',
    'GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a',
    'GPU-fe4ca05a-b7fd-efa7-8192-201e24573825',
    'GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26',
    'GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5',
    'GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e',
    'GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a']))


def build():
    spec = json.loads((HERE / 'experiment_spec.json').read_text())
    stages = []
    parents = {p['id']: p for p in spec['parents']}

    def stage(name, dependencies, command, receipt, required, gpu=None):
        job = dict(id=name, gpu=gpu, command=[PYTHON, '-u'] + list(map(str, command)),
                   cwd=str(HERE), env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp',
                                      'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                      'OPENBLAS_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1',
                                      'TRANSFORMERS_OFFLINE': '1'},
                   receipt=str(receipt), receipt_require=required)
        if gpu is not None:
            job['gpu_pool'] = [gpu] + [g for g in GPUS if g != gpu]
            job['preflight_command'] = [PYTHON, str(HERE / 'env_witness.py'),
                                        '--output', str(HERE / 'witness' / (name + '.json'))]
        stages.append(dict(id=name, depends_on=dependencies, jobs=[job]))

    preparations = []
    for i, parent in enumerate(spec['parents']):
        name = 'prepare_' + parent['id']
        preparations.append(name)
        reuse = parent['action'] == 'reuse_verified_existing_raw'
        cmd = [HERE / 'prepare_fisher.py', '--anchor-method', parent['anchor_method'],
               '--seed', parent['seed']]
        cmd += ['--reuse-existing'] if reuse else ['--gpu', '{gpu}']
        stage(name, [], cmd, parent['raw_fisher_receipt'],
              dict(status='complete', seed=parent['seed'], anchor_method=parent['anchor_method']),
              None if reuse else GPUS[(i - 1) % len(GPUS)])

    def train_command(trial, smoke=False):
        parent = parents[trial['parent_id']]
        root = Path(parent['directory'])
        name = ('smoke_' if smoke else '') + trial['trial_id']
        cmd = [HERE / 'train.py', '--arm', trial['arm'], '--anchor-method', trial['anchor_method'],
               '--future-optimizer', trial['future_optimizer'], '--seed', trial['seed'],
               '--gpu', '{gpu}', '--ewc-lambda', '20000', '--anchor-dir', root / 'checkpoints',
               '--anchor-metrics', root / 'trainer_outputs', '--prepared-fisher', parent['raw_fisher'],
               '--run-name', name, '--output-root', HERE / ('smoke/training' if smoke else 'training'),
               '--resume-last']
        if smoke:
            cmd.append('--smoke')
        return cmd

    smokes = []
    for trial in spec['trials']:
        if trial['seed'] != 0:
            continue
        name = 'smoke_' + trial['trial_id']
        smokes.append(name)
        stage(name, preparations, train_command(trial, smoke=True),
              HERE / 'smoke/training' / name / 'SMOKE_COMPLETE.json',
              dict(status='smoke_complete', completed_tasks=2, arm=trial['arm'], smoke_only=True),
              trial['preferred_gpu'])
    stage('validate_smokes', smokes, [HERE / 'validate_smokes.py'], HERE / 'smoke/VALIDATED.json',
          dict(status='passed', validated_smokes=4, formal_training_released=True))

    for seed in spec['seeds']:
        for trial in [t for t in spec['trials'] if t['seed'] == seed]:
            dependencies = ['validate_smokes']
            if seed:
                dependencies.append(trial['arm'] + '_seed' + str(seed - 1))
            stage(trial['trial_id'], dependencies, train_command(trial),
                  Path(trial['run_dir']) / 'COMPLETE.json',
                  dict(status='complete', completed_tasks=2, arm=trial['arm'], smoke_only=False),
                  trial['preferred_gpu'])
    stage('report', [t['trial_id'] for t in spec['trials']],
          [HERE / 'report.py', '--run-root', HERE, '--output-dir', HERE / 'report'],
          HERE / 'report/COMPLETE.json', dict(status='complete', expected_trials=12, completed_trials=12))

    old = json.loads((BASE / 'ewc_lambda_refinement_round2_20260922/plan.json').read_text())
    # Preserve dependencies actually imported by these adapters and frozen source.
    files = [p for p in old['code_files'] if '/ewc_lambda_refinement_round' not in p]
    files += [str(HERE / f) for f in ['build_plan.py', 'campaign_runner.py', 'env_witness.py',
              'train.py', 'prepare_fisher.py', 'validate_smokes.py', 'report.py',
              'SELECTION.json', 'experiment_spec.json']]
    files += [str(BASE / 'seven_row_extension_20260921/train.py')]
    for parent in spec['parents']:
        root = Path(parent['directory'])
        files += [str(root / f) for f in ['run_manifest.json', 'COMPLETE.json']]
    plan = dict(version=1, purpose='User-fixed lambda=20000; four raw-Fisher EWC arms x three seeds',
                gpus=GPUS, max_parallel=6, run_dir=str(HERE / 'campaign'),
                gpu_lock_dir=str(BASE.parents[3] / '.cache/gpu_locks'),
                start_after={'gpu_uuids': {str(g): UUIDS[g] for g in GPUS}},
                all_gpu_start_gate={'required': False}, gpu_poll_seconds=900,
                workload=dict(new_formal_continuations=12, new_fisher_passes=5,
                              verified_raw_fisher_reuses=1, new_smokes=4, smoke_acceptance=1,
                              cpu_report=1, total_jobs=24),
                stages=stages, code_files=sorted(set(files)))
    assert Path(plan['gpu_lock_dir']).parent.parent == Path('/data/140-1/users/liying/O-LoRA')
    assert len(stages) == 24
    (HERE / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'plan': str(HERE / 'plan.json'), 'stages': len(stages),
                      'formal_trials': 12, 'gpus': GPUS, 'all_gpu_start_gate': False}))


if __name__ == '__main__':
    build()
