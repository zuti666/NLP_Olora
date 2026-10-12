"""Build the authorized seed0 two-task normalized-Fisher sweep, without launching it."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
PY = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
LAMBDAS = [0, 2000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]


def job(jid, script, argv, receipt, require, gpu=True):
    result = {'id': jid, 'gpu': 0 if gpu else None, 'gpu_pool': list(range(7)) if gpu else [],
              'command': [PY, '-u', str(HERE / script)] + [str(x) for x in argv],
              'cwd': str(HERE), 'env': {'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
              'receipt': str(receipt), 'receipt_require': require}
    if gpu:
        result['preflight_command'] = [PY, str(HERE / 'env_witness.py'), '--output', str(HERE / 'witness' / (jid + '.json'))]
    return result


def main():
    previous = json.loads((FORMAL / 'formal_plan.json').read_text())
    spec = {'seed': 0, 'dataset': 'imageneta', 'anchor_method': 'sgd', 'future_optimizer': 'sgd',
            'tasks_total': 2, 'new_task_epochs': 50, 'batch_size': 128, 'learning_rate': .02,
            'fisher_policy': 'Reuse original trace-normalized Task0 Fisher unchanged; no raw-Fisher restoration or recomputation',
            'ewc_normalize_fisher': True, 'ewc_normalize_accumulated_fisher': True,
            'lambdas': LAMBDAS, 'new_lambdas': LAMBDAS[2:], 'reused_lambdas': [0, 2000],
            'reused_runs': {'0': str(FORMAL / 'formal_training/B3_A_sgd_lam0_seed0'),
                            '2000': str(FORMAL / 'formal_training/sequence_sgd_seed0')},
            'shared_anchor': str(FORMAL / 'formal_training/anchor_sgd_seed0'),
            'smoke_lambda': 2000, 'scope': 'Single-seed exploratory protection-strength sweep; preserves complete seven-row three-seed results',
            'selection': 'Report all observed points, strongest retention and best average among finite completed points; test-set exploration is not confirmatory evidence',
            'failure_policy': 'Explicit nonfinite loss/gradient/parameter divergence is a reported outcome; OOM, missing files, data or code failures stop new work',
            'resource_policy': 'Wait for all GPU0-6 to be idle before the first GPU job; poll every900seconds; per-job shared locks and idle checks remain required'}
    (HERE / 'sweep_spec.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    smoke = job('smoke_sgd_lambda2000', 'train.py', ['--seed', 0, '--ewc-lambda', 2000,
        '--run-name', 'smoke_lambda2000_seed0', '--gpu', '{gpu}', '--smoke'],
        HERE / 'smoke/training/smoke_lambda2000_seed0/SMOKE_COMPLETE.json',
        {'status': 'smoke_complete', 'completed_tasks': 2})
    validate = job('validate_smoke', 'validate_smoke.py', [], HERE / 'smoke/VALIDATED.json', {'status': 'passed'}, gpu=False)
    stages = [{'id': smoke['id'], 'depends_on': [], 'jobs': [smoke]},
              {'id': validate['id'], 'depends_on': [smoke['id']], 'jobs': [validate]}]
    trials = []
    for lam in spec['new_lambdas']:
        name = f'lambda_{lam}_seed0'
        trial = job(name, 'run_trial.py', ['--ewc-lambda', lam, '--gpu', '{gpu}'],
                    HERE / 'training' / name / 'TRIAL_RESULT.json', {'status': 'observed', 'ewc_lambda': lam})
        stages.append({'id': name, 'depends_on': ['validate_smoke'], 'jobs': [trial]})
        trials.append(name)
    report = job('sweep_report', 'report.py', [], HERE / 'report/COMPLETE.json', {'status': 'complete', 'observed_lambda_points': len(LAMBDAS)}, gpu=False)
    stages.append({'id': report['id'], 'depends_on': trials, 'jobs': [report]})
    own = ['campaign_runner.py', 'train.py', 'run_trial.py', 'report.py', 'validate_smoke.py',
           'env_witness.py', 'build_plan.py', 'sweep_spec.json']
    plan = {'version': 1, 'purpose': spec['scope'], 'gpus': list(range(7)), 'max_parallel': 6,
            'run_dir': str(HERE / 'campaign'), 'gpu_lock_dir': previous['gpu_lock_dir'],
            'start_after': {'gpu_uuids': previous['start_after']['gpu_uuids']},
            'all_gpu_start_gate': {'required': True, 'gpus': list(range(7)), 'poll_seconds': 900},
            'code_files': list(dict.fromkeys(previous['code_files'] + [str(HERE / x) for x in own])),
            'stages': stages, 'gpu_poll_seconds': 900,
            'workload': {'reused_points': 2, 'new_formal_continuations': 9, 'smokes': 1,
                         'smoke_validation': 1, 'report': 1, 'total_jobs': len(stages)}}
    (HERE / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'execute': False, 'lambda_points': len(LAMBDAS), 'new_formal': 9,
                      'smoke': 1, 'total_jobs': len(stages), 'wait_for_all_gpus': True, 'poll_seconds': 900}))


if __name__ == '__main__':
    main()
