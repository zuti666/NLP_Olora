"""Create the seven-row supplement DAG; this script never starts a process."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
N = E.parent
PY = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
POOL = list(range(7))


def job(jid, script, arguments, receipt, require, gpu=True):
    result = {'id': jid, 'gpu': 0 if gpu else None, 'gpu_pool': POOL if gpu else [],
              'command': [PY, '-u', str(E / script)] + list(map(str, arguments)),
              'cwd': str(E), 'env': {'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
              'receipt': str(receipt), 'receipt_require': require}
    if gpu:
        result['preflight_command'] = [PY, str(E / 'env_witness.py'), '--output', str(E / 'witness' / f'{jid}.json')]
    return result


def train_job(arm, parent, lam, seed, smoke=False):
    name = f'{arm}_seed{seed}'
    ancestor = N / 'formal_training' / f'anchor_{parent}_seed{seed}'
    args = ['--arm', arm, '--anchor-method', parent, '--future-optimizer', 'gam',
            '--seed', seed, '--ewc-lambda', lam, '--anchor-dir', ancestor / 'checkpoints',
            '--anchor-metrics', ancestor / 'trainer_outputs', '--run-name', name,
            '--output-root', E / 'training', '--workers', 4, '--gpu', '{gpu}', '--resume-last']
    if smoke:
        args.append('--smoke')
        receipt = E / 'smoke' / 'training' / name / 'SMOKE_COMPLETE.json'
        require = {'status': 'smoke_complete', 'completed_tasks': 2}
    else:
        receipt = E / 'training' / name / 'COMPLETE.json'
        require = {'status': 'complete', 'completed_tasks': 2}
    return job(('smoke_' if smoke else '') + name, 'train.py', args, receipt, require)


def main():
    old = json.loads((N / 'formal_plan.json').read_text())
    arms = [('shaping_only', 'gam', 0), ('task2_shaping', 'sgd', 2000)]
    stages = []
    for arm, parent, lam in arms:
        j = train_job(arm, parent, lam, 0, smoke=True)
        stages.append({'id': j['id'], 'depends_on': [], 'jobs': [j]})
    smoke_ids = [s['id'] for s in stages]
    gate = job('smoke_validation', 'validate_smokes.py', [], E / 'smoke' / 'VALIDATED.json',
               {'status': 'passed', 'validated_smokes': 2}, gpu=False)
    stages.append({'id': gate['id'], 'depends_on': smoke_ids, 'jobs': [gate]})
    training_ids = []
    for seed in range(3):
        for arm, parent, lam in arms:
            j = train_job(arm, parent, lam, seed)
            training_ids.append(j['id'])
            stages.append({'id': j['id'], 'depends_on': ['smoke_validation'], 'jobs': [j]})
    report = job('seven_row_report', 'report.py', ['--extension-root', E, '--output-dir', E / 'final_report'],
                 E / 'final_report' / 'COMPLETE.json', {'status': 'complete'}, gpu=False)
    stages.append({'id': report['id'], 'depends_on': training_ids, 'jobs': [report]})
    code_files = old['code_files'] + [str(E / p) for p in ['train.py', 'report.py', 'env_witness.py', 'validate_smokes.py', 'build_plan.py']]
    code_files.append(str(N / 'six_row_imageneta_20260921' / 'build_table.py'))
    plan = {'version': 1, 'purpose': 'Authorized seven-row ImageNet-A table supplement: two missing future-GAM arms, seeds0/1/2',
            'seeds': [0, 1, 2], 'gpus': POOL, 'max_parallel': 6,
            'run_dir': str(E / 'campaign'), 'gpu_lock_dir': old['gpu_lock_dir'],
            'start_after': {'gpu_uuids': old['start_after']['gpu_uuids']},
            'code_files': list(dict.fromkeys(code_files)), 'stages': stages,
            'workload': {'smoke_training_jobs': 2, 'smoke_validation_jobs': 1, 'formal_training_jobs': 6,
                         'formal_new_task_segments': 6, 'formal_epochs_per_segment': 50, 'report_jobs': 1, 'total_jobs': 10},
            'resource_policy': 'Claim only GPU0-6 with memory<500MiB, utilization<5%, no compute process and shared UUID lock; retry availability every15s; no preemption.',
            'failure_policy': 'Any nonzero exit or invalid receipt prevents subsequent launches; active jobs drain; no automatic batch/protocol changes.',
            'scientific_scope': 'Seven rows =21 seed cells; reuse15 completed, train6 new. Not the full8-cell factorial.'}
    (E / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'stages':len(stages), 'new_formal_training':6, 'smokes':2, 'gpus':POOL, 'execute':False}))


if __name__ == '__main__':
    main()
