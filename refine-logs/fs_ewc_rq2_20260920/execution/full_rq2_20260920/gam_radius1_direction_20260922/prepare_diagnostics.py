"""Prepare the two-direction diagnostics and CPU publisher without launching."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'


def main():
    campaign = HERE / 'diagnostic_campaign'
    if (campaign / 'pipeline_state.json').exists() or (campaign / 'queue_state.json').exists():
        raise RuntimeError('Do not regenerate a launched diagnostic plan.')
    campaign.mkdir(exist_ok=True)
    train_plan = json.loads((HERE / 'training_campaign/plan.json').read_text())
    files = set(train_plan['code_files'])
    files |= {str(HERE / name) for name in ['diagnostic.py', 'check_diagnostic.py', 'publish.py',
              'prepare_diagnostics.py', 'wait_and_run_diagnostics.py', 'launch_diagnostics.py',
              'RUN_DIAGNOSTICS.md']}
    files |= {str(EXECUTION / name) for name in ['diagnostic.py', 'rq2_common.py']}
    files |= {str(BASE / 'diagnostic_full.py'), str(HERE / 'training_campaign/plan.json')}
    files.add(str(BASE / 'figures_formal_20260921/plot_common.py'))
    files |= {str(Path('/home-local/liying/.fonts') / name)
              for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']}
    for folder in ['fixed_gradient_extension_20260921', 'task1_gradient_extension_20260921']:
        files |= {str(p) for p in (BASE / folder).rglob('*.py')}
    jobs = []
    for arm in ['probe005_norm1', 'probe1_norm1']:
        for seed in range(3):
            config = HERE / f'configs/{arm}_seed{seed}.json'
            files.add(str(config))
            cfg = json.loads(config.read_text())
            assert cfg['radii'] == [i / 2 for i in range(11)]
            assert cfg['intervention_arm'] == arm and cfg['training_seed'] == seed
            for task, name, sign in [(0, 'task0_positive', 1), (1, 'task1_negative', -1)]:
                jid = f'{arm}_seed{seed}_{name}'
                receipt = Path(cfg['output_dir']) / 'gam' / name / 'extension_receipt.json'
                jobs.append(dict(id=jid, gpu=0, gpu_pool=[0, 1], cwd=str(HERE),
                    command=[PYTHON, '-u', str(HERE / 'diagnostic.py'), '--config', str(config),
                             '--gradient-task', str(task), '--device', 'cuda:0'],
                    env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                    preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                       '--output', str(campaign / f'{jid}_env.json')],
                    receipt=str(receipt), receipt_require=dict(status='complete', seed=seed,
                    method='gam', gradient_task=task, direction_sign=sign,
                    evaluation_task=0, expected_states=11)))
    publisher = dict(id='publish_radius_comparison', gpu=None, cwd=str(HERE),
        command=[PYTHON, '-u', str(HERE / 'publish.py'), '--execute'],
        receipt=str(HERE / 'report/radius_intervention_complete.json'),
        receipt_require=dict(status='complete', expected_jobs=12, validated_jobs=12,
                             new_states=132, plotted_rows=264, training_anchors=6))
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    plan = dict(schema='rq2_radius_pair_fixed_direction_v1', gpus=[0, 1], max_parallel=2,
        run_dir=str(campaign), gpu_lock_dir=train_plan['gpu_lock_dir'],
        start_after=train_plan['start_after'], code_files=sorted(files),
        stages=[dict(id='task0_positive', depends_on=[],
                     jobs=[j for j in jobs if j['receipt_require']['gradient_task'] == 0]),
                dict(id='task1_negative', depends_on=['task0_positive'],
                     jobs=[j for j in jobs if j['receipt_require']['gradient_task'] == 1]),
                dict(id='publication', depends_on=['task1_negative'], jobs=[publisher])])
    (campaign / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(status='prepared', jobs=13, gpu_jobs=12, states=132, gpus=[0, 1])))


if __name__ == '__main__':
    main()
