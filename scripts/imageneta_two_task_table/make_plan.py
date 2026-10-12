#!/usr/bin/env python3
"""Prepare the exact six-row table DAG. Does not start jobs."""
import argparse
import json
from pathlib import Path
import sys

from train import RULES, runtime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'


def build(campaign):
    run_root = campaign.parent / 'runs'
    smoke_root = campaign.parent / 'smoke/runs'
    stages = []
    config = str(HERE / 'config.json')
    env_probe = ROOT / 'refine-logs/fs_ewc_rq2_20260920/execution/env_probe.py'

    def stage(name, deps, jobs):
        stages.append({'id': name, 'depends_on': deps, 'jobs': jobs})

    def job(name, command, gpu, receipt, require, witness=False):
        result = {'id': name, 'command': command, 'gpu': gpu, 'receipt': str(receipt),
                  'receipt_require': require, 'cwd': str(ROOT)}
        if witness:
            result['preflight_command'] = [PYTHON, str(env_probe), '--output',
                                            str(campaign / f'{name}_environment.json')]
        return result

    def cohort(prefix, root, seeds, prerequisite, smoke):
        previous = list(prerequisite)
        for seed in seeds:
            tag = f'{prefix}_s{seed}'
            anchor_jobs = []
            for method, gpu in [('sgd', 4), ('gam', 6)]:
                cmd = [PYTHON, str(HERE / 'train.py'), '--stage', 'task0', '--method', method,
                       '--seed', str(seed), '--gpu', str(gpu), '--output-root', str(root), '--config', config]
                if smoke:
                    cmd += ['--smoke']
                anchor_jobs.append(job(f'{tag}_anchor_{method}', cmd, gpu,
                    root / 'anchors' / method / f'seed{seed}/TASK0_COMPLETE.json',
                    {'status': 'complete', 'completed_tasks': 1, 'smoke': smoke}, witness=True))
            stage(tag + '_anchors', previous, anchor_jobs)
            branches = []
            # Both cards may execute either optimizer, balancing expensive GAM branches.
            gpu_by_method = {'gam_ewc': 4, 'shaping_only': 6, 'task1_shaping': 4,
                             'ewc': 6, 'task0_shaping': 6, 'vanilla': 6}
            order = ['gam_ewc', 'shaping_only', 'task1_shaping', 'ewc', 'task0_shaping', 'vanilla']
            for method in order:
                gpu = gpu_by_method[method]
                cmd = [PYTHON, str(HERE / 'train.py'), '--stage', 'task1', '--method', method,
                       '--seed', str(seed), '--gpu', str(gpu), '--output-root', str(root), '--config', config]
                if smoke:
                    cmd += ['--smoke']
                branches.append(job(f'{tag}_{method}_train', cmd, gpu,
                    root / method / f'seed{seed}/COMPLETE.json',
                    {'status': 'complete', 'completed_tasks': 2, 'smoke': smoke}))
            stage(tag + '_branches', [tag + '_anchors'], branches)
            evaluations = []
            for idx, method in enumerate(RULES):
                run = root / method / f'seed{seed}'
                evaluations.append(job(f'{tag}_{method}_eval',
                    [PYTHON, str(HERE / 'evaluate.py'), '--run-dir', str(run), '--device', 'cuda',
                     '--fisher-samples', '8' if smoke else '256'], 4 if idx % 2 == 0 else 6,
                    run / 'evaluation/COMPLETE.json', {'status': 'complete'}))
            stage(tag + '_evaluations', [tag + '_branches'], evaluations)
            selected = [s for s in seeds if s <= seed]
            # Each seed gets a persistent receipt; aggregate tables can grow from 1 to 3 seeds.
            cmd = [PYTHON, str(HERE / 'summarize_stage.py'), '--output-root', str(root),
                   '--seeds', ','.join(map(str, selected))] + (['--allow-smoke'] if smoke else [])
            stage(tag + '_summary', [tag + '_evaluations'], [job(tag + '_summary', cmd, None,
                root / 'comparison' / f'COMPLETE_seed{seed}.json',
                {'status': 'complete', 'smoke': smoke, 'runs': 6 * len(selected)})])
            previous = [tag + '_summary']
        return previous

    smoke_final = cohort('smoke', smoke_root, [0], [], True)
    stage('smoke_verified', smoke_final, [job('smoke_verified',
        [PYTHON, str(HERE / 'verify_smoke.py'), '--output-root', str(smoke_root)], None,
        smoke_root / 'SMOKE_VERIFIED.json', {'status': 'passed', 'paper_evidence_allowed': False})])
    cohort('formal', run_root, [0, 1, 2], ['smoke_verified'], False)
    code = [str(f) for f in sorted(HERE.glob('*.py'))] + [config, str(env_probe)]
    code += [str(HERE.parent / 'imagenetr_gam_ewc_1x1' / f) for f in ('train.py', 'learner.py')]
    code += [str(Path('/data/140-1/users/liying/CopyFlatCV') / f) for f in runtime.SOURCE_FILES]
    return {'project': 'imageneta_two_task_six_row_table', 'run_dir': str(campaign),
            'gpus': [4, 6], 'max_parallel': 2, 'code_files': code, 'stages': stages,
            'formal_runs': 18, 'formal_anchor_trainings': 6, 'formal_branch_trainings': 18,
            'failure_policy': 'stop dependent work; preserve failed attempt; no automatic budget changes',
            'start_after': {'gpu_uuids': {'4': 'GPU-e97b92ec-d10c-0913-e3f8-67941b20ee0c',
                                         '6': 'GPU-854d8366-b2d5-676d-1c34-70a08132c88f'}}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-dir', type=Path,
                   default=ROOT / 'refine-logs/imageneta_two_task_table_20260920/campaign')
    args = p.parse_args()
    path = args.campaign_dir.resolve()
    path.mkdir(parents=True, exist_ok=True)
    target = path / 'plan.json'
    if target.exists():
        raise FileExistsError(target)
    plan = build(path)
    target.write_text(json.dumps(plan, indent=2) + '\n')
    print(target)
