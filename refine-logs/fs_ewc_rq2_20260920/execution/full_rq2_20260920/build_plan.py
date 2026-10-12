"""Materialize the authorized, fresh three-seed RQ2 DAG without launching it."""
import json
from pathlib import Path

N = Path(__file__).resolve().parent
E = N.parent
PY = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
TRAIN_POOL = [4, 5, 6, 2, 3]
DIAG_POOL = [0, 1, 2, 3, 4, 5, 6]
ENV = {'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def job(jid, script, arguments, receipt, require, pool=DIAG_POOL):
    return dict(id=jid, gpu=pool[0] if pool else None, gpu_pool=pool,
                command=[PY, '-u', str(N / script)] + list(map(str, arguments)),
                cwd=str(N), env=ENV, receipt=str(receipt), receipt_require=require)


def stage(j, dependencies):
    return dict(id=j['id'], depends_on=dependencies, jobs=[j])


def train(jid, mode, method, seed, lam, anchor=None):
    args = ['--mode', mode, '--method', method, '--seed', seed,
            '--ewc-lambda', lam, '--run-name', jid, '--output-root', N / 'formal_training',
            '--workers', 4, '--gpu', '{gpu}', '--resume-last']
    if anchor:
        parent = N / 'formal_training' / anchor
        args += ['--anchor-dir', parent / 'checkpoints', '--anchor-metrics', parent / 'trainer_outputs']
    count = {'anchor': 1, 'continuation': 2, 'sequence': 10}[mode]
    return job(jid, 'train.py', args, N / 'formal_training' / jid / 'COMPLETE.json',
               {'status': 'complete', 'completed_tasks': count}, TRAIN_POOL)


def main():
    groups = {k: [] for k in ['anchors', 'b3_training', 'sequences', 'endpoints',
                              'sparse', 'trajectory', 'path', 'bridge', 'dense']}
    config_paths = []
    b3_inputs = {}
    for seed in range(3):
        config = json.loads((E / 'diagnostic_config.json').read_text())
        config.update(output_dir=str(N / 'formal_results' / f'seed{seed}'), training_seed=seed,
                      workers=4, subset_seed=20260920, direction_seed=20260920,
                      dense_radii=[i / 4 for i in range(81)], dense_directions=100,
                      sparse_train_radii=[0, .5, 1, 2, 5, 10, 20],
                      sparse_test_radii=[0, 1, 10, 20], sparse_train_full_radii=[0, 1, 10, 20],
                      sparse_directions=16, full_train_direction_indices=[0, 1, 2, 3],
                      bridge_base_radii=[0, .25, .5, 1, 1.5, 2], bridge_random_directions=100,
                      quadrature_atol=1e-8, quadrature_rtol=.05, bound_rtol=.02,
                      path_peak_neighbor_factor=4.)
        config['checkpoints'] = {}
        cfg_path = N / 'configs' / f'seed{seed}.json'
        config_paths.append(cfg_path)
        b3_inputs[str(seed)] = {}
        for method in ['sgd', 'gam']:
            anchor = f'anchor_{method}_seed{seed}'
            sequence = f'sequence_{method}_seed{seed}'
            groups['anchors'].append(stage(train(anchor, 'anchor', method, seed, 2000), []))
            groups['sequences'].append(stage(train(sequence, 'sequence', method, seed, 2000, anchor), [anchor]))
            config['checkpoints'][method] = {
                str(t): str(N / 'formal_training' / (anchor if t == 0 else sequence) /
                            'checkpoints' / f'task_{t:02d}.pt') for t in range(10)}
            branches = [('A', 0), ('C', 2000)] if method == 'sgd' else [('B', 0), ('D', 2000)]
            for branch, lam in branches:
                run_name = sequence if branch == 'C' else f'B3_{branch}_{method}_lam{lam}_seed{seed}'
                if branch != 'C':
                    groups['b3_training'].append(stage(train(run_name, 'continuation', method, seed, lam, anchor), [anchor]))
                output = N / 'formal_tables' / f'seed{seed}' / branch
                ep = job(f'endpoint_{branch}_seed{seed}', 'endpoint.py',
                         ['--config', cfg_path, '--method', method, '--run-dir', N / 'formal_training' / run_name,
                          '--out-dir', output], output / 'summary.json',
                         {'status': 'complete', 'method': method, 'seed': seed, 'ewc_lambda': lam,
                          'future_optimizer': 'sgd', 'shaping_in_task1': False})
                groups['endpoints'].append(stage(ep, [run_name]))
                b3_inputs[str(seed)][branch] = str(output / 'summary.json')
            for kind in ['sparse_train', 'sparse_test', 'sparse_train_full', 'trajectory_fisher', 'path', 'bridge_full', 'dense']:
                tasks = [1, 5, 9] if kind == 'path' else [None]
                for task in tasks:
                    label = f'path_t{task:02d}' if task else kind
                    args = ['--config', cfg_path, '--method', method, '--job', kind]
                    if task:
                        args += ['--path-task', task]
                    # CUDA_VISIBLE_DEVICES is the queue's UUID. Do not override it with an ordinal.
                    requirement = {'status': 'complete', 'method': method, 'job': kind, 'seed': seed}
                    if task:
                        requirement.update(target_task=task, samples=1259, numerical_gate_passed=True,
                                           quadrature_converged=True)
                    if kind == 'dense':
                        requirement.update(forward_only=True, expected_states=8001)
                    if kind == 'trajectory_fisher':
                        requirement.update(task_count=10, train_fisher_samples=1259)
                    out = N / 'formal_results' / f'seed{seed}' / method / label / 'summary.json'
                    j = job(f'{method}_seed{seed}_{label}', 'diagnostic_full.py', args, out, requirement)
                    deps = ([f'sequence_sgd_seed{seed}', f'sequence_gam_seed{seed}'] if kind == 'bridge_full'
                            else [sequence] if kind in ('trajectory_fisher', 'path') else [anchor])
                    group = ('sparse' if kind.startswith('sparse') else 'trajectory' if kind == 'trajectory_fisher'
                             else 'bridge' if kind == 'bridge_full' else kind)
                    groups[group].append(stage(j, deps))
        dump(cfg_path, config)
    stages = [s for group in groups.values() for s in group]
    report = job('final_report', 'report.py', ['--root', N, '--formal-plan', N / 'formal_plan.json'],
                 N / 'final_report' / 'COMPLETE.json', {'status': 'complete'}, [])
    stages.append(stage(report, [s['id'] for s in stages]))
    source_files = [E / 'source' / f['path'] for f in json.loads((E / 'source_frozen.json').read_text())['files']]
    code = [N / name for name in ('campaign_runner.py', 'train.py', 'endpoint.py', 'diagnostic_full.py', 'report.py', 'build_plan.py')]
    code += [E / 'source_frozen.json', E / 'rq2_common.py', E / 'diagnostic.py'] + source_files + config_paths
    pilot = json.loads((N / 'pilot_plan_paired.json').read_text())
    plan = dict(version=1, purpose='Full RQ2 B1/B2/B3; homogeneous fresh FP32/RNG seeds 0,1,2; historical pilot kept separate',
                seeds=[0, 1, 2], gpus=list(range(7)), max_parallel=7,
                run_dir=str(N / 'formal_queue'), gpu_lock_dir=pilot['gpu_lock_dir'],
                start_after=pilot['start_after'], code_files=list(map(str, code)), stages=stages,
                report_inputs={'b3': b3_inputs},
                schedule={'training_pool': TRAIN_POOL, 'diagnostic_preference': DIAG_POOL,
                          'reason': 'GPU01 available for diagnostics while GPU23 and 456 train; diagnostics can claim any idle GPU.'},
                workload={'training_jobs': 21, 'training_task_segments': 69, 'epochs_per_segment': 50,
                          'diagnostic_jobs': 54, 'B3_endpoint_jobs': 12, 'report_jobs': 1, 'total_jobs': len(stages)},
                failure_policy='Nonzero exit or invalid receipt stops new launches; active jobs drain; no blind change of batch size or protocol.')
    dump(N / 'formal_plan.json', plan)
    print(json.dumps(plan['workload'], indent=2))


if __name__ == '__main__':
    main()
