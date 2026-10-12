"""Create immutable formal/smoke configs from the reviewed mechanism plan."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
TARGET = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920')
OLD = TARGET.parent / 's3-reference-hyperparams-20260920'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    plan = json.loads((HERE / 'REVIEWER_MECHANISM_PLAN.json').read_text())
    jobs = plan['jobs']
    base = json.loads((HERE / 'base_config.json').read_text())
    parent = json.loads((HERE / 'source_parent_manifest.json').read_text())
    files = dict(parent['files'])
    for path, expected in files.items():
        assert sha(HERE / 'source' / path) == expected, path
    added = 'utils/s3_mechanism_controls.py'
    files[added] = sha(HERE / 'source' / added)
    write(HERE / 'source_manifest.json', {
        'origin': str(OLD / 'source'),
        'parent_manifest_sha256': sha(HERE / 'source_parent_manifest.json'),
        'unchanged_parent_files': len(parent['files']),
        'added_files': [added], 'files': files,
    })
    old_suite = json.loads((OLD / 'suite.json').read_text())
    suite = {k: v for k, v in old_suite.items() if k != 'jobs'}
    suite.update(allowed_gpus=list(range(7)), max_parallel=7, source_root=str(TARGET / 'source'),
                 source_manifest=str(TARGET / 'source_manifest.json'),
                 final_report_script=str(TARGET / 'finish_analysis.py'),
                 required_preflight_review='preflight/smoke_review.json',
                 integration_campaign='preflight/smoke_attempt2', jobs=[])
    formal_configs = []
    for j in jobs:
        ident = j['id']
        cfg = {**base, **j['overrides'], 'seed': [j['seed']],
               'outputs_root': str(TARGET / 'outputs' / ident),
               'prefix': ident, 'optimizer_tag_override': 's3_native_mechanism',
               's3_boundary_capture': True}
        cfg.setdefault('s3_fisher_location', 'retained')
        path = HERE / 'configs' / (ident + '.json')
        write(path, cfg)
        formal_configs.append(cfg)
        suite['jobs'].append({
            'id': ident, 'seed': j['seed'], 'axis': j.get('axis', ident),
            'value': j.get('value'), 'config': str(TARGET / 'configs' / path.name),
            'config_sha256': sha(path), 'depends_on': j.get('depends_on', []),
            'what_it_tests': j['what_it_tests'],
        })
    write(HERE / 'suite.json', suite)
    smoke_here = HERE / 'preflight' / 'smoke_attempt2'
    smoke_target = TARGET / 'preflight' / 'smoke_attempt2'
    smoke_here.mkdir(parents=True)
    for name in ('common.py', 'run_job.py', 'launch_queue.py', 'collect_results.py'):
        shutil.copy2(HERE / name, smoke_here / name)
    (smoke_here / 'preflight').mkdir()
    shutil.copy2(HERE / 'preflight/verify_boundaries.py', smoke_here / 'preflight/verify_boundaries.py')
    smoke_suite = {**suite, 'expected_tasks': 2, 'jobs': []}
    smoke_suite.pop('final_report_script', None)
    smoke_suite.pop('required_preflight_review', None)
    # Two seed2 center executions certify that the observer does not change training.
    observer_pair = [dict(base, prefix='observer_on_s2', s3_boundary_capture=True),
                     dict(base, prefix='observer_off_s2', s3_boundary_capture=False)]
    for cfg in observer_pair + formal_configs:
        ident = 'smoke_' + cfg['prefix']
        cfg = {**cfg, 'prefix': ident, 'outputs_root': str(smoke_target / 'outputs' / ident),
               'optimizer_tag_override': 's3_mechanism_smoke',
               'max_train_tasks': 2, 'nb_tasks': 2, 'total_sessions': 2, 'init_epoch': 2, 'epochs': 2,
               'in_task_fisher_interval': 1}
        path = smoke_here / 'configs' / (ident + '.json')
        write(path, cfg)
        smoke_suite['jobs'].append({'id': ident, 'seed': cfg['seed'][0],
                                   'axis': 'integration_only', 'value': None,
                                   'config': str(smoke_target / 'configs' / path.name),
                                   'config_sha256': sha(path), 'depends_on': []})
    write(smoke_here / 'suite.json', smoke_suite)
    write(HERE / 'runtime_manifest.json', {
        'files': {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*.py')
                  if not p.is_relative_to(HERE / 'source')},
        'suite_sha256': sha(HERE / 'suite.json'),
        'smoke_suite_sha256': sha(smoke_here / 'suite.json'),
    })
    print(json.dumps({'formal_jobs': len(suite['jobs']), 'smoke_jobs': len(smoke_suite['jobs']),
                      'parent_files_unchanged': len(parent['files']), 'source_files': len(files)}))


if __name__ == '__main__':
    main()
