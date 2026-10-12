"""Build a finite, two-seed, one-axis S3 search, reusing verified completed cells."""
import hashlib
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs')
OLD = ROOT / 's3-reference-hyperparams-20260920'
MECH = ROOT / 's3-mechanisms-20260920'
TARGET = ROOT / 's3-hparam-search-20260921'
BUILD = HERE / 'campaign'
AXES = {
    'ewc_lambda': [0, 200, 2000, 20000],
    'gam_grad_norm_rho': [.1, .2, .5],
    'ewc_fisher_damping_alpha': [0., .1, .2, .5, 1.],
    'in_task_fisher_omega': [.05, .1, .2, .3, .4, .5],
}
ALIASES = {'ewc_lambda': 'lambda', 'gam_grad_norm_rho': 'radius',
           'ewc_fisher_damping_alpha': 'damping', 'in_task_fisher_omega': 'weightmerge'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(rel, obj):
    p = BUILD / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def token(value):
    return str(value).replace('.', 'p')


def main():
    BUILD.mkdir(exist_ok=False)
    native_suite = read(OLD / 'suite.json')
    base = read(OLD / 'configs/s3_center_s2.json')
    old_source = read(OLD / 'source_manifest.json')
    for rel, expected in old_source['files'].items():
        assert sha(OLD / 'source' / rel) == expected
        dst = BUILD / 'source' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD / 'source' / rel, dst)
    # Exact native source and native runner from the already successful 8-run search.
    shutil.copy2(OLD / 'source_manifest.json', BUILD / 'source_manifest.json')
    shutil.copy2(OLD / 'run_job.py', BUILD / 'run_job.py')
    shutil.copy2(OLD / 'collect_results.py', BUILD / 'collect_results.py')
    # Current helpers/queue already support all seven authorized physical GPUs.
    for name in ['common.py', 'launch_queue.py']:
        shutil.copy2(MECH / name, BUILD / name)
    shutil.copy2(OLD / 'env_witness.json', BUILD / 'reused_env_witness.json')
    witness = read(OLD / 'env_witness.json')
    assert witness['passed'] and sha(witness['spec']) == witness['spec_raw_sha256']
    write('base_config.json', base)
    reused_specs = {
        ('center', None, 2): (OLD, 's3_center_s2'),
        ('center', None, 3): (MECH, 's3_mech_center_s3'),
        ('ewc_lambda', 0, 2): (OLD, 's3_lambda0_s2'),
        ('ewc_lambda', 200, 2): (OLD, 's3_lambda200_s2'),
        ('ewc_lambda', 20000, 2): (OLD, 's3_lambda20000_s2'),
        ('gam_grad_norm_rho', .1, 2): (OLD, 's3_radius01_s2'),
        ('gam_grad_norm_rho', .5, 2): (OLD, 's3_radius05_s2'),
        ('ewc_fisher_damping_alpha', 0., 2): (MECH, 's3_mech_fisher_alpha0_s2'),
        ('ewc_fisher_damping_alpha', 1., 2): (MECH, 's3_mech_fisher_alpha1_s2'),
    }
    reuse = []
    ignored = {'prefix', 'outputs_root', 'optimizer_tag_override', 's3_boundary_capture', 's3_fisher_location'}
    for (axis, value, seed), (cam, ident) in reused_specs.items():
        cfg = read(cam / 'configs' / (ident + '.json'))
        expected = dict(base, seed=[seed])
        if axis != 'center':
            expected[axis] = value
        differences = {k: [expected.get(k), cfg.get(k)] for k in set(expected) | set(cfg)
                       if k not in ignored and expected.get(k) != cfg.get(k)}
        assert not differences, (ident, differences)
        assert cfg.get('s3_fisher_location', 'retained') == 'retained'
        receipt_path = cam / 'runtime/jobs' / ident / 'completed.json'
        receipt = read(receipt_path)
        assert receipt['status'] == 'completed'
        assert receipt['config_sha256'] == sha(cam / 'configs' / (ident + '.json'))
        assert receipt['source_manifest_sha256'] == sha(cam / 'source_manifest.json')
        current_sources = read(cam / 'source_manifest.json')['files']
        assert all(current_sources.get(k) == v for k, v in old_source['files'].items())
        result = receipt['result']
        mp = Path(result['metrics_path'])
        assert sha(mp) == result['metrics_sha256']
        m = read(mp)['cnn']['matrices']['final']
        assert len(m) == 10 and all(len(row) == 10 for row in m)
        assert all(math.isfinite(v) for i, row in enumerate(m) for v in row[:i+1])
        assert abs(sum(m[-1]) / 10 - result['faa_task_macro']) < 1e-9
        if cfg.get('s3_boundary_capture'):
            assert result['boundary_validation']['passed']
        reuse.append({'axis': axis, 'value': value, 'seed': seed, 'id': ident,
                      'config_path': str(cam / 'configs' / (ident + '.json')),
                      'config_sha256': receipt['config_sha256'], 'receipt_path': str(receipt_path),
                      'receipt_sha256': sha(receipt_path), 'metrics_path': str(mp),
                      'metrics_sha256': result['metrics_sha256'],
                      'source_manifest_sha256': receipt['source_manifest_sha256'],
                      'uses_boundary_observer': bool(cfg.get('s3_boundary_capture')),
                      'same_native_source_files': len(old_source['files']),
                      'scope': 'Scientific configuration and native source match; observer runs passed boundary preservation. Historical June29 scores remain separate.'})
    jobs = []
    cells = []
    for axis, values in AXES.items():
        for value in values:
            for seed in [2, 3]:
                key = ('center', None, seed) if value == base[axis] else (axis, value, seed)
                if key in reused_specs:
                    cam, ident = reused_specs[key]
                    cells.append({'axis': axis, 'value': value, 'seed': seed, 'id': ident,
                                  'source': 'external_completed', 'campaign': str(cam)})
                    continue
                if seed != 2:
                    continue  # User: new training only seed2; reuse existing seed3.
                ident = f's3_hp_{ALIASES[axis]}_{token(value)}_s{seed}'
                cfg = dict(base, seed=[seed], prefix=ident,
                           outputs_root=str(TARGET / 'outputs' / ident),
                           optimizer_tag_override='s3_reference_native_hparam_search')
                cfg[axis] = value
                assert {k for k in cfg if cfg[k] != dict(base, seed=[seed]).get(k)} <= {
                    axis, 'prefix', 'outputs_root', 'optimizer_tag_override'}
                write('configs/' + ident + '.json', cfg)
                job = {'id': ident, 'axis': axis, 'value': value, 'seed': seed,
                       'config': str(TARGET / 'configs' / (ident + '.json')),
                       'config_sha256': sha(BUILD / 'configs' / (ident + '.json')), 'depends_on': []}
                jobs.append(job)
                cells.append({'axis': axis, 'value': value, 'seed': seed, 'id': ident,
                              'source': 'new_search', 'campaign': str(TARGET)})
    # Prioritize missing seed3 radius confirmation and the new positive-omega curve.
    axis_priority = {'gam_grad_norm_rho': 0, 'in_task_fisher_omega': 1,
                     'ewc_fisher_damping_alpha': 2, 'ewc_lambda': 3}
    jobs.sort(key=lambda j: (axis_priority[j['axis']], 0 if j['value'] == .5 else 1,
                             j['value'], j['seed']))
    assert len(reuse) == 9 and len(jobs) == 7 and len(cells) == 22
    suite = {k: v for k, v in native_suite.items() if k not in ['jobs', 'final_report_script', 'required_preflight_review', 'integration_campaign']}
    suite.update(allowed_gpus=list(range(7)), max_parallel=7,
                 source_root=str(TARGET / 'source'), source_manifest=str(TARGET / 'source_manifest.json'),
                 final_report_script=str(TARGET / 'summarize_search.py'), jobs=jobs)
    write('suite.json', suite)
    write('reused_cells.json', reuse)
    write('grid_cells.json', cells)
    write('SEARCH_PLAN.json', {
        'reference_logs': [str(ROOT.parent / 'logs_try/logs_s3_damp_c3_multiseed' / f's3_damp_c3_alpha02_int20_s{s}_gpu{g}.log') for s, g in [(2, 3), (3, 4)]],
        'axes': AXES, 'new_training_seeds': [2], 'statistics_included_seeds': [2, 3], 'design': 'one_axis_at_a_time', 'unique_config_seed_cells': 16,
        'axis_table_rows_with_repeated_centers': 22, 'reused_unique_cells': 9, 'new_jobs': 7,
        'weightmerge_parameter': 'in_task_fisher_omega is the weight-only coefficient in WMergeThenFisher; no trajectory Fisher is computed.',
        'zero_merge_control': 'Existing matched_no_merge seed2/3 remains a separate CE-matched control; native omega=0 disables snapshot CE and is not the same intervention.',
        'full_training_budget': '10 tasks x 50 epochs, batch128, fixed task order, no replay',
        'selection_scope': 'Exploratory fixed-grid sensitivity analysis; historical and new repeat scores are not interchangeable and test-score selection is not confirmation.',
        'native_source_manifest_sha256': sha(BUILD / 'source_manifest.json'),
        'environment_spec_sha256': sha(witness['spec']), 'environment_changed': False,
        'existing_mechanism_queue_modified': False,
    })
    print(json.dumps({'staging': str(BUILD), 'target': str(TARGET), 'reused': len(reuse),
                      'new': len(jobs), 'first_jobs': [j['id'] for j in jobs[:4]]}))


if __name__ == '__main__':
    main()
