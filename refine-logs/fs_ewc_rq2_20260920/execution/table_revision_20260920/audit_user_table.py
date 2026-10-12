"""Independently re-read the exact historical ImageNet-R cohort; no GPU work."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics

OUT = Path(__file__).resolve().parent
OLD = Path('/data/115-2/users/liying/low-rank-cl-new')
TAG = 'verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7'
METHODS = ['baseline', 'gam_only', 'control', 'full', 'setting1', 'setting2']
REPORTED = [
    [90.73, 19.885, 84.85, 89.59, -5.87, 22.469, 1.560],
    [93.34, 10.946, 84.80, 93.81, -8.55, 10.471, 3.131],
    [90.73, 19.885, 87.95, 83.45, -2.78, 24.599, .0106],
    [93.34, 10.946, 89.10, 88.33, -4.24, 17.931, .0204],
    [93.34, 10.946, 91.82, 82.57, -1.52, 18.734, .392],
    [90.73, 19.885, 83.07, 89.21, -7.66, 41.316, .129],
]
FIELDS = ['a11', 'F1_batch', 'a21', 'a22', 'BWT', 'F2_batch', 'r_KV']
source_csv = OLD / 'result_summary' / TAG / 'table_source.csv'
with source_csv.open() as f:
    reference = {(r['setting'], int(r['seed'])): r for r in csv.DictReader(f)}
runs = []
for method in METHODS:
    for seed in (0, 1, 2):
        log = OLD / 'logs' / TAG / 'shell' / f'{method}_s012_rho0p05_normrho0p5_lam1e7_seed{seed}.log'
        cfg = OLD / 'configs_sweeps' / TAG / f'seed{seed}' / f'{method}.json'
        accuracy, mechanism, lines = [], {}, {}
        for line_no, line in enumerate(log.read_text().splitlines(), 1):
            if '=> Accuracy: {' in line:
                accuracy.append(ast.literal_eval(line[line.index('{'):]))
                lines[f'accuracy_{len(accuracy)-1}'] = line_no
            if '[Mechanism] {' in line:
                rec = ast.literal_eval(line[line.index('{'):])
                mechanism[int(rec['task'])] = rec
                lines[f'mechanism_{rec["task"]}'] = line_no
        assert len(accuracy) == 2 and set(mechanism) == {0, 1}
        m0, m1 = mechanism[0], mechanism[1]
        v = {
            'a11': accuracy[0]['00-19'],
            'F1_batch': m0['endpoint_fisher_total_mass'],
            'a21': accuracy[1]['00-19'],
            'a22': accuracy[1]['20-39'],
            'BWT': accuracy[1]['00-19'] - accuracy[0]['00-19'],
            'F2_batch': m1['endpoint_fisher_total_mass'],
            'r_KV': math.sqrt(m1['delta_norm']),
            'mean_accuracy': (accuracy[1]['00-19'] + accuracy[1]['20-39']) / 2,
            'r_KV_squared': m1['delta_norm'],
            'q_F1': m1['fisher_weighted_drift'],
            'q_over_r_squared': m1['normalized_fisher_drift'],
            'ewc_endpoint_penalty': m1['ewc_penalty_value'],
            'accumulated_F2_batch': m1['accumulated_fisher_total_mass'],
        }
        assert all(math.isfinite(x) for x in v.values())
        assert math.isclose(m1['lambda_ewc'] * v['q_F1'] / 2,
                            v['ewc_endpoint_penalty'], rel_tol=1e-6, abs_tol=1e-8)
        ref = reference[(method, seed)]
        maps = {'a11': 'R1_A_after_T1', 'a21': 'R2_A_after_T2', 'a22': 'R2_B_after_T2',
                'F1_batch': 'after_T1_fisher_total_mass', 'F2_batch': 'after_T2_fisher_total_mass',
                'r_KV_squared': 'theta2_minus_theta1_norm_sq'}
        for field, column in maps.items():
            assert math.isclose(v[field], float(ref[column]), rel_tol=1e-9, abs_tol=1e-10)
        runs.append({'setting': method, 'seed': seed, 'values': v,
                     'config_values': json.loads(cfg.read_text()), 'log': str(log),
                     'config': str(cfg), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
                     'config_sha256': hashlib.sha256(cfg.read_bytes()).hexdigest(),
                     'line_numbers': lines,
                     'checkpoint_disabled_in_log': 'save_ckp: False' in log.read_text()})
summary = []
for method, reported in zip(METHODS, REPORTED):
    rr = [r['values'] for r in runs if r['setting'] == method]
    mean = {k: statistics.mean(r[k] for r in rr) for k in rr[0]}
    sd = {k: statistics.stdev(r[k] for r in rr) for k in rr[0]}
    decimals = [2, 3, 2, 2, 2, 3, 4 if method in {'control', 'full'} else 3]
    checks = {k: round(mean[k], dp) == value for k, dp, value in zip(FIELDS, decimals, reported)}
    assert all(checks.values()), checks
    summary.append({'setting': method, 'seeds': [0, 1, 2], 'n': 3, 'mean': mean,
                    'sample_sd_ddof1': sd, 'matches_user_table': checks})
control = {r['seed']: r['values'] for r in runs if r['setting'] == 'control'}
shaped = {r['seed']: r['values'] for r in runs if r['setting'] == 'setting1'}
paired = [{ 'seed': s, **{k: shaped[s][k] - control[s][k]
                        for k in ['a11', 'a21', 'a22', 'BWT', 'mean_accuracy']}}
          for s in (0, 1, 2)]
result = {'scope': 'Original ImageNet-R logs/configs, independent re-read and aggregation; no GPU or historical checkpoint loading.',
          'validated_runs': len(runs), 'expected_runs': 18,
          'all_42_displayed_values_match': all(all(s['matches_user_table'].values()) for s in summary),
          'source_csv': str(source_csv),
          'source_csv_sha256': hashlib.sha256(source_csv.read_bytes()).hexdigest(),
          'runs': runs, 'summary': summary, 'setting1_minus_control_paired': paired,
          'limitations': ['All 18 logs disable checkpoint saving; this verifies original logs and aggregation, not historical tensors.',
                          'Historical run code SHA was not recorded; exact estimator interpretation uses matching current source and earlier code audit.',
                          'The Fisher columns are per-current-task batch-gradient buffers, not the new ImageNet-A per-example old-task raw Fisher.']}
(OUT / 'USER_TABLE_PROVENANCE.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in {'runs', 'summary'}}, indent=2, ensure_ascii=False))
for s in summary:
    print(s['setting'], s['mean'])
