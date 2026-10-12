#!/usr/bin/env python3
"""Recompute figure metrics from hash-checked logs and verify exported SVG points."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLOT = ROOT / 'figures/lora_taskwise_20260918/aa_times'
sys.path.insert(0, str(PLOT))
import plot_lora_aa as base

manifest, groups, records = base.load_runs(PLOT / 'sources.json')
output = PLOT / 'output'
stem = 'lora_taskwise_aa_compact_tighty'
display = json.loads((output / f'{stem}_display.json').read_text())
order = display['dataset_order']
methods = ['InfLoRA', 'GAM-EWC+Merge']
metric_keys = {'task_macro_faa': 'aa', 'task_macro_aaa': 'aaa',
               'sample_weighted_faa': 'sample_accuracy',
               'sample_weighted_aaa': 'previous_sample_aaa'}
summary = []
for dataset in order:
    for method in methods:
        runs = groups[(dataset, method)]
        row = {'dataset': dataset, 'method': method,
               'n': len(runs), 'seeds': [r['seed'] for r in runs]}
        for label, key in metric_keys.items():
            mean, sd = base.summarize(runs, key)
            row[label] = float(mean[-1])
            row[label + '_sd'] = float(sd[-1]) if len(runs) > 1 else None
        row['aa_curve'] = base.summarize(runs, 'aa')[0].tolist()
        row['source_logs'] = [r['log'] for r in runs]
        row['historical_table_snapshot'] = next(
            t for t in manifest['reported_table']
            if t['dataset'] == dataset and t['method'] == method)
        summary.append(row)

with (output / 'stage_statistics.csv').open() as f:
    stats = list(csv.DictReader(f))
csv_checked = 0
for row in stats:
    if row['method'] not in methods:
        continue
    mean, sd = base.summarize(groups[(row['dataset'], row['method'])], row['metric'])
    stage = int(row['stage']) - 1
    np.testing.assert_allclose(float(row['mean']), mean[stage], atol=1e-10, rtol=0)
    if row['std']:
        np.testing.assert_allclose(float(row['std']), sd[stage], atol=1e-10, rtol=0)
    csv_checked += 1

ns = {'s': 'http://www.w3.org/2000/svg'}
tree = ET.parse(output / f'{stem}.svg')
axes = [g for g in tree.findall('.//s:g', ns)
        if g.get('id', '').startswith('axes_')]
assert len(axes) == len(order)
number = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
svg_checks = []
for ax, dataset in zip(axes, order):
    patch = next(g for g in ax if g.get('id', '').startswith('patch_'))
    bounds = np.array([float(v) for v in number.findall(patch[0].get('d'))]).reshape(-1, 2)
    xleft, ytop = bounds.min(axis=0)
    xright, ybottom = bounds.max(axis=0)
    ymin, ymax = display['y_limits'][dataset]
    steps = base.DATASETS[dataset][1]
    for method, color in [('InfLoRA', '#0072b2'), ('GAM-EWC+Merge', '#cc3377')]:
        candidates = [(g, p) for g in ax if g.get('id', '').startswith('line2d_')
                      for p in g if p.tag.endswith('path') and color in p.get('style', '')]
        assert len(candidates) == 1
        group, path = candidates[0]
        assert re.sub(number, '', path.get('d')).strip().split() == ['M'] + ['L'] * (steps - 1)
        coords = np.array([float(v) for v in number.findall(path.get('d'))]).reshape(-1, 2)
        recovered_x = 0.8 + (coords[:, 0] - xleft) / (xright - xleft) * (steps - 0.6)
        recovered_y = ymax - (coords[:, 1] - ytop) / (ybottom - ytop) * (ymax - ymin)
        expected = base.summarize(groups[(dataset, method)], 'aa')[0]
        np.testing.assert_allclose(recovered_x, np.arange(1, steps + 1), atol=1e-6, rtol=0)
        np.testing.assert_allclose(recovered_y, expected, atol=1e-6, rtol=0)
        assert len(group.findall('.//s:use', ns)) == steps
        svg_checks.append({'dataset': dataset, 'method': method, 'points': steps,
                           'max_abs_y_error_pp': float(np.max(np.abs(recovered_y - expected))),
                           'only_observed_integer_task_points': True})

deltas = []
for dataset in order:
    inf, fs = [next(r for r in summary if r['dataset'] == dataset and r['method'] == m)
               for m in methods]
    row = {'dataset': dataset}
    for key in metric_keys:
        row[key + '_delta_pp'] = fs[key] - inf[key]
        row[key + '_relative_percent'] = 100 * (fs[key] - inf[key]) / inf[key]
    row['task_macro_aa_curve_delta_pp'] = (np.array(fs['aa_curve']) - inf['aa_curve']).tolist()
    old_inf, old_fs = inf['historical_table_snapshot'], fs['historical_table_snapshot']
    row['historical_table_faa_delta_pp'] = old_fs['accuracy_mean'] - old_inf['accuracy_mean']
    row['historical_table_aaa_delta_pp'] = old_fs['aaa_mean'] - old_inf['aaa_mean']
    deltas.append(row)

selected = [r for r in records if r['method'] == 'InfLoRA' or
            (r['method'] == 'GAM-EWC+Merge' and r['dataset'] in ['imagenetr', 'imageneta'])]
per_seed = []
for r in selected:
    row = {'dataset': r['dataset'], 'method': r['method'], 'seed': r['seed'],
           'log': r['log'], 'sha256': r['sha256'], 'task_row_lines': r['task_row_lines'],
           'config_from_manifest': r['logged_config'],
           'task_macro_aa_curve': r['aa'], 'sample_weighted_curve': r['sample_accuracy']}
    for label, key in metric_keys.items():
        row[label] = float(r[key][-1])
    per_seed.append(row)

report = {'scope': 'Current figure value/provenance check; not a re-evaluation of model checkpoints.',
          'figure': str(output / f'{stem}.pdf'),
          'definitions': {'task_macro_faa': 'AA_T = mean_j a[T,j]',
                          'task_macro_aaa': 'mean_t mean_{j<=t} a[t,j]',
                          'sample_weighted_faa': 'last classifier top1 across test samples',
                          'sample_weighted_aaa': 'mean of per-stage classifier global top1'},
          'summary': summary, 'fs_minus_inflora': deltas, 'per_seed': per_seed,
          'csv_rows_checked': csv_checked, 'svg_point_checks': svg_checks,
          'source_log_hashes_verified_by_loader': len(records),
          'artifact_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [PLOT/'sources.json', PLOT/'plot_lora_aa.py',
                                       PLOT/'restyle_aa_main.py', output/f'{stem}.svg',
                                       output/f'{stem}.pdf']}}
(HERE / 'value_audit.json').write_text(json.dumps(report, indent=2) + '\n')
with (HERE / 'per_seed_metrics.csv').open('w') as f:
    fields = ['dataset', 'method', 'seed', *metric_keys, 'log']
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(per_seed)
print(json.dumps({'csv_rows_checked': csv_checked, 'svg_checks': svg_checks,
                  'fs_minus_inflora': deltas}, indent=2))
