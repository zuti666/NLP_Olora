"""Compute seed variance only for audited replicates of the selected recipe.

The current figure/table remains a collection of single-run results.
Additional seed aggregates are reported separately and never overwrite it.
"""
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CURVES = WORKSPACE / 'figures/finegrained_taskwise_aa_20260920/data/curves.json'
OLD_AUDIT = Path('/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/ewc_cars_flowers_wiki_20260918')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    assert len(values) >= 2
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values), 'sample_variance': statistics.variance(values)}


def main():
    current = read(CURVES)
    assert len(current) == 35 and all(r['n_runs'] == 1 for r in current)
    selected = {(r['dataset'], r['method']): r for r in current}
    per_run = {r['source']: r for r in read(OLD_AUDIT / 'per_run.json')}
    groups = [g for g in read(OLD_AUDIT / 'seed_groups.json')
              if g['family'] == 'alpha0_both_rng_preserved'
              and (g['dataset'], g['lambda']) in [('cars196', 80000), ('flowers', 10)]]
    assert len(groups) == 2

    # Recheck the five current EWC recipe directories for additional result files.
    coverage = []
    for r in current:
        if r['method'] != 'EWCLoRA':
            continue
        source = Path(r['original_source'])
        aliases = {r['dataset'], 'cars' if r['dataset'] == 'cars196' else r['dataset']}
        directory = next(p for p in source.parents if p.name in aliases)
        result = subprocess.run(['rg', '--files', str(directory), '-g', '*_cl_metrics.json'],
                                capture_output=True, text=True, check=True)
        sources = sorted(result.stdout.splitlines())
        expected = next((sorted(g['sources']) for g in groups if g['dataset'] == r['dataset']),
                        [r['original_source']])
        assert sources == expected, f'Additional EWC candidates need inspection: {directory}'
        coverage.append({'dataset': r['dataset'], 'searched_directory': str(directory),
                         'metric_files': sources, 'complete_selected_recipe_runs': len(sources)})

    raw_rows, stage_rows, summaries, source_records = [], [], [], []
    ignore = {'seed', 'prefix', 'device', 'config', 'filepath', 'feature_flat_save_path', 'CNN', 'NME'}
    for group in groups:
        assert group['shared_logged_parameter_differences'] == {}
        old_rows = [per_run[s] for s in group['sources']]
        keys = set.union(*(set(r['params']) for r in old_rows)) - ignore
        differences = {k: [r['params'].get(k) for r in old_rows] for k in keys
                       if len({json.dumps(r['params'].get(k), sort_keys=True) for r in old_rows}) > 1}
        assert not differences, differences
        chosen = selected[(group['dataset'], 'EWCLoRA')]
        assert chosen['original_source'] in group['sources']
        runs = []
        for old in old_rows:
            assert old['status'] == 'verified' and not old['training_nonfinite']
            assert sha(old['source']) == old['sha256']
            matrix = read(old['source'])['cnn']['matrices']['final']
            lower = [[float(v) for v in row[:i+1]] for i, row in enumerate(matrix)]
            assert len(lower) == 10 and all(len(row) == i+1 for i, row in enumerate(lower))
            assert all(math.isfinite(v) and 0 <= v <= 100 for row in lower for v in row)
            aa = [statistics.mean(row) for row in lower]
            faa, aaa = aa[-1], statistics.mean(aa)
            assert abs(faa-old['faa']) < 1e-10 and abs(aaa-old['aaa_stage']) < 1e-10
            if old['source'] == chosen['original_source']:
                assert aa == chosen['aa']
            item = {'dataset': group['dataset'], 'method': 'EWCLoRA', 'ewc_lambda': group['lambda'],
                    'seed': int(old['seed']), 'faa': faa, 'aaa_stage': aaa,
                    'selected_in_current_table': old['source'] == chosen['original_source']}
            raw_rows.append(item)
            runs.append({**item, 'aa': aa})
            source_records.append({**item, 'metrics_path': old['source'], 'metrics_sha256': old['sha256'],
                                   'log_path': old['log'], 'log_sha256': sha(old['log']),
                                   'recorded_training_parameter_differences': differences})
        runs.sort(key=lambda r: r['seed'])
        faa = stats([r['faa'] for r in runs])
        aaa = stats([r['aaa_stage'] for r in runs])
        assert abs(faa['mean']-group['faa_mean']) < 1e-10
        assert abs(faa['sample_sd']-group['faa_sample_sd']) < 1e-10
        assert abs(aaa['mean']-group['aaa_mean']) < 1e-10
        assert abs(aaa['sample_sd']-group['aaa_sample_sd']) < 1e-10
        summary = {'dataset': group['dataset'], 'method': 'EWCLoRA', 'ewc_lambda': group['lambda'],
                   'seeds': ','.join(str(r['seed']) for r in runs), 'n': len(runs),
                   'faa_mean': faa['mean'], 'faa_sample_sd': faa['sample_sd'],
                   'faa_sample_variance_pp2': faa['sample_variance'],
                   'aaa_mean': aaa['mean'], 'aaa_sample_sd': aaa['sample_sd'],
                   'aaa_sample_variance_pp2': aaa['sample_variance']}
        summaries.append(summary)
        for i in range(10):
            s = stats([r['aa'][i] for r in runs])
            stage_rows.append({'dataset': group['dataset'], 'method': 'EWCLoRA', 'task': i+1,
                               'seed1_aa': runs[0]['aa'][i], 'seed2_aa': runs[1]['aa'][i],
                               'n': s['n'], 'aa_mean': s['mean'], 'aa_sample_sd': s['sample_sd'],
                               'aa_sample_variance_pp2': s['sample_variance']})

    current_rows = [{'dataset': r['dataset'], 'method': r['method'], 'displayed_seed': r['seed'],
                     'n_used_in_current_table': 1, 'faa': r['faa'], 'aaa_stage': r['aaa_stage'],
                     'faa_seed_variance': None, 'aaa_seed_variance': None,
                     'variance_status': 'undefined for n=1; not zero'} for r in current]
    save_json(HERE / 'current_selection.json', current_rows)
    save_csv(HERE / 'current_selection.csv', current_rows)
    save_csv(HERE / 'replicate_raw_metrics.csv', raw_rows)
    save_csv(HERE / 'replicate_summary.csv', summaries)
    save_csv(HERE / 'replicate_stage_variance.csv', stage_rows)
    save_json(HERE / 'variance_audit.json', {
        'date': '2026-09-22', 'current_table_n_per_cell': 1,
        'current_curve_file': str(CURVES), 'current_curve_sha256': sha(CURVES),
        'variance_estimator': 'sample variance, ddof=1: sum((x-mean)^2)/(n-1)',
        'variance_unit': 'percentage points squared (pp^2)',
        'sd_unit': 'percentage points (pp)',
        'metric_definition': 'For each seed: AA_t is lower-triangle row mean; FAA=AA_T; AAA=mean_t AA_t. Then aggregate across seeds.',
        'current_figure_band': 'Fixed +/-0.2 pp illustrative shading, not empirical SD or variance.',
        'summary': summaries, 'raw_metrics': raw_rows, 'sources': source_records,
        'ewc_directory_coverage': coverage,
        'scope': 'Selected recipes and nearby source directories; no claim of exhaustive whole-disk search.',
        'historical_limit': 'Matching recorded configurations and checkpoint IDs; historical code/data/weight bytes are not all immutably bound.',
    })
    print(json.dumps({'raw': raw_rows, 'summary': summaries}, indent=2))


if __name__ == '__main__':
    main()
