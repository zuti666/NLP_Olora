"""Freeze the current table's runs and derive task-macro stage accuracy."""
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXPERIMENT = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2')
DATASETS = ['aircraft', 'cars196', 'cub200', 'flowers', 'het5']
METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']
TASKS = {'aircraft': 10, 'cars196': 10, 'cub200': 10, 'flowers': 10, 'het5': 5}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def slug(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def main():
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    selections_path = WORKSPACE / 'refine-logs/cllora_comparison_20260918/results.json'
    ewc_path = WORKSPACE / 'refine-logs/finegrained_complete_20260919/ewc_best_selection.json'
    table_path = WORKSPACE / 'refine-logs/finegrained_complete_20260919/finegrained_complete.tex'
    records = {(r['dataset'], r['method']): r for r in read(selections_path)
               if r['dataset'] in DATASETS and r['method'] in METHODS}
    for r in read(ewc_path)['selected_runs']:
        if r['dataset'] in DATASETS:
            records[(r['dataset'], 'EWCLoRA')] = {
                **r, 'method': 'EWCLoRA', 'sha256': r['metrics_sha256'],
                'selection': 'Highest audited eligible archived FAA; paired AAA; retrospective test-score selection.',
            }
    receipt_path = EXPERIMENT / 'receipts/gam_ewc_het5_s0.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    assert digest(receipt['config']) == receipt['config_sha256']
    assert digest(receipt['log']) == receipt['log_sha256']
    records[('het5', 'GAM-EWCLoRA')] = {
        'dataset': 'het5', 'method': 'GAM-EWCLoRA', 'source': receipt['metric_path'],
        'sha256': receipt['metric_sha256'], 'log': receipt['log'], 'seed': receipt['seed'],
        'params': read(receipt['config']), 'config_path': receipt['config'],
        'faa': receipt['metrics']['faa'], 'aaa_stage': receipt['metrics']['aaa_stage'],
        'receipt': str(receipt_path),
    }
    assert len(records) == len(DATASETS) * len(METHODS) == 35

    # Preserve the displayed table's original rounding conventions; raw curves are never rounded.
    table_order = ['aircraft', 'cars196', 'cub200', 'flowers', 'pets', 'het5']
    table_values = {}
    for line in table_path.read_text().splitlines():
        method = line.split(' & ')[0]
        if method in METHODS:
            cells = line.split(' & ')[1:]
            assert len(cells) == 12
            vals = [float(re.search(r'\d+\.\d+', c).group()) for c in cells]
            table_values[method] = dict(zip(table_order, [vals[i:i+2] for i in range(0, 12, 2)]))

    curves, points, final_rows, manifest = [], [], [], []
    for dataset in DATASETS:
        for method in METHODS:
            r = records[(dataset, method)]
            source = Path(r['source'])
            assert digest(source) == r['sha256'], source
            raw = read(source)
            matrix = raw['cnn']['matrices']['final']
            total = TASKS[dataset]
            assert len(matrix) == total
            lower = [[float(x) for x in row[:i+1]] for i, row in enumerate(matrix)]
            assert all(len(row) == i+1 for i, row in enumerate(lower))
            assert all(math.isfinite(x) and 0 <= x <= 100 for row in lower for x in row)
            aa = [statistics.mean(row) for row in lower]
            faa, aaa = aa[-1], statistics.mean(aa)
            assert abs(faa-r['faa']) < 1e-8 and abs(aaa-r['aaa_stage']) < 1e-8
            shown_faa, shown_aaa = table_values[method][dataset]
            assert abs(faa-shown_faa) <= .011 and abs(aaa-shown_aaa) <= .011
            params = r['params']
            seed = r.get('seed', params.get('seed'))
            if isinstance(seed, list):
                seed = seed[0]
            frozen_path = data / f'{dataset}_{slug(method)}.json'
            item = {'dataset': dataset, 'method': method, 'seed': int(seed), 'n_runs': 1,
                    'tasks': total, 'task_index': list(range(1, total+1)),
                    'matrix_lower_triangle': lower, 'aa': aa, 'faa': faa, 'aaa_stage': aaa,
                    'original_source': str(source), 'original_sha256': r['sha256'],
                    'epochs_per_task': int(params['epochs']),
                    'learning_rate': float(params['lrate']),
                    'ewc_lambda': float(params['ewc_lambda']) if 'ewc_lambda' in params else None,
                    'ewc_implementation': r.get('family') if method == 'EWCLoRA' else None}
            write_json(frozen_path, item)
            curves.append(item)
            for task, value in enumerate(aa, 1):
                points.append({'dataset': dataset, 'method': method, 'seed': int(seed),
                               'learned_tasks': task, 'aa_percent': value})
            final_rows.append({'dataset': dataset, 'method': method, 'seed': int(seed),
                               'tasks': total, 'faa_from_last_point': faa,
                               'aaa_from_curve_mean': aaa, 'displayed_faa': shown_faa,
                               'displayed_aaa': shown_aaa, 'table_agrees_within_0_011_pp': True})
            manifest.append({'dataset': dataset, 'method': method,
                             'source': str(source), 'source_sha256': r['sha256'],
                             'log': r.get('log'),
                             'log_sha256': digest(r['log']) if r.get('log') else None,
                             'config': r.get('config_path'),
                             'frozen_matrix': str(frozen_path.relative_to(HERE)),
                             'frozen_matrix_sha256': digest(frozen_path)})

    write_json(data / 'curves.json', curves)
    write_json(HERE / 'source_manifest.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_order': DATASETS, 'method_order': METHODS,
        'metric': 'AA_t = (1/t) sum_{j=1}^t a[t,j], including the current task; task-macro row mean.',
        'last_point': 'FAA', 'mean_of_all_points': 'AAA_stage',
        'replication': 'One selected run per method/dataset; no SD/CI or illustrative bands.',
        'selection_files': {str(p): digest(p) for p in [selections_path, ewc_path, table_path, receipt_path]},
        'runs': manifest,
    })
    for filename, values in [('stage_accuracy.csv', points), ('final_summary.csv', final_rows)]:
        with (data / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(f'Frozen {len(curves)} verified runs and {len(points)} task-macro stage points.')
    for dataset in DATASETS:
        print(dataset, {r['method']: [round(v, 2) for v in r['aa']] for r in curves if r['dataset'] == dataset})


if __name__ == '__main__':
    main()
