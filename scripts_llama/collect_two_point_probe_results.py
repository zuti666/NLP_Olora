#!/usr/bin/env python3
"""Collect only accepted v2 two-point-probe chains; no legacy baselines."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path

SCHEMA = 'TwoPointProbeSixChainQueue_v1'
_s = importlib.util.spec_from_file_location('_probe_shared_metrics', Path(__file__).with_name('collect_g2_orders_results.py'))
metrics = importlib.util.module_from_spec(_s)
_s.loader.exec_module(metrics)
NAMES = {'sgd': 'IncLoRA (clean gradient)', 'sam': 'SAM direction + AdamW',
         'reverse': 'Reflected descent probe + AdamW'}
DIRECTIONS = {'sgd': 'g0', 'sam': 'g1_ascent', 'reverse': '2g0-g1_descent'}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> dict:
    if metrics.sha256(root / 'manifest.json') != (root / 'manifest.sha256').read_text().strip():
        raise ValueError('Prepared probe manifest changed.')
    manifest = json.loads((root / 'manifest.json').read_text())
    state = json.loads((root / 'runtime/queue_state.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('decoder_pipeline_version') != 'decoder_token_ids_v2':
        raise ValueError('Result collection requires the registered v2 probe campaign.')
    source = Path(manifest['source'])
    paths = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if paths != set(manifest['source_sha256']):
        raise ValueError('Frozen probe source population changed.')
    for relative, expected in manifest['source_sha256'].items():
        if metrics.sha256(source / relative) != expected:
            raise ValueError(f'Frozen probe source changed: {relative}')
    completed = {r['key']: r for r in state['completed_stages']}
    order_rows, stage_rows = [], []
    for job in manifest['jobs']:
        tasks = job['config']['tasks']
        gold = {}
        for task in tasks:
            path = Path(job['config']['data_dir']) / metrics.TASK_RELATIVE[task] / 'test.json'
            metrics.verify_pinned_input(manifest, path)
            gold[task] = json.loads(path.read_text())
        by_stage = {}
        ce_total = replay_total = update_total = 0
        directions, changes, update_ratios = [], [], []
        for spec in job['plan']['stages']:
            if spec['smoke']:
                continue
            key = f'{job["id"]}:s{spec["seed"]}:formal:{spec["stage"]}'
            if key not in completed:
                continue
            accepted = completed[key]
            receipt = accepted['receipt']
            identity = {'status': 'pass', 'method': 'TwoPointProbe-Llama',
                'mode': job['mode'], 'model_size': job['model_size'],
                'order': job['order'], 'seed': job['seed'], 'smoke': False,
                'stage': spec['stage'], 'task': spec['task'],
                'global_step': spec['expected_updates']}
            if (accepted.get('job') != job['id'] or accepted.get('smoke') is not False
                    or accepted.get('stage') != spec['stage']
                    or any(receipt.get(k) != v for k, v in identity.items())):
                raise ValueError('Formal receipt identity or budget differs from the registered chain.')
            output = Path(spec['output_dir'])
            audit = output / 'two_point_probe_training_audit.json'
            if metrics.sha256(audit) != receipt['audit_sha256']:
                raise ValueError('Accepted runtime audit changed.')
            adapter = next((output / 'adapter' / name for name in ['adapter_model.bin', 'adapter_model.safetensors']
                            if (output / 'adapter' / name).is_file()), None)
            if adapter is None or metrics.sha256(adapter) != receipt['adapter_sha256']:
                raise ValueError('Accepted adapter changed.')
            if metrics.sha256(output / 'adapter/adapter_config.json') != receipt['adapter_config_sha256']:
                raise ValueError('Accepted adapter configuration changed.')
            predictions = output / 'predict_eval_predictions.jsonl'
            if metrics.sha256(predictions) != receipt['prediction_sha256']:
                raise ValueError('Accepted predictions changed.')
            scored = metrics.prediction_metrics(predictions, spec['seen_tasks'], gold=gold)
            by_stage[spec['stage']] = scored
            trace = output / 'probe_step_trace.jsonl'
            if metrics.sha256(trace) != receipt['trace_sha256']:
                raise ValueError('Accepted gradient trace changed.')
            rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            if len(rows) != spec['expected_updates']:
                raise ValueError('Trace update count differs.')
            ce = sum(row['probe_ce_gradient_phases'] for row in rows)
            replay = sum(row['probe_replay_closure_calls'] for row in rows)
            commits = sum(row['probe_base_optimizer_steps'] for row in rows)
            if commits != spec['expected_updates'] or ce != commits * (1 if job['mode'] == 'sgd' else 2):
                raise ValueError('Observed method cost differs from registered probe budget.')
            ce_total += ce
            replay_total += replay
            update_total += commits
            runtimes = {}
            for filename, field in [('train_results.json', 'train_runtime'),
                                    ('predict_results.json', 'predict_runtime')]:
                value = json.loads((output / filename).read_text()).get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid actual timing: {field}')
                runtimes[field + '_seconds'] = value
            for row in rows:
                for field, values in [('probe_cos_g0_submitted', directions),
                        ('probe_submitted_minus_g0_relative_norm', changes),
                        ('probe_actual_update_over_probe_radius', update_ratios)]:
                    value = row.get(field)
                    if value is not None:
                        values.append(value)
            stage_rows.append({'model': job['model_size'], 'mode': job['mode'],
                'method': NAMES[job['mode']], 'order': job['order'], 'seed': job['seed'],
                'stage': spec['stage'], 'task': spec['task'],
                'seen_task_accuracy': scored['seen_task_accuracy'],
                'ce_phases': ce, 'replay_phases': replay, 'optimizer_commits': commits,
                'prediction_sha256': scored['prediction_sha256'], **runtimes,
                **{f'{task}_accuracy': value for task, value in scored['accuracy'].items()}})
        row = {'model': job['model_size'], 'method': NAMES[job['mode']],
            'mode': job['mode'], 'gradient': DIRECTIONS[job['mode']], 'order': job['order'],
            'seed': job['seed'], 'decoder_pipeline_version': 'decoder_token_ids_v2',
            'status': 'completed' if len(by_stage) == 4 else 'pending',
            'completed_stages': len(by_stage), 'FAA': '', 'AAA': '', 'BWT': '',
            'delta_FAA_vs_clean': '', 'delta_FAA_vs_SAM': '',
            'learning_accuracy': '', 'delta_AAA_vs_clean': '', 'delta_AAA_vs_SAM': '',
            'delta_BWT_vs_clean': '', 'delta_BWT_vs_SAM': '',
            'delta_learning_accuracy_vs_clean': '', 'delta_learning_accuracy_vs_SAM': '',
            'optimizer_commits': update_total, 'ce_gradient_phases': ce_total,
            'replay_closure_calls': replay_total,
            'train_runtime_seconds': sum(r['train_runtime_seconds'] for r in stage_rows
                if r['model'] == job['model_size'] and r['mode'] == job['mode']
                and r['order'] == job['order'] and r['seed'] == job['seed']),
            'predict_runtime_seconds': sum(r['predict_runtime_seconds'] for r in stage_rows
                if r['model'] == job['model_size'] and r['mode'] == job['mode']
                and r['order'] == job['order'] and r['seed'] == job['seed']),
            'median_cos_g0_submitted': statistics.median(directions) if directions else '',
            'median_relative_gradient_change': statistics.median(changes) if changes else '',
            'median_update_over_probe_radius': statistics.median(update_ratios) if update_ratios else ''}
        if len(by_stage) == 4:
            row.update(metrics.continual_metrics(tasks, by_stage))
            row['learning_accuracy'] = sum(by_stage[i + 1]['accuracy'][task]
                for i, task in enumerate(tasks)) / len(tasks)
        order_rows.append(row)
    for row in order_rows:
        if row['status'] != 'completed':
            continue
        for mode, field in [('sgd', 'delta_FAA_vs_clean'), ('sam', 'delta_FAA_vs_SAM')]:
            ref = next((r for r in order_rows if r['model'] == row['model'] and r['order'] == row['order']
                        and r['seed'] == row['seed'] and r['mode'] == mode and r['status'] == 'completed'), None)
            if ref is not None:
                row[field] = row['FAA'] - ref['FAA']
                suffix = 'clean' if mode == 'sgd' else 'SAM'
                for metric in ['AAA', 'BWT', 'learning_accuracy']:
                    row[f'delta_{metric}_vs_{suffix}'] = row[metric] - ref[metric]
    results = root / 'results'
    results.mkdir(exist_ok=True)
    _write_csv(results / 'orders.csv', order_rows)
    _write_csv(results / 'stages.csv', stage_rows)
    lines = ['# Corrected two-point NLP probe results', '',
        f'Queue status: {state.get("status")}. All results use decoder_token_ids_v2.', '',
        '| Model | Method | Gradient | Stages | FAA (%) | AAA (%) | BWT (pp) | ΔFAA vs clean | ΔFAA vs SAM |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def value(v):
        return f'{v:.6f}' if isinstance(v, (int, float)) else str(v)
    for row in order_rows:
        lines.append('| ' + ' | '.join(value(row[key]) for key in
            ['model', 'method', 'gradient', 'completed_stages', 'FAA', 'AAA', 'BWT',
             'delta_FAA_vs_clean', 'delta_FAA_vs_SAM']) + ' |')
    lines += ['', 'Only complete four-stage chains receive FAA/AAA/BWT. Smoke scores and',
        'partial chains are excluded. Comparators are fresh, same-model/Order/seed,',
        'corrected-pipeline controls from this campaign. Seed0 is exploratory;',
        'three-seed confirmation remains separate. Actual optimizer commits are',
        'matched; clean-gradient control uses one CE phase, probe methods two.']
    (results / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    summary = {'queue_status': state.get('status'), 'completed_orders': sum(r['status'] == 'completed' for r in order_rows),
        'total_orders': len(order_rows), 'order_results': order_rows, 'stage_results': stage_rows,
        'old_baseline_used': False}
    (results / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-root', type=Path, required=True)
    args = p.parse_args()
    result = collect(args.queue_root.resolve())
    print(json.dumps({'completed_orders': result['completed_orders'], 'total_orders': result['total_orders']}))
