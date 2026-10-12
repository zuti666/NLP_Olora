#!/usr/bin/env python3
"""Collect completed paired runs, verify pairing, export CSV/LaTeX statistics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

FIELDS = ['a11', 'a21', 'a22', 'bwt', 'old_train_loss_delta', 'old_test_loss_delta',
          'r_eff', 'old_trace_task0', 'old_trace_task1', 'new_trace_task1']
COMMON_CONFIG = ['dataset', 'class_shuffle', 'init_cls', 'increment', 'lora_rank',
                 'comparison_base_optimizer', 'init_lr', 'lrate', 'init_epoch', 'epochs',
                 'batch_size', 'momentum', 'weight_decay', 'ewc_lambda', 'ewc_gamma',
                 'ewc_eta', 'fisher_mode', 'ewc_max_batches', 'ewc_fisher_source',
                 'ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher',
                 'ewc_fisher_damping_beta', 'frozen_backbone_checkpoint']


def read_json(path):
    return json.loads(path.read_text())


def rng_signature(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'rng_receipts.jsonl').read_text().splitlines()]
    keys = {
        'head_initialized': ['task', 'old_classes', 'total_classes', 'new_head_sha256'],
        'loader_iteration': ['task', 'loader_serial', 'iteration', 'generator_seed', 'samples',
                             'indices_sha256', 'first_batch_pixels_targets_sha256'],
    }
    out = []
    for row in rows:
        if row['event'] in keys:
            out.append({'event': row['event'], **{k: row[k] for k in keys[row['event']]}})
    if {r['task'] for r in out if r['event'] == 'head_initialized'} != {0, 1}:
        raise ValueError(f'Missing task head initialization receipts: {run_dir}')
    if not any(r['event'] == 'loader_iteration' and r['task'] == 1 for r in out):
        raise ValueError(f'Missing Task1 training-loader receipts: {run_dir}')
    return out


def collect(root, seeds):
    records, pair_checks = [], []
    for seed in seeds:
        pair = []
        for method in ('ewc', 'gam_ewc'):
            run = root / method / f'seed{seed}'
            marker = read_json(run / 'COMPLETE.json')
            if marker.get('completed_tasks') != 2:
                raise ValueError(f'Training is not complete: {run}')
            path = run / 'evaluation/summary.json'
            evaluation = read_json(path)
            receipt = read_json(path.parent / 'COMPLETE.json')
            if evaluation['status'] != 'complete' or receipt['summary_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'Evaluation is incomplete or changed: {path}')
            if evaluation['method'] != method or evaluation['seed'] != seed:
                raise ValueError(f'Method/seed mismatch: {path}')
            record = {'method': method, 'seed': seed, **{k: evaluation[k] for k in FIELDS},
                      'source': str(path)}
            records.append(record)
            pair.append((run, evaluation))
        left, right = (p[1] for p in pair)
        for name in COMMON_CONFIG:
            if left['inputs']['config'][name] != right['inputs']['config'][name]:
                raise ValueError(f'Unmatched training config {name}, seed{seed}')
        if left['inputs']['code_sha256'] != right['inputs']['code_sha256']:
            raise ValueError('Different empirical-Fisher evaluation implementations')
        for task, scope in ((0, 'old_local20'), (1, 'old_local20'), (1, 'new_local20')):
            a, b = (p['endpoints'][task]['fisher'][scope] for p in (left, right))
            if a['input_tensors_sha256'] != b['input_tensors_sha256']:
                raise ValueError(f'Fisher samples differ: seed{seed} task{task} {scope}')
        if rng_signature(pair[0][0]) != rng_signature(pair[1][0]):
            raise ValueError(f'Training head/data RNG receipts differ: seed{seed}')
        pair_checks.append({'seed': seed, 'same_config': True, 'same_fisher_inputs': True,
                            'same_heads_data_order_and_first_augmented_batches': True})
    aggregate = []
    for method in ('ewc', 'gam_ewc'):
        rows = [row for row in records if row['method'] == method]
        for field in FIELDS:
            values = [row[field] for row in rows]
            aggregate.append({'method': method, 'metric': field, 'n': len(values),
                              'mean': statistics.mean(values),
                              'sample_sd': statistics.stdev(values) if len(values) > 1 else None})
    differences = {}
    for field in FIELDS:
        values = [next(r[field] for r in records if r['method'] == 'gam_ewc' and r['seed'] == seed) -
                  next(r[field] for r in records if r['method'] == 'ewc' and r['seed'] == seed)
                  for seed in seeds]
        differences[field] = {'values_GAM_minus_EWC': values, 'mean': statistics.mean(values),
                              'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
    return records, aggregate, pair_checks, differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seeds', default='3,4,5')
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct seeds required.')
    rows, aggregate, checks, differences = collect(args.output_root.resolve(), seeds)
    output = args.output_root.resolve() / 'comparison'
    output.mkdir(exist_ok=True)
    for name, values in [('per_seed.csv', rows), ('mean_sd.csv', aggregate)]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'seeds': seeds, 'pair_checks': checks, 'per_seed': rows,
               'statistics': aggregate, 'paired_differences': differences,
               'statistical_scope': 'single-seed case study' if len(seeds) == 1 else 'training-seed statistics'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def value(method, metric):
        cell = next(r for r in aggregate if r['method'] == method and r['metric'] == metric)
        mean, sd = cell['mean'], cell['sample_sd']
        return f'${mean:.3f}$' if sd is None else f'${mean:.3f}\\pm{sd:.3f}$'
    columns = ['a11', 'a21', 'a22', 'bwt', 'old_test_loss_delta', 'r_eff',
               'old_trace_task0', 'old_trace_task1', 'new_trace_task1']
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
             r'\caption{Native GAM-EWC versus EWC on ImageNet-R (20+20 classes). '
             r'Accuracies use the seen-class head; old-task CE damage uses the fixed old 20-class head. '
             r'$\tau_{k|t}$ is the unnormalized per-example empirical Fisher trace on task $k$ data '
             r'at endpoint $t$, using the local 20-class head of task $k$, '
             r'in effective Q/V weight coordinates, excluding classifier parameters. '
             r'$r_{QV}$ is the net difference of effective Q/V weights between endpoints. '
             + f'Means over {len(seeds)} training seed(s); sample standard deviations shown when available.}}',
             r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
             r'Method & $a_{11}$ & $a_{21}$ & $a_{22}$ & BWT & $\Delta L_{old}$ & $r_{QV}$ & $\tau_{1|1}$ & $\tau_{1|2}$ & $\tau_{2|2}$ \\',
             r'\midrule']
    for method, label in [('ewc', 'EWC'), ('gam_ewc', 'GAM-EWC')]:
        lines.append(label + ' & ' + ' & '.join(value(method, metric) for metric in columns) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'comparison.tex').write_text('\n'.join(lines) + '\n')
    print(f'Validated paired results: {output}')


if __name__ == '__main__':
    main()
