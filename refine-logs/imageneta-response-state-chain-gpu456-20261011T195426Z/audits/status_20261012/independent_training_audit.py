#!/usr/bin/env python3
"""Read-only independent acceptance and raw-source paired result audit."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'trainer'))
from validate_chain import validate_output, validate_pair


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, label='root'):
    count = 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError('Nonfinite observed numeric value: ' + label)
        count += 1
    elif isinstance(value, dict):
        for key, item in value.items():
            count += finite(item, label + '.' + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            count += finite(item, label + '[' + str(i) + ']')
    return count


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def main():
    suite = read(ROOT / 'suite.json')
    state = read(ROOT / 'runtime/queue_state.json')
    pins = read(ROOT / 'RUN_PINS.json')['files']
    for path, sha in pins.items():
        assert digest(ROOT / path) == sha, ('Frozen execution file changed', path)
    manifest = read(ROOT / 'source_manifest.json')
    for path, sha in manifest['files'].items():
        assert digest(Path(suite['source_root']) / path) == sha, ('Native source changed', path)
    jobs = [j for j in suite['jobs'] if j.get('formal')]
    assert len(jobs) == 6
    assert {(j['seed'], j['chain_branch']) for j in jobs} == {(s, b) for s in [0, 1, 2] for b in ['S', 'R']}
    validations = []
    raw = []
    outputs = {}
    total_steps = total_reads = 0
    all_numeric = 0
    for job in jobs:
        out = Path(job['output_dir'])
        config = read(job['config'])
        assert digest(job['config']) == job['config_sha256']
        expected = {'dataset': 'imageneta', 'class_shuffle': False, 'init_cls': 20, 'increment': 20,
                    'lora_rank': 10, 'batch_size': 128, 'epochs': 50, 'init_epoch': 50,
                    'init_lr': .04, 'lrate': .02, 'momentum': 0, 'weight_decay': 0,
                    'ewc_lambda': 0., 'ewc_skip_task_fisher': True, 'in_task_weight_merge': False,
                    'task_endpoint_weight_merge': False, 'approximate_fisher_in_task': False,
                    'history_weight': 0., 'chain_role': 'branch', 'chain_smoke': False,
                    'chain_switch_after_epoch': 25, 'chain_branch': job['chain_branch']}
        for key, value in expected.items():
            assert config[key] == value, (job['id'], key, config[key], value)
        assert config['seed'] == [job['seed']]
        assert state['jobs'][job['id']]['status'] == 'completed'
        checked = validate_output(job['config'], out)
        receipt = read(ROOT / 'runtime/jobs' / job['id'] / 'completed.json')
        exit_receipt = read(ROOT / 'runtime/jobs' / job['id'] / 'exit.json')
        summary = read(out / 'final_summary.json')
        assert receipt['summary_sha256'] == digest(out / 'final_summary.json')
        assert exit_receipt['returncode'] == receipt['exit']['returncode'] == 0
        assert receipt['full_formal_training'] is True
        assert receipt['updates'] == 400
        assert receipt['gradient_reads'] == (600 if job['chain_branch'] == 'S' else 800)
        assert summary['trainer_sha256'] == pins['trainer/response_chain_train.py']
        assert summary['algorithm_sha256'] == pins['trainer/chain_step.py']
        assert summary['training_tasks_this_job'] == 1 and summary['trained_task_ids'] == [1]
        parent = summary['parent_restore_receipt']
        assert digest(parent['parent_completion_receipt']) == parent['parent_completion_receipt_sha256']
        assert digest(Path(parent['parent_output_dir']) / 'final_summary.json') == parent['parent_summary_sha256']
        for name in ['final_summary.json', 'retention_curve.json', 'factor_state_curve.json',
                     'factor_update_decomposition.json']:
            all_numeric += finite(read(out / name), job['id'] + '/' + name)
        traces = [json.loads(line) for line in (out / 'step_trace.jsonl').read_text().splitlines() if line.strip()]
        all_numeric += finite(traces, job['id'] + '/trace')
        for trace in traces:
            for field in ['loss0', 'train_ce', 'post_update_ce', 'nominal_lr', 'actual_parameter_step_norm']:
                assert isinstance(trace[field], (float, int)) and math.isfinite(trace[field])
            assert trace['gradient_clipping_calls'] == trace['qr_calls'] == 0
            assert trace['natural_sgd_exact'] is True
        retention = read(out / 'retention_curve.json')['epochs']
        assert retention == summary['retention_curve']
        initial_old = retention[0]['old_test_allseen40']['accuracy']
        for epoch in [25, 50]:
            point = retention[epoch]
            assert point['epoch'] == epoch
            old, new = point['old_test_allseen40'], point['current_test_allseen40']
            assert 0 <= old['accuracy'] <= 100 and 0 <= new['accuracy'] <= 100
            faa = (old['accuracy'] + new['accuracy']) / 2
            assert math.isclose(faa, point['allseen40_task_macro_accuracy'], abs_tol=1e-12)
            assert math.isclose(initial_old - old['accuracy'], point['old_drop_since_start'], abs_tol=1e-12)
            raw.append({'run': job['id'], 'seed': job['seed'], 'branch': job['chain_branch'], 'epoch': epoch,
                        'old_accuracy': old['accuracy'], 'new_accuracy': new['accuracy'], 'FAA': faa,
                        'old_decline': initial_old - old['accuracy'], 'old_ce': old['ce'], 'new_ce': new['ce']})
        outputs[(job['seed'], job['chain_branch'])] = out
        total_steps += receipt['updates']
        total_reads += receipt['gradient_reads']
        validations.append({k: v for k, v in checked.items() if k != 'check_labels'} | {
            'run': job['id'], 'completed_at': receipt['completed_at'],
            'receipt_sha256': digest(ROOT / 'runtime/jobs' / job['id'] / 'completed.json'),
            'expected_updates': 400, 'expected_gradient_reads': receipt['gradient_reads']})
    pairs = [validate_pair(outputs[(seed, 'S')], outputs[(seed, 'R')]) for seed in [0, 1, 2]]
    aggregate = []
    differences = []
    paired = []
    metrics = ['old_accuracy', 'new_accuracy', 'FAA', 'old_decline', 'old_ce', 'new_ce']
    for epoch in [25, 50]:
        for branch in ['S', 'R']:
            group = [r for r in raw if r['epoch'] == epoch and r['branch'] == branch]
            aggregate.append({'epoch': epoch, 'branch': branch, **{m: stats([r[m] for r in group]) for m in metrics}})
        for seed in [0, 1, 2]:
            s = next(r for r in raw if r['seed'] == seed and r['epoch'] == epoch and r['branch'] == 'S')
            r = next(r for r in raw if r['seed'] == seed and r['epoch'] == epoch and r['branch'] == 'R')
            differences.append({'seed': seed, 'epoch': epoch, **{m: r[m] - s[m] for m in metrics}})
        ds = [r for r in differences if r['epoch'] == epoch]
        contrast = {'epoch': epoch, 'contrast': 'R-minus-S', 'paired_seeds': [0, 1, 2]}
        for m in metrics:
            values = [r[m] for r in ds]
            sign = -1 if m in ['old_decline', 'old_ce', 'new_ce'] else 1
            contrast[m] = stats(values) | {'wins': sum(sign * v > 1e-12 for v in values),
                                         'ties': sum(abs(v) <= 1e-12 for v in values),
                                         'losses': sum(sign * v < -1e-12 for v in values),
                                         'per_seed_deltas': {str(r['seed']): r[m] for r in ds}}
        paired.append(contrast)
    collected = list(csv.DictReader((ROOT / 'FORMAL_RESULTS.csv').open()))
    assert len(collected) == len(raw)
    for row in raw:
        match = [r for r in collected if r['run'] == row['run'] and int(r['epoch']) == row['epoch']]
        assert len(match) == 1
        for m in metrics:
            assert math.isclose(float(match[0][m]), row[m], abs_tol=1e-12)
    counts = {}
    for job in state['jobs'].values():
        counts[job['status']] = counts.get(job['status'], 0) + 1
    report = {'audit_status': 'accepted_formal_training_only',
              'audited_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'root': str(ROOT), 'formal_validated': 6, 'formal_expected': 6,
              'formal_failed': 0, 'formal_missing': 0, 'independent_seed_count': 3,
              'native_source_files_verified': len(manifest['files']), 'execution_pins_verified': len(pins),
              'finite_observed_float_values_verified': all_numeric,
              'total_SGD_updates': total_steps, 'total_gradient_reads': total_reads,
              'run_validations': validations, 'paired_validation': pairs, 'raw_endpoints': raw,
              'mean_sample_sd': aggregate, 'paired_differences': differences, 'paired_stats': paired,
              'queue_status': state['status'], 'actual_job_status_counts': counts,
              'reported_queue_counts': state.get('counts'),
              'offline_status': {j['id']: state['jobs'][j['id']]['status'] for j in suite['jobs'] if j['id'].startswith('offline_')},
              'limitations': ['Training performance validity does not imply offline Hessian gate validity.',
                              'This is a two-task protocol with a common Task0 SGD parent; it is not a ten-task FAA result.',
                              'Independent units are three seeds, not epochs, minibatches or LoRA layers.',
                              'Old final accuracy gains are one win and two ties, so consistent forgetting reduction is unsupported.',
                              'Natural branch steps differ; no causal direction-only claim from these six trainings.']}
    folder = Path(__file__).resolve().parent
    (folder / 'INDEPENDENT_TRAINING_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (folder / 'INDEPENDENT_RAW_ENDPOINTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
    lines = ['# Independent training audit — 2026-10-12', '',
             'All 6/6 formal branches passed validation; no failed or missing formal training cells. Three paired independent seeds: 0, 1, 2.',
             f'Verified {len(pins)} frozen execution pins and {len(manifest["files"])} native source files; {all_numeric} observed float fields are finite.',
             f'Total: {total_steps} actual SGD commits and {total_reads} native gradient reads. Each seed has 400 paired augmented minibatches, identical initial expanded head, model, RNG and first base gradient.', '',
             '| Branch | Epoch | Old accuracy (%) | New accuracy (%) | FAA (%) | Old decline (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for group in aggregate:
        fields = [f'{group[m]["mean"]:.3f} ± {group[m]["sample_sd"]:.3f}' for m in metrics[:4]]
        lines.append('| ' + ' | '.join([group['branch'], str(group['epoch'])] + fields) + ' |')
    lines += ['', '| Epoch | R−S FAA (pp) | FAA wins/ties/losses | R−S old (pp) | Old wins/ties/losses | R−S new (pp) | New wins/ties/losses |',
              '|---|---:|---|---:|---|---:|---|']
    for group in paired:
        fields = [str(group['epoch'])]
        for metric in ['FAA', 'old_accuracy', 'new_accuracy']:
            v = group[metric]
            fields += [f'{v["mean"]:+.3f} ± {v["sample_sd"]:.3f}', f'{v["wins"]}/{v["ties"]}/{v["losses"]}']
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', 'FAA improves in all three seeds at epochs 25 and 50. Final old accuracy improves only for seed 0 and ties for seeds 1 and 2; the mean gain mainly comes from new-task accuracy.',
              '', f'Queue status: `{state["status"]}`. Recomputed statuses: `{counts}`; stored queue counts are stale and should not be used as a live-process count.',
              'Offline seed2 failed; seeds0/1 remain pending. This does not invalidate the accepted native training results, but the full three-layer mechanism study is incomplete.',
              '', 'The valid table is a two-task experiment with a common 50-epoch Task0 SGD parent, followed by 25 epochs SAM or S_F g2 and 25 shared SGD epochs on Task1. Do not mix with ten-task results or earlier 20-epoch formation experiments.']
    (folder / 'INDEPENDENT_TRAINING_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['audit_status'], 'formal_validated': 6,
                      'paired_seeds': [0, 1, 2], 'checks_each': [v['checks'] for v in validations],
                      'finite_observed_floats': all_numeric, 'total_updates': total_steps,
                      'total_gradient_reads': total_reads, 'actual_queue_counts': counts,
                      'output': str(folder / 'INDEPENDENT_TRAINING_AUDIT.json')}, allow_nan=False))


if __name__ == '__main__':
    main()
