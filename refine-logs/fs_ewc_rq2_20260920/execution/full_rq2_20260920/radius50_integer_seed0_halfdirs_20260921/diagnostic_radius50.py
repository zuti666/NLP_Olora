#!/usr/bin/env python3
"""Fill missing integer radii 25--50 for seed0; halve dense directions only.

Dense: first 50 public directions, full 1259/307 train/test evaluation, no Fisher.
Sparse_train: first 16 identical directions, 256-example raw empirical Fisher.
The origin is repeated for identity checks. Extension-only tolerance values
must not be interpreted before merging the original 0--20 dense grid.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
EXTRA_RADII = [0, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49]
DIRECTION_COUNTS = {'dense': 50, 'sparse_train': 16}
HELPER_PATH = BASE / 'stage_path_completion_20260921/diagnostic_extension.py'
HELPER_SHA256 = '37b059f42420c37450cfaaed4e6258a1ac52d6c63380322d7057fe43ca5d10a8'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError('The frozen formal extension validation helper changed')
_spec = importlib.util.spec_from_file_location('_formal_extension_validation', HELPER_PATH)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
read, sha, require, close = _helper.read, _helper.sha, _helper.require, _helper.close
verify_protocol = _helper.verify_protocol
load_reference = _helper.load_reference
validate_loaded_identity = _helper.validate_loaded_identity


def validate_configuration(config, formal, job):
    require(job in ('dense', 'sparse_train'), 'Unsupported radius diagnostic')
    require(type(config.get('training_seed')) is int and config['training_seed'] == 0, 'Invalid formal seed')
    require(set(config) == set(formal), 'Formal config keys must remain unchanged')
    radius_key = 'dense_radii' if job == 'dense' else 'sparse_train_radii'
    changed = {key for key in config if config[key] != formal[key]}
    allowed = {'output_dir', radius_key} | ({'dense_directions'} if job == 'dense' else set())
    require(changed == allowed, f'Unauthorized config differences: {sorted(changed)}')
    require(config[radius_key] == EXTRA_RADII, 'Radius extension must contain origin and the 20 missing integer radii from 25 through 50')
    require(Path(config['output_dir']).resolve() == HERE / f"results/seed{config['training_seed']}", 'Unexpected output root')
    require(config['dense_directions'] == (50 if job == 'dense' else 100) and config['sparse_directions'] == 16, 'Dense must use 50 directions; Fisher retains all 16 formal directions')
    require(config['fisher_samples'] == 256 and config['fisher_batch_size'] == 8, 'Formal Fisher estimator scope changed')
    require(config['eval_batch_size'] == 64, 'Formal evaluation batch size changed')
    return sorted(changed)


def preflight(args):
    config_path = Path(args.config).resolve()
    config = read(config_path)
    seed = config.get('training_seed')
    require(type(seed) is int and seed == 0, 'Invalid formal seed')
    require(args.method in ('sgd', 'gam'), 'Invalid method')
    formal_path = BASE / f'configs/seed{seed}.json'
    formal = read(formal_path)
    changed = validate_configuration(config, formal, args.job)
    coverage_path = BASE / 'final_report/coverage.json'
    coverage = read(coverage_path)
    require(not coverage['errors'], 'Formal coverage contains errors')
    reference = load_reference(seed, args.method, args.job, coverage)
    trajectory = load_reference(seed, args.method, 'trajectory_fisher', coverage)
    meta = reference['meta']
    require(meta['config'] == formal and meta['config_sha256'] == sha(formal_path), 'Formal reference config changed')
    require(meta['counts'] == {'train': 1259, 'test': 307, 'subset': 256}, 'Formal data counts changed')
    code = {BASE / 'diagnostic_full.py': meta['diagnostic_code_sha256'],
            EXECUTION / 'rq2_common.py': meta['common_code_sha256'],
            EXECUTION / 'diagnostic.py': meta['pilot_runner_code_sha256'],
            HELPER_PATH: HELPER_SHA256}
    source = Path(config['source_path'])
    code.update({source / relative: digest for relative, digest in meta['source_files'].items()})
    for path, digest in code.items():
        require(sha(path) == digest, f'Formal measurement source changed: {path}')
    for method, tasks in config['checkpoints'].items():
        for task, checkpoint in tasks.items():
            require(Path(checkpoint).is_file(), f'Missing formal checkpoint: {method}/Task{task}')
    directions = direction_library(reference['rows'])
    count = DIRECTION_COUNTS[args.job]
    formal_count = 100 if args.job == 'dense' else 16
    require(set(directions) == set(range(formal_count)), 'Formal public direction library incomplete')
    require(set(range(count)).issubset(directions), 'Requested prefix direction library incomplete')
    evidence = dict(status='passed', seed=seed, method=args.method, job=args.job,
        config_changes=changed, radii=EXTRA_RADII, expected_states=1 + (len(EXTRA_RADII) - 1) * count,
        direction_indices=list(range(count)), formal_direction_count=formal_count,
        config=dict(path=str(config_path), sha256=sha(config_path)),
        formal_config=dict(path=str(formal_path), sha256=sha(formal_path)),
        coverage=dict(path=str(coverage_path), sha256=sha(coverage_path)),
        reference_sources=reference['sources'], trajectory_sources=trajectory['sources'],
        original_code=[dict(path=str(path), sha256=digest) for path, digest in code.items()],
        wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    return config, reference, trajectory, evidence


def direction_library(rows):
    library = {}
    for row in rows:
        direction = row['point'].get('direction')
        if direction is not None:
            index = direction['index']
            require(index not in library or library[index] == direction, 'Direction identity changed between radii')
            library[index] = direction
    return library


def validate_origin(origin, reference, job):
    if job == 'sparse_train':
        return _helper.validate_origin(origin, reference, job)
    baseline = next(row for row in reference['rows'] if row['state_id'] == 'r0')
    require(origin['fisher_by_scope'] == baseline['fisher_by_scope'] == {}, 'Dense origin must be forward-only')
    checks = {}
    for split, count in [('train', 1259), ('test', 307)]:
        require(origin['metrics'][split]['samples'] == baseline['metrics'][split]['samples'] == count, 'Origin evaluation count changed')
        for key in ('loss', 'accuracy'):
            checks[f'{split}_{key}'] = close(origin['metrics'][split][key], baseline['metrics'][split][key], f'Origin {split} {key}')
    return dict(passed=True, samples={'train': 1259, 'test': 307}, scope='forward_only', checks=checks)


def finite_tree(value, label='result'):
    if isinstance(value, float):
        require(math.isfinite(value), f'Nonfinite {label}')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f'{label}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f'{label}[{index}]')


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = math.floor(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])


def numerical_equal(actual, expected, label):
    return close(float(actual), float(expected), label, atol=1e-10, rtol=1e-10)


def validate_finished(summary, meta, rows, reference, job, radii=EXTRA_RADII, direction_count=None):
    """Recompute every plotted statistic from finite, identity-bound raw rows."""
    verify_protocol(meta)
    finite_tree(summary, 'summary')
    count = DIRECTION_COUNTS[job] if direction_count is None else direction_count
    require(type(count) is int and 0 < count <= (100 if job == 'dense' else 16), 'Invalid validation direction count')
    scope = [] if job == 'dense' else ['subset']
    expected_ids = {'r0'} | {f'r{r:.12g}_u{i:03d}' for r in radii if r for i in range(count)}
    require(summary['status'] == 'complete' and summary['job'] == meta['job'] == job, 'Requested job did not complete')
    require(summary['seed'] == meta['seed'] == reference['meta']['seed'] and summary['method'] == meta['method'] == reference['meta']['method'], 'Result identity changed')
    require(summary['protocol_sha256'] == meta['protocol_sha256'], 'Summary protocol mismatch')
    require(summary['expected_states'] == summary['states_completed'] == len(rows) == len(expected_ids), 'Incomplete state count')
    require({row['state_id'] for row in rows} == expected_ids, 'State grid mismatch or duplicate states')
    require(summary['direction_indices'] == list(range(count)) and summary['paired_public_direction_library'] is True, 'Direction coverage changed')
    require(summary['gradient_scope'] == meta['fisher_scopes'] == scope and summary['forward_only'] == (job == 'dense'), 'Fisher evaluation scope changed')
    require([item['radius'] for item in summary['radius_summary']] == radii, 'Summary radius grid changed')
    library = direction_library(reference['rows'])
    by_id = {row['state_id']: row for row in rows}
    origin = by_id['r0']
    origin_check = validate_origin(origin, reference, job)
    for row in rows:
        finite_tree(row, row['state_id'])
        require(row['protocol_sha256'] == meta['protocol_sha256'] and row['anchor_restored_exactly'] is True, 'Raw row protocol/restoration mismatch')
        require(row['seed'] == meta['seed'] and row['method'] == meta['method'], 'Raw row seed/method mismatch')
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == meta[key], f'Raw row {key} mismatch')
        require(row['checkpoint_sha256'] == meta['checkpoint_sha256'][meta['method']], 'Raw row checkpoint mismatch')
        radius = row['point']['radius']
        require(radius in radii and row['amount'] == radius, 'Raw row radius/amount mismatch')
        close(row['effective_weight_drift'], float(radius), 'Measured drift', atol=1e-5, rtol=2e-6)
        numerical_equal(row['effective_weight_drift_squared'], row['effective_weight_drift'] ** 2, 'Squared measured drift')
        if radius:
            direction = row['point']['direction']
            index = direction['index']
            require(index in range(count) and direction == library[index], 'Public direction differs from formal run')
            require(row['point']['kind'] == job and row['state_id'] == f'r{radius:.12g}_u{index:03d}', 'Raw state ID disagrees with radius/direction')
        else:
            require(row['state_id'] == 'r0' and row['point']['kind'] == 'anchor', 'Malformed origin')
        for split, samples in [('train', 1259), ('test', 307)]:
            metric = row['metrics'][split]
            require(metric['samples'] == samples and metric['loss'] >= 0 and 0 <= metric['accuracy'] <= 100, 'Invalid evaluation metric or count')
        require(sorted(row['fisher_by_scope']) == scope, 'Raw Fisher scope changed')
        if job == 'sparse_train':
            fisher = row['fisher_by_scope']['subset']
            baseline = origin['fisher_by_scope']['subset']
            for key in ('samples', 'normalized', 'floor', 'regularizer', 'estimator', 'coordinate', 'gradient_batch_size'):
                require(fisher[key] == baseline[key], f'Fisher metadata changed: {key}')
            require(fisher['samples'] == 256 and fisher['normalized'] is False and fisher['regularizer'] is None and fisher['floor'] is None, 'Fisher was normalized or regularized')
            require(fisher['raw_trace'] >= 0 and fisher['subset_loss'] >= 0 and fisher['q'] == {}, 'Invalid raw Fisher value')
    for item in summary['radius_summary']:
        radius = item['radius']
        selected = [origin] if radius == 0 else [by_id[f'r{radius:.12g}_u{i:03d}'] for i in range(count)]
        require(item['directions'] == len(selected), 'Radius direction count mismatch')
        for split in ('train', 'test'):
            actual = item[split]
            signed = [row['metrics'][split]['loss'] - origin['metrics'][split]['loss'] for row in selected]
            positive = [max(value, 0.) for value in signed]
            require(len(actual['D_each']) == len(signed), 'Saved per-direction loss count mismatch')
            for got, expected in zip(actual['D_each'], signed):
                numerical_equal(got, expected, 'Per-direction CE change')
            mean_accuracy = sum(row['metrics'][split]['accuracy'] for row in selected) / len(selected)
            recomputed = dict(D_mean=sum(signed) / len(signed), G=sum(positive) / len(positive),
                Q95=quantile(positive, .95), sampled_max=max(positive),
                absolute_loss_mean=sum(row['metrics'][split]['loss'] for row in selected) / len(selected),
                accuracy_mean=mean_accuracy,
                accuracy_drop_mean_pp=origin['metrics'][split]['accuracy'] - mean_accuracy)
            for key, expected in recomputed.items():
                numerical_equal(actual[key], expected, f'{split}/r{radius}/{key}')
        if job == 'sparse_train':
            actual = item['raw_trace_by_scope']['subset']
            traces = [row['fisher_by_scope']['subset']['raw_trace'] for row in selected]
            require(actual['samples'] == 256 and actual['each'] == traces, 'Saved raw Fisher list differs from states')
            for key, expected in dict(mean=sum(traces)/len(traces), Q95=quantile(traces, .95), sampled_max=max(traces)).items():
                numerical_equal(actual[key], expected, f'Fisher/r{radius}/{key}')
    for item in summary['tolerance']:
        radius = 0
        for row in summary['radius_summary']:
            if row[item['split']][item['metric']] > item['epsilon']:
                break
            radius = row['radius']
        require(item['prefix_radius'] == radius and item['right_censored'] == (radius == max(radii)), 'Extension-grid tolerance summary mismatch')
    ending = summary['end_restoration']
    require(ending['effective_sha256'] == meta['anchor_effective_sha256'] and ending['predictions_match'] is True, 'Final anchor restoration mismatch')
    for key in ('loss', 'accuracy'):
        numerical_equal(ending['test'][key], origin['metrics']['test'][key], f'Final anchor test {key}')
    return dict(passed=True, samples=256 if job == 'sparse_train' else {'train': 1259, 'test': 307},
        measured_states=len(rows), radii=radii, directions_per_nonzero_radius=count,
        raw_states_finite=True, all_statistics_recomputed=True, direction_identities_match_formal=True,
        origin_check=origin_check,
        interpretation='Append measured radii only; derive tolerance from merged original and extension dense grid, never this sparse extension grid alone')


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=('sgd', 'gam'))
    parser.add_argument('--job', required=True, choices=('dense', 'sparse_train'))
    parser.add_argument('--path-task', type=int, default=9, choices=range(1, 10))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--validate-only', action='store_true', help='Pure CPU file/config/source validation; no model import')
    return parser


def main():
    args = argument_parser().parse_args()
    _, reference, trajectory, evidence = preflight(args)
    if args.validate_only:
        print(json.dumps(evidence, indent=2, allow_nan=False))
        return
    sys.path.insert(0, str(BASE))
    from diagnostic_full import FullRunner
    from rq2_common import atomic_json
    runner = FullRunner(args)
    identity = validate_loaded_identity(runner, reference, trajectory)
    origin = runner.origin_full([] if args.job == 'dense' else ['subset'])
    origin_check = validate_origin(origin, reference, args.job)
    runner.neighborhood_full()
    summary_path = runner.directory / 'summary.json'
    summary = read(summary_path)
    rows = [json.loads(line) for line in (runner.directory / 'states.jsonl').read_text().splitlines()]
    finished = validate_finished(summary, runner.meta, rows, reference, args.job)
    receipt = dict(status='complete', schema='rq2_radius50_integer_seed0_halfdirs_v1', job=args.job,
        method=args.method, seed=runner.seed, samples=finished['samples'],
        protocol_sha256=runner.protocol, summary=str(summary_path),
        summary_sha256=sha(summary_path), manifest_sha256=sha(runner.directory / 'manifest.json'),
        states_sha256=sha(runner.directory / 'states.jsonl'), wrapper_sha256=sha(__file__),
        preflight=evidence, identity_check=identity, origin_check=origin_check, result_check=finished,
        endpoint_check=finished)
    atomic_json(runner.directory / 'extension_receipt.json', receipt)
    print(json.dumps(dict(event='extension_complete', receipt=str(runner.directory / 'extension_receipt.json'),
        summary_sha256=receipt['summary_sha256']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
