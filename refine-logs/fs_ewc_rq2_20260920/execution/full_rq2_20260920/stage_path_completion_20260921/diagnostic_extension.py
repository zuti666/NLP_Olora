#!/usr/bin/env python3
"""Complete formal-cohort diagnostics through the unchanged FullRunner.

Only CLI target coverage and isolated output paths are extended. The formal
1259-example directional path integral and 256-example neighborhood estimator
are reused verbatim. A separate receipt binds this adapter and the formal
reference records, without changing or rewriting the original protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
EXTRA_RADII = [0] + list(range(11, 20))
COMMON_IDENTITY = (
    'seed', 'method', 'coordinate_order', 'coordinate_numel',
    'coordinate_sha256', 'data_manifest_sha256', 'head_sha256',
    'anchor_effective_sha256', 'source_files', 'source_path', 'counts',
    'direction_seed', 'production_precision', 'fisher_definition',
    'torch_version', 'diagnostic_code_sha256', 'common_code_sha256',
    'pilot_runner_code_sha256',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_protocol(meta):
    original = dict(meta)
    digest = original.pop('protocol_sha256')
    require(canonical(original) == digest, 'Manifest canonical protocol hash mismatch')


def close(actual, expected, label, *, atol=3e-6, rtol=1e-7):
    require(math.isfinite(actual) and math.isfinite(expected), f'Nonfinite {label}')
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
            f'{label} changed: {actual} versus formal {expected}')
    return dict(actual=actual, formal=expected, absolute_difference=abs(actual - expected),
                atol=atol, rtol=rtol, passed=True)


def validate_configuration(config, formal, job, output_root=HERE):
    require(job in ('path', 'sparse_train'), 'Unsupported diagnostic job')
    require(type(config.get('training_seed')) is int and config['training_seed'] in range(3),
            'Only the formal training seeds 0, 1, 2 are supported')
    require(set(config) == set(formal), 'Formal config keys must remain unchanged')
    allowed = {'output_dir'} | ({'sparse_train_radii'} if job == 'sparse_train' else set())
    changed = {key for key in config if config[key] != formal[key]}
    require(changed <= allowed and 'output_dir' in changed,
            f'Unauthorized config differences: {sorted(changed)}')
    output = Path(config['output_dir']).resolve()
    require(output.is_relative_to(Path(output_root).resolve()) and output != Path(output_root).resolve(),
            'Output must be inside the isolated formal extension directory')
    require(config['fisher_samples'] == 256 and config['fisher_batch_size'] == 8,
            'Preserve the formal 256-example subset and gradient batch size 8')
    require(config['eval_batch_size'] == 64, 'Preserve formal evaluation batch size 64')
    require(config['sparse_directions'] == 16, 'Preserve all 16 public neighborhood directions')
    if job == 'sparse_train':
        require(config['sparse_train_radii'] == EXTRA_RADII,
                'Radius extension is restricted to origin and radii 11 through 19')
    return sorted(changed)


def load_reference(seed, method, job, coverage):
    directory = BASE / f'formal_results/seed{seed}/{method}/{job}'
    summary_path, meta_path = directory / 'summary.json', directory / 'manifest.json'
    summary, meta = read(summary_path), read(meta_path)
    matches = [r for r in coverage['diagnostics']
               if (r['seed'], r['method'], r['job']) == (seed, method, job)]
    require(len(matches) == 1, f'Formal coverage missing or duplicate: {seed}/{method}/{job}')
    item = matches[0]
    require(sha(summary_path) == item['sha256'], 'Formal summary differs from completed coverage receipt')
    verify_protocol(meta)
    require(summary['protocol_sha256'] == meta['protocol_sha256'] == item['protocol_sha256'],
            'Formal summary/manifest/coverage protocol mismatch')
    require(summary['status'] == 'complete' and summary['method'] == method and summary['seed'] == seed,
            'Reference diagnostic is not a completed matching formal run')
    rows_path = directory / 'states.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    require(len(rows) == summary['states_completed'] == item['states'], 'Formal raw-state count mismatch')
    require(len({r['state_id'] for r in rows}) == len(rows), 'Duplicate formal state IDs')
    for row in rows:
        require(row['protocol_sha256'] == meta['protocol_sha256'], 'Formal raw-state protocol mismatch')
        require(row['seed'] == seed and row['method'] == method and row['anchor_restored_exactly'],
                'Formal raw-state identity or restoration mismatch')
    return dict(summary=summary, meta=meta, rows=rows,
                sources=[dict(path=str(p), sha256=sha(p)) for p in (summary_path, meta_path, rows_path)])


def preflight(args):
    config_path = Path(args.config).resolve()
    config = read(config_path)
    seed = config.get('training_seed')
    require(type(seed) is int and seed in range(3), 'Invalid formal seed')
    formal_config_path = BASE / f'configs/seed{seed}.json'
    formal_config = read(formal_config_path)
    changed = validate_configuration(config, formal_config, args.job)
    require(args.method in ('sgd', 'gam'), 'Invalid method')
    require(args.job != 'path' or args.path_task in range(1, 10), 'Path target must be Task1 through Task9')
    coverage_path = BASE / 'final_report/coverage.json'
    coverage = read(coverage_path)
    require(not coverage['errors'], 'The formal reference coverage contains errors')
    reference_job = 'path_t01' if args.job == 'path' else 'sparse_train'
    reference = load_reference(seed, args.method, reference_job, coverage)
    trajectory = load_reference(seed, args.method, 'trajectory_fisher', coverage)
    meta = reference['meta']
    require(meta['config'] == formal_config and meta['config_sha256'] == sha(formal_config_path),
            'The formal config was changed after reference measurement')
    require(meta['counts'] == {'train': 1259, 'test': 307, 'subset': 256}, 'Formal data counts changed')
    code = {
        BASE / 'diagnostic_full.py': meta['diagnostic_code_sha256'],
        EXECUTION / 'rq2_common.py': meta['common_code_sha256'],
        EXECUTION / 'diagnostic.py': meta['pilot_runner_code_sha256'],
    }
    source = Path(config['source_path'])
    code.update({source / relative: digest for relative, digest in meta['source_files'].items()})
    for path, digest in code.items():
        require(sha(path) == digest, f'Formal measurement source changed: {path}')
    for method, tasks in config['checkpoints'].items():
        for task, checkpoint in tasks.items():
            require(Path(checkpoint).is_file(), f'Missing formal checkpoint: {method}/Task{task}')
    evidence = dict(
        status='passed', seed=seed, method=args.method, job=args.job,
        target_task=args.path_task if args.job == 'path' else None,
        config_changes=changed, config=dict(path=str(config_path), sha256=sha(config_path)),
        formal_config=dict(path=str(formal_config_path), sha256=sha(formal_config_path)),
        coverage=dict(path=str(coverage_path), sha256=sha(coverage_path)),
        reference_sources=reference['sources'], trajectory_sources=trajectory['sources'],
        original_code=[dict(path=str(p), sha256=digest) for p, digest in code.items()],
        wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
    )
    return config, reference, trajectory, evidence


def validate_loaded_identity(runner, reference, trajectory):
    formal = reference['meta']
    for key in COMMON_IDENTITY:
        require(runner.meta[key] == formal[key], f'Loaded formal identity changed: {key}')
    require(runner.meta['fisher_scopes'] == formal['fisher_scopes'], 'Fisher scope changed')
    method = runner.args.method
    for task, digest in runner.meta['checkpoint_sha256'][method].items():
        require(digest == trajectory['meta']['checkpoint_sha256'][method][task],
                f'Loaded checkpoint differs from formal trajectory: {method}/Task{task}')
    if runner.args.job == 'path':
        t = runner.args.path_task
        endpoint = next(item for item in trajectory['summary']['tasks'] if item['task'] == t)
        name = f'{method}_t{t:02d}'
        require(runner.meta['actual_direction_sha256'][name] == endpoint['endpoint_recovery']['effective_delta_sha256'],
                'Path direction differs from completed formal trajectory')
    return dict(passed=True, matched_fields=list(COMMON_IDENTITY),
                checkpoint_sha256=runner.meta['checkpoint_sha256'])


def validate_origin(origin, reference, job):
    formal = next(row for row in reference['rows'] if row['state_id'] == 'r0')
    scope = 'train' if job == 'path' else 'subset'
    samples = 1259 if job == 'path' else 256
    checks = {}
    for split in ('train', 'test'):
        require(origin['metrics'][split]['samples'] == formal['metrics'][split]['samples'], 'Evaluation count changed')
        for field in ('loss', 'accuracy'):
            checks[f'{split}_{field}'] = close(origin['metrics'][split][field], formal['metrics'][split][field],
                                               f'Origin {split} {field}')
    current, baseline = origin['fisher_by_scope'][scope], formal['fisher_by_scope'][scope]
    require(current['samples'] == baseline['samples'] == samples, 'Origin Fisher sample count changed')
    for key in ('normalized', 'floor', 'regularizer', 'estimator', 'coordinate', 'gradient_batch_size'):
        require(current[key] == baseline[key], f'Origin Fisher estimator changed: {key}')
    checks['raw_trace'] = close(current['raw_trace'], baseline['raw_trace'], 'Origin raw Fisher trace', atol=1e-5, rtol=1e-5)
    checks['fisher_scope_loss'] = close(current['subset_loss'], baseline['subset_loss'], 'Origin Fisher scope loss')
    # q is intentionally not compared: each target task has its own direction.
    return dict(passed=True, samples=samples, scope=scope, checks=checks)


def validate_finished(summary, trajectory, args):
    require(summary['status'] == 'complete' and summary['method'] == args.method and summary['job'] == args.job,
            'Extension did not finish the requested diagnostic')
    if args.job == 'sparse_train':
        require(summary['expected_states'] == summary['states_completed'] == 145,
                'Radius extension must have 1 + 9 x 16 states')
        require([row['radius'] for row in summary['radius_summary']] == EXTRA_RADII,
                'Completed radius grid changed')
        require(summary['direction_indices'] == list(range(16)), 'Completed direction library changed')
        return dict(passed=True, samples=256, measured_states=145, radii=EXTRA_RADII,
                    interpretation='Only appended radii; recompute tolerance from the full merged grid if needed')
    require(summary['target_task'] == args.path_task and summary['samples'] == 1259,
            'Path target or full training sample scope changed')
    require(summary['selected_nodes'] in (9, 17, 33), 'Unexpected quadrature grid')
    for key in ('numerical_gate_passed', 'quadrature_converged', 'bound_passed_with_numeric_tolerance'):
        require(summary[key] is True, f'Path numerical gate failed: {key}')
    endpoint = next(item for item in trajectory['summary']['tasks'] if item['task'] == args.path_task)
    checks = {'drift': close(summary['drift'], endpoint['effective_weight_drift'], 'Path endpoint drift', atol=2e-6)}
    for split in ('train', 'test'):
        checks[f'{split}_damage'] = close(summary[f'D_{split}'], endpoint['D0'][split], f'Path endpoint {split} damage')
        for field in ('loss', 'accuracy'):
            checks[f'{split}_{field}'] = close(summary['path'][-1]['metrics'][split][field],
                                               endpoint['metrics'][split][field], f'Path endpoint {split} {field}')
    close(summary['S_path'], summary['integral'] / summary['drift_squared'], 'Normalized path sensitivity', atol=1e-12, rtol=1e-10)
    require(summary['endpoint_recovery']['effective_delta_sha256'] == endpoint['endpoint_recovery']['effective_delta_sha256'],
            'Finished path endpoint direction changed')
    return dict(passed=True, samples=1259, selected_nodes=summary['selected_nodes'], checks=checks,
                numerical_gate_passed=True, quadrature_converged=True,
                bound_passed_with_numeric_tolerance=True)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=('sgd', 'gam'))
    parser.add_argument('--job', required=True, choices=('path', 'sparse_train'))
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
    directions = ({f'{args.method}_t{args.path_task:02d}': runner.deltas[f'{args.method}_t{args.path_task:02d}']}
                  if args.job == 'path' else None)
    origin = runner.origin_full(['train' if args.job == 'path' else 'subset'], directions)
    origin_check = validate_origin(origin, reference, args.job)
    if args.job == 'path':
        runner.path()
    else:
        runner.neighborhood_full()
    summary_path = runner.directory / 'summary.json'
    summary = read(summary_path)
    finished = validate_finished(summary, trajectory, args)
    receipt = dict(
        status='complete', schema='rq2_formal_completion_v1', job=args.job,
        method=args.method, seed=runner.seed,
        target_task=args.path_task if args.job == 'path' else None,
        samples=finished['samples'],
        protocol_sha256=runner.protocol, summary=str(summary_path),
        summary_sha256=sha(summary_path), manifest_sha256=sha(runner.directory / 'manifest.json'),
        states_sha256=sha(runner.directory / 'states.jsonl'), wrapper_sha256=sha(__file__),
        numerical_gate_passed=summary.get('numerical_gate_passed'),
        quadrature_converged=summary.get('quadrature_converged'),
        bound_passed_with_numeric_tolerance=summary.get('bound_passed_with_numeric_tolerance'),
        preflight=evidence, identity_check=identity, origin_check=origin_check, endpoint_check=finished,
    )
    atomic_json(runner.directory / 'extension_receipt.json', receipt)
    print(json.dumps(dict(event='extension_complete', receipt=str(runner.directory / 'extension_receipt.json'),
                          summary_sha256=receipt['summary_sha256']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
