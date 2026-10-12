#!/usr/bin/env python3
"""A fixed positive Task0 full-training CE gradient ray in effective Q/V space.

This diagnostic computes the gradient once at each method's own Task0 anchor.
It is not an optimizer step, a LoRA-factor gradient, or iterative training.
All 93 points independently restore theta0 then apply theta0 + r * u.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
RADII = sorted(set([i / 4 for i in range(81)] + [25, 30, 35, 40, 45, 50] + [.001, .005, .01, .025, .05, .1]))
WITNESS_STEPS = [1e-4, 5e-4, 1e-3]
GRADIENT_BATCH_SIZE = 8
HELPER_PATH = BASE / 'radius50_extension_20260921/diagnostic_radius50.py'
HELPER_SHA256 = '29b8d4306f17a1d5945643f961df6d5ec7ff56363f416d69ac99a852b298a5bc'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError('Frozen radius-extension helper changed')
_spec = importlib.util.spec_from_file_location('_radius50_validation', HELPER_PATH)
support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(support)
read, sha, require, close = support.read, support.sha, support.require, support.close
canonical = support._helper.canonical
finite_tree = support.finite_tree
numerical_equal = support.numerical_equal


def validate_configuration(config, formal):
    require(set(config) == set(formal), 'Formal configuration keys changed')
    require({key for key in config if config[key] != formal[key]} == {'output_dir'}, 'Only isolated output_dir may differ')
    seed = config['training_seed']
    require(type(seed) is int and seed in range(3), 'Only formal seeds 0, 1, 2 are supported')
    require(Path(config['output_dir']).resolve() == HERE / f'results/seed{seed}', 'Unexpected output directory')
    require(config['fisher_samples'] == 256 and config['fisher_batch_size'] == 8 and config['eval_batch_size'] == 64, 'Formal evaluation/Fisher setting changed')


def preflight(args):
    require(args.job == 'fixed_gradient' and args.gradient_task == 0, 'This protocol is restricted to the user-selected Task0 gradient')
    require(args.method in ('sgd', 'gam'), 'Unknown method')
    config_path = Path(args.config).resolve()
    config = read(config_path)
    seed = config.get('training_seed')
    require(type(seed) is int and seed in range(3), 'Invalid seed')
    formal_path = BASE / f'configs/seed{seed}.json'
    formal = read(formal_path)
    validate_configuration(config, formal)
    coverage_path = BASE / 'final_report/coverage.json'
    coverage = read(coverage_path)
    require(not coverage['errors'], 'Formal source coverage contains errors')
    reference = support.load_reference(seed, args.method, 'sparse_train', coverage)
    trajectory = support.load_reference(seed, args.method, 'trajectory_fisher', coverage)
    meta = reference['meta']
    require(meta['config'] == formal and meta['config_sha256'] == sha(formal_path), 'Formal source configuration changed')
    require(meta['counts'] == {'train': 1259, 'test': 307, 'subset': 256}, 'Formal dataset size changed')
    code = {BASE / 'diagnostic_full.py': meta['diagnostic_code_sha256'],
            EXECUTION / 'rq2_common.py': meta['common_code_sha256'],
            EXECUTION / 'diagnostic.py': meta['pilot_runner_code_sha256'],
            HELPER_PATH: HELPER_SHA256,
            support.HELPER_PATH: support.HELPER_SHA256}
    code.update({Path(config['source_path']) / relative: digest for relative, digest in meta['source_files'].items()})
    for path, digest in code.items():
        require(sha(path) == digest, f'Formal source changed: {path}')
    for checkpoint in config['checkpoints'][args.method].values():
        require(Path(checkpoint).is_file(), f'Formal checkpoint missing: {checkpoint}')
    evidence = dict(status='passed', seed=seed, method=args.method, job='fixed_gradient', gradient_task=0, direction_sign=1,
        radii=RADII, expected_states=len(RADII), config_changes=['output_dir'],
        config=dict(path=str(config_path), sha256=sha(config_path)),
        formal_config=dict(path=str(formal_path), sha256=sha(formal_path)),
        coverage=dict(path=str(coverage_path), sha256=sha(coverage_path)),
        reference_sources=reference['sources'], trajectory_sources=trajectory['sources'],
        original_code=[dict(path=str(path), sha256=digest) for path, digest in code.items()],
        wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    return config, reference, trajectory, evidence


def validate_direction_scalars(meta):
    finite_tree(meta, 'gradient_manifest')
    require(meta['gradient_task'] == 0 and meta['direction_sign'] == 1 and meta['samples'] == 1259, 'Gradient objective scope changed')
    require(meta['gradient_batch_size'] == 8 and meta['sample_reduction'] == 'sum of CE gradients over every training example, divided once by N=1259', 'Gradient batching/reduction changed')
    require(meta['regularizer'] is None and meta['optimizer_step'] is False and meta['held_fixed_across_radii'] is True, 'Direction is not a fixed plain-CE diagnostic')
    require(meta['coordinate_numel'] == 14155776 and meta['raw_grad_norm'] > 0, 'Invalid effective gradient')
    close(meta['direction_norm'], 1., 'Direction global Frobenius norm', atol=5e-7, rtol=0.)
    close(meta['gradient_dot_direction'], meta['raw_grad_norm'], 'Positive-gradient directional derivative', atol=1e-9, rtol=2e-6)
    require(meta['gradient_dot_direction'] > 0, 'Direction is not an ascent direction')
    return dict(passed=True, unit_global_frobenius_norm=True, analytic_ascent=True,
                raw_grad_norm=meta['raw_grad_norm'], direction_norm=meta['direction_norm'],
                gradient_dot_direction=meta['gradient_dot_direction'])


def accumulate_full_gradient(model, params, x, y, device, batch_size=8):
    """FP32 autograd of batch sums; FP64 accumulation; divide by N once."""
    import torch
    from torch.nn import functional as F
    require(len(y) > 0, 'Empty gradient sample set')
    accumulated = [torch.zeros_like(param, dtype=torch.float64, device=device) for param in params]
    loss_sum = 0.
    for start in range(0, len(y), batch_size):
        model.zero_grad(set_to_none=True)
        losses = F.cross_entropy(model(x[start:start + batch_size].to(device)), y[start:start + batch_size].to(device), reduction='none')
        losses.sum().backward()
        with torch.no_grad():
            for destination, param in zip(accumulated, params):
                require(param.grad is not None and bool(torch.isfinite(param.grad).all()), 'Missing/nonfinite effective gradient')
                destination.add_(param.grad.detach().double())
        loss_sum += float(losses.detach().double().sum())
    model.zero_grad(set_to_none=True)
    return [value / len(y) for value in accumulated], loss_sum / len(y)


def normalize_positive_gradient(gradient):
    import torch
    norm = math.sqrt(sum(float(value.double().square().sum()) for value in gradient))
    require(math.isfinite(norm) and norm > 0, 'Zero or nonfinite full gradient')
    direction = [(value.double() / norm).to(torch.float32) for value in gradient]
    direction_norm = math.sqrt(sum(float(value.double().square().sum()) for value in direction))
    dot = sum(float((g.double() * u.double()).sum()) for g, u in zip(gradient, direction))
    return direction, norm, direction_norm, dot


def validate_gradient_tensors(gradient, direction):
    """Check the actual saved vector, independently of claimed metadata."""
    import torch
    expected, norm, direction_norm, dot = normalize_positive_gradient(gradient)
    require(len(direction) == len(expected), 'Direction tensor count differs from gradient')
    require(all(torch.equal(actual, wanted) for actual, wanted in zip(direction, expected)),
            'Actual direction differs from the positive normalized gradient')
    require(dot > 0, 'Actual direction has the wrong sign')
    return norm, direction_norm, dot


def gradient_binding(runner):
    from rq2_common import tensor_hash
    return dict(seed=runner.seed, method=runner.args.method, gradient_task=0,
        checkpoint_sha256=runner.meta['checkpoint_sha256'][runner.args.method],
        anchor_effective_sha256=runner.meta['anchor_effective_sha256'],
        coordinate_sha256=runner.meta['coordinate_sha256'], coordinate_numel=runner.meta['coordinate_numel'],
        head_sha256=runner.meta['head_sha256'], data_manifest_sha256=runner.meta['data_manifest_sha256'],
        gradient_samples_sha256=canonical(runner.data['records']['train']),
        decoded_train_sha256=runner.data['decoded_tensor_sha256']['train'],
        training_labels_sha256=tensor_hash([runner.y['train']]),
        wrapper_sha256=sha(__file__))


def prepare_gradient(runner, directory):
    import torch
    from rq2_common import atomic_json, tensor_hash
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / 'gradient_manifest.json'
    artifact_path = directory / 'gradient_direction.pt'
    sample_path = directory / 'gradient_sample_manifest.json'
    binding = gradient_binding(runner)
    if meta_path.exists():
        meta = read(meta_path)
        for key, value in binding.items():
            require(meta[key] == value, f'Persisted gradient identity changed: {key}')
        require(meta['gradient_artifact_sha256'] == sha(artifact_path) and meta['sample_manifest_sha256'] == sha(sample_path), 'Persisted gradient artifact/sample hash changed')
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
        gradient = artifact['gradient_mean_fp64']
        direction = artifact['unit_positive_gradient_fp32']
        require(artifact['coordinate_order'] == runner.meta['coordinate_order'], 'Gradient coordinate order changed')
        require(len(gradient) == len(direction) == 24, 'Saved gradient tensor count changed')
        require(all(tuple(value.shape) == (768, 768) and value.dtype == torch.float64 and bool(torch.isfinite(value).all()) for value in gradient), 'Saved gradient shape/dtype/values invalid')
        require(all(tuple(value.shape) == (768, 768) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in direction), 'Saved direction shape/dtype/values invalid')
        norm, direction_norm, dot = validate_gradient_tensors(gradient, direction)
        require(tensor_hash(gradient) == meta['gradient_sha256'] and tensor_hash(direction) == meta['direction_sha256'], 'Saved gradient/direction tensor hash changed')
        for key, value in [('raw_grad_norm', norm), ('direction_norm', direction_norm), ('gradient_dot_direction', dot)]:
            close(meta[key], value, f'Reloaded {key}', atol=1e-10, rtol=1e-10)
        validate_direction_scalars(meta)
        return [value.to(runner.device) for value in direction], meta
    require(not runner.collector.active, 'Per-example Fisher collector unexpectedly active during full gradient')
    runner.restore()
    gradient, objective = accumulate_full_gradient(runner.model, runner.params, runner.x['train'], runner.y['train'], runner.device, GRADIENT_BATCH_SIZE)
    # Store the full mean gradient in FP64, not merely a direction or diagonal.
    # Normalize on CPU both initially and after resume to make the saved FP32
    # vector bitwise reproducible across device reduction implementations.
    gradient_cpu = [value.cpu() for value in gradient]
    direction_cpu, norm, direction_norm, dot = normalize_positive_gradient(gradient_cpu)
    direction = [value.to(runner.device) for value in direction_cpu]
    temporary = artifact_path.with_name(artifact_path.name + f'.{os.getpid()}.tmp')
    torch.save(dict(gradient_mean_fp64=gradient_cpu, unit_positive_gradient_fp32=direction_cpu,
                    coordinate_order=runner.meta['coordinate_order']), temporary)
    os.replace(temporary, artifact_path)
    atomic_json(sample_path, dict(gradient_task=0, split='train', samples=1259,
                                records=runner.data['records']['train'], records_sha256=binding['gradient_samples_sha256']))
    meta = dict(**binding, schema='fixed_task0_gradient_v1', direction_sign=1, samples=1259, gradient_batch_size=8,
        sample_reduction='sum of CE gradients over every training example, divided once by N=1259',
        objective='true-label Task0 full-training mean cross-entropy with the fixed old20 head',
        objective_at_anchor=objective, regularizer=None, optimizer_step=False, held_fixed_across_radii=True,
        gradient_precision='FP32 model and batch-sum gradients; FP64 accumulation and sample averaging; CPU FP64 global normalization; FP32 direction',
        raw_grad_norm=norm, direction_norm=direction_norm, gradient_dot_direction=dot,
        gradient_sha256=tensor_hash(gradient_cpu), direction_sha256=tensor_hash(direction_cpu),
        gradient_artifact_sha256=sha(artifact_path), sample_manifest_sha256=sha(sample_path))
    validate_direction_scalars(meta)
    runner.restore()
    atomic_json(meta_path, meta)
    return direction, meta


def validate_witness(witness, gradient_meta):
    finite_tree(witness, 'witness')
    require(witness['direction_sha256'] == gradient_meta['direction_sha256'], 'Witness direction changed')
    require(witness['samples'] == 1259 and witness['gradient_task'] == 0 and witness['direction_sign'] == 1, 'Witness objective/sign changed')
    require([row['h'] for row in witness['steps']] == WITNESS_STEPS, 'Witness finite-difference grid changed')
    norm = gradient_meta['raw_grad_norm']
    for row in witness['steps']:
        h = row['h']
        numerical_equal(row['central_derivative'], (row['loss_plus'] - row['loss_minus']) / (2 * h), 'Central finite difference')
        numerical_equal(row['forward_derivative'], (row['loss_plus'] - witness['origin_loss']) / h, 'Forward finite difference')
        numerical_equal(row['relative_derivative_error'], abs(row['central_derivative'] - norm) / norm, 'Derivative relative error')
        require(row['anchor_restored_exactly'] is True, 'Witness anchor not restored')
    first = witness['steps'][0]
    require(first['loss_plus'] > witness['origin_loss'] + max(1e-8, .1 * first['h'] * norm), 'Small positive step did not increase Task0 training CE')
    require(min(row['relative_derivative_error'] for row in witness['steps']) <= .05, 'Finite-difference derivative disagrees with full gradient by more than 5%')
    require(witness['origin_restored_after_witness'] is True, 'Witness final restoration failed')
    return dict(passed=True, small_step_ascent=True, central_difference_agrees=True,
                minimum_relative_derivative_error=min(row['relative_derivative_error'] for row in witness['steps']))


def finite_difference_witness(runner, direction, gradient_meta, directory):
    from rq2_common import atomic_json, evaluate, set_point
    path = directory / 'finite_difference_witness.json'
    if path.exists():
        witness = read(path)
        require(witness['gradient_manifest_sha256'] == sha(directory / 'gradient_manifest.json'), 'Witness gradient manifest changed')
        return witness, validate_witness(witness, gradient_meta)
    runner.restore()
    origin = evaluate(runner.model, runner.x['train'], runner.y['train'], runner.device, runner.eval_batch)['loss']
    close(origin, gradient_meta['objective_at_anchor'], 'Full-gradient objective CE', atol=3e-6, rtol=1e-7)
    steps = []
    for h in WITNESS_STEPS:
        losses = {}
        for sign in (-1, 1):
            try:
                set_point(runner.params, runner.anchor, direction, sign * h)
                losses[sign] = evaluate(runner.model, runner.x['train'], runner.y['train'], runner.device, runner.eval_batch)['loss']
            finally:
                runner.restore()
        derivative = (losses[1] - losses[-1]) / (2 * h)
        steps.append(dict(h=h, loss_minus=losses[-1], loss_plus=losses[1], central_derivative=derivative,
                          forward_derivative=(losses[1] - origin) / h,
                          relative_derivative_error=abs(derivative - gradient_meta['raw_grad_norm']) / gradient_meta['raw_grad_norm'],
                          anchor_restored_exactly=True))
    runner.check_anchor()
    witness = dict(gradient_task=0, direction_sign=1, samples=1259, direction_sha256=gradient_meta['direction_sha256'],
        gradient_manifest_sha256=sha(directory / 'gradient_manifest.json'), origin_loss=origin, steps=steps,
        origin_restored_after_witness=True)
    result = validate_witness(witness, gradient_meta)
    atomic_json(path, witness)
    return witness, result


def summarize_point(row, origin):
    item = dict(radius=row['point']['radius'], directions=1)
    for split in ('train', 'test'):
        value, baseline = row['metrics'][split], origin['metrics'][split]
        signed = value['loss'] - baseline['loss']
        item[split] = dict(D_mean=signed, G=max(signed, 0.), Q95=max(signed, 0.), sampled_max=max(signed, 0.),
            D_each=[signed], absolute_loss_mean=value['loss'], accuracy_mean=value['accuracy'],
            accuracy_drop_mean_pp=baseline['accuracy'] - value['accuracy'])
    fisher = row['fisher_by_scope']['subset']
    item['raw_trace_by_scope'] = {'subset': dict(mean=fisher['raw_trace'], Q95=fisher['raw_trace'],
        sampled_max=fisher['raw_trace'], each=[fisher['raw_trace']], samples=fisher['samples'])}
    return item


def validate_finished(summary, meta, rows, reference, gradient_meta, witness):
    support.verify_protocol(meta)
    finite_tree(summary, 'summary')
    require(summary['status'] == 'complete' and summary['job'] == meta['job'] == 'fixed_gradient', 'Job not complete')
    require(summary['gradient_task'] == meta['gradient_task'] == 0 and summary['direction_sign'] == meta['direction_sign'] == 1, 'Task0 ascent sign changed')
    require(summary['seed'] == meta['seed'] == reference['meta']['seed'] and summary['method'] == meta['method'] == reference['meta']['method'], 'Run identity changed')
    require(summary['protocol_sha256'] == meta['protocol_sha256'], 'Summary/manifest protocol mismatch')
    expected = {'r0'} | {f'r{r:.12g}_gradient' for r in RADII if r}
    require(len(rows) == summary['states_completed'] == summary['expected_states'] == len(expected), 'Incomplete radius coverage')
    require({row['state_id'] for row in rows} == expected, 'Duplicate/missing fixed-gradient states')
    require(meta['radii'] == RADII and summary['radii'] == RADII, 'Radius grid changed')
    require(summary['gradient_scope'] == ['subset'] and summary['directions_per_radius'] == 1, 'Measurement scope changed')
    require(summary['direction_sha256'] == meta['fixed_gradient']['direction_sha256'] == gradient_meta['direction_sha256'], 'Fixed direction identity changed')
    direction_check = validate_direction_scalars(gradient_meta)
    witness_check = validate_witness(witness, gradient_meta)
    by_id = {row['state_id']: row for row in rows}
    origin = by_id['r0']
    origin_check = support.validate_origin(origin, reference, 'sparse_train')
    for row in rows:
        finite_tree(row, row['state_id'])
        require(row['seed'] == meta['seed'] and row['method'] == meta['method'] and row['protocol_sha256'] == meta['protocol_sha256'], 'Raw row identity mismatch')
        require(row['anchor_restored_exactly'] is True, 'Raw point anchor restoration failed')
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == meta[key], f'Raw row {key} mismatch')
        require(row['checkpoint_sha256'] == meta['checkpoint_sha256'][meta['method']], 'Checkpoint mismatch')
        r = row['point']['radius']
        require(r in RADII and row['amount'] == r, 'Radius/amount changed')
        close(row['effective_weight_drift'], r, 'Realized fixed-gradient drift', atol=1e-5, rtol=2e-6)
        numerical_equal(row['effective_weight_drift_squared'], row['effective_weight_drift'] ** 2, 'Squared drift')
        if r:
            require(row['state_id'] == f'r{r:.12g}_gradient' and row['point']['kind'] == 'fixed_gradient', 'Point state ID mismatch')
            require(row['point']['direction'] == meta['direction_record'], 'Direction changed across radii')
        else:
            require(row['state_id'] == 'r0' and row['point']['kind'] == 'anchor', 'Malformed anchor')
        for split, samples in [('train', 1259), ('test', 307)]:
            value = row['metrics'][split]
            require(value['samples'] == samples and value['loss'] >= 0 and 0 <= value['accuracy'] <= 100, 'Invalid evaluation metric')
        require(set(row['fisher_by_scope']) == {'subset'}, 'Unexpected Fisher scope')
        fisher = row['fisher_by_scope']['subset']
        baseline = origin['fisher_by_scope']['subset']
        for key in ('samples', 'normalized', 'floor', 'regularizer', 'estimator', 'coordinate', 'gradient_batch_size'):
            require(fisher[key] == baseline[key], f'Fisher metadata changed: {key}')
        require(fisher['samples'] == 256 and fisher['normalized'] is False and fisher['floor'] is None and fisher['regularizer'] is None and fisher['raw_trace'] >= 0 and fisher['q'] == {}, 'Invalid or scaled Fisher estimate')
    expected_summary = [summarize_point(by_id['r0' if r == 0 else f'r{r:.12g}_gradient'], origin) for r in RADII]
    require(summary['radius_summary'] == expected_summary, 'Summary differs from raw-state recomputation')
    require(summary['end_restoration']['effective_sha256'] == meta['anchor_effective_sha256'], 'Final effective anchor changed')
    require(summary['frozen_state_sha256_before'] == summary['frozen_state_sha256_after'] and summary['fixed_state_restored'] is True, 'Fixed backbone/head changed')
    require(summary['direction_sha256_after'] == gradient_meta['direction_sha256'], 'Direction tensors mutated')
    return dict(passed=True, measured_states=len(rows), all_statistics_recomputed=True,
                raw_states_finite=True, direction_constant=True, fixed_backbone_and_head=True,
                direction_check=direction_check, witness_check=witness_check, origin_check=origin_check,
                interpretation='Signed Task0 CE change along one fixed Task0 ascent direction; not future-task learning')


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=('sgd', 'gam'))
    parser.add_argument('--job', default='fixed_gradient', choices=('fixed_gradient',))
    parser.add_argument('--gradient-task', type=int, default=0, choices=(0,))
    parser.add_argument('--path-task', type=int, default=9)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--validate-only', action='store_true')
    return parser


def main():
    args = argument_parser().parse_args()
    _, reference, trajectory, evidence = preflight(args)
    if args.validate_only:
        print(json.dumps(evidence, indent=2, allow_nan=False))
        return
    sys.path.insert(0, str(BASE))
    from diagnostic_full import FullRunner
    from diagnostic import Journal
    from rq2_common import atomic_json, tensor_hash
    loader_args = argparse.Namespace(**vars(args))
    loader_args.job = 'sparse_train'
    runner = FullRunner(loader_args)
    identity = support.validate_loaded_identity(runner, reference, trajectory)
    # FullRunner's original sparse_train manifest/sanity are an isolated loader
    # witness only; no states are measured in that directory.
    require(not runner.journal.rows, 'Loader-only journal unexpectedly contains measurements')
    frozen = [value for name, value in runner.model.state_dict().items()
              if not name.endswith(('.effective_q', '.effective_v'))]
    frozen_hash = tensor_hash(frozen)
    directory = runner.output / 'fixed_gradient'
    direction, gradient_meta = prepare_gradient(runner, directory)
    witness, witness_check = finite_difference_witness(runner, direction, gradient_meta, directory)
    direction_record = dict(kind='positive_full_task0_training_ce_gradient', task=0, direction_sign=1,
        sha256=gradient_meta['direction_sha256'], gradient_sha256=gradient_meta['gradient_sha256'],
        held_fixed_across_radii=True)
    runner.args.job = 'fixed_gradient'
    runner.directory = directory
    runner.meta.pop('protocol_sha256')
    runner.meta.update(schema='rq2_fixed_task0_gradient_v1', job='fixed_gradient', gradient_task=0, direction_sign=1, radii=RADII,
        fixed_gradient=gradient_meta, direction_record=direction_record,
        wrapper_sha256=sha(__file__), gradient_manifest_sha256=sha(directory / 'gradient_manifest.json'),
        gradient_artifact_sha256=sha(directory / 'gradient_direction.pt'),
        witness_sha256=sha(directory / 'finite_difference_witness.json'),
        frozen_state_sha256=frozen_hash)
    runner.protocol = canonical(runner.meta)
    runner.meta['protocol_sha256'] = runner.protocol
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        require(read(manifest_path)['protocol_sha256'] == runner.protocol, 'Fixed-gradient protocol changed; cannot resume')
    atomic_json(manifest_path, runner.meta)
    runner.journal = Journal(directory, runner.protocol)
    origin = runner.origin_full(['subset'])
    origin_check = support.validate_origin(origin, reference, 'sparse_train')
    close(origin['metrics']['train']['loss'], gradient_meta['objective_at_anchor'], 'Gradient/ray shared objective', atol=3e-6, rtol=1e-7)
    for r in RADII:
        if r:
            runner.measure_full(f'r{r:.12g}_gradient', direction, float(r), fisher_scopes=['subset'],
                                kind='fixed_gradient', radius=r, direction=direction_record)
    direction_hash_after = tensor_hash(direction)
    frozen_after = tensor_hash(frozen)
    require(direction_hash_after == gradient_meta['direction_sha256'], 'Fixed direction mutated during measurement')
    require(frozen_after == frozen_hash, 'Frozen backbone/head changed during measurement')
    radius_summary = [summarize_point(runner.journal.rows['r0' if r == 0 else f'r{r:.12g}_gradient'], origin) for r in RADII]
    runner.finish(dict(radius_summary=radius_summary, radii=RADII, expected_states=len(RADII),
        directions_per_radius=1, gradient_scope=['subset'], gradient_task=0, direction_sign=1,
        direction_sha256=gradient_meta['direction_sha256'], direction_sha256_after=direction_hash_after,
        signed_damage_primary=True, positive_damage_also_saved=True,
        frozen_state_sha256_before=frozen_hash, frozen_state_sha256_after=frozen_after,
        fixed_state_restored=True))
    summary_path = directory / 'summary.json'
    summary = read(summary_path)
    rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
    result_check = validate_finished(summary, runner.meta, rows, reference, gradient_meta, witness)
    receipt = dict(status='complete', schema='rq2_fixed_task0_gradient_v1', job='fixed_gradient',
        seed=runner.seed, method=args.method, gradient_task=0, direction_sign=1, samples=256,
        protocol_sha256=runner.protocol, summary=str(summary_path), summary_sha256=sha(summary_path),
        manifest_sha256=sha(manifest_path), states_sha256=sha(directory / 'states.jsonl'),
        gradient_artifact_sha256=sha(directory / 'gradient_direction.pt'),
        gradient_manifest_sha256=sha(directory / 'gradient_manifest.json'),
        sample_manifest_sha256=sha(directory / 'gradient_sample_manifest.json'),
        witness_sha256=sha(directory / 'finite_difference_witness.json'), wrapper_sha256=sha(__file__),
        preflight=evidence, identity_check=identity, origin_check=origin_check,
        witness_check=witness_check, result_check=result_check, endpoint_check=result_check)
    atomic_json(directory / 'extension_receipt.json', receipt)
    print(json.dumps(dict(event='fixed_gradient_complete', receipt=str(directory / 'extension_receipt.json'))), flush=True)


if __name__ == '__main__':
    main()
