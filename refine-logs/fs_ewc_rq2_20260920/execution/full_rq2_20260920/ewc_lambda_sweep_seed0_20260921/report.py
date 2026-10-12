#!/usr/bin/env python3
"""Strict CPU-only report for the seed0 SGD/raw-Fisher lambda sweep."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
GRID = [0, 1, 10, 30, 60, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
FISHER_POLICY = 'recomputed_native_task0_raw_fisher_no_trace_normalization'
METRICS = ['a11', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont',
           'raw_fisher_weighted_drift_squared', 'CE_mean', 'P_mean', 'P_over_CE',
           'grad_P_over_grad_CE', 'grad_CE_norm_mean', 'grad_P_norm_mean',
           'endpoint_P', 'delta_a21_pp', 'delta_a22_pp', 'delta_average_pp']
PAIR_FIELDS = ('generator_seed', 'num_workers', 'dataset_ids_sha256', 'sample_count')
ITER_FIELDS = PAIR_FIELDS + ('indices_sha256', 'first_batch_pixels_targets_sha256', 'samples', 'batches')
COMMON_CONFIG = ('dataset', 'class_shuffle', 'lora_rank', 'init_cls', 'increment', 'batch_size',
                 'epochs', 'lrate', 'momentum', 'weight_decay', 'scheduler', 'ewc_gamma', 'ewc_eta',
                 'ewc_max_batches', 'fisher_mode', 'ewc_fisher_source', 'fisher_split',
                 'train_num_workers', 'eval_num_workers', 'persistent_workers')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, label):
    result = float(value)
    require(math.isfinite(result), f'Nonfinite {label}')
    return result


def read_json(path, *, metric_matrices=False):
    def bad(value):
        raise ValueError(f'Nonfinite JSON in {path}: {value}')
    # Native matrices intentionally use NaN above the evaluated lower triangle.
    # Only metrics files permit this placeholder; reported cells are checked finite.
    return json.loads(Path(path).read_text(), **({} if metric_matrices else {'parse_constant': bad}))


def jsonl(path):
    if not Path(path).is_file():
        return []
    result = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Assets:
    def __init__(self):
        self.cache = {}

    def hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            self.cache[key] = sha(path)
        return self.cache[key]

    def verify(self, path, digest, label):
        require(self.hash(path) == digest, f'{label} changed: {path}')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_spec(spec):
    require(spec['lambdas'] == spec['new_lambdas'] == GRID and spec['reused_lambdas'] == [],
            'This report requires all 14 newly trained raw-Fisher points, including lambda 0')
    require(spec['seed'] == 0 and spec['anchor_method'] == spec['future_optimizer'] == 'sgd',
            'Report is pinned to seed0 SGD to SGD')
    require(spec['ewc_normalize_fisher'] is False and spec['ewc_normalize_accumulated_fisher'] is False,
            'Raw sweep cannot use trace-normalized Fisher')
    require(spec['efficacy_thresholds'] == dict(retention_gain_pp=1.0, max_new_task_drop_pp=5.0,
                                              nondecreasing_average_gain_pp=0.0),
            'Predefined efficacy thresholds changed')


def validate_prepared(root, spec, assets):
    path = Path(spec['prepared_fisher']).resolve()
    require(path == (root / 'prepared_fisher/fisher_state_raw.pt').resolve(), 'Raw Fisher must belong to this sweep')
    receipt_path = path.with_name('PREPARED.json')
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    expected = dict(status='complete', seed=0, task=0, fisher_mode='batch_empirical', fisher_source='fit',
                    ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                    tensor_count=24, numel=14155776, weights_unchanged=True)
    require(all(receipt.get(k) == v for k, v in expected.items()), 'Prepared raw Fisher policy mismatch')
    require(receipt.get('exact_inputs_receipt', {}).get('matched') is True, 'Prepared Fisher input pairing missing')
    anchor = Path(spec['shared_anchor']).resolve()
    require(anchor == (root.parent / 'formal_training/anchor_sgd_seed0').resolve(), 'Unexpected Task0 parent')
    parent_checkpoint = anchor / 'checkpoints/task_00.pt'
    require(Path(receipt['parent_checkpoint_path']).resolve() == parent_checkpoint, 'Prepared Fisher parent path mismatch')
    require(Path(receipt['payload_path']).resolve() == path, 'Prepared Fisher payload path mismatch')
    assets.verify(path, receipt['payload_sha256'], 'Prepared raw Fisher payload')
    assets.verify(parent_checkpoint, receipt['parent_checkpoint_sha256'], 'Common Task0 checkpoint')
    assets.verify(root.parent.parent / 'source_frozen.json', receipt['source_manifest_sha256'], 'Frozen source manifest')
    require(numeric(receipt['raw_trace'], 'raw trace') > 0, 'Raw trace must be positive')
    require(math.isclose(numeric(receipt['mean'], 'raw mean'), receipt['raw_trace'] / receipt['numel'], rel_tol=1e-8),
            'Raw Fisher mean/trace mismatch')
    parent_complete = read_json(anchor / 'COMPLETE.json')
    parent_manifest = read_json(anchor / 'run_manifest.json')
    require(parent_complete['status'] == 'complete' and parent_complete['completed_tasks'] == 1,
            'Common Task0 parent must be complete')
    require(parent_complete['identity_sha256'] == parent_manifest['identity_sha256'], 'Task0 identity mismatch')
    assets.verify(parent_checkpoint, parent_complete['validation']['checkpoint_assets']['checkpoints/task_00.pt']['sha256'],
                  'Completed Task0 checkpoint')
    metrics_path = Path(parent_complete['validation']['metrics'])
    assets.verify(metrics_path, parent_complete['validation']['metrics_sha256'], 'Task0 metrics')
    return dict(receipt=receipt, receipt_path=str(receipt_path), receipt_sha256=assets.hash(receipt_path),
                payload_path=str(path), payload_sha256=receipt['payload_sha256'],
                parent_checkpoint=str(parent_checkpoint), parent_checkpoint_sha256=receipt['parent_checkpoint_sha256'],
                anchor=str(anchor), a11=numeric(read_json(metrics_path, metric_matrices=True)['cnn']['matrices']['final'][0][0], 'Task0 a11'))


def validate_prepared_identity(identity, prepared, label):
    require(identity.get('status') == 'verified' and identity.get('execution_ready') is True,
            f'{label}: prepared Fisher is not verified')
    for key in ('payload_sha256', 'receipt_sha256', 'parent_checkpoint_sha256'):
        require(identity.get(key) == prepared[key], f'{label}: shared raw Fisher {key} mismatch')
    require(Path(identity['path']).resolve() == Path(prepared['payload_path']) and
            Path(identity['receipt_path']).resolve() == Path(prepared['receipt_path']), f'{label}: raw Fisher path mismatch')
    require(identity.get('fisher_policy') == FISHER_POLICY, f'{label}: raw Fisher policy mismatch')


def validate_pairing(run, complete, assets, reference_path):
    reference = jsonl(reference_path)
    reference_complete = read_json(reference_path.parent / 'COMPLETE.json')
    assets.verify(reference_path, reference_complete['validation']['rng_receipts_sha256'], 'Formal pairing reference')
    observed = jsonl(run / 'rng_receipts.jsonl')
    def select(rows, event):
        return [r for r in rows if r.get('task') == 1 and r.get('event') == event
                and (event not in ('loader_created', 'loader_iteration') or r.get('shuffle') is True)]
    for event, fields in [('head_initialized', ('old_classes', 'total_classes', 'task_seed', 'new_head_sha256')),
                          ('task_rng_reset', ('task_seed', 'rng_protocol', 'optimizer_type')),
                          ('loader_created', PAIR_FIELDS)]:
        got, ref = select(observed, event), select(reference, event)
        require(len(got) == len(ref) == 1, f'{run.name}: missing or duplicated {event}')
        require(all(got[0].get(k) == ref[0].get(k) and got[0].get(k) is not None for k in fields),
                f'{run.name}: Task1 {event} pairing mismatch')
    iterations, ref_iters = select(observed, 'loader_iteration'), select(reference, 'loader_iteration')
    require([r['iteration'] for r in iterations] == list(range(len(iterations))), f'{run.name}: duplicated/missing loader pass')
    require(len(iterations) <= 51 and (not complete or len(iterations) == 51), f'{run.name}: loader pass count mismatch')
    full_passes = partial_passes = 0
    for index, row in enumerate(iterations):
        ref = ref_iters[index]
        full = row['samples'] == row['sample_count']
        fields = ITER_FIELDS if full else PAIR_FIELDS + ('first_batch_pixels_targets_sha256',)
        require(all(row.get(k) == ref.get(k) and row.get(k) is not None for k in fields),
                f'{run.name}: Task1 loader pass {index} pairing mismatch')
        if full:
            full_passes += 1
        else:
            require(not complete and index == len(iterations) - 1 and 0 < row['samples'] < row['sample_count'],
                    f'{run.name}: invalid incomplete loader pass')
            partial_passes += 1
    require(iterations or not complete, f'{run.name}: no training-input evidence')
    return dict(passed=True, reference=str(reference_path), reference_sha256=assets.hash(reference_path),
                new_head_sha256=select(observed, 'head_initialized')[0]['new_head_sha256'],
                full_passes=full_passes, partial_passes=partial_passes,
                scope='All completed pass indices and first augmented batch; partial pass only sampler metadata and first batch; not every augmentation')


def drift_from_states(old, new, fisher):
    """Continuous 24 Q/V BA displacement; no classifier or A/B-factor distance."""
    import torch
    require(set(old) == set(new), 'Checkpoint model-state keys changed')
    require(old['fc.weight'].shape == (20, 768) and new['fc.weight'].shape == (40, 768), 'Classifier shape mismatch')
    for key in ('fc.weight', 'fc.bias'):
        require(torch.equal(old[key], new[key][:20]), 'Old classifier changed')
    for key, value in old.items():
        if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
            require(torch.equal(value, new[key]), f'Frozen backbone changed: {key}')
    expected = {f'backbone.lora_vit.blocks.{i}.attn.qkv.{role}' for i in range(12) for role in ('q', 'v')}
    require(set(fisher) == expected, 'Raw Fisher must cover exactly 24 Q/V blocks')
    drift_sq = weighted = 0.0
    for key in sorted(expected):
        prefix, role = key.rsplit('.', 1)
        a, b = prefix + f'.linear_a_{role}.weight', prefix + f'.linear_b_{role}.weight'
        require(old[a].shape == new[a].shape == (10, 768) and old[b].shape == new[b].shape == (768, 10),
                f'Unexpected LoRA shapes: {key}')
        require(fisher[key].shape == (768, 768) and bool(torch.isfinite(fisher[key]).all())
                and bool((fisher[key] >= 0).all()), f'Invalid raw Fisher tensor: {key}')
        # Native FP32 BA products, then FP64 reductions; mirrors existing report.
        difference = (new[b] @ new[a] - old[b] @ old[a]).double()
        require(bool(torch.isfinite(difference).all()), f'Nonfinite effective displacement: {key}')
        squared = difference.square()
        drift_sq += float(squared.sum())
        weighted += float((fisher[key].double() * squared).sum())
    return dict(d_QV_cont=math.sqrt(drift_sq), raw_fisher_weighted_drift_squared=weighted)


def telemetry(run, lam, complete, validation):
    rows = jsonl(run / 'training_batch_metrics.jsonl')
    require([r['step'] for r in rows] == list(range(len(rows))), f'{run.name}: batch steps missing or duplicated')
    require(not complete or len(rows) == 400, f'{run.name}: formal training requires 400 finite update records')
    for row in rows:
        require(row['task'] == 1 and row['ewc_lambda'] == lam and row['parameters_finite_after_step'] is True,
                f'{run.name}: invalid finite-step receipt')
        for key in ('ce_clean', 'ewc', 'loss', 'grad_combined_norm'):
            numeric(row[key], f'{run.name}/{row["step"]}/{key}')
        require(row['ce_clean'] >= 0 and row['ewc'] >= 0, 'Negative CE or EWC penalty')
        require(row['gradient_components_sampled'] is (row['step'] == 1 or row['step'] % 8 == 0),
                'Gradient sampling schedule changed')
    sampled = [r for r in rows if r['gradient_components_sampled']]
    result = dict(observed_finite_steps=len(rows), gradient_component_samples=len(sampled),
                  telemetry_scope='all 400 completed steps' if complete else 'finite prefix before divergence')
    if rows:
        ce, penalty = statistics.mean(r['ce_clean'] for r in rows), statistics.mean(r['ewc'] for r in rows)
        result.update(CE_mean=ce, P_mean=penalty, P_over_CE=penalty / ce if ce else None)
    if sampled:
        for row in sampled:
            for key in ('grad_clean_norm', 'grad_fisher_norm', 'fisher_to_clean_norm_ratio'):
                numeric(row[key], key)
            require(row['grad_clean_norm'] >= 0 and row['grad_fisher_norm'] >= 0, 'Negative gradient norm')
            require(math.isclose(row['fisher_to_clean_norm_ratio'], row['grad_fisher_norm'] / (row['grad_clean_norm'] + 1e-12),
                                 rel_tol=1e-9, abs_tol=1e-12), 'Gradient ratio does not match component norms')
        result.update(grad_CE_norm_mean=statistics.mean(r['grad_clean_norm'] for r in sampled),
                      grad_P_norm_mean=statistics.mean(r['grad_fisher_norm'] for r in sampled),
                      grad_P_over_grad_CE=statistics.mean(r['fisher_to_clean_norm_ratio'] for r in sampled))
    if complete:
        for output, key in [('CE_mean', 'train_clean_loss_mean'), ('P_mean', 'train_fisher_loss_mean')]:
            require(math.isclose(result[output], numeric(validation['losses'][key], key), rel_tol=1e-7, abs_tol=1e-10),
                    f'{run.name}: aggregate {key} differs from batch telemetry')
        for output, key in [('grad_CE_norm_mean', 'grad_clean_norm_mean'), ('grad_P_norm_mean', 'grad_fisher_norm_mean'),
                            ('grad_P_over_grad_CE', 'fisher_to_clean_norm_ratio_mean')]:
            require(math.isclose(result[output], numeric(validation['gradients'][key], key), rel_tol=1e-9, abs_tol=1e-12),
                    f'{run.name}: aggregate {key} differs from batch telemetry')
        require(validation['gradients']['component_sample_count'] == len(sampled), 'Gradient sample count changed')
    return result


def validate_trial(root, lam, prepared, assets, tensor_cache):
    run = root / 'training' / f'lambda_{lam}_seed0'
    row = dict(ewc_lambda=lam, status='pending', outcome=None, **{k: None for k in METRICS}, run_dir=str(run))
    result_path = run / 'TRIAL_RESULT.json'
    if not result_path.is_file():
        row.update(status='failed' if (run / 'FAILED.json').exists() else 'pending',
                   note='No validated observed trial receipt; native checkpoints alone do not count')
        return row, None
    require(prepared is not None, f'{run.name}: observed trial lacks common prepared Fisher')
    trial = read_json(result_path)
    outcome = trial.get('outcome')
    require(trial.get('status') == 'observed' and outcome in ('complete', 'diverged') and
            trial.get('seed') == 0 and trial.get('ewc_lambda') == lam, f'{run.name}: invalid trial receipt')
    complete = outcome == 'complete'
    native_path = run / ('COMPLETE.json' if complete else 'DIVERGED.json')
    manifest_path = run / 'run_manifest.json'
    require(Path(trial['native_receipt']).resolve() == native_path and Path(trial['manifest']).resolve() == manifest_path,
            f'{run.name}: receipt paths must identify this freshly trained point')
    require(not (run / ('DIVERGED.json' if complete else 'COMPLETE.json')).exists(), 'Conflicting native terminal receipts')
    require(not (run / 'FAILED.json').exists(), 'Unexpected failure cannot be classified as an observed trial')
    assets.verify(native_path, trial['native_receipt_sha256'], 'Native terminal receipt')
    assets.verify(manifest_path, trial['manifest_sha256'], 'Trial manifest')
    native, manifest = read_json(native_path), read_json(manifest_path)
    require(native['status'] == outcome and trial['native_exit_code'] == (0 if complete else 86), 'Native outcome/exit mismatch')
    require(trial['identity_sha256'] == native['identity_sha256'] == manifest['identity_sha256'], 'Trial identity mismatch')
    require(manifest['seed'] == 0 and manifest['anchor_method'] == manifest['future_optimizer'] == 'sgd'
            and manifest['smoke_only'] is False and manifest['task0_shaping'] is False and manifest['task1_shaping'] is False,
            'Formal seed0 SGD to SGD manifest required')
    require(manifest['matmul_allow_tf32'] is False and manifest['cudnn_allow_tf32'] is False, 'FP32 precision flags changed')
    assets.verify(root / 'train.py', manifest['runner_sha256'], 'Trial trainer')
    require(manifest['source_manifest_sha256'] == prepared['receipt']['source_manifest_sha256'], 'Trial source differs from Fisher source')
    cfg = manifest['requested_config']
    expected = dict(optimizer_type='sgd', lambda_flat=0.0, ewc_lambda=lam, epochs=50, batch_size=128,
                    lrate=.02, momentum=0, weight_decay=0.0, max_train_tasks=2, resume_from_task=0,
                    ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                    fisher_mode='batch_empirical', ewc_fisher_source='fit', ewc_eta=0.0,
                    approximate_fisher_in_task=False, ewc_dual_ascent=False, ewc_project_grad=False,
                    delta_w_penalty=False, rq2_smoke_only=False, rq2_fisher_policy=FISHER_POLICY)
    for config in (cfg, manifest['config']):
        require(all(config.get(k) == v for k, v in expected.items()), f'{run.name}: scientific config mismatch')
    parent = manifest['parent']
    require(Path(parent['task_00.pt']['path']).resolve() == Path(prepared['parent_checkpoint']) and
            parent['task_00.pt']['sha256'] == prepared['parent_checkpoint_sha256'], 'Different Task0 start')
    for name, asset in parent.items():
        require(Path(asset['path']).resolve().is_relative_to(Path(prepared['anchor'])), f'Unexpected parent asset: {name}')
        assets.verify(asset['path'], asset['sha256'], f'Task0 {name}')
    validate_prepared_identity(manifest['prepared_fisher'], prepared, run.name)
    runtime = jsonl(run / 'runtime_receipts.jsonl')
    restored = [e for e in runtime if e.get('event') == 'complete_resume' and e.get('task') == 0]
    replaced = [e for e in runtime if e.get('event') == 'prepared_raw_fisher_loaded' and e.get('task') == 0]
    require(len(restored) == len(replaced) == 1 and restored[0].get('verified_exact') is True and
            restored[0]['checkpoint_sha256'] == prepared['parent_checkpoint_sha256'], 'Exact Task0 restoration missing')
    require(replaced[0].get('verified_exact') is True and replaced[0].get('replaced_original_normalized_fisher') is True
            and replaced[0].get('ewc_normalize_fisher') is False and replaced[0].get('ewc_normalize_accumulated_fisher') is False,
            'Raw Fisher runtime replacement not verified')
    validate_prepared_identity(replaced[0]['prepared_fisher'], prepared, run.name + '/runtime')
    reference_path = root.parent / 'formal_training/sequence_sgd_seed0/rng_receipts.jsonl'
    pairing = validate_pairing(run, complete, assets, reference_path)
    validation = native.get('validation', {})
    row.update(status='observed', outcome=outcome, trial_receipt_sha256=assets.hash(result_path),
               native_receipt_sha256=trial['native_receipt_sha256'], manifest_sha256=trial['manifest_sha256'],
               raw_fisher_sha256=prepared['payload_sha256'], parent_checkpoint_sha256=prepared['parent_checkpoint_sha256'],
               pairing=pairing, **telemetry(run, lam, complete, validation))
    if not complete:
        require(native.get('smoke_only') is False and native.get('fisher_policy') == FISHER_POLICY and
                native.get('hyperparameters_changed') is False, 'Unclassified divergence receipt')
        require(native['last_completed_training_steps'] == row['observed_finite_steps'], 'Divergence finite-step count mismatch')
        row['note'] = native['reason']
    else:
        require(native['completed_tasks'] == 2 and native['smoke_only'] is False and
                validation['epochs_per_new_task'] == 50 and validation['validated_tasks'] == [1] and
                validation['smoke_only'] is False and validation['fisher_policy'] == FISHER_POLICY,
                'Native COMPLETE is not formal Task1 training')
        validate_prepared_identity(validation['prepared_fisher'], prepared, run.name + '/completion')
        for filename, field in [('rng_receipts.jsonl', 'rng_receipts_sha256'),
                                ('runtime_receipts.jsonl', 'runtime_receipts_sha256'),
                                ('training_batch_metrics.jsonl', 'training_batch_metrics_sha256')]:
            assets.verify(run / filename, validation[field], filename)
        metrics_path = Path(validation['metrics'])
        require(metrics_path.resolve().is_relative_to(run / 'trainer_outputs'), 'Metrics do not belong to this point')
        assets.verify(metrics_path, validation['metrics_sha256'], 'Task1 CIL metrics')
        metrics = read_json(metrics_path, metric_matrices=True)
        matrix = metrics['cnn']['matrices']['final']
        require(len(matrix) == 2 and all(len(r) == 2 for r in matrix), 'Final two-task matrix missing')
        t01 = metrics['cnn']['matrices']['t01']
        require(len(t01) == 2 and all(len(r) == 2 for r in t01) and
                all(t01[i][j] == matrix[i][j] for i in range(2) for j in range(i + 1)), 'Final matrix differs from t01')
        accuracy = dict(a11=numeric(matrix[0][0], 'a11'), a21=numeric(matrix[1][0], 'a21'), a22=numeric(matrix[1][1], 'a22'))
        require(accuracy == validation['accuracy_CIL'] and accuracy['a11'] == prepared['a11'] and
                all(0 <= v <= 100 for v in accuracy.values()), 'Accuracy or shared Task0 metric mismatch')
        checkpoint_path = run / 'checkpoints/task_01.pt'
        for relative, asset in validation['checkpoint_assets'].items():
            require(Path(relative).parts[0] == 'checkpoints' and '..' not in Path(relative).parts, 'Invalid checkpoint asset path')
            assets.verify(run / relative, asset['sha256'], 'Completed checkpoint asset')
        require('checkpoints/task_01.pt' in validation['checkpoint_assets'], 'Endpoint checkpoint is not certified')
        import torch
        torch.set_num_threads(4)
        if not tensor_cache:
            tensor_cache['old'] = torch.load(prepared['parent_checkpoint'], map_location='cpu', weights_only=True, mmap=True)['network_state']
            payload = torch.load(prepared['payload_path'], map_location='cpu', weights_only=True, mmap=True)
            tensor_cache['fisher'] = payload['fisher_past_delta']
            trace = sum(float(t.double().sum()) for t in tensor_cache['fisher'].values())
            require(math.isclose(trace, prepared['receipt']['raw_trace'], rel_tol=1e-8), 'Raw payload trace mismatch')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        require(checkpoint['cur_task'] == 1 and checkpoint['total_classes'] == 40 and checkpoint['optimizer_type'] == 'sgd',
                'Endpoint checkpoint identity mismatch')
        drift = drift_from_states(tensor_cache['old'], checkpoint['network_state'], tensor_cache['fisher'])
        require(math.isclose(drift['d_QV_cont'], numeric(validation['effective_weight_displacement'], 'recorded drift'),
                             rel_tol=1e-5, abs_tol=1e-8), 'CPU BA displacement differs from native endpoint witness')
        row.update(**accuracy, **drift, BWT=accuracy['a21'] - accuracy['a11'],
                   average_accuracy=(accuracy['a21'] + accuracy['a22']) / 2,
                   endpoint_P=.5 * lam * drift['raw_fisher_weighted_drift_squared'])
    return row, {k: cfg.get(k) for k in COMMON_CONFIG}


def summarize(rows, thresholds):
    completed = [r for r in rows if r['outcome'] == 'complete']
    baseline = next((r for r in completed if r['ewc_lambda'] == 0), None)
    result = dict(baseline_complete=baseline is not None, best_old_retention=[], best_average=[],
                  observable_retention_lambdas=[], balanced_retention_lambdas=[])
    if completed:
        for field, key in [('a21', 'best_old_retention'), ('average_accuracy', 'best_average')]:
            best = max(r[field] for r in completed)
            result[key] = [r['ewc_lambda'] for r in completed if math.isclose(r[field], best, abs_tol=1e-12)]
    if baseline:
        for row in completed:
            row.update(delta_a21_pp=row['a21'] - baseline['a21'], delta_a22_pp=row['a22'] - baseline['a22'],
                       delta_average_pp=row['average_accuracy'] - baseline['average_accuracy'])
            if row['ewc_lambda'] > 0 and row['delta_a21_pp'] >= thresholds['retention_gain_pp'] - 1e-12:
                result['observable_retention_lambdas'].append(row['ewc_lambda'])
                if (row['delta_a22_pp'] >= -thresholds['max_new_task_drop_pp'] - 1e-12 and
                        row['delta_average_pp'] >= thresholds['nondecreasing_average_gain_pp'] - 1e-12):
                    result['balanced_retention_lambdas'].append(row['ewc_lambda'])
    for key in ('observable_retention_lambdas', 'balanced_retention_lambdas'):
        values = result[key]
        result[key + '_observed_span'] = [min(values), max(values)] if values else None
    return result


def render_markdown(data):
    def fmt(value):
        return '—' if value is None else f'{value:.6g}' if isinstance(value, (int, float)) else str(value).replace('|', '/')
    lines = ['# Raw Fisher λ 扫描（seed0，SGD → SGD）', '',
             f"状态：{data['status']}；已观察 {data['observed_lambda_points']}/14 点，成功完成 {data['completed_lambda_points']} 点，科学发散 {data['diverged_lambda_points']} 点。", '',
             '原 Task0 模型保持一致，重新计算一次 native fit batch-empirical Fisher；关闭两处 trace normalization。',
             '以下仅为单 seed、在测试集上探索选择的结果，不是跨 seed 确证。空格/破折号不是零，也不表示训练完成。', '',
             '| λ | 终态 | a11 | a21 | a22 | BWT | 平均准确率 | d_QV_cont | Σ rawF·ΔBA² | CE mean | P mean | P/CE | gradP/gradCE |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    fields = ['ewc_lambda', 'outcome', 'a11', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont',
              'raw_fisher_weighted_drift_squared', 'CE_mean', 'P_mean', 'P_over_CE', 'grad_P_over_grad_CE']
    for row in data['individual']:
        values = [row.get(key) if key != 'outcome' else row.get('outcome') or row['status'] for key in fields]
        lines.append('| ' + ' | '.join(map(fmt, values)) + ' |')
    ranking = data['selection']
    lines += ['', f"已完成点中，旧任务保持最高 λ：{ranking['best_old_retention']}；两任务平均最高 λ：{ranking['best_average']}。", '',
              '预设可观测保护阈值：相对新跑 λ=0，a21 提升至少 1 个百分点。折中条件额外要求 a22 降幅不超过 5 个百分点，且平均准确率不下降。',
              f"达到可观测保护阈值的实测 λ：{ranking['observable_retention_lambdas']}；实测跨度：{ranking['observable_retention_lambdas_observed_span']}。",
              f"同时满足折中条件的实测 λ：{ranking['balanced_retention_lambdas']}；实测跨度：{ranking['balanced_retention_lambdas_observed_span']}。",
              '跨度仅说明满足条件的离散观测最小/最大值，不对中间未测 λ 或缺失/发散点插值；λ=0 未完成时不判断相对效力。', '',
              '| λ | Δa21 (pp) | Δa22 (pp) | Δ平均 (pp) | 有限更新数 | 梯度分量样本数 |',
              '|---|---|---|---|---|---|']
    for row in data['individual']:
        lines.append('| ' + ' | '.join(fmt(row.get(k)) for k in ['ewc_lambda', 'delta_a21_pp', 'delta_a22_pp',
                     'delta_average_pp', 'observed_finite_steps', 'gradient_component_samples']) + ' |')
    lines += ['', '数值口径：', '',
              '- a11/a21/a22 使用训练器记录的 CIL 矩阵精度，单位为百分比；BWT=a21−a11，平均=(a21+a22)/2。',
              '- d_QV_cont 为延续同一 LoRA 分支的 24 个 Q/V 有效 BA 从 Task0 到 Task1 的 Frobenius 位移，排除分类头。rawF 加权位移平方为 Σ rawF·(BA1−BA0)²；端点 P=λ/2 倍该值，另存于 CSV/JSON。',
              '- P mean 已含 λ/2 系数。CE mean 和 P mean 按实际更新批次求均值；P/CE 是这两个均值之比。',
              '- gradP/gradCE 为零起始 steps 0、1、8、16… 更新前梯度范数比的均值，分母包括实际优化器全部可训练参数的 CE 梯度（包含新分类头），P 只作用于 Q/V。不是所有步的比值，也不是两个均值之比。',
              '- 发散点的准确率、端点位移与正式排名留空；如有有限前缀，损失及梯度只汇总发散前已完成的有限更新，不能与完整训练均值直接等同。',
              '- pairing 核验共同 Task0、raw Fisher 文件哈希、Task1 新分类头和各完成遍历的样本顺序/首批增强；未完成遍历只核验采样器信息及首批，不声称验证了每个增强样本。', '',
              '历史参考单列保存在 results.json/historical_references_not_raw_points：旧 trace-normalized λ=2000 没有进入 raw Fisher 网格、baseline、排名或效力范围。', '',
              '各点状态说明：', '']
    for row in data['individual']:
        if row.get('note'):
            lines.append(f"- λ={row['ewc_lambda']}：{row['note']}")
    return '\n'.join(lines) + '\n'


def build_report(root, output, allow_partial=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and output != root, 'Report outputs must remain in a new subdirectory of this sweep')
    require(not any(output.is_relative_to(root / name) for name in ('training', 'prepared_fisher', 'before_raw_fisher')),
            'Report cannot overwrite experiment data or historical snapshots')
    spec_path = root / 'sweep_spec.json'
    spec = read_json(spec_path)
    validate_spec(spec)
    assets, tensor_cache = Assets(), {}
    prepared = validate_prepared(root, spec, assets)
    rows, scientific = [], None
    for lam in GRID:
        row, config = validate_trial(root, lam, prepared, assets, tensor_cache)
        if config is not None:
            require(scientific is None or scientific == config, 'Scientific settings changed between lambda points')
            scientific = config
        rows.append(row)
    selection = summarize(rows, spec['efficacy_thresholds'])
    observed = sum(r['status'] == 'observed' for r in rows)
    complete = observed == len(GRID) and selection['baseline_complete']
    data = dict(status='complete' if complete else 'partial', generated_utc=datetime.now(timezone.utc).isoformat(),
                seed=0, grid=GRID, observed_lambda_points=observed,
                completed_lambda_points=sum(r['outcome'] == 'complete' for r in rows),
                diverged_lambda_points=sum(r['outcome'] == 'diverged' for r in rows),
                missing_lambda_points=[r['ewc_lambda'] for r in rows if r['status'] != 'observed'],
                prepared_fisher=prepared, efficacy_thresholds=spec['efficacy_thresholds'], selection=selection,
                individual=rows, shared_scientific_config=scientific, gpu_evaluation_performed=False,
                reporter_sha256=assets.hash(__file__), spec_sha256=assets.hash(spec_path),
                historical_references_not_raw_points=spec.get('historical_references_not_raw_points', {}))
    output.mkdir(parents=True, exist_ok=True)
    # A partial regeneration must not leave a stale success receipt beside it.
    (output / 'COMPLETE.json').unlink(missing_ok=True)
    atomic_json(output / 'results.json', data)
    flat_fields = ['ewc_lambda', 'status', 'outcome'] + METRICS + ['observed_finite_steps', 'gradient_component_samples',
                   'telemetry_scope', 'raw_fisher_sha256', 'parent_checkpoint_sha256', 'run_dir', 'note']
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'RESULTS.md').write_text(render_markdown(data))
    if complete:
        atomic_json(output / 'COMPLETE.json', dict(status='complete', observed_lambda_points=14,
                    completed_lambda_points=data['completed_lambda_points'], diverged_lambda_points=data['diverged_lambda_points'],
                    seed=0, baseline_lambda0_complete=True, gpu_evaluation_performed=False,
                    reporter_sha256=data['reporter_sha256'], spec_sha256=data['spec_sha256'],
                    raw_fisher_sha256=prepared['payload_sha256'],
                    outputs_sha256={name: sha(output / name) for name in ('RESULTS.md', 'results.csv', 'results.json')}))
    elif not allow_partial:
        raise RuntimeError(f'Report remains partial: {observed}/14 observed; lambda0 complete={selection["baseline_complete"]}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    data = build_report(args.sweep_root, args.output_dir or args.sweep_root / 'report', args.allow_partial)
    print(json.dumps({key: data[key] for key in ('status', 'observed_lambda_points', 'completed_lambda_points', 'diverged_lambda_points')}))


if __name__ == '__main__':
    main()
