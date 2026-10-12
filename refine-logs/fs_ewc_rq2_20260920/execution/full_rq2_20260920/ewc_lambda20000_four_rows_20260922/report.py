#!/usr/bin/env python3
"""CPU-only, fail-closed four-arm report for user-fixed raw-Fisher lambda 20000."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics

# Set before the only optional tensor import. Collection never evaluates a model.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_variable] = '4'
HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
HELPER = FORMAL / 'ewc_lambda_sweep_seed0_20260921/report.py'
HELPER_SHA = '1808e5b1c9767e8efb61ae3ff7ff76f511760ded087dfd904a710baa62377e67'
ARMS = {'ewclora': ('sgd', 'sgd', 'C', 'C+P'),
        'fs_ewclora': ('gam', 'gam', 'C+F', 'C+F+P'),
        'task1_shaping': ('gam', 'sgd', 'C+F', 'C+P'),
        'task2_shaping': ('sgd', 'gam', 'C', 'C+F+P')}
SEEDS = [0, 1, 2]
LAMBDA = 20000.0
FISHER_POLICY = 'recomputed_native_task0_raw_fisher_no_trace_normalization'
METRICS = ['a11', 'a21', 'a22', 'BWT', 'A2', 'B1_batch_proxy', 'd_QV_cont',
           'raw_fisher_trace', 'raw_fisher_weighted_drift_squared', 'endpoint_P',
           'CE_mean', 'P_mean', 'P_over_CE', 'grad_reference_norm_mean',
           'grad_P_norm_mean', 'grad_P_over_reference', 'grad_gam_direction_norm_mean']
IDENTITY_KEYS = ('protocol', 'mode', 'anchor_method', 'future_optimizer', 'arm', 'smoke_only',
                 'task0_shaping', 'task1_shaping', 'parent_validation', 'prepared_fisher',
                 'execution_ready', 'seed', 'requested_config', 'parent',
                 'source_manifest_sha256', 'runner_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, name):
    require(not isinstance(value, bool), 'Boolean numeric value: ' + name)
    value = float(value)
    require(math.isfinite(value), 'Nonfinite ' + name)
    return value


def read_json(path, matrices=False):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON: {path}: {value}')
    return json.loads(Path(path).read_text(), **({} if matrices else {'parse_constant': invalid}))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Assets:
    def __init__(self):
        self.cache, self.ledger = {}, {}

    def hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self.cache:
            self.cache[key] = sha(path)
        digest = self.cache[key]
        require(str(path) not in self.ledger or self.ledger[str(path)] == digest,
                f'Evidence changed during collection: {path}')
        self.ledger[str(path)] = digest
        return digest

    def verify(self, path, digest, label):
        require(self.hash(path) == digest, f'{label} SHA mismatch: {path}')


def old_helpers(assets):
    assets.verify(HELPER, HELPER_SHA, 'Frozen CPU helper')
    spec = importlib.util.spec_from_file_location('raw_fisher_cpu_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_spec(root, assets):
    spec_path, selection_path = root / 'experiment_spec.json', root / 'SELECTION.json'
    spec, selection = read_json(spec_path), read_json(selection_path)
    assets.hash(spec_path)
    require(Path(spec['selection_receipt']).resolve() == selection_path, 'Selection path changed')
    assets.verify(selection_path, spec['selection_sha256'], 'User selection')
    require(spec['lambda_star'] == selection['lambda_star'] == LAMBDA and spec['seeds'] == SEEDS,
            'Fixed lambda or seeds changed')
    require(selection['status'] == 'fixed_by_user' and
            selection['selection_method'] == 'explicit_user_fixed_common_lambda' and
            selection['not_claimed_optimal_or_protection_verified'] is True and
            selection['predefined_protection_criterion_met'] is False and
            selection['exact_formal_continuations'] == 12 and
            selection['no_additional_lambda0_training_in_this_campaign'] is True,
            'Manual selection must not become an optimization or protection claim')
    require([(a['arm'], a['anchor_method'], a['future_optimizer'], a['task0_objective'],
              a['task1_objective']) for a in spec['arms']] == [(a, *v) for a, v in ARMS.items()],
            'Four-arm scientific mapping changed')
    trials = spec['trials']
    require([(t['arm'], t['seed']) for t in trials] == [(a, s) for a in ARMS for s in SEEDS],
            'Exactly 12 unique arm/seed cells are required')
    for trial in trials:
        arm, seed = trial['arm'], trial['seed']
        method, optimizer, _, _ = ARMS[arm]
        require(trial['trial_id'] == f'{arm}_seed{seed}' and trial['anchor_method'] == method and
                trial['future_optimizer'] == optimizer and trial['parent_id'] == f'{method}_seed{seed}' and
                trial['ewc_lambda'] == LAMBDA and
                Path(trial['run_dir']).resolve() == root / 'training' / trial['trial_id'],
                'Trial identity, path or lambda changed')
    parents = spec['parents']
    require([(p['anchor_method'], p['seed']) for p in parents] == [(m, s) for m in ('sgd', 'gam') for s in SEEDS],
            'Exactly six same-seed anchors are required')
    for parent in parents:
        method, seed = parent['anchor_method'], parent['seed']
        expected = FORMAL / 'formal_training' / f'anchor_{method}_seed{seed}'
        require(parent['id'] == f'{method}_seed{seed}' and Path(parent['directory']).resolve() == expected and
                Path(parent['checkpoint']).resolve() == expected / 'checkpoints/task_00.pt' and
                Path(parent['raw_fisher']).resolve() == root / 'prepared_fisher' / parent['id'] / 'fisher_state_raw.pt' and
                Path(parent['raw_fisher_receipt']).resolve() == root / 'prepared_fisher' / parent['id'] / 'PREPARED.json' and
                parent['task0_training'] is False, 'Parent or prepared-Fisher path changed')
    required_protocol = dict(epochs=50, batch_size=128, learning_rate=.02, workers=4,
                             ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                             fisher_mode='batch_empirical', fisher_source='fit', precision='fp32',
                             tf32=False, paired_task1_inputs_and_head=True, trajectory_averaging=False,
                             same_adapter_continued=True, train_task_indices=[1], tasks_total=2)
    require(all(spec['protocol'].get(k) == v for k, v in required_protocol.items()), 'Scientific protocol changed')
    require(Path(spec['source_manifest']).resolve() == FORMAL.parent / 'source_frozen.json', 'Source path changed')
    assets.verify(spec['source_manifest'], spec['source_manifest_sha256'], 'Frozen source manifest')
    for entry in read_json(spec['source_manifest'])['files']:
        path = FORMAL.parent / 'source' / entry['path']
        require(path.resolve().is_relative_to(FORMAL.parent / 'source'), 'Source path escaped')
        assets.verify(path, entry['sha256'], 'Frozen source file')
    return spec, selection


def validate_anchor(parent, assets):
    anchor, method, seed = Path(parent['directory']), parent['anchor_method'], parent['seed']
    complete_path, manifest_path = anchor / 'COMPLETE.json', anchor / 'run_manifest.json'
    assets.verify(complete_path, parent['parent_complete_sha256'], 'Frozen Task0 completion')
    assets.verify(manifest_path, parent['parent_manifest_sha256'], 'Frozen Task0 manifest')
    receipt, manifest = read_json(complete_path), read_json(manifest_path)
    require(receipt['status'] == 'complete' and receipt['completed_tasks'] == 1 and
            receipt['identity_sha256'] == manifest['identity_sha256'] and
            Path(receipt['manifest']).resolve() == manifest_path and
            manifest['seed'] == seed and manifest['anchor_method'] == method,
            'Task0 complete/identity/method/seed mismatch')
    valid = receipt['validation']
    require(valid['epochs_per_new_task'] == 50 and valid['validated_tasks'] == [0], 'Task0 is not formal 50 epochs')
    checkpoint = anchor / 'checkpoints/task_00.pt'
    assets.verify(checkpoint, parent['checkpoint_sha256'], 'Task0 checkpoint')
    assets.verify(checkpoint, valid['checkpoint_assets']['checkpoints/task_00.pt']['sha256'], 'Certified Task0 checkpoint')
    assets.verify(anchor / 'checkpoints/fisher_state_0.pt',
                  valid['checkpoint_assets']['checkpoints/fisher_state_0.pt']['sha256'], 'Original Task0 Fisher')
    for name in ('rng_receipts', 'runtime_receipts'):
        assets.verify(anchor / (name + '.jsonl'), valid[name + '_sha256'], 'Task0 ' + name)
    metrics_path = Path(valid['metrics']).resolve()
    require(metrics_path.is_relative_to(anchor / 'trainer_outputs'), 'Task0 metrics source escaped')
    assets.verify(metrics_path, valid['metrics_sha256'], 'Task0 metrics')
    metrics = read_json(metrics_path, matrices=True)
    mechanism = metrics['mechanism']['steps']['0']
    require(mechanism['fisher_estimator_family'] == 'batch_gradient_square_proxy' and
            mechanism['fisher_is_per_sample_empirical'] is False and mechanism['fisher_num_samples'] == 1259 and
            mechanism['in_task_fisher_used_path'] is False and mechanism['in_task_weight_merge_applied'] == 0 and
            mechanism['train_ewc_mean'] == 0, 'Task0 B1 estimator/source changed')
    a11 = numeric(metrics['cnn']['matrices']['final'][0][0], 'a11')
    b1 = numeric(mechanism['in_task_fisher_final_raw_trace'], 'B1 batch proxy')
    require(0 <= a11 <= 100 and b1 > 0, 'Invalid Task0 metric')
    return dict(anchor=str(anchor), method=method, seed=seed, parent_checkpoint=str(checkpoint),
                parent_checkpoint_sha256=parent['checkpoint_sha256'], metrics=str(metrics_path),
                metrics_sha256=valid['metrics_sha256'], a11=a11, B1_batch_proxy=b1,
                B1_source=str(metrics_path), rng_receipts_sha256=valid['rng_receipts_sha256'])


def validate_prepared(root, parent, anchor, assets):
    receipt_path, payload_path = Path(parent['raw_fisher_receipt']), Path(parent['raw_fisher'])
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    expected = dict(status='complete', task=0, seed=parent['seed'], anchor_method=parent['anchor_method'],
                    fisher_mode='batch_empirical', fisher_source='fit', ewc_normalize_fisher=False,
                    ewc_normalize_accumulated_fisher=False, tensor_count=24, numel=14155776, weights_unchanged=True)
    require(all(receipt.get(k) == v for k, v in expected.items()), 'Prepared raw-Fisher policy/seed/method mismatch')
    require(Path(receipt['payload_path']).resolve() == payload_path and
            Path(receipt['parent_checkpoint_path']).resolve() == Path(anchor['parent_checkpoint']),
            'Prepared raw-Fisher source path mismatch')
    assets.verify(payload_path, receipt['payload_sha256'], 'Raw Fisher payload')
    assets.verify(anchor['parent_checkpoint'], receipt['parent_checkpoint_sha256'], 'Raw Fisher parent')
    assets.verify(Path(anchor['anchor']) / 'run_manifest.json', receipt['parent_manifest_sha256'], 'Raw Fisher parent manifest')
    assets.verify(Path(anchor['anchor']) / 'rng_receipts.jsonl', receipt['parent_rng_receipts_sha256'], 'Raw Fisher RNG reference')
    assets.verify(FORMAL.parent / 'source_frozen.json', receipt['source_manifest_sha256'], 'Raw Fisher source')
    exact = receipt['exact_inputs_receipt']
    require(exact['matched'] is True, 'Raw Fisher inputs not paired')
    reference = [r for r in jsonl(Path(anchor['anchor']) / 'rng_receipts.jsonl') if
                 r.get('event') == 'loader_iteration' and r.get('task') == 0 and r.get('shuffle') is True and r.get('iteration') == 50]
    require(len(reference) == 1, 'Task0 native fit-Fisher pass missing or duplicated')
    for key in ('indices_sha256', 'first_batch_pixels_targets_sha256', 'dataset_ids_sha256',
                'generator_seed', 'num_workers', 'samples', 'batches'):
        require(exact.get(key) == reference[0].get(key) and exact.get(key) is not None, 'Raw Fisher replay mismatch: ' + key)
    trace = numeric(receipt['raw_trace'], 'raw Fisher trace')
    require(trace > 0 and math.isclose(numeric(receipt['mean'], 'raw Fisher mean'), trace / receipt['numel'], rel_tol=1e-8),
            'Raw Fisher mean/trace mismatch')
    return dict(receipt=receipt, receipt_path=str(receipt_path), receipt_sha256=assets.hash(receipt_path),
                payload_path=str(payload_path), payload_sha256=receipt['payload_sha256'],
                parent_checkpoint=anchor['parent_checkpoint'], parent_checkpoint_sha256=anchor['parent_checkpoint_sha256'],
                anchor=anchor['anchor'], a11=anchor['a11'])


def prepared_identity(identity, prepared):
    require(identity.get('status') == 'verified' and identity.get('execution_ready') is True and
            identity.get('fisher_policy') == FISHER_POLICY, 'Prepared identity not verified')
    for key in ('payload_sha256', 'receipt_sha256', 'parent_checkpoint_sha256'):
        require(identity.get(key) == prepared[key], 'Prepared identity changed: ' + key)
    require(Path(identity['path']).resolve() == Path(prepared['payload_path']) and
            Path(identity['receipt_path']).resolve() == Path(prepared['receipt_path']), 'Prepared identity path changed')
    for key in ('seed', 'anchor_method'):
        require(identity.get(key) == prepared['receipt'][key], 'Prepared identity seed/method changed')


def validate_manifest(root, trial, manifest, anchor, prepared, assets):
    method, optimizer, _, _ = ARMS[trial['arm']]
    expected = dict(protocol='rq2_shared_task_rng_v1', mode='continuation', seed=trial['seed'],
                    anchor_method=method, future_optimizer=optimizer, arm=trial['arm'], smoke_only=False,
                    task0_shaping=method == 'gam', task1_shaping=optimizer == 'gam')
    require(all(manifest.get(k) == v for k, v in expected.items()), 'Formal manifest arm/seed/optimizer identity mismatch')
    require(manifest['run_name'] == trial['trial_id'] and manifest['matmul_allow_tf32'] is False and
            manifest['cudnn_allow_tf32'] is False, 'Run identity or FP32 policy changed')
    require(manifest['prepared_fisher'].get('execution_ready') is True, 'Manifest prepared Fisher is not execution-ready')
    identity = {key: (manifest['prepared_fisher']['execution_ready'] if key == 'execution_ready' else manifest[key])
                for key in IDENTITY_KEYS}
    require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == manifest['identity_sha256'],
            'Manifest identity hash mismatch')
    assets.verify(root / 'train.py', manifest['runner_sha256'], 'Frozen trainer')
    assets.verify(FORMAL.parent / 'source_frozen.json', manifest['source_manifest_sha256'], 'Training source')
    required = dict(dataset='imageneta', optimizer_type=optimizer, lambda_flat=float(optimizer == 'gam'),
                    ewc_lambda=LAMBDA, epochs=50, lrate=.02, batch_size=128, lora_rank=10, scheduler='cosine',
                    momentum=0, weight_decay=0., init_cls=20, increment=20, class_shuffle=False,
                    max_train_tasks=2, resume_from_task=0, gam_base_optimizer='sgd', gam_grad_rho=.05,
                    gam_grad_norm_rho=.2, memory_size=0, memory_per_class=0, ewc_gamma=.9, ewc_eta=0.,
                    ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                    fisher_mode='batch_empirical', ewc_fisher_source='fit', approximate_fisher_in_task=False,
                    ewc_dual_ascent=False, ewc_project_grad=False, delta_w_penalty=False,
                    weight_merge=False, in_task_weight_merge=False, rq2_rng_protocol='rq2_shared_task_rng_v1',
                    rq2_smoke_only=False, rq2_fisher_policy=FISHER_POLICY, rq2_anchor_method=method,
                    rq2_extension_arm=trial['arm'], endpoint_weight_merge=False,
                    gam_adaptive=False, gam_grad_gamma=.1, gam_grad_beta_1=1., gam_grad_beta_2=1.,
                    gam_grad_beta_3=1., ewc_max_batches=100, fisher_split=False,
                    train_num_workers=4, eval_num_workers=4, persistent_workers=False)
    run = root / 'training' / trial['trial_id']
    for key in ('requested_config', 'config'):
        cfg = manifest[key]
        require(all(cfg.get(k) == v for k, v in required.items()), 'Scientific config mismatch: ' + key)
        require(cfg['seed'] in (trial['seed'], [trial['seed']]), 'Config seed mismatch')
        require(Path(cfg['resume_from_checkpoint_dir']).resolve() == Path(anchor['parent_checkpoint']).parent and
                Path(cfg['save_task_checkpoint_dir']).resolve() == run / 'checkpoints' and
                Path(cfg['rq2_prepared_fisher']).resolve() == Path(prepared['payload_path']), 'Config source/output path mismatch')
    parent = manifest['parent']
    paths = {'task_00.pt': anchor['parent_checkpoint'], 'fisher_state_0.pt': str(Path(anchor['anchor']) / 'checkpoints/fisher_state_0.pt'),
             'metrics': anchor['metrics'], 'run_manifest.json': str(Path(anchor['anchor']) / 'run_manifest.json'),
             'COMPLETE.json': str(Path(anchor['anchor']) / 'COMPLETE.json')}
    require(set(parent) == set(paths), 'Parent assets incomplete or unexpected')
    for name, path in paths.items():
        require(Path(parent[name]['path']).resolve() == Path(path), 'Wrong same-seed parent: ' + name)
        assets.verify(path, parent[name]['sha256'], 'Training parent ' + name)
    pv = manifest['parent_validation']
    require(pv['passed'] is True and pv['seed'] == trial['seed'] and pv['anchor_method'] == method and
            Path(pv['formal_parent']).resolve() == Path(anchor['anchor']), 'Parent validation identity changed')
    prepared_identity(manifest['prepared_fisher'], prepared)


def telemetry(run, optimizer, complete, valid):
    rows = jsonl(run / 'training_batch_metrics.jsonl')
    require([r['step'] for r in rows] == list(range(len(rows))) and len(rows) <= 400 and
            (not complete or len(rows) == 400), 'Training steps missing, duplicated or not formal 400')
    source = f'actual_native_{optimizer}_base_optimizer_gradients_immediately_before_step'
    component = ('native_clean_and_fisher_grad_caches; gam_direction=actual_combined-EWC; float64_norms'
                 if optimizer == 'gam' else 'native_EWC_autograd; clean=actual_combined-EWC; float64_norms')
    sampled = []
    for row in rows:
        require(row['task'] == 1 and row['ewc_lambda'] == LAMBDA and row['parameters_finite_after_step'] is True,
                'Invalid finite-step identity')
        require(row.get('gradient_source') == source, 'Batch gradient source mismatch')
        require(row.get('native_ewc_evaluations') == 1 and row.get('single_ewc_injection_verified') is True,
                'Native update did not certify exactly one EWC injection')
        for key in ('ce_clean', 'ewc', 'loss', 'grad_combined_norm'):
            numeric(row[key], key)
        require(row['ce_clean'] >= 0 and row['ewc'] >= 0, 'Negative loss or penalty')
        require(math.isclose(row['loss'], row['ce_clean'] + row['ewc'], rel_tol=1e-6, abs_tol=1e-8),
                'Batch scalar loss does not equal native clean CE plus P')
        sampled_flag = row['step'] == 1 or row['step'] % 8 == 0
        require(row['gradient_components_sampled'] is sampled_flag, 'Gradient component schedule changed')
        if sampled_flag:
            require(row.get('component_source') == component, 'Batch component source mismatch')
            for key in ('grad_clean_norm', 'grad_fisher_norm', 'fisher_to_clean_norm_ratio'):
                require(numeric(row[key], key) >= 0, 'Negative gradient component')
            require(math.isclose(row['fisher_to_clean_norm_ratio'], row['grad_fisher_norm'] / (row['grad_clean_norm'] + 1e-12),
                                 rel_tol=1e-9, abs_tol=1e-12), 'Gradient component ratio mismatch')
            sampled.append(row)
    result = dict(observed_finite_steps=len(rows), gradient_component_samples=len(sampled),
                  gradient_source=source, gradient_component_source=component,
                  gradient_reference=('native GAM clean_grad_cache' if optimizer == 'gam' else 'native SGD clean CE gradient'),
                  telemetry_scope='all 400 completed updates' if complete else 'finite prefix before divergence',
                  batch_loss_equals_CE_plus_P=True)
    if rows:
        ce, penalty = statistics.mean(r['ce_clean'] for r in rows), statistics.mean(r['ewc'] for r in rows)
        result.update(CE_mean=ce, P_mean=penalty, P_over_CE=penalty / ce if ce else None)
    if sampled:
        result.update(grad_reference_norm_mean=statistics.mean(r['grad_clean_norm'] for r in sampled),
                      grad_P_norm_mean=statistics.mean(r['grad_fisher_norm'] for r in sampled),
                      grad_P_over_reference=statistics.mean(r['fisher_to_clean_norm_ratio'] for r in sampled))
        step1 = next((r for r in sampled if r['step'] == 1), None)
        result['grad_P_norm_step1'] = step1['grad_fisher_norm'] if step1 else None
    if complete:
        gradients = valid['gradients']
        require(gradients['source'] == source and gradients['component_source'] == component, 'Native gradient source mismatch')
        require(gradients['component_sample_count'] == len(sampled), 'Native gradient sample count changed')
        for output, key in [('CE_mean', 'train_clean_loss_mean'), ('P_mean', 'train_fisher_loss_mean')]:
            require(math.isclose(result[output], numeric(valid['losses'][key], key), rel_tol=1e-7, abs_tol=1e-10), 'Native mean loss mismatch')
        for output, key in [('grad_reference_norm_mean', 'grad_clean_norm_mean'), ('grad_P_norm_mean', 'grad_fisher_norm_mean'),
                            ('grad_P_over_reference', 'fisher_to_clean_norm_ratio_mean')]:
            require(math.isclose(result[output], numeric(gradients[key], key), rel_tol=1e-9, abs_tol=1e-12), 'Native mean gradient mismatch')
        if optimizer == 'gam':
            direction = [numeric(r['grad_gam_direction_norm'], 'GAM direction') for r in sampled]
            result['grad_gam_direction_norm_mean'] = statistics.mean(direction)
            require(math.isclose(result['grad_gam_direction_norm_mean'], gradients['grad_gam_direction_norm_mean'],
                                 rel_tol=1e-9, abs_tol=1e-12), 'GAM shaping direction mean mismatch')
        require(result['P_mean'] > 0 and result['grad_P_norm_mean'] > 0 and result['grad_P_norm_step1'] > 0,
                'Positive EWC loss/gradient and second-update witness required')
        require(gradients['single_ewc_injection_verified'] is True and
                math.isclose(gradients['step1_ewc'], rows[1]['ewc'], rel_tol=1e-9, abs_tol=1e-12) and
                math.isclose(gradients['step1_grad_fisher_norm'], rows[1]['grad_fisher_norm'], rel_tol=1e-9, abs_tol=1e-12),
                'Native EWC injection/step1 witness mismatch')
    return result


def read_matrix(metrics, validation, anchor_a11):
    matrices = metrics['cnn']['matrices']
    final, t01 = matrices['final'], matrices['t01']
    require(len(final) == len(t01) == 2 and all(len(row) == 2 for row in final + t01), 'Missing two-task CIL matrix')
    require(all(final[i][j] == t01[i][j] for i in range(2) for j in range(i + 1)), 'Final and t01 matrices disagree')
    values = dict(a11=numeric(final[0][0], 'a11'), a21=numeric(final[1][0], 'a21'), a22=numeric(final[1][1], 'a22'))
    require(values == validation['accuracy_CIL'] and values['a11'] == anchor_a11 and
            all(0 <= v <= 100 for v in values.values()), 'CIL accuracy or inherited Task0 mismatch')
    return dict(**values, BWT=values['a21'] - values['a11'], A2=(values['a21'] + values['a22']) / 2)


def validate_trial(root, trial, anchor, prepared, assets, helper):
    run = root / 'training' / trial['trial_id']
    row = dict(trial_id=trial['trial_id'], arm=trial['arm'], seed=trial['seed'], ewc_lambda=LAMBDA,
               anchor_method=trial['anchor_method'], future_optimizer=trial['future_optimizer'],
               status='pending', run_dir=str(run), **{k: None for k in METRICS})
    if anchor:
        row.update(a11=anchor['a11'], B1_batch_proxy=anchor['B1_batch_proxy'], B1_source=anchor['B1_source'],
                   parent_checkpoint_sha256=anchor['parent_checkpoint_sha256'])
    terminal = [name for name in ('COMPLETE.json', 'DIVERGED.json', 'FAILED.json') if (run / name).exists()]
    require(len(terminal) <= 1, 'Conflicting native terminal receipts')
    if not terminal:
        row['note'] = 'No formal native terminal receipt; inherited Task0 metrics do not imply Task1 completion.'
        return row
    if terminal == ['FAILED.json']:
        row.update(status='failed', note=str(read_json(run / 'FAILED.json')), failure_sha256=assets.hash(run / 'FAILED.json'))
        return row
    require(anchor is not None and prepared is not None, 'Terminal result lacks validated parent/raw Fisher')
    complete = terminal == ['COMPLETE.json']
    receipt_path, manifest_path = run / terminal[0], run / 'run_manifest.json'
    receipt, manifest = read_json(receipt_path), read_json(manifest_path)
    validate_manifest(root, trial, manifest, anchor, prepared, assets)
    require(receipt['identity_sha256'] == manifest['identity_sha256'] and receipt['smoke_only'] is False and
            receipt['arm'] == trial['arm'] and receipt['status'] == ('complete' if complete else 'diverged'),
            'Terminal receipt identity mismatch')
    valid = receipt.get('validation', {})
    runtime = jsonl(run / 'runtime_receipts.jsonl')
    for filename in ('rng_receipts.jsonl', 'runtime_receipts.jsonl', 'training_batch_metrics.jsonl'):
        assets.hash(run / filename)
        if complete:
            assets.verify(run / filename, valid[filename.replace('.jsonl', '_sha256')], 'Native telemetry')
    restored = [r for r in runtime if r.get('event') == 'complete_resume' and r.get('task') == 0]
    replaced = [r for r in runtime if r.get('event') == 'prepared_raw_fisher_loaded' and r.get('task') == 0]
    require(len(restored) == len(replaced) == 1 and restored[0]['verified_exact'] is True and
            restored[0]['checkpoint_sha256'] == anchor['parent_checkpoint_sha256'], 'Exact same-seed Task0 restoration missing')
    require(replaced[0]['verified_exact'] is True and replaced[0]['replaced_original_normalized_fisher'] is True and
            replaced[0]['ewc_normalize_fisher'] is False and replaced[0]['ewc_normalize_accumulated_fisher'] is False,
            'Native raw-Fisher replacement missing')
    prepared_identity(replaced[0]['prepared_fisher'], prepared)
    reference = FORMAL / 'formal_training' / f"sequence_{trial['future_optimizer']}_seed{trial['seed']}" / 'rng_receipts.jsonl'
    pairing = helper.validate_pairing(run, complete, assets, reference)
    row.update(**telemetry(run, trial['future_optimizer'], complete, valid), pairing=pairing,
               native_receipt_sha256=assets.hash(receipt_path), manifest_sha256=assets.hash(manifest_path),
               raw_fisher_sha256=prepared['payload_sha256'], raw_fisher_trace=prepared['receipt']['raw_trace'])
    if not complete:
        require(receipt['seed'] == trial['seed'] and receipt['ewc_lambda'] == LAMBDA and
                receipt['fisher_policy'] == FISHER_POLICY and receipt['hyperparameters_changed'] is False and
                receipt['last_completed_training_steps'] == row['observed_finite_steps'], 'Invalid divergence receipt')
        prepared_identity(receipt['prepared_fisher'], prepared)
        row.update(status='diverged', note=receipt['reason'])
        return row
    manifest_reference = receipt.get('manifest')
    if manifest_reference is None:
        require(receipt.get('recovered_after_final_task') is True, 'Completion lacks its manifest reference')
        manifest_reference = str(manifest_path)
    require(Path(manifest_reference).resolve() == manifest_path and receipt['completed_tasks'] == 2 and
            valid['epochs_per_new_task'] == 50 and valid['validated_tasks'] == [1] and
            valid['smoke_only'] is False and valid['fisher_policy'] == FISHER_POLICY, 'Not formal Task1 completion')
    prepared_identity(valid['prepared_fisher'], prepared)
    metrics_path = Path(valid['metrics']).resolve()
    require(metrics_path.is_relative_to(run / 'trainer_outputs'), 'CIL metrics do not belong to this cell')
    assets.verify(metrics_path, valid['metrics_sha256'], 'Native CIL matrix')
    local_path = run / 'task0_local_metrics.jsonl'
    assets.verify(local_path, valid['task0_local_metrics_sha256'], 'Native endpoint local metrics')
    local = [r for r in jsonl(local_path) if r.get('task') == 1]
    require(len(local) == 1, 'Native local endpoint receipt missing or duplicated')
    for key in ('effective_weight_displacement', 'fisher_weighted_displacement', 'endpoint_ewc_penalty'):
        require(local[0][key] == valid[key], 'Native endpoint local/COMPLETE mismatch: ' + key)
    logs = sorted((run / 'trainer_outputs').rglob('*.log'))
    require(logs and any(re.search(r'Task 1, Epoch 50/50\b', p.read_text()) for p in logs), 'Actual final epoch log missing')
    for path in logs:
        assets.hash(path)
    metrics = read_json(metrics_path, matrices=True)
    row.update(read_matrix(metrics, valid, anchor['a11']))
    mechanism = metrics['mechanism']['steps']['1']
    require(mechanism['lambda_ewc'] == LAMBDA and mechanism['lambda_flat'] == float(trial['future_optimizer'] == 'gam'),
            'Native mechanism coefficients mismatch')
    if trial['future_optimizer'] == 'gam':
        for name in ('grad_clean_norm_mean', 'grad_gam_norm_mean', 'grad_fisher_norm_mean', 'grad_combined_norm_mean'):
            require(numeric(mechanism[name], name) >= 0, 'Invalid native GAM mechanism gradient')
        require(mechanism['grad_gam_norm_mean'] > 0 and mechanism['grad_fisher_norm_mean'] > 0,
                'Native shaping and EWC mechanisms were not both exercised')
    endpoint = run / 'checkpoints/task_01.pt'
    require('checkpoints/task_01.pt' in valid['checkpoint_assets'], 'Endpoint checkpoint not certified')
    for relative, asset in valid['checkpoint_assets'].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run / 'checkpoints'), 'Checkpoint asset path escaped')
        require(path.stat().st_size == asset['bytes'], 'Checkpoint asset size mismatch')
        assets.verify(path, asset['sha256'], 'Endpoint checkpoint asset')
    require(any(r.get('event') == 'checkpoint_full_reload_verified' and r.get('task') == 1 and
                r.get('verified_exact') is True for r in runtime), 'Native full-state endpoint reload missing')
    require(any(r.get('event') == 'task_complete' and r.get('task') == 1 for r in runtime), 'Task1 runtime completion missing')
    import torch
    torch.set_num_threads(4)
    old = torch.load(anchor['parent_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    payload = torch.load(prepared['payload_path'], map_location='cpu', weights_only=True, mmap=True)
    new = torch.load(endpoint, map_location='cpu', weights_only=True, mmap=True)
    require(old['cur_task'] == 0 and old['total_classes'] == 20 and old['optimizer_type'] == trial['anchor_method'] and
            new['cur_task'] == 1 and new['total_classes'] == 40 and new['optimizer_type'] == trial['future_optimizer'],
            'Stored checkpoint task/optimizer identity mismatch')
    for name, tensor in new['network_state'].items():
        require(bool(torch.isfinite(tensor).all()), 'Nonfinite endpoint state: ' + name)
    fisher = payload['fisher_past_delta']
    endpoint_fisher = torch.load(run / 'checkpoints/fisher_state_1.pt', map_location='cpu', weights_only=True, mmap=True)['fisher_past_delta']
    require(set(endpoint_fisher) == set(fisher) and all(bool(torch.isfinite(t).all()) and bool((t >= 0).all())
                                                     for t in endpoint_fisher.values()), 'Invalid native Task1 Fisher state')
    trace = sum(float(t.double().sum()) for t in fisher.values())
    require(math.isclose(trace, row['raw_fisher_trace'], rel_tol=1e-8), 'Raw Fisher payload trace mismatch')
    drift = helper.drift_from_states(old['network_state'], new['network_state'], fisher)
    require(math.isclose(drift['d_QV_cont'], numeric(valid['effective_weight_displacement'], 'native drift'),
                         rel_tol=1e-5, abs_tol=1e-8), 'CPU/native displacement mismatch')
    require(math.isclose(drift['raw_fisher_weighted_drift_squared'], numeric(valid['fisher_weighted_displacement'], 'native weighted drift'),
                         rel_tol=1e-5, abs_tol=1e-10), 'CPU/native raw-Fisher weighted displacement mismatch')
    require(math.isclose(drift['d_QV_cont'] ** 2, numeric(mechanism['delta_norm'], 'native delta squared'),
                         rel_tol=1e-5, abs_tol=1e-8), 'Native mechanism displacement mismatch')
    require(math.isclose(.5 * LAMBDA * drift['raw_fisher_weighted_drift_squared'], valid['endpoint_ewc_penalty'],
                         rel_tol=1e-5, abs_tol=1e-9), 'Native endpoint EWC scaling mismatch')
    row.update(drift, status='complete', endpoint_P=.5 * LAMBDA * drift['raw_fisher_weighted_drift_squared'],
               metrics_source=str(metrics_path), endpoint_checkpoint=str(endpoint),
               endpoint_checkpoint_sha256=assets.hash(endpoint), note='All native and CPU evidence gates passed.')
    return row


def summarize(rows):
    require([(r['arm'], r['seed']) for r in rows] == [(a, s) for a in ARMS for s in SEEDS],
            'Aggregation requires exactly 12 ordered unique cells')
    summaries = []
    for arm in ARMS:
        cells = [r for r in rows if r['arm'] == arm]
        n = sum(r['status'] == 'complete' for r in cells)
        row = dict(arm=arm, n_complete=n, expected_seeds=3, status='complete' if n == 3 else 'partial',
                   task0_objective=ARMS[arm][2], task1_objective=ARMS[arm][3], statistics={})
        for metric in METRICS:
            values = [r.get(metric) for r in cells]
            stats = {'mean': None, 'sample_sd': None, 'n': n}
            if n == 3 and all(v is not None for v in values):
                numbers = [numeric(v, metric) for v in values]
                stats.update(mean=statistics.mean(numbers), sample_sd=statistics.stdev(numbers))
            row['statistics'][metric] = stats
        summaries.append(row)
    return summaries


def format_value(value):
    return '—' if value is None else f'{value:.6g}' if isinstance(value, (int, float)) else str(value).replace('|', '/')


def render_markdown(data):
    lines = ['# 固定 λ=20000：四分支、三个 seed', '',
             'λ=20000 由用户手动固定；不是扫描最优值，也没有被先前扫描证实达到保护门槛。',
             f"状态 {data['status']}：{data['completed_trials']}/12 有效完成，{data['diverged_trials']} 发散，{data['invalid_trials']} 无效。", '',
             '同一分支的 seeds 0/1/2 全部有效完成时才报告 mean ± sample SD（ddof=1）。发散、失败、缺失不填零，不参与凑均值。', '',
             '| 分支 | Task0 | Task1 | 有效 seed | a11 | B1 batch proxy | a21 | a22 | BWT | A2 | d_QV_cont |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    main = ['a11', 'B1_batch_proxy', 'a21', 'a22', 'BWT', 'A2', 'd_QV_cont']
    for row in data['summary']:
        values = []
        for key in main:
            stat = row['statistics'][key]
            values.append('—' if stat['mean'] is None else f"{stat['mean']:.5g} ± {stat['sample_sd']:.3g}")
        lines.append('| ' + ' | '.join([row['arm'], row['task0_objective'], row['task1_objective'], str(row['n_complete'])] + values) + ' |')
    fields = ['trial_id', 'status', 'a11', 'B1_batch_proxy', 'a21', 'a22', 'BWT', 'A2', 'd_QV_cont',
              'raw_fisher_trace', 'raw_fisher_weighted_drift_squared', 'P_over_CE']
    lines += ['', '逐 seed 原生观测：', '', '| ' + ' | '.join(fields) + ' |', '|' + '|'.join(['---'] * len(fields)) + '|']
    for row in data['individual']:
        lines.append('| ' + ' | '.join(format_value(row.get(k)) for k in fields) + ' |')
    lines += ['', '指标和证据范围：', '',
              '- a11/a21/a22 为原生 CIL 矩阵的百分比值；BWT=a21−a11（百分点），A2=(a21+a22)/2。',
              '- B1_batch_proxy 取同 seed 原生 Task0 anchor 日志的 in_task_fisher_final_raw_trace：fit loader、1259 样本的 batch-gradient-square proxy，未归一化、非逐样本 Fisher、无轨迹平均。',
              '- raw_fisher_trace 来自本轮复用/重算的 Task0 raw Fisher，独立核验 payload；它与历史固定 train256 逐样本 Fisher 不同，不能混填 B1。',
              '- d_QV_cont 是同一 LoRA 分支从 Task0 到 Task1 的 24 个 Q/V 有效 BA 位移的 Frobenius 范数，排除分类头。rawF 加权平方为 Σ rawF·(BA1−BA0)²；endpoint_P=λ/2 倍该量。',
              '- CE_mean、P_mean 按全部 400 个实际更新求均值；P 已含 λ/2，P/CE 为两均值之比。发散行如有遥测只表示有限前缀，不能与完整训练均值等同。',
              '- 梯度只采样零起始 steps 0、1、8、16…；SGD reference 是真实 CE 分量，GAM reference 是 native clean_grad_cache。GAM 实际更新减 EWC 的残差另称 GAM direction，包含 shaping，绝不称干净 CE 梯度。每个 seed 的 gradient_source/component_source 保留于 CSV/JSON。',
              '- 所有继承的 a11/B1 都来自已经完成的父 Task0；待运行行出现这些值不代表 Task1 已执行。',
              '- 本轮没有新增 λ=0 训练；既有 matched λ=0 对照未在本报告重新核验，因此相对无 EWC 的保护收益记为不可判定。历史 trace-normalized λ=2000 与本轮 raw λ=20000 尺度不同，不混并为正式因果比较。',
              '- report/COMPLETE.json 只在全 12 个原生结果和证据链有效完成后生成；不表示固定 λ 已证明最优或通过保护门槛。', '']
    for row in data['individual']:
        if row['status'] != 'complete':
            lines.append(f"- {row['trial_id']} [{row['status']}]：{row.get('note', '')}")
    return '\n'.join(lines) + '\n'


def render_tex(data):
    fields = ['a11', 'B1_batch_proxy', 'a21', 'a22', 'BWT', 'A2', 'd_QV_cont']
    lines = [r'\begin{table*}[t]', r'\centering', r'\begin{tabular}{lllrrrrrrr}', r'\hline',
             r'Variant & Task 0 & Task 1 & $a_{11}$ & $\mathcal B_1$ & $a_{21}$ & $a_{22}$ & BWT & $A_2$ & $d_{QV}^{\rm cont}$ \\', r'\hline']
    for row in data['summary']:
        values = []
        for key in fields:
            stat = row['statistics'][key]
            values.append('--' if stat['mean'] is None else f"${stat['mean']:.4g}_{{\\pm {stat['sample_sd']:.3g}}}$")
        lines.append(' & '.join([row['arm'].replace('_', r'\_'), row['task0_objective'], row['task1_objective']] + values) + r' \\')
    lines += [r'\hline', r'\end{tabular}',
              r'\caption{User-fixed raw-Fisher $\lambda=20000$; not an automatically selected optimum or a verified protection threshold. Only complete seed sets (0,1,2) have means and sample SDs. $\mathcal B_1$ is the native Task-0 batch-gradient-square proxy, not per-example Fisher. Missing or divergent cells are not imputed. $d_{QV}^{\rm cont}$ excludes classifier heads.}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def build_report(root, output, allow_partial=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root == HERE and output.is_relative_to(root) and output != root,
            'Collector is pinned to this campaign; output must be its own subdirectory')
    require(not any(output.is_relative_to(root / name) for name in
                    ('training', 'prepared_fisher', 'smoke', 'campaign', 'witness', 'cpu_validation')),
            'Report cannot overwrite experiment evidence')
    (output / 'COMPLETE.json').unlink(missing_ok=True)
    assets = Assets()
    spec, selection = validate_spec(root, assets)
    helper = old_helpers(assets)
    trials = spec['trials']
    expected = {t['trial_id'] for t in trials}
    unexpected = [p.name for p in (root / 'training').glob('*') if p.is_dir() and p.name not in expected]
    require(not unexpected, 'Unexpected formal training cells: ' + ', '.join(unexpected))
    parents, errors = {}, {}
    for parent in spec['parents']:
        key = parent['id']
        try:
            anchor = validate_anchor(parent, assets)
            prepared = validate_prepared(root, parent, anchor, assets)
            parents[key] = (anchor, prepared)
        except (ValueError, KeyError, FileNotFoundError, TypeError, IndexError) as exc:
            parents[key] = (None, None)
            errors[key] = f'{type(exc).__name__}: {exc}'
    rows = []
    for trial in trials:
        try:
            require(trial['parent_id'] not in errors, errors.get(trial['parent_id'], ''))
            row = validate_trial(root, trial, *parents[trial['parent_id']], assets, helper)
        except (ValueError, KeyError, FileNotFoundError, TypeError, IndexError, RuntimeError) as exc:
            row = dict(trial_id=trial['trial_id'], arm=trial['arm'], seed=trial['seed'], ewc_lambda=LAMBDA,
                       status='invalid', run_dir=trial['run_dir'], note=f'{type(exc).__name__}: {exc}',
                       **{k: None for k in METRICS})
        rows.append(row)
    summary = summarize(rows)
    n = sum(r['status'] == 'complete' for r in rows)
    data = dict(status='complete' if n == 12 else 'partial', expected_trials=12, completed_trials=n,
                diverged_trials=sum(r['status'] == 'diverged' for r in rows),
                invalid_trials=sum(r['status'] == 'invalid' for r in rows),
                pending_trials=sum(r['status'] == 'pending' for r in rows),
                failed_trials=sum(r['status'] == 'failed' for r in rows),
                validated_seeds=SEEDS if n == 12 else [], ewc_lambda=LAMBDA, lambda_star=LAMBDA,
                selection_method=selection['selection_method'], scientific_status='manual_fixed_lambda_observations_only',
                protection_efficacy_vs_matched_lambda0='not_assessed_controls_not_revalidated',
                matched_controls={'status': 'not_revalidated', 'new_lambda0_training': False,
                                  'causal_protection_gain_available': False},
                generated_utc=datetime.now(timezone.utc).isoformat(), individual=rows, summary=summary,
                parent_errors=errors, selection=selection, reporter_sha256=assets.hash(__file__),
                spec_sha256=assets.hash(root / 'experiment_spec.json'), selection_sha256=assets.hash(root / 'SELECTION.json'),
                gpu_evaluation_performed=False, additional_training_launched=False)
    data['evidence_sha256'] = dict(assets.ledger)
    for path, digest in data['evidence_sha256'].items():
        assets.verify(path, digest, 'Publication-time evidence')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'results.json', data)
    fields = ['trial_id', 'arm', 'seed', 'status', 'ewc_lambda', 'anchor_method', 'future_optimizer'] + METRICS + [
        'observed_finite_steps', 'gradient_component_samples', 'gradient_source', 'gradient_component_source',
        'gradient_reference', 'grad_P_norm_step1', 'telemetry_scope', 'parent_checkpoint_sha256', 'raw_fisher_sha256',
        'endpoint_checkpoint_sha256', 'native_receipt_sha256', 'B1_source', 'metrics_source', 'run_dir', 'note']
    with (output / 'individual.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    flat = [{**{k: row[k] for k in ('arm', 'status', 'n_complete', 'expected_seeds', 'task0_objective', 'task1_objective')},
             **{f'{metric}_{stat}': row['statistics'][metric][stat] for metric in METRICS for stat in ('mean', 'sample_sd')}} for row in summary]
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / 'RESULTS.md').write_text(render_markdown(data))
    (output / 'table.tex').write_text(render_tex(data))
    if n == 12:
        atomic_json(output / 'COMPLETE.json', dict(status='complete', expected_trials=12, completed_trials=12,
                    validated_seeds=SEEDS, ewc_lambda=LAMBDA, lambda_star=LAMBDA,
                    selection_method=selection['selection_method'], scientific_status=data['scientific_status'],
                    gpu_evaluation_performed=False, additional_training_launched=False,
                    reporter_sha256=data['reporter_sha256'], spec_sha256=data['spec_sha256'], selection_sha256=data['selection_sha256'],
                    outputs_sha256={name: sha(output / name) for name in ('results.json', 'individual.csv', 'summary.csv', 'RESULTS.md', 'table.tex')}))
    elif not allow_partial:
        raise RuntimeError(f'Only {n}/12 valid complete trials; partial results preserved, no COMPLETE issued')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    data = build_report(args.run_root, args.output_dir or args.run_root / 'report', args.allow_partial)
    print(json.dumps({key: data[key] for key in ('status', 'expected_trials', 'completed_trials', 'diverged_trials', 'invalid_trials', 'pending_trials')}))


if __name__ == '__main__':
    main()
