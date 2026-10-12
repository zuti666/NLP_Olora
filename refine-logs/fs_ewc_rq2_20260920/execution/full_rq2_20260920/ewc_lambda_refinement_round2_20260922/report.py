#!/usr/bin/env python3
"""CPU-only round2 collector: frozen initial14 + round1six + six new observations."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'ewc_lambda_refinement_round1_20260922'
PRIOR_REPORTER_SHA256 = '2fe58a17c769daa15371742354643a14c7ebc139c2ced5ed92d4b177c8edb58e'
PRIOR_COMPLETE_SHA256 = 'e06a75e2c0eaa76020fc5409cf7b170b3a07e898d5ee426d4a7bc8124a8c1037'
PRIOR_SPEC_SHA256 = 'e50e4da13283b4e495da67bf3e4928885ab4f7731b6d31fed5a7b5e3d921bd12'
INTERVALS = ((3000., 10000.), (173205.0807568877, 227950.70569547775))


def load_prior_helpers():
    path = PRIOR / 'report.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR_REPORTER_SHA256:
        raise ValueError('Frozen round1 report helper changed')
    spec = importlib.util.spec_from_file_location('frozen_round1_raw_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_prior_helpers()
old = prior.old
INITIAL = prior.INITIAL
INITIAL_REPORTER_SHA256 = prior.INITIAL_REPORTER_SHA256
INITIAL_COMPLETE_SHA256 = prior.INITIAL_COMPLETE_SHA256
require, read_json, jsonl = old.require, old.read_json, old.jsonl
numeric, sha, atomic_json = old.numeric, old.sha, old.atomic_json
validate_prepared_identity, validate_pairing = old.validate_prepared_identity, old.validate_pairing
telemetry, drift_from_states = old.telemetry, old.drift_from_states
FISHER_POLICY, METRICS, COMMON_CONFIG = old.FISHER_POLICY, old.METRICS, old.COMMON_CONFIG
validate_initial = prior.validate_initial
mechanism_evidence, select_points = prior.mechanism_evidence, prior.select_points


class EvidenceAssets(old.Assets):
    """Keep every checked dependency in the publication-time evidence ledger."""
    def __init__(self):
        super().__init__()
        self.ledger = {}

    def hash(self, path):
        digest = super().hash(path)
        key = str(Path(path).resolve())
        require(key not in self.ledger or self.ledger[key] == digest,
                f'Evidence changed during collection: {path}')
        self.ledger[key] = digest
        return digest


def validate_spec(spec):
    require(spec.get('round') == 2 and spec.get('seed') == 0, 'Only the second seed0 refinement round is supported')
    require(Path(spec['initial_sweep']).resolve() == INITIAL.resolve(), 'Initial sweep source changed')
    require(spec.get('prior_refinement_rounds') == [dict(round=1, root=str(PRIOR),
                complete_sha256=PRIOR_COMPLETE_SHA256, reporter_sha256=PRIOR_REPORTER_SHA256,
                spec_sha256=PRIOR_SPEC_SHA256)], 'Frozen prior refinement history changed')
    prior_spec = read_json(PRIOR / 'refinement_spec.json')
    require(sha(PRIOR / 'refinement_spec.json') == PRIOR_SPEC_SHA256, 'Frozen round1 specification changed')
    prior.validate_spec(prior_spec)
    historical_grid = sorted(old.GRID + [t['ewc_lambda'] for t in prior_spec['trials']])
    require(spec.get('tested_lambda_union') == historical_grid, 'Tested lambda union must preserve all 20 historical points')
    require(Path(spec['prepared_fisher']).resolve() == INITIAL / 'prepared_fisher/fisher_state_raw.pt',
            'Round2 must reuse the same raw Fisher payload')
    trials = spec['trials']
    require(len(trials) == 6, 'Second refinement round requires six trials')
    ids, lambdas = set(), set(historical_grid)
    for index, trial in enumerate(trials):
        trial_id = trial['trial_id']
        require(trial_id == f'refine2_{index + 1:02d}_seed0' and trial_id not in ids,
                'Round2 trial IDs or order changed')
        lam = numeric(trial['ewc_lambda'], 'refinement lambda')
        interval, fraction = INTERVALS[index // 3], (.25, .5, .75)[index % 3]
        require(trial.get('interval') == list(interval) and trial.get('log_fraction') == fraction,
                'Round2 authorized interval/fraction changed')
        expected = interval[0] * (interval[1] / interval[0]) ** fraction
        require(lam > 0 and lam not in lambdas and math.isclose(lam, expected, rel_tol=1e-12),
                'Round2 lambda must be a new unique log-quarter point of the authorized intervals')
        ids.add(trial_id)
        lambdas.add(lam)
    require(spec['efficacy_thresholds'] == dict(retention_gain_pp=1.0, max_new_task_drop_pp=5.0,
                                              nondecreasing_average_gain_pp=0.0), 'Predefined efficacy thresholds changed')
    return trials


def validate_history(assets, tensor_cache):
    """Revalidate frozen round1 native evidence, then match its complete 20-point report."""
    folder = PRIOR / 'report'
    assets.verify(PRIOR / 'report.py', PRIOR_REPORTER_SHA256, 'Frozen round1 reporter')
    assets.verify(PRIOR / 'refinement_spec.json', PRIOR_SPEC_SHA256, 'Frozen round1 specification')
    assets.verify(folder / 'COMPLETE.json', PRIOR_COMPLETE_SHA256, 'Frozen round1 completion')
    complete = read_json(folder / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['round_status'] == 'observed'
            and complete['observed_refinement_points'] == 6 and complete['total_observed_points'] == 20
            and complete['baseline_lambda0_complete'] is True, 'Round1 is not fully observed')
    require(complete['reporter_sha256'] == PRIOR_REPORTER_SHA256 and complete['spec_sha256'] == PRIOR_SPEC_SHA256,
            'Round1 report chain mismatch')
    for name in ('RESULTS.md', 'results.csv', 'results.json'):
        assets.verify(folder / name, complete['outputs_sha256'][name], 'Frozen round1 report output')
    frozen = read_json(folder / 'results.json')
    previous_spec = read_json(PRIOR / 'refinement_spec.json')
    trials = prior.validate_spec(previous_spec)
    for path, digest in previous_spec.get('input_fingerprints', {}).items():
        assets.verify(path, digest, 'Round1 specification input')
    rows, prepared, shared_config, initial_ledger = validate_initial(assets)
    require(frozen['prepared_fisher'] == prepared and frozen['shared_scientific_config'] == shared_config,
            'Round1 shared Fisher or scientific config differs from the initial sweep')
    require(complete['raw_fisher_sha256'] == prepared['payload_sha256'], 'Round1 completion raw Fisher mismatch')
    for trial in trials:
        row, config = prior.validate_refinement_trial(PRIOR, trial, prepared, assets, tensor_cache)
        require(row['status'] == 'observed' and config == shared_config, 'Frozen round1 trial is missing or scientifically changed')
        row.update(source='refinement', source_round=1, trial_id=trial['trial_id'])
        # Native divergence receipts do not hash finite-prefix telemetry. Pin all
        # three files here, and compare the recomputed summaries below.
        for name in ('training_batch_metrics.jsonl', 'rng_receipts.jsonl', 'runtime_receipts.jsonl'):
            path = Path(row['run_dir']) / name
            if path.is_file():
                assets.hash(path)
        rows.append(row)
    rows.sort(key=lambda row: row['ewc_lambda'])
    require(len(rows) == len({row['ewc_lambda'] for row in rows}) == 20, 'Frozen history must have 20 unique lambdas')
    for row in rows:
        mechanism_evidence(row)
    selection = select_points(rows, previous_spec['efficacy_thresholds'])
    require(rows == frozen['individual'], 'Frozen round1 report rows differ from revalidated native evidence')
    require(selection == frozen['selection'], 'Frozen round1 selection differs from revalidated evidence')
    require(sum(row['outcome'] == 'complete' for row in rows) == 17
            and sum(row['outcome'] == 'diverged' for row in rows) == 3,
            'Historical completed/divergent observations changed')
    return rows, prepared, shared_config, initial_ledger


# Copied verbatim from the SHA-pinned round1 native verifier, except receipt
# round validation now reads and validates the actual round2 specification.
# No receipt is rewritten; every native science/telemetry/CPU displacement
# check is preserved. Round1 itself uses its original verifier above.
def validate_refinement_trial(root, trial_spec, prepared, assets, tensor_cache):
    lam = trial_spec['ewc_lambda']
    run = root / 'training' / trial_spec['trial_id']
    row = dict(ewc_lambda=lam, status='pending', outcome=None, **{k: None for k in METRICS}, run_dir=str(run))
    result_path = run / 'TRIAL_RESULT.json'
    if not result_path.is_file():
        row.update(status='failed' if (run / 'FAILED.json').exists() else 'pending',
                   note='No validated observed trial receipt; native checkpoints alone do not count')
        return row, None
    require(prepared is not None, f'{run.name}: observed trial lacks common prepared Fisher')
    trial = read_json(result_path)
    require(trial.get('trial_id') == trial_spec['trial_id'], 'Refinement wrapper trial ID mismatch')
    actual_spec = read_json(root / 'refinement_spec.json')
    validate_spec(actual_spec)
    require(trial_spec in actual_spec['trials'], 'Trial does not belong to the actual round2 specification')
    require(trial.get('round') == actual_spec['round'] and Path(trial['spec']).resolve() == root / 'refinement_spec.json',
            'Refinement receipt round/spec mismatch')
    assets.verify(root / 'refinement_spec.json', trial['spec_sha256'], 'Refinement receipt specification')
    assets.verify(root / 'run_trial.py', trial['wrapper_sha256'], 'Refinement wrapper')
    validate_prepared_identity(trial['prepared_fisher'], prepared, run.name + '/wrapper')
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



def render_markdown(data):
    # Preserve the frozen scientific definitions; only round/history labels vary.
    return (prior.render_markdown(data)
            .replace('# Raw Fisher λ 细化：第 1 轮', '# Raw Fisher λ 细化：第 2 轮', 1)
            .replace(f"合并已观察 {data['total_observed_points']}/20", f"合并已观察 {data['total_observed_points']}/26", 1)
            .replace('初始 14 点及其报告保持原样；此报告只读核验其哈希与原生收据，再合并本轮 6 个实测点。',
                     '初始 14 点和第 1 轮 6 点及其报告保持原样；此报告只读核验两轮历史的哈希与原生收据，'
                     '保留所有负结果和发散点，再合并本轮 6 个实测点。', 1))


def build_report(root, output, allow_partial=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not any(root.is_relative_to(path.resolve()) for path in (INITIAL, PRIOR))
            and output.is_relative_to(root) and output != root,
            'Round2 report must use its own output subdirectory; historical sources are immutable')
    require(not any(output.is_relative_to(root / name) for name in ('training', 'prepared_fisher')),
            'Report cannot overwrite training or Fisher artifacts')
    # Fail closed when recollection finds missing or invalid source evidence.
    (output / 'COMPLETE.json').unlink(missing_ok=True)
    spec_path = root / 'refinement_spec.json'
    spec = read_json(spec_path)
    trials = validate_spec(spec)
    assets, tensor_cache = EvidenceAssets(), {}
    for path, digest in spec.get('input_fingerprints', {}).items():
        assets.verify(path, digest, 'Round2 specification input')
    rows, prepared, shared_config, initial_ledger = validate_history(assets, tensor_cache)
    require([row['ewc_lambda'] for row in rows] == spec['tested_lambda_union'], 'Verified historical lambda union changed')
    require(all(row['status'] == 'observed' for row in rows), 'Historical observations must all be terminal')
    history_ledger = dict(assets.ledger)
    refinement_rows = []
    for trial in trials:
        row, config = validate_refinement_trial(root, trial, prepared, assets, tensor_cache)
        require(config is None or config == shared_config, 'Round2 settings differ from the initial raw sweep')
        row.update(source='refinement', source_round=2, trial_id=trial['trial_id'])
        refinement_rows.append(row)
        rows.append(row)
        for name in ('training_batch_metrics.jsonl', 'rng_receipts.jsonl', 'runtime_receipts.jsonl'):
            path = root / 'training' / trial['trial_id'] / name
            if path.is_file():
                assets.hash(path)
    rows.sort(key=lambda row: row['ewc_lambda'])
    require(len(rows) == len({row['ewc_lambda'] for row in rows}) == 26,
            'Combined report must have 26 unique points')
    require(len({row['trial_id'] for row in rows}) == 26, 'Combined trial IDs must be unique')
    for row in rows:
        mechanism_evidence(row)
    selection = select_points(rows, spec['efficacy_thresholds'])
    observed = sum(row['status'] == 'observed' for row in refinement_rows)
    all_observed = observed == 6 and selection['baseline_complete']
    scientific_status = ('criterion_met' if selection['qualified_lambdas'] else 'refinement_required') if all_observed else 'pending_round'
    data = dict(status=scientific_status if all_observed else 'partial', round=2,
                round_status='observed' if all_observed else 'partial', scientific_status=scientific_status,
                generated_utc=datetime.now(timezone.utc).isoformat(), seed=0,
                expected_refinement_points=6, observed_refinement_points=observed,
                total_expected_points=26, total_observed_points=sum(row['status'] == 'observed' for row in rows),
                completed_lambda_points=sum(row['outcome'] == 'complete' for row in rows),
                diverged_lambda_points=sum(row['outcome'] == 'diverged' for row in rows),
                missing_refinement_trials=[row['trial_id'] for row in refinement_rows if row['status'] != 'observed'],
                individual=rows, selection=selection, efficacy_thresholds=spec['efficacy_thresholds'],
                prepared_fisher=prepared, initial_evidence_sha256=initial_ledger,
                historical_evidence_sha256=history_ledger,
                prior_refinement_rounds=spec['prior_refinement_rounds'],
                input_evidence_sha256=spec.get('input_fingerprints', {}),
                initial_complete_sha256=INITIAL_COMPLETE_SHA256, initial_reporter_sha256=INITIAL_REPORTER_SHA256,
                prior_complete_sha256=PRIOR_COMPLETE_SHA256, prior_reporter_sha256=PRIOR_REPORTER_SHA256,
                reporter_sha256=assets.hash(__file__), spec_sha256=assets.hash(spec_path),
                shared_scientific_config=shared_config, gpu_evaluation_performed=False,
                additional_round_launched=False,
                historical_references_not_raw_points=read_json(INITIAL / 'sweep_spec.json').get('historical_references_not_raw_points', {}))
    data['evidence_sha256'] = dict(assets.ledger)
    for path, digest in data['evidence_sha256'].items():
        assets.verify(path, digest, 'Evidence during collection')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'results.json', data)
    fields = ['ewc_lambda', 'trial_id', 'source', 'source_round', 'status', 'outcome'] + METRICS + [
        'observed_finite_steps', 'gradient_component_samples', 'telemetry_scope', 'raw_fisher_sha256',
        'parent_checkpoint_sha256', 'grad_P_norm_step1', 'batch_loss_equals_CE_plus_P',
        'delta_d_QV_cont', 'delta_raw_fisher_weighted_drift_squared', 'raw_fisher_weighted_drift_ratio_vs_lambda0',
        'run_dir', 'note']
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'RESULTS.md').write_text(render_markdown(data))
    if all_observed:
        atomic_json(output / 'COMPLETE.json', dict(status='complete', round=2, round_status='observed',
                    scientific_status=scientific_status, observed_refinement_points=6, total_observed_points=26,
                    completed_lambda_points=data['completed_lambda_points'], diverged_lambda_points=data['diverged_lambda_points'],
                    seed=0, baseline_lambda0_complete=True, qualified_lambdas=selection['qualified_lambdas'],
                    gpu_evaluation_performed=False, additional_round_launched=False,
                    initial_complete_sha256=INITIAL_COMPLETE_SHA256, prior_complete_sha256=PRIOR_COMPLETE_SHA256,
                    reporter_sha256=data['reporter_sha256'], spec_sha256=data['spec_sha256'],
                    raw_fisher_sha256=prepared['payload_sha256'],
                    outputs_sha256={name: sha(output / name) for name in ('RESULTS.md', 'results.csv', 'results.json')}))
    elif not allow_partial:
        raise RuntimeError(f'Refinement remains partial: {observed}/6 observed')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement-root', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    data = build_report(args.refinement_root, args.output_dir or args.refinement_root / 'report', args.allow_partial)
    print(json.dumps({key: data[key] for key in ('status', 'round_status', 'scientific_status',
                                               'observed_refinement_points', 'total_observed_points')}))


if __name__ == '__main__':
    main()
