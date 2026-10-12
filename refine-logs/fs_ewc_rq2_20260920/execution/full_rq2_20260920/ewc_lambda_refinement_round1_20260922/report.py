#!/usr/bin/env python3
"""CPU collector for six raw-Fisher refinement points plus the immutable first 14."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
INITIAL = HERE.parent / 'ewc_lambda_sweep_seed0_20260921'
INITIAL_REPORTER_SHA256 = '1808e5b1c9767e8efb61ae3ff7ff76f511760ded087dfd904a710baa62377e67'
INITIAL_COMPLETE_SHA256 = '15aa8beef331a59a24ffd360ebe16dc3261bd26d9f6854898c01040c8ebd1140'


def load_initial_helpers():
    import hashlib
    path = INITIAL / 'report.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != INITIAL_REPORTER_SHA256:
        raise ValueError('Frozen initial report helper changed')
    spec = importlib.util.spec_from_file_location('frozen_initial_raw_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_initial_helpers()
require, read_json, jsonl = old.require, old.read_json, old.jsonl
numeric, sha, atomic_json = old.numeric, old.sha, old.atomic_json
validate_prepared_identity, validate_pairing = old.validate_prepared_identity, old.validate_pairing
telemetry, drift_from_states = old.telemetry, old.drift_from_states
FISHER_POLICY, METRICS, COMMON_CONFIG = old.FISHER_POLICY, old.METRICS, old.COMMON_CONFIG


def validate_spec(spec):
    require(spec.get('round') == 1 and spec.get('seed') == 0, 'Only the first seed0 refinement round is supported')
    require(Path(spec['initial_sweep']).resolve() == INITIAL.resolve(), 'Initial sweep must be the immutable completed raw sweep')
    trials = spec['trials']
    require(len(trials) == 6, 'First refinement round requires six trials')
    ids, lambdas = set(), []
    for trial in trials:
        trial_id = trial['trial_id']
        require(isinstance(trial_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', trial_id)
                and trial_id not in ids, 'Trial IDs must be unique safe directory names')
        lam = numeric(trial['ewc_lambda'], 'refinement lambda')
        require(lam > 0 and lam not in old.GRID and lam not in lambdas, 'Refinement lambda must be new, unique, and positive')
        ids.add(trial_id)
        lambdas.append(lam)
    expected = sorted(base * 3 ** fraction for base in (10000., 100000.) for fraction in (.25, .5, .75))
    require(all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(sorted(lambdas), expected)),
            'Six first-round lambdas must be the log-quarter points of the two authorized intervals')
    require(spec['efficacy_thresholds'] == dict(retention_gain_pp=1.0, max_new_task_drop_pp=5.0,
                                              nondecreasing_average_gain_pp=0.0), 'Predefined efficacy thresholds changed')
    return trials


def validate_initial(assets):
    """Trust the pinned report only after rechecking its receipt and asset hash chain."""
    folder = INITIAL / 'report'
    ledger = {}
    def verified(path, digest, label):
        assets.verify(path, digest, label)
        ledger[str(Path(path).resolve())] = digest
    verified(INITIAL / 'report.py', INITIAL_REPORTER_SHA256, 'Initial report helper')
    verified(folder / 'COMPLETE.json', INITIAL_COMPLETE_SHA256, 'Immutable initial completion')
    complete = read_json(folder / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['observed_lambda_points'] == 14
            and complete['baseline_lambda0_complete'] is True, 'Initial 14-point sweep is not complete')
    for name in ('RESULTS.md', 'results.csv', 'results.json'):
        verified(folder / name, complete['outputs_sha256'][name], 'Immutable initial report output')
    verified(INITIAL / 'sweep_spec.json', complete['spec_sha256'], 'Initial sweep specification')
    data = read_json(folder / 'results.json')
    initial_spec = read_json(INITIAL / 'sweep_spec.json')
    old.validate_spec(initial_spec)
    prepared = old.validate_prepared(INITIAL, initial_spec, assets)
    require(prepared is not None and prepared == data['prepared_fisher'], 'Initial raw Fisher or shared parent changed')
    verified(prepared['receipt_path'], prepared['receipt_sha256'], 'Shared Fisher preparation receipt')
    verified(prepared['payload_path'], prepared['payload_sha256'], 'Shared raw Fisher')
    rows = copy.deepcopy(data['individual'])
    require([r['ewc_lambda'] for r in rows] == old.GRID and len(rows) == 14, 'Initial point grid changed')
    for row in rows:
        lam = row['ewc_lambda']
        run = INITIAL / 'training' / f'lambda_{lam}_seed0'
        require(Path(row['run_dir']).resolve() == run and row['status'] == 'observed'
                and row['outcome'] in ('complete', 'diverged'), 'Initial row is not an observed original point')
        trial_path, manifest_path = run / 'TRIAL_RESULT.json', run / 'run_manifest.json'
        native_path = run / ('COMPLETE.json' if row['outcome'] == 'complete' else 'DIVERGED.json')
        verified(trial_path, row['trial_receipt_sha256'], 'Initial trial receipt')
        verified(native_path, row['native_receipt_sha256'], 'Initial native receipt')
        verified(manifest_path, row['manifest_sha256'], 'Initial manifest')
        trial, native, manifest = read_json(trial_path), read_json(native_path), read_json(manifest_path)
        require(trial['outcome'] == native['status'] == row['outcome'] and trial['status'] == 'observed'
                and trial['ewc_lambda'] == lam and trial['seed'] == 0, 'Initial trial outcome changed')
        require(trial['identity_sha256'] == native['identity_sha256'] == manifest['identity_sha256'], 'Initial identity chain mismatch')
        require(trial['native_receipt_sha256'] == row['native_receipt_sha256']
                and trial['manifest_sha256'] == row['manifest_sha256'], 'Initial receipt chain mismatch')
        require(row['raw_fisher_sha256'] == prepared['payload_sha256']
                and row['parent_checkpoint_sha256'] == prepared['parent_checkpoint_sha256'], 'Initial Fisher/parent mismatch')
        validate_prepared_identity(manifest['prepared_fisher'], prepared, run.name)
        for key in ('ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher'):
            require(manifest['requested_config'][key] is False and manifest['config'][key] is False,
                    'Initial point is not raw Fisher')
        verified(INITIAL / 'train.py', manifest['runner_sha256'], 'Initial trainer')
        if row['outcome'] == 'complete':
            validation = native['validation']
            require(validation['accuracy_CIL'] == {k: row[k] for k in ('a11', 'a21', 'a22')}, 'Initial metric receipt mismatch')
            require(validation['epochs_per_new_task'] == 50 and row['observed_finite_steps'] == 400,
                    'Initial successful point is not full training')
            for name, key in [('runtime_receipts.jsonl', 'runtime_receipts_sha256'),
                              ('rng_receipts.jsonl', 'rng_receipts_sha256'),
                              ('training_batch_metrics.jsonl', 'training_batch_metrics_sha256')]:
                verified(run / name, validation[key], 'Initial completed telemetry')
            verified(validation['metrics'], validation['metrics_sha256'], 'Initial accuracy matrix')
            for relative, asset in validation['checkpoint_assets'].items():
                require(Path(relative).parts[0] == 'checkpoints' and '..' not in Path(relative).parts,
                        'Invalid initial checkpoint asset path')
                verified(run / relative, asset['sha256'], 'Initial completed checkpoint')
        else:
            require(all(row[k] is None for k in ('a11', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont')),
                    'Initial divergence cannot have formal endpoint metrics')
            # These prefix files were not covered by native DIVERGED, so verify
            # their actual contents against the already pinned report summaries.
            actual = telemetry(run, lam, False, {})
            require(all(row.get(k) == value for k, value in actual.items()), 'Initial divergence prefix changed')
            pairing = validate_pairing(run, False, assets,
                                      INITIAL.parent / 'formal_training/sequence_sgd_seed0/rng_receipts.jsonl')
            require(pairing == row['pairing'], 'Initial divergence pairing changed')
            for name in ('training_batch_metrics.jsonl', 'rng_receipts.jsonl', 'runtime_receipts.jsonl'):
                if (run / name).is_file():
                    ledger[str(run / name)] = assets.hash(run / name)
        row.update(source='initial_sweep', source_round=0, trial_id=run.name)
    require(sum(r['outcome'] == 'complete' for r in rows) == 12 and sum(r['outcome'] == 'diverged' for r in rows) == 2,
            'Initial terminal counts changed')
    return rows, prepared, data['shared_scientific_config'], ledger


# The full native trial verifier below is copied from the SHA-pinned initial
# report. Only the run ID / float-lambda entry point and wrapper trial ID check
# differ. Every loss, raw-Fisher, RNG, 50-epoch, and endpoint check is preserved.
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
    require(trial.get('round') == 1 and Path(trial['spec']).resolve() == root / 'refinement_spec.json',
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


def select_points(rows, thresholds):
    selection = old.summarize(rows, thresholds)
    active = [row['ewc_lambda'] for row in rows if row['outcome'] == 'complete' and row['ewc_lambda'] > 0
              and numeric(row['P_mean'], 'mean P') > 0 and numeric(row['grad_P_norm_step1'], 'grad P after first update') > 0
              and row.get('batch_loss_equals_CE_plus_P') is True]
    selection['mechanistically_active_lambdas'] = active
    selection['qualified_lambdas'] = [lam for lam in selection['balanced_retention_lambdas'] if lam in active]
    selection['qualified_observed_span'] = ([min(selection['qualified_lambdas']), max(selection['qualified_lambdas'])]
                                           if selection['qualified_lambdas'] else None)
    selection['qualification_rule'] = ('Completed finite point; lambda>0; mean P>0 and mean ||grad P||>0; '
                                       'delta a21>=1 pp; delta a22>=-5 pp; delta average>=0 pp relative to initial fresh lambda0')
    selection['qualification_rule'] = selection['qualification_rule'].replace('mean ||grad P||>0', 'step1 ||grad P||>0 and loss=CE+P')
    baseline = next(row for row in rows if row['ewc_lambda'] == 0 and row['outcome'] == 'complete')
    for row in rows:
        if row['outcome'] == 'complete':
            row['delta_d_QV_cont'] = row['d_QV_cont'] - baseline['d_QV_cont']
            row['delta_raw_fisher_weighted_drift_squared'] = (
                row['raw_fisher_weighted_drift_squared'] - baseline['raw_fisher_weighted_drift_squared'])
            row['raw_fisher_weighted_drift_ratio_vs_lambda0'] = (
                row['raw_fisher_weighted_drift_squared'] / baseline['raw_fisher_weighted_drift_squared']
                if baseline['raw_fisher_weighted_drift_squared'] > 0 else None)
    qualifying = [row for row in rows if row['ewc_lambda'] in selection['qualified_lambdas']]
    selection['ranked_qualified_lambdas'] = [row['ewc_lambda'] for row in sorted(
        qualifying, key=lambda r: (-r['average_accuracy'], -r['a21'], r['ewc_lambda']))]
    selection['geometry_review_required_lambdas'] = [row['ewc_lambda'] for row in qualifying
                                                   if row['delta_raw_fisher_weighted_drift_squared'] >= 0]
    selection['lambda_locked'] = False
    return selection


def mechanism_evidence(row):
    if row['outcome'] != 'complete':
        return
    batches = jsonl(Path(row['run_dir']) / 'training_batch_metrics.jsonl')
    second = [r for r in batches if r['step'] == 1]
    require(len(second) == 1 and second[0]['gradient_components_sampled'] is True,
            'Successful point lacks gradient evidence after first update')
    row['grad_P_norm_step1'] = numeric(second[0]['grad_fisher_norm'], 'grad P after first update')
    require(len(batches) == 400 and all(math.isclose(r['loss'], r['ce_clean'] + r['ewc'],
                                                   rel_tol=1e-6, abs_tol=1e-7) for r in batches),
            'Recorded native SGD objective differs from CE+P')
    row['batch_loss_equals_CE_plus_P'] = True


def render_markdown(data):
    def fmt(value):
        if value is None:
            return '—'
        if isinstance(value, (int, float)):
            return f'{value:.12g}'
        return str(value).replace('|', '/')
    lines = ['# Raw Fisher λ 细化：第 1 轮', '',
             f"本轮状态：{data['round_status']}；科学判定：{data['scientific_status']}。",
             f"新点已观察 {data['observed_refinement_points']}/6；合并已观察 {data['total_observed_points']}/20。"
             f"成功训练 {data['completed_lambda_points']} 点，科学发散 {data['diverged_lambda_points']} 点。", '',
             '初始 14 点及其报告保持原样；此报告只读核验其哈希与原生收据，再合并本轮 6 个实测点。'
             '所有点均为同一 seed0、SGD → SGD、同一 Task0 模型和同一未做 trace normalization 的 Fisher。', '',
             '| λ | 来源轮次 | 终态 | a11 | a21 | a22 | BWT | 两任务平均 | d_QV_cont | ΣrawF·ΔBA² | CE mean | P mean | P/CE | gradP/gradCE |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    columns = ['ewc_lambda', 'source_round', 'outcome', 'a11', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont',
               'raw_fisher_weighted_drift_squared', 'CE_mean', 'P_mean', 'P_over_CE', 'grad_P_over_grad_CE']
    for row in data['individual']:
        lines.append('| ' + ' | '.join(fmt(row.get(k) if k != 'outcome' else row.get(k) or row['status']) for k in columns) + ' |')
    selection = data['selection']
    lines += ['', f"所有成功训练点中，旧任务保持最高 λ：{selection['best_old_retention']}；平均准确率最高 λ：{selection['best_average']}。", '',
              '判定条件预先固定为：实际 P mean 为正、第一次更新后的 step1 正则梯度为正，且各批次 loss=CE+P；相对初始新跑 λ=0，a21 至少提高 1 个百分点、a22 降幅不超过 5 个百分点、平均准确率不下降。',
              f"达到全部条件的实测 λ：{selection['qualified_lambdas']}。实测跨度：{selection['qualified_observed_span']}。",
              '跨度只表示合格离散观测点的最小/最大值，不推断中间未测点有效。', '',
              '| λ | Δa21 (pp) | Δa22 (pp) | Δ平均 (pp) | 有限更新数 | 梯度分量样本数 |',
              '|---|---|---|---|---|---|']
    for row in data['individual']:
        lines.append('| ' + ' | '.join(fmt(row.get(k)) for k in ['ewc_lambda', 'delta_a21_pp', 'delta_a22_pp',
                     'delta_average_pp', 'observed_finite_steps', 'gradient_component_samples']) + ' |')
    if data['scientific_status'] == 'refinement_required':
        lines += ['', '本轮全部点已观察，但仍没有点同时达到以上条件；科学状态为 refinement_required。'
                  '该状态不等于队列失败，也不把“表现最好的点”当作已证实有效。此报告不会启动后续训练。']
    lines += ['', f"按平均准确率、旧任务准确率、较小 λ 排序的候选：{selection['ranked_qualified_lambdas']}。",
              f"准确率达标但 rawF 加权位移没有减少、需先解释的候选：{selection['geometry_review_required_lambdas']}。",
              '本报告未锁定 λ，也未创建 SELECTION.json；几何证据与相对 λ=0 的位移变化完整保存在 CSV/JSON。']
    lines += ['', '数值口径与限制：', '',
              '- 准确率来自 CIL 矩阵，单位为百分比；BWT=a21−a11。仅单 seed、测试集探索，不能代替跨 seed 确认。',
              '- d_QV_cont 与 rawF 加权位移使用持续同一分支的 24 个 Q/V BA，从 Task0 到 Task1；排除分类头，不是 A/B 因子距离。',
              '- P 已含 λ/2；P/CE 是批次均值之比。gradP/gradCE 是 steps 0、1、8、16… 的梯度范数比均值；分母包括新分类头的 CE 梯度。',
              '- 发散点没有正式准确率、端点位移或排名；其损失和梯度若有数值，仅代表发散前有限前缀。',
              '- 完成点要求 50 轮、400 个有限更新、配对 Task1 新头和 51 次 loader 遍历。部分遍历仅验证采样器和首批；不声称哈希覆盖每个增强批次。',
              '- 旧 trace-normalized λ=2000 仅为历史参考，不属于本报告的 rawF 网格、baseline 或效力判定。', '', '未完成/发散说明：', '']
    for row in data['individual']:
        if row.get('note'):
            lines.append(f"- λ={fmt(row['ewc_lambda'])}（{row['trial_id']}）：{row['note']}")
    return '\n'.join(lines) + '\n'


def build_report(root, output, allow_partial=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root != INITIAL.resolve() and output.is_relative_to(root) and output != root,
            'Refinement report must use its own output subdirectory; initial report is immutable')
    require(not any(output.is_relative_to(root / name) for name in ('training', 'prepared_fisher')),
            'Report cannot overwrite training or Fisher artifacts')
    spec_path = root / 'refinement_spec.json'
    spec = read_json(spec_path)
    trials = validate_spec(spec)
    assets, tensor_cache = old.Assets(), {}
    for path, digest in spec.get('input_fingerprints', {}).items():
        assets.verify(path, digest, 'Refinement specification input')
    initial_rows, prepared, shared_config, initial_ledger = validate_initial(assets)
    rows = list(initial_rows)
    refinement_rows = []
    for trial in trials:
        row, config = validate_refinement_trial(root, trial, prepared, assets, tensor_cache)
        require(config is None or config == shared_config, 'Refinement settings differ from the initial raw sweep')
        row.update(source='refinement', source_round=1, trial_id=trial['trial_id'])
        refinement_rows.append(row)
        rows.append(row)
    rows.sort(key=lambda r: r['ewc_lambda'])
    require(len(rows) == 20 and len({r['ewc_lambda'] for r in rows}) == 20, 'Combined report must have 20 unique points')
    for row in rows:
        mechanism_evidence(row)
    selection = select_points(rows, spec['efficacy_thresholds'])
    observed = sum(row['status'] == 'observed' for row in refinement_rows)
    all_observed = observed == 6 and selection['baseline_complete']
    scientific_status = ('criterion_met' if selection['qualified_lambdas'] else 'refinement_required') if all_observed else 'pending_round'
    data = dict(status=scientific_status if all_observed else 'partial', round=1,
                round_status='observed' if all_observed else 'partial', scientific_status=scientific_status,
                generated_utc=datetime.now(timezone.utc).isoformat(), seed=0,
                expected_refinement_points=6, observed_refinement_points=observed,
                total_expected_points=20, total_observed_points=14 + observed,
                completed_lambda_points=sum(r['outcome'] == 'complete' for r in rows),
                diverged_lambda_points=sum(r['outcome'] == 'diverged' for r in rows),
                missing_refinement_trials=[r['trial_id'] for r in refinement_rows if r['status'] != 'observed'],
                individual=rows, selection=selection, efficacy_thresholds=spec['efficacy_thresholds'],
                prepared_fisher=prepared, initial_evidence_sha256=initial_ledger,
                input_evidence_sha256=spec.get('input_fingerprints', {}),
                initial_complete_sha256=INITIAL_COMPLETE_SHA256, initial_reporter_sha256=INITIAL_REPORTER_SHA256,
                reporter_sha256=assets.hash(__file__), spec_sha256=assets.hash(spec_path),
                shared_scientific_config=shared_config, gpu_evaluation_performed=False,
                additional_round_launched=False,
                historical_references_not_raw_points=read_json(INITIAL / 'sweep_spec.json').get('historical_references_not_raw_points', {}))
    # Recheck immutable inputs immediately before publication, with no writes to
    # INITIAL. Assets caches include mtime and size, so changed files are rehashed.
    for path, digest in initial_ledger.items():
        assets.verify(path, digest, 'Initial evidence during collection')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'COMPLETE.json').unlink(missing_ok=True)
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
        atomic_json(output / 'COMPLETE.json', dict(status='complete', round_status='observed',
                    scientific_status=scientific_status, observed_refinement_points=6, total_observed_points=20,
                    completed_lambda_points=data['completed_lambda_points'], diverged_lambda_points=data['diverged_lambda_points'],
                    seed=0, baseline_lambda0_complete=True, qualified_lambdas=selection['qualified_lambdas'],
                    gpu_evaluation_performed=False, additional_round_launched=False,
                    initial_complete_sha256=INITIAL_COMPLETE_SHA256, reporter_sha256=data['reporter_sha256'],
                    spec_sha256=data['spec_sha256'], raw_fisher_sha256=prepared['payload_sha256'],
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
