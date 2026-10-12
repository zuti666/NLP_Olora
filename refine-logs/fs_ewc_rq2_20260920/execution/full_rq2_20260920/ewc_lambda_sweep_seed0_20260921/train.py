#!/usr/bin/env python3
"""Seed0 SGD -> SGD+EWC continuation with recomputed native RAW Task0 Fisher.

The historical learner implements every optimizer update. Runtime adapters only
make pretrained loading, complete resume, RNG pairing, and receipts explicit.
Nothing is written to the original CopyFlatCV tree or frozen source snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import math
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
EXECUTION = FORMAL.parent
ORIGINAL = Path('/data/140-1/users/liying/CopyFlatCV')
ANCHORS = ORIGINAL / 'outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50'
PROTOCOL = 'rq2_shared_task_rng_v1'
ARM = 'ewc_lambda_sweep'
MAX_EWC_LAMBDA = 1e9
PARENT = FORMAL / 'formal_training/anchor_sgd_seed0'
FISHER_POLICY = 'recomputed_native_task0_raw_fisher_no_trace_normalization'


def finite_ewc_lambda(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= MAX_EWC_LAMBDA:
        raise argparse.ArgumentTypeError('EWC lambda must be finite and in [0, 1e9].')
    return value


def finite_number(value, name):
    if not math.isfinite(float(value)):
        raise FloatingPointError(f'Nonfinite {name}: {value}')
    return float(value)


def sample_gradient_components(step):
    # Include the second batch: Task1 starts exactly at the EWC anchor, so the
    # first batch's EWC gradient is zero even when lambda is nonzero.
    return step == 1 or step % 8 == 0


def observe_sgd_gradients(model, param_groups, sample_components):
    """Read actual update gradients; optional EWC probe never writes .grad or RNG.

    The frozen SGD branch adds only CE and EWC (all other penalties disabled).
    Thus clean = actual combined - re-differentiated native EWC. The subtraction
    uses float64 for measurement, while the optimizer's float32 update is intact.
    """
    import torch
    params = [p for group in param_groups for p in group['params'] if p.requires_grad]
    combined_sq = 0.
    for index, param in enumerate(params):
        if param.grad is not None:
            if not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError(f'Nonfinite actual SGD gradient at parameter {index}')
            combined_sq += float(param.grad.detach().double().square().sum().item())
    record = {'grad_combined_norm': finite_number(math.sqrt(combined_sq), 'combined gradient norm'),
              'gradient_components_sampled': bool(sample_components)}
    if not sample_components:
        return record
    fisher_grads = [None] * len(params)
    if float(model._ewc_lambda) != 0. and model._fisher_past_delta:
        with torch.enable_grad():
            penalty = model._past_fisher_penalty(model._unwrap_network())
            finite_number(penalty.detach().item(), 'diagnostic EWC penalty')
            if penalty.requires_grad:
                fisher_grads = torch.autograd.grad(penalty, params, allow_unused=True)
    clean_sq = fisher_sq = dot = 0.
    for param, fisher_grad in zip(params, fisher_grads):
        if param.grad is None and fisher_grad is None:
            continue
        combined = torch.zeros_like(param, dtype=torch.float64) if param.grad is None else param.grad.detach().double()
        fisher = torch.zeros_like(combined) if fisher_grad is None else fisher_grad.detach().double()
        if not bool(torch.isfinite(fisher).all()):
            raise FloatingPointError('Nonfinite diagnostic EWC gradient')
        clean = combined - fisher
        clean_sq += float(clean.square().sum().item())
        fisher_sq += float(fisher.square().sum().item())
        dot += float((clean * fisher).sum().item())
    clean_norm, fisher_norm = math.sqrt(clean_sq), math.sqrt(fisher_sq)
    record.update(grad_clean_norm=clean_norm, grad_fisher_norm=fisher_norm,
                  fisher_to_clean_norm_ratio=fisher_norm / (clean_norm + 1e-12),
                  cos_clean_fisher=dot / (clean_norm * fisher_norm + 1e-12),
                  component_source='native_EWC_autograd; clean=actual_combined-EWC; float64_norms')
    for key in ('grad_clean_norm', 'grad_fisher_norm', 'fisher_to_clean_norm_ratio', 'cos_clean_fisher'):
        finite_number(record[key], key)
    return record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_fisher_identity(cfg, *, allow_missing=False):
    """Bind the native recomputation receipt and payload before any execution."""
    path = Path(cfg['rq2_prepared_fisher']).resolve()
    receipt_path = path.parent / 'PREPARED.json'
    if not path.is_file() or not receipt_path.is_file():
        if allow_missing:
            return {'status': 'pending', 'execution_ready': False, 'path': str(path),
                    'receipt_path': str(receipt_path), 'fisher_policy': FISHER_POLICY}
        raise FileNotFoundError(f'Prepared raw Fisher payload and PREPARED.json are mandatory: {path}')
    receipt = json.loads(receipt_path.read_text())
    required = {'status': 'complete', 'seed': 0, 'task': 0,
                'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
                'fisher_mode': 'batch_empirical', 'fisher_source': 'fit',
                'tensor_count': 24, 'weights_unchanged': True}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError('Prepared Fisher receipt is not complete native raw Task0/seed0 recomputation.')
    if receipt.get('exact_inputs_receipt', {}).get('matched') is not True:
        raise ValueError('Prepared Fisher must certify exact formal Task0 fit-Fisher input pairing.')
    if (receipt.get('parent_checkpoint_sha256') != sha256(PARENT / 'checkpoints/task_00.pt')
            or receipt.get('source_manifest_sha256') != sha256(EXECUTION / 'source_frozen.json')
            or receipt.get('payload_sha256') != sha256(path)):
        raise ValueError('Prepared Fisher parent/source/payload fingerprint mismatch.')
    if not math.isfinite(float(receipt.get('raw_trace', math.nan))) or float(receipt['raw_trace']) <= 0:
        raise ValueError('Prepared raw Fisher trace must be finite and positive.')
    if not isinstance(receipt.get('numel'), int) or receipt['numel'] <= 0:
        raise ValueError('Prepared raw Fisher element count is missing or invalid.')
    return {'status': 'verified', 'execution_ready': True, 'path': str(path),
            'receipt_path': str(receipt_path), 'receipt_sha256': sha256(receipt_path),
            'payload_sha256': receipt['payload_sha256'], 'raw_trace': receipt['raw_trace'],
            'tensor_count': receipt['tensor_count'], 'numel': receipt['numel'],
            'parent_checkpoint_sha256': receipt['parent_checkpoint_sha256'],
            'source_manifest_sha256': receipt['source_manifest_sha256'], 'fisher_policy': FISHER_POLICY}


def load_prepared_fisher(cfg, expected_shapes):
    import torch
    identity = prepared_fisher_identity(cfg)
    payload = torch.load(identity['path'], map_location='cpu', weights_only=True, mmap=True)
    fisher = payload.get('fisher_past_delta')
    if not isinstance(fisher, dict) or len(fisher) != 24 or set(fisher) != set(expected_shapes):
        raise ValueError('Prepared raw Fisher must contain exactly the 24 parent Q/V tensors.')
    for key, value in fisher.items():
        if (not torch.is_tensor(value) or tuple(value.shape) != tuple(expected_shapes[key])
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())):
            raise ValueError(f'Invalid prepared raw Fisher tensor: {key}')
    trace = sum(float(value.double().sum().item()) for value in fisher.values())
    if sum(value.numel() for value in fisher.values()) != identity['numel']:
        raise ValueError('Prepared raw Fisher element count differs from the receipt.')
    if trace <= 0 or not math.isfinite(trace):
        raise ValueError('Prepared raw Fisher must have finite positive trace.')
    if any(not math.isclose(trace, float(value), rel_tol=2e-6, abs_tol=1e-9)
           for value in (payload.get('raw_trace', math.nan), identity['raw_trace'])):
        raise ValueError('Prepared raw Fisher tensor trace differs from its recorded raw trace.')
    if not isinstance(payload.get('stats'), dict):
        raise ValueError('Prepared raw Fisher tensor statistics are missing.')
    return payload, identity


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def append_json(path, payload):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + '\n')


def task_seed(seed, task):
    return 20260920 + 100003 * int(seed) + 1009 * int(task)


def anchor_assets(method):
    manifest = ANCHORS / f'manifests/{method}_seed0.env'
    values = dict(line.split('=', 1) for line in manifest.read_text().splitlines() if '=' in line)
    return Path(values['S1_CKPT_DIR']), Path(values['S1_METRICS_JSON'])


def is_frozen_backbone(key):
    return key.startswith('backbone.') and not any(
        part in key for part in ('linear_a_', 'linear_b_', 'w_As.', 'w_Bs.')
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['continuation'], default='continuation')
    p.add_argument('--arm', choices=[ARM], default=ARM)
    p.add_argument('--anchor-method', choices=['sgd'], default='sgd')
    p.add_argument('--future-optimizer', choices=['sgd'], default='sgd')
    seeds = p.add_mutually_exclusive_group(required=True)
    seeds.add_argument('--seed', type=int, choices=[0])
    seeds.add_argument('--seed0', dest='seed', action='store_const', const=0)
    p.add_argument('--gpu', choices=list(map(str, range(7))))
    p.add_argument('--source', type=Path, default=EXECUTION / 'source')
    p.add_argument('--output-root', type=Path, default=HERE / 'training')
    p.add_argument('--run-name', required=True)
    p.add_argument('--ewc-lambda', type=finite_ewc_lambda, required=True)
    p.add_argument('--max-train-tasks', type=int, choices=[2], default=2)
    p.add_argument('--anchor-dir', type=Path, default=PARENT / 'checkpoints')
    p.add_argument('--anchor-metrics', type=Path, default=PARENT / 'trainer_outputs')
    p.add_argument('--prepared-fisher', type=Path, default=HERE / 'prepared_fisher/fisher_state_raw.pt')
    p.add_argument('--workers', type=int, choices=[4], default=4)
    p.add_argument('--smoke', action='store_true',
                   help='Explicit one-epoch integration test; isolated outputs and SMOKE_COMPLETE only.')
    p.add_argument('--resume-last', action='store_true')
    p.add_argument('--print-config', action='store_true')
    p.add_argument('--validate-cpu', action='store_true',
                   help='Build and restore model on CPU; no forward, gradients, or training.')
    return p


def build_config(ns):
    import yaml
    base = ns.source / 'config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml'
    cfg = yaml.safe_load(base.read_text())
    label = ns.run_name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
        raise ValueError('--run-name must be a single safe path component.')
    if (ns.arm, ns.anchor_method, ns.future_optimizer, ns.seed) != (ARM, 'sgd', 'sgd', 0):
        raise ValueError('Only seed0, arm=ewc_lambda_sweep, SGD anchor and SGD future are allowed.')
    finite_ewc_lambda(ns.ewc_lambda)
    if ns.mode != 'continuation' or ns.max_train_tasks != 2 or ns.workers != 4:
        raise ValueError('Sweep requires task0 resume, task1 only, and workers=4.')
    if ns.source.resolve() != (EXECUTION / 'source').resolve():
        raise ValueError('Only the original frozen source is allowed.')
    if not ns.output_root.resolve().is_relative_to(HERE):
        raise ValueError('Outputs must stay inside this new extension directory.')
    if not ns.smoke and any(ns.output_root.resolve().is_relative_to(HERE / folder)
                            for folder in ('smoke', 'cpu_validation')):
        raise ValueError('Formal training cannot use smoke or CPU validation output directories.')
    output_root = HERE / 'smoke/training' if ns.smoke else ns.output_root.resolve()
    run_dir = output_root / label
    method = ns.future_optimizer
    cfg.update({
        'seed': [ns.seed], 'device': ['0'], 'prefix': label,
        'outputs_root': str(run_dir / 'trainer_outputs'),
        'save_task_checkpoint_dir': str(run_dir / 'checkpoints'),
        'save_task_checkpoint_required': True, 'save_final_model': True,
        'checkpoint_require_nonempty_fisher': True, 'resume_require_fisher_state': True,
        'resume_require_experiment_identity': False, 'resume_require_model_sidecars': False,
        'optimizer_type': method, 'optimizer_type_by_task': None,
        'optimizer_tag_override': label, 'lambda_flat': 0.0,
        'batch_size': 128, 'init_epoch': 50, 'epochs': 50,
        'init_lr': .04, 'lrate': .02, 'momentum': 0, 'init_momentum': 0,
        'weight_decay': 0., 'init_weight_decay': 0., 'scheduler': 'cosine',
        'ewc_lambda': ns.ewc_lambda, 'ewc_gamma': .9, 'ewc_eta': 0.,
        'ewc_max_batches': 100, 'fisher_mode': 'batch_empirical',
        'ewc_fisher_source': 'fit', 'fisher_split': False,
        'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
        'ewc_dual_ascent': False, 'ewc_project_grad': False,
        'approximate_fisher_in_task': False, 'weight_merge': False,
        'in_task_weight_merge': False, 'endpoint_weight_merge': False,
        'delta_w_penalty': False, 'gam_base_optimizer': 'sgd',
        'gam_adaptive': False, 'gam_grad_rho': .05, 'gam_grad_norm_rho': .2,
        'as1_rho': .05, 'as1_norm_rho': .2,
        'gam_grad_gamma': .1, 'gam_grad_beta_1': 1., 'gam_grad_beta_2': 1.,
        'gam_grad_beta_3': 1., 'data_root': '/data/115-2/users/liying/datasets',
        'train_num_workers': ns.workers, 'eval_num_workers': ns.workers,
        'persistent_workers': False, 'compute_nme_class_means': False,
        'mechanism_eval': True, 'mechanism_grad_stats': True,
        'compute_full_fisher': False, 'endpoint_eval': False,
        'task0_trajectory_eval': False, 'task0_fisher_recompute_eval': False,
        'flat_eval': False, 'probe_flatness_eval': False, 'multi_fisher_eval': False,
        'feature_drift_mode': 'none', 'feature_drift_eval': False,
        'max_train_tasks': 2,
        'rq2_rng_protocol': PROTOCOL,
        'rq2_extension_arm': ns.arm, 'rq2_smoke_only': bool(ns.smoke),
        'rq2_fisher_policy': FISHER_POLICY,
        'rq2_prepared_fisher': str(ns.prepared_fisher.resolve()),
    })
    cfg.update(resume_from_task=0, resume_from_checkpoint_dir=str(ns.anchor_dir.resolve()),
               resume_from_metrics_json=str(ns.anchor_metrics.resolve()))
    if ns.smoke:
        cfg['epochs'] = 1
    return cfg, run_dir


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_parent(ns, cfg, parent):
    """Require the completed, same-seed formal anchor, never a historical fallback."""
    expected = FORMAL / 'formal_training' / f'anchor_{ns.anchor_method}_seed{ns.seed}'
    if Path(cfg['resume_from_checkpoint_dir']).resolve() != (expected / 'checkpoints').resolve():
        raise ValueError(f'Expected formal same-seed parent {expected / "checkpoints"}.')
    manifest_path, complete_path = expected / 'run_manifest.json', expected / 'COMPLETE.json'
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    pcfg = manifest['requested_config']
    if (manifest.get('seed') != ns.seed or manifest.get('anchor_method') != ns.anchor_method
            or manifest.get('mode') != 'anchor' or manifest.get('protocol') != PROTOCOL
            or complete.get('status') != 'complete' or complete.get('completed_tasks') != 1
            or manifest.get('identity_sha256') != complete.get('identity_sha256')):
        raise ValueError('Parent seed/method/mode/protocol/complete identity does not match.')
    expected_config = {'seed': [ns.seed], 'optimizer_type': ns.anchor_method,
        'lambda_flat': 1. if ns.anchor_method == 'gam' else 0.,
        'ewc_lambda': 2000., 'init_epoch': 50, 'epochs': 50, 'batch_size': 128,
        'max_train_tasks': 1, 'train_num_workers': 4, 'eval_num_workers': 4,
        'dataset': 'imageneta', 'class_shuffle': False, 'lora_rank': 10,
        'fisher_mode': 'batch_empirical', 'ewc_normalize_fisher': True,
        'ewc_normalize_accumulated_fisher': True, 'approximate_fisher_in_task': False}
    if any(pcfg.get(k) != v for k, v in expected_config.items()):
        raise ValueError('Parent scientific configuration differs from the completed formal protocol.')
    if (manifest.get('matmul_allow_tf32') is not False or manifest.get('cudnn_allow_tf32') is not False
            or manifest['source_manifest_sha256'] != sha256(EXECUTION / 'source_frozen.json')
            or manifest['runner_sha256'] != sha256(FORMAL / 'train.py')):
        raise ValueError('Parent precision/source/runner fingerprint changed.')
    validation = complete['validation']
    if validation['validated_tasks'] != [0] or validation['epochs_per_new_task'] != 50:
        raise ValueError('Parent does not certify a complete 50-epoch Task0.')
    for name in ('task_00.pt', 'fisher_state_0.pt'):
        if parent[name]['sha256'] != validation['checkpoint_assets'][f'checkpoints/{name}']['sha256']:
            raise ValueError(f'Parent completed asset changed: {name}')
    metric_path = Path(parent['metrics']['path'])
    if (metric_path.resolve() != Path(validation['metrics']).resolve()
            or parent['metrics']['sha256'] != validation['metrics_sha256']):
        raise ValueError('Anchor metrics must be exactly the completed formal Task0 metrics.')
    raw = json.loads(metric_path.read_text())
    matrix = raw['cnn']['matrices']['final']
    if len(matrix) != 1 or len(matrix[0]) != 1 or not math.isfinite(float(matrix[0][0])):
        raise ValueError('Parent metrics must contain Task0 only.')
    for filename, key in [('runtime_receipts.jsonl', 'runtime_receipts_sha256'),
                          ('rng_receipts.jsonl', 'rng_receipts_sha256')]:
        if sha256(expected / filename) != validation[key]:
            raise ValueError(f'Parent receipt changed: {filename}')
    parent['run_manifest.json'] = {'path': str(manifest_path), 'sha256': sha256(manifest_path)}
    parent['COMPLETE.json'] = {'path': str(complete_path), 'sha256': sha256(complete_path)}
    return {'passed': True, 'seed': ns.seed, 'anchor_method': ns.anchor_method,
            'formal_parent': str(expected), 'parent_identity_sha256': manifest['identity_sha256']}


def experiment_identity(ns, cfg, *, allow_missing_prepared=False):
    """GPU choice and resume execution do not alter the requested experiment."""
    parent = None
    if 'resume_from_checkpoint_dir' in cfg:
        folder = Path(cfg['resume_from_checkpoint_dir'])
        parent = {name: {'path': str(folder / name), 'sha256': sha256(folder / name)}
                  for name in ('task_00.pt', 'fisher_state_0.pt')}
        metrics = Path(cfg['resume_from_metrics_json'])
        if metrics.is_dir():
            matches = sorted(metrics.rglob('*_cl_metrics.json'))
            if len(matches) != 1:
                raise ValueError(f'Expected one anchor metrics JSON below {metrics}; got {matches}')
            metrics = matches[0]
            cfg['resume_from_metrics_json'] = str(metrics)
        parent['metrics'] = {'path': str(metrics), 'sha256': sha256(metrics)}
    parent_validation = validate_parent(ns, cfg, parent)
    prepared = prepared_fisher_identity(cfg, allow_missing=allow_missing_prepared)
    source_manifest = json.loads((EXECUTION / 'source_frozen.json').read_text())
    for entry in source_manifest['files']:
        if sha256(ns.source / entry['path']) != entry['sha256']:
            raise ValueError(f'Frozen source changed: {entry["path"]}')
    payload = {'protocol': PROTOCOL, 'mode': ns.mode, 'anchor_method': ns.anchor_method,
               'future_optimizer': ns.future_optimizer, 'arm': ns.arm, 'smoke_only': ns.smoke,
               'task0_shaping': ns.anchor_method == 'gam', 'task1_shaping': ns.future_optimizer == 'gam',
               'parent_validation': parent_validation,
               'prepared_fisher': prepared, 'execution_ready': prepared['execution_ready'],
               'seed': ns.seed, 'requested_config': copy.deepcopy(cfg), 'parent': parent,
               'source_manifest_sha256': sha256(EXECUTION / 'source_frozen.json'),
               'runner_sha256': sha256(__file__)}
    payload['identity_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def validate_loader_passes(iterations, epochs):
    """The fit-Fisher estimator deliberately traverses the shuffled train loader once."""
    if len(iterations) != epochs + 1:
        raise ValueError(f'Expected {epochs} training passes plus one fit-Fisher pass; got {len(iterations)}.')
    if any(e.get('samples', 0) <= 0 or e.get('samples') != e.get('sample_count') for e in iterations):
        raise ValueError('Training or final fit-Fisher traversal is incomplete.')
    if [e.get('iteration') for e in iterations] != list(range(epochs + 1)):
        raise ValueError('Loader iterations are duplicated, missing, or out of order.')


def validate_final_epoch_log(logs, task, epochs):
    # The frozen trainer's file logger stores only the last epoch; tqdm's other
    # epochs go to the queue console. Full pass counts are verified separately.
    if not re.search(rf'Task {task}, Epoch {epochs}/{epochs}\b', logs):
        raise ValueError(f'Task {task} final {epochs}/{epochs} epoch log missing.')


def validate_completion(run_dir, cfg, mode):
    """A checkpoint alone never implies training or metric completion."""
    count = int(cfg['max_train_tasks'])
    if (run_dir / 'DIVERGED.json').exists():
        raise ValueError('A diverged run cannot be marked complete.')
    epochs = 1 if cfg.get('rq2_smoke_only') else 50
    if mode != 'continuation' or count != 2 or cfg.get('resume_from_task') != 0 or cfg['epochs'] != epochs:
        raise ValueError('Completion must certify only Task1 from Task0 at the explicit epoch budget.')
    tasks = range(1 if 'resume_from_checkpoint_dir' in cfg else 0, count)
    metrics = sorted((run_dir / 'trainer_outputs').rglob('*_cl_metrics.json'))
    if len(metrics) != 1:
        raise ValueError(f'Expected one completed metrics file: {metrics}')
    content = json.loads(metrics[0].read_text())
    matrix = content['cnn']['matrices'].get('final')
    if matrix is None or len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError(f'Final CNN matrix must be {count}x{count}.')
    if any(not math.isfinite(float(matrix[i][j])) for i in range(count) for j in range(i + 1)):
        raise ValueError('Final CNN matrix contains unfinished/nonfinite evaluated cells.')
    t01 = content['cnn']['matrices'].get('t01')
    if t01 is None or any(t01[i][j] != matrix[i][j] for i in range(2) for j in range(i + 1)):
        raise ValueError('Final CIL matrix does not equal the two-task t01 matrix.')
    step1 = content.get('mechanism', {}).get('steps', {}).get('1')
    if (step1 is None or float(step1['lambda_flat']) != 0.
            or float(step1['lambda_ewc']) != float(cfg['ewc_lambda'])):
        raise ValueError('Task1 mechanism metrics are missing or belong to a different objective.')
    for key in ('train_clean_loss_mean', 'train_fisher_loss_mean'):
        finite_number(step1[key], f'mechanism.{key}')
    if step1.get('ewc_normalize_accumulated_fisher') is not False:
        raise ValueError('Raw accumulated Fisher policy changed.')
    prepared = prepared_fisher_identity(cfg)
    events = read_jsonl(run_dir / 'runtime_receipts.jsonl')
    rng = read_jsonl(run_dir / 'rng_receipts.jsonl')
    logs = '\n'.join(p.read_text() for p in (run_dir / 'trainer_outputs').rglob('*.log'))
    seen_tasks = {int(e['task']) for e in rng if e.get('event') == 'task_rng_reset'}
    if seen_tasks != {1} or any(int(t) != 1 for t in re.findall(r'Task (\d+), Epoch \d+/', logs)):
        raise ValueError('Extension trained a task other than Task1.')
    if not any(e.get('event') == 'complete_resume' and e.get('task') == 0
               and e.get('verified_exact') is True for e in events):
        raise ValueError('Exact full Task0 restoration receipt missing.')
    raw_restores = [e for e in events if e.get('event') == 'prepared_raw_fisher_loaded']
    if (len(raw_restores) != 1 or raw_restores[0].get('prepared_fisher') != prepared
            or raw_restores[0].get('verified_exact') is not True):
        raise ValueError('Verified replacement of old Fisher by prepared raw Fisher is missing.')
    if not any(e.get('event') == 'checkpoint_full_reload_verified' and e.get('task') == 1 for e in events):
        raise ValueError('Saved Task1 checkpoint was not fully reloaded and verified.')
    if re.search(r'(?:Loss|CE|GAM)\s+(?:nan|[+-]?inf)(?:[,\s]|$)', logs, re.I):
        raise ValueError('Nonfinite loss in training log.')
    assets = {}
    for task in tasks:
        if not any(e.get('event') == 'task_complete' and e.get('task') == task for e in events):
            raise ValueError(f'Task {task} completion receipt missing.')
        validate_final_epoch_log(logs, task, epochs)
        iterations = [e for e in rng if e.get('event') == 'loader_iteration' and e.get('task') == task and e.get('shuffle')]
        validate_loader_passes(iterations, epochs)
        for name in (f'task_{task:02d}.pt', f'fisher_state_{task}.pt', f'rng_state_{task:02d}.pt'):
            path = run_dir / 'checkpoints' / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing completed task asset: {path}')
            assets[str(path.relative_to(run_dir))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    # Pairing checks compare the actual input stream, not merely matching seed labels.
    seed = int(cfg['seed'][0])
    reference_path = FORMAL / 'formal_training' / f'sequence_sgd_seed{seed}' / 'rng_receipts.jsonl'
    reference = read_jsonl(reference_path)
    head = [e['new_head_sha256'] for e in rng if e.get('event') == 'head_initialized' and e.get('task') == 1]
    ref_head = [e['new_head_sha256'] for e in reference if e.get('event') == 'head_initialized' and e.get('task') == 1]
    if len(head) != 1 or len(ref_head) != 1 or head != ref_head:
        raise ValueError('Task1 new head does not match the formal same-seed initialization.')
    reference_complete = json.loads((reference_path.parent / 'COMPLETE.json').read_text())
    if sha256(reference_path) != reference_complete['validation']['rng_receipts_sha256']:
        raise ValueError('Formal input-pairing reference changed after its completion.')
    # In a one-epoch smoke the Fisher traversal is the second iterator invocation:
    # compare its input stream to formal iteration 1, not formal Fisher iteration 50.
    ref_iters = [e for e in reference if e.get('event') == 'loader_iteration' and e.get('task') == 1 and e.get('shuffle')][:epochs + 1]
    compare = ('generator_seed', 'num_workers', 'dataset_ids_sha256', 'indices_sha256',
               'first_batch_pixels_targets_sha256', 'samples', 'sample_count', 'batches')
    if len(ref_iters) != epochs + 1 or any(any(a[k] != b[k] for k in compare) for a, b in zip(iterations, ref_iters)):
        raise ValueError('Task1 loader/input receipts do not match the formal same-seed protocol.')
    local_rows = read_jsonl(run_dir / 'task0_local_metrics.jsonl')
    old = [r for r in local_rows if r['task'] == 1]
    if len(old) != 1 or not math.isfinite(float(old[0]['effective_weight_displacement'])):
        raise ValueError('Task1 effective Q/V BA displacement is missing or invalid.')
    for split in ('train', 'test'):
        for key in ('local_ce', 'seen_ce', 'local_accuracy', 'seen_accuracy'):
            finite_number(old[0][split][key], f'task0.{split}.{key}')
    batches = read_jsonl(run_dir / 'training_batch_metrics.jsonl')
    expected_batches = sum(row['batches'] for row in iterations[:epochs])
    if len(batches) != expected_batches or [row['step'] for row in batches] != list(range(expected_batches)):
        raise ValueError('Finite per-batch loss/update-gradient receipts are incomplete.')
    for row in batches:
        if row.get('task') != 1 or row.get('parameters_finite_after_step') is not True:
            raise ValueError('Training batch lacks Task1 finite-parameter validation.')
        for key in ('loss', 'ce_clean', 'ewc', 'grad_combined_norm'):
            finite_number(row[key], f'batch {row["step"]}.{key}')
        if row.get('gradient_components_sampled') is not sample_gradient_components(row['step']):
            raise ValueError('Gradient component sampling schedule differs from step1 plus every eighth step.')
        if row['gradient_components_sampled']:
            for key in ('grad_clean_norm', 'grad_fisher_norm', 'fisher_to_clean_norm_ratio', 'cos_clean_fisher'):
                finite_number(row[key], f'batch {row["step"]}.{key}')
    sampled = [row for row in batches if row['gradient_components_sampled']]
    import torch
    cp = torch.load(run_dir / 'checkpoints/task_01.pt', map_location='cpu', weights_only=True, mmap=True)
    fisher = torch.load(run_dir / 'checkpoints/fisher_state_1.pt', map_location='cpu', weights_only=True, mmap=True)
    if cp['cur_task'] != 1 or cp['total_classes'] != 40 or cp['optimizer_type'] != cfg['optimizer_type']:
        raise ValueError('Saved checkpoint task/class/optimizer identity mismatch.')
    if any(not bool(torch.isfinite(value).all()) for value in cp['network_state'].values()):
        raise FloatingPointError('Saved model contains nonfinite state.')
    fpast = fisher['fisher_past_delta']
    if not fpast or any(not bool(torch.isfinite(x).all()) for x in fpast.values()):
        raise ValueError('Task1 Fisher state is empty or nonfinite.')
    return {'metrics': str(metrics[0]), 'metrics_sha256': sha256(metrics[0]),
            'epochs_per_new_task': epochs, 'validated_tasks': list(tasks), 'checkpoint_assets': assets,
            'smoke_only': bool(cfg.get('rq2_smoke_only')), 'mechanism_task1_verified': True,
            'accuracy_CIL': {'a11': matrix[0][0], 'a21': matrix[1][0], 'a22': matrix[1][1]},
            'effective_weight_displacement': old[0]['effective_weight_displacement'],
            'fisher_policy': FISHER_POLICY, 'prepared_fisher': prepared,
            'losses': {key: step1[key] for key in ('train_clean_loss_mean', 'train_fisher_loss_mean')},
            'gradients': {'grad_combined_norm_mean': sum(row['grad_combined_norm'] for row in batches) / len(batches),
                          'grad_combined_norm_max': max(row['grad_combined_norm'] for row in batches),
                          'source': 'actual_native_sgd_parameter_gradients_immediately_before_step',
                          'component_norms_available': True,
                          'component_source': 'native_EWC_autograd; clean=actual_combined-EWC; float64_norms',
                          'component_sampling': 'zero-based steps 0, 1, 8, 16, ...',
                          'component_sample_count': len(sampled),
                          **{key + '_mean': sum(row[key] for row in sampled) / len(sampled)
                             for key in ('grad_clean_norm', 'grad_fisher_norm', 'fisher_to_clean_norm_ratio', 'cos_clean_fisher')},
                          'batches': len(batches)},
            'training_batch_metrics_sha256': sha256(run_dir / 'training_batch_metrics.jsonl'),
            'pairing': {'passed': True, 'reference': str(reference_path), 'reference_sha256': sha256(reference_path),
                        'epochs': epochs, 'fit_fisher_passes': 1, 'new_head_sha256': head[0],
                        'scope': 'Every sample index and first augmented batch for all training and the final fit-Fisher loader traversals, not every augmented batch'},
            'runtime_receipts_sha256': sha256(run_dir / 'runtime_receipts.jsonl'),
            'rng_receipts_sha256': sha256(run_dir / 'rng_receipts.jsonl')}


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def install_runtime(ns, cfg, run_dir):
    import numpy as np
    import torch
    import timm

    # Timm's mutable default alias must never select/download another pretrained
    # state. Reuse the audited frozen base in the complete historical manifest.
    reference_dir, _ = anchor_assets('sgd')
    reference_path = reference_dir / 'task_00.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=True,
                           mmap=True)['network_state']
    frozen = {key: value for key, value in reference.items() if is_frozen_backbone(key)}
    if len(frozen) != 300:
        raise ValueError(f'Unexpected frozen-backbone state count: {len(frozen)}')
    original_create = timm.create_model

    def local_create_model(model_name, *a, **kw):
        if model_name != 'vit_base_patch16_224':
            raise ValueError(f'Unexpected model constructor {model_name!r}')
        kw['pretrained'] = False
        return original_create(model_name, *a, **kw)

    timm.create_model = local_create_model
    from utils import factory
    from utils import data_manager as dm_module
    from utils.inc_net import IncrementalNet
    from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
    from models_LoRAbasedCL.seqlora import Learner as SeqLearner
    from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner as NativeLearner
    original_factory = factory.get_model
    original_load = LoraBaseLearner.load_task_checkpoint
    original_save = LoraBaseLearner.save_task_checkpoint
    original_incremental = SeqLearner.incremental_train
    original_update_fc = IncrementalNet.update_fc
    original_build_loader = dm_module.DataManager.build_dataloader
    original_loader_class = dm_module.DataLoader
    original_batch = NativeLearner._step_batch_with_log
    original_task_loss = NativeLearner._task_loss
    original_fisher_penalty = NativeLearner._past_fisher_penalty
    context = {'task': -1, 'loader_serial': 0, 'learner': None, 'anchor_delta': None,
               'anchor_head': None, 'train_step': 0}

    def checked_task_loss(model, *args, **kwargs):
        result = original_task_loss(model, *args, **kwargs)
        finite_number(result[0].detach().item(), 'native task loss')
        return result

    def checked_fisher_penalty(model, *args, **kwargs):
        result = original_fisher_penalty(model, *args, **kwargs)
        finite_number(result.detach().item(), 'native EWC loss')
        return result

    def checked_batch(model, optimizer, inputs, targets, class_offset):
        if model._optimizer_type != 'sgd' or context['task'] != 1:
            raise ValueError('Only native Task1 SGD is authorized.')
        step = context['train_step']
        original_step = optimizer.step
        observed = []

        def checked_step(*args, **kwargs):
            record = observe_sgd_gradients(model, optimizer.param_groups,
                                           sample_components=sample_gradient_components(step))
            result = original_step(*args, **kwargs)
            for index, parameter in enumerate(p for group in optimizer.param_groups for p in group['params']):
                if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(f'Nonfinite parameter after SGD step {step}, parameter {index}')
            record['parameters_finite_after_step'] = True
            observed.append(record)
            return result

        optimizer.step = checked_step
        try:
            logits, loss, log_dict = original_batch(model, optimizer, inputs, targets, class_offset)
        finally:
            optimizer.step = original_step
        finite_number(loss, f'total loss at step {step}')
        for key in ('ce_clean', 'ewc'):
            finite_number(log_dict[key], f'{key} at step {step}')
        if len(observed) != 1:
            raise ValueError('Native SGD must perform exactly one update per batch.')
        append_json(run_dir / 'training_batch_metrics.jsonl', {
            'task': 1, 'step': step, 'loss': loss, 'ce_clean': log_dict['ce_clean'],
            'ewc': log_dict['ewc'], 'ewc_lambda': cfg['ewc_lambda'],
            'learning_rates': [float(group['lr']) for group in optimizer.param_groups], **observed[0],
        })
        context['train_step'] += 1
        return logits, loss, log_dict

    def reset_rng(seed):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def capture_rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    @contextlib.contextmanager
    def preserve_rng():
        state = capture_rng()
        try:
            yield
        finally:
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu'])
            if state['torch_cuda'] is not None:
                torch.cuda.set_rng_state_all(state['torch_cuda'])

    def patched_factory(name, args):
        model = original_factory(name, args)
        net = model._unwrap_network()
        current = net.state_dict()
        expected = {key for key in current if is_frozen_backbone(key)}
        if expected != set(frozen):
            raise ValueError('Frozen-backbone keys differ from audited historical architecture.')
        current.update(frozen)
        net.load_state_dict(current, strict=True)
        if not all(torch.equal(net.state_dict()[key].cpu(), val) for key, val in frozen.items()):
            raise ValueError('Frozen backbone restoration failed.')
        context['learner'] = model
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'frozen_backbone_loaded', 'reference': str(reference_path),
            'frozen_state_sha256': tensor_hash(frozen.items()), 'frozen_keys': len(frozen),
            'new_adapter_state_sha256': tensor_hash((k, v) for k, v in current.items() if not is_frozen_backbone(k)),
        })
        return model

    def remember_anchor(model):
        net = model._unwrap_network()
        context['anchor_delta'] = {k: value.detach().cpu().clone()
                                   for k, value in model._iter_delta_terms(net)}
        context['anchor_head'] = {k: value[:20].detach().cpu().clone()
                                  for k, value in net.fc.state_dict().items()}

    def patched_load(model, checkpoint_dir, task_idx, data_manager):
        if task_idx != 0 or Path(checkpoint_dir).resolve() != ns.anchor_dir.resolve():
            raise ValueError('This extension only restores its explicit formal Task0 parent.')
        # Preserve lifecycle/Fisher sidecar logic, then override *every* weight.
        original_load(model, checkpoint_dir, task_idx, data_manager)
        checkpoint = Path(checkpoint_dir) / f'task_{task_idx:02d}.pt'
        full = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        net = model._unwrap_network()
        net.load_state_dict(full['network_state'], strict=True)
        if not all(torch.equal(net.state_dict()[key].cpu(), value)
                   for key, value in full['network_state'].items()):
            raise ValueError('Complete manifest resume did not restore every parameter.')
        model._num_finished_tasks = int(task_idx) + 1
        model._data_manager = model.data_manager = data_manager
        if (model._optimizer_type != 'sgd' or float(model._lambda_flat) != 0.
                or float(model._ewc_lambda) != float(cfg['ewc_lambda'])
                or float(model._lambda_ewc) != float(cfg['ewc_lambda'])
                or model._normalize_fisher is not False or model._normalize_accumulated_fisher is not False):
            raise ValueError('Actual restored SGD/lambda/raw-Fisher configuration differs.')
        parent_fisher = torch.load(Path(checkpoint_dir) / 'fisher_state_0.pt',
                                   map_location='cpu', weights_only=True, mmap=True)['fisher_past_delta']
        if (set(parent_fisher) != set(model._fisher_past_delta) or not parent_fisher
                or not all(torch.equal(value.cpu(), model._fisher_past_delta[key].cpu())
                           for key, value in parent_fisher.items())):
            raise ValueError('Original Task0 Fisher must be restored unchanged, tensor for tensor.')
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'complete_resume', 'task': task_idx, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'fisher_sha256': sha256(Path(checkpoint_dir) / f'fisher_state_{task_idx}.pt'),
            'state_keys': len(full['network_state']), 'verified_exact': True,
            'fisher_verified_exact': True, 'fisher_tensors': len(parent_fisher),
            'actual_ewc_lambda': float(model._ewc_lambda), 'lambda_flat': 0.,
            'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
        })
        prepared_payload, prepared_identity = load_prepared_fisher(
            cfg, {key: value.shape for key, value in parent_fisher.items()})
        model._fisher_past_delta = {key: value.detach().cpu().clone()
                                   for key, value in prepared_payload['fisher_past_delta'].items()}
        model._fisher_past_trace_mass = 0.0
        model._fisher_new_task_raw_trace = float(prepared_payload['raw_trace'])
        model._last_task_new_fisher_stats = copy.deepcopy(prepared_payload['stats'])
        if not all(torch.equal(value, model._fisher_past_delta[key])
                   for key, value in prepared_payload['fisher_past_delta'].items()):
            raise ValueError('Prepared raw Fisher replacement is not tensor-for-tensor exact.')
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'prepared_raw_fisher_loaded', 'task': 0, 'verified_exact': True,
            'replaced_original_normalized_fisher': True, 'prepared_fisher': prepared_identity,
            'fisher_tensors': len(model._fisher_past_delta),
            'fisher_tensor_sha256': tensor_hash(model._fisher_past_delta.items()),
            'actual_raw_trace': sum(float(value.double().sum().item()) for value in model._fisher_past_delta.values()),
            'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
        })
        if task_idx == 0:
            remember_anchor(model)
            if not ns.validate_cpu:
                evaluate_old_task(model, task_idx)
        else:
            anchor_path = ((ns.anchor_dir.resolve() / 'task_00.pt') if ns.anchor_dir is not None
                           else run_dir / 'checkpoints/task_00.pt')
            anchor_state = torch.load(anchor_path, map_location='cpu', weights_only=True,
                                     mmap=True)['network_state']
            # Shapes beyond the old head differ; only old-head prefix is needed.
            delta = {}
            actual_keys = [key for key, _ in model._iter_delta_terms(net)]
            for block in range(12):
                for role in ('q', 'v'):
                    prefix = f'backbone.lora_vit.blocks.{block}.attn.qkv.'
                    # Use actual learner keys instead of assuming their spelling.
                    actual_key = actual_keys[2 * block + (role == 'v')]
                    delta[actual_key] = (anchor_state[prefix + f'linear_b_{role}.weight'] @
                                         anchor_state[prefix + f'linear_a_{role}.weight'])
            context['anchor_delta'] = delta
            context['anchor_head'] = {k: anchor_state['fc.' + k][:20].clone() for k in net.fc.state_dict()}

    def patched_update_fc(network, nb_classes):
        old_classes = 0 if network.fc is None else network.fc.out_features
        result = original_update_fc(network, nb_classes)
        if context['task'] >= 0:
            append_json(run_dir / 'rng_receipts.jsonl', {
                'event': 'head_initialized', 'task': context['task'],
                'old_classes': old_classes, 'total_classes': nb_classes,
                'task_seed': task_seed(ns.seed, context['task']),
                'new_head_sha256': tensor_hash((k, v[old_classes:]) for k, v in network.fc.state_dict().items()),
            })
        return result

    class AuditedLoader(original_loader_class):
        def __iter__(self):
            receipt = getattr(self, '_rq2_receipt', None)
            if receipt is None or not receipt['shuffle']:
                yield from super().__iter__()
                return
            iteration = getattr(self, '_rq2_iterations', 0)
            self._rq2_iterations = iteration + 1
            sample_digest = hashlib.sha256()
            first_pixels = None
            batches = samples = 0
            try:
                for batch in super().__iter__():
                    idx, inputs, targets = batch
                    sample_digest.update(idx.detach().cpu().numpy().tobytes())
                    if first_pixels is None:
                        first_pixels = tensor_hash([('input', inputs), ('target', targets)])
                    batches += 1
                    samples += len(targets)
                    yield batch
            finally:
                append_json(run_dir / 'rng_receipts.jsonl', {
                    **receipt, 'event': 'loader_iteration', 'iteration': iteration,
                    'batches': batches, 'samples': samples,
                    'indices_sha256': sample_digest.hexdigest(),
                    'first_batch_pixels_targets_sha256': first_pixels,
                })

    def patched_build_loader(manager, dataset, *, batch_size, shuffle, num_workers, **kwargs):
        serial = context['loader_serial']
        context['loader_serial'] += 1
        seed = task_seed(ns.seed, max(0, context['task'])) + 10000019 + 7919 * serial
        kwargs['generator'] = torch.Generator().manual_seed(seed)
        kwargs['persistent_workers'] = False
        loader = original_build_loader(manager, dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, **kwargs)
        ids = list(zip(map(str, getattr(dataset, 'images', [])),
                       map(int, getattr(dataset, 'labels', []))))
        loader._rq2_receipt = {'task': context['task'], 'loader_serial': serial,
                               'shuffle': bool(shuffle), 'generator_seed': seed,
                               'num_workers': num_workers, 'sample_count': len(dataset),
                               'dataset_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        append_json(run_dir / 'rng_receipts.jsonl', {**loader._rq2_receipt, 'event': 'loader_created'})
        return loader

    def patched_incremental(model, data_manager):
        task = model._cur_task + 1
        if task != 1:
            raise ValueError(f'Extension attempted to train Task{task}; only Task1 is authorized.')
        context.update(task=task, loader_serial=0)
        reset_rng(task_seed(ns.seed, task))
        append_json(run_dir / 'rng_receipts.jsonl', {
            'event': 'task_rng_reset', 'task': task, 'task_seed': task_seed(ns.seed, task),
            'rng_protocol': PROTOCOL, 'optimizer_type': cfg['optimizer_type'],
        })
        return original_incremental(model, data_manager)

    def evaluate_old_task(model, task):
        if context['anchor_delta'] is None:
            return
        net = model._unwrap_network()
        for name, value in context['anchor_head'].items():
            if not torch.equal(net.fc.state_dict()[name][:20].cpu(), value):
                raise ValueError(f'Old Task0 head changed at task {task}.')
        device = model._device
        record = {'task': task, 'coordinate': 'effective_QV_BA', 'old_local_head': [0, 20],
                  'train_transform': 'deterministic_test', 'trace_measured': False}
        with torch.no_grad():
            record['effective_weight_displacement'] = float(sum(
                (value.double() - context['anchor_delta'][key].to(device).double()).square().sum()
                for key, value in model._iter_delta_terms(net)).sqrt().item())
        was_training = net.training
        with preserve_rng(), torch.no_grad():
            net.eval()
            for split in ('train', 'test'):
                dataset = model._data_manager.get_dataset(np.arange(20), source=split, mode='test')
                loader = original_loader_class(dataset, batch_size=64, shuffle=False,
                                               num_workers=ns.workers)
                total = local_correct = seen_correct = 0
                local_loss = seen_loss = 0.
                for _, images, targets in loader:
                    targets = targets.to(device)
                    logits = net(images.to(device))['logits']
                    local_logits = logits[:, :20]
                    local_loss += torch.nn.functional.cross_entropy(local_logits, targets, reduction='none').double().sum().item()
                    seen_loss += torch.nn.functional.cross_entropy(logits, targets, reduction='none').double().sum().item()
                    local_correct += int((local_logits.argmax(1) == targets).sum().item())
                    seen_correct += int((logits.argmax(1) == targets).sum().item())
                    total += len(targets)
                record[split] = {'n': total, 'local_ce': local_loss / total,
                                 'seen_ce': seen_loss / total, 'local_accuracy': 100 * local_correct / total,
                                 'seen_accuracy': 100 * seen_correct / total}
            net.train(was_training)
        append_json(run_dir / 'task0_local_metrics.jsonl', record)

    def patched_save(model, ckpt_dir, task_idx):
        if task_idx != 1:
            raise ValueError('This extension may only save newly trained Task1.')
        for name, parameter in model._unwrap_network().named_parameters():
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f'Non-finite trained parameter at task {task_idx}: {name}')
        original_save(model, ckpt_dir, task_idx)
        checkpoint = Path(ckpt_dir) / 'task_01.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        net = model._unwrap_network()
        net.load_state_dict(saved['network_state'], strict=True)
        if not all(torch.equal(net.state_dict()[key].detach().cpu(), value)
                   for key, value in saved['network_state'].items()):
            raise ValueError('Saved Task1 full-state reload was not exact.')
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'checkpoint_full_reload_verified', 'task': 1,
            'state_keys': len(saved['network_state']), 'verified_exact': True,
        })
        torch.save({'rng_protocol': PROTOCOL, 'seed': ns.seed, 'task': task_idx,
                    'state': capture_rng()}, Path(ckpt_dir) / f'rng_state_{task_idx:02d}.pt')
        if task_idx == 0:
            remember_anchor(model)
        evaluate_old_task(model, task_idx)
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'task_complete', 'task': task_idx, 'time_unix': time.time(),
            'checkpoint': str(Path(ckpt_dir) / f'task_{task_idx:02d}.pt'),
        })

    factory.get_model = patched_factory
    LoraBaseLearner.load_task_checkpoint = patched_load
    LoraBaseLearner.save_task_checkpoint = patched_save
    SeqLearner.incremental_train = patched_incremental
    IncrementalNet.update_fc = patched_update_fc
    dm_module.DataLoader = AuditedLoader
    dm_module.DataManager.build_dataloader = patched_build_loader
    NativeLearner._task_loss = checked_task_loss
    NativeLearner._past_fisher_penalty = checked_fisher_penalty
    NativeLearner._step_batch_with_log = checked_batch
    return {'factory': factory, 'DataManager': dm_module.DataManager,
            'reset_rng': reset_rng, 'context': context, 'reference_path': reference_path}


def main():
    ns = parser().parse_args()
    if ns.seed != 0:
        raise ValueError('This sweep only defines seed0.')
    if ns.workers != 4:
        raise ValueError('Workers must equal the formal paired protocol value 4.')
    ns.source = ns.source.resolve()
    sys.path.insert(0, str(ns.source))
    cfg, run_dir = build_config(ns)
    identity = experiment_identity(ns, cfg, allow_missing_prepared=ns.print_config)
    if ns.print_config:
        print(json.dumps({'run_dir': str(run_dir), 'config': cfg, 'identity': identity}, indent=2))
        return
    requested_cfg = copy.deepcopy(cfg)
    completion_name = 'SMOKE_COMPLETE.json' if ns.smoke else 'COMPLETE.json'
    completion_status = 'smoke_complete' if ns.smoke else 'complete'
    if not ns.validate_cpu and (run_dir / 'run_manifest.json').exists():
        previous = json.loads((run_dir / 'run_manifest.json').read_text())
        if previous.get('identity_sha256') != identity['identity_sha256']:
            raise ValueError('Existing run identity differs: change --run-name; never reuse incompatible outputs.')
    if not ns.validate_cpu and (run_dir / 'DIVERGED.json').exists():
        raise ValueError('Divergence is a terminal observation; preserve this run and use a new run name.')
    if not ns.validate_cpu and (run_dir / completion_name).exists():
        previous = json.loads((run_dir / completion_name).read_text())
        if previous.get('identity_sha256') != identity['identity_sha256']:
            raise ValueError('COMPLETE receipt belongs to another experiment.')
        validation = validate_completion(run_dir, cfg, ns.mode)
        if validation != previous.get('validation'):
            raise ValueError('Completed experiment assets changed since completion.')
        print(f'Already complete and validated: {run_dir}', flush=True)
        return
    if ns.validate_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpu_uuid = None
        run_dir = HERE / 'cpu_validation' / run_dir.name
    else:
        if ns.gpu is None:
            raise ValueError('Training requires an authorized physical GPU via --gpu 0..6.')
        gpu_uuid = subprocess.check_output([
            'nvidia-smi', '-i', ns.gpu, '--query-gpu=uuid', '--format=csv,noheader'
        ], text=True).strip()
        if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
            raise RuntimeError(f'Cannot resolve requested physical GPU {ns.gpu}: {gpu_uuid!r}')
        current_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if current_visible.startswith('GPU-') and current_visible != gpu_uuid:
            raise RuntimeError(f'Queue GPU UUID {current_visible} differs from physical GPU {ns.gpu}: {gpu_uuid}')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.setdefault('TMPDIR', '/tmp')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ['PYTHONHASHSEED'] = str(ns.seed)
    os.environ['FLATNESS_CV_DATA_ROOT'] = cfg['data_root']
    if not ns.validate_cpu and (run_dir / 'run_manifest.json').exists() and not ns.resume_last:
        raise FileExistsError(f'Run already exists: {run_dir}; use --resume-last for task-boundary resume.')
    run_dir.mkdir(parents=True, exist_ok=True)
    if ns.resume_last and not ns.validate_cpu:
        saved = sorted((run_dir / 'checkpoints').glob('task_??.pt'))
        if saved:
            if len(saved) != 1 or saved[0].name != 'task_01.pt':
                raise ValueError('Unexpected checkpoint; this extension only produces task_01.pt.')
            latest = int(saved[-1].stem.split('_')[1])
            metrics = sorted((run_dir / 'trainer_outputs').rglob('*_cl_metrics.json'))
            if len(metrics) != 1:
                raise ValueError(f'Expected one metrics JSON for resume, found {metrics}')
            if latest + 1 >= cfg['max_train_tasks']:
                validation = validate_completion(run_dir, cfg, ns.mode)
                write_json(run_dir / completion_name, {'status': completion_status, 'completed_tasks': latest + 1,
                    'identity_sha256': identity['identity_sha256'], 'validation': validation,
                    'arm': ns.arm, 'smoke_only': ns.smoke,
                    'recovered_after_final_task': True, 'time_unix': time.time()})
                return
        elif (run_dir / 'run_manifest.json').exists():
            raise ValueError('Partial Task1 has no complete boundary checkpoint; use a fresh run directory.')
    import torch
    import timm
    import torchvision
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {'protocol': PROTOCOL, 'mode': ns.mode, 'anchor_method': ns.anchor_method,
                'future_optimizer': ns.future_optimizer, 'arm': ns.arm, 'smoke_only': ns.smoke,
                'task0_shaping': ns.anchor_method == 'gam', 'task1_shaping': ns.future_optimizer == 'gam',
                'physical_gpu': ns.gpu if not ns.validate_cpu else None, 'seed': ns.seed,
                'gpu_uuid': gpu_uuid,
                'source': str(ns.source), 'source_manifest_sha256': sha256(EXECUTION / 'source_frozen.json'),
                'runner_sha256': sha256(__file__), 'torch': torch.__version__,
                'torchvision': torchvision.__version__, 'timm': timm.__version__,
                'production_precision': 'FP32; CUDA matmul and cuDNN TF32 explicitly disabled',
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'config': cfg, 'start_unix': time.time(),
                'scope': 'CPU restoration only' if ns.validate_cpu else ('isolated one-epoch smoke' if ns.smoke else 'training'),
                'history_compatibility': 'Same frozen learner, FP32 settings and shared-task RNG as completed formal RQ2.'}
    manifest.update(identity_sha256=identity['identity_sha256'], requested_config=requested_cfg,
                    parent=identity['parent'], run_name=ns.run_name, ewc_lambda=ns.ewc_lambda,
                    parent_validation=identity['parent_validation'],
                    prepared_fisher=identity['prepared_fisher'], fisher_policy=FISHER_POLICY,
                    initialization='Complete formal Task0 weights; original normalized Fisher replaced entirely by verified native recomputed raw Fisher; new Task1 head')
    if ns.resume_last and (run_dir / 'run_manifest.json').exists():
        previous_path = run_dir / f'run_manifest_before_resume_{time.time_ns()}.json'
        previous_path.write_text((run_dir / 'run_manifest.json').read_text())
    write_json(run_dir / ('cpu_manifest.json' if ns.validate_cpu else 'run_manifest.json'), manifest)
    runtime = install_runtime(ns, cfg, run_dir)
    if ns.validate_cpu:
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.update(seed=ns.seed, device=[torch.device('cpu')], filepath=str(run_dir / 'checkpoints'))
        Path(cpu_cfg['filepath']).mkdir(parents=True, exist_ok=True)
        runtime['reset_rng'](ns.seed)
        manager = runtime['DataManager']('imageneta', False, ns.seed, 20, 20, cpu_cfg)
        cpu_cfg['task_increments'] = list(manager._increments)
        cpu_cfg['nb_classes'] = manager.nb_classes
        model = runtime['factory'].get_model(cpu_cfg['model_name'], cpu_cfg)
        if 'resume_from_checkpoint_dir' in cpu_cfg:
            model.load_task_checkpoint(cpu_cfg['resume_from_checkpoint_dir'], 0, manager)
        next_task = 1 if 'resume_from_checkpoint_dir' in cpu_cfg else 0
        runtime['context']['task'] = next_task
        runtime['reset_rng'](task_seed(ns.seed, next_task))
        model._unwrap_network().update_fc(20 * (next_task + 1))
        assert model._optimizer_type == cfg['optimizer_type']
        assert float(model._ewc_lambda) == float(cfg['ewc_lambda'])
        result = {'status': 'passed', 'scope': 'CPU model construction/full-state/Fisher restoration/new-head initialization',
                  'forward_tested': False, 'training_started': False,
                  'new_head_sha256': tensor_hash((k, v[next_task * 20:])
                                                for k, v in model._unwrap_network().fc.state_dict().items()),
                  'fisher_tensors': len(model._fisher_past_delta), 'run_dir': str(run_dir),
                  'actual_optimizer_type': model._optimizer_type, 'actual_ewc_lambda': float(model._ewc_lambda),
                  'actual_lambda_flat': float(model._lambda_flat),
                  'ewc_normalize_fisher': model._normalize_fisher,
                  'ewc_normalize_accumulated_fisher': model._normalize_accumulated_fisher,
                  'prepared_fisher': identity['prepared_fisher'],
                  'actual_raw_trace': sum(float(value.double().sum().item()) for value in model._fisher_past_delta.values()),
                  'runner_sha256': sha256(__file__)}
        write_json(run_dir / 'CPU_VALIDATED.json', result)
        print(json.dumps(result), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one authorized GPU must be visible.')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from src.trainer import train
    try:
        train(cfg)
        validation = validate_completion(run_dir, requested_cfg, ns.mode)
    except FloatingPointError as exc:
        write_json(run_dir / 'DIVERGED.json', {
            'status': 'diverged', 'identity_sha256': identity['identity_sha256'],
            'seed': ns.seed, 'ewc_lambda': ns.ewc_lambda, 'smoke_only': ns.smoke,
            'arm': ns.arm, 'reason': str(exc), 'error': repr(exc), 'time_unix': time.time(),
            'last_completed_training_steps': runtime['context']['train_step'],
            'fisher_policy': FISHER_POLICY, 'prepared_fisher': identity['prepared_fisher'],
            'hyperparameters_changed': False,
        })
        print(f'Numerical divergence at lambda={ns.ewc_lambda}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(86) from exc
    except BaseException as exc:
        write_json(run_dir / 'FAILED.json', {'error': repr(exc), 'time_unix': time.time()})
        raise
    write_json(run_dir / completion_name, {'status': completion_status, 'completed_tasks': cfg['max_train_tasks'],
              'arm': ns.arm, 'smoke_only': ns.smoke,
              'time_unix': time.time(), 'manifest': str(run_dir / 'run_manifest.json'),
              'identity_sha256': identity['identity_sha256'], 'validation': validation})


if __name__ == '__main__':
    main()
