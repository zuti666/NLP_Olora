"""Publish the two controlled GAM-radius interventions; all operations are CPU-only.

--validate-only is read-only and accepts missing future results, while validating
the complete old controls, configuration, font, and every new completed job.
--execute refuses incomplete data and publishes only inside this new run folder.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
EXECUTION = BASE.parent
RADII = [i / 2 for i in range(11)]
ARMS = {'probe005_norm1': (.05, 1.), 'probe1_norm1': (1., 1.)}
GROUPS = ('ewc', 'old_gam', *ARMS)
JOBS = {0: 'task0_positive', 1: 'task1_negative'}
OLD = {
    0: ('fixed_gradient_extension_20260921', 'fixed_gradient', 'fixed_gradient_complete.json'),
    1: ('task1_gradient_extension_20260921', 'task1_gradient', 'task1_gradient_complete.json'),
}
PLOT_METRICS = ('D_train', 'train_accuracy_percent', 'raw_fisher_trace')
METRICS = (*PLOT_METRICS, 'absolute_train_ce')
LABELS = {'ewc': 'EWCLoRA', 'old_gam': 'FS-EWCLoRA (0.05, 0.2)',
          'probe005_norm1': 'FS-EWCLoRA (0.05, 1)', 'probe1_norm1': 'FS-EWCLoRA (1, 1)'}
COLORS = {'ewc': '#0072B2', 'old_gam': '#D55E00',
          'probe005_norm1': '#009E73', 'probe1_norm1': '#CC79A7'}
MARKERS = {'ewc': 'o', 'old_gam': 'D', 'probe005_norm1': 's', 'probe1_norm1': '^'}
LINES = {'ewc': '-', 'old_gam': (0, (4, 2)), 'probe005_norm1': '-.', 'probe1_norm1': ':'}
SAME_DIAGNOSTIC = ('coordinate_order', 'coordinate_numel', 'coordinate_sha256',
                   'data_manifest_sha256', 'counts', 'production_precision', 'fisher_definition')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)
    elif isinstance(value, (float, int)):
        require(math.isfinite(value), 'Nonfinite result value')


def close(actual, expected, label, *, atol=1e-12, rtol=1e-10):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
            f'{label}: {actual} differs from {expected}')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    require(rows, f'Cannot write empty comparison: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class Sources:
    def __init__(self):
        self.values = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f'Source changed: {path}')
        if str(path) in self.values:
            require(self.values[str(path)] == digest, f'Source changed while publishing: {path}')
        self.values[str(path)] = digest
        return digest

    def records(self):
        return [dict(path=path, sha256=digest) for path, digest in sorted(self.values.items())]

    def recheck(self):
        for path, expected in self.values.items():
            require(sha(path) == expected, f'Source changed during publishing: {path}')


def load_module(path, name):
    # This run also contains diagnostic.py. Never allow it to shadow the older
    # canonical execution/diagnostic.py imported by the measurement helpers.
    sys.path[:] = [str(EXECUTION), str(BASE)] + [p for p in sys.path
        if p and Path(p).resolve() not in (ROOT, EXECUTION, BASE)]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def result_paths(arm, seed, task):
    return ROOT / f'results/{arm}/seed{seed}/gam/{JOBS[task]}'


def result_values(row, origin, task, group, seed):
    output = dict(gradient_task=task, direction_sign=1 if task == 0 else -1,
                  group=group, label=LABELS[group], seed=seed, radius=float(row['point']['radius']),
                  directions=1, train_samples=1259, test_samples=307, fisher_samples=256)
    for split, count in (('train', 1259), ('test', 307)):
        metric = row['metrics'][split]
        require(metric['samples'] == count and 0 <= metric['accuracy'] <= 100 and metric['loss'] >= 0,
                f'Invalid {split} metric/sample count')
        damage = float(metric['loss']) - float(origin['metrics'][split]['loss'])
        output.update({f'D_{split}': damage, f'G_positive_{split}': max(damage, 0.),
                       f'{split}_ce': float(metric['loss']),
                       f'{split}_accuracy_percent': float(metric['accuracy'])})
    require(set(row['fisher_by_scope']) == {'subset'}, 'Changed Fisher scope')
    fisher = row['fisher_by_scope']['subset']
    require(fisher['samples'] == 256 and fisher['normalized'] is False and
            fisher['floor'] is None and fisher['regularizer'] is None and
            fisher['gradient_batch_size'] == 8 and fisher['raw_trace'] >= 0 and fisher['q'] == {},
            'Fisher must be unscaled per-example raw trace on the fixed 256 samples')
    output['raw_fisher_trace'] = float(fisher['raw_trace'])
    output['absolute_train_ce'] = output['train_ce']
    output['anchor_train_ce'] = float(origin['metrics']['train']['loss'])
    output['anchor_train_accuracy_percent'] = float(origin['metrics']['train']['accuracy'])
    output['anchor_raw_fisher_trace'] = float(origin['fisher_by_scope']['subset']['raw_trace'])
    finite(output)
    return output


def validate_summary_point(summary_point, record):
    require(summary_point['directions'] == 1, 'A fixed direction is not multiple replicates')
    for split in ('train', 'test'):
        value = summary_point[split]
        damage = record[f'D_{split}']; positive = record[f'G_positive_{split}']
        for field, expected in [('D_mean', damage), ('G', positive), ('Q95', positive),
                                ('sampled_max', positive), ('absolute_loss_mean', record[f'{split}_ce']),
                                ('accuracy_mean', record[f'{split}_accuracy_percent'])]:
            close(value[field], expected, f'Summary {split}/{field}')
        require(len(value['D_each']) == 1, 'Changed per-direction cardinality')
        close(value['D_each'][0], damage, 'Summary D_each')
    value = summary_point['raw_trace_by_scope']['subset']
    require(value['samples'] == 256 and len(value['each']) == 1, 'Changed Fisher sample/cardinality')
    for field in ('mean', 'Q95', 'sampled_max'):
        close(value[field], record['raw_fisher_trace'], f'Summary Fisher/{field}')
    close(value['each'][0], record['raw_fisher_trace'], 'Summary Fisher/each')


def completed_hashes(directory, sources):
    receipt_path = directory / 'extension_receipt.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'complete', f'Job is not complete: {directory}')
    sources.add(receipt_path)
    for key, name in (('states_sha256', 'states.jsonl'), ('summary_sha256', 'summary.json'),
                      ('manifest_sha256', 'manifest.json'), ('gradient_manifest_sha256', 'gradient_manifest.json'),
                      ('witness_sha256', 'finite_difference_witness.json')):
        sources.add(directory / name, receipt[key])
    return receipt


def load_old_controls(sources):
    """Read only the requested 11 measured radii, verifying raw values behind CSV."""
    output, manifests, checks = [], {}, []
    for task, (folder, job, report_name) in OLD.items():
        old_root = BASE / folder
        report_path = old_root / 'report' / report_name
        report = read(report_path)
        require(report['status'] == 'complete' and report['expected_jobs'] == report['validated_jobs'] == 6,
                f'Old control report incomplete: {report_path}')
        sources.add(report_path)
        trusted_sources = {x['path']: x['sha256'] for x in report['source_files']}
        csv_path = old_root / 'figures/all_per_seed.csv'
        saved = {x['path']: x['sha256'] for x in report['output_files']}
        sources.add(csv_path, saved[str(csv_path)])
        table = list(csv.DictReader(csv_path.open()))
        require(len(table) == 558, 'Old complete CSV must contain 6 x 93 rows')
        indexed = {(int(x['seed']), x['method'], float(x['radius'])): x for x in table}
        require(len(indexed) == 558, 'Duplicate old-control CSV rows')
        for seed in range(3):
            for method, group in (('sgd', 'ewc'), ('gam', 'old_gam')):
                directory = old_root / f'results/seed{seed}/{method}/{job}'
                receipt = completed_hashes(directory, sources)
                names = ['extension_receipt.json', 'manifest.json', 'summary.json', 'states.jsonl',
                         'gradient_manifest.json', 'gradient_sample_manifest.json', 'finite_difference_witness.json']
                if task == 1:
                    names += ['task1_head_manifest.json', 'task1_initial_head.pt']
                for name in names:
                    path = directory / name
                    require(str(path) in trusted_sources, f'Old published source receipt missing: {path}')
                    sources.add(path, trusted_sources[str(path)])
                manifest, summary = read(directory / 'manifest.json'), read(directory / 'summary.json')
                gradient, witness = read(directory / 'gradient_manifest.json'), read(directory / 'finite_difference_witness.json')
                sign = 1 if task == 0 else -1
                require(receipt['gradient_task'] == gradient['gradient_task'] == task and
                        receipt['direction_sign'] == gradient['direction_sign'] == sign,
                        'Old control objective/sign changed')
                require(receipt['result_check']['passed'] and receipt['witness_check']['passed'], 'Old control validation failed')
                close(gradient['direction_norm'], 1., 'Old direction norm', atol=5e-7, rtol=0)
                require(gradient['gradient_dot_direction'] * sign > 0 and
                        (witness['steps'][0]['loss_plus'] - witness['origin_loss']) * sign > 0,
                        'Old finite-difference orientation changed')
                raw = lines(directory / 'states.jsonl')
                by_radius = {float(x['point']['radius']): x for x in raw}
                require(len(raw) == len(by_radius) == summary['states_completed'] == 93, 'Old states incomplete')
                points = {float(x['radius']): x for x in summary['radius_summary']}
                for radius in RADII:
                    row = by_radius[radius]
                    require(row['anchor_restored_exactly'], 'Old anchor not restored')
                    if radius:
                        require(row['point']['direction']['sha256'] == gradient['direction_sha256'],
                                'Old direction changed across radii')
                    values = result_values(row, by_radius[0.], task, group, seed)
                    for key in ('D_train', 'G_positive_train', 'train_ce', 'train_accuracy_percent',
                                'D_test', 'G_positive_test', 'test_ce', 'test_accuracy_percent', 'raw_fisher_trace'):
                        close(values[key], float(indexed[(seed, method, radius)][key]), f'Old raw versus CSV/{key}')
                    validate_summary_point(points[radius], values)
                    output.append(values)
                manifests[(task, seed, method)] = manifest
                checks.append(dict(kind='old_control', seed=seed, method=method, gradient_task=task,
                                   selected_measured_radii=RADII, raw_csv_summary_verified=True))
    return output, manifests, checks


def validate_config(arm, seed, sources):
    path = ROOT / f'configs/{arm}_seed{seed}.json'
    config = read(path); sources.add(path)
    rho, norm = ARMS[arm]
    require(config['intervention_arm'] == arm and config['training_seed'] == seed and
            config['training_probe_radius'] == rho and config['training_norm_radius'] == norm and
            config['radii'] == RADII, 'Intervention config identity/radii changed')
    require(Path(config['output_dir']).resolve() == (ROOT / f'results/{arm}/seed{seed}').resolve(), 'Config output path mismatch')
    training_dir = ROOT / f'training/{arm}_seed{seed}'
    require(Path(config['training_receipt']).resolve() == (training_dir / 'COMPLETE.json').resolve(), 'Training receipt path mismatch')
    require(config['checkpoints'] == {'gam': {'0': str(training_dir / 'checkpoints/task_00.pt')}},
            'New direction must use only the new trained Task0 checkpoint')
    old = read(BASE / f'configs/seed{seed}.json')
    for key in ('data_root', 'source_path', 'workers', 'fisher_samples', 'fisher_batch_size',
                'eval_batch_size', 'subset_seed', 'direction_seed'):
        require(config[key] == old[key], f'Diagnostic protocol changed: {key}')
    sources.add(BASE / f'configs/seed{seed}.json')
    return config


def validate_training(arm, seed, sources):
    directory = ROOT / f'training/{arm}_seed{seed}'
    old_directory = BASE / f'formal_training/anchor_gam_seed{seed}'
    receipt, manifest = read(directory / 'COMPLETE.json'), read(directory / 'run_manifest.json')
    old = read(old_directory / 'run_manifest.json')
    rho, norm = ARMS[arm]
    require(receipt['status'] == 'complete' and receipt['completed_tasks'] == 1 and
            receipt['identity_sha256'] == manifest['identity_sha256'] and manifest['mode'] == 'anchor' and
            manifest['parent'] is None and manifest['seed'] == seed, 'Training is not completed fresh Task0')
    cfg = manifest['requested_config']; old_cfg = old['requested_config']
    for key, expected in [('gam_grad_rho', rho), ('as1_rho', rho),
                          ('gam_grad_norm_rho', norm), ('as1_norm_rho', norm)]:
        close(cfg[key], expected, f'Training {key}')
        close(manifest['config'][key], expected, f'Effective training {key}')
    allowed = {'gam_grad_rho', 'as1_rho', 'gam_grad_norm_rho', 'as1_norm_rho', 'rq2_radius_intervention',
               'prefix', 'outputs_root', 'save_task_checkpoint_dir', 'optimizer_tag_override'}
    changes = {k for k in set(cfg) | set(old_cfg) if cfg.get(k) != old_cfg.get(k)}
    require(changes <= allowed, f'This is not a radius-only intervention: {sorted(changes - allowed)}')
    require(cfg['rq2_radius_intervention']['arm'] == arm and cfg['max_train_tasks'] == 1,
            'Training arm/task budget mismatch')
    require(manifest['matmul_allow_tf32'] is False and manifest['cudnn_allow_tf32'] is False,
            'Training precision changed')
    validation = receipt['validation']
    require(validation['epochs_per_new_task'] == 50 and validation['validated_tasks'] == [0], 'Training budget incomplete')
    sources.add(directory / 'COMPLETE.json'); sources.add(directory / 'run_manifest.json')
    sources.add(old_directory / 'run_manifest.json')
    for key, name in [('rng_receipts_sha256', 'rng_receipts.jsonl'), ('runtime_receipts_sha256', 'runtime_receipts.jsonl')]:
        sources.add(directory / name, validation[key])
    checkpoint = directory / 'checkpoints/task_00.pt'
    sources.add(checkpoint, validation['checkpoint_assets']['checkpoints/task_00.pt']['sha256'])
    runtime, rng = lines(directory / 'runtime_receipts.jsonl'), lines(directory / 'rng_receipts.jsonl')
    old_rng, old_runtime = lines(old_directory / 'rng_receipts.jsonl'), lines(old_directory / 'runtime_receipts.jsonl')
    for name in ('rng_receipts.jsonl', 'runtime_receipts.jsonl'):
        sources.add(old_directory / name)
    initial = next(x for x in runtime if x['event'] == 'frozen_backbone_loaded')
    old_initial = next(x for x in old_runtime if x['event'] == 'frozen_backbone_loaded')
    for key in ('frozen_state_sha256', 'new_adapter_state_sha256', 'frozen_keys'):
        require(initial[key] == old_initial[key], f'Fresh initialization pairing changed: {key}')
    new_head = next(x for x in rng if x['event'] == 'head_initialized' and x['task'] == 0)
    old_head = next(x for x in old_rng if x['event'] == 'head_initialized' and x['task'] == 0)
    require(new_head == old_head, 'Task0 initial classifier is not paired with the old seed')
    new_epochs = [x for x in rng if x['event'] == 'loader_iteration' and x.get('shuffle') and x['task'] == 0][:50]
    old_epochs = [x for x in old_rng if x['event'] == 'loader_iteration' and x.get('shuffle') and x['task'] == 0][:50]
    require(len(new_epochs) == len(old_epochs) == 50 and new_epochs == old_epochs,
            'Training sample order/first augmented batches or budget are not paired')
    observed = [x for x in runtime if x['event'] == 'actual_optimizer_radii_verified']
    require(observed and all(not x['cpu_construction_only'] for x in observed), 'Actual GPU optimizer radius receipt missing')
    for event in observed:
        close(event['probe_radius'], rho, 'Actual optimizer probe radius', atol=1e-7)
        close(event['norm_radius'], norm, 'Actual optimizer norm radius', atol=1e-7)
    return dict(passed=True, intervention_arm=arm, seed=seed, training_probe_radius=rho,
                training_norm_radius=norm, same_initial_lora_head_backbone=True,
                same_50_training_orders_and_first_augmented_batches=True,
                configuration_changes=sorted(changes), checkpoint_sha256=sha(checkpoint),
                runtime_seconds=receipt['time_unix'] - manifest['start_unix'])


def tensor_hash(values):
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode())
        digest.update(memoryview(value.numpy()).cast('B'))
    return digest.hexdigest()


def validate_vector_witness(directory, manifest, task, seed, sources):
    """Independent CPU reconstruction rejects an incorrect sign even if metadata agrees."""
    import torch
    torch.set_num_threads(4)
    gm = read(directory / 'gradient_manifest.json'); witness = read(directory / 'finite_difference_witness.json')
    sign = 1 if task == 0 else -1
    require(gm['gradient_task'] == task and gm['direction_sign'] == sign and gm['samples'] == (1259 if task == 0 else 997),
            'Gradient objective/sign/sample scope changed')
    require(gm['gradient_batch_size'] == 8 and gm['regularizer'] is None and
            not gm['optimizer_step'] and gm['held_fixed_across_radii'], 'Direction is not the fixed pure CE gradient')
    for key in ('anchor_effective_sha256', 'coordinate_sha256', 'coordinate_numel', 'head_sha256', 'data_manifest_sha256'):
        require(gm[key] == manifest[key], f'Gradient/new-anchor mismatch: {key}')
    sources.add(directory / 'gradient_direction.pt', gm['gradient_artifact_sha256'])
    artifact = torch.load(directory / 'gradient_direction.pt', map_location='cpu', weights_only=True)
    gradients = artifact['gradient_mean_fp64']
    directions = artifact['unit_positive_gradient_fp32' if task == 0 else 'unit_negative_gradient_fp32']
    require(artifact['coordinate_order'] == manifest['coordinate_order'] and len(gradients) == len(directions) == 24,
            'Wrong effective coordinate order/tensor count')
    for values, dtype in ((gradients, torch.float64), (directions, torch.float32)):
        require(all(v.shape == (768, 768) and v.dtype == dtype and bool(torch.isfinite(v).all()) for v in values),
                'Gradient tensor shape/dtype/finiteness failed')
    norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
    unorm = math.sqrt(sum(float(u.double().square().sum()) for u in directions))
    dot = sum(float((g * u.double()).sum()) for g, u in zip(gradients, directions))
    require(norm > 0 and dot * sign > 0 and all(torch.equal(u, (sign * g / norm).float()) for g, u in zip(gradients, directions)),
            'Saved direction is not the requested signed unit gradient')
    require(tensor_hash(gradients) == gm['gradient_sha256'] and tensor_hash(directions) == gm['direction_sha256'],
            'Saved gradient tensor hashes differ')
    close(unorm, 1., 'Global direction norm', atol=5e-7, rtol=0)
    for field, actual in [('raw_grad_norm', norm), ('direction_norm', unorm), ('gradient_dot_direction', dot)]:
        close(gm[field], actual, f'Independent {field}')
    require(witness['gradient_task'] == task and witness['direction_sign'] == sign and
            witness['samples'] == gm['samples'] and witness['direction_sha256'] == gm['direction_sha256'], 'FD objective changed')
    require([x['h'] for x in witness['steps']] == [1e-4, 5e-4, 1e-3] and witness['origin_restored_after_witness'], 'FD grid/restoration changed')
    errors = []
    for step in witness['steps']:
        central = (step['loss_plus'] - step['loss_minus']) / (2 * step['h'])
        close(step['central_derivative'], central, 'FD central derivative')
        close(step['forward_derivative'], (step['loss_plus'] - witness['origin_loss']) / step['h'], 'FD forward derivative')
        errors.append(abs(central - sign * norm) / norm)
        require(step['anchor_restored_exactly'], 'FD point restoration failed')
    require(min(errors) <= .05 and (witness['steps'][0]['loss_plus'] - witness['origin_loss']) * sign > 0,
            'Finite-difference sign/gradient agreement failed')
    old_folder, old_job, _ = OLD[task]
    old_dir = BASE / old_folder / f'results/seed{seed}/gam/{old_job}'
    data = read(directory / 'gradient_sample_manifest.json')
    old_data = read(old_dir / 'gradient_sample_manifest.json')
    require(data['samples'] == gm['samples'] and data['gradient_task'] == task and data['split'] == 'train' and
            data['records'] == old_data['records'] and data['records_sha256'] == old_data['records_sha256'],
            'Gradient sample identities differ from matched old diagnostic')
    sources.add(directory / 'gradient_sample_manifest.json', gm['sample_manifest_sha256'])
    sources.add(old_dir / 'gradient_sample_manifest.json')
    if task == 1:
        head = read(directory / 'task1_head_manifest.json'); old_head = read(old_dir / 'task1_head_manifest.json')
        require(head['trained'] is False and head['future_trained_checkpoint_used'] is False and
                head['named_tensor_sha256'] == old_head['named_tensor_sha256'] == gm['task1_initial_head_sha256'] and
                head['old_task0_head_sha256'] == manifest['head_sha256'], 'New/old classifier head isolation failed')
        require(head['global_classes'] == list(range(20, 40)) and head['local_classes'] == list(range(20)), 'Task1 labels changed')
        sources.add(directory / 'task1_head_manifest.json'); sources.add(old_dir / 'task1_head_manifest.json')
        sources.add(directory / 'task1_initial_head.pt', head['head_artifact_sha256'])
        a = torch.load(directory / 'task1_initial_head.pt', map_location='cpu', weights_only=True)
        b = torch.load(old_dir / 'task1_initial_head.pt', map_location='cpu', weights_only=True)
        require(set(a) == set(b) == {'weight', 'bias'} and all(torch.equal(a[k], b[k]) for k in a),
                'Actual Task1 classifier differs from paired untrained head')
        sources.add(old_dir / 'task1_initial_head.pt', old_head['head_artifact_sha256'])
    return dict(passed=True, actual_signed_unit_gradient_recomputed=True, raw_grad_norm=norm,
                gradient_dot_direction=dot, direction_norm=unorm, minimum_fd_relative_error=min(errors),
                direction_sha256=gm['direction_sha256'], samples=gm['samples'])


def validate_new_job(arm, seed, task, sources, controls, worker, training):
    import torch
    # CPU reduction order is part of the saved bitwise unit-vector protocol.
    torch.set_num_threads(4)
    directory = result_paths(arm, seed, task)
    worker_check = worker.validate_artifacts(directory, load_tensors=True)
    receipt = completed_hashes(directory, sources)
    manifest, summary, raw = read(directory / 'manifest.json'), read(directory / 'summary.json'), lines(directory / 'states.jsonl')
    sign = 1 if task == 0 else -1
    require(receipt['gradient_task'] == task and receipt['direction_sign'] == sign and receipt['evaluation_task'] == 0 and
            receipt['intervention_arm'] == arm and receipt['seed'] == seed and receipt['method'] == 'gam', 'New job identity changed')
    require(receipt['result_check']['passed'] and receipt['witness_check']['passed'], 'New job numeric validation failed')
    rho, norm = ARMS[arm]
    for value in (receipt, manifest, summary):
        require(value['intervention_arm'] == arm and value['training_probe_radius'] == rho and
                value['training_norm_radius'] == norm, 'New radius intervention identity changed')
    old = controls[(task, seed, 'gam')]
    for key in SAME_DIAGNOSTIC:
        require(manifest[key] == old[key], f'New/old diagnostic protocol mismatch: {key}')
    require(manifest['checkpoint_sha256'] == {'gam': {'0': training['checkpoint_sha256']}}, 'Wrong new checkpoint')
    by_radius = {float(x['point']['radius']): x for x in raw}
    require(len(raw) == len(by_radius) == 11 and sorted(by_radius) == RADII and
            summary['states_completed'] == summary['expected_states'] == 11, 'New grid missing or duplicated')
    require(summary['directions_per_radius'] == 1 and summary['radii'] == RADII, 'New direction/grid count changed')
    points = {float(x['radius']): x for x in summary['radius_summary']}
    require(sorted(points) == RADII, 'Summary grid mismatch')
    gradient = validate_vector_witness(directory, manifest, task, seed, sources)
    output = []
    for radius in RADII:
        row = by_radius[radius]
        require(row['seed'] == seed and row['method'] == 'gam' and row['anchor_restored_exactly'], 'Raw state identity/restoration failed')
        require(row['protocol_sha256'] == manifest['protocol_sha256'] and
                row['checkpoint_sha256'] == manifest['checkpoint_sha256']['gam'], 'Raw checkpoint/protocol mismatch')
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == manifest[key], f'Raw {key} mismatch')
        close(row['amount'], radius, 'Raw radius amount')
        close(row['effective_weight_drift'], radius, 'Actual weight drift', atol=1e-5, rtol=2e-6)
        if radius:
            direction = row['point']['direction']
            require(direction['sha256'] == gradient['direction_sha256'] and direction['direction_sign'] == sign and
                    direction['task'] == task and direction['held_fixed_across_radii'], 'Raw direction is not fixed')
        values = result_values(row, by_radius[0.], task, arm, seed)
        validate_summary_point(points[radius], values)
        output.append(values)
    return output, dict(kind='new_intervention', intervention_arm=arm, seed=seed, gradient_task=task,
                        status='validated', states=11, independent_raw_statistics_recomputed=True,
                        independent_vector_witness=gradient, worker_check=worker_check)


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['gradient_task'], row['group'], row['radius'])].append(row)
    output = []
    for (task, group, radius), values in sorted(groups.items()):
        require(sorted(x['seed'] for x in values) == [0, 1, 2], 'Every plotted mark requires 3 independent training seeds')
        for metric in METRICS:
            vector = [x[metric] for x in sorted(values, key=lambda x: x['seed'])]
            output.append(dict(gradient_task=task, group=group, label=LABELS[group], radius=radius,
                               metric=metric, mean=statistics.mean(vector), sample_sd=statistics.stdev(vector), n=3))
    return output


def paired_statistics(rows):
    indexed = {(x['gradient_task'], x['group'], x['seed'], x['radius']): x for x in rows}
    each, summary = [], []
    for task in JOBS:
        for arm in ARMS:
            for control in ('ewc', 'old_gam'):
                for radius in RADII:
                    changes = []
                    for seed in range(3):
                        new, old = indexed[(task, arm, seed, radius)], indexed[(task, control, seed, radius)]
                        entry = dict(gradient_task=task, intervention_arm=arm, comparator=control,
                                     seed=seed, radius=radius, difference_definition='new minus comparator')
                        entry.update({metric: new[metric] - old[metric] for metric in METRICS})
                        each.append(entry); changes.append(entry)
                    for metric in METRICS:
                        values = [x[metric] for x in changes]
                        summary.append(dict(gradient_task=task, intervention_arm=arm, comparator=control,
                            radius=radius, metric=metric, paired_n=3,
                            paired_mean_change=statistics.mean(values), paired_sample_sd=statistics.stdev(values),
                            delta_seed0=values[0], delta_seed1=values[1], delta_seed2=values[2],
                            new_lower_seeds=sum(x < 0 for x in values), new_higher_seeds=sum(x > 0 for x in values)))
    return each, summary


def plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rq2-gam-radius-intervention-mpl'))
    pc = load_module(BASE / 'figures_formal_20260921/plot_common.py', 'rq2_radius_intervention_plot_style')
    return pc, pc.setup_style()


def caption(task):
    direction = ('the positive gradient of mean Task0 training CE over all 1259 images'
                 if task == 0 else 'the negative gradient of mean Task1 training CE over all 997 images, using the paired untrained new20 classifier')
    return (
        'Controlled GAM training-radius intervention on formal ImageNet-A (ViT-B/16, rank-10 Q/V LoRA; seeds 0, 1, 2). '
        'The four curves are the existing EWCLoRA control, existing FS-EWCLoRA, and two newly trained FS-EWCLoRA anchors. '
        'The ordered pair in each FS-EWCLoRA legend is (rho, r_train): the normalized-gradient probe radius and the '
        'gradient-difference ascent radius used during GAM training. The existing pair is (0.05, 0.2); the two interventions '
        'are (0.05, 1) and (1, 1). Both active as1 aliases are changed consistently. All other training settings, '
        'the pretrained frozen backbone, fresh LoRA/head initialization, 50-epoch training budget, and paired data order '
        'are held fixed. These training radii are measured in trainable LoRA-factor/classifier coordinates and are distinct '
        'from the horizontal-axis diagnostic radius r in effective Q/V weights. At each trained anchor, compute once '
        + direction + ', globally normalize the 24 effective Q/V gradient matrices, and hold that one direction fixed '
        'through theta(r)=theta(0)+r*u. Each method and intervention uses its own newly computed direction. '
        'The horizontal grid is the 11 measured points 0, 0.5, ..., 5; old controls are selected from their completed denser scans. '
        'Left: signed Task0 training CE change D0(r)=L0(theta(r))-L0(theta(0)); the overbar denotes the mean over three training seeds. '
        'L0 is mean CE per image over all 1259 Task0 training images. Negative loss changes are retained without clipping. '
        'Middle: accuracy (%) on the same images. Right: raw empirical Fisher trace, mean_i ||gradient CE_i||^2 on the unchanged '
        '256-image Task0 training subset, in the same effective Q/V coordinates. The original Task0 20-class classifier '
        'and all other model state remain fixed throughout each scan. No extra Fisher normalization, floor, or EWC lambda '
        'is applied. Lines and shading show arithmetic mean and one sample standard deviation (ddof=1) across three '
        'training seeds, not SEM or confidence intervals. Connecting segments are guides and introduce no measured points. '
        'The separate CSV includes absolute CE, positive-part damage and test metrics, while these panels show signed training damage. '
        'A separate origin table reports the fit at every trained anchor. A smaller CE increment alone does not imply '
        'a lower absolute perturbed CE or a better-trained anchor; both quantities must be inspected before a robustness claim. '
        'Small finite-difference probes verify the requested gradient sign; no sign or monotonicity condition is imposed '
        'on finite-radius Task0 loss changes. This fixed effective-weight ray is a diagnostic, not an iterative optimization '
        'trajectory or a completed Task1 learning experiment.'
    )


def render(pc, font, aggregates, task, sources, checks, output=None, fixture=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    out = Path(output) if output is not None else ROOT / f'figures/{JOBS[task]}'
    out.mkdir(parents=True, exist_ok=True)
    pc.HERE = out
    fig, axes = pc.plt.subplots(1, 3, figsize=(9.1, 2.35))
    fig.subplots_adjust(left=.067, right=.992, bottom=.255, top=.965, wspace=.43)
    labels = (r'Loss change $\overline{D_0(r)}$', 'Accuracy (%)', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$')
    ranges = []
    for ax, metric, label in zip(axes, PLOT_METRICS, labels):
        shown = [x for x in aggregates if x['gradient_task'] == task and x['metric'] == metric]
        require(len(shown) == len(GROUPS) * len(RADII), 'A plotted series is missing data')
        for group in GROUPS:
            series = sorted((x for x in shown if x['group'] == group), key=lambda x: x['radius'])
            xx = [x['radius'] for x in series]; yy = [x['mean'] for x in series]; sd = [x['sample_sd'] for x in series]
            require(xx == RADII, 'Plotted radius grid differs from the measured grid')
            ax.fill_between(xx, [a-b for a,b in zip(yy,sd)], [a+b for a,b in zip(yy,sd)],
                            color=COLORS[group], alpha=.12, linewidth=0)
            ax.plot(xx, yy, color=COLORS[group], linestyle=LINES[group], marker=MARKERS[group],
                    markersize=3.0, linewidth=1.2, label=LABELS[group], zorder=3)
        lo = min(x['mean']-x['sample_sd'] for x in shown); hi = max(x['mean']+x['sample_sd'] for x in shown)
        if metric == 'D_train':
            lo = min(lo, 0.); ax.axhline(0, color='#999999', linewidth=.45, zorder=0)
        pad = max((hi-lo)*.05, 1e-8)
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.06, 5.06), xticks=list(range(6)), ylim=(lo-pad, hi+pad))
        ticks = MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]).tick_values(lo-pad, hi+pad)
        ax.set_yticks([tick for tick in ticks if lo-pad <= tick <= hi+pad])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2g}' if abs(value)>=1e4 or 0<abs(value)<1e-3 else f'{value:g}'))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.legend(loc='best', frameon=False, fontsize=7.3, handlelength=1.35,
                  handletextpad=.3, labelspacing=.22, borderaxespad=.35, borderpad=0)
        ranges.append([lo-pad,hi+pad])
    name = 'Figure_' + JOBS[task] + '_training_radius_intervention'
    metadata = dict(font=font, gradient_task=task, direction_sign=1 if task == 0 else -1,
                    groups=list(GROUPS), legend_pair_definition='(gradient probe rho, gradient-difference ascent r_train)',
                    measured_radii=RADII, axis_ranges=ranges, source_files=sources,
                    checks=checks, signed_negative_damage_retained=True, extra_fisher_normalization_or_lambda=False,
                    experimental_results=not fixture, layout_fixture_only=fixture)
    result = pc.save_figure(fig, name, metadata)
    if not fixture:
        text = caption(task)
        (out / 'CAPTION.md').write_text(text + '\n')
        tex = text.replace('_', '\\_').replace('%', '\\%')
        (out / 'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'
            + name + '.pdf}\n\\caption{' + tex + '}\n\\label{fig:' + JOBS[task].replace('_','-') + '-radius-intervention}\n\\end{figure*}\n')
    return result


def write_numeric_report(data, aggregates, paired):
    out = ROOT / 'report'
    out.mkdir(parents=True, exist_ok=True)
    origin_rows = [x for x in data if x['gradient_task'] == 0 and x['radius'] == 0]
    require(len(origin_rows) == 12, 'Origin comparison needs 4 groups x 3 seeds')
    write_csv(ROOT / 'figures/anchor_per_seed.csv', origin_rows)
    origin_mean = [x for x in aggregates if x['gradient_task'] == 0 and x['radius'] == 0
                   and x['metric'] in ('absolute_train_ce', 'train_accuracy_percent', 'raw_fisher_trace')]
    write_csv(ROOT / 'figures/anchor_mean_sample_sd.csv', origin_mean)
    lookup = {(x['gradient_task'], x['group'], x['radius'], x['metric']): x for x in aggregates}
    text = ['# Controlled GAM training-radius comparison', '',
            'All six new Task0 training runs and all twelve fixed-direction diagnostics have completed. '
            'Old controls are selected from completed, hash-verified measurements at the same eleven radii. '
            'This report uses formal ImageNet-A; it does not relabel these results as historical ImageNet-R.', '',
            'Legend pairs denote (GAM probe radius, gradient-difference ascent radius), in training parameter coordinates. '
            'The horizontal diagnostic radius is a different quantity: global effective Q/V weight drift.', '',
            '## Anchor fit before perturbation', '',
            'Mean ± sample SD over three training seeds. Each anchor appears once here; both diagnostic directions '
            'must reproduce this same origin.', '',
            '| Model / training radii | Mean training CE | Training accuracy (%) | Raw Fisher trace |',
            '|---|---:|---:|---:|']
    for group in GROUPS:
        cells = []
        for metric in ('absolute_train_ce', 'train_accuracy_percent', 'raw_fisher_trace'):
            row = lookup[(0, group, 0., metric)]
            cells.append(f"{row['mean']:.6g} ± {row['sample_sd']:.3g}")
        text.append('| ' + LABELS[group] + ' | ' + ' | '.join(cells) + ' |')
    text += ['', '**Interpretation rule:** signed damage subtracts each model\'s own origin CE. '
             'A smaller increment can coexist with a worse-fitted anchor or larger absolute perturbed loss. '
             'Read the origin fit, signed damage, absolute CE and accuracy together; the report does not treat '
             'smaller damage alone as a robustness improvement.', '']
    for task in JOBS:
        text += ['## ' + ('Task0 positive-gradient direction' if task == 0 else 'Task1 negative-gradient direction'), '',
                 '| r | Model / training radii | Signed old-task damage | Absolute old-task CE | Accuracy (%) | Raw Fisher |',
                 '|---:|---|---:|---:|---:|---:|']
        for radius in (.5, 1., 2., 5.):
            for group in GROUPS:
                vals = [lookup[(task, group, radius, metric)]['mean']
                        for metric in ('D_train', 'absolute_train_ce', 'train_accuracy_percent', 'raw_fisher_trace')]
                text.append(f'| {radius:g} | {LABELS[group]} | ' + ' | '.join(f'{v:.6g}' for v in vals) + ' |')
        text += ['', 'Paired differences are computed within the same training seed, then summarized across seeds. '
                 'Different anchors have their own gradient directions; this comparison changes both the trained '
                 'anchor and the direction it induces.', '',
                 '| New arm vs control | Lower / higher mean signed damage over 10 nonzero radii | '
                 'Lower damage but higher absolute CE |', '|---|---:|---:|']
        for arm in ARMS:
            for control in ('ewc', 'old_gam'):
                ds = {x['radius']: x['paired_mean_change'] for x in paired
                      if x['gradient_task'] == task and x['intervention_arm'] == arm and
                      x['comparator'] == control and x['metric'] == 'D_train' and x['radius'] > 0}
                absolute = {x['radius']: x['paired_mean_change'] for x in paired
                            if x['gradient_task'] == task and x['intervention_arm'] == arm and
                            x['comparator'] == control and x['metric'] == 'absolute_train_ce' and x['radius'] > 0}
                text.append(f'| {LABELS[arm]} vs {LABELS[control]} | {sum(v < 0 for v in ds.values())} / '
                            f'{sum(v > 0 for v in ds.values())} | '
                            f'{sum(ds[r] < 0 and absolute[r] > 0 for r in ds)} |')
        text += ['', 'These counts describe the measured radius grid; radii are correlated and are not independent '
                 'replicates or a probability of superiority.', '']
    text += ['## Measurement scope', '',
             'Loss is mean cross-entropy per image, not a dataset sum. Each diagnostic direction is computed once '
             'at the trained Task0 anchor: +∇L0 from 1259 old training images, or −∇L1 from 997 next-task training '
             'images and a paired untrained new20 classifier. All plotted evaluations use the anchor\'s own fixed '
             'old20 classifier. Raw Fisher is the mean squared norm of per-example true-label CE gradients on the '
             'same 256 old training images, with no EWC lambda or extra normalization. Signed negative damage '
             'is retained. The curves are fixed effective-weight rays, not real sequential optimizer trajectories.', '',
             'Figures: `figures/task0_positive/Figure_task0_positive_training_radius_intervention.pdf` and '
             '`figures/task1_negative/Figure_task1_negative_training_radius_intervention.pdf`. '
             'The corresponding directories contain PNG, caption and LaTeX include files.', '']
    (out / 'RESULTS.md').write_text('\n'.join(text))


def run(execute=False):
    sources = Sources(); sources.add(__file__); sources.add(ROOT / 'train.py')
    sources.add(BASE / 'figures_formal_20260921/plot_common.py')
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        sources.add(Path('/home-local/liying/.fonts') / name)
    data, controls, checks = load_old_controls(sources)
    pc, font = plotting()
    configs = {(arm, seed): validate_config(arm, seed, sources) for arm in ARMS for seed in range(3)}
    worker_path = ROOT / 'diagnostic.py'
    require(worker_path.is_file(), 'New diagnostic worker is not ready')
    worker = load_module(worker_path, 'rq2_training_radius_intervention_diagnostic')
    require(hasattr(worker, 'validate_artifacts'), 'Worker lacks the completed-artifact validation API')
    sources.add(worker_path)
    training = {}; missing = []; new_checks = []
    for arm in ARMS:
        for seed in range(3):
            trained = (ROOT / f'training/{arm}_seed{seed}/COMPLETE.json').is_file()
            if trained:
                training[(arm, seed)] = validate_training(arm, seed, sources)
            for task in JOBS:
                directory = result_paths(arm, seed, task)
                if not (directory / 'extension_receipt.json').is_file():
                    missing.append(dict(intervention_arm=arm, seed=seed, gradient_task=task, path=str(directory)))
                    continue
                require(trained, 'Diagnostic completed before its training anchor was validated')
                rows, check = validate_new_job(arm, seed, task, sources, controls, worker, training[(arm, seed)])
                data.extend(rows); new_checks.append(check)
    status = 'ready_awaiting_results' if missing else 'validated_complete_inputs'
    result = dict(status=status, old_control_jobs=12, training_anchors=len(training), expected_jobs=12,
                  validated_jobs=len(new_checks), missing_jobs=missing, font=font,
                  read_only=not execute, new_gpu_evaluations=False)
    if not execute:
        print(json.dumps(result, indent=2, ensure_ascii=False)); return result
    require(not missing and len(new_checks) == 12 and len(training) == 6, 'Refusing to publish incomplete intervention results')
    require(len(data) == 264, 'Expected 4 groups x 3 seeds x 2 directions x 11 radii')
    # Two directions for each newly trained anchor must repeat the same Task0 origin.
    indexed = {(x['gradient_task'], x['group'], x['seed'], x['radius']): x for x in data}
    for group in GROUPS:
        for seed in range(3):
            for metric in ('train_ce', 'test_ce', 'train_accuracy_percent', 'test_accuracy_percent', 'raw_fisher_trace'):
                close(indexed[(0,group,seed,0.)][metric], indexed[(1,group,seed,0.)][metric],
                      'Cross-direction origin/'+metric, atol=3e-7, rtol=1e-5)
    aggregates = aggregate(data); each, paired = paired_statistics(data)
    sources.recheck()
    figures = ROOT / 'figures'
    write_csv(figures / 'all_per_seed.csv', data)
    write_csv(figures / 'all_mean_sample_sd.csv', aggregates)
    write_csv(figures / 'paired_per_seed.csv', each)
    write_csv(figures / 'paired_mean_sample_sd.csv', paired)
    write_numeric_report(data, aggregates, paired)
    rendered = [render(pc, font, aggregates, task, sources.records(), new_checks) for task in JOBS]
    sources.recheck()
    files = [dict(path=str(p), sha256=sha(p)) for p in sorted(figures.rglob('*')) if p.is_file()]
    files.append(dict(path=str(ROOT / 'report/RESULTS.md'), sha256=sha(ROOT / 'report/RESULTS.md')))
    receipt = dict(status='complete', expected_jobs=12, validated_jobs=12, new_states=132,
                   plotted_rows=264, training_anchors=6, training_checks=list(training.values()),
                   new_job_checks=new_checks, old_control_checks=checks, groups=list(GROUPS),
                   measured_radii=RADII, gradient_tasks=[0,1], evaluation_task=0,
                   extra_fisher_normalization_or_lambda=False, signed_loss_changes_preserved=True,
                   statistics='Paired 3 training seeds; arithmetic mean and sample SD ddof=1; no SEM/CI',
                   source_files=sources.records(), output_files=files,
                   figures=rendered, old_results_preserved=True,
                   generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   rendered_visual_review='pending actual rendered-image inspection by root/reviewer')
    atomic_json(ROOT / 'report/radius_intervention_complete.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_files','output_files','figures','training_checks','new_job_checks','old_control_checks')}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--validate-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(execute=args.execute)


if __name__ == '__main__':
    main()
