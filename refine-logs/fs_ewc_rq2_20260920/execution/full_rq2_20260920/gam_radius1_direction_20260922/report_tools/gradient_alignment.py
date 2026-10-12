#!/usr/bin/env python3
"""Read-only CPU alignment analysis of existing, completed diagnostic artifacts.

Default: report real available pairs and explicitly mark missing new jobs pending.
--old-only: verify the twelve historical jobs and report their six within-anchor
alignments. No result is simulated, no model is constructed, and no file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
ARMS = {'probe005_norm1': (.05, 1.), 'probe1_norm1': (1., 1.)}
COORDINATES = [f'block.{block}.effective_{kind}[out,in]' for block in range(12) for kind in ('q', 'v')]
NUMEL = 24 * 768 * 768
JOBS = {0: 'task0_positive', 1: 'task1_negative'}
OLD = {0: ('fixed_gradient_extension_20260921', 'fixed_gradient'),
       1: ('task1_gradient_extension_20260921', 'task1_gradient')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(memoryview(value.numpy()).cast('B'))
    return digest.hexdigest()


def close(actual, expected, label, atol=1e-10, rtol=1e-10):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
            f'{label}: {actual} != {expected}')


def directory_for(group, seed, task):
    if group in ARMS:
        return ROOT / f'results/{group}/seed{seed}/gam/{JOBS[task]}'
    folder, job = OLD[task]
    method = 'sgd' if group == 'ewc' else 'gam'
    return BASE / folder / f'results/seed{seed}/{method}/{job}'


def inspect_job(group, seed, task):
    """Require a durable completion receipt before opening any large tensors."""
    directory = directory_for(group, seed, task)
    required = ['extension_receipt.json', 'manifest.json', 'gradient_manifest.json',
                'gradient_direction.pt', 'summary.json']
    missing = [str(directory / name) for name in required if not (directory / name).is_file()]
    if missing:
        return dict(status='pending', group=group, seed=seed, gradient_task=task, missing=missing)
    receipt = read(directory / 'extension_receipt.json')
    require(receipt['status'] == 'complete', f'Existing receipt is not complete: {directory}')
    for field in ('identity_check', 'origin_check', 'witness_check', 'result_check'):
        require(receipt[field]['passed'] is True, f'Failed completed-job check {field}: {directory}')
    sources = []
    for filename, key in [('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256'),
                          ('gradient_manifest.json', 'gradient_manifest_sha256'),
                          ('gradient_direction.pt', 'gradient_artifact_sha256')]:
        digest = sha(directory / filename)
        require(digest == receipt[key], f'Completed artifact hash mismatch: {directory / filename}')
        sources.append(dict(path=str(directory / filename), sha256=digest))
    sources.append(dict(path=str(directory / 'extension_receipt.json'), sha256=sha(directory / 'extension_receipt.json')))
    meta, gm, summary = read(directory / 'manifest.json'), read(directory / 'gradient_manifest.json'), read(directory / 'summary.json')
    method, sign = ('sgd' if group == 'ewc' else 'gam'), (1 if task == 0 else -1)
    for value in (receipt, meta, gm, summary):
        require(value['seed'] == seed and value['method'] == method and value['gradient_task'] == task and
                value['direction_sign'] == sign, f'Job identity/sign changed: {directory}')
    require(meta.get('evaluation_task', 0) == receipt.get('evaluation_task', 0) == 0, 'Expected Task0 evaluation')
    require(canonical({k: v for k, v in meta.items() if k != 'protocol_sha256'}) == meta['protocol_sha256'] ==
            receipt['protocol_sha256'] == summary['protocol_sha256'], f'Protocol hash mismatch: {directory}')
    require(meta['coordinate_order'] == COORDINATES and meta['coordinate_numel'] == gm['coordinate_numel'] == NUMEL and
            meta['coordinate_sha256'] == gm['coordinate_sha256'] == canonical(COORDINATES),
            f'Effective Q/V coordinate schema mismatch: {directory}')
    for key in ('anchor_effective_sha256', 'head_sha256', 'data_manifest_sha256'):
        require(gm[key] == meta[key], f'Gradient {key} does not match its anchor')
    require(gm['checkpoint_sha256'] == meta['checkpoint_sha256'][method], 'Gradient checkpoint mismatch')
    require(gm['gradient_artifact_sha256'] == receipt['gradient_artifact_sha256'], 'Gradient artifact binding mismatch')
    require(gm['samples'] == (1259 if task == 0 else 997) and gm['gradient_batch_size'] == 8 and
            gm['regularizer'] is None and gm['optimizer_step'] is False and gm['held_fixed_across_radii'] is True,
            'Expected one fixed, unregularized full-training CE direction')
    require(summary['direction_sha256'] == gm['direction_sha256'] and summary['fixed_state_restored'] is True,
            'Completed direction/restoration mismatch')
    if group in ARMS:
        probe, norm = ARMS[group]
        for value in (receipt, meta, summary):
            require(value['intervention_arm'] == group and value['training_probe_radius'] == probe and
                    value['training_norm_radius'] == norm, 'Intervention arm/radius mismatch')
        require(meta['job'] == JOBS[task] and summary['states_completed'] == summary['expected_states'] == 11,
                'New diagnostic coverage is incomplete')
        expected_checkpoint = ROOT / f'training/{group}_seed{seed}/checkpoints/task_00.pt'
        require(meta['checkpoints'] == {'gam': {'0': str(expected_checkpoint)}}, 'Expected this intervention new Task0 checkpoint')
    if task == 1:
        require(gm['task1_head_trained'] is False and gm['evaluation_task'] == 0, 'Expected the untrained Task1 classifier')
    return dict(status='ready', group=group, seed=seed, gradient_task=task, directory=directory,
                meta=meta, gm=gm, summary=summary, sources=sources)


def load_vectors(job):
    import torch
    artifact = torch.load(job['directory'] / 'gradient_direction.pt', map_location='cpu', weights_only=True, mmap=True)
    task, gm = job['gradient_task'], job['gm']
    gradients = artifact['gradient_mean_fp64']
    directions = artifact['unit_positive_gradient_fp32' if task == 0 else 'unit_negative_gradient_fp32']
    require(artifact['coordinate_order'] == job['meta']['coordinate_order'] == COORDINATES, 'Artifact coordinate order mismatch')
    require(len(gradients) == len(directions) == len(COORDINATES), 'Coordinate tensor count mismatch')
    for values, dtype in ((gradients, torch.float64), (directions, torch.float32)):
        require(all(tuple(value.shape) == (768, 768) and value.dtype == dtype and value.device.type == 'cpu' and
                    bool(torch.isfinite(value).all()) for value in values), 'Wrong coordinate shape/dtype/device/values')
    require(tensor_hash(gradients) == gm['gradient_sha256'] and tensor_hash(directions) == gm['direction_sha256'],
            'Actual saved gradient/direction tensor hash mismatch')
    norm = math.sqrt(sum(float(value.square().sum()) for value in gradients))
    direction_norm = math.sqrt(sum(float(value.double().square().sum()) for value in directions))
    require(norm > 0 and math.isfinite(norm), 'Zero/nonfinite gradient')
    sign = 1 if task == 0 else -1
    require(all(torch.equal(u, (sign * g / norm).float()) for g, u in zip(gradients, directions)),
            'Actual direction does not equal the requested normalized gradient')
    dot = dot_product(gradients, directions)
    close(norm, gm['raw_grad_norm'], 'Gradient norm')
    close(direction_norm, gm['direction_norm'], 'Direction norm')
    close(direction_norm, 1., 'Unit global Frobenius norm', atol=5e-7, rtol=0.)
    close(dot, gm['gradient_dot_direction'], 'Signed directional derivative')
    return dict(job=job, gradients=gradients, directions=directions, norm=norm, direction_norm=direction_norm)


def dot_product(left, right):
    require(len(left) == len(right) == len(COORDINATES), 'Cannot take an unstructured flattened dot product')
    require(all(tuple(a.shape) == tuple(b.shape) == (768, 768) for a, b in zip(left, right)),
            'Coordinate-wise tensor shapes do not match')
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))


def same_coordinates(left, right):
    a, b = left['job']['meta'], right['job']['meta']
    for key in ('coordinate_order', 'coordinate_sha256', 'coordinate_numel', 'source_path', 'source_files',
                'data_manifest_sha256', 'counts', 'production_precision', 'fisher_definition'):
        require(a[key] == b[key], f'Incompatible diagnostic coordinate/data schema: {key}')


def cosine(left, right):
    same_coordinates(left, right)
    raw = dot_product(left['directions'], right['directions'])
    normalized = raw / (left['direction_norm'] * right['direction_norm'])
    require(math.isfinite(normalized) and abs(normalized) <= 1. + 1e-10, 'Invalid direction cosine')
    return normalized, raw


def within_anchor(task0, task1):
    a, b = task0['job'], task1['job']
    require(a['gradient_task'] == 0 and b['gradient_task'] == 1 and
            (a['group'], a['seed']) == (b['group'], b['seed']), 'Within-anchor pair identity mismatch')
    same_coordinates(task0, task1)
    for key in ('checkpoint_sha256', 'anchor_effective_sha256', 'head_sha256'):
        require(a['meta'][key] == b['meta'][key], f'Two directions belong to different anchors: {key}')
    alignment, raw_dot = cosine(task0, task1)
    derivative = dot_product(task0['gradients'], task1['directions'])
    norm_cosine = task0['norm'] * alignment
    # The identity is exact for ideal unit vectors; stored FP32 normalization
    # introduces rounding. Retain the direct g0 dot u1 as the actual ray slope.
    close(derivative, norm_cosine, 'Local slope versus norm-times-cosine', atol=task0['norm'] * 5e-7 + 1e-10, rtol=0.)
    origin0 = next(row for row in a['summary']['radius_summary'] if row['radius'] == 0)
    origin1 = next(row for row in b['summary']['radius_summary'] if row['radius'] == 0)
    close(origin0['train']['absolute_loss_mean'], origin1['train']['absolute_loss_mean'], 'Shared Task0 anchor CE', atol=3e-7, rtol=1e-5)
    close(origin0['train']['absolute_loss_mean'], a['gm']['objective_at_anchor'], 'Task0 mean-gradient objective', atol=3e-6, rtol=1e-7)
    return dict(group=a['group'], seed=a['seed'], cosine_u0_positive_u1_negative=alignment,
                saved_direction_inner_product=raw_dot,
                task0_gradient_norm=task0['norm'], task1_gradient_norm=task1['norm'],
                task0_direction_norm=task0['direction_norm'], task1_direction_norm=task1['direction_norm'],
                dL0_dr_along_u1_direct=derivative, dL0_dr_norm_times_cosine=norm_cosine,
                fp32_normalization_residual=derivative - norm_cosine,
                task0_anchor_train_ce=origin0['train']['absolute_loss_mean'],
                task0_anchor_train_accuracy_percent=origin0['train']['accuracy_mean'],
                task0_anchor_raw_fisher=origin0['raw_trace_by_scope']['subset']['mean'],
                task1_anchor_train_ce=b['gm']['objective_at_anchor'],
                anchor_effective_sha256=a['meta']['anchor_effective_sha256'], head_sha256=a['meta']['head_sha256'],
                task0_direction_sha256=a['gm']['direction_sha256'], task1_direction_sha256=b['gm']['direction_sha256'])


def between_anchors(new, old):
    a, b = new['job'], old['job']
    require(a['group'] in ARMS and b['group'] == 'old_gam' and
            (a['seed'], a['gradient_task']) == (b['seed'], b['gradient_task']), 'New/old GAM comparison not paired')
    for key in ('gradient_samples_sha256', 'training_labels_sha256', 'decoded_train_sha256'):
        require(a['gm'][key] == b['gm'][key], f'New/old gradient objective sample identity changed: {key}')
    if a['gradient_task'] == 1:
        require(a['gm']['task1_initial_head_sha256'] == b['gm']['task1_initial_head_sha256'], 'Task1 initial classifier is not paired')
    alignment, raw_dot = cosine(new, old)
    return dict(intervention_arm=a['group'], seed=a['seed'], gradient_task=a['gradient_task'],
                direction_sign=1 if a['gradient_task'] == 0 else -1,
                cosine_new_direction_old_gam_direction=alignment, saved_direction_inner_product=raw_dot,
                new_gradient_norm=new['norm'], old_gam_gradient_norm=old['norm'],
                new_objective_at_anchor=a['gm']['objective_at_anchor'], old_gam_objective_at_anchor=b['gm']['objective_at_anchor'],
                new_anchor_effective_sha256=a['meta']['anchor_effective_sha256'],
                old_gam_anchor_effective_sha256=b['meta']['anchor_effective_sha256'],
                new_direction_sha256=a['gm']['direction_sha256'], old_gam_direction_sha256=b['gm']['direction_sha256'],
                different_task0_heads_permitted=True)


def run(old_only=False):
    import torch
    torch.set_num_threads(4)
    groups = ('ewc', 'old_gam') if old_only else ('ewc', 'old_gam', *ARMS)
    inventory = {(group, seed, task): inspect_job(group, seed, task)
                 for group in groups for seed in range(3) for task in (0, 1)}
    missing = [job for job in inventory.values() if job['status'] == 'pending']
    ready = [job for job in inventory.values() if job['status'] == 'ready']
    within, between = [], []
    for seed in range(3):
        # Retain old GAM vectors while comparing one intervention at a time.
        # mmap+coordinate-wise reduction avoids flattening or GPU allocations.
        old_gam = {task: load_vectors(inventory[('old_gam', seed, task)]) for task in (0, 1)
                   if inventory[('old_gam', seed, task)]['status'] == 'ready'}
        if len(old_gam) == 2:
            within.append(within_anchor(old_gam[0], old_gam[1]))
        for group in groups:
            if group == 'old_gam':
                continue
            pair = {task: load_vectors(inventory[(group, seed, task)]) for task in (0, 1)
                    if inventory[(group, seed, task)]['status'] == 'ready'}
            if len(pair) == 2:
                within.append(within_anchor(pair[0], pair[1]))
            if group in ARMS:
                for task in pair.keys() & old_gam.keys():
                    between.append(between_anchors(pair[task], old_gam[task]))
            del pair
        del old_gam
    sources = {record['path']: record['sha256'] for job in ready for record in job['sources']}
    # Detect concurrent edits without writing a source snapshot.
    for path, digest in sources.items():
        require(sha(path) == digest, f'Input changed while computing alignment: {path}')
    return dict(status='pending' if missing else 'passed', scope='historical_only' if old_only else 'historical_and_interventions',
                read_only=True, gpu_used=False, production_models_constructed=0,
                tool_sha256=sha(__file__), expected_jobs=len(inventory), validated_existing_jobs=len(ready), pending_jobs=missing,
                coordinate_schema=dict(order=COORDINATES, shapes=[[768, 768]] * len(COORDINATES),
                                       coordinate_numel=NUMEL, coordinate_sha256=canonical(COORDINATES)),
                within_anchor_alignment=sorted(within, key=lambda row: (row['group'], row['seed'])),
                intervention_vs_old_gam_direction=sorted(between, key=lambda row: (row['intervention_arm'], row['seed'], row['gradient_task'])),
                interpretation=['u0 is the Task0 ascent direction and u1 is the Task1 descent direction.',
                    'Negative within-anchor cosine predicts a local decrease of Task0 CE along Task1 descent; positive cosine predicts a local increase.',
                    'The directly computed g0 dot u1 is the slope at radius zero, not a prediction of the entire finite-radius curve.',
                    'New/old GAM direction cosines compare the same named effective Q/V axes across different anchors; they do not require equal trained Task0 heads.',
                    'Anchor CE/Fisher and directional alignment are complementary descriptions, not a causal decomposition of the training intervention.'],
                source_files=[dict(path=path, sha256=digest) for path, digest in sorted(sources.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-only', action='store_true')
    parser.add_argument('--require-complete', action='store_true', help='Return exit 2 if any requested completed job is missing')
    args = parser.parse_args()
    try:
        result = run(old_only=args.old_only)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps(dict(status='invalid', read_only=True, error=str(error)), indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.require_complete and result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
