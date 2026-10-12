#!/usr/bin/env python3
"""Publish the fixed Task-0 positive-gradient diagnostic using CPU only.

Default/--check-only is a read-only coverage and completed-file-hash inventory.
--execute requires six complete, valid seed/method jobs before rendering either
the full radius-50 figure or its radius-2 companion. No GPU or training launch.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
METHODS = ('sgd', 'gam')
RADII = sorted(set([i / 4 for i in range(81)] + [25, 30, 35, 40, 45, 50]
                   + [.001, .005, .01, .025, .05, .1]))
SMALL_RADII = [.001, .005, .01, .025, .05, .1]
STYLE = BASE / 'figures_formal_20260921'
COMMON_IDENTITY = ('seed', 'method', 'coordinate_order', 'coordinate_numel',
                  'coordinate_sha256', 'data_manifest_sha256', 'head_sha256',
                  'anchor_effective_sha256', 'source_files', 'source_path', 'counts',
                  'production_precision', 'fisher_definition', 'torch_version',
                  'diagnostic_code_sha256', 'common_code_sha256', 'pilot_runner_code_sha256')
REQUIRED_FILES = ('summary.json', 'manifest.json', 'states.jsonl', 'extension_receipt.json',
                  'gradient_direction.pt', 'gradient_manifest.json', 'gradient_sample_manifest.json',
                  'finite_difference_witness.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_receipt(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, *, atol=1e-12, rtol=1e-10, label='value'):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f'{label}: nonfinite')
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
            f'{label}: {actual} != {expected}')


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite numeric source value')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def directory_for(seed, method):
    return ROOT / f'results/seed{seed}/{method}/fixed_gradient'


def check_completed_hashes(directory):
    receipt = read(directory / 'extension_receipt.json')
    require(receipt['status'] == 'complete', f'Incomplete worker receipt: {directory}')
    for field, filename in (('summary_sha256', 'summary.json'), ('manifest_sha256', 'manifest.json'),
                            ('states_sha256', 'states.jsonl'),
                            ('gradient_artifact_sha256', 'gradient_direction.pt'),
                            ('gradient_manifest_sha256', 'gradient_manifest.json'),
                            ('sample_manifest_sha256', 'gradient_sample_manifest.json'),
                            ('witness_sha256', 'finite_difference_witness.json')):
        require(receipt[field] == sha(directory / filename), f'Worker {field} mismatch: {directory}')
    return receipt


def inventory():
    jobs = []
    for seed in range(3):
        for method in METHODS:
            directory = directory_for(seed, method)
            files = [directory / name for name in REQUIRED_FILES] + [ROOT / f'configs/seed{seed}.json']
            missing = [str(path) for path in files if not path.is_file()]
            status, error = ('missing' if missing else 'incomplete'), None
            if not missing:
                try:
                    check_completed_hashes(directory)
                    require(read(directory / 'summary.json')['status'] == 'complete', 'Summary incomplete')
                    status = 'complete'
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    status, error = 'invalid', str(exc)
            jobs.append(dict(seed=seed, method=method, status=status, missing=missing, error=error))
    complete = sum(item['status'] == 'complete' for item in jobs)
    return dict(status='ready_to_validate' if complete == 6 else 'pending', expected_jobs=6,
                complete_job_files=complete, jobs=jobs, gradient_task=0, radii=RADII,
                states_per_job=len(RADII), final_receipt=str(ROOT / 'report/fixed_gradient_complete.json'),
                check_only_is_read_only=True)


def read_formal(seed, method):
    directory = BASE / f'formal_results/seed{seed}/{method}/sparse_train'
    summary, manifest = read(directory / 'summary.json'), read(directory / 'manifest.json')
    coverage_path = BASE / 'final_report/coverage.json'
    matches = [row for row in read(coverage_path)['diagnostics']
               if (row['seed'], row['method'], row['job']) == (seed, method, 'sparse_train')]
    require(len(matches) == 1 and matches[0]['sha256'] == sha(directory / 'summary.json'), 'Formal summary hash changed')
    require(canonical({k: v for k, v in manifest.items() if k != 'protocol_sha256'}) ==
            manifest['protocol_sha256'] == summary['protocol_sha256'], 'Formal protocol hash mismatch')
    rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
    require(summary['status'] == 'complete' and len(rows) == matches[0]['states'], 'Formal reference incomplete')
    origin = next(row for row in rows if row['point']['radius'] == 0)
    sources = [file_receipt(path) for path in (coverage_path, directory / 'summary.json',
                                              directory / 'manifest.json', directory / 'states.jsonl')]
    return manifest, origin, sources


def validate_gradient(directory, manifest, rows, seed, method):
    """Independently rehash/reduce the saved gradient and unit ray on CPU."""
    import torch
    torch.set_num_threads(4)
    gradient_meta = read(directory / 'gradient_manifest.json')
    sample_meta = read(directory / 'gradient_sample_manifest.json')
    witness = read(directory / 'finite_difference_witness.json')
    finite(gradient_meta); finite(sample_meta); finite(witness)
    require(manifest['fixed_gradient'] == gradient_meta, 'Manifest gradient metadata differs from saved metadata')
    require(gradient_meta['gradient_task'] == 0 and gradient_meta['direction_sign'] == 1 and
            gradient_meta['samples'] == 1259 and gradient_meta['seed'] == seed and gradient_meta['method'] == method,
            'Gradient is not the matching full Task0 positive-gradient objective')
    require(gradient_meta['gradient_batch_size'] == 8 and gradient_meta['sample_reduction'] ==
            'sum of CE gradients over every training example, divided once by N=1259', 'Gradient reduction changed')
    require(gradient_meta['regularizer'] is None and gradient_meta['optimizer_step'] is False and
            gradient_meta['held_fixed_across_radii'] is True, 'Gradient is regularized, optimized, or not fixed')
    for key in ('anchor_effective_sha256', 'coordinate_sha256', 'coordinate_numel', 'head_sha256', 'data_manifest_sha256'):
        require(gradient_meta[key] == manifest[key], f'Gradient {key} identity changed')
    require(gradient_meta['checkpoint_sha256'] == manifest['checkpoint_sha256'][method], 'Gradient checkpoint mismatch')
    require(gradient_meta['wrapper_sha256'] == sha(ROOT / 'diagnostic_fixed_gradient.py'), 'Gradient worker changed')
    for field, filename in (('gradient_manifest_sha256', 'gradient_manifest.json'),
                            ('gradient_artifact_sha256', 'gradient_direction.pt'),
                            ('witness_sha256', 'finite_difference_witness.json')):
        require(manifest[field] == sha(directory / filename), f'Manifest {field} mismatch')
    require(gradient_meta['gradient_artifact_sha256'] == sha(directory / 'gradient_direction.pt') and
            gradient_meta['sample_manifest_sha256'] == sha(directory / 'gradient_sample_manifest.json'),
            'Gradient artifact/sample hash mismatch')
    formal_data_path = BASE / f'formal_results/seed{seed}/{method}/data_manifest.json'
    formal_data = read(formal_data_path)
    require(canonical({k: v for k, v in formal_data.items() if k != 'manifest_sha256'}) ==
            formal_data['manifest_sha256'] == manifest['data_manifest_sha256'], 'Formal data manifest hash mismatch')
    require(sample_meta['gradient_task'] == 0 and sample_meta['split'] == 'train' and sample_meta['samples'] == 1259,
            'Gradient sample scope changed')
    records = sample_meta['records']
    require(records == formal_data['records']['train'] and len(records) == len({row['id'] for row in records}) == 1259,
            'Gradient samples are not exactly all formal Task0 training examples')
    require(sample_meta['records_sha256'] == canonical(records) == gradient_meta['gradient_samples_sha256'],
            'Gradient sample identity hash mismatch')
    require(gradient_meta['decoded_train_sha256'] == formal_data['decoded_tensor_sha256']['train'],
            'Gradient decoded training images differ')

    def tensor_hash(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            tensor = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(memoryview(tensor.numpy()).cast('B'))
        return digest.hexdigest()

    labels = torch.tensor([row['label'] for row in records], dtype=torch.int64)
    require(tensor_hash([labels]) == gradient_meta['training_labels_sha256'], 'Full-gradient labels mismatch')
    artifact = torch.load(directory / 'gradient_direction.pt', map_location='cpu', weights_only=True)
    gradient, direction = artifact['gradient_mean_fp64'], artifact['unit_positive_gradient_fp32']
    require(artifact['coordinate_order'] == manifest['coordinate_order'], 'Gradient tensor coordinate order changed')
    require(len(gradient) == len(direction) == 24, 'Effective gradient does not contain 24 Q/V tensors')
    for tensors, dtype in ((gradient, torch.float64), (direction, torch.float32)):
        require(all(tuple(value.shape) == (768, 768) and value.dtype == dtype and
                    value.device.type == 'cpu' and bool(torch.isfinite(value).all()) for value in tensors),
                'Saved gradient/direction tensor shape/dtype/finiteness invalid')
    require(tensor_hash(gradient) == gradient_meta['gradient_sha256'] and
            tensor_hash(direction) == gradient_meta['direction_sha256'], 'Saved gradient/direction tensor hash mismatch')
    norm = math.sqrt(sum(float(value.square().sum()) for value in gradient))
    direction_norm = math.sqrt(sum(float(value.double().square().sum()) for value in direction))
    dot = sum(float((g * u.double()).sum()) for g, u in zip(gradient, direction))
    require(norm > 0 and dot > 0, 'Saved direction is not an ascent direction')
    require(all(torch.equal(u, (g / norm).to(torch.float32)) for g, u in zip(gradient, direction)),
            'Saved ray differs from the positive normalized mean gradient')
    close(direction_norm, 1, atol=5e-7, rtol=0, label='Global unit direction norm')
    close(dot, norm, atol=1e-9, rtol=2e-6, label='Positive gradient directional derivative')
    for key, value in (('raw_grad_norm', norm), ('direction_norm', direction_norm), ('gradient_dot_direction', dot)):
        close(gradient_meta[key], value, atol=1e-10, rtol=1e-10, label=f'Recomputed {key}')
    require(gradient_meta['coordinate_numel'] == sum(t.numel() for t in gradient) == 14155776,
            'Gradient coordinate dimension changed')
    record = manifest['direction_record']
    require(record['direction_sign'] == 1 and record['task'] == 0 and
            record['sha256'] == gradient_meta['direction_sha256'] and
            record['gradient_sha256'] == gradient_meta['gradient_sha256'], 'Direction record hash/orientation mismatch')
    require(all(row['point']['direction'] == record for row in rows if row['point']['radius']),
            'Raw scan direction does not match saved unit gradient')
    origin = next(row for row in rows if row['point']['radius'] == 0)
    close(gradient_meta['objective_at_anchor'], origin['metrics']['train']['loss'],
          atol=3e-6, rtol=1e-7, label='Gradient and scan share full-training mean CE')
    require(witness['gradient_task'] == 0 and witness['direction_sign'] == 1 and witness['samples'] == 1259 and
            witness['direction_sha256'] == gradient_meta['direction_sha256'] and
            witness['gradient_manifest_sha256'] == sha(directory / 'gradient_manifest.json'),
            'Finite-difference witness identity changed')
    close(witness['origin_loss'], origin['metrics']['train']['loss'], atol=3e-7, rtol=1e-6, label='Witness origin CE')
    require([step['h'] for step in witness['steps']] == [1e-4, 5e-4, 1e-3], 'Witness radius grid changed')
    relative_errors = []
    for step in witness['steps']:
        central = (step['loss_plus'] - step['loss_minus']) / (2 * step['h'])
        forward = (step['loss_plus'] - witness['origin_loss']) / step['h']
        relative = abs(central - norm) / norm
        for key, value in (('central_derivative', central), ('forward_derivative', forward),
                           ('relative_derivative_error', relative)):
            close(step[key], value, label=f'Witness {key}')
        require(step['anchor_restored_exactly'], 'Witness restoration failed')
        relative_errors.append(relative)
    first = witness['steps'][0]
    require(first['loss_plus'] > witness['origin_loss'] + max(1e-8, .1 * first['h'] * norm),
            'Small positive step does not increase Task0 CE')
    require(min(relative_errors) <= .05 and witness['origin_restored_after_witness'],
            'Finite-difference derivative or final witness restoration failed')
    at001 = next(row for row in rows if row['point']['radius'] == .001)
    close(witness['steps'][-1]['loss_plus'], at001['metrics']['train']['loss'],
          atol=3e-7, rtol=1e-6, label='Witness and measured scan at r=.001')
    return dict(passed=True, cpu_saved_tensor_hashes_verified=True, full_training_membership_verified=True,
                positive_normalized_mean_gradient_verified=True, gradient_task=0, direction_sign=1,
                raw_grad_norm=norm, direction_norm=direction_norm, gradient_dot_direction=dot,
                small_step_ascent_verified=True, minimum_relative_derivative_error=min(relative_errors),
                formal_data_source=file_receipt(formal_data_path))


def validate_job(seed, method):
    directory = directory_for(seed, method)
    receipt = check_completed_hashes(directory)
    summary, manifest = read(directory / 'summary.json'), read(directory / 'manifest.json')
    rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
    require(receipt['schema'] == 'rq2_fixed_task0_gradient_v1' and receipt['seed'] == seed and
            receipt['method'] == method and receipt['gradient_task'] == 0 and receipt['direction_sign'] == 1,
            'Worker schema/seed/method/positive-Task0 identity mismatch')
    require(receipt['protocol_sha256'] == manifest['protocol_sha256'], 'Worker protocol mismatch')
    require(receipt['wrapper_sha256'] == sha(ROOT / 'diagnostic_fixed_gradient.py'), 'Worker source changed')
    require(receipt['preflight']['status'] == 'passed', 'Worker preflight failed')
    for field in ('result_check', 'identity_check', 'origin_check', 'witness_check'):
        require(receipt[field]['passed'] is True, f'Worker {field} failed')
    require(canonical({k: v for k, v in manifest.items() if k != 'protocol_sha256'}) ==
            manifest['protocol_sha256'], 'Manifest canonical hash mismatch')
    formal, formal_origin, sources = read_formal(seed, method)
    for key in COMMON_IDENTITY:
        require(manifest[key] == formal[key], f'Formal anchor identity changed: {key}')
    require(manifest['checkpoint_sha256'] == formal['checkpoint_sha256'], 'Task0 checkpoint identity changed')
    config_path, formal_config_path = ROOT / f'configs/seed{seed}.json', BASE / f'configs/seed{seed}.json'
    config, old_config = read(config_path), read(formal_config_path)
    require(config == manifest['config'] and sha(config_path) == manifest['config_sha256'], 'Actual config does not match manifest')
    require(old_config == formal['config'] and sha(formal_config_path) == formal['config_sha256'], 'Formal config changed')
    differences = {key for key in set(config) | set(old_config) if config.get(key) != old_config.get(key)}
    require(differences == {'output_dir'}, f'Unexpected config changes: {sorted(differences)}')
    require(Path(config['output_dir']).resolve() == (ROOT / f'results/seed{seed}').resolve(), 'Unexpected result directory')
    require(receipt['preflight']['config']['sha256'] == sha(config_path), 'Preflight config changed')
    require(summary['status'] == 'complete' and summary['seed'] == seed and summary['method'] == method and
            summary['job'] == 'fixed_gradient' and summary['protocol_sha256'] == manifest['protocol_sha256'],
            'Summary identity/status mismatch')
    require(len(rows) == len({row['state_id'] for row in rows}) == len(RADII) == 93, 'Missing/duplicate fixed-direction states')
    require(summary['states_completed'] == len(RADII), 'Summary count mismatch')
    require(summary['expected_states'] == 93 and summary['directions_per_radius'] == 1 and
            summary['gradient_scope'] == ['subset'] and summary['gradient_task'] == 0 and
            summary['radii'] == manifest['radii'] == RADII, 'Summary grid/objective/scope changed')
    require(summary['frozen_state_sha256_before'] == summary['frozen_state_sha256_after'] ==
            manifest['frozen_state_sha256'] and summary['fixed_state_restored'], 'Fixed backbone/head changed')
    require(summary['direction_sha256'] == summary['direction_sha256_after'] ==
            manifest['fixed_gradient']['direction_sha256'], 'Direction tensors mutated during scan')
    require([entry['radius'] for entry in summary['radius_summary']] == RADII, 'Summary radius grid mismatch')
    finite(summary); finite(rows)
    by_radius = {row['point']['radius']: row for row in rows}
    require(sorted(by_radius) == RADII, 'Raw radius grid mismatch')
    origin = by_radius[0]
    direction_identity = None
    for radius, row in sorted(by_radius.items()):
        require(row['seed'] == seed and row['method'] == method and row['protocol_sha256'] == manifest['protocol_sha256'],
                'Raw state identity mismatch')
        require(row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method], 'Raw Task0 checkpoint mismatch')
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == manifest[key], f'Raw {key} mismatch')
        require(row['anchor_restored_exactly'] and row['amount'] == radius, 'Restoration or radius/amount mismatch')
        require(row['state_id'] == ('r0' if radius == 0 else f'r{radius:.12g}_gradient'), 'Unexpected state ID')
        close(row['effective_weight_drift'], radius, rtol=2e-6, atol=1e-5, label='Realized unit-direction radius')
        close(row['effective_weight_drift_squared'], row['effective_weight_drift'] ** 2, label='Drift squared')
        if radius:
            direction = row['point']['direction']
            require(direction['kind'] == 'positive_full_task0_training_ce_gradient' and direction['task'] == 0 and
                    direction['held_fixed_across_radii'] is True, 'Direction is not fixed positive Task0 gradient')
            if direction_identity is None:
                direction_identity = direction
            require(direction == direction_identity, 'Direction changed along radius scan')
        for split, count in (('train', 1259), ('test', 307)):
            metric = row['metrics'][split]
            require(metric['samples'] == count and metric['loss'] >= 0 and 0 <= metric['accuracy'] <= 100,
                    'Invalid CE/accuracy/sample count')
        require(set(row['fisher_by_scope']) == {'subset'}, 'Fisher scope changed')
        fisher = row['fisher_by_scope']['subset']
        require(fisher['samples'] == 256 and fisher['normalized'] is False and fisher['floor'] is None and
                fisher['regularizer'] is None and fisher['raw_trace'] >= 0 and fisher['q'] == {}, 'Fisher definition changed')
        for key in ('estimator', 'coordinate', 'gradient_batch_size'):
            require(fisher[key] == formal_origin['fisher_by_scope']['subset'][key], f'Fisher {key} changed')
    for split in ('train', 'test'):
        close(origin['metrics'][split]['loss'], formal_origin['metrics'][split]['loss'],
              rtol=1e-6, atol=3e-7, label='Repeated formal origin CE')
        close(origin['metrics'][split]['accuracy'], formal_origin['metrics'][split]['accuracy'], label='Origin accuracy')
    close(origin['fisher_by_scope']['subset']['raw_trace'], formal_origin['fisher_by_scope']['subset']['raw_trace'],
          rtol=1e-5, atol=3e-7, label='Repeated formal origin raw Fisher')
    ending = summary['end_restoration']
    require(ending['predictions_match'] and ending['max_logit_error'] == 0 and
            ending['effective_sha256'] == manifest['anchor_effective_sha256'], 'Final restoration failed')
    for metric in ('loss', 'accuracy'):
        close(ending['test'][metric], origin['metrics']['test'][metric], label='Restored origin metric')
    gradient_check = validate_gradient(directory, manifest, rows, seed, method)
    output = []
    for entry in summary['radius_summary']:
        radius = entry['radius']; row = by_radius[radius]
        require(entry['directions'] == 1, 'Fixed direction was counted as multiple replicates')
        record = dict(seed=seed, method=method, radius=radius, directions=1, gradient_task=0,
                      train_samples=1259, test_samples=307, fisher_samples=256)
        for split in ('train', 'test'):
            loss, accuracy = row['metrics'][split]['loss'], row['metrics'][split]['accuracy']
            damage = loss - origin['metrics'][split]['loss']
            positive = max(damage, 0)
            expect = dict(D_mean=damage, G=positive, Q95=positive, sampled_max=positive,
                          absolute_loss_mean=loss, accuracy_mean=accuracy,
                          accuracy_drop_mean_pp=origin['metrics'][split]['accuracy'] - accuracy)
            for key, value in expect.items():
                close(entry[split][key], value, label=f'Signed/positive raw recomputation {split}/{key}')
            require(len(entry[split]['D_each']) == 1, 'One-direction damage cardinality mismatch')
            close(entry[split]['D_each'][0], damage, label='D_each')
            record.update({f'D_{split}': damage, f'G_positive_{split}': positive,
                           f'{split}_ce': loss, f'{split}_accuracy_percent': accuracy})
        trace = row['fisher_by_scope']['subset']['raw_trace']
        saved = entry['raw_trace_by_scope']['subset']
        require(saved['samples'] == 256 and len(saved['each']) == 1, 'Fisher sample/replicate mismatch')
        for key in ('mean', 'Q95', 'sampled_max'):
            close(saved[key], trace, label=f'Raw Fisher {key}')
        close(saved['each'][0], trace, label='Raw Fisher single direction')
        record['raw_fisher_trace'] = trace
        output.append(record)
    # Local orientation is checked by the separate tiny finite-difference probe.
    # No positivity or monotonicity condition is imposed on the plotted radius grid.
    sources.extend(file_receipt(path) for path in [directory / name for name in REQUIRED_FILES]
                   + [config_path, formal_config_path, ROOT / 'diagnostic_fixed_gradient.py'])
    sources.append(gradient_check['formal_data_source'])
    check = dict(seed=seed, method=method, status='validated', states=len(rows), gradient_task=0,
                 raw_statistics_recomputed=True, fixed_direction_identity_verified=True,
                 formal_origin_identity_verified=True, finite_values=True,
                 positive_small_step_witness_verified=True, gradient_tensor_check=gradient_check,
                 smallest_plotted_radius_D_train=next(item['D_train'] for item in output if item['radius'] == .001))
    return output, sources, check


def load_plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rq2-fixed-gradient-publisher-mpl'))
    sys.path.insert(0, str(STYLE))
    import plot_common as pc
    return pc


def render(pc, data, sources, checks, *, zoom=False):
    from matplotlib.ticker import FuncFormatter
    maximum = 2 if zoom else 50
    tag = 'near_origin' if zoom else 'full'
    out = ROOT / f'figures/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    pc.HERE = out
    font = pc.setup_style()
    frame = data[data.radius <= maximum].copy()
    fig, axes = pc.plt.subplots(1, 3, figsize=(7.6, 2.15))
    fig.subplots_adjust(left=.083, right=.99, bottom=.27, top=.965, wspace=.48)
    specs = [('D_train', r'Signed CE change $D_0(r)$', 'upper left'),
             ('train_accuracy_percent', 'Accuracy (%)', 'lower left'),
             ('raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', 'upper left')]
    grid = sorted(frame.radius.unique())
    marker_indices = list(range(len(grid))) if zoom else [i for i, r in enumerate(grid) if r == 0 or r >= 25 or float(r).is_integer()]
    aggregates, ranges = [], []
    for ax, (value, label, legend) in zip(axes, specs):
        agg = pc.draw_mean_sd(ax, frame, 'radius', value, marker_every=marker_indices)
        ranges.append(pc.limits_from_sd(ax, agg, include_zero=value == 'D_train'))
        # Tick labels use inline scientific notation when needed; the plotted raw
        # values and SD are never rescaled, normalized, clipped, or smoothed.
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda tick, position: f'{tick:.2g}' if abs(tick) >= 1e4 or 0 < abs(tick) < 1e-3 else f'{tick:g}'))
        ax.yaxis.get_offset_text().set_visible(False)
        if value == 'D_train':
            ax.axhline(0, color='#777777', linewidth=.5, zorder=0)
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.025 if zoom else -.5, maximum + (.025 if zoom else .5)),
               xticks=[0, 1, 2] if zoom else [0, 25, 50])
        pc.internal_legend(ax, legend)
        agg['metric'] = value; aggregates.append(agg)
    pc.pd.concat(aggregates, ignore_index=True).to_csv(out / 'mean_sample_sd.csv', index=False)
    frame.to_csv(out / 'per_seed.csv', index=False)
    name = f'Figure_B1_fixed_task0_gradient_{tag}'
    metadata = dict(font=font, source_files=sources, expected_jobs=6, validated_jobs=6,
        gradient_task=0, direction_sign=1, job_checks=checks, signed_negative_damage_retained=True,
        radius_window=[0, maximum], axis_ranges=ranges, measured_radii=grid,
        direction_count_per_seed_method=1, gradient_samples=1259, fisher_samples=256,
        same_direction_across_methods=False, direction_is_held_fixed_across_radii=True,
        gradient_recomputed_after_each_step=False, actual_lora_optimizer_trajectory=False,
        new_model_evaluations_by_publisher=False, extra_fisher_normalization_or_lambda=False)
    result = pc.save_figure(fig, name, metadata)
    caption = (
        'Fixed Task-0 positive-gradient response on the formal ImageNet-A seeds 0, 1, and 2 '
        '(ViT-B/16, rank-10 Q/V LoRA). At each method/seed own Task-0 checkpoint, compute once '
        'the gradient g of mean true-label cross-entropy over all 1,259 Task-0 training examples '
        'with respect to the 24 effective Q/V weight matrices, then set u=+g/||g||. '
        'Evaluate theta(r)=theta(0)+r*u with this single direction held fixed for the whole radius scan; '
        'the old 20-class head and the remaining model state are fixed. Each method uses its own '
        'gradient direction; directions are not shared between EWCLoRA and FS-EWCLoRA. '
        'Left: signed Task-0 training CE difference D_0(r)=L_0(theta(r))-L_0(theta(0)), retaining '
        'any negative values without clipping. Middle: Task-0 training accuracy (%). Right: raw empirical Fisher '
        'trace on the unchanged fixed 256-example training subset, computed from per-example '
        'true-label CE gradients in the same effective Q/V coordinates. Fisher has no extra '
        'normalization, floor, regularizer, or multiplication by the EWC coefficient. '
        'The full scan contains 93 measured radii: 0–20 in steps of 0.25, '
        '25, 30, 35, 40, 45, 50, and additional near-origin radii 0.001, 0.005, 0.01, 0.025, 0.05, 0.1. '
        + ('This companion shows only the measured 0–2 window to reveal the initial response. '
           if zoom else 'This main figure shows the full measured 0–50 range; the separate 0–2 companion reveals the initial response. ')
        + 'Lines and shading are the arithmetic mean and one sample standard deviation (ddof=1) '
        'across the three training seeds. Each seed/method supplies one direction; directions '
        'are not additional independent replicates. Connecting segments are visual guides, with '
        'no interpolation treated as a measurement. The CSV also exports max(D_0(r),0), alongside '
        'signed training/test CE changes and accuracy, but the displayed loss panel uses signed D_0. '
        'Gradient unit norm, saved direction identity, full Task-0 sample membership, formal origin '
        'recovery, and an initial finite-difference ascent witness are checked before publishing. '
        'This is a fixed direction in effective weights, not an actual LoRA optimizer update path, '
        'iterative gradient descent, a Task-1 update, or an experiment measuring future-task forgetting.'
    )
    (out / 'CAPTION.md').write_text(caption + '\n')
    tex = caption.replace('_', '\\_').replace('%', '\\%')
    (out / 'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'
        + name + '.pdf}\n\\caption{' + tex + '}\n\\label{fig:fixed-task0-gradient-' + tag.replace('_', '-')
        + '}\n\\end{figure*}\n')
    return result


def publish():
    # Recheck sources just before rendering, then leave all old figures untouched.
    old_figures = STYLE / 'B1_merged_latest_20260921'
    old_snapshot = {str(path): sha(path) for path in old_figures.iterdir() if path.is_file()}
    rows, sources, checks = [], [], []
    for seed in range(3):
        for method in METHODS:
            measured, provenance, checked = validate_job(seed, method)
            rows.extend(measured); sources.extend(provenance); checks.append(checked)
    sources.extend(file_receipt(path) for path in (Path(__file__), STYLE / 'plot_common.py'))
    sources = list({item['path']: item for item in sources}.values())
    for item in sources:
        require(sha(item['path']) == item['sha256'], f'Source changed during validation: {item["path"]}')
    pc = load_plotting()
    data = pc.pd.DataFrame(rows).sort_values(['method', 'seed', 'radius'])
    require(len(data) == 558 and not data.duplicated(['method', 'seed', 'radius']).any(), 'Merged coverage is not 6×93')
    for _, group in data.groupby(['method', 'seed']):
        require(group.radius.tolist() == RADII, 'Merged grid incomplete')
    figures = [render(pc, data, sources, checks, zoom=zoom) for zoom in (False, True)]
    out = ROOT / 'figures'
    data.to_csv(out / 'all_per_seed.csv', index=False)
    aggregates = []
    for metric in ('D_train', 'G_positive_train', 'D_test', 'G_positive_test', 'train_ce', 'test_ce',
                   'train_accuracy_percent', 'test_accuracy_percent', 'raw_fisher_trace'):
        frame = pc.mean_sd(data, 'radius', metric); frame['metric'] = metric; aggregates.append(frame)
    pc.pd.concat(aggregates, ignore_index=True).to_csv(out / 'all_mean_sample_sd.csv', index=False)
    for path, digest in old_snapshot.items():
        require(sha(path) == digest, f'Previous figure changed: {path}')
    complete = dict(status='complete', expected_jobs=6, validated_jobs=6, states=558, jobs=checks,
        gradient_task=0, direction_sign=1, automatic_numeric_and_layout_checks='passed', source_files=sources,
        output_files=[file_receipt(path) for path in sorted(out.rglob('*')) if path.is_file()],
        figures=[figure['outputs'] for figure in figures], previous_figures_preserved=True,
        rendered_visual_review='pending actual rendered-image inspection',
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(ROOT / 'report/fixed_gradient_complete.json', complete)
    atomic_json(ROOT / 'publisher_receipt.json', complete)
    return dict(status='complete', expected_jobs=6, validated_jobs=6,
                receipt=str(ROOT / 'report/fixed_gradient_complete.json'), figures=complete['figures'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Read-only inventory and completed file hash checks (default)')
    mode.add_argument('--execute', action='store_true', help='Require six valid complete jobs and publish both figures')
    args = parser.parse_args(argv)
    try:
        result = inventory()
        if args.execute:
            if result['complete_job_files'] != 6:
                print(json.dumps(result, indent=2)); return 2
            result = publish()
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (AssertionError, ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps(dict(status='invalid', error=f'{type(exc).__name__}: {exc}',
                             complete_receipt_written=False), indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
