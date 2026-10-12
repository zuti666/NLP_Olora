#!/usr/bin/env python3
"""CPU-only B1 seed-0 integer-radius publisher (50 dense / 16 Fisher directions): default inventory is read-only; never opens a GPU.

--check-only: report extension coverage and verify any completed receipt hashes.
--execute: validate all 4 new jobs and the completed prior measurements, then publish.
New radii fill every missing integer in 25–50; no interpolation or invented points.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
COARSE = BASE / 'radius50_extension_20260921'
STYLE = BASE / 'figures_formal_20260921/B1_merged_latest_20260921'
METHODS = ('sgd', 'gam')
SEEDS = (0,)
EXPECTED_JOBS = 4
JOBS = ('dense', 'sparse_train')
COARSE_RADII = [25, 30, 35, 40, 45, 50]
RADII = [0] + [r for r in range(25, 51) if r not in COARSE_RADII]
EPSILONS = [.005, .01, .02, .04]
FISHER_RADII = [0, .5, 1, 2, 5, 10] + list(range(11, 21))


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


def close(actual, expected, *, rtol=1e-10, atol=1e-12, label='value'):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f'{label}: nonfinite')
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
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


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_receipt_files(directory):
    wrapper = read(directory / 'extension_receipt.json')
    require(wrapper['status'] == 'complete', f'Incomplete receipt: {directory}')
    for name in ('summary', 'manifest', 'states'):
        path = directory / (name + ('.jsonl' if name == 'states' else '.json'))
        require(wrapper[name + '_sha256'] == sha(path), f'Receipt {name} SHA mismatch: {directory}')
    return wrapper


def inventory():
    jobs = []
    for seed in SEEDS:
        for method in METHODS:
            for job in JOBS:
                directory = ROOT / f'results/seed{seed}/{method}/{job}'
                files = [directory / name for name in
                         ('summary.json', 'manifest.json', 'states.jsonl', 'extension_receipt.json')]
                files.append(ROOT / f'configs/{job}_seed{seed}.json')
                missing = [str(p) for p in files if not p.is_file()]
                status = 'missing' if missing else 'incomplete'
                error = None
                if not missing:
                    try:
                        wrapper = check_receipt_files(directory)
                        require(read(files[0])['status'] == 'complete', 'Summary incomplete')
                        status = 'complete'
                    except (ValueError, KeyError, OSError, TypeError) as exc:
                        status, error = 'invalid', str(exc)
                jobs.append(dict(seed=seed, method=method, job=job, status=status,
                                 missing=missing, error=error))
    count = sum(j['status'] == 'complete' for j in jobs)
    return dict(status='ready_to_validate' if count == EXPECTED_JOBS else 'pending', expected_jobs=EXPECTED_JOBS,
                complete_job_files=count, jobs=jobs, extension_radii=RADII,
                final_receipt=str(ROOT / 'report/radius50_complete.json'),
                check_only_is_read_only=True)


def load_prior():
    """Recompute original and coarse r<=50 raw data after verifying its receipts."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rq2-radius50-integer-publisher-mpl'))
    sys.path.insert(0, str(BASE / 'figures_formal_20260921'))
    path = COARSE / 'publish_radius50.py'
    spec = importlib.util.spec_from_file_location('rq2_coarse_radius50_publisher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt_path = COARSE / 'report/radius50_complete.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'complete' and receipt['validated_jobs'] == receipt['expected_jobs'] == 12,
            'Coarse radius-50 publication is incomplete')
    require(receipt['automatic_numeric_and_layout_checks'] == 'passed', 'Coarse publication validation failed')
    for item in receipt['source_files'] + receipt['output_files']:
        require(sha(item['path']) == item['sha256'], f'Coarse source/output changed: {item["path"]}')
    pc, dense, fisher, metadata = module.load_prior()
    reused_rows, reused_checks = [], []
    for seed in range(3):
        for method in METHODS:
            for job in JOBS:
                rows, sources, check = module.validate_extension(seed, method, job, pc)
                reused_rows.extend(rows)
                reused_checks.append(check)
                metadata['source_files'].extend(sources)
    coarse = pc.pd.DataFrame(reused_rows)
    dense = pc.pd.concat([dense, coarse[(coarse.job == 'dense') & (coarse.radius > 0)]], ignore_index=True)
    fisher = pc.pd.concat([fisher, coarse[(coarse.job == 'sparse_train') & (coarse.radius > 0)]], ignore_index=True)
    for frame, grid in ((dense, [i / 4 for i in range(81)] + COARSE_RADII),
                        (fisher, FISHER_RADII + COARSE_RADII)):
        require(not frame.duplicated(['method', 'seed', 'radius']).any(), 'Duplicate reused point')
        require(sorted(set(zip(frame.method, frame.seed))) ==
                sorted((method, seed) for method in METHODS for seed in range(3)), 'Reused cohort incomplete')
        for _, group in frame.groupby(['method', 'seed']):
            require(sorted(group.radius.tolist()) == grid, 'Incomplete reused grid')
    metadata['source_files'].extend(file_receipt(p) for p in (receipt_path, path))
    metadata['original_raw_states_recomputed'] += sum(item['states'] for item in reused_checks)
    metadata['reused_coarse_jobs'] = reused_checks
    metadata['reused_coarse_receipt'] = file_receipt(receipt_path)
    dense = recompute_prior_dense_prefix(pc)
    fisher = fisher[fisher.seed == 0].copy()
    metadata['source_cohort'] = 'seed0_dense50_fisher16'
    return pc, dense, fisher, metadata


def recompute_prior_dense_prefix(pc):
    """Use the first 50 persisted directions at EVERY old radius, never old 100-dir means."""
    output = []
    for method in METHODS:
        library = {}
        origin_reference = None
        for partition, directory in (
            ('original_formal_seed0_first50', BASE / f'formal_results/seed0/{method}/dense'),
            ('coarse_radius50_seed0_first50', COARSE / f'results/seed0/{method}/dense')):
            rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
            origins = [row for row in rows if row['point']['radius'] == 0]
            require(len(origins) == 1, 'Reused dense origin missing/duplicated')
            origin = origins[0]
            if origin_reference is None:
                origin_reference = origin
            else:
                close(origin['metrics']['train']['loss'], origin_reference['metrics']['train']['loss'],
                      rtol=1e-6, atol=3e-7, label='Reused half-direction origin')
            grouped = defaultdict(list)
            for row in rows:
                radius = row['point']['radius']
                if radius == 0:
                    if partition.startswith('original'):
                        grouped[radius].append(row)
                    continue
                direction = row['point']['direction']
                index = direction['index']
                if index >= 50:
                    continue
                require(index not in library or library[index] == direction,
                        'Reused first-50 direction identity changed')
                library[index] = direction
                grouped[radius].append(row)
            for radius, points in sorted(grouped.items()):
                require(len(points) == (1 if radius == 0 else 50), 'Incomplete reused first-50 coverage')
                if radius:
                    require({r['point']['direction']['index'] for r in points} == set(range(50)),
                            'Reused dense direction prefix mismatch')
                output.append(dict(seed=0, method=method, job='dense', radius=radius,
                    directions=len(points), train_loss_samples=1259,
                    G_train=math.fsum(max(r['metrics']['train']['loss'] - origin['metrics']['train']['loss'], 0)
                                      for r in points) / len(points),
                    train_accuracy_percent=math.fsum(r['metrics']['train']['accuracy'] for r in points) / len(points),
                    source_partition=partition))
        require(set(library) == set(range(50)), 'Incomplete reused dense library')
    frame = pc.pd.DataFrame(output)
    for _, group in frame.groupby(['method', 'seed']):
        require(sorted(group.radius.tolist()) == [i / 4 for i in range(81)] + COARSE_RADII,
                'First-50 reused dense grid incomplete')
    return frame


def validate_extension(seed, method, job, pc):
    directory = ROOT / f'results/seed{seed}/{method}/{job}'
    summary_path, manifest_path, states_path = [directory / name for name in
                                               ('summary.json', 'manifest.json', 'states.jsonl')]
    config_path = ROOT / f'configs/{job}_seed{seed}.json'
    summary, manifest, config = read(summary_path), read(manifest_path), read(config_path)
    wrapper = check_receipt_files(directory)
    count = 50 if job == 'dense' else 16
    expected = 1 + (len(RADII) - 1) * count
    require(wrapper['schema'] == 'rq2_radius50_integer_seed0_halfdirs_v1', 'Unexpected worker receipt schema')
    require(wrapper['seed'] == seed and wrapper['method'] == method and wrapper['job'] == job,
            'Extension receipt belongs to another job')
    require(Path(wrapper['summary']).resolve() == summary_path.resolve(), 'Extension summary path mismatch')
    require(wrapper['protocol_sha256'] == manifest['protocol_sha256'], 'Extension protocol mismatch')
    require(wrapper['wrapper_sha256'] == sha(ROOT / 'diagnostic_radius50.py'), 'Worker changed after receipt')
    require(wrapper['preflight']['status'] == 'passed', 'Worker preflight failed')
    for key in ('identity_check', 'origin_check', 'endpoint_check'):
        require(wrapper[key]['passed'] is True, f'Worker {key} failed')
    for key in ('all_statistics_recomputed', 'direction_identities_match_formal', 'raw_states_finite'):
        require(wrapper['result_check'][key] is True, f'Worker result check {key} failed')
    require(wrapper['preflight']['config']['sha256'] == sha(config_path), 'Preflight config hash mismatch')
    require(wrapper['endpoint_check']['measured_states'] == expected, 'Worker state coverage mismatch')
    require(wrapper['endpoint_check']['radii'] == RADII, 'Worker radii mismatch')
    old_summary, old_manifest, old_sources = pc.audited_summary(seed, method, job)
    old_config_path = BASE / f'configs/seed{seed}.json'
    old_config = read(old_config_path)
    require(old_config == old_manifest['config'] and sha(old_config_path) == old_manifest['config_sha256'],
            'Original config differs from formal manifest')
    require(config == manifest['config'] and sha(config_path) == manifest['config_sha256'],
            'Extension config differs from manifest')
    grid_key = 'dense_radii' if job == 'dense' else 'sparse_train_radii'
    changed = {key for key in set(config) | set(old_config) if config.get(key) != old_config.get(key)}
    expected_changed = {'output_dir', grid_key} | ({'dense_directions'} if job == 'dense' else set())
    require(changed == expected_changed, f'Unexpected configuration changes: {sorted(changed)}')
    require(seed == 0 and config['training_seed'] == 0, 'Only seed 0 is authorized')
    require(config['dense_directions'] == (50 if job == 'dense' else 100), 'Dense direction count changed')
    require(config['sparse_directions'] == 16, 'Fisher must keep 16 directions')
    require(config[grid_key] == RADII, 'Unexpected extension grid')
    require(Path(config['output_dir']).resolve() == (ROOT / f'results/seed{seed}').resolve(),
            'Extension output directory mismatch')
    require(canonical({k: v for k, v in manifest.items() if k != 'protocol_sha256'}) ==
            manifest['protocol_sha256'], 'Canonical manifest hash mismatch')
    mutable = {'config', 'config_sha256', 'protocol_sha256'}
    mismatches = {key for key in set(manifest) | set(old_manifest)
                  if key not in mutable and manifest.get(key) != old_manifest.get(key)}
    require(not mismatches, f'Anchor/code/data/head/coordinate identity mismatch: {sorted(mismatches)}')
    require(summary['status'] == 'complete' and summary['seed'] == seed and summary['method'] == method and
            summary['job'] == job and summary['protocol_sha256'] == manifest['protocol_sha256'],
            'Summary identity/status mismatch')
    require(summary['direction_indices'] == list(range(count)) and summary['paired_public_direction_library'],
            'Public directions changed')
    require(summary['gradient_scope'] == ([] if job == 'dense' else ['subset']) and
            summary['forward_only'] == (job == 'dense'), 'Gradient scope changed')
    require([entry['radius'] for entry in summary['radius_summary']] == RADII, 'Summary radius grid mismatch')
    restored = summary['end_restoration']
    require(restored['predictions_match'] and restored['max_logit_error'] == 0 and
            restored['effective_sha256'] == manifest['anchor_effective_sha256'], 'Final restoration failed')
    old_states_path = BASE / f'formal_results/seed{seed}/{method}/{job}/states.jsonl'
    old_rows = [json.loads(line) for line in old_states_path.read_text().splitlines()]
    old_origin = next(row for row in old_rows if row['point']['radius'] == 0)
    library = {}
    for row in old_rows:
        if row['point']['radius']:
            direction = row['point']['direction']
            index = direction['index']
            require(index not in library or library[index] == direction, 'Original direction library inconsistent')
            library[index] = direction
    require(set(library) == set(range(100 if job == 'dense' else 16)), 'Original direction library incomplete')
    rows = [json.loads(line) for line in states_path.read_text().splitlines()]
    finite(summary); finite(rows)
    require(len(rows) == len({row['state_id'] for row in rows}) == expected, 'Duplicate/missing raw states')
    require(summary['expected_states'] == summary['states_completed'] == expected, 'Summary states mismatch')
    origins = [row for row in rows if row['point']['radius'] == 0]
    require(len(origins) == 1, 'Expected exactly one repeat origin')
    origin = origins[0]
    grouped = defaultdict(list)
    for row in rows:
        require(row['seed'] == seed and row['method'] == method and
                row['protocol_sha256'] == manifest['protocol_sha256'], 'Raw identity mismatch')
        require(row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method], 'Raw checkpoint mismatch')
        for key in ('data_manifest_sha256', 'coordinate_sha256'):
            require(row[key] == manifest[key], f'Raw {key} mismatch')
        require(row['anchor_restored_exactly'], 'Point restoration failed')
        radius = row['point']['radius']
        require(radius in RADII, 'Unexpected raw radius')
        close(row['effective_weight_drift'], radius, rtol=2e-6, atol=1e-9, label='realized drift')
        close(row['effective_weight_drift_squared'], radius ** 2, rtol=4e-6, atol=1e-9, label='drift squared')
        if radius:
            direction = row['point']['direction']
            require(direction == library.get(direction['index']), 'Direction hash/index/seed/norm changed')
        for split, samples in (('train', 1259), ('test', 307)):
            require(row['metrics'][split]['samples'] == samples, 'Metric sample count changed')
        require(set(row['fisher_by_scope']) == (set() if job == 'dense' else {'subset'}), 'Fisher scope changed')
        if job == 'sparse_train':
            fisher = row['fisher_by_scope']['subset']
            require(fisher['samples'] == 256 and fisher['normalized'] is False and
                    fisher['floor'] is None and fisher['regularizer'] is None and fisher['raw_trace'] >= 0,
                    'Raw Fisher definition changed')
        grouped[radius].append(row)
    for split in ('train', 'test'):
        close(origin['metrics'][split]['loss'], old_origin['metrics'][split]['loss'],
              rtol=1e-6, atol=3e-7, label='repeat origin CE')
        close(origin['metrics'][split]['accuracy'], old_origin['metrics'][split]['accuracy'], label='repeat origin accuracy')
    if job == 'sparse_train':
        for metric in ('raw_trace', 'subset_loss'):
            close(origin['fisher_by_scope']['subset'][metric], old_origin['fisher_by_scope']['subset'][metric],
                  rtol=1e-5, atol=3e-7, label='repeat origin Fisher')
    output = []
    for entry in summary['radius_summary']:
        radius = entry['radius']; points = grouped[radius]
        require(len(points) == (1 if radius == 0 else count), 'Missing radius/direction pairs')
        if radius:
            require({row['point']['direction']['index'] for row in points} == set(range(count)), 'Direction coverage mismatch')
        damage = math.fsum(max(row['metrics']['train']['loss'] - origin['metrics']['train']['loss'], 0)
                           for row in points) / len(points)
        accuracy = math.fsum(row['metrics']['train']['accuracy'] for row in points) / len(points)
        close(damage, entry['train']['G'], label='Recomputed G')
        close(accuracy, entry['train']['accuracy_mean'], label='Recomputed accuracy')
        item = dict(seed=seed, method=method, job=job, radius=radius, directions=len(points),
                    train_loss_samples=1259, G_train=damage, train_accuracy_percent=accuracy,
                    source_partition='seed0_halfdirs_repeat_origin' if radius == 0 else 'seed0_halfdirs_integer_extension')
        if job == 'sparse_train':
            trace = math.fsum(row['fisher_by_scope']['subset']['raw_trace'] for row in points) / len(points)
            close(trace, entry['raw_trace_by_scope']['subset']['mean'], label='Recomputed Fisher')
            item.update(raw_fisher_trace=trace, fisher_samples=256)
        output.append(item)
    files = old_sources + [file_receipt(path) for path in (summary_path, manifest_path, states_path,
              config_path, old_config_path, old_states_path, directory / 'extension_receipt.json',
              ROOT / 'diagnostic_radius50.py')]
    checked = dict(seed=seed, method=method, job=job, status='validated', states=expected,
                   protocol_sha256=manifest['protocol_sha256'], anchor_identity_matches=True,
                   code_data_head_coordinate_identity_matches=True, direction_library_matches=True,
                   origin_repeat_matches=True, metrics_recomputed_from_raw=True)
    return output, files, checked


def tolerance_tables(pc, dense):
    """Return measured-grid bounds; never interpolate or average a censored boundary."""
    per_seed = []
    for (method, seed), curve in dense.groupby(['method', 'seed']):
        curve = curve.sort_values('radius')
        radii = curve.radius.to_numpy(dtype=float)
        prefix = pc.np.maximum.accumulate(curve.G_train.to_numpy(dtype=float))
        for epsilon in EPSILONS:
            failed = pc.np.flatnonzero(prefix > epsilon)
            first_failed = int(failed[0]) if len(failed) else None
            require(first_failed != 0, 'Origin cannot fail a positive damage threshold')
            last_safe = len(radii) - 1 if first_failed is None else first_failed - 1
            low = float(radii[last_safe])
            high = float(radii[first_failed]) if first_failed is not None else None
            per_seed.append(dict(method=method, seed=int(seed), epsilon=epsilon,
                 largest_safe_measured_radius=low, first_failed_measured_radius=high,
                 bracket_width=None if high is None else high - low,
                 right_censored=high is None, scan_max_radius=float(radii[-1]),
                 continuous_boundary_identified=False, interpolation_used=False))
    seeds = pc.pd.DataFrame(per_seed)
    aggregate = []
    for (method, epsilon), group in seeds.groupby(['method', 'epsilon']):
        require(group.seed.tolist() == [0], 'Tolerance seed-0 cohort mismatch')
        values = group.largest_safe_measured_radius.to_numpy(dtype=float)
        censored = int(group.right_censored.sum())
        # A censored seed-0 bound is not plotted as an observed tolerance radius.
        aggregate.append(dict(method=method, epsilon=epsilon, n=1, censored_seeds=censored,
             plotted=censored == 0, mean=float(values.mean()) if not censored else None,
             sd=None,
             mean_largest_safe_measured_radius=float(values.mean()),
             interpretation='measured_grid_statistic' if not censored else 'censored_bound_only'))
    return seeds, pc.pd.DataFrame(aggregate)


def draw_single_seed(pc, ax, frame, value, marker_every):
    rows = []
    require(set(frame.seed.unique()) == {0}, 'Plot contains another training seed')
    for method in METHODS:
        curve = frame[frame.method == method].sort_values('radius')
        require(not curve.radius.duplicated().any(), 'Duplicate single-seed plot point')
        require(pc.np.isfinite(curve[value]).all(), 'Nonfinite single-seed curve')
        ax.plot(curve.radius, curve[value], color=pc.COLORS[method], linestyle=pc.LINES[method],
                linewidth=1.25, marker=pc.MARKERS[method], markersize=3.1, markevery=marker_every,
                label=pc.LABELS[method], zorder=3)
        rows.extend(dict(method=method, seed=0, radius=row.radius, mean=getattr(row, value),
                         n_training_seeds=1) for row in curve.itertuples())
    return pc.pd.DataFrame(rows)


def save_single_seed_figure(pc, fig, name, metadata):
    for ax in fig.axes:
        ax.tick_params(axis='both', length=3, pad=3)
        ax.xaxis.set_minor_locator(pc.NullLocator()); ax.yaxis.set_minor_locator(pc.NullLocator())
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    for text in fig.findobj(pc.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= fig_box.x0 - 1 and box.x1 <= fig_box.x1 + 1, ('clipped-x', text.get_text())
        assert box.y0 >= fig_box.y0 - 1 and box.y1 <= fig_box.y1 + 1, ('clipped-y', text.get_text())
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            box, bounds = legend.get_window_extent(renderer), ax.get_window_extent(renderer)
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), 'legend outside axes'
        boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_visible()]
        assert all(not a.overlaps(b) for a,b in zip(boxes, boxes[1:])), 'overlapping x ticks'
    outputs = {}
    for suffix in ('pdf', 'png'):
        path = pc.HERE / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white')
        outputs[suffix] = dict(path=str(path), sha256=sha(path))
    fonts = sorted(set(x.decode() for x in pc.re.findall(rb'/(?:BaseFont|FontName)\s*/([^\s/>]+)', (pc.HERE / f'{name}.pdf').read_bytes())))
    assert fonts and all('TimesNewRoman' in name for name in fonts), fonts
    metadata.update(status='passed', outputs=outputs, embedded_fonts=fonts,
                    figure_inches=list(fig.get_size_inches()),
                    statistics='One training seed (seed 0); mean over 50 dense / 16 Fisher paired directions; no uncertainty band or error bar',
                    source_cohort='seed0_dense50_fisher16_exploratory',
                    no_test_neighborhood_damage_curve=True,
                    rendered_visual_review='pending root and independent inspection')
    (pc.HERE / f'{name}_validation.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    pc.plt.close(fig)
    return metadata

def publish():
    prior_snapshot = {str(path): sha(path) for directory in (STYLE, COARSE / 'figures/radius50')
                      for path in directory.iterdir() if path.is_file()}
    pc, dense, fisher, meta = load_prior()
    checks, new_rows = [], []
    for seed in SEEDS:
        for method in METHODS:
            for job in JOBS:
                rows, sources, check = validate_extension(seed, method, job, pc)
                new_rows.extend(rows); checks.append(check); meta['source_files'].extend(sources)
    new = pc.pd.DataFrame(new_rows)
    dense = pc.pd.concat([dense, new[(new.job == 'dense') & (new.radius > 0)]], ignore_index=True)
    fisher = pc.pd.concat([fisher, new[(new.job == 'sparse_train') & (new.radius > 0)]], ignore_index=True)
    expected_dense = [i / 4 for i in range(81)] + list(range(25, 51))
    expected_fisher = FISHER_RADII + list(range(25, 51))
    for frame, grid in ((dense, expected_dense), (fisher, expected_fisher)):
        require(not frame.duplicated(['method', 'seed', 'radius']).any(), 'Duplicate merged source point')
        for _, group in frame.groupby(['method', 'seed']):
            require(sorted(group.radius.tolist()) == grid, 'Incomplete merged grid')
    tolerances, tolerance_stats = tolerance_tables(pc, dense)
    tag = 'radius50_integer_seed0_dense50_fisher16'
    out = ROOT / f'figures/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    dense = dense.drop(columns=['raw_fisher_trace', 'fisher_samples'], errors='ignore')
    dense.sort_values(['method', 'seed', 'radius']).to_csv(out / 'dense_training_per_seed.csv', index=False)
    fisher.sort_values(['method', 'seed', 'radius']).to_csv(out / 'fisher_training_per_seed.csv', index=False)
    tolerances.to_csv(out / 'tolerance_bounds_per_seed.csv', index=False)
    tolerance_stats.drop(columns=['sd']).to_csv(out / 'fourth_panel_single_seed.csv', index=False)
    if new_rows:
        pc.pd.DataFrame(new_rows).to_csv(out / 'extension_recomputed_per_seed.csv', index=False)
    pc.HERE = out
    font = pc.setup_style()
    fig, axes = pc.plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.32, top=.966, wspace=.56)
    specs = [(dense, 'G_train', r'Loss change $\overline{D_t(\boldsymbol{\theta},\boldsymbol{\Delta})}$', 'upper left'),
             (dense, 'train_accuracy_percent', 'Accuracy (%)', 'lower left'),
             (fisher, 'raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', 'upper left')]
    summaries = []
    maximum = 50
    for ax, (frame, value, label, location) in zip(axes[:3], specs):
        markers = 1 if value == 'raw_fisher_trace' else list(range(0, 81, 4)) + list(range(81, len(expected_dense)))
        agg = draw_single_seed(pc, ax, frame, value, marker_every=markers)
        pc.limits_from_sd(ax, agg.assign(sd=0.), include_zero=value == 'G_train')
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-1.25, maximum + 1.25),
               xticks=[0, 25, 50])
        pc.internal_legend(ax, location)
        agg['metric'] = value; summaries.append(agg)
    combined_stats = pc.pd.concat(summaries, ignore_index=True)
    combined_stats.to_csv(out / 'first_three_single_seed.csv', index=False)
    ax = axes[3]
    for method in METHODS:
        rows = tolerance_stats[tolerance_stats.method == method].sort_values('epsilon')
        # NaNs intentionally preserve line gaps at unavailable aggregate points.
        ax.plot(rows.epsilon, rows['mean'], color=pc.COLORS[method], linestyle=pc.LINES[method],
                linewidth=1.25, marker=pc.MARKERS[method], markersize=3.1, label=pc.LABELS[method], zorder=3)
    shown = tolerance_stats[tolerance_stats.plotted]
    upper = float(shown['mean'].max()) if len(shown) else maximum
    tolerance_ylim = max(upper * 1.12, 5)
    ax.set(xlabel=r'Allowed damage $\epsilon$', ylabel=r'Tolerance radius $\hat r_\epsilon$',
           xlim=(0, .0415), ylim=(-.55, tolerance_ylim), yticks=list(range(0, int(tolerance_ylim) + 1, 5)), xticks=EPSILONS,
           xticklabels=['0.005', '0.01', '0.02', '0.04'])
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    pc.internal_legend(ax, 'lower right')
    fig.align_xlabels(axes)
    name = f'Figure_B1_neighborhood_{tag}_epsilon4'
    sources = list({item['path']: item for item in meta['source_files']}.values())
    sources.append(file_receipt(Path(__file__)))
    for item in sources:
        require(sha(item['path']) == item['sha256'], f'Source changed during validation: {item["path"]}')
    figure_meta = dict(font=font, source_files=sources, integer_refinement=True,
         displayed_thresholds=EPSILONS, dense_radii=expected_dense, fisher_radii=expected_fisher,
         source_cohort='seed0_dense50_fisher16_exploratory', training_seeds=[0],
         dense_direction_indices=list(range(50)), fisher_direction_indices=list(range(16)),
         old_dense_means_recomputed_from_first_50_raw_directions=True,
         across_seed_uncertainty_displayed=False,
         original_raw_states_recomputed=meta['original_raw_states_recomputed'],
         new_raw_states_recomputed=sum(item['states'] for item in checks),
         validated_jobs=len(checks), expected_jobs=EXPECTED_JOBS,
         censored_aggregate_policy='omit method×epsilon point and line segment if seed 0 is censored',
         tolerance_is_prefix_safe_on_measured_grid=True, tolerance_interpolation_used=False,
         unsampled_radius_20_to_25_gap=True,
         reused_coarse_validated_jobs=len(meta['reused_coarse_jobs']),
         reused_coarse_jobs=meta['reused_coarse_jobs'],
         reused_coarse_receipt=meta['reused_coarse_receipt'],
         measured_fisher_features_preserved='All seed-0 means retained; no clipping or smoothing; no cross-seed SD',
         previous_figure_sha256=prior_snapshot, new_model_evaluations_by_publisher=False)
    result = save_single_seed_figure(pc, fig, name, figure_meta)
    caption = (
        'Exploratory training-neighborhood diagnostics for seed 0 of the formal ImageNet-A cohort '
        '(ViT-B/16, rank-10 Q/V LoRA), at fixed Task-0 anchors and fixed old 20-class heads. '
        'Panels show mean positive CE damage, Task-0 training accuracy (%), unnormalized empirical Fisher '
        'trace, and measured-grid tolerance radius. Loss and accuracy use all 1,259 old training examples '
        f'and the first 50 paired unit directions at {len(expected_dense)} measured radii. '
        'All earlier loss/accuracy points were recomputed from those same first 50 persisted directions; '
        'the former 100-direction means are never mixed into these curves. The original dense grid is '
        '0–20 in steps of 0.25, followed by every integer radius from 25 through 50. The open interval '
        '20–25 remains unsampled. Fisher retains all 16 paired directions and a fixed 256-example '
        f'training subset at {len(expected_fisher)} measured radii: '
        + ', '.join(f'{r:g}' for r in expected_fisher) + '. '
        'Repeated extension origins are validation checks and are not counted twice. Fisher is the mean '
        'squared per-example true-label CE gradient norm, without EWC scaling, trace normalization, '
        'parameter-count normalization, floor, or regularizer. All measured features are retained without '
        'clipping or smoothing. Damage is G(r)=mean_direction(max(mean_example CE(theta+r*u) − '
        'mean_example CE(theta), 0)); clipping occurs after sample averaging for each direction. '
        'The fourth panel uses epsilon = 0.005, 0.01, 0.02, and 0.04. For seed 0 independently in each '
        'method, take the prefix maximum of G on the ordered measured grid and retain the largest measured '
        'radius whose entire measured prefix is <= epsilon. If seed 0 remains safe at the scan limit, '
        'that method×epsilon point and its connecting segment are omitted. The CSV gives the largest safe '
        'measured radius and first failed measured radius; these bracket observations on a finite grid, '
        'not a continuous safe interval or an interpolated exact threshold. A 20/25 pair still leaves '
        'the crossing unresolved between those sampled points. Lines are visual guides only. This is '
        'one training seed, with means across paired directions; no across-seed standard deviation, '
        'confidence band, or error bar is displayed. These exploratory curves do not replace the prior '
        'three-seed formal results. The CPU publisher recomputes plotted quantities from persisted raw '
        'measurements and checks receipts, source hashes, origin restoration, model/data/head/coordinate '
        'identities, and the shared direction library.'
    )
    (out / 'CAPTION.md').write_text(caption + '\n')
    latex_caption = (caption.replace('_', '\\_').replace('%', '\\%').replace('<=', '$\\leq$')
                     .replace('×', '$\\times$').replace('−', '$-$'))
    (out / 'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'
             + name + '.pdf}\n\\caption{' + latex_caption
             + '}\n\\label{fig:rq2-' + tag.replace('_', '-') + '-epsilon4}\n\\end{figure*}\n')
    (out / 'README.md').write_text(
        f'# B1 {tag}: four thresholds\n\n[PNG]({name}.png) · [PDF]({name}.pdf) · [Caption](CAPTION.md)\n\n'
        'Times New Roman and the prior one-row four-panel style are retained. All inputs are read-only. '
        'Only seed 0 is plotted, with 50 dense directions and 16 Fisher directions at all radii. No across-seed uncertainty is shown.\n\n'
        'See `tolerance_bounds_per_seed.csv` for per-seed last-safe/first-failed measured radii. '
        'A missing first-failed radius means right censoring; the corresponding method×epsilon mean '
        'is blank in `fourth_panel_single_seed.csv`, and no point is drawn. '
        'No interpolation estimates the threshold within radius-grid gaps.\n\n'
        'Reproduce: `python ../../publish_radius50.py --execute`.\n')
    for path, digest in prior_snapshot.items():
        require(sha(path) == digest, f'Prior figure modified: {path}')
    complete = dict(status='complete', integer_refinement=True, expected_jobs=EXPECTED_JOBS,
         validated_jobs=len(checks), jobs=checks, automatic_numeric_and_layout_checks='passed',
         reused_coarse_validated_jobs=len(meta['reused_coarse_jobs']), reused_coarse_jobs=meta['reused_coarse_jobs'],
         reused_coarse_receipt=meta['reused_coarse_receipt'],
         source_files=sources, output_files=[file_receipt(path) for path in sorted(out.iterdir()) if path.is_file()],
         figure=result['outputs'], previous_figures_preserved=True,
         rendered_visual_review='pending actual rendered-image inspection',
         generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path = ROOT / 'report/radius50_complete.json'
    atomic_json(receipt_path, complete)
    atomic_json(ROOT / 'publisher_receipt.json', complete)
    return dict(status='complete', validated_jobs=len(checks), receipt=str(receipt_path),
                outputs=result['outputs'], censored_method_epsilon_points=int((~tolerance_stats.plotted).sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Read-only inventory and completed receipt SHA checks (default)')
    mode.add_argument('--execute', action='store_true', help='Validate all 4 seed-0 extension jobs and publish radius50')
    args = parser.parse_args(argv)
    try:
        result = inventory()
        if args.execute:
            if result['complete_job_files'] != EXPECTED_JOBS:
                print(json.dumps(result, indent=2)); return 2
            result = publish()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (AssertionError, ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps(dict(status='invalid', error=f'{type(exc).__name__}: {exc}',
                              complete_receipt_written=False), indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
