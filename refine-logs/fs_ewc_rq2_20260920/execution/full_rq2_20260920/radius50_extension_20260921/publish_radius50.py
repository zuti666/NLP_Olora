#!/usr/bin/env python3
"""CPU-only B1 publisher: default inventory is read-only; never opens a GPU.

--check-only: report extension coverage and verify any completed receipt hashes.
--execute: require and validate all 12 jobs, then publish the merged radius-50 plot.
--current-only: validate existing radius <=20 data and render the revised epsilon plot.
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
PRIOR = BASE / 'stage_path_completion_20260921'
STYLE = BASE / 'figures_formal_20260921/B1_merged_latest_20260921'
METHODS = ('sgd', 'gam')
JOBS = ('dense', 'sparse_train')
RADII = [0, 25, 30, 35, 40, 45, 50]
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
    for seed in range(3):
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
    return dict(status='ready_to_validate' if count == 12 else 'pending', expected_jobs=12,
                complete_job_files=count, jobs=jobs, extension_radii=RADII,
                final_receipt=str(ROOT / 'report/radius50_complete.json'),
                check_only_is_read_only=True)


def load_prior():
    """Recompute old dense and all 16 Fisher radii from raw states, not plot CSVs."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rq2-radius50-publisher-mpl'))
    sys.path.insert(0, str(BASE / 'figures_formal_20260921'))
    path = PRIOR / 'publish_radius.py'
    spec = importlib.util.spec_from_file_location('rq2_prior_radius_publisher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt_path = PRIOR / 'report/radius_complete.json'
    prior_receipt = read(receipt_path)
    require(prior_receipt['status'] == 'complete' and prior_receipt['validated_jobs'] == 6,
            'Prior 16-radius Fisher publication is incomplete')
    for item in prior_receipt['source_files']:
        require(sha(item['path']) == item['sha256'], f'Prior source changed: {item["path"]}')
    pc, dense, fisher, _, metadata = module.validate_all()
    dense = dense.drop(columns=['raw_fisher_trace', 'fisher_samples'], errors='ignore')
    dense['source_partition'] = 'original_formal'
    require(sorted(fisher.radius.unique()) == FISHER_RADII, 'Prior Fisher grid is not the latest 16-radius grid')
    style_manifest_path = STYLE / 'ARTIFACT_MANIFEST.json'
    style_manifest = read(style_manifest_path)
    for item in style_manifest['files']:
        require(sha(item['path']) == item['sha256'], f'Prior figure file changed: {item["path"]}')
    # The previous figure serves only as a style/provenance reference. Compare its
    # displayed inputs against freshly rebuilt raw measurements before reusing style.
    for frame, filename, value in ((dense, 'dense_training_per_seed.csv', 'G_train'),
                                    (fisher, 'fisher_training_per_seed.csv', 'raw_fisher_trace')):
        previous = pc.pd.read_csv(STYLE / filename)
        keys = ['method', 'seed', 'radius']
        joined = frame.merge(previous, on=keys, suffixes=('_raw', '_prior'), validate='one_to_one')
        require(len(joined) == len(frame) == len(previous), f'Prior plot coverage changed: {filename}')
        pc.np.testing.assert_allclose(joined[value + '_raw'], joined[value + '_prior'], rtol=1e-10, atol=1e-12)
    metadata['source_files'].extend(file_receipt(p) for p in
                                   (receipt_path, path, style_manifest_path, STYLE / 'plot_b1_merged_latest.py'))
    metadata['original_raw_states_recomputed'] += metadata.pop('new_raw_states_recomputed')
    return pc, dense, fisher, metadata


def validate_extension(seed, method, job, pc):
    directory = ROOT / f'results/seed{seed}/{method}/{job}'
    summary_path, manifest_path, states_path = [directory / name for name in
                                               ('summary.json', 'manifest.json', 'states.jsonl')]
    config_path = ROOT / f'configs/{job}_seed{seed}.json'
    summary, manifest, config = read(summary_path), read(manifest_path), read(config_path)
    wrapper = check_receipt_files(directory)
    count = 100 if job == 'dense' else 16
    expected = 1 + 6 * count
    require(wrapper['schema'] == 'rq2_radius50_extension_v1', 'Unexpected worker receipt schema')
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
    require(changed == {'output_dir', grid_key}, f'Unexpected configuration changes: {sorted(changed)}')
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
    require(set(library) == set(range(count)), 'Original direction library incomplete')
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
                    source_partition='radius50_repeat_origin' if radius == 0 else 'radius50_extension')
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
        require(sorted(group.seed.tolist()) == [0, 1, 2], 'Tolerance seed cohort incomplete')
        values = group.largest_safe_measured_radius.to_numpy(dtype=float)
        censored = int(group.right_censored.sum())
        # Even one censored seed makes a method×epsilon aggregate unavailable.
        aggregate.append(dict(method=method, epsilon=epsilon, n=3, censored_seeds=censored,
             plotted=censored == 0, mean=float(values.mean()) if not censored else None,
             sd=float(values.std(ddof=1)) if not censored else None,
             mean_largest_safe_measured_radius=float(values.mean()),
             interpretation='measured_grid_statistic' if not censored else 'censored_bound_only'))
    return seeds, pc.pd.DataFrame(aggregate)


def publish(current_only=False):
    prior_snapshot = {str(path): sha(path) for path in STYLE.iterdir() if path.is_file()}
    pc, dense, fisher, meta = load_prior()
    checks, new_rows = [], []
    if not current_only:
        for seed in range(3):
            for method in METHODS:
                for job in JOBS:
                    rows, sources, check = validate_extension(seed, method, job, pc)
                    new_rows.extend(rows); checks.append(check); meta['source_files'].extend(sources)
        new = pc.pd.DataFrame(new_rows)
        dense = pc.pd.concat([dense, new[(new.job == 'dense') & (new.radius > 0)]], ignore_index=True)
        fisher = pc.pd.concat([fisher, new[(new.job == 'sparse_train') & (new.radius > 0)]], ignore_index=True)
    expected_dense = [i / 4 for i in range(81)] + ([] if current_only else RADII[1:])
    expected_fisher = FISHER_RADII + ([] if current_only else RADII[1:])
    for frame, grid in ((dense, expected_dense), (fisher, expected_fisher)):
        require(not frame.duplicated(['method', 'seed', 'radius']).any(), 'Duplicate merged source point')
        for _, group in frame.groupby(['method', 'seed']):
            require(sorted(group.radius.tolist()) == grid, 'Incomplete merged grid')
    tolerances, tolerance_stats = tolerance_tables(pc, dense)
    tag = 'current_r20' if current_only else 'radius50'
    out = ROOT / f'figures/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    dense = dense.drop(columns=['raw_fisher_trace', 'fisher_samples'], errors='ignore')
    dense.sort_values(['method', 'seed', 'radius']).to_csv(out / 'dense_training_per_seed.csv', index=False)
    fisher.sort_values(['method', 'seed', 'radius']).to_csv(out / 'fisher_training_per_seed.csv', index=False)
    tolerances.to_csv(out / 'tolerance_bounds_per_seed.csv', index=False)
    tolerance_stats.to_csv(out / 'fourth_panel_mean_sd.csv', index=False)
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
    maximum = 20 if current_only else 50
    for ax, (frame, value, label, location) in zip(axes[:3], specs):
        markers = 1 if value == 'raw_fisher_trace' else list(range(0, 81, 4)) + list(range(81, len(expected_dense)))
        agg = pc.draw_mean_sd(ax, frame, 'radius', value, marker_every=markers)
        pc.limits_from_sd(ax, agg, include_zero=value == 'G_train')
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.5, maximum + .5),
               xticks=[0, 10, 20] if current_only else [0, 25, 50])
        pc.internal_legend(ax, location)
        agg['metric'] = value; summaries.append(agg)
    combined_stats = pc.pd.concat(summaries, ignore_index=True)
    combined_stats.to_csv(out / 'first_three_mean_sd.csv', index=False)
    ax = axes[3]
    for method in METHODS:
        rows = tolerance_stats[tolerance_stats.method == method].sort_values('epsilon')
        # NaNs intentionally preserve line gaps at unavailable aggregate points.
        ax.plot(rows.epsilon, rows['mean'], color=pc.COLORS[method], linestyle=pc.LINES[method],
                linewidth=1.25, marker=pc.MARKERS[method], markersize=3.1, label=pc.LABELS[method], zorder=3)
        shown = rows[rows.plotted]
        ax.errorbar(shown.epsilon, shown['mean'], yerr=shown.sd, color=pc.COLORS[method],
                    linestyle='none', capsize=2, linewidth=.8, capthick=.8, zorder=2)
    shown = tolerance_stats[tolerance_stats.plotted]
    upper = float((shown['mean'] + shown.sd).max()) if len(shown) else maximum
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
    figure_meta = dict(font=font, source_files=sources, current_only=current_only,
         displayed_thresholds=EPSILONS, dense_radii=expected_dense, fisher_radii=expected_fisher,
         original_raw_states_recomputed=meta['original_raw_states_recomputed'],
         new_raw_states_recomputed=sum(item['states'] for item in checks),
         validated_jobs=len(checks), expected_jobs=0 if current_only else 12,
         censored_aggregate_policy='omit method×epsilon point and line segment if any of its 3 seeds is censored',
         tolerance_is_prefix_safe_on_measured_grid=True, tolerance_interpolation_used=False,
         unsampled_radius_20_to_25_gap=not current_only,
         fisher_spike_preserved='r17 EWCLoRA and full sample SD; no clipping or smoothing',
         previous_figure_sha256=prior_snapshot, new_model_evaluations_by_publisher=False)
    result = pc.save_figure(fig, name, figure_meta)
    caption = (
        'Training-neighborhood diagnostics for the formal ImageNet-A seeds 0, 1, and 2 '
        '(ViT-B/16, rank-10 Q/V LoRA), at fixed Task-0 anchors and fixed old 20-class heads. '
        'Panels show mean positive CE damage, Task-0 training accuracy (%), raw empirical Fisher trace, '
        'and measured-grid tolerance radius. Loss and accuracy use all 1,259 old training examples '
        f'and 100 paired unit directions at {len(expected_dense)} measured radii. '
        'The original dense grid is 0–20 in steps of 0.25. '
        + ('' if current_only else 'New radii are 25, 30, 35, 40, 45, and 50; the interval 20–25 is not sampled. ')
        + 'Fisher uses the same coordinates and anchors, a fixed 256-example training subset, and the first '
        f'16 paired directions at {len(expected_fisher)} measured radii: '
        + ', '.join(f'{r:g}' for r in expected_fisher) + '. '
        'Repeated extension origins are consistency checks and are not included twice. '
        'Fisher is the mean squared per-example true-label CE gradient norm, with no EWC scaling, '
        'normalization, floor, or regularizer. The r=17 EWCLoRA spike and its full sample SD are retained. '
        'Damage is G(r)=mean_direction(max(mean_example CE(theta+r*u) − mean_example CE(theta), 0)); '
        'clipping occurs after sample averaging, separately for each direction. '
        'The fourth panel uses epsilon = 0.005, 0.01, 0.02, and 0.04. For each seed separately, take the '
        'prefix maximum of G on the ordered measured grid and retain the largest measured radius whose '
        'entire measured prefix is <= epsilon; then compute the three-seed mean and sample SD. '
        'If any seed of a method remains safe at the scan limit, that method×epsilon aggregate is omitted '
        'entirely (no marker, cap, or connecting segment). Censored bounds are tabulated separately. '
        'The CSV reports the largest safe measured radius and the first failed measured radius for every '
        'seed. These are measured-grid bracketing observations; no continuous safe interval or exact '
        'threshold within a gap is established. '
        + ('' if current_only else 'In particular, a 20/25 pair leaves the crossing unresolved between those sampled points; ')
        + 'no interpolation is used. Lines between measured nodes are visual '
        'guides only. Shading in panels 1–3 and error bars in panel 4 are one sample standard deviation '
        'across three training seeds (ddof=1), not confidence intervals. Directions are averaged within '
        'each seed and are not independent training repeats. All panels concern training diagnostics. '
        'The CPU publisher recomputes plotted quantities from persisted raw measurements and checks '
        'receipt hashes, origins, head/data/coordinate identities, and the shared direction library.'
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
        'Every plotted aggregate uses exactly seeds 0/1/2 and sample SD.\n\n'
        'See `tolerance_bounds_per_seed.csv` for per-seed last-safe/first-failed measured radii. '
        'A missing first-failed radius means right censoring; the corresponding method×epsilon mean '
        'and SD are blank in `fourth_panel_mean_sd.csv`, and no point is drawn. '
        'No interpolation estimates the threshold within radius-grid gaps.\n\n'
        f'Reproduce: `python ../../publish_radius50.py {"--current-only" if current_only else "--execute"}`.\n')
    for path, digest in prior_snapshot.items():
        require(sha(path) == digest, f'Prior figure modified: {path}')
    complete = dict(status='complete', current_only=current_only, expected_jobs=0 if current_only else 12,
         validated_jobs=len(checks), jobs=checks, automatic_numeric_and_layout_checks='passed',
         source_files=sources, output_files=[file_receipt(path) for path in sorted(out.iterdir()) if path.is_file()],
         figure=result['outputs'], previous_figures_preserved=True,
         rendered_visual_review='pending actual rendered-image inspection',
         generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path = ROOT / ('report/current_r20_complete.json' if current_only else 'report/radius50_complete.json')
    atomic_json(receipt_path, complete)
    if not current_only:
        atomic_json(ROOT / 'publisher_receipt.json', complete)
    return dict(status='complete', validated_jobs=len(checks), receipt=str(receipt_path),
                outputs=result['outputs'], censored_method_epsilon_points=int((~tolerance_stats.plotted).sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Read-only inventory and completed receipt SHA checks (default)')
    mode.add_argument('--execute', action='store_true', help='Validate all 12 extension jobs and publish radius50')
    mode.add_argument('--current-only', action='store_true', help='Validate existing radius<=20 and render new four thresholds now')
    args = parser.parse_args(argv)
    try:
        if args.current_only:
            result = publish(current_only=True)
        else:
            result = inventory()
            if args.execute:
                if result['complete_job_files'] != 12:
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
