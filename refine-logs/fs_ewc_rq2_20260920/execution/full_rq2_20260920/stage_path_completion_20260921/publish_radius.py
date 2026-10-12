#!/usr/bin/env python3
"""Validate and publish the formal three-seed B1 radius extension (CPU only).

Default/--check-only prints coverage without writing files. --execute publishes
only after all six jobs and the original formal inputs pass the checks below.
No checkpoint is trained, no GPU is opened, and no historical plot is replaced.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile


EXT = Path(__file__).resolve().parent
BASE = EXT.parent
FIGURES = EXT / 'figures'
REPORT = EXT / 'report'
METHODS = ('sgd', 'gam')
NEW_RADII = [0] + list(range(11, 20))
COMBINED_RADII = [0, .5, 1, 2, 5, 10] + list(range(11, 21))
FIGURE_NAME = 'Figure_B1_neighborhood_extended'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, *, rtol=1e-10, atol=1e-12, label='numeric value'):
    require(math.isfinite(float(a)) and math.isfinite(float(b)), f'{label}: non-finite')
    require(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol), f'{label}: {a} != {b}')


def receipt(path):
    return {'path': str(path), 'sha256': sha(path)}


def inventory():
    jobs = []
    for seed in range(3):
        for method in METHODS:
            directory = EXT / f'results/seed{seed}/{method}/sparse_train'
            files = [directory / name for name in ('summary.json', 'manifest.json', 'states.jsonl', 'extension_receipt.json')]
            files.append(EXT / f'configs/radius_seed{seed}.json')
            missing = [str(p) for p in files if not p.is_file()]
            state = 'missing' if missing else read(files[3]).get('status', 'unknown')
            if not missing and read(files[0]).get('status') != 'complete':
                state = 'summary_incomplete'
            jobs.append(dict(seed=seed, method=method, directory=str(directory),
                             status=state, missing=missing))
    ready = sum(j['status'] == 'complete' and not j['missing'] for j in jobs)
    return dict(status='ready_to_validate' if ready == 6 else 'pending',
                expected_jobs=6, complete_job_files=ready,
                missing_or_incomplete_jobs=6-ready, jobs=jobs,
                measurement_radii=NEW_RADII, merged_fisher_radii=COMBINED_RADII,
                expected_new_nonzero_states=864, expected_states_with_origins=870)


def validate_job(seed, method, common):
    """Recompute new points and prove they extend the same formal protocol."""
    directory = EXT / f'results/seed{seed}/{method}/sparse_train'
    summary_path, manifest_path, states_path = [directory / name for name in
                                               ('summary.json', 'manifest.json', 'states.jsonl')]
    config_path = EXT / f'configs/radius_seed{seed}.json'
    summary, manifest, config = read(summary_path), read(manifest_path), read(config_path)
    wrapper_path = EXT / 'diagnostic_extension.py'
    extension_path = directory / 'extension_receipt.json'
    extension = read(extension_path)
    require(extension['status'] == 'complete' and extension['schema'] == 'rq2_formal_completion_v1' and
            extension['seed'] == seed and extension['method'] == method and
            extension['job'] == 'sparse_train' and extension['target_task'] is None and extension['samples'] == 256,
            'Wrapper completion receipt missing or belongs to another diagnostic')
    require(Path(extension['summary']).resolve() == summary_path.resolve(), 'Wrapper summary path mismatch')
    for key, path in (('summary_sha256', summary_path), ('manifest_sha256', manifest_path),
                      ('states_sha256', states_path), ('wrapper_sha256', wrapper_path)):
        require(extension[key] == sha(path), f'Wrapper receipt {key} mismatch')
    require(extension['protocol_sha256'] == manifest['protocol_sha256'], 'Wrapper protocol mismatch')
    require(extension['preflight']['status'] == 'passed', 'Wrapper preflight failed')
    for check in ('identity_check', 'origin_check', 'endpoint_check'):
        require(extension[check]['passed'] is True, f'Wrapper {check} failed')
    require(extension['preflight']['wrapper']['sha256'] == sha(wrapper_path), 'Preflight wrapper SHA changed')
    require(extension['preflight']['config']['sha256'] == sha(config_path), 'Preflight radius config SHA changed')
    require(extension['endpoint_check']['measured_states'] == 145 and
            extension['endpoint_check']['radii'] == NEW_RADII, 'Wrapper completed grid mismatch')
    old_summary, old_manifest, old_sources = common.audited_summary(seed, method, 'sparse_train')
    old_states_path = BASE / f'formal_results/seed{seed}/{method}/sparse_train/states.jsonl'
    old_rows = [json.loads(line) for line in old_states_path.read_text().splitlines()]
    old_origin = next(r for r in old_rows if r['point']['radius'] == 0)
    old_config_path = BASE / f'configs/seed{seed}.json'
    old_config = read(old_config_path)
    require(old_config == old_manifest['config'], 'Original config differs from archived formal manifest')
    require(sha(old_config_path) == old_manifest['config_sha256'], 'Original config SHA changed')
    require(config == manifest['config'], 'New manifest does not contain the actual radius config')
    require(sha(config_path) == manifest['config_sha256'], 'Radius config SHA mismatch')
    changed = {key for key in set(config) | set(old_config) if config.get(key) != old_config.get(key)}
    require(changed == {'output_dir', 'sparse_train_radii'}, f'Unexpected config changes: {sorted(changed)}')
    require(config['sparse_train_radii'] == NEW_RADII, 'Incorrect extension radius grid')
    require(Path(config['output_dir']).resolve() == (EXT / f'results/seed{seed}').resolve(),
            'Radius config points outside the formal extension result directory')
    require(canonical({k: v for k, v in manifest.items() if k != 'protocol_sha256'}) ==
            manifest['protocol_sha256'], 'Manifest canonical protocol hash mismatch')
    mutable = {'config', 'config_sha256', 'protocol_sha256'}
    identity = {key for key in set(manifest) | set(old_manifest)
                if key not in mutable and manifest.get(key) != old_manifest.get(key)}
    require(not identity, f'Formal model/data/code/protocol identity changed: {sorted(identity)}')
    require(summary['status'] == 'complete' and summary['seed'] == seed and
            summary['method'] == method and summary['job'] == 'sparse_train', 'Summary identity/status mismatch')
    require(summary['protocol_sha256'] == manifest['protocol_sha256'], 'Summary manifest mismatch')
    require(summary['direction_indices'] == list(range(16)), 'Not the original first 16 public directions')
    require(summary['gradient_scope'] == ['subset'] and not summary['forward_only'], 'Incorrect Fisher scope')
    require(summary['paired_public_direction_library'] is True, 'Directions are not paired')
    require([r['radius'] for r in summary['radius_summary']] == NEW_RADII, 'Summary radii mismatch')
    restored = summary['end_restoration']
    require(restored['predictions_match'] is True and restored['max_logit_error'] == 0 and
            restored['effective_sha256'] == manifest['anchor_effective_sha256'], 'Final anchor restoration failed')

    direction_library = {}
    for row in old_rows:
        if row['point']['radius']:
            direction = row['point']['direction']
            index = direction['index']
            if index in direction_library:
                require(direction_library[index] == direction, 'Original direction library changed across radii')
            direction_library[index] = direction
    require(set(direction_library) == set(range(16)), 'Incomplete original direction library')
    rows = [json.loads(line) for line in states_path.read_text().splitlines()]
    require(len(rows) == len({r['state_id'] for r in rows}) == 145, 'Expected 145 unique states per job')
    require(summary['expected_states'] == summary['states_completed'] == 145, 'Summary state count mismatch')
    origins = [r for r in rows if r['point']['radius'] == 0]
    require(len(origins) == 1, 'Expected exactly one origin observation')
    origin = origins[0]
    grouped = defaultdict(list)
    for row in rows:
        require(row['seed'] == seed and row['method'] == method, 'Raw row belongs to another run')
        require(row['protocol_sha256'] == manifest['protocol_sha256'], 'Raw protocol mismatch')
        require(row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method], 'Raw checkpoint mismatch')
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == manifest[key], f'Raw {key} mismatch')
        require(row['anchor_restored_exactly'] is True, 'A point did not restore its anchor')
        radius = row['point']['radius']
        require(radius in NEW_RADII, 'Unexpected radius in raw rows')
        close(row['effective_weight_drift'], radius, rtol=2e-6, atol=1e-9, label='actual drift')
        close(row['effective_weight_drift_squared'], radius**2, rtol=4e-6, atol=1e-9, label='drift squared')
        if radius:
            direction = row['point']['direction']
            require(direction == direction_library.get(direction['index']), 'Public direction hash/seed/norm mismatch')
        require(row['metrics']['train']['samples'] == 1259, 'Training metric sample count changed')
        require(row['metrics']['test']['samples'] == 307, 'Test metric sample count changed')
        require(set(row['fisher_by_scope']) == {'subset'}, 'Unexpected raw Fisher scope')
        fisher = row['fisher_by_scope']['subset']
        require(fisher['samples'] == 256 and fisher['normalized'] is False and
                fisher['floor'] is None and fisher['regularizer'] is None, 'Raw Fisher definition changed')
        require(math.isfinite(fisher['raw_trace']) and fisher['raw_trace'] >= 0, 'Invalid raw Fisher')
        grouped[radius].append(row)

    # Repeating the anchor is a consistency witness, not an extra direction or seed.
    close(origin['fisher_by_scope']['subset']['raw_trace'], old_origin['fisher_by_scope']['subset']['raw_trace'],
          rtol=1e-5, atol=1e-7, label='repeated origin raw Fisher')
    close(origin['fisher_by_scope']['subset']['subset_loss'], old_origin['fisher_by_scope']['subset']['subset_loss'],
          rtol=1e-6, atol=3e-7, label='repeated origin subset CE')
    for split in ('train', 'test'):
        close(origin['metrics'][split]['loss'], old_origin['metrics'][split]['loss'],
              rtol=1e-6, atol=3e-7, label=f'repeated origin {split} CE')
        close(origin['metrics'][split]['accuracy'], old_origin['metrics'][split]['accuracy'],
              atol=1e-9, label=f'repeated origin {split} accuracy')
    output = []
    for entry in summary['radius_summary']:
        radius = entry['radius']; points = grouped[radius]
        require(len(points) == (1 if radius == 0 else 16), 'Missing or duplicate radius/direction pair')
        if radius:
            require({r['point']['direction']['index'] for r in points} == set(range(16)), 'Missing paired direction')
        trace = math.fsum(r['fisher_by_scope']['subset']['raw_trace'] for r in points) / len(points)
        damage = [max(r['metrics']['train']['loss'] - origin['metrics']['train']['loss'], 0) for r in points]
        loss = math.fsum(damage) / len(points)
        accuracy = math.fsum(r['metrics']['train']['accuracy'] for r in points) / len(points)
        close(trace, entry['raw_trace_by_scope']['subset']['mean'], label='summary raw Fisher mean')
        close(loss, entry['train']['G'], label='summary mean positive damage')
        close(accuracy, entry['train']['accuracy_mean'], label='summary training accuracy')
        output.append(dict(seed=seed, method=method, job='sparse_train', radius=radius,
                           directions=len(points), train_loss_samples=1259, G_train=loss,
                           train_accuracy_percent=accuracy, raw_fisher_trace=trace, fisher_samples=256,
                           source_partition='extension_repeat_origin' if radius == 0 else 'extension'))
    sources = old_sources + [receipt(p) for p in (old_states_path, old_config_path, config_path,
                                                 summary_path, manifest_path, states_path,
                                                 wrapper_path, extension_path)]
    check = dict(seed=seed, method=method, status='validated', states=145, new_nonzero_states=144,
                 protocol_sha256=manifest['protocol_sha256'], anchor_identity_matches=True,
                 code_data_head_coordinate_identity_matches=True, direction_library_matches=True,
                 origin_repeat_matches=True, metrics_recomputed_from_raw=True)
    return output, sources, check


def validate_all():
    # Import only once all job files exist; no GPU/model import is needed.
    sys.path.insert(0, str(BASE / 'figures_formal_20260921'))
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rq2-radius-publisher-mpl'))
    import plot_common as common
    from plot_b1 import load_points
    old, _, sources, old_checked = load_points()
    new_rows, jobs = [], []
    for seed in range(3):
        for method in METHODS:
            rows, files, check = validate_job(seed, method, common)
            new_rows.extend(rows); sources.extend(files); jobs.append(check)
    new = common.pd.DataFrame(new_rows)
    old_fisher = old.query("job == 'sparse_train'").copy()
    old_fisher['source_partition'] = 'original_formal'
    sparse = common.pd.concat([old_fisher, new.query('radius > 0')], ignore_index=True)
    for (seed, method), group in sparse.groupby(['seed', 'method']):
        require(sorted(group.radius.tolist()) == COMBINED_RADII, f'Incomplete merged grid: {seed}/{method}')
    dense = old.query("job == 'dense'").copy()
    sources = list({s['path']: s for s in sources}.values())
    sources.extend(receipt(p) for p in (Path(__file__).resolve(), BASE / 'figures_formal_20260921/plot_b1.py',
                                       BASE / 'figures_formal_20260921/plot_common.py'))
    metadata = dict(status='validated', expected_jobs=6, validated_jobs=6, jobs=jobs,
                    original_raw_states_recomputed=old_checked, new_raw_states_recomputed=870,
                    new_nonzero_states=864, merged_fisher_radii=COMBINED_RADII, source_files=sources,
                    original_formal_figure_unchanged=True, historical_data_used=False,
                    checkpoint_physical_files_rehashed=False,
                    checkpoint_identity='raw rows and manifests matched to the original audited formal cohort',
                    visual_review='pending actual rendered-image inspection')
    return common, dense, sparse, new, metadata


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def publish(common, dense, sparse, new, metadata):
    FIGURES.mkdir(parents=True, exist_ok=True)
    common.HERE = FIGURES
    font = common.setup_style()
    dense.drop(columns=['raw_fisher_trace', 'fisher_samples']).to_csv(FIGURES / 'B1_dense_per_seed.csv', index=False)
    sparse.sort_values(['method', 'seed', 'radius']).to_csv(FIGURES / 'B1_fisher_per_seed.csv', index=False)
    new.to_csv(FIGURES / 'B1_radius_extension_recomputed.csv', index=False)
    fig, axes = common.plt.subplots(1, 3, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.07, right=.99, bottom=.25, top=.965, wspace=.42)
    specs = [
        (dense, 'G_train', r'Loss $\overline{[D_t(\boldsymbol{\theta},\boldsymbol{\Delta})]_+}$', 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy', 'lower left'),
        (sparse, 'raw_fisher_trace', 'Raw Fisher trace', 'upper left'),
    ]
    aggregates = []
    for ax, (frame, value, label, location) in zip(axes, specs):
        aggregate = common.draw_mean_sd(ax, frame, 'radius', value,
                                       marker_every=1 if value == 'raw_fisher_trace' else 4)
        aggregate['metric'] = value; aggregates.append(aggregate)
        ax.set_xlabel('Weight drift'); ax.set_ylabel(label)
        ax.set_xlim(-.35, 20.35); ax.set_xticks([0, 5, 10, 15, 20])
        common.limits_from_sd(ax, aggregate, include_zero=value == 'G_train')
        common.internal_legend(ax, location)
    common.pd.concat(aggregates, ignore_index=True).to_csv(FIGURES / 'B1_mean_sd.csv', index=False)
    figure_meta = dict(metadata, font=font, source_cohort='formal ImageNet-A seed0/1/2',
                       dense_grid=dict(points=81, step=.25, directions=100, training_samples=1259),
                       fisher_grid=dict(radii=COMBINED_RADII, directions=16, training_subset_samples=256),
                       mean_and_sample_sd_across_training_seeds=True,
                       no_interpolated_fisher_measurements=True)
    common.save_figure(fig, FIGURE_NAME, figure_meta)
    caption = '''# Formal B1 neighborhood radius extension

Three training-side diagnostics on the formal ImageNet-A three-seed cohort: mean positive loss increase, Task-0 training accuracy (%), and raw empirical Fisher trace. Left and middle reuse all 81 measured radii (0–20, step 0.25), 100 paired directions, and all 1,259 old-task training examples from the original formal dense scan. Dense markers are displayed every 1.0; the line uses every measured radius. Right uses the same formal Task-0 checkpoint, fixed old-task head, effective Q/V coordinate system and first 16 public directions, with the same 256-example training subset. The measured Fisher radii are 0, 0.5, 1, 2, 5, and every integer from 10 to 20 (16 radii). Radii 11–19 are new measurements; all original seven radii are retained. Six repeated origin measurements validate consistency and are not additional samples in the displayed averages.

Curves show arithmetic means and one sample standard deviation across training seeds 0/1/2, after averaging directions within each seed. Directions are not counted as independent training repeats. Raw Fisher is the mean squared per-example true-label CE gradient norm; it has no EWC coefficient, normalization, floor, or regularizer. Line segments between Fisher measurements are guides, not additional evaluations. No test damage, Q95, or tolerance-radius panel is shown. These curves establish training-side diagnostics and do not imply improved test-side perturbation tolerance.

中文：保留正式三 seed 数据口径。损伤和准确率复用完整训练集的 81×100 原始测量；Fisher 保留 256 个固定训练样本和 16 个配对方向，新增 11–19 共九个半径，合并为 16 个实测半径。原点复测仅用于核验，没有重复计入曲线。阴影为跨训练 seed 的样本标准差，未混入历史 checkpoint。无需重训。

Layout and numerical checks are automated. Actual rendered-image visual review remains pending and is not claimed complete by this publisher. Per-seed values, aggregate values, original-file hashes, and identity checks are recorded alongside the figure and in ../report/radius_complete.json. Physical checkpoint hashes are inherited from the previous audit; current raw-result and manifest identities are checked again.
'''
    (FIGURES / 'B1_radius_extended_caption.md').write_text(caption)
    (FIGURES / 'B1_radius_extended_include.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figure_B1_neighborhood_extended.pdf}
\caption{Formal three-seed training-neighborhood diagnostics: mean positive loss increase, accuracy, and raw empirical Fisher trace. Curves and shading show the mean and one sample standard deviation across training seeds. Loss and accuracy use 1,259 examples and 100 paired directions at 81 radii. Fisher uses a fixed 256-example subset and 16 paired directions at 16 measured radii, with no EWC scaling.}
\label{fig:rq2-formal-neighborhood-extended}
\end{figure*}
''')
    outputs = [p for p in FIGURES.iterdir() if p.name.startswith(('B1_', FIGURE_NAME)) and p.is_file()]
    complete = dict(metadata, status='complete', figure=str(FIGURES / f'{FIGURE_NAME}.pdf'),
                    output_files=[receipt(p) for p in sorted(outputs)],
                    rendered_visual_review='pending actual rendered-image inspection',
                    automatic_numeric_and_layout_checks='passed')
    # The completion receipt is the final write, after all validation and exports.
    atomic_json(REPORT / 'radius_complete.json', complete)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Validate available completion; never write artifacts (default)')
    mode.add_argument('--execute', action='store_true', help='Publish only when all six complete jobs pass validation')
    args = parser.parse_args(argv)
    try:
        coverage = inventory()
        if coverage['complete_job_files'] != 6:
            print(json.dumps(coverage, indent=2))
            return 2 if args.execute else 0
        validated = validate_all()
        if not args.execute:
            print(json.dumps(validated[-1], indent=2, allow_nan=False))
            return 0
        result = publish(*validated)
        print(json.dumps(dict(status=result['status'], validated_jobs=6, figure=result['figure'],
                              receipt=str(REPORT / 'radius_complete.json'),
                              rendered_visual_review=result['rendered_visual_review']), indent=2))
        return 0
    except (AssertionError, ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps(dict(status='invalid', error=f'{type(exc).__name__}: {exc}',
                              complete_receipt_written=False), indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
