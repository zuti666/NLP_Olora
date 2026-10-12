#!/usr/bin/env python3
"""Plot the updated four-method figure using one consistent metric per figure.

python plot_updated_lora_aa.py --metric both
Raw data are re-read from hash-pinned logs; the six new matrices are cross-checked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lora-updated-20260925-mpl')
HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/'lora_taskwise_20260918/aa_times'
sys.path.insert(0, str(REFERENCE))
import plot_lora_aa as base
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

METHODS = ['Vanilla LoRA', 'SD-LoRA', 'EWCLoRA', 'GAM-EWC+Merge']
LABELS = ['Vanilla LoRA', 'SD-LoRA', 'EWCLoRA', 'FS-EWC']
METRICS = {'sample': 'sample_accuracy', 'task': 'aa'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_updated_runs(manifest_path=HERE/'sources.json'):
    """Read original logs and validate the latest six runs against their raw JSON."""
    manifest, groups, records = base.load_runs(manifest_path)
    assert len(records) == 41
    checks = []
    for run in records:
        if 'metrics_path' not in run:
            continue
        path = Path(run['metrics_path'])
        assert digest(path) == run['metrics_sha256'], f'Changed metrics: {path}'
        metrics = json.loads(path.read_text())
        matrix = np.asarray(metrics['cnn']['matrices']['final'], dtype=float)
        assert matrix.shape == (10, 10)
        for stage, row in enumerate(run['task_accuracies'], 1):
            np.testing.assert_array_equal(matrix[stage-1, :stage], row)
        assert np.isnan(matrix[np.triu_indices(10, k=1)]).all()
        method_types = {step['method_type'] for step in metrics['mechanism']['steps'].values()}
        assert method_types == {'as1_normfisher_gam_intask_damp_weightmergethenfisher'}
        sample = np.asarray(run['sample_accuracy'])
        np.testing.assert_allclose(sample[-1], run['expected_sample_faa'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(sample.mean(), run['expected_sample_aaa'], atol=1e-9, rtol=0)
        checks.append({'dataset': run['dataset'], 'seed': run['seed'],
                       'log': run['log'], 'metrics_path': str(path),
                       'matrix_matches_log': True, 'sample_metrics_match_supplied_summary': True,
                       'actual_method_type': sorted(method_types)[0]})
    assert len(checks) == 6
    for dataset in ['imagenetr', 'imageneta']:
        assert [r['seed'] for r in groups[(dataset, 'GAM-EWC+Merge')]] == [0, 1, 2]
    return manifest, groups, records, checks


def export_data(output, groups, records, order):
    stage_rows, final_rows, raw_rows = [], [], []
    for dataset in order:
        for method in METHODS:
            runs = groups[(dataset, method)]
            for metric, key in METRICS.items():
                values = np.asarray([run[key] for run in runs])
                mean, sd = base.summarize(runs, key)
                aaa = values.mean(axis=1)
                final_rows.append({'dataset': dataset, 'method': method, 'metric': metric,
                                   'n': len(runs), 'FAA': float(mean[-1]),
                                   'FAA_sample_sd': float(sd[-1]) if len(runs)>1 else '',
                                   'AAA': float(aaa.mean()),
                                   'AAA_sample_sd': float(aaa.std(ddof=1)) if len(runs)>1 else ''})
                for t, (m, s) in enumerate(zip(mean, sd), 1):
                    band = float(s) if len(runs)>1 else 0.2
                    stage_rows.append({'dataset': dataset, 'method': method, 'metric': metric,
                                       'stage': t, 'n': len(runs), 'mean': float(m),
                                       'sample_sd': float(s) if len(runs)>1 else '',
                                       'band_kind': 'sample_sd' if len(runs)>1 else 'illustrative_not_sd',
                                       'band_half_width': band, 'lower': float(m-band), 'upper': float(m+band)})
                for run in runs:
                    curve = run[key]
                    raw_rows.append({'dataset': dataset, 'method': method, 'metric': metric,
                                     'seed': run['seed'], 'FAA': curve[-1],
                                     'AAA': float(np.mean(curve)), 'log': run['log']})
    base.write_csv(output/'stage_statistics.csv', stage_rows)
    base.write_csv(output/'summary_metrics.csv', final_rows)
    base.write_csv(output/'per_seed_metrics.csv', raw_rows)
    (output/'parsed_raw_runs.json').write_text(json.dumps(records, indent=2)+'\n')
    return final_rows


def plot_updated_dataset(ax, groups, dataset, metric, reference_limits):
    """Use the reference styling; extend limits only if a new curve/band needs it."""
    key = METRICS[metric]
    steps = base.DATASETS[dataset][1]
    x = np.arange(1, steps+1)
    low, high = np.inf, -np.inf
    for method, label in zip(METHODS, LABELS):
        index = base.METHODS.index(method)  # Preserve method colors after removing InfLoRA.
        runs = groups[(dataset, method)]
        mean, sd = base.summarize(runs, key)
        width = sd if len(runs)>1 else np.full_like(mean, 0.2)
        ax.plot(x, mean, color=base.COLORS[index], marker=base.MARKERS[index],
                linestyle='-', linewidth=1.5, markersize=4, label=label, zorder=3)
        ax.fill_between(x, mean-width, mean+width, color=base.COLORS[index],
                        alpha=0.14, linewidth=0, zorder=1)
        low = min(low, float(np.min(mean-width)))
        high = max(high, float(np.max(mean+width)))
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(), mean)
    lo, hi = reference_limits
    pad = max((high-low)*0.03, 0.2)
    if low < lo:
        lo = float(np.floor((low-pad)*2)/2)
    if high > hi:
        hi = float(np.ceil((high+pad)*2)/2)
    assert lo <= low <= high <= hi
    ax.set_ylim(lo, hi)
    ax.set_title(base.DATASETS[dataset][0], fontsize=13, fontweight='bold', pad=7)
    ax.set_xlabel('Number of learned tasks', fontsize=14)
    ax.set_ylabel('Average accuracy (%)', fontsize=14)
    ax.tick_params(labelsize=10, pad=3.5)
    ax.set_xticks(x)
    ax.set_xlim(0.8, steps+0.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, color='#d8d8d8', linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='#dddddd',
              framealpha=0.94, fontsize=8.6, handlelength=1.65,
              handletextpad=0.5, borderpad=0.3, labelspacing=0.28)
    return {'limits': [lo, hi], 'full_band_range': [low, high], 'all_bands_visible': True}


def plot_updated_figure(groups, order, reference_style, metric, output):
    """Render the no-InfLoRA 1x4 figure with a single consistent metric."""
    fig, axes = plt.subplots(1, 4, figsize=reference_style['figsize_inches'])
    ranges = {dataset: plot_updated_dataset(ax, groups, dataset, metric,
                                           reference_style['y_limits'][dataset])
              for ax, dataset in zip(axes, order)}
    ewc_n = '/'.join(str(len(groups[(d, 'EWCLoRA')])) for d in order)
    fs_n = '/'.join(str(len(groups[(d, 'GAM-EWC+Merge')])) for d in order)
    weighting = 'Samples weighted equally' if metric == 'sample' else 'Seen tasks weighted equally'
    note = ('Vanilla LoRA, SD-LoRA: n=1; fixed ±0.2 pp illustrative band (not SD/CI). Other bands: mean ± sample SD.\n'
            f'EWCLoRA n={ewc_n}; FS-EWC n={fs_n}. {weighting}; historical method-specific training recipes.')
    fig.text(0.515, 0.025, note, ha='center', fontsize=9)
    fig.subplots_adjust(**reference_style['subplot_layout'])
    fig.canvas.draw()
    for ax in axes:
        assert len(ax.lines) == len(ax.collections) == 4
        assert [text.get_text() for text in ax.get_legend().get_texts()] == LABELS
    stem = f'lora_taskwise_{metric}_aa_updated'
    base.save(fig, output/stem)
    metadata = {'metric': metric, 'dataset_order': order, 'legend_labels': LABELS,
                'figsize_inches': reference_style['figsize_inches'],
                'subplot_layout': reference_style['subplot_layout'], 'y_ranges': ranges,
                'plotted_methods': METHODS, 'excluded_methods': ['InfLoRA'],
                'updated_group_seeds': {'imagenetr': [0,1,2], 'imageneta': [0,1,2]},
                'updated_group_gam_grad_norm_rho': 1.0, 'raw_curves_unchanged_by_plot': True,
                'display_alias': {'FS-EWC': 'GAM-EWC+Merge / latest ImageNet-R,A GAM-EWC+A'},
                'footnote': note, 'source_manifest': str(HERE/'sources.json')}
    (output/f'{stem}_display.json').write_text(json.dumps(metadata, indent=2)+'\n')
    caption = ('Current-stage accuracy on ImageNet-R, ImageNet-A, DomainNet and CIFAR-100. '
               + ('All curves use sample-weighted CNN top-1 accuracy. ' if metric=='sample'
                  else 'All curves use equal-weight means over seen tasks. ')
               + 'The FS-EWC display label denotes GAM-EWC+Merge; ImageNet-R/A use the latest '
               'GAM-EWC+A outer-radius-1.0 runs, seeds 0, 1 and 2. '
               f'EWCLoRA uses n={ewc_n} and FS-EWC n={fs_n} seeds in panel order. '
               'Multi-seed bands show sample standard deviations. Vanilla LoRA and SD-LoRA '
               'have one seed and illustrative $\\pm0.2$ percentage-point bands, not uncertainty estimates. '
               'Historical method-specific training recipes differ.')
    (output/f'{stem}.tex').write_text(
        '\\begin{figure*}[t]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:lora-updated-{metric}}}\n\\end{{figure*}}\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=['sample', 'task', 'both'], default='both')
    parser.add_argument('--manifest', type=Path, default=HERE/'sources.json')
    parser.add_argument('--output', type=Path, default=HERE/'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, groups, records, checks = load_updated_runs(args.manifest)
    style = json.loads(Path(manifest['style_reference']).read_text())
    font = base.configure_times()
    summary = export_data(args.output, groups, records, manifest['dataset_order'])
    metrics = list(METRICS) if args.metric=='both' else [args.metric]
    plots = [plot_updated_figure(groups, manifest['dataset_order'], style, m, args.output) for m in metrics]
    (args.output/'source_verification.json').write_text(json.dumps({
        'hash_checked_raw_logs': len(records), 'new_run_checks': checks,
        'source_manifest_sha256': digest(args.manifest), 'font': font, 'plots': plots}, indent=2)+'\n')
    for row in summary:
        if row['method']=='GAM-EWC+Merge' and row['dataset'] in ['imagenetr','imageneta']:
            print(row)
    print(f'Saved {len(plots)} figures under {args.output}')


if __name__ == '__main__':
    main()
