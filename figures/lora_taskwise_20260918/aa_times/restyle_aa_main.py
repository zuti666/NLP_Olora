#!/usr/bin/env python3
"""Render the requested AA main figure with larger axis labels and FS-EWC alias.

Only presentation changes: original data, task-macro metric and bands are reused.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plot_lora_aa as base

HERE = Path(__file__).resolve().parent


def tighten_y_limits(ax):
    """Fit all plotted curves and bands, with 3% padding and outward rounding."""
    values = [line.get_ydata() for line in ax.lines]
    values.extend(path.vertices[:, 1] for band in ax.collections
                  for path in band.get_paths())
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    padding = max(0.03 * (high - low), 0.2)
    lower = float(np.floor((low - padding) * 2) / 2)
    upper = float(np.ceil((high + padding) * 2) / 2)
    if low >= 0:
        lower = max(0.0, lower)
    if high <= 100:
        upper = min(100.0, upper)
    previous = list(ax.get_ylim())
    ax.set_ylim(lower, upper)
    assert lower <= low <= high <= upper
    return {'previous_y_limits': previous, 'full_curve_and_band_range': [low, high],
            'y_limits': [lower, upper], 'padding_fraction': 0.03,
            'all_curves_and_bands_visible': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'output')
    parser.add_argument('--dataset-order', nargs=4, choices=list(base.DATASETS),
                        default=list(base.DATASETS), help='Left-to-right dataset order.')
    parser.add_argument('--stem', default='lora_taskwise_aa',
                        help='Output filename stem for PDF, PNG, SVG and metadata.')
    parser.add_argument('--compact', action='store_true',
                        help='Reduce the figure and axes height; preserve data and y limits.')
    parser.add_argument('--tight-y', action='store_true',
                        help='Tighten y limits around all curves and bands with 3%% padding.')
    parser.add_argument('--exclude-inflora', action='store_true',
                        help='Hide InfLoRA while preserving the original axis limits and styles.')
    args = parser.parse_args()
    if len(set(args.dataset_order)) != len(base.DATASETS):
        parser.error('--dataset-order must contain each dataset exactly once')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, groups, records = base.load_runs(HERE/'sources.json')
    base.configure_times()
    figsize = (13.4, 2.95) if args.compact else (13.4, 3.65)
    layout = dict(left=0.05, right=0.996, top=0.88 if args.compact else 0.90,
                  bottom=0.29 if args.compact else 0.24, wspace=0.36)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    plotted_methods = [method for method in base.METHODS
                       if not (args.exclude_inflora and method == 'InfLoRA')]
    label_map = {method: ('FS-EWC' if method == 'GAM-EWC+Merge' else method)
                 for method in base.METHODS}
    display_labels = [label_map[method] for method in plotted_methods]
    y_range_checks = {}
    for ax, dataset in zip(axes, args.dataset_order):
        legend = base.plot_dataset(ax, groups, dataset, metric='aa', single_seed_band=0.2)
        if args.tight_y:
            y_range_checks[dataset] = tighten_y_limits(ax)
        # Set limits before removing artists so this variant retains the reference framing.
        if args.exclude_inflora:
            index = base.METHODS.index('InfLoRA')
            ax.lines[index].remove()
            ax.collections[index].remove()
            legend.remove()
            legend = ax.legend(handles=list(ax.lines), labels=display_labels,
                               loc='best', frameon=True, facecolor='white',
                               edgecolor='#dddddd', framealpha=0.94, fontsize=8.6,
                               handlelength=1.65, handletextpad=0.5,
                               borderpad=0.3, labelspacing=0.28)
        ax.set_xlabel('Number of learned tasks', fontsize=14)
        ax.set_ylabel('Average accuracy (%)', fontsize=14)
        ax.set_title(base.DATASETS[dataset][0], fontsize=13, fontweight='bold', pad=7)
        for line, label, legend_text in zip(ax.lines, display_labels, legend.get_texts()):
            line.set_label(label)
            legend_text.set_text(label)
    ewc_counts = '/'.join(str(len(groups[(dataset, 'EWCLoRA')]))
                          for dataset in args.dataset_order)
    fs_counts = '/'.join(str(len(groups[(dataset, 'GAM-EWC+Merge')]))
                         for dataset in args.dataset_order)
    single_seed_labels = 'Vanilla LoRA, SD-LoRA' if args.exclude_inflora else 'Vanilla LoRA, InfLoRA, SD-LoRA'
    note = (f'{single_seed_labels}: n=1; fixed ±0.2 pp illustrative band (not SD/CI). '
            'Other bands: mean ± sample SD.\n'
            f'EWCLoRA n={ewc_counts}; FS-EWC n={fs_counts}. Seen tasks weighted equally; '
            'historical method-specific training recipes.')
    fig.text(0.515, 0.025, note, ha='center', fontsize=9)
    fig.subplots_adjust(**layout)
    # Validate data identity and requested presentation on the actual plot objects.
    for ax, dataset in zip(axes, args.dataset_order):
        assert ax.get_xlabel() == 'Number of learned tasks'
        assert ax.get_ylabel() == 'Average accuracy (%)'
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 14
        assert ax.title.get_fontweight() == 'bold'
        assert [t.get_text() for t in ax.get_legend().get_texts()] == display_labels
        assert len(ax.lines) == len(ax.collections) == len(plotted_methods)
        for line, method in zip(ax.lines, plotted_methods):
            expected, _ = base.summarize(groups[(dataset, method)], 'aa')
            np.testing.assert_array_equal(line.get_ydata(), expected)
            assert line.get_linestyle() == '-'
    y_limits = {dataset: list(ax.get_ylim())
                for ax, dataset in zip(axes, args.dataset_order)}
    base.save(fig, args.output/args.stem)
    metadata = {'y_label':'Average accuracy (%)', 'x_label':'Number of learned tasks',
        'axis_label_fontsize_pt':14, 'title_fontsize_pt':13, 'title_weight':'bold',
        'font_family':'Times New Roman', 'legend_labels':display_labels,
        'display_alias':{'FS-EWC':'GAM-EWC+Merge'},
        'display_alias_only':'Original source runs, method identifiers and numerical values are retained.',
        'single_seed_band_half_width_pp':0.2,
        'single_seed_band':'Illustrative, not an estimate of SD, variance or CI.',
        'raw_logs':len(records), 'numerical_curve_identity_check':'passed',
        'dataset_order':args.dataset_order, 'output_stem':args.stem,
        'compact':args.compact, 'figsize_inches':list(figsize),
        'subplot_layout':layout, 'y_limits':y_limits,
        'tight_y':args.tight_y, 'y_range_checks':y_range_checks,
        'plotted_methods':plotted_methods,
        'excluded_methods':['InfLoRA'] if args.exclude_inflora else [],
        'axis_limits_basis_methods':list(base.METHODS),
        'plotted_raw_logs':sum(r['method'] in plotted_methods for r in records)}
    (args.output/f'{args.stem}_display.json').write_text(json.dumps(metadata, indent=2))
    print('Saved AA main figure: axis labels 14pt, bold titles, clean legends, FS-EWC display alias.')


if __name__ == '__main__':
    main()
