#!/usr/bin/env python3
"""Render the table-ordered 1x5 overview, with explicitly defined shading.

The frozen data contain n=1 per method/dataset. An optional fixed-width band
is illustrative only and is never presented as SD, SE or a confidence interval.
"""
import argparse
import json
from pathlib import Path

from paper_plot_style import HERE, DATASETS, METHODS, STYLES, configure
from plot_taskwise_aa import load_curves, digest
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

OUTPUT = HERE / 'output'
SETTINGS = HERE / 'horizontal_settings.json'
DEFAULT_SETTINGS = {'band_type': 'none', 'band_half_width_pp': 0.0,
                    'note': 'No uncertainty estimate is available from one run.'}


def render_overview(records=None, settings=None, preview=False):
    font = configure()
    records = load_curves() if records is None else records
    if settings is None:
        settings = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else DEFAULT_SETTINGS
    half = float(settings['band_half_width_pp'])
    assert settings['band_type'] in {'none', 'illustrative_fixed_width'}
    assert 0 <= half <= 5
    assert (half == 0) == (settings['band_type'] == 'none')
    fig, axes = plt.subplots(1, 5, figsize=(16.8, 4.1))
    fig.subplots_adjust(left=.042, right=.996, top=.89, bottom=.34, wspace=.34)
    panel_checks = []
    for ax, dataset in zip(axes, DATASETS):
        tasks = records[(dataset, METHODS[0])]['tasks']
        for method in METHODS:
            r = records[(dataset, method)]
            x, y = np.array(r['task_index']), np.array(r['aa'])
            if half:
                # Same illustrative rule as the single-run reference, not a statistic.
                ax.fill_between(x, np.maximum(0, y-half), np.minimum(100, y+half),
                                color=STYLES[method]['color'], alpha=.14,
                                linewidth=0, zorder=1)
            ax.plot(x, y, label=method, **STYLES[method],
                    linestyle='-', linewidth=1.5, markersize=4, zorder=3)
        ax.set_title(DATASETS[dataset][0], fontsize=13, fontweight='bold', pad=7)
        ax.set_xlabel('Number of learned tasks', fontsize=14, labelpad=5)
        ax.set_ylabel('Average accuracy (%)', fontsize=14, labelpad=5)
        ax.set_xticks(range(1, tasks+1))
        ax.set_xlim(.8, tasks+.2)
        ax.margins(y=.09)
        low, high = ax.get_ylim()
        ax.set_ylim(max(0, low), min(100, high))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=10, pad=3.5)
        ax.set_axisbelow(True)
        ax.grid(color='#d8d8d8', linewidth=.55, alpha=.7)
        for line, method in zip(ax.lines, METHODS):
            np.testing.assert_array_equal(line.get_ydata(), records[(dataset, method)]['aa'])
            np.testing.assert_array_equal(line.get_xdata(), records[(dataset, method)]['task_index'])
        assert len(ax.collections) == (7 if half else 0)
        panel_checks.append({'dataset': dataset, 'title': DATASETS[dataset][0],
                             'curves': 7, 'tasks': tasks, 'n_per_curve': 1,
                             'curve_values_match_frozen_data': True,
                             'shaded_bands': len(ax.collections)})
    legend = fig.legend(list(axes[0].lines), METHODS, loc='lower center',
                        bbox_to_anchor=(.52, .105), ncol=7,
                        frameon=True, facecolor='white', edgecolor='#dddddd',
                        framealpha=.94, fontsize=10, handlelength=1.65,
                        handletextpad=.5, borderpad=.4, columnspacing=1.8)
    first = ('All curves: n=1. Shading: fixed ±' + f'{half:g}' +
             ' pp illustrative band (not SD, SE, or CI).') if half else (
             'All curves: n=1; no uncertainty band is estimated.')
    note = fig.text(.52, .024, first + '\n'
                    'Seen tasks weighted equally; archived method-specific configurations. '
                    'EWCLoRA: highest-FAA eligible archived run per dataset.',
                    ha='center', va='bottom', fontsize=8.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    note_box = note.get_window_extent(renderer)
    assert not legend_box.overlaps(note_box)
    for ax in axes:
        assert not legend_box.overlaps(ax.get_tightbbox(renderer))
    assert fig.bbox.contains(*legend_box.p0) and fig.bbox.contains(*legend_box.p1)
    assert fig.bbox.contains(*note_box.p0) and fig.bbox.contains(*note_box.p1)
    assert [ax.get_title() for ax in axes] == [d[0] for d in DATASETS.values()]
    assert [text.get_text() for text in legend.get_texts()] == METHODS
    OUTPUT.mkdir(exist_ok=True)
    stem = 'horizontal_layout_preview' if preview else 'all_datasets_taskwise_aa'
    files = []
    for ext in ['pdf', 'png', 'svg']:
        path = OUTPUT / f'{stem}.{ext}'
        meta = {'metadata': {'Title': 'Aircraft, Cars, CUB-200, Flowers, 5Datasets: taskwise AA',
                             'Subject': first}} if ext == 'pdf' else {}
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.055, **meta)
        files.append({'path': str(path.relative_to(HERE)), 'sha256': digest(path)})
    plt.close(fig)
    (OUTPUT / f'{stem}_verification.json').write_text(json.dumps({
        'layout': '1 row x 5 columns, single page', 'font': font,
        'panel_order': [d[0] for d in DATASETS.values()],
        'band_settings': settings, 'legend_overlaps_data': False,
        'titles_requested_by_user': True,
        'data_sha256': digest(HERE / 'data/curves.json'),
        'panels': panel_checks, 'files': files,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved {stem}: 1x5, band={settings["band_type"]}, half-width={half:g} pp.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='Save layout preview without changing final overview.')
    args = parser.parse_args()
    render_overview(preview=args.preview)
