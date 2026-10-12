#!/usr/bin/env python3
"""Render verified per-stage task-macro AA from the archived run matrices.

Usage: python plot_taskwise_aa.py [aircraft cars196 cub200 flowers het5]
No training or original experiment directories are needed to render frozen data.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from paper_plot_style import HERE, DATASETS, METHODS, STYLES, configure
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

OUTPUT = HERE / 'output'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_curves():
    path = HERE / 'data/curves.json'
    curves = json.loads(path.read_text())
    records = {(r['dataset'], r['method']): r for r in curves}
    assert len(records) == len(curves) == 35
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for entry in manifest['runs']:
        frozen = HERE / entry['frozen_matrix']
        assert digest(frozen) == entry['frozen_matrix_sha256'], frozen
        assert json.loads(frozen.read_text()) == records[(entry['dataset'], entry['method'])]
    for r in curves:
        matrix = r['matrix_lower_triangle']
        assert all(len(row) == i + 1 for i, row in enumerate(matrix))
        actual = [statistics.mean(row) for row in matrix]
        np.testing.assert_array_equal(actual, r['aa'])
        assert r['n_runs'] == 1 and len(actual) == r['tasks']
        assert np.isfinite(actual).all() and min(actual) >= 0 and max(actual) <= 100
        assert r['faa'] == actual[-1] and r['aaa_stage'] == statistics.mean(actual)
    return records


def make_figure(dataset, records):
    fig, ax = plt.subplots(figsize=(5.6, 4.35))
    fig.subplots_adjust(left=.14, right=.982, bottom=.27, top=.97)
    tasks = records[(dataset, METHODS[0])]['tasks']
    for method in METHODS:
        r = records[(dataset, method)]
        ax.plot(r['task_index'], r['aa'], label=method, **STYLES[method],
                linestyle='-', linewidth=1.5, markersize=4, zorder=3)
    ax.set_xlabel('Number of learned tasks', labelpad=6)
    ax.set_ylabel('Average accuracy (%)', labelpad=7)
    ax.set_xticks(range(1, tasks + 1))
    ax.set_xlim(.8, tasks + .2)
    ax.margins(y=.09)
    bottom, top = ax.get_ylim()
    ax.set_ylim(max(0, bottom), min(100, top))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=10, pad=3.5)
    ax.set_axisbelow(True)
    ax.grid(color='#d8d8d8', linewidth=.55, alpha=.7)

    # Seven entries sit below the axes, so the shared legend never hides a curve.
    # Matplotlib fills columns first; reorder handles to preserve table order by row.
    order = [0, 4, 1, 5, 2, 6, 3]
    handles = [ax.lines[i] for i in order]
    legend = fig.legend(handles, [METHODS[i] for i in order],
                        loc='lower center', bbox_to_anchor=(.54, .025), ncol=4,
                        frameon=True, facecolor='white', edgecolor='#dddddd',
                        framealpha=.94, fontsize=8.6, handlelength=1.65,
                        handletextpad=.5, borderpad=.4, labelspacing=.6,
                        columnspacing=1.25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    xlabel_box = ax.xaxis.label.get_window_extent(renderer)
    assert not legend_box.overlaps(ax.get_window_extent(renderer))
    assert not legend_box.overlaps(xlabel_box)
    assert fig.bbox.contains(*legend_box.p0) and fig.bbox.contains(*legend_box.p1)
    assert not ax.get_title()
    assert [line.get_label() for line in ax.lines] == METHODS
    for line, method in zip(ax.lines, METHODS):
        r = records[(dataset, method)]
        np.testing.assert_array_equal(line.get_xdata(), r['task_index'])
        np.testing.assert_array_equal(line.get_ydata(), r['aa'])
    assert len(ax.collections) == 0  # Single-run curves have no uncertainty bands.
    return fig, ax


def main(datasets=None):
    font = configure()
    records = load_curves()
    if datasets is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('datasets', nargs='*', choices=list(DATASETS))
        args = parser.parse_args()
        datasets = args.datasets or list(DATASETS)
    OUTPUT.mkdir(exist_ok=True)
    checks = []
    combined = PdfPages(OUTPUT / 'all_datasets_taskwise_aa_multipage.pdf') if datasets == list(DATASETS) else None
    try:
        for dataset in datasets:
            fig, ax = make_figure(dataset, records)
            name, stem = DATASETS[dataset]
            files = []
            for extension in ['pdf', 'png', 'svg']:
                path = OUTPUT / f'{stem}_taskwise_aa.{extension}'
                kwargs = {'metadata': {'Title': f'{name}: task-macro stage accuracy',
                                      'Subject': 'One archived run per method; differing documented configurations.'}} if extension == 'pdf' else {}
                fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.055, **kwargs)
                assert path.stat().st_size > 0
                files.append({'path': str(path.relative_to(HERE)), 'sha256': digest(path)})
            if combined is not None:
                combined.savefig(fig, bbox_inches='tight', pad_inches=.055)
            checks.append({'dataset': dataset, 'tasks': records[(dataset, METHODS[0])]['tasks'],
                           'lines': 7, 'all_line_values_equal_frozen_row_means': True,
                           'legend_overlaps_data': False, 'uncertainty_bands': 0,
                           'axis_label_fontsize_pt': 14, 'ylim': list(ax.get_ylim()),
                           'files': files})
            plt.close(fig)
            print(f'Saved {name}: 7 curves, PDF/PNG/SVG.')
    finally:
        if combined is not None:
            combined.close()
    suffix = '' if datasets == list(DATASETS) else '_' + '_'.join(datasets)
    (OUTPUT / f'render_verification{suffix}.json').write_text(json.dumps({
        'matplotlib_version': matplotlib.__version__, 'font': font,
        'data_sha256': digest(HERE / 'data/curves.json'),
        'metric': 'AA_t = mean(a[t, 1:t]), including current task; tasks weighted equally.',
        'replication': 'n=1 archived run per dataset/method; no uncertainty estimated.',
        'reference': 'lora_taskwise_aa.pdf (final restyled main figure)',
        'style_changes': ['Seven-entry legend below axes to avoid covering data.',
                          'Dataset names provided in captions, with no in-figure title.',
                          'OLoRA and CL-LoRA receive new colors/markers.',
                          'Actual GAM-EWCLoRA name retained; magenta/star style reused.',
                          'No illustrative bands or cross-run uncertainty bands.'],
        'legend_order_by_row': METHODS, 'styles': STYLES, 'figures': checks,
    }, ensure_ascii=False, indent=2) + '\n')
    if datasets == list(DATASETS):
        from plot_horizontal_aa import render_overview
        render_overview(records=records)


if __name__ == '__main__':
    main()
