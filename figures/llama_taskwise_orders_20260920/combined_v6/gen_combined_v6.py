"""Compact 2 x 3 figure, derived from methodcolors_v5; real data unchanged."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_plot_style import plt, LABELS
from matplotlib.lines import Line2D

MODELS = ['Llama-3.2-1B', 'Llama-3.2-3B']
METHOD_COLORS = {
    'SeqLoRA': '#667085', 'IncLoRA': '#0072B2', 'OLoRA+GAM': '#009E73',
    'TreeLoRA': '#9467BD', 'EWCLoRA': '#D89000', 'FS-EWCLoRA': '#E31A1C',
}
DATASET_STYLES = {
    'dbpedia': '-', 'amazon': (0, (5, 2.4)),
    'yahoo': (0, (1, 1.8)), 'agnews': (0, (5, 1.8, 1, 1.8)),
}
STYLE_NAMES = {'dbpedia': 'solid', 'amazon': 'dashed', 'yahoo': 'dotted', 'agnews': 'dash-dot'}
DATASET_MARKERS = {'dbpedia': 'o', 'amazon': 's', 'yahoo': '^', 'agnews': 'D'}
MARKER_SIZE_PT = 2.8
STEM = 'llama_1b_3b_taskwise_orders_combined_v6'


def draw():
    with (HERE / 'taskwise_accuracy.csv').open(encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    display_labels = json.loads((HERE / 'display_labels.json').read_text())
    # Use identical limits for the two models in each task order. Both ends
    # include >= 2 percentage points of padding; no observation is cropped.
    limits = {}
    for order in range(1, 4):
        values = [float(r['accuracy_percent']) for r in data if int(r['order']) == order]
        limits[order] = (math.floor(min(values) - 2), math.ceil(max(values) + 2))

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), sharey='col')
    fig.subplots_adjust(left=.075, right=.99, bottom=.11, top=.865, wspace=.225, hspace=.46)
    plotted, panels = [], []
    for row, model in enumerate(MODELS):
        for col, order in enumerate(range(1, 4)):
            ax = axes[row, col]
            sequence = manifest['orders'][str(order)]
            ax.set_xlim(.87, 4.18)
            ax.set_ylim(*limits[order])
            ax.set_yticks([v for v in [30, 50, 70, 90] if limits[order][0] <= v <= limits[order][1]])
            # Keep all tick labels visible because the range differs by order.
            ax.tick_params(labelleft=True, axis='both', labelsize=8.5, length=2.4, width=.6, pad=2)
            for dataset in DATASET_STYLES:
                for method, color in METHOD_COLORS.items():
                    rows = sorted([r for r in data if r['model'] == model and r['method'] == method
                                   and int(r['order']) == order and r['evaluated_dataset'] == dataset],
                                  key=lambda r: int(r['stage']))
                    xs = [int(r['stage']) for r in rows]
                    ys = [float(r['accuracy_percent']) for r in rows]
                    assert xs == list(range(sequence.index(dataset) + 1, 5))
                    singleton = len(xs) == 1
                    ax.plot(xs, ys, color=color, linestyle='None' if singleton else DATASET_STYLES[dataset],
                            linewidth=1.5 if method == 'FS-EWCLoRA' else .98,
                            marker=DATASET_MARKERS[dataset], markersize=MARKER_SIZE_PT,
                            markeredgewidth=.4, markerfacecolor=color, markeredgecolor=color,
                            solid_capstyle='butt', dash_capstyle='butt',
                            zorder=5 if method == 'FS-EWCLoRA' else 3)
                    plotted.append(dict(model=model, order=order, method=method,
                                        display_label=display_labels[method], dataset=dataset, x=xs, y=ys,
                                        color=color, dataset_line_style=STYLE_NAMES[dataset],
                                        marker=DATASET_MARKERS[dataset], markersize_pt=MARKER_SIZE_PT,
                                        display='single_marker_no_line' if singleton else 'measured_polyline_with_markers'))
            short_labels = dict(LABELS, dbpedia='DBP')
            ax.set_xticks([1, 2, 3, 4], [short_labels[t] for t in sequence])
            ax.grid(axis='y', color='.87', linewidth=.5, zorder=0)
            ax.grid(axis='x', color='.94', linewidth=.4, zorder=0)
            panel_label = f'({chr(97 + row * 3 + col)}) Llama-{model.rsplit("-", 1)[-1]}–Order {order}'
            ax.set_xlabel(panel_label, fontsize=10.8, fontweight='bold', labelpad=5.5)
            if col == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=9.5, labelpad=3)
            panels.append(dict(model=model, order=order, label=panel_label, ylim=list(limits[order]),
                               measured_min=min(y for s in plotted if s['model'] == model and s['order'] == order for y in s['y']),
                               measured_max=max(y for s in plotted if s['model'] == model and s['order'] == order for y in s['y'])))

    method_handles = [Line2D([], [], color=color, linewidth=1.8 if m == 'FS-EWCLoRA' else 1.25,
                            label=display_labels[m]) for m, color in METHOD_COLORS.items()]
    method_legend = fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(.55, .993),
                              ncol=6, frameon=False, handlelength=1.45, columnspacing=.8,
                              handletextpad=.35, borderaxespad=0, fontsize=9)
    for label in method_legend.get_texts():
        if label.get_text() == 'FS-EWCLoRA':
            label.set_color(METHOD_COLORS['FS-EWCLoRA'])
            label.set_fontweight('bold')
    dataset_handles = [Line2D([], [], color='.15', linestyle=DATASET_STYLES[d], linewidth=1.15,
                             marker=DATASET_MARKERS[d], markersize=MARKER_SIZE_PT, markeredgewidth=.4,
                             label=LABELS[d]) for d in DATASET_STYLES]
    dataset_legend = fig.legend(handles=dataset_handles, loc='upper center', bbox_to_anchor=(.55, .939),
                               ncol=4, frameon=False, handlelength=2.8, columnspacing=1.4,
                               handletextpad=.5, borderaxespad=0, fontsize=9)
    fig.text(.075, .969, 'Method', fontsize=9.2, va='center')
    fig.text(.075, .915, 'Dataset', fontsize=9.2, va='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for legend in [method_legend, dataset_legend]:
        bbox = legend.get_window_extent(renderer)
        assert all(bbox.y0 > ax.get_window_extent(renderer).y1 for ax in axes.flat)
    assert len(plotted) == 144 and sum(len(s['x']) for s in plotted) == 360
    assert sum(len(s['x']) == 1 for s in plotted) == 36
    for ax in axes.flat:
        for line in ax.lines:
            assert all(ax.get_ylim()[0] < y < ax.get_ylim()[1] for y in line.get_ydata())
            if len(line.get_xdata()) == 1:
                assert line.get_linestyle() == 'None'
    for ext in ['pdf', 'png', 'svg']:
        path = HERE / f'{STEM}.{ext}'
        fig.savefig(path, facecolor='white')
        print(path)
    (HERE / f'{STEM}_plotted_data.json').write_text(json.dumps(plotted, indent=2) + '\n')
    check = dict(figure_size_inches=[7.2, 4.35], layout='2 rows x 3 columns', panels=panels,
                 method_legend_count=1, dataset_legend_count=1, legends_above_axes=True,
                 x_axis_training_stage_title=False, shared_y_limits='within each order, across both models',
                 y_limits_rule='floor(min - 2), ceil(max + 2) per order, both models pooled',
                 cropped_observations=0, series=144, measured_points=360, singleton_markers=36,
                 data_sha256=hashlib.sha256((HERE / 'taskwise_accuracy.csv').read_bytes()).hexdigest())
    (HERE / f'{STEM}_style_check.json').write_text(json.dumps(check, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    draw()
