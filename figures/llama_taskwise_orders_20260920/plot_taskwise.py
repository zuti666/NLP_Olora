"""Shared plotting implementation; scores are read from the extracted CSV."""
import csv
import json
from paper_plot_style import HERE, plt, COLORS, LABELS, METHODS, STYLES, save_figure
from matplotlib.lines import Line2D


def draw(model, seqlora_only=False):
    with (HERE/'taskwise_accuracy.csv').open(encoding='utf-8-sig') as f:
        data = [r for r in csv.DictReader(f) if r['model']==model]
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    available = {r['method'] for r in data}
    methods = ['SeqLoRA'] if seqlora_only else [m for m in METHODS if m in available]
    missing = [m for m in METHODS if m not in available] if not seqlora_only else []
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.65), sharey=True)
    fig.subplots_adjust(left=.073, right=.991, bottom=.23,
                        top=.84 if seqlora_only else .745, wspace=.18)
    render_checks = []
    for order, ax in enumerate(axes, 1):
        sequence = manifest['orders'][str(order)]
        for dataset in COLORS:
            for method in methods:
                rows = sorted([r for r in data if r['order']==str(order) and
                               r['evaluated_dataset']==dataset and r['method']==method],
                              key=lambda r:int(r['stage']))
                xs = [int(r['stage']) for r in rows]
                ys = [float(r['accuracy_percent']) for r in rows]
                assert xs == list(range(sequence.index(dataset)+1,5)), (model, order, dataset, method)
                style = STYLES[method]
                ax.plot(xs, ys, color=COLORS[dataset], **style,
                        linewidth=1.25 if method=='FS-EWCLoRA' or seqlora_only else .85,
                        markersize=5.0 if method=='FS-EWCLoRA' else 3.0,
                        markeredgewidth=.6, markerfacecolor='white' if method=='TreeLoRA' else COLORS[dataset],
                        alpha=1, zorder=4 if method=='FS-EWCLoRA' else 3)
                render_checks.append(dict(model=model, order=order, method=method,
                                          dataset=dataset, x=xs, y=ys))
        ax.set_xlim(.85,4.15)
        ax.set_ylim(20,100)
        ax.set_yticks([20,40,60,80,100])
        short_labels = dict(LABELS, dbpedia='DBP')
        ax.set_xticks([1,2,3,4], [f'{i}\n{short_labels[t]}' for i,t in enumerate(sequence,1)])
        ax.tick_params(axis='both', length=2.5, width=.6, pad=2)
        ax.grid(axis='y', color='.85', linewidth=.55, zorder=0)
        ax.grid(axis='x', color='.91', linewidth=.45, zorder=0)
        ax.set_xlabel('Training stage', labelpad=5)
        bounds = ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2, .072, f'({chr(96+order)}) Order {order}',
                 ha='center', va='center', fontsize=10)
    axes[0].set_ylabel('Accuracy (%)', labelpad=3)
    dataset_handles = [Line2D([],[],color=COLORS[d],linewidth=1.6,label=LABELS[d]) for d in COLORS]
    fig.legend(handles=dataset_handles, loc='upper center', bbox_to_anchor=(.535,.993),
               ncol=4, frameon=False, handlelength=1.9, columnspacing=1.9,
               title='Dataset', title_fontsize=8.5, borderaxespad=0)
    if not seqlora_only:
        method_handles = [Line2D([],[],color='.16',**STYLES[m],markersize=4.3,
                                 markerfacecolor='white' if m=='TreeLoRA' else '.16',
                                 linewidth=1,label=m) for m in methods]
        fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(.535,.877),
                   ncol=3, frameon=False, handlelength=2.5, columnspacing=1.8,
                   handletextpad=.6, borderaxespad=0, labelspacing=.55)
    if missing:
        fig.text(.53,.018,'OLoRA (1B): matching source unavailable; curve omitted.',
                 ha='center',va='bottom',fontsize=8,color='.25')
    size = '1b' if model.endswith('1B') else '3b'
    stem = f'llama_{size}_'+('seqlora_' if seqlora_only else '')+'taskwise_orders'
    save_figure(fig,stem)
    (HERE/f'{stem}_plotted_data.json').write_text(json.dumps(render_checks,indent=2)+'\n')
    plt.close(fig)
