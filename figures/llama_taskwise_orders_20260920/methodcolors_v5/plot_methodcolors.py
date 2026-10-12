"""Version 5: compact dataset markers, singleton points, top legends; olra is an OLoRA+GAM display alias: method colors, dataset line styles.

Adapted from ../plot_taskwise.py. Reuses its data extraction, order masking,
axis structure, font settings, and source CSV; no existing output is written.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_plot_style import plt, LABELS, METHODS as BASE_METHODS
METHODS = ["OLoRA+GAM" if m == "OLoRA" else m for m in BASE_METHODS]
from matplotlib.lines import Line2D

METHOD_COLORS = {
    'SeqLoRA': '#667085',
    'IncLoRA': '#0072B2',
    'OLoRA+GAM': '#009E73',
    'TreeLoRA': '#9467BD',
    'EWCLoRA': '#D89000',
    'FS-EWCLoRA': '#E31A1C',
}
DATASET_STYLES = {
    'dbpedia': '-',
    'amazon': (0, (5, 2.4)),
    'yahoo': (0, (1, 1.8)),
    'agnews': (0, (5, 1.8, 1, 1.8)),
}
STYLE_NAMES = {'dbpedia': 'solid', 'amazon': 'dashed', 'yahoo': 'dotted', 'agnews': 'dash-dot'}
DATASET_MARKERS = {'dbpedia': 'o', 'amazon': 's', 'yahoo': '^', 'agnews': 'D'}
MARKER_SIZE_PT = 3.0


def draw(model):
    with (HERE/'taskwise_accuracy.csv').open(encoding='utf-8-sig') as f:
        data = [r for r in csv.DictReader(f) if r['model']==model]
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    display_labels = json.loads((HERE/'display_labels.json').read_text())
    methods = [m for m in METHODS if any(r['method']==m for r in data)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.85), sharey=True)
    fig.subplots_adjust(left=.073, right=.991, bottom=.23, top=.81, wspace=.18)
    plotted=[]
    singleton_count=0
    for order, ax in enumerate(axes, 1):
        sequence=manifest['orders'][str(order)]
        ax.set_xlim(.85,4.25)
        ax.set_ylim(20,100)
        ax.set_yticks([20,40,60,80,100])
        for dataset in DATASET_STYLES:
            for method in methods:
                rows=sorted([r for r in data if r['method']==method and r['order']==str(order)
                             and r['evaluated_dataset']==dataset],key=lambda r:int(r['stage']))
                xs=[int(r['stage']) for r in rows]
                ys=[float(r['accuracy_percent']) for r in rows]
                assert xs==list(range(sequence.index(dataset)+1,5))
                props=dict(color=METHOD_COLORS[method],
                           linestyle=DATASET_STYLES[dataset] if len(xs)>1 else 'None',
                           linewidth=1.6 if method=='FS-EWCLoRA' else 1.02,
                           marker=DATASET_MARKERS[dataset], markersize=MARKER_SIZE_PT,
                           markeredgewidth=.45, markerfacecolor=METHOD_COLORS[method],
                           markeredgecolor=METHOD_COLORS[method],
                           solid_capstyle='butt', dash_capstyle='butt',
                           zorder=5 if method=='FS-EWCLoRA' else 3)
                ax.plot(xs,ys,**props)
                if len(xs)==1: singleton_count+=1
                plotted.append(dict(model=model,order=order,method=method,dataset=dataset,
                                    x=xs,y=ys,color=METHOD_COLORS[method],
                                    display_label=display_labels[method],
                                    dataset_line_style=STYLE_NAMES[dataset],marker=DATASET_MARKERS[dataset],
                                    markersize_pt=MARKER_SIZE_PT,
                                    display='single_marker_no_line' if len(xs)==1 else 'measured_polyline_with_markers'))
        short_labels=dict(LABELS,dbpedia='DBP')
        ax.set_xticks([1,2,3,4],[f'{i}\n{short_labels[t]}' for i,t in enumerate(sequence,1)])
        ax.tick_params(axis='both',length=2.5,width=.6,pad=2)
        ax.grid(axis='y',color='.86',linewidth=.55,zorder=0)
        ax.grid(axis='x',color='.92',linewidth=.45,zorder=0)
        ax.set_xlabel('Training stage',labelpad=5)
        bounds=ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2,.073,f'({chr(96+order)}) Order {order}',
                 ha='center',va='center',fontsize=10)
    axes[0].set_ylabel('Accuracy (%)',labelpad=3)
    method_handles=[Line2D([],[],color=METHOD_COLORS[m],linestyle='-',marker=None,
                          linewidth=1.8 if m=='FS-EWCLoRA' else 1.3,label=display_labels[m]) for m in methods]
    method_legend=fig.legend(handles=method_handles,loc='upper center',bbox_to_anchor=(.555,.986),
                            ncol=len(methods),frameon=False,handlelength=1.65,columnspacing=1.05,
                            handletextpad=.4,borderaxespad=0,fontsize=8.1)
    for label in method_legend.get_texts():
        if label.get_text()=='FS-EWCLoRA':
            label.set_color(METHOD_COLORS['FS-EWCLoRA'])
            label.set_fontweight('bold')
    fig.text(.073,.962,'Method',ha='left',va='center',fontsize=8.5)
    dataset_handles=[Line2D([],[],color='.15',linestyle=DATASET_STYLES[d],linewidth=1.2,
                           marker=DATASET_MARKERS[d],markersize=MARKER_SIZE_PT,
                           markeredgewidth=.45,label=LABELS[d]) for d in DATASET_STYLES]
    dataset_legend=fig.legend(handles=dataset_handles,loc='upper center',bbox_to_anchor=(.555,.91),
                             ncol=4,frameon=False,handlelength=3.2,columnspacing=1.5,
                             handletextpad=.6,borderaxespad=0,fontsize=8.3)
    fig.text(.073,.887,'Dataset',ha='left',va='center',fontsize=8.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legend_bounds=[leg.get_window_extent(renderer) for leg in [method_legend,dataset_legend]]
    assert all(not bounds.overlaps(ax.get_window_extent(renderer)) for bounds in legend_bounds for ax in axes)
    assert all(bounds.y0>ax.get_window_extent(renderer).y1 for bounds in legend_bounds for ax in axes)
    assert len(set(DATASET_MARKERS.values()))==4
    for ax in axes:
        for line in ax.lines:
            assert line.get_marker() in DATASET_MARKERS.values()
            assert line.get_markersize()==MARKER_SIZE_PT
            if len(line.get_xdata())==1: assert line.get_linestyle()=='None'
    size='1b' if model.endswith('1B') else '3b'
    stem=f'llama_{size}_taskwise_orders_methodcolors_v5'
    for ext in ['pdf','svg','png']:
        file=HERE/f'{stem}.{ext}'
        fig.savefig(file,facecolor='white')
        print(file)
    (HERE/f'{stem}_plotted_data.json').write_text(json.dumps(plotted,indent=2)+'\n')
    check=dict(model=model,source_csv_sha256=hashlib.sha256((HERE/'taskwise_accuracy.csv').read_bytes()).hexdigest(),
               methods=methods,series=len(plotted),measured_points=sum(len(r['x']) for r in plotted),
               single_observation_markers=singleton_count,single_observation_line_glyphs=0,
               dataset_marker_count=len(DATASET_MARKERS),markersize_pt=MARKER_SIZE_PT,
               dataset_style_count=len(DATASET_STYLES),fs_ewclora_color=METHOD_COLORS['FS-EWCLoRA'],
               legends_outside_axes=True,legends_above_axes=True,
               displayed_method_aliases=display_labels)
    (HERE/f'{stem}_style_check.json').write_text(json.dumps(check,indent=2)+'\n')
    plt.close(fig)
