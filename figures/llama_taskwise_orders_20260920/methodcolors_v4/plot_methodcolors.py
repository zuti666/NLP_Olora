"""Version 4: OLoRA+GAM replaces pure OLoRA in both models: method colors, dataset line styles.

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
from matplotlib.transforms import Affine2D, ScaledTranslation

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


def draw(model):
    with (HERE/'taskwise_accuracy.csv').open(encoding='utf-8-sig') as f:
        data = [r for r in csv.DictReader(f) if r['model']==model]
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    methods = [m for m in METHODS if any(r['method']==m for r in data)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.85), sharey=True)
    fig.subplots_adjust(left=.073, right=.991, bottom=.385, top=.967, wspace=.18)
    plotted=[]
    glyph_count=0
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
                props=dict(color=METHOD_COLORS[method],linestyle=DATASET_STYLES[dataset],
                           linewidth=1.6 if method=='FS-EWCLoRA' else 1.02,
                           marker=None, solid_capstyle='butt', dash_capstyle='butt',
                           zorder=5 if method=='FS-EWCLoRA' else 3)
                if len(xs)>1:
                    ax.plot(xs,ys,**props)
                else:
                    # An 18-point-wide line glyph at the single measured (x,y).
                    # Width is in display points, not a fabricated task interval.
                    transform=(Affine2D().scale(1/72)+fig.dpi_scale_trans+
                               ScaledTranslation(xs[0],ys[0],ax.transData))
                    ax.add_line(Line2D([-9,9],[0,0],transform=transform,**props))
                    glyph_count+=1
                plotted.append(dict(model=model,order=order,method=method,dataset=dataset,
                                    x=xs,y=ys,color=METHOD_COLORS[method],
                                    dataset_line_style=STYLE_NAMES[dataset],marker=None,
                                    display='single_observation_line_glyph_18pt' if len(xs)==1 else 'measured_polyline'))
        short_labels=dict(LABELS,dbpedia='DBP')
        ax.set_xticks([1,2,3,4],[f'{i}\n{short_labels[t]}' for i,t in enumerate(sequence,1)])
        ax.tick_params(axis='both',length=2.5,width=.6,pad=2)
        ax.grid(axis='y',color='.86',linewidth=.55,zorder=0)
        ax.grid(axis='x',color='.92',linewidth=.45,zorder=0)
        ax.set_xlabel('Training stage',labelpad=5)
        bounds=ax.get_position()
        fig.text((bounds.x0+bounds.x1)/2,.234,f'({chr(96+order)}) Order {order}',
                 ha='center',va='center',fontsize=10)
    axes[0].set_ylabel('Accuracy (%)',labelpad=3)
    method_handles=[Line2D([],[],color=METHOD_COLORS[m],linestyle='-',marker=None,
                          linewidth=1.8 if m=='FS-EWCLoRA' else 1.3,label=m) for m in methods]
    method_legend=fig.legend(handles=method_handles,loc='upper center',bbox_to_anchor=(.555,.169),
                            ncol=len(methods),frameon=False,handlelength=1.65,columnspacing=1.05,
                            handletextpad=.4,borderaxespad=0,fontsize=8.1)
    for label in method_legend.get_texts():
        if label.get_text()=='FS-EWCLoRA':
            label.set_color(METHOD_COLORS['FS-EWCLoRA'])
            label.set_fontweight('bold')
    fig.text(.073,.145,'Method',ha='left',va='center',fontsize=8.5)
    dataset_handles=[Line2D([],[],color='.15',linestyle=DATASET_STYLES[d],linewidth=1.2,
                           marker=None,label=LABELS[d]) for d in DATASET_STYLES]
    dataset_legend=fig.legend(handles=dataset_handles,loc='upper center',bbox_to_anchor=(.555,.097),
                             ncol=4,frameon=False,handlelength=3.2,columnspacing=1.5,
                             handletextpad=.6,borderaxespad=0,fontsize=8.3)
    fig.text(.073,.074,'Dataset',ha='left',va='center',fontsize=8.5)
    if len(methods)<len(METHODS):
        fig.text(.53,.013,'OLoRA (1B): matching source unavailable; curve omitted.',
                 ha='center',va='bottom',fontsize=7.5,color='.28')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legend_bounds=[leg.get_window_extent(renderer) for leg in [method_legend,dataset_legend]]
    assert all(not bounds.overlaps(ax.get_window_extent(renderer)) for bounds in legend_bounds for ax in axes)
    assert all(line.get_marker() in [None,'None',''] for ax in axes for line in ax.lines)
    size='1b' if model.endswith('1B') else '3b'
    stem=f'llama_{size}_taskwise_orders_methodcolors_v4'
    for ext in ['pdf','svg','png']:
        file=HERE/f'{stem}.{ext}'
        fig.savefig(file,facecolor='white')
        print(file)
    (HERE/f'{stem}_plotted_data.json').write_text(json.dumps(plotted,indent=2)+'\n')
    check=dict(model=model,source_csv_sha256=hashlib.sha256((HERE/'taskwise_accuracy.csv').read_bytes()).hexdigest(),
               methods=methods,series=len(plotted),measured_points=sum(len(r['x']) for r in plotted),
               single_observation_line_glyphs=glyph_count,no_point_markers=True,
               dataset_style_count=len(DATASET_STYLES),fs_ewclora_color=METHOD_COLORS['FS-EWCLoRA'],
               legends_outside_axes=True)
    (HERE/f'{stem}_style_check.json').write_text(json.dumps(check,indent=2)+'\n')
    plt.close(fig)

