#!/usr/bin/env python3
"""CPU-only figures from frozen copies of audited CSVs; no fitted values or error bars."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/results-audit-matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
INPUTS=HERE/'inputs'
INPUTS.mkdir(exist_ok=True)
sources=[ROOT/'task0/epoch_curves.csv',ROOT/'task0/final_results.csv',ROOT/'ten_task/training_summary.csv']
pins=[]
for source in sources:
    dest=INPUTS/source.name
    content=source.read_bytes()
    if dest.exists() and dest.read_bytes()!=content:
        raise RuntimeError('Audited input changed; use a versioned figures directory: '+str(source))
    if not dest.exists():dest.write_bytes(content)
    pins.append({'source':str(source),'snapshot':str(dest),'sha256':hashlib.sha256(content).hexdigest()})

def rows(name):
    with (INPUTS/name).open() as f:return list(csv.DictReader(f))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.axisbelow':True,'grid.color':'#dedede','grid.linewidth':.65,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
colors={'ce':'#62666a','s':'#0072b2','g':'#d55e00','mean':'#8855a1','sum':'#009e73'}
labels={'ce':'CE: $g_0$','s':'S: $g_1$','g':'G: $g_3$',
        'mean':'Mean: $(g_1+g_3)/2$','sum':'Sum: $g_1+g_3$'}
styles={'ce':(0,(3,2)),'s':'-','g':(0,(5,2)),'mean':(0,(4,1,1,1)),'sum':'-'}
checks=[]

def save(fig,name,metadata):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    outside=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > w+1 or box.y1 > h+1):
            outside.append({'text':artist.get_text(),'bounds':list(box.bounds)})
    if outside:raise AssertionError('Text outside canvas: '+repr(outside))
    fig.savefig(HERE/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    checks.append({'figure':name,'canvas_text_clipping_check':'PASS','outside_text':outside,
                   'png_dpi':300,'pdf_vector':True,'error_bars':False,**metadata})
    plt.close(fig)

# Independent Task0; only the five completed, audited new runs are eligible.
curve=rows('epoch_curves.csv');final=rows('final_results.csv')
order=['ce','s','g','mean','sum']
assert {r['direction'] for r in final}==set(order)
assert len({r['job'] for r in final})==len(final)==5
fig,axes=plt.subplots(1,2,figsize=(11.5,4.9))
fig.subplots_adjust(left=.075,right=.982,bottom=.18,top=.77,wspace=.24)
for direction in order:
    endpoint=next(r for r in final if r['direction']==direction)
    series=sorted([r for r in curve if r['job']==endpoint['job']],key=lambda r:int(r['epoch']))
    assert [int(r['epoch']) for r in series]==list(range(1,51))
    assert int(endpoint['epochs'])==50
    assert math.isclose(float(series[-1]['eval_accuracy']),float(endpoint['a11_sample']),abs_tol=1e-10)
    assert math.isclose(float(series[-1]['eval_ce']),float(endpoint['eval_ce']),abs_tol=1e-10)
    x=[int(r['epoch']) for r in series]
    for ax,key in zip(axes,['eval_accuracy','eval_ce']):
        y=[float(r[key]) for r in series]
        assert all(math.isfinite(v) for v in y)
        ax.plot(x,y,color=colors[direction],ls=styles[direction],lw=1.75,label=labels[direction],alpha=.95)
        ax.scatter([x[-1]],[y[-1]],color=colors[direction],s=24,zorder=4)
for ax in axes:
    ax.set_xlim(0,51);ax.set_xticks([0,10,20,30,40,50]);ax.set_xlabel('Training epoch');ax.grid(axis='y')
axes[0].set_title('(a) Held-out Task0 accuracy');axes[0].set_ylabel('$a_{11}$, sample accuracy (%)')
axes[1].set_title('(b) Held-out cross-entropy');axes[1].set_ylabel('Evaluation CE (nats)')
fig.suptitle('Independent Task0 learning curves',y=.98,fontsize=15,fontweight='semibold')
fig.text(.5,.916,'Stage1 snapshot: 5 of 10 planned arms completed | seed 2 only',ha='center',fontsize=10,color='#555555')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.895),ncol=5,frameon=False,handlelength=2.5,columnspacing=1.65)
fig.text(.5,.048,'Every shown run completed 50 epochs. Final results use epoch 50, never the best observed epoch.',ha='center',fontsize=9,color='#555555')
fig.text(.5,.015,'Separate Stage1 protocol with isolated data RNG; no Task0 results borrowed from ten-task runs.',ha='center',fontsize=9,color='#555555')
save(fig,'task0_learning_curves',{'input_rows':len(curve),'shown_arms':order,'epochs':50,'campaign_status':'incomplete_5_of_10'})

training=rows('training_summary.csv')
composition=[]
for direction,coeff in [('s',(1.,0.)),('g',(0.,1.)),('mean',(.5,.5)),('sum',(1.,1.))]:
    match=[r for r in training if float(r['gamma'])==0 and float(r['outer_radius'])==1
           and (float(r['beta1']),float(r['beta3']))==coeff]
    assert len(match)==1
    composition.append((direction,match[0]))
fig,axes=plt.subplots(1,2,figsize=(10.4,4.45),sharey=True)
fig.subplots_adjust(left=.20,right=.972,bottom=.20,top=.78,wspace=.31)
for ax,key,title in zip(axes,['aaa_task_macro','faa_task_macro'],['(a) Task AAA','(b) Task FAA']):
    values=[float(r[key]) for _,r in composition];span=max(values)-min(values)
    for idx,(direction,row) in enumerate(composition):
        value=float(row[key]);ax.scatter(value,idx,color=colors[direction],s=67,zorder=3)
        ax.annotate(f'{value:.2f}',(value,idx),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_xlim(min(values)-.12*span,max(values)+.35*span)
    ax.set_ylim(len(composition)-.5,-.5);ax.set_xlabel('Accuracy (%)');ax.set_title(title);ax.grid(axis='x')
    lo,hi=ax.get_xlim()
    ax.set_xticks([tick for tick in MaxNLocator(5).tick_values(lo,hi) if lo <= tick <= hi])
axes[0].set_yticks(range(len(composition)),[labels[d] for d,_ in composition]);axes[0].tick_params(axis='y',length=0)
fig.suptitle('Ten-task direction composition',y=.98,fontsize=15,fontweight='semibold')
fig.text(.5,.883,r'Complete ten-task runs | seed 2 | $\gamma=0$, $r=1$, $\rho=0.05$',ha='center',fontsize=10,color='#555555')
fig.text(.5,.070,'Task AAA averages the task-mean accuracy after each stage; Task FAA is the final task mean.',ha='center',fontsize=9,color='#555555')
fig.text(.5,.025,'Single seed: no error bars. This protocol is separate from the independent Stage1 Task0 experiment.',ha='center',fontsize=9,color='#555555')
save(fig,'ten_task_composition',{'shown_arms':[r['arm'] for _,r in composition],'seed_count':1})

radius=sorted([r for r in training if float(r['gamma'])==0 and float(r['beta1'])==0 and float(r['beta3'])==1],key=lambda r:float(r['outer_radius']))
x=[float(r['outer_radius']) for r in radius];assert x==[0,.1,.25,.5,1]
fig,axes=plt.subplots(1,2,figsize=(10.4,4.45))
fig.subplots_adjust(left=.075,right=.977,bottom=.20,top=.78,wspace=.23)
for ax,key,title in zip(axes,['aaa_task_macro','faa_task_macro'],['(a) Task AAA','(b) Task FAA']):
    y=[float(r[key]) for r in radius];span=max(y)-min(y)
    ax.plot(x,y,'o-',color=colors['g'],lw=1.85,ms=5.5)
    for px,py in zip(x,y):ax.annotate(f'{py:.2f}',(px,py),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xlim(-.055,1.07);ax.set_ylim(min(y)-.12*span,max(y)+.25*span)
    ax.set_xticks(x,['0','0.1','0.25','0.5','1']);ax.set_xlabel('Outer radius $r$');ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.grid(axis='y')
fig.suptitle('Ten-task $g_3$ outer-radius sweep',y=.98,fontsize=15,fontweight='semibold')
fig.text(.5,.883,r'Complete ten-task runs | seed 2 | $\beta_1=0$, $\beta_3=1$, $\gamma=0$, $\rho=0.05$',ha='center',fontsize=10,color='#555555')
fig.text(.5,.070,'Numeric radius spacing is preserved. Values are final audited results, not peak-selected checkpoints.',ha='center',fontsize=9,color='#555555')
fig.text(.5,.025,'Single seed: no uncertainty estimates; small differences do not establish a robust optimum.',ha='center',fontsize=9,color='#555555')
save(fig,'ten_task_g3_radius',{'shown_arms':[r['arm'] for r in radius],'seed_count':1,'radii':x})

manifest={'status':'PASS','inputs':pins,'figures':checks,'matplotlib_version':matplotlib.__version__,
          'visual_check':'pending manual image review','numeric_values_hardcoded':False}
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PASS','figures':[x['figure'] for x in checks],'inputs_pinned':len(pins)}))
