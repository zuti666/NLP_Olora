#!/usr/bin/env python3
"""Independent Stage1 radius endpoints; preserves the earlier five-arm snapshot."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PIN=HERE/'inputs'/'task0_g3_radius'
PIN.mkdir(parents=True,exist_ok=True)
pins=[]
for name in ['radius_final_results.csv','final_results.csv','radius_completed_audit.json']:
    src=ROOT/'task0'/name
    content=src.read_bytes();dest=PIN/name
    if dest.exists() and dest.read_bytes()!=content:
        raise RuntimeError('Pinned input changed: '+str(src))
    if not dest.exists():dest.write_bytes(content)
    pins.append({'source':str(src),'snapshot':str(dest),'sha256':hashlib.sha256(content).hexdigest()})
def read_csv(name):
    with (PIN/name).open() as f:return list(csv.DictReader(f))
audit=json.loads((PIN/'radius_completed_audit.json').read_text())
assert audit['status']=='passed' and audit['completed_audited']==4
radius=read_csv('radius_final_results.csv')
original=[r for r in read_csv('final_results.csv') if r['direction']=='g']
assert len(original)==1 and len(radius)==4
# r=1 is the completed G arm from this same independent Stage1 campaign.
radius += [dict(original[0],radius='1.0')]
radius.sort(key=lambda r:float(r['radius']))
x=[float(r['radius']) for r in radius]
assert x==[0,.1,.25,.5,1]
for r in radius:
    assert r['direction']=='g' and int(r['epochs'])==50 and int(r['eval_samples'])==307
    assert math.isclose(float(r['a11_sample']),100*int(r['correct'])/int(r['eval_samples']),abs_tol=1e-10)
    assert all(math.isfinite(float(r[k])) for k in ['a11_sample','a11_class_macro','eval_ce'])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.axisbelow':True,'grid.color':'#dedede','grid.linewidth':.65,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(11.2,4.7))
fig.subplots_adjust(left=.085,right=.977,bottom=.22,top=.77,wspace=.26)
for ax,key,title,ylabel,fmt in zip(axes,['a11_sample','eval_ce'],
        ['(a) Held-out Task0 accuracy','(b) Held-out cross-entropy'],
        ['$a_{11}$, sample accuracy (%)','Evaluation CE (nats)'],['.2f','.4f']):
    y=[float(r[key]) for r in radius];span=max(y)-min(y)
    ax.plot(x,y,'o-',color='#d55e00',lw=1.85,ms=5.5)
    for idx,(px,py) in enumerate(zip(x,y)):
        offset=-4 if key=='a11_sample' and idx==0 else (4 if key=='a11_sample' and idx==1 else 0)
        ax.annotate(format(py,fmt),(px,py),xytext=(offset,8),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xlim(-.075,1.07);ax.set_ylim(min(y)-.17*span,max(y)+.26*span)
    ax.set_xticks(x,['0','0.1','0.25','0.5','1']);ax.set_xlabel('Outer radius $r$')
    ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y')
fig.suptitle('Independent Task0 $g_3$ outer-radius sweep',y=.98,fontsize=15,fontweight='semibold')
fig.text(.5,.90,'Stage1 snapshot: 9 of 10 planned arms completed; standard SAM running',ha='center',fontsize=10,color='#555555')
fig.text(.5,.85,r'Seed 2 only | 307 evaluation samples | $\beta_1=0$, $\beta_3=1$, $\gamma=0$, $\rho=0.05$',ha='center',fontsize=10,color='#555555')
fig.text(.5,.090,'All five radius arms completed 50 epochs. Endpoints are fixed at epoch 50, never selected by peak accuracy.',ha='center',fontsize=9,color='#555555')
fig.text(.5,.040,'Independent Stage1 protocol; no ten-task values included. Single seed: no error bars or established optimal radius.',ha='center',fontsize=9,color='#555555')
fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
outside=[]
for artist in fig.findobj(matplotlib.text.Text):
    if not artist.get_visible() or not artist.get_text():continue
    box=artist.get_window_extent(renderer)
    if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1>w+1 or box.y1>h+1):
        outside.append({'text':artist.get_text(),'bounds':list(box.bounds)})
assert not outside,outside
for ext in ['png','pdf']:
    fig.savefig(HERE/('task0_g3_radius.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
values=[{k:r[k] for k in ['job','radius','epochs','eval_samples','correct','a11_sample','a11_class_macro','eval_ce']} for r in radius]
with (HERE/'task0_g3_radius_values.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
manifest={'status':'PASS','inputs':pins,'audit_time_utc':audit['audited_at_utc'],
    'campaign_completed_arms':9,'campaign_planned_arms':10,'sam_status_at_snapshot':'running',
    'fixed_epoch':50,'evaluation_samples':307,'training_seeds':[2],'radii':x,
    'numeric_results_hardcoded':False,'old_five_arm_snapshot_modified':False,
    'canvas_text_clipping_check':'PASS','visual_check':'pending image inspection',
    'error_bars':False,'matplotlib_version':matplotlib.__version__,
    'outputs':[{'name':'task0_g3_radius.'+ext,'sha256':hashlib.sha256((HERE/('task0_g3_radius.'+ext)).read_bytes()).hexdigest()} for ext in ['png','pdf']]}
(HERE/'task0_g3_radius_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PASS','radius_arms':len(radius),'inputs_pinned':len(pins)}))
