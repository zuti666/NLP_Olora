#!/usr/bin/env python3
"""Plot current-stage task-macro AA and its running AAA from original logs.

python plot_lora_aa.py                 # both metrics, all datasets
python plot_lora_aa.py --metric aa     # requested AA figure only
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = {'cifar100':('CIFAR-100',10), 'domainnet':('DomainNet',5),
            'imagenetr':('ImageNet-R',10), 'imageneta':('ImageNet-A',10)}
METHODS = ['Vanilla LoRA','InfLoRA','SD-LoRA','EWCLoRA','GAM-EWC+Merge']
COLORS = ['#777777','#0072B2','#D55E00','#009E73','#CC3377']
MARKERS = ['o','s','^','P','*']
SCALAR = re.compile(r'(?:np\.|numpy\.)?float(?:16|32|64)?\(([^()]*)\)')
GROUP_ROW = re.compile(r'=> (?:Accuracy|CNN):\s*(\{.*\})\s*$')
SAMPLE_CURVE = re.compile(r'(?:CNN top1 curve:|\(curve\) top1 Acc:)\s*(\[[^\n]*\])')


def parse_literal(text):
    """Parse only numeric log containers; no eval and no executable code."""
    text = SCALAR.sub(r'\1', text)
    text = re.sub(r'\bnan\b','None',text)
    return ast.literal_eval(text)


def read_task_accuracies(run):
    """Read a lower-triangular accuracy matrix a[t,j], for j <= t.

    Rows are classifier results using the global seen-class label space.
    Task ranges are checked against the logged init_cls / increment; unseen
    task NaNs never enter a mean. A logged total cross-check binds these
    dictionaries to the same run as the previously plotted top1 curve.
    """
    path = Path(run['log']); raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != run['sha256']:
        raise ValueError(f'Changed raw log: {path}')
    text = raw.decode('utf-8',errors='replace')
    if re.search(r'\b(?:loss\w*|grad_norm)\s*[:= ]\s*(?:nan|[-+]?inf)\b',text,re.I):
        raise ValueError(f'Non-finite training: {path}')
    n_tasks = DATASETS[run['dataset']][1]
    config = run['logged_config']
    init_cls, increment = int(config['init_cls']), int(config['increment'])
    ranges = [(0,init_cls-1)] + [(init_cls+(j-1)*increment,init_cls+j*increment-1)
                                for j in range(1,n_tasks)]
    rows = []
    for line_no, line in enumerate(text.splitlines(),1):
        match = GROUP_ROW.search(line)
        if match:
            values = parse_literal(match[1])
            tasks = {tuple(map(int,k.split('-'))):v for k,v in values.items()
                     if re.fullmatch(r'\d+-\d+',k)}
            rows.append((line_no,values['total'],tasks))
    if len(rows)<n_tasks:
        raise ValueError(f'Incomplete per-task rows: {path}')
    selected = rows[-n_tasks:]
    samples = list(SAMPLE_CURVE.finditer(text))
    if not samples:
        raise ValueError(f'No classifier curve for cross-check: {path}')
    sample = np.asarray(parse_literal(samples[-1][1]),dtype=float)
    if sample.shape!=(n_tasks,) or not np.isfinite(sample).all():
        raise ValueError(f'Invalid classifier curve: {path}')
    task_rows, lines, totals = [], [], []
    for t,(line_no,total,values) in enumerate(selected,1):
        finite_keys = {k for k,v in values.items() if v is not None and np.isfinite(v)}
        if finite_keys != set(ranges[:t]):
            raise ValueError(f'Stage {t} task ranges mismatch at {path}:{line_no}')
        row = np.asarray([values[k] for k in ranges[:t]],dtype=float)
        if not np.isfinite(row).all() or np.any((row<0)|(row>100)):
            raise ValueError(f'Invalid task accuracies: {path}:{line_no}')
        task_rows.append(row.tolist()); lines.append(line_no); totals.append(total)
    if not np.allclose(totals,sample,atol=0.000001,rtol=0):
        raise ValueError(f'Task rows and classifier curve are not aligned: {path}')
    aa = np.asarray([np.mean(row) for row in task_rows])
    return dict(run,task_accuracies=task_rows,task_ranges=ranges,task_row_lines=lines,
                aa=aa.tolist(),aaa=(np.cumsum(aa)/np.arange(1,n_tasks+1)).tolist(),
                sample_accuracy=sample.tolist(),
                previous_sample_aaa=(np.cumsum(sample)/np.arange(1,n_tasks+1)).tolist())


def load_runs(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    groups = defaultdict(list); records=[]; seen=set(); hashes=set()
    for run in manifest['runs']:
        if not run.get('include',True) or run['method'] not in METHODS:
            continue
        key=(run['dataset'],run['method'],int(run['seed']))
        if key in seen or run['sha256'] in hashes:
            raise ValueError(f'Duplicate run: {key}')
        record=read_task_accuracies(run)
        seen.add(key);hashes.add(run['sha256'])
        groups[key[:2]].append(record);records.append(record)
    for runs in groups.values():runs.sort(key=lambda r:int(r['seed']))
    return manifest,groups,records


def configure_times():
    for path in sorted((HERE/'fonts').glob('*.TTF')):
        font_manager.fontManager.addfont(str(path))
    regular=font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    if font_manager.FontProperties(fname=regular).get_name()!='Times New Roman':
        raise RuntimeError('An authentic Times New Roman font is required.')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,
        'axes.labelsize':11,'axes.titlesize':13,'xtick.labelsize':10,'ytick.labelsize':10,
        'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold'})
    return {'family':'Times New Roman','resolved_path':regular,
            'sha256':hashlib.sha256(Path(regular).read_bytes()).hexdigest()}


def summarize(runs,metric):
    values=np.asarray([r[metric] for r in runs])
    mean=values.mean(axis=0)
    sd=values.std(axis=0,ddof=1) if len(runs)>1 else np.full_like(mean,np.nan)
    return mean,sd


def plot_dataset(ax,groups,dataset,metric='aa',single_seed_band=0.2):
    """Plot current-stage AA or cumulative AAA with all-solid lines and inside legend."""
    title,steps=DATASETS[dataset];x=np.arange(1,steps+1)
    for i,method in enumerate(METHODS):
        runs=groups[(dataset,method)];mean,sd=summarize(runs,metric)
        label=method+(' †' if len(runs)==1 else '')
        ax.plot(x,mean,color=COLORS[i],marker=MARKERS[i],linestyle='-',
                linewidth=1.5,markersize=4,label=label,zorder=3)
        width=sd if len(runs)>1 else single_seed_band
        ax.fill_between(x,mean-width,mean+width,color=COLORS[i],alpha=0.14,linewidth=0,zorder=1)
    ax.set_title(title,pad=7,fontsize=13)
    ax.set_xlabel('Number of learned tasks',fontsize=11)
    ax.set_ylabel('Average accuracy, AA (%)' if metric=='aa' else 'Cumulative AAA (%)',fontsize=11)
    ax.tick_params(labelsize=10,pad=3.5)
    ax.set_xticks(x);ax.set_xlim(0.8,steps+0.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True,color='#d8d8d8',linewidth=0.55,alpha=0.7);ax.set_axisbelow(True)
    ax.margins(y=0.09);lo,hi=ax.get_ylim();ax.set_ylim(max(0,lo),min(100,hi))
    # Reserve clear space for the long legend on this otherwise crowded panel.
    legend_location='best'
    if dataset=='imagenetr' and metric=='aaa':
        ax.set_ylim(max(0,lo),100)
        legend_location='upper right'
    legend=ax.legend(loc=legend_location,frameon=True,facecolor='white',edgecolor='#dddddd',
                     framealpha=0.94,fontsize=8.6,handlelength=1.65,
                     handletextpad=0.5,borderpad=0.3,labelspacing=0.28)
    return legend


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export_data(output,manifest,groups,records,band_width,font):
    raw=[];tasks=[];summary=[];bands=[];comparison=[]
    for r in records:
        for i,row in enumerate(r['task_accuracies']):
            common={'dataset':r['dataset'],'method':r['method'],'seed':r['seed'],'stage':i+1}
            raw.append(dict(common,aa=r['aa'][i],aaa=r['aaa'][i],sample_accuracy=r['sample_accuracy'][i],previous_sample_aaa=r['previous_sample_aaa'][i],log=r['log'],line=r['task_row_lines'][i]))
            for j,value in enumerate(row):
                tasks.append(dict(common,evaluated_task=j+1,accuracy=value,
                                  class_start=r['task_ranges'][j][0],class_end=r['task_ranges'][j][1],
                                  log=r['log'],line=r['task_row_lines'][i]))
    for dataset in DATASETS:
        for method in METHODS:
            runs=groups[(dataset,method)]
            for metric in ['aa','aaa']:
                mean,sd=summarize(runs,metric)
                for i,(m,s) in enumerate(zip(mean,sd),1):
                    row={'dataset':dataset,'method':method,'metric':metric,'stage':i,'n':len(runs),
                         'mean':m,'std':s if len(runs)>1 else '',
                         'variance':s*s if len(runs)>1 else ''}
                    summary.append(row)
                    width=s if len(runs)>1 else band_width
                    bands.append(dict(row,band_kind='empirical_sample_sd' if len(runs)>1 else 'illustrative_fixed_width_not_statistical',half_width_pp=width,lower=m-width,upper=m+width))
            aa,aa_sd=summarize(runs,'aa');aaa,aaa_sd=summarize(runs,'aaa')
            sample,sample_sd=summarize(runs,'sample_accuracy');old_aaa,old_sd=summarize(runs,'previous_sample_aaa')
            comparison.append({'dataset':dataset,'method':method,'n':len(runs),
                'final_task_macro_aa':aa[-1],'final_sample_weighted_accuracy':sample[-1],
                'difference_task_macro_minus_sample_pp':aa[-1]-sample[-1],
                'final_task_macro_aaa':aaa[-1],'previous_sample_weighted_aaa':old_aaa[-1]})
    for name,rows in [('raw_per_seed',raw),('raw_task_accuracy',tasks),('stage_statistics',summary),
                       ('band_metadata',bands),('metric_comparison',comparison)]:
        write_csv(output/f'{name}.csv',rows)
    for dataset in DATASETS:
        write_csv(output/f'{dataset}_stage_statistics.csv',[r for r in summary if r['dataset']==dataset])
    (output/'parsed_raw_runs.json').write_text(json.dumps(records,indent=2))
    metadata={'aa':'AA_t = (1/t) sum_{j=1}^t a[t,j]; task-macro current-stage accuracy',
              'aaa':'AAA_t = (1/t) sum_{s=1}^t AA_s; task-macro cumulative accuracy',
              'previous_figures':'Prior AAA used sample-weighted stage accuracy, a different weighting convention.',
              'future_tasks':'Not included; standard seen-task evaluation only.',
              'single_seed_band_half_width_pp':band_width,'single_seed_band':'illustrative only, not SD/variance/CI',
              'multiseed_band':'mean +/- sample SD across seeds (ddof=1)',
              'font':font,'line_style':'solid for every method','legend':'inside each axes',
              'task_accuracy_precision':'Task scores have the decimal precision retained by original logs.',
              'excluded_runs':[r for r in manifest['runs'] if not r.get('include',True)]}
    (output/'plot_metadata.json').write_text(json.dumps(metadata,indent=2))


def save(fig,stem):
    for ext in ['pdf','png','svg']:
        fig.savefig(f'{stem}.{ext}',bbox_inches='tight',pad_inches=0.055)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=HERE/'sources.json')
    parser.add_argument('--output',type=Path,default=HERE/'output')
    parser.add_argument('--metric',choices=['aa','aaa','both'],default='both')
    parser.add_argument('--single-seed-band',type=float,default=0.2)
    args=parser.parse_args()
    if not np.isfinite(args.single_seed_band) or args.single_seed_band<0:
        parser.error('single-seed-band must be nonnegative and finite')
    manifest,groups,records=load_runs(args.manifest)
    font=configure_times();args.output.mkdir(parents=True,exist_ok=True)
    export_data(args.output,manifest,groups,records,args.single_seed_band,font)
    metrics=['aa','aaa'] if args.metric=='both' else [args.metric]
    captions=[]
    for metric in metrics:
        for dataset in DATASETS:
            fig,ax=plt.subplots(figsize=(5.0,3.75))
            plot_dataset(ax,groups,dataset,metric,args.single_seed_band)
            ewc_n=len(groups[(dataset,'EWCLoRA')]);gam_n=len(groups[(dataset,'GAM-EWC+Merge')])
            note=f'† n=1; ±{args.single_seed_band:g} pp illustrative band (not SD/CI).\nEWC n={ewc_n}, GAM n={gam_n}: mean ± sample SD. Seen tasks weighted equally.'
            fig.text(0.53,0.028,note,ha='center',fontsize=8.3)
            fig.subplots_adjust(left=0.145,right=0.98,top=0.90,bottom=0.23)
            save(fig,args.output/f'{dataset}_{metric}')
        fig,axes=plt.subplots(1,4,figsize=(13.4,3.65))
        for ax,dataset in zip(axes,DATASETS):plot_dataset(ax,groups,dataset,metric,args.single_seed_band)
        note=f'† n=1; fixed ±{args.single_seed_band:g} pp illustrative band (not SD/CI). Other bands: mean ± sample SD.\nEWC n=5/3/4/5; GAM n=4/6/6/6. Seen tasks weighted equally; historical method-specific training recipes.'
        fig.text(0.515,0.025,note,ha='center',fontsize=9)
        fig.subplots_adjust(left=0.05,right=0.996,top=0.90,bottom=0.24,wspace=0.36)
        save(fig,args.output/f'lora_taskwise_{metric}')
        # Paper-width companion keeps >=8 pt legends after insertion at textwidth.
        fig,axes=plt.subplots(2,2,figsize=(7.2,6.2))
        for ax,dataset in zip(axes.flat,DATASETS):plot_dataset(ax,groups,dataset,metric,args.single_seed_band)
        print_note=f'† n=1; fixed ±{args.single_seed_band:g} pp illustrative band (not SD/CI). Others: mean ± sample SD.\nEWC n=5/3/4/5; GAM n=4/6/6/6 (row order). Seen tasks weighted equally; recipes differ.'
        fig.text(0.53,0.025,print_note,ha='center',fontsize=8.5)
        fig.subplots_adjust(left=0.09,right=0.988,top=0.945,bottom=0.135,wspace=0.30,hspace=0.39)
        save(fig,args.output/f'lora_taskwise_{metric}_paper_2x2')
        meaning=(r'Current-stage task-macro $AA_t=t^{-1}\sum_{j=1}^t a_{t,j}$' if metric=='aa'
                 else r'Cumulative task-macro $AAA_t=t^{-1}\sum_{s=1}^t AA_s$')
        caption=meaning+'. Each task has equal weight; future tasks are excluded. Daggered baselines use their available seed-0 logs and fixed illustrative bands of '+str(args.single_seed_band)+' percentage points, not estimated uncertainty. EWC and GAM curves show means with sample standard deviations; seed counts in dataset order are 5/3/4/5 and 4/6/6/6, respectively. Historical training recipes differ. The earlier plots used sample-weighted stage accuracy; these task-macro values may differ. GAM-EWC+Merge denotes the historical implementation traced from the former FS-EWCLoRA table row.'
        captions += ['% Print-size version. The 1x4 version is a wider screen/native-size companion.',r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{'+args.output.name+'/lora_taskwise_'+metric+r'_paper_2x2.pdf}',r'\caption{'+caption+'}',r'\label{fig:lora_task_macro_'+metric+'}',r'\end{figure*}','']
    (args.output/'latex_includes.tex').write_text('\n'.join(captions))
    print(f'Read {len(records)} original logs; task-macro AA/AAA plotted with Times New Roman.')


if __name__=='__main__':main()
