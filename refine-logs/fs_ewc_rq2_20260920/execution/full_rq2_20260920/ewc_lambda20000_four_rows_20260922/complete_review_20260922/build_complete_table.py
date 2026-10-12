#!/usr/bin/env python3
"""CPU-only synthesis of audited lambda=0 controls and raw-Fisher lambda=20000 results."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import statistics

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
BASE = RUN.parent
ARMS = [
 ('vanilla','Vanilla LoRA','C','C'),
 ('shaping_only','Shaping only','C+F','C+F'),
 ('ewclora','EWCLoRA','C','C+P'),
 ('fs_ewclora','FS-EWCLoRA','C+F','C+F+P'),
 ('task1_shaping_no_ewc','Task-1 shaping without EWC','C+F','C'),
 ('task1_shaping','Task-1 shaping','C+F','C+P'),
 ('task2_shaping','Task-2 shaping','C','C+F+P'),
]
METRICS = ['a11','B1_batch_proxy','a21','a22','BWT','d_QV_cont','A2',
           'B2_newtask_batch_proxy','raw_fisher_weighted_drift_squared']
CONTROL_PAIR = {'ewclora':'vanilla','fs_ewclora':'shaping_only','task1_shaping':'task1_shaping_no_ewc'}
LABEL = {a:b for a,b,_,_ in ARMS}
OBJECTIVES = {a:(c,d) for a,_,c,d in ARMS}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v, n=2):
    if v is None: return '—'
    return str(Decimal(f'{v:.10f}').quantize(Decimal(10) ** -n, rounding=ROUND_HALF_UP))
def stats(values):
    assert len(values)==3 and all(math.isfinite(float(v)) for v in values)
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'n':3}
def ms(s,n=2): return fmt(s['mean'],n)+'±'+fmt(s['sample_sd'],n)
def csvfile(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def main():
    inputs = [HERE/'revalidated_four_rows/results.json',HERE/'revalidated_four_rows/COMPLETE.json',
              HERE/'CONTROL_AUDIT.json',RUN/'experiment_spec.json',RUN/'SELECTION.json',
              BASE/'seven_row_extension_20260921/final_report/table_data.json',HERE/'PROTOCOL_AUDIT.json']
    new, complete, controls = map(read,inputs[:3])
    assert new['status']=='complete' and new['completed_trials']==12
    assert complete['status']=='complete' and complete['completed_trials']==12
    assert controls['status'] in ('passed','complete','validated'), controls.get('status')
    assert len(controls['controls'])==9
    for f,h in complete.get('output_sha256',{}).items():
        assert sha(f)==h
    rows=[]
    for r in controls['controls']:
        assert r['status']=='complete_verified',r
        q={k:r[k] for k in ['arm','seed','a11','a21','a22','B1_batch_proxy','d_QV_cont','run_dir','metrics_source','parent_checkpoint_sha256']}
        q.update(ewc_lambda=0.,cohort='revalidated_existing_lambda0',status='validated',
                 raw_fisher_weighted_drift_squared=r['raw_fisher_weighted_drift_squared_vs_common_parent_rawF'],
                 original_exact_a11=r.get('a11_exact'))
        rows.append(q)
    for r in new['individual']:
        q=dict(r);q['cohort']='raw_fisher_lambda20000';q['status']='validated';rows.append(q)
    assert len({(r['arm'],r['seed']) for r in rows})==21
    for r in rows:
        m=read(r['metrics_source'])
        r['B2_newtask_batch_proxy']=m['mechanism']['steps']['1']['in_task_fisher_final_raw_trace']
        r['A2']=(r['a21']+r['a22'])/2
        r['BWT']=r['a21']-r['a11']
        r['variant']=LABEL[r['arm']]
        r['task1_objective'],r['task2_objective']=OBJECTIVES[r['arm']]
        assert all(math.isfinite(r[k]) for k in METRICS)
    rows.sort(key=lambda r:([a for a,_,_,_ in ARMS].index(r['arm']),r['seed']))
    cells={(r['arm'],r['seed']):r for r in rows}
    summaries={a:{k:stats([cells[a,s][k] for s in range(3)]) for k in METRICS} for a,_,_,_ in ARMS}
    contrasts=[]
    pairs=list(CONTROL_PAIR.items())+[
        ('fs_ewclora','ewclora'),('task1_shaping','ewclora'),('task2_shaping','ewclora'),
        ('fs_ewclora','task1_shaping'),('fs_ewclora','task2_shaping'),
        ('shaping_only','vanilla'),('task1_shaping_no_ewc','vanilla')]
    for a,b in pairs:
        for k in ['a21','a22','BWT','A2','d_QV_cont','raw_fisher_weighted_drift_squared']:
            ds=[cells[a,s][k]-cells[b,s][k] for s in range(3)]
            contrasts.append(dict(contrast=a+'-'+b,metric=k,**stats(ds),per_seed=ds,
                                  positive_count=sum(v>1e-10 for v in ds),negative_count=sum(v< -1e-10 for v in ds)))
    efficacy=[]
    for a,b in CONTROL_PAIR.items():
        for s in range(3):assert cells[a,s]['parent_checkpoint_sha256']==cells[b,s]['parent_checkpoint_sha256']
        efficacy.append(dict(arm=a,control=b,
            d_QV_reduction_percent=100*(1-summaries[a]['d_QV_cont']['mean']/summaries[b]['d_QV_cont']['mean']),
            fisher_weighted_squared_reduction_percent=100*(1-summaries[a]['raw_fisher_weighted_drift_squared']['mean']/summaries[b]['raw_fisher_weighted_drift_squared']['mean']),
            old_task_accuracy_gain=next(r for r in contrasts if r['contrast']==a+'-'+b and r['metric']=='a21')))
    old=read(inputs[5]);oldcells={(r['arm'],r['seed']):r for r in old['individual']}
    historical=[]
    for a in ['ewclora','fs_ewclora','task1_shaping','task2_shaping']:
        for k,oldk in [('a21','a21'),('a22','a22'),('A2','average_accuracy'),('d_QV_cont','d_QV_cont')]:
            vs=[oldcells[a,s][oldk] for s in range(3)]
            ds=[cells[a,s][k]-oldcells[a,s][oldk] for s in range(3)]
            historical.append(dict(arm=a,metric=k,old_normalized2000=stats(vs),new_raw20000=summaries[a][k],delta=stats(ds),per_seed_delta=ds,scope='joint normalization-and-lambda protocol change; not lambda-only causal effect'))
    mechanism=[]
    for a in ['ewclora','fs_ewclora','task1_shaping','task2_shaping']:
        mechanism.append(dict(arm=a,**{k:stats([cells[a,s][k] for s in range(3)]) for k in
            ['P_over_CE','grad_P_over_reference','endpoint_P','raw_fisher_trace','raw_fisher_weighted_drift_squared','B2_newtask_batch_proxy']},
            d_QV_reduction_percent=next((e['d_QV_reduction_percent'] for e in efficacy if e['arm']==a),None),
            weighted_squared_reduction_percent=next((e['fisher_weighted_squared_reduction_percent'] for e in efficacy if e['arm']==a),None)))
    data=dict(status='validated',validated_main_cells=21,expected_main_cells=21,seeds=[0,1,2],
              generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),individual=rows,
              summary=summaries,paired_contrasts=contrasts,ewc_effects=efficacy,mechanism=mechanism,
              historical_protocol_comparison=historical,source_sha256={str(p):sha(p) for p in inputs},
              accuracy_convention='Use native CIL two-decimal per-seed accuracies consistently for all seven rows. Derive BWT and equal-task A2 per seed before sample SD.',
              rounding='Decimal half-up after numerical floating precision cleanup; small 0.01 differences from old tables can reflect precision unification.',
              missing_counterfactual='SGD→GAM with lambda0 is absent; direct EWC effect for Task-2 shaping is not identifiable from this seven-row table.',
              selected_lambda_scope='User-fixed after exploratory seed0 sweep; not an independently validated optimum. Seeds1/2 provide out-of-search confirmation only.')
    (HERE/'table_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    csvfile('individual.csv',rows);csvfile('paired_contrasts.csv',contrasts);csvfile('mechanism.csv',mechanism)
    csvfile('historical_protocol_comparison.csv',historical)
    csvfile('summary.csv',[dict(arm=a,metric=k,**s) for a,ms in summaries.items() for k,s in ms.items()])
    header='| Variant | Task 1 | Task 2 | a11 | B1 batch proxy | a21 | a22 | BWT | d_QV_cont | 两任务平均 |'
    table=[header,'|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,label,o1,o2 in ARMS:
        ss=summaries[a]
        table.append('| '+' | '.join([label,o1,o2]+[ms(ss[k],3 if k=='B1_batch_proxy' else 4 if k=='d_QV_cont' else 2) for k in ['a11','B1_batch_proxy','a21','a22','BWT','d_QV_cont','A2']])+' |')
    (HERE/'TABLE_IMAGENETA.md').write_text('# ImageNet-A：统一 raw-Fisher λ=20000 七行表\n\n21/21 个 seed×分支核验通过；12条本轮含EWC续训＋9条已核验λ0对照。数值为seed0/1/2均值±样本标准差。含P行λ=20000、两归一化开关关闭，其余λ=0。\n\n'+'\n'.join(table)+'\n\nB1 为第一任务端点的原生batch-gradient平方proxy，非逐样本Fisher；d_QV_cont为同一Q/V LoRA分支的有效BA端点位移，排除分类头。a21/a22为40类joint CIL任务分组准确率；BWT=a21−a11；两任务平均=(a21+a22)/2。\n\n所有行统一由原生两位CIL值计算；旧表a11曾用未舍入比例，因此个别SD与末位0.01差异属于显示口径统一。详细配置、配对对比、Fisher与结论见FINAL_RESULTS_AND_CONCLUSIONS.md。\n')
    tex=[r'\providecommand{\ExpRes}[2]{\ensuremath{#1_{\pm #2}}}',r'\begin{table*}[t]',r'\centering',
         r'\caption{Two-task diagnostics with rank-10 Q/V LoRA on ImageNet-A. Results are means over three seeds with sample standard deviations. All protected variants use unnormalized native batch-gradient Fisher and a common, user-fixed $\lambda=20000$; unprotected variants use $\lambda=0$. $\mathcal B_1$ is the Task-1 batch-gradient Fisher proxy. $d_{QV}^{\mathrm{cont}}$ measures the effective Q/V weight displacement of the continued adapter and excludes classifier heads.}',
         r'\label{tab:two_task_lora_diagnostics_imageneta_raw20000}',r'\scriptsize\setlength{\tabcolsep}{2.4pt}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}',r'\toprule',
         r'Variant & Task 1 & Task 2 & $a_{11}\uparrow$ & $\mathcal B_1$ & $a_{21}\uparrow$ & $a_{22}\uparrow$ & BWT$\uparrow$ & $d_{QV}^{\mathrm{cont}}$ & $A_2\uparrow$ \\',r'\midrule']
    for a,label,o1,o2 in ARMS:
        values=[]
        for k in ['a11','B1_batch_proxy','a21','a22','BWT','d_QV_cont','A2']:
            dp=3 if k=='B1_batch_proxy' else 4 if k=='d_QV_cont' else 2;s=summaries[a][k]
            values.append('\\ExpRes{'+fmt(s['mean'],dp)+'}{'+fmt(s['sample_sd'],dp)+'}')
        def obj(o):return '$'+'+'.join('\\mathsf '+c for c in o.split('+'))+'$'
        tex.append(' & '.join([label,obj(o1),obj(o2)]+values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
    (HERE/'table_imageneta.tex').write_text('\n'.join(tex)+'\n')
    ledger=dict(status='validated',expected_cells=21,validated_cells=21,failed_cells=0,missing_cells=0,
                controls=9,new_formal=12,source_sha256=data['source_sha256'],
                outputs_sha256={str(HERE/f):sha(HERE/f) for f in ['table_data.json','individual.csv','summary.csv','paired_contrasts.csv','mechanism.csv','historical_protocol_comparison.csv','TABLE_IMAGENETA.md','table_imageneta.tex']},
                script_sha256=sha(__file__),training_launched=False,gpu_evaluation_performed=False)
    (HERE/'VALIDITY_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print('\n'.join(table));print('EWC effects:',json.dumps(efficacy,ensure_ascii=False));print('mechanism:',json.dumps(mechanism,ensure_ascii=False))


if __name__=='__main__':main()
