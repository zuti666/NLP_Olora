#!/usr/bin/env python3
"""Collect accepted independent20epoch runs, exact-scope curves and paired contrasts."""
import csv,io,json,statistics
from pathlib import Path
from common import atomic_json,atomic_text,now,sha256
from validate_job import validate_job,CONDITIONS
H=Path(__file__).resolve().parent
LABELS={'sgd':'SGD throughout','sam':'SAM throughout','gam_residual':'Residual GAM throughout','sf_g2':'S_F g2 throughout','sf_g2_then_sgd':'S_F g2 early, SGD late','sgd_then_sf_g2':'SGD early, S_F g2 late'}
METRICS=('task0_end20','old_start40','old_final40','current_final40','old_drop_since_task1_start','training_BWT','expansion_damage','FAA','traditional_BWT','stage_AAA','retention_auc')
CONTRASTS=(('sam','sgd','SAM minus SGD'),('gam_residual','sgd','Residual GAM minus SGD'),('sf_g2','sgd','S_F g2 minus SGD'),('sf_g2_then_sgd','sgd_then_sf_g2','Early g2 minus late g2'),('sf_g2_then_sgd','sf_g2','Early g2 minus full g2'),('sgd_then_sf_g2','sf_g2','Late g2 minus full g2'))
def publish_csv(path,rows,columns):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows);atomic_text(path,stream.getvalue())
def summarize(values):return dict(mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None)
def collect():
    suite=json.loads((H/'suite.json').read_text());statepath=H/'runtime/queue_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'jobs':{},'status':'not started'}
    rows=[];curves=[];task0curves=[];evidence={}
    for j in suite['jobs']:
        if j['kind']!='training' or j['smoke'] or state['jobs'].get(j['id'],{}).get('status')!='completed':continue
        out=Path(j['output_dir']);checked=validate_job(j);receipt_path=H/'runtime/jobs'/j['id']/'completed.json';receipt=json.loads(receipt_path.read_text())
        if checked['summary_sha256']!=receipt['summary_sha256'] or checked['summary_sha256']!=state['jobs'][j['id']]['summary_sha256']:raise AssertionError('Accepted formal receipt changed')
        summary=json.loads((out/'final_summary.json').read_text());m=summary['formation_metrics']
        r=dict(job=j['id'],condition=j['formation_condition_id'],method=LABELS[j['formation_condition_id']],seed=j['seed'],protocol=summary['protocol'],epochs_task0=20,epochs_task1=20,updates=summary['completed_steps'],gradient_reads=checked['gradient_reads'],
            native_training_compute_seconds=summary['native_training_compute_seconds'],training_peak_bytes=summary['native_training_peak_allocated_bytes'],current_dev_ce=summary['endpoint_dev']['dev_current_task_ce'],summary_path=str(out/'final_summary.json'),summary_sha256=checked['summary_sha256'])
        r.update({k:m[k] for k in METRICS});rows.append(r)
        for x in summary['retention_curve']:
            curves.append(dict(job=j['id'],condition=j['formation_condition_id'],seed=j['seed'],task=1,epoch=x['epoch'],old_accuracy_joint40=x['old_test_allseen40']['accuracy'],old_ce_joint40=x['old_test_allseen40']['ce'],old_accuracy_local20=x['old_test_allseen40']['old20_only_accuracy'],new_accuracy_joint40=x['current_test_allseen40']['accuracy'],new_ce_joint40=x['current_test_allseen40']['ce'],FAA_joint40=x['allseen40_task_macro_accuracy'],evaluation_is_passive=x['evaluation_state_rng_restored']))
        t0path=out/'task0_learning_curve.json'
        for x in json.loads(t0path.read_text())['epochs']:
            task0curves.append(dict(job=j['id'],condition=j['formation_condition_id'],seed=j['seed'],task=0,epoch=x['epoch'],active_method=x['active_method'],accuracy_20classes=x['test']['accuracy'],ce_20classes=x['test']['ce'],dev_ce=x['dev']['dev_current_task_ce']))
        for p in (out/'final_summary.json',receipt_path,out/'step_trace.jsonl',t0path):evidence[str(p)]=sha256(p)
    cols=['job','condition','method','seed','protocol','epochs_task0','epochs_task1','updates','gradient_reads',*METRICS,'current_dev_ce','native_training_compute_seconds','training_peak_bytes','summary_path','summary_sha256']
    publish_csv(H/'FORMAL_RESULTS.csv',rows,cols)
    publish_csv(H/'TASK1_RETENTION_CURVES.csv',curves,['job','condition','seed','task','epoch','old_accuracy_joint40','old_ce_joint40','old_accuracy_local20','new_accuracy_joint40','new_ce_joint40','FAA_joint40','evaluation_is_passive'])
    publish_csv(H/'TASK0_LEARNING_CURVES.csv',task0curves,['job','condition','seed','task','epoch','active_method','accuracy_20classes','ce_20classes','dev_ce'])
    groups=[]
    for c in CONDITIONS:
        rs=sorted((r for r in rows if r['condition']==c),key=lambda r:r['seed']);g=dict(condition=c,method=LABELS[c],seeds=[r['seed'] for r in rs],n=len(rs),complete_three_seed_cohort=len(rs)==3)
        for metric in METRICS:
            stats=summarize([r[metric] for r in rs]);g[metric+'_mean']=stats['mean'];g[metric+'_sample_sd']=stats['sample_sd']
        groups.append(g)
    contrasts=[];contrastrows=[]
    indexed={(r['condition'],r['seed']):r for r in rows}
    for lhs,rhs,label in CONTRASTS:
        seeds=[s for s in (0,1,2) if (lhs,s) in indexed and (rhs,s) in indexed]
        x=dict(contrast=label,lhs=lhs,rhs=rhs,seeds=seeds,n=len(seeds),complete_paired_three_seed_cohort=len(seeds)==3,definition='lhs minus rhs within identical seed; positive old_drop difference means more training-induced old-task damage')
        for metric in METRICS:
            differences=[indexed[lhs,s][metric]-indexed[rhs,s][metric] for s in seeds];x[metric]=dict(**summarize(differences),per_seed={str(s):v for s,v in zip(seeds,differences)})
        contrasts.append(x)
        for s in seeds:
            contrastrows.append(dict(contrast=label,seed=s,lhs=lhs,rhs=rhs,**{k:indexed[lhs,s][k]-indexed[rhs,s][k] for k in METRICS}))
    publish_csv(H/'PAIRED_SEED_DIFFERENCES.csv',contrastrows,['contrast','seed','lhs','rhs',*METRICS])
    atomic_json(H/'AGGREGATED_RESULTS.json',dict(updated_at=now(),protocol='task0_formation_20ep_task1_sgd_v1',groups=groups,paired_contrasts=contrasts,evidence_hashes=evidence,
        definitions=dict(primary_accuracy='exact official test count ratios, percent, equal task weighting for FAA',old_drop='post-expansion epoch0 old joint40 accuracy minus epoch20 old joint40 accuracy',sample_sd='unbiased sample standard deviation across accepted seeds; null for n<2',stage_AAA='mean of Task0 endpoint accuracy and Task1 FAA; separate from mean epoch retention AUC',old50_results_pooled=False)))
    counts={s:sum(r.get('status')==s for r in state['jobs'].values()) for s in ('pending','running','completed','failed')}
    text=['# Task0 formation: independent 2 x 20 epoch results','',f'Updated UTC: {now()}',f'Queue: {state.get("status")}; {counts}',f'Accepted formal runs: {len(rows)}/18. Six bounded smoke runs and their pairing gate are excluded.','',
        '| Condition | Seed | Task0 end | Old start (40) | Old final (40) | New final (40) | Old drop | FAA |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append('| '+' | '.join([r['method'],str(r['seed'])]+[f'{r[k]:.3f}' for k in ('task0_end20','old_start40','old_final40','current_final40','old_drop_since_task1_start','FAA')])+' |')
    text+=['','| Condition | Accepted seeds | FAA mean +/- sample SD | Old drop mean +/- sample SD |','|---|---|---:|---:|']
    def display(g,key):
        m,s=g[key+'_mean'],g[key+'_sample_sd'];return 'pending' if m is None else f'{m:.3f}' if s is None else f'{m:.3f} +/- {s:.3f}'
    for g in groups:text.append('| '+g['method']+' | '+str(g['seeds'])+' | '+display(g,'FAA')+' | '+display(g,'old_drop_since_task1_start')+' |')
    text+=['','All Task1 updates use natural SGD. Method switching uses a continuous Task0 cosine schedule and model at epoch11. Old drop starts after head expansion, before Task1 training. The same official old/new tests use the full 40-class classifier.','',
        'Reported ratios are computed from exact test sample counts. Native rounded-per-task FAA/AAA/BWT remain in each run summary for provenance. Historical50-epoch results are never pooled. Partial seed cohorts are descriptive; formal paired conclusions require all three identical seeds. Formation/phase effects do not independently establish a trajectory-only causal mechanism.']
    atomic_text(H/'RESULTS.md','\n'.join(text)+'\n')
    return dict(accepted_formal=len(rows),expected_formal=18,counts=counts,status=state.get('status'))
if __name__=='__main__':print(json.dumps(collect()))
