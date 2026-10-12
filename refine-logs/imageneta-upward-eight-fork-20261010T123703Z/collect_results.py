#!/usr/bin/env python3
"""Publish only validated completed formal runs; never fill missing results."""
import csv,json,statistics
from pathlib import Path
from common import atomic_json,now,sha256
H=Path(__file__).resolve().parent
def collect():
    suite=json.loads((H/'suite.json').read_text());statepath=H/'runtime/queue_state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'jobs':{}}
    rows=[]
    for j in suite['jobs']:
        if j['kind']!='training' or j['smoke'] or state['jobs'].get(j['id'],{}).get('status')!='completed':continue
        out=Path(j['output_dir']);a=json.loads((out/'final_summary.json').read_text());receipt=json.loads((H/'runtime/jobs'/j['id']/'completed.json').read_text())
        if receipt['summary_sha256']!=sha256(out/'final_summary.json'):raise AssertionError('Completed result changed')
        m=a['final_metrics'];single=j['layout']=='single200'
        row=dict(job=j['id'],seed=j['seed'],panel='single200' if single else 'parent' if j['phase']=='parent' else 'task1_fork',
            direction=j['upward_direction'],r=j['r'],rho=j['rho'],epochs=a['completed_epochs'],updates=a['completed_steps'],
            FAA=None if single else m['FAA_task_macro'],AAA=None if single else m['AAA_task_macro'],BWT=None if single else m['BWT'],
            sample_accuracy=a['endpoint_test']['test_accuracy'] if single else m['final_test_accuracy'],
            original10group_accuracy=a['endpoint_test']['test_task_macro_accuracy'] if single else None,
            test_ce=a['endpoint_test']['test_ce'] if single else None,
            current_dev_ce=a.get('endpoint_dev',{}).get('dev_current_task_ce'),
            retention_auc=a.get('retention_auc'),old_training_delta=a.get('old_training_delta'),
            native_training_compute_seconds=a['native_training_compute_seconds'],training_peak_bytes=a['native_training_peak_allocated_bytes'],
            summary_path=str(out/'final_summary.json'),summary_sha256=receipt['summary_sha256'])
        rows.append(row)
    cols=list(rows[0]) if rows else ['job','seed','panel','direction','r','rho','FAA','AAA','BWT']
    with (H/'FORMAL_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:groups.setdefault((r['panel'],r['direction'],r['r']),[]).append(r)
    aggregated=[]
    for key,rs in groups.items():
        x=dict(panel=key[0],direction=key[1],r=key[2],seeds=[r['seed'] for r in rs],n=len(rs))
        for metric in ('FAA','AAA','BWT','sample_accuracy','original10group_accuracy','test_ce','current_dev_ce','retention_auc','old_training_delta'):
            vals=[r[metric] for r in rs if r.get(metric) is not None]
            x[metric+'_mean']=statistics.mean(vals) if vals else None
            x[metric+'_sample_sd']=statistics.stdev(vals) if len(vals)>1 else None
        aggregated.append(x)
    atomic_json(H/'AGGREGATED_RESULTS.json',dict(updated_at=now(),groups=aggregated,definition='mean/sample SD over accepted seeds; distinct panels and radii stay separate'))
    counts={s:sum(r.get('status')==s for r in state['jobs'].values()) for s in ('pending','running','completed','failed')}
    active=[(jid,r.get('gpu')) for jid,r in state['jobs'].items() if r.get('status')=='running']
    text=['# Upward-side experiment progress','',f'Updated: {now()}',f'Queue: {state.get("status","not started")}; {counts}',f'Active: {active}',f'Accepted formal training: {len(rows)}/45. Smoke and offline jobs are excluded.','',
        '| Panel | Direction | r | Seed | AAA | FAA | BWT |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['panel']!='single200':text.append('| '+ ' | '.join(str(r[k]) if r[k] is not None else 'N/A' for k in ('panel','direction','r','seed','AAA','FAA','BWT'))+' |')
    text+=['','| Single200 direction | Seed | Sample Acc | 10-group Acc | Test CE |','|---|---:|---:|---:|---:|']
    for r in rows:
        if r['panel']=='single200':text.append('| '+' | '.join(str(r[k]) for k in ('direction','seed','sample_accuracy','original10group_accuracy','test_ce'))+' |')
    text+=['','Task1 AAA is the two-stage average; RetentionAUC is a separate epoch retention statistic. Single200 BWT is undefined. No test-based selection.']
    (H/'RESULTS.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':collect()
