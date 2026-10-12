#!/usr/bin/env python3
"""Export finite observed results and paired deltas; no absent-result filling."""
import csv
import json
import math
from pathlib import Path
import statistics
from common import atomic_json,atomic_text,now
from validate_job import validate_job,receipt_matches

HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text())
rows=[];futures=[]
for j in suite['jobs']:
    if j['smoke']: continue
    rp=HERE/'runtime/jobs'/j['id']/'completed.json'
    if not rp.exists(): continue
    receipt=json.loads(rp.read_text())
    actual=validate_job(j)
    if receipt['job_id']!=j['id'] or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,actual): raise ValueError('unverified result '+j['id'])
    out=Path(j['output_dir']);p=out/'final_summary.json'
    if not p.exists(): continue
    s=json.loads(p.read_text())
    if s['status']!='completed': continue
    if j['kind']=='formation':
        row={'job_id':j['id'],'arm':j['arm'],'seed':j['seed'],'epochs':s['completed_epochs'],
             'accuracy':s['final']['eval_accuracy'],'macro_accuracy':s['final']['eval_macro_accuracy'],
             'ce':s['final']['eval_ce'],'steps':s['step_trace_steps'],
             'max_match_relative_error':s['step_checks']['max_match_relative_error']}
        rows.append(row)
    else:
        ev=json.loads((out/'evaluations.json').read_text());k=ev['milestones'];end=str(j['expected_steps'])
        for subset in ('full','diagnostic','reporting'):
            for space in ('local','all_seen'):
                old0=k['0']['task0'][subset][space];oldk=k[end]['task0'][subset][space];newk=k[end]['task1'][subset][space]
                futures.append({'job_id':j['id'],'arm':j['arm'],'seed':j['seed'],'formation_epoch':j['formation_epoch'],
                    'future_rule':j['future_rule'],'steps':j['expected_steps'],'subset':subset,'label_space':space,
                    'old_ce_K0':old0['ce'],'old_ce_K100':oldk['ce'],'old_ce_increase':oldk['ce']-old0['ce'],
                    'old_accuracy_K0':old0['accuracy'],'old_accuracy_K100':oldk['accuracy'],
                    'new_ce_K100':newk['ce'],'new_accuracy_K100':newk['accuracy']})
results=HERE/'results';results.mkdir(exist_ok=True)
def write_csv(name,rr):
    if not rr:return
    import io
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    atomic_text(results/name,stream.getvalue())
write_csv('formation_endpoints.csv',rows);write_csv('future_endpoints.csv',futures)
pairs=[]
for seed in (0,1,2):
    s=next((r for r in futures if r['arm']=='s' and r['seed']==seed and r['formation_epoch']==50 and r['future_rule']=='s' and r['subset']=='reporting' and r['label_space']=='all_seen'),None)
    g=next((r for r in futures if r['arm']=='g_match' and r['seed']==seed and r['formation_epoch']==50 and r['future_rule']=='s' and r['subset']=='reporting' and r['label_space']=='all_seen'),None)
    if s and g: pairs.append({'seed':seed,'Gmatch_minus_S_old_ce_increase':g['old_ce_increase']-s['old_ce_increase'],
        'Gmatch_minus_S_new_accuracy':g['new_accuracy_K100']-s['new_accuracy_K100']})
atomic_json(results/'progress_summary.json',{'updated_at':now(),'formation_completed':len(rows),'future_trajectories_completed':len(futures)//6,
    'primary_fixed_epoch_pairs':pairs,'mean_old_ce_increase_difference':statistics.mean(p['Gmatch_minus_S_old_ce_increase'] for p in pairs) if len(pairs)==3 else None,
    'phase2_gate_status':'await full diagnostic matched-acquisition/progress audit; never auto-launch phase2/3',
    'missing_values_policy':'absent results omitted','seed2_scope':'exploratory'})
