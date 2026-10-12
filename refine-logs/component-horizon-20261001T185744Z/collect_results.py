#!/usr/bin/env python3
"""Validated single-seed screening metrics; deterministic pilot selection, no claims."""
import csv
import io
import json
from pathlib import Path
from common import atomic_json,atomic_text,now
from validate_job import validate_job,receipt_matches
HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text());results=HERE/'results';results.mkdir(exist_ok=True)
rows=[];diagnostics=[]
for job in suite['jobs']:
    if job['smoke']:continue
    p=HERE/'runtime/jobs'/job['id']/'completed.json'
    if not p.exists():continue
    receipt=json.loads(p.read_text());actual=validate_job(job)
    if receipt['job_id']!=job['id'] or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,actual):raise ValueError('result lacks valid completion receipt '+job['id'])
    s=actual['summary']
    if job['kind']=='curvature':
        diagnostics.append({'job':job['id'],'task':job['task'],'rule':job['rule'],'hvp_all_available':s['hvp_all_available']});continue
    m=s['final_metrics']
    rows.append({'job_id':job['id'],'rule':job['rule'],'scale':job['scale'],'seed':job['seed'],'radius':job['radius'],
        **{key:m[key] for key in ('AAA_task_macro','FAA_task_macro','LA','BWT','forgetting','AAA_sample_weighted','FAA_sample_weighted')},
        'max_match_relative_error':s['max_match_relative_error'],'effective_reference':'candidate own-state projected g1','completed_tasks':s['completed_tasks']})
if rows:
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(results/'screen_metrics.csv',stream.getvalue())
base=next((r for r in rows if r['rule']=='us' and r['scale']=='matched'),None)
comparisons=[];chosen=None;complete=len(rows)==9
if base:
    for row in rows:
        if row['scale']!='matched' or row['rule'] not in ('ug','uo','ufull'):continue
        aaa=row['AAA_task_macro']-base['AAA_task_macro'];faa=row['FAA_task_macro']-base['FAA_task_macro']
        comparisons.append({'rule':row['rule'],'AAA_gain_pp':aaa,'FAA_gain_pp':faa,'LA_gain_pp':row['LA']-base['LA'],
            'BWT_gain_pp':row['BWT']-base['BWT'],'eligible_for_three_seed_confirmation':aaa>=1. and faa>=-1.})
    if complete:
        eligible=[c for c in comparisons if c['eligible_for_three_seed_confirmation']]
        if eligible:chosen=sorted(eligible,key=lambda c:(-c['AAA_gain_pp'],-c['FAA_gain_pp'],['uo','ug','ufull'].index(c['rule'])))[0]['rule']
paired=[]
for rule in ('u0','ug','uo','ufull'):
    a=next((r for r in rows if r['rule']==rule and r['scale']=='natural'),None)
    b=next((r for r in rows if r['rule']==rule and r['scale']=='matched'),None)
    if a and b:paired.append({'rule':rule,'natural_minus_matched_AAA_pp':a['AAA_task_macro']-b['AAA_task_macro'],
        'natural_minus_matched_FAA_pp':a['FAA_task_macro']-b['FAA_task_macro']})
atomic_json(results/'screen_decision.json',{'updated_at':now(),'fresh_runs_validated':len(rows),'fresh_runs_expected':9,
    'diagnostics_validated':diagnostics,'pilot_comparisons':comparisons,'chosen_X':chosen if complete else None,
    'screen_complete':complete,'natural_vs_matched_same_cohort':paired,
    'interpretation':'single seed exploratory screening; no statistical equivalence or Hessian mediation claim',
    'phase2_phase3_auto_launch':False,'next_step':'independent 3-seed SS/XS/SX/XX if X selected; else diagnose magnitude/schedule'})
