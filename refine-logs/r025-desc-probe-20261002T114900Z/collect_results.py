"""Collect validated natural A/B and reused r=.25 baselines; never partial scores."""
import csv,io,json
from pathlib import Path
from common import atomic_json,atomic_text,now
from validate_job import validate_job,receipt_matches
H=Path(__file__).resolve().parent
s=json.loads((H/'suite.json').read_text());(H/'results').mkdir(exist_ok=True)
proof=json.loads((H/'runtime/BASELINES_VALIDATED.json').read_text());rows=[]
keys=('AAA_task_macro','FAA_task_macro','LA','BWT','forgetting','AAA_sample_weighted','FAA_sample_weighted')
for r in proof['baselines']:
 if r['smoke'] or (r['scale']!='natural' and r['rule']!='us'):continue
 rows.append(dict(job_id=r['job_id'],rule=r['rule'],provenance='reused validated reference',scale='natural' if r['rule']!='us' else 'nominal LR identity (reused matched g1)',**{k:r['summary']['final_metrics'][k] for k in keys}))
fresh=[]
for j in s['jobs']:
 if j['smoke']:continue
 p=H/'runtime/jobs'/j['id']/'completed.json'
 if not p.exists():continue
 receipt=json.loads(p.read_text());v=validate_job(j)
 if receipt['job_id']!=j['id'] or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,v):raise ValueError('invalid completion '+j['id'])
 row=dict(job_id=j['id'],rule=j['rule'],provenance='new full A/B training',scale=j['scale'],**{k:v['summary']['final_metrics'][k] for k in keys})
 rows.append(row);fresh.append(row)
stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic_text(H/'results/comparison.csv',stream.getvalue())
comparisons=[]
for row in fresh:
 for ref in rows:
  if ref['rule'] not in ('u0','us','uo'):continue
  comparisons.append(dict(rule=row['rule'],reference=ref['rule'],AAA_gain_pp=row['AAA_task_macro']-ref['AAA_task_macro'],FAA_gain_pp=row['FAA_task_macro']-ref['FAA_task_macro'],BWT_gain_pp=row['BWT']-ref['BWT']))
atomic_json(H/'results/status.json',dict(updated_at=now(),completed_full=len(fresh),expected_full=2,comparisons=comparisons,interpretation='Single seed natural-step controls. Removing the parallel residual does not match effective update magnitude. No Hessian mediation claim.',additional_full_jobs_auto_launch=False))
