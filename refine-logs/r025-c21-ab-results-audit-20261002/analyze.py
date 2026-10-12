from pathlib import Path
import json,csv,math,statistics,hashlib
H=Path(__file__).resolve().parent
N=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r025-c21-ab-20261002T080000Z')
O=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z')
rows=[];details={};stages=[]
for root in [O,N]:
 suite=json.loads((root/'suite.json').read_text())
 for job in suite['jobs']:
  if job['smoke'] or job['kind']!='horizon':continue
  out=Path(job['output_dir']);summary=json.loads((out/'final_summary.json').read_text());raw=json.loads((out/'task_matrix.json').read_text());mat=raw['matrix'];n=len(mat)
  assert n==10 and all(len(r)==i+1 and all(math.isfinite(x) for x in r) for i,r in enumerate(mat))
  m=dict(AAA_task_macro=sum(sum(r)/len(r) for r in mat)/n,FAA_task_macro=sum(mat[-1])/n,LA=sum(mat[i][i] for i in range(n))/n,BWT=sum(mat[-1][i]-mat[i][i] for i in range(n-1))/(n-1),forgetting=sum(max(mat[t][i] for t in range(i,n))-mat[-1][i] for i in range(n-1))/(n-1))
  for key,value in m.items():assert abs(value-raw[key])<1e-9,(job['id'],key)
  assert abs(m['FAA_task_macro']-m['LA']-.9*m['BWT'])<1e-9
  records=[json.loads(l) for l in (out/'step_trace.jsonl').read_text().splitlines()];assert len(records)==2550
  ratios=[r['actual_to_reference_norm_ratio'] for r in records]
  row=dict(job_id=job['id'],rule=job['rule'],scale=job['scale'],source='new' if root==N else 'reused',seed=2,**m,AAA_sample_weighted=raw['AAA_sample_weighted'],FAA_sample_weighted=raw['FAA_sample_weighted'],mean_actual_effective_step_ratio_to_own_state_g1=statistics.mean(ratios))
  rows.append(row)
  stages.extend(dict(job_id=job['id'],rule=job['rule'],scale=job['scale'],task=t,stage_task_mean=sum(r)/len(r),diagonal=r[-1]) for t,r in enumerate(mat))
  if root==N:
   a=[r['c21_audit'] for r in records]
   details[job['rule']]=dict(steps=len(records),norm_match_used=False,scale_equals_nominal_every_step=all(r['step_scale']==r['lr_nominal'] for r in records),Fisher_passes=summary['counters']['fisher_estimation_passes'],weight_merge_updates=summary['counters']['weight_merge_updates'],max_orthogonality_error=max(r['orthogonality_relative_error'] for r in a),mean_coefficient=statistics.mean(r['coefficient_a'] for r in a),mean_perp_over_c21=statistics.mean(r['perp_over_c21'] for r in a),mean_cos_c10_c21=statistics.mean(r['cos_c10_c21'] for r in a),mean_effective_ratio=statistics.mean(ratios),median_effective_ratio=statistics.median(ratios),matrix_sha256=hashlib.sha256((out/'task_matrix.json').read_bytes()).hexdigest())
by={r['rule']:r for r in rows if r['scale']=='natural' or r['rule']=='us'}
diffs=[]
for a,b in [('c21_base','u0'),('c21_base','us'),('c21_base','uo'),('c21_base','ug'),('c21_perp','us'),('c21_perp','uo'),('c21_perp','c21_base')]:
 ra,rb=by[a],by[b];d={k:ra[k]-rb[k] for k in ['AAA_task_macro','FAA_task_macro','LA','BWT','forgetting']};d['FAA_from_LA_difference']=d['LA'];d['FAA_from_BWT_difference']=.9*d['BWT'];diffs.append(dict(candidate=a,reference=b,**d))
retained={m:(by['c21_perp'][m]-by['us'][m])/(by['uo'][m]-by['us'][m]) for m in ['AAA_task_macro','FAA_task_macro']}
for fn,rs in [('all_results.csv',rows),('comparisons.csv',diffs),('per_stage.csv',stages)]:
 with (H/fn).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
report=dict(formal_new_validated=2,formal_new_expected=2,old_training_baselines_validated=9,smokes_validated=3,running=0,failed=0,pending=0,new_queue_last_completion_UTC='2026-10-02T10:47:54.283772+00:00',details=details,differences=diffs,B_fraction_of_g2_gain_over_g1=retained,statistical_n=1,sample_sd=None,analysis_note='Natural and matched results are separate protocols; within-step counterfactual ratios are not independent g1 trajectories; single-seed results cannot establish equivalence or Hessian causality.')
(H/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
