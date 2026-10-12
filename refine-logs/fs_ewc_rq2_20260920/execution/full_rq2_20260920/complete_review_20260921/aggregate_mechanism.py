#!/usr/bin/env python3
"""Pure-CPU reaggregation of already-completed RQ2 receipts; no model execution."""
import json,statistics as st,hashlib,math,datetime
from pathlib import Path
O=Path(__file__).resolve().parent;N=O.parent
old=json.load(open(N/'MECHANISM_STATUS_20260921.json'))
def ms(x):return {'mean':st.mean(x),'sample_sd':st.stdev(x) if len(x)>1 else None,'n':len(x),'values_seed_order':x}
def count(a,b):return {'gam_lower':sum(y<x for x,y in zip(a,b)),'tie':sum(y==x for x,y in zip(a,b)),'gam_higher':sum(y>x for x,y in zip(a,b)),'total':len(a)}
def cmp(a,b):return {'sgd':ms(a),'gam':ms(b),'paired_gam_minus_sgd':ms([y-x for x,y in zip(a,b)]),'comparison_count':count(a,b)}
def quantile(x,q):
 x=sorted(x);p=(len(x)-1)*q;i=int(p);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(p-i)
def finite(o):
 if isinstance(o,float):return math.isfinite(o)
 if isinstance(o,dict):return all(finite(v) for v in o.values())
 if isinstance(o,list):return all(finite(v) for v in o)
 return True
D={}; hashes={};checks={};status=[]
for seed in range(3):
 for m in ['sgd','gam']:
  for job in ['dense','sparse_train','sparse_train_full','sparse_test','trajectory_fisher','path_t01','path_t05','path_t09','bridge_full']:
   p=N/f'formal_results/seed{seed}/{m}/{job}/summary.json';raw=p.read_bytes();j=json.loads(raw);hashes[str(p)]=hashlib.sha256(raw).hexdigest();D[seed,m,job]=j
   status.append({'seed':seed,'method':m,'job':job,'status':j['status']})
checks['all_54_complete']=all(x['status']=='complete' for x in status)
checks['all_54_summaries_finite']=all(finite(x) for x in D.values())
checks['prior_53_receipts_unchanged']=all(hashes[k]==v for k,v in old['summary_sha256'].items())
checks['6_dense_full_8001_states']=all(D[s,m,'dense']['states_completed']==D[s,m,'dense']['expected_states']==8001 for s in range(3) for m in ['sgd','gam'])
checks['6_dense_81_radii_100_directions']=all([r['radius'] for r in D[s,m,'dense']['radius_summary']]==[i/4 for i in range(81)] and D[s,m,'dense']['direction_indices']==list(range(100)) for s in range(3) for m in ['sgd','gam'])
checks['dense_restoration_all_passed']=all(D[s,m,'dense']['end_restoration']['max_logit_error']==0 and D[s,m,'dense']['end_restoration']['predictions_match'] for s in range(3) for m in ['sgd','gam'])
checks['dense_scalar_recomputation']=True;checks['dense_tolerance_recomputation']=True
for s in range(3):
 for m in ['sgd','gam']:
  j=D[s,m,'dense']
  for r in j['radius_summary']:
   for sp in ['train','test']:
    v=r[sp]['D_each'];p=[max(0,x) for x in v]
    for k,x in [('D_mean',st.mean(v)),('G',st.mean(p)),('Q95',quantile(p,.95))]:
     if abs(x-r[sp][k])>1e-12:checks['dense_scalar_recomputation']=False
  for t in j['tolerance']:
   radius=0
   for r in j['radius_summary']:
    if r[t['split']][t['metric']]>t['epsilon']:break
    radius=r['radius']
   if radius!=t['prefix_radius']:checks['dense_tolerance_recomputation']=False
radii=[]
for i in range(81):
 rec={'radius':i/4,'direction_count':1 if i==0 else 100}
 for sp in ['train','test']:
  rec[sp]={k:cmp([D[s,'sgd','dense']['radius_summary'][i][sp][k] for s in range(3)],[D[s,'gam','dense']['radius_summary'][i][sp][k] for s in range(3)]) for k in ['D_mean','G','Q95','absolute_loss_mean','accuracy_drop_mean_pp']}
  rec[sp]['paired_direction_damage_counts']={mode:{'per_seed':[count([max(0,x) if mode=='positive' else x for x in D[s,'sgd','dense']['radius_summary'][i][sp]['D_each']],[max(0,x) if mode=='positive' else x for x in D[s,'gam','dense']['radius_summary'][i][sp]['D_each']]) for s in range(3)]} for mode in ['signed','positive']}
 radii.append(rec)
win={}
for sp in ['train','test']:
 win[sp]={}
 for k in ['D_mean','G','Q95']:
  per=[]
  for s in range(3):per.append(count([r[sp][k] for r in D[s,'sgd','dense']['radius_summary'][1:]],[r[sp][k] for r in D[s,'gam','dense']['radius_summary'][1:]]))
  win[sp][k]={'per_seed':per,'all_seed_radius_counts':{c:sum(x[c] for x in per) for c in per[0]},'three_seed_mean_by_radius_counts':count([r[sp][k]['sgd']['mean'] for r in radii[1:]],[r[sp][k]['gam']['mean'] for r in radii[1:]])}
 for mode in ['signed','positive']:
  per=[{c:sum(r[sp]['paired_direction_damage_counts'][mode]['per_seed'][s][c] for r in radii[1:]) for c in ['gam_lower','tie','gam_higher','total']} for s in range(3)]
  win[sp]['directions_'+mode]={'per_seed':per,'all_seed_radius_direction_counts':{c:sum(x[c] for x in per) for c in per[0]}}
tols=[]
for t in D[0,'sgd','dense']['tolerance']:
 sp,k,e=t['split'],t['metric'],t['epsilon']
 v={m:[next(x for x in D[s,m,'dense']['tolerance'] if (x['split'],x['metric'],x['epsilon'])==(sp,k,e)) for s in range(3)] for m in ['sgd','gam']}
 tols.append({'split':sp,'metric':k,'epsilon':e,'radius':cmp([x['prefix_radius'] for x in v['sgd']],[x['prefix_radius'] for x in v['gam']]),'right_censored':{m:[x['right_censored'] for x in v[m]] for m in v}})
# Previously computed sparse/path/trajectory/bridge statistics are valid only if all their immutable source receipts match.
assert checks['prior_53_receipts_unchanged']
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign_completed_utc':'2026-09-21T09:57:36+00:00','scope':'Completed new formal RQ2 three-seed B1/B2; no new training or model evaluation','checks':checks,'all_checks_passed':all(checks.values()),'status':status,'summary_sha256':hashes,'dense_radius_aggregates':radii,'dense_win_counts':win,'dense_tolerance':tols,**{k:old[k] for k in ['B1_sparse','B2_trajectory','B2_paths','B1_bridge_directions','B1_actual_vs_random_at_own_norm']},'paper_claim':'Across three seeds, GAM reduces old-task training-neighborhood positive CE damage and the directional sensitivity of endpoint interpolation paths under continued GAM training, while the corresponding reductions in empirical Fisher trace do not imply uniformly smaller test-neighborhood damage.','limits':['All descriptive means and SD use 3 training seeds; radii and directions are correlated probes, not independent training replicates.','Dense 100-direction loss scan has all 6 branches; sparse Fisher coverage remains its predeclared smaller radius/direction grids.','Paths interpolate endpoints t0 to t1/5/9 and are not chronological optimization paths.','B2 later tasks continue GAM, unlike B3 GAM anchor followed by SGD.','Higher prefix radius means more tolerance, unlike damage quantities where lower is better; right-censored 20 means at least20, not exact tolerance.']}
(O/'RQ2_MECHANISM_FINAL.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_checks_passed':all(checks.values()),'checks':checks,'r20':{sp:{k:radii[-1][sp][k] for k in ['G','Q95']} for sp in ['train','test']},'win':win},indent=2))
