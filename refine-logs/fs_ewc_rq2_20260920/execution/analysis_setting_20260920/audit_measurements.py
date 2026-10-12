#!/usr/bin/env python3
"""Read-only CPU audit of final RQ2 measurements; writes only analysis outputs."""
import csv, hashlib, json, math, statistics
from pathlib import Path
B=Path(__file__).resolve().parents[1]
R=B/'results_seed0_recovery1'; O=Path(__file__).resolve().parent
jread=lambda p:json.loads(Path(p).read_text())
def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8<<20),b''): h.update(c)
 return h.hexdigest()
def canon(j): return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def nonfinite(j,path=''):
 if isinstance(j,float) and not math.isfinite(j): return [path]
 if isinstance(j,dict): return sum([nonfinite(v,path+'/'+str(k)) for k,v in j.items()],[])
 if isinstance(j,list): return sum([nonfinite(v,path+'/'+str(k)) for k,v in enumerate(j)],[])
 return []
def quantile(v,q):
 v=sorted(v); p=(len(v)-1)*q; a=int(p); b=min(a+1,len(v)-1)
 return v[a]+(v[b]-v[a])*(p-a)
def rows(p): return list(csv.DictReader(open(p)))
report=jread(B/'report_seed0/MANIFEST.json')
checks={}; artifacts={}
for k,v in report['sources'].items():
 actual=filehash(v['path']); checks['receipt_hash/'+k]=actual==v['sha256']; artifacts[v['path']]=actual
checks['report_config_hash']=filehash(report['config'])==report['config_sha256']
checks['postprocess_code_hash']=filehash(B/'postprocess.py')==report['code_sha256']
finite=[]; statecounts={}; manifests={}; neighborhood={}; A={}; trajectories={}; paths={}; bridges={}
for m in ['sgd','gam']:
 for job in ['sanity','neighborhood','bridge','trajectory','path','table_a']:
  s=jread(R/m/job/'summary.json'); finite+=nonfinite(s,f'{m}/{job}/summary')
  states=R/m/job/'states.jsonl'
  if states.exists():
   ss=[json.loads(x) for x in states.read_text().splitlines() if x.strip()]
   statecounts[m+'/'+job]=len(ss)
   for x in ss: finite+=nonfinite(x,f'{m}/{job}/state/{x["state_id"]}')
   checks['unique_states/'+m+'/'+job]=len(set(x['state_id'] for x in ss))==len(ss)
   checks['protocol_match/'+m+'/'+job]=all(x['protocol_sha256']==s['protocol_sha256'] for x in ss)
  if job=='neighborhood': neighborhood[m]=s
  elif job=='table_a': A[m]=s
  elif job=='path': paths[m]=s
  elif job=='trajectory': trajectories[m]=s
  elif job=='bridge': bridges[m]=s
 for sub in ['', 'table_a/']:
  p=R/m/sub/'data_manifest.json'; d=jread(p); stored=d.pop('manifest_sha256'); checks['data_manifest_hash/'+m+'/'+sub]=canon(d)==stored; d['manifest_sha256']=stored; manifests[m+'/'+sub]=d
 rp=B/f'training/A_{m}_to_sgd_seed0/rng_receipts.jsonl'
 checks['rng_artifact_hash/'+m]=filehash(rp)==report['A_paired_rng']['artifacts'][m]['sha256']
checks['all_diagnostic_numbers_finite']=not finite
checks['data_manifests_byte_identical']=len(set(filehash(R/m/'data_manifest.json') for m in ['sgd','gam']))==1
checks['four_data_manifests_equal']=all(x==next(iter(manifests.values())) for x in manifests.values())
checks['paired_rng_receipts_byte_identical']=filehash(B/'training/A_sgd_to_sgd_seed0/rng_receipts.jsonl')==filehash(B/'training/A_gam_to_sgd_seed0/rng_receipts.jsonl')
stateby={m:{x['state_id']:x for x in [json.loads(z) for z in (R/m/'neighborhood/states.jsonl').read_text().splitlines()]} for m in ['sgd','gam']}
paired=[k for k in stateby['sgd'] if k!='r0']
checks['neighborhood_96_perturbed_states_paired']=len(paired)==96 and set(stateby['sgd'])==set(stateby['gam'])
checks['neighborhood_direction_hashes_paired']=all(stateby['sgd'][k]['point']['direction']==stateby['gam'][k]['point']['direction'] for k in paired)
recompute_errors=[]
for m in ['sgd','gam']:
 origin=stateby[m]['r0']
 for r in neighborhood[m]['radius_summary']:
  selected=[origin] if r['radius']==0 else [x for x in stateby[m].values() if x['point'].get('radius')==r['radius']]
  for split in ['train','test']:
   dmg=[x['metrics'][split]['loss']-origin['metrics'][split]['loss'] for x in selected]; positive=[max(0,x) for x in dmg]
   for key,val in [('D_mean',statistics.mean(dmg)),('G',statistics.mean(positive)),('Q95',quantile(positive,.95))]:
    if abs(val-r[split][key])>1e-12: recompute_errors.append([m,r['radius'],split,key,val,r[split][key]])
checks['neighborhood_summaries_recomputed']=not recompute_errors
local={m:[json.loads(x) for x in (B/f'training/A_{m}_to_sgd_seed0/task0_local_metrics.jsonl').read_text().splitlines()] for m in ['sgd','gam']}
# The task labels use sorted ImageFolder classes (class_shuffle=false); count files with supported image suffixes.
data=Path('/data/115-2/users/liying/datasets/imagenet-a'); counts={}; extensions={'.jpg','.jpeg','.png','.ppm','.bmp','.pgm','.tif','.tiff','.webp'}
for split in ['train','test']:
 cls=sorted(x for x in (data/split).iterdir() if x.is_dir())
 counts[split]={f'task{t}':sum(sum(1 for f in d.rglob('*') if f.is_file() and f.suffix.lower() in extensions) for d in cls[t*20:(t+1)*20]) for t in [0,1]}
new_local={'sgd':88.19,'gam':87.40}; detail={}
for m in ['sgd','gam']:
 a=A[m]; nold=counts['test']['task0']; nnew=counts['test']['task1']
 oldlocal=round(local[m][-1]['test']['local_accuracy']*nold/100); oldcil=round(local[m][-1]['test']['seen_accuracy']*nold/100)
 newlocal=round(new_local[m]*nnew/100); newcil=round(a['accuracy_CIL']['a22']*nnew/100)
 detail[m]={
  'CIL':a['accuracy_CIL'],'macro_two_task_accuracy_from_rounded_matrix':(a['accuracy_CIL']['a21']+a['accuracy_CIL']['a22'])/2,
  'exact_count_macro_accuracy':(oldcil/nold+newcil/nnew)*50,'exact_count_micro_accuracy':100*(oldcil+newcil)/(nold+nnew),
  'old_task_correct':{'anchor_local':round(a['accuracy_CIL']['a11']*nold/100),'endpoint_local':oldlocal,'endpoint_CIL':oldcil,'n':nold,'local_to_CIL_additional_errors':oldlocal-oldcil},
  'new_task_correct':{'endpoint_local':newlocal,'endpoint_CIL':newcil,'n':nnew,'local_to_CIL_additional_errors':newlocal-newcil,'count_evidence':'Inferred exact integer correct counts from rounded accuracy with known 254-image denominator; logs verified against local-head evaluation code.'},
  'old_task_local':{'anchor':a['anchor']['metrics'],'endpoint':a['endpoint']['metrics'],'D0':a['D0'],'D0_subset':a['D0_subset']},
  'raw_trace_t0':a['anchor']['fisher']['raw_trace'],'raw_trace_t1':a['endpoint']['fisher']['raw_trace'],'effective_weight_drift':a['effective_weight_drift'],
  'old_test_local_metrics_training_receipt':local[m],
 }
 mech=jread(a['metrics_json'])['mechanism']['steps']['1']
 detail[m]['training_mechanism']={k:mech[k] for k in ['train_ce_clean_mean','train_ewc_mean','ewc_penalty_value','delta_norm','fisher_past_trace_mass','fisher_mode','fisher_estimator_family','fisher_is_per_sample_empirical']}
 checks['A_local_endpoint_matches_training/'+m]=a['endpoint']['metrics']['test']['accuracy']==local[m][-1]['test']['local_accuracy'] and a['endpoint']['metrics']['test']['loss']==local[m][-1]['test']['local_ce']
 checks['new_local_rounded_count_matches/'+m]=round(100*newlocal/nnew,2)==new_local[m]
 checks['new_CIL_rounded_count_matches/'+m]=round(100*newcil/nnew,2)==a['accuracy_CIL']['a22']
 checks['training_delta_norm_is_squared/'+m]=abs(mech['delta_norm']-a['effective_weight_drift']**2)<1e-8
comparisons=[]
for s,g in zip(neighborhood['sgd']['radius_summary'],neighborhood['gam']['radius_summary']):
 rec={'radius':s['radius'],'raw_trace_sgd':s['raw_trace_mean'],'raw_trace_gam':g['raw_trace_mean'],'raw_trace_ratio':g['raw_trace_mean']/s['raw_trace_mean']}
 for split in ['train','test']:
  rec[split]={key:{'sgd':s[split][key],'gam':g[split][key],'gam_minus_sgd':g[split][key]-s[split][key]} for key in ['D_mean','G','Q95']}
 comparisons.append(rec)
C={}
for m in ['sgd','gam']:
 p=paths[m]; rr=p['path']; q=[x['q'] for x in rr]; I=(q[0]+q[-1]+4*sum(q[1:-1:2])+2*sum(q[2:-1:2]))/(3*(len(q)-1))
 checks['path_integral_recomputed/'+m]=abs(I-p['integral'])<1e-12
 checks['path_normalization_recomputed/'+m]=abs(I/p['drift']**2-p['S_path'])<1e-12
 C[m]={'path':p,'task0':trajectories[m]['tasks'][0],'task9':trajectories[m]['tasks'][-1]}
summary={'audit_scope':'Read-only CPU recomputation, no GPU or experiment mutation','report_time_utc':report['created_utc'],'checks':checks,'all_checks_passed':all(checks.values()),'state_counts':statecounts,'nonfinite_diagnostic_fields':finite,'summary_recompute_errors':recompute_errors,'artifact_hashes':artifacts,'data_counts':counts,'A':detail,'A_difference_using_rounded_matrix':{'old_CIL_pp':84.04-84.36,'new_CIL_pp':72.05-74.8,'macro_pp':78.045-79.58,'new_task_share_of_gap':(74.8-72.05)/((84.36-84.04)+(74.8-72.05))},'B_comparisons':comparisons,'B_tolerance':{m:neighborhood[m]['tolerance'] for m in ['sgd','gam']},'B_bridge':bridges,'C':C,'limitations':['Single training seed only; directions are not independent training seeds.','No per-example prediction IDs were exported here; count decompositions do not assert sample overlap.','Bridge and C use old Task0-to-Task9 checkpoints, not the new A Task1 endpoint.','Different inherited EWC matrices remain; no unique causal attribution to starting flatness.','Formal diagnostic values are finite; undefined fields for untrained task pairs and unrun NME in training metrics JSON remain NaN by design.']}
(O/'MEASUREMENT_ANALYSIS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'all_checks_passed':all(checks.values()),'checks':len(checks),'failed':{k:v for k,v in checks.items() if not v},'state_counts':statecounts,'data_counts':counts,'new_task_detail':{m:detail[m]['new_task_correct'] for m in ['sgd','gam']}},indent=2))
