#!/usr/bin/env python3
from pathlib import Path
import json,csv,re,hashlib,math,statistics,datetime
ROOT=Path('/data/140-1/users/liying/CopyFlatCV');OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'fs_ewc_rq2_20260920';TAG='ratiofs_l12c20_full_twostage_s345_v1'
def info(p,hash_file=True):
 p=Path(p);r={'path':str(p),'exists':p.exists()}
 if p.exists():
  s=p.stat();r.update(bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat())
  if hash_file:r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 return r
def cfglog(p):
 out={}
 for line in Path(p).read_text().splitlines()[:140]:
  m=re.match(r'.*\[trainer.py\] => ([A-Za-z_0-9]+): (.*)',line)
  if m:out[m[1]]=m[2]
 return out
out={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'read only source; no GPU/training/process mutation','imageneta':{},'imagenetr_ratio':[],'imagenetr_fixedfs':[],'current_diagnostics':{}}
assets=json.loads((OLD/'task_assets_manifest.json').read_text());traj=list(csv.DictReader((OLD/'task_trajectory_recomputed.csv').open()))
out['imageneta']={'prior_tensor_audit':info(OLD/'task_assets_manifest.json'),'prior_results':info(OLD/'task_trajectory_recomputed.csv'),'all_checkpoints_still_exist':all(Path(x['checkpoint']).exists() for x in assets['assets']),'checkpoints':[info(x['checkpoint'],False) for x in assets['assets']], 'metrics_sources':{m:info(next(x['metrics_json'] for x in traj if x['method']==m)) for m in ['sgd','gam']},'selected_prior_cpu_results':[x for x in traj if x['task'] in ['0','1','9']]}
for setting in ['baseline','fs_only','control','fs_ewc','factored_s1','factored_s2']:
 for seed in [3,4,5]:
  matches=list((ROOT/'outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage').glob(f'**/ratiofs_full_twostage_{setting}_{TAG}/imagenetr/{seed}/*/exp_run/20/*cl_metrics.json'));assert len(matches)==1
  met=matches[0];x=json.loads(met.read_text());a=x['cnn']['matrices']['final'];m=x['mechanism']['steps']['1'];log=ROOT/'logs/verify_imagenet_r_1x1_lora_ratio_full_twostage'/TAG/'stage2'/f'{setting}_seed{seed}.log';cfg=cfglog(log);anchor=Path(cfg['resume_from_checkpoint_dir']);end=met.parent.parent/'checkpoints'
  checks={f'{where}_{file}':info(d/file,False) for where,d,task in [('anchor',anchor,0),('endpoint',end,1)] for file in [f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'fc_state_{task}.pt',f'lora_meta_{task}.json']}
  out['imagenetr_ratio'].append({'setting':setting,'seed':seed,'metrics':info(met),'log':info(log),'logged_config':cfg,'anchor_dir':str(anchor),'endpoint_dir':str(end),'checkpoint_asset_files':checks,'a11':a[0][0],'a21':a[1][0],'a22':a[1][1],'BWT':a[1][0]-a[0][0],'delta_norm_squared':m['delta_norm'],'r_eff_from_logged_energy':math.sqrt(m['delta_norm']),'pointers':{'a11':'/cnn/matrices/final/0/0','a21':'/cnn/matrices/final/1/0','a22':'/cnn/matrices/final/1/1','delta_norm_squared':'/mechanism/steps/1/delta_norm'},'fisher_fields':{k:v for k,v in m.items() if 'fisher' in k and any(w in k for w in ['trace','raw','norm','damp'])}})
for p in (ROOT/'outputs_logs/verify_imagenet_r_1x1_lora_fixedfs').rglob('*cl_metrics.json'):
 if 'fixedfs_exploratory_s012_a005_lam4e5_rho005_v1/imagenetr/0/' not in str(p) or 'stage2_' not in str(p):continue
 x=json.loads(p.read_text());a=x['cnn']['matrices']['final'];m=x['mechanism']['steps'].get('1',{})
 if len(a)!=2:continue
 out['imagenetr_fixedfs'].append({'metrics':info(p),'a11':a[0][0],'a21':a[1][0],'a22':a[1][1],'delta_norm_squared':m.get('delta_norm'),'r_eff_from_logged_energy':math.sqrt(m['delta_norm']) if 'delta_norm' in m else None,'endpoint_assets':[info(p.parent.parent/'checkpoints'/f'lora_w_{z}_1.pt',False) for z in ['a','b']]})
r=OLD/'execution';sp=r/'campaign_seed0_dualline_gpu03/pipeline_state.json'
if sp.exists():
 x=json.loads(sp.read_text());out['current_diagnostics']['pipeline_snapshot']={'source':info(sp),'updated_utc':x['updated_utc'],'status':x['status'],'jobs':{k:v.get('status') for k,v in x['jobs'].items()},'error':x.get('error')}
for method in ['sgd','gam']:
 p=r/'results_seed0'/method/'sanity/summary.json';ap=p.parent.parent/'anchor_measurement.json'
 if p.exists():
  x=json.loads(p.read_text());a=json.loads(ap.read_text());out['current_diagnostics'][method]={'sanity':info(p),'status':x['status'],'passed':x['passed'],'anchor':info(ap),'anchor_fisher':a['fisher'],'anchor_metrics':a['metrics'],'data_manifest_sha256':a['data_manifest_sha256'],'protocol_sha256':a['protocol_sha256']}
with (OUT/'new_lora_sources.json').open('w') as f:json.dump(out,f,indent=2,ensure_ascii=False)
summary=[]
for setting in ['baseline','fs_only','control','fs_ewc','factored_s1','factored_s2']:
 rr=[x for x in out['imagenetr_ratio'] if x['setting']==setting]
 summary.append({'setting':setting,'n':len(rr),**{k:statistics.mean(x[k] for x in rr) for k in ['a11','a21','a22','BWT','delta_norm_squared','r_eff_from_logged_energy']},'all_paired_sidecars_present':all(v['exists'] for x in rr for v in x['checkpoint_asset_files'].values())})
with (OUT/'new_imagenetr_ratio_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(json.dumps(summary,indent=2))
