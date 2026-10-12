#!/usr/bin/env python3
"""Read-only extraction of historical full-model two-task table provenance."""
import csv, datetime, hashlib, json, math, re
from pathlib import Path
ROOT=Path('/data/140-1/users/liying/CopyFlatCV')
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'GAM-EWC/results/section6_two_phase_20260917.csv').open()))
rows+=list(csv.DictReader((ROOT/'GAM-EWC/results/section6_two_phase_old_20260917.csv').open()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
res=[]
for row in rows:
 p=ROOT/row['path']; log=ROOT/row['log']; j=json.loads(p.read_text()); m=j['mechanism']['steps']['1']; txt=log.read_text(errors='replace')
 cfg={a:b for a,b in re.findall(r'\[trainer.py\] => ([a-zA-Z0-9_]+): ([^\n]+)',txt) if a in ['ewc_lambda','ewc_gamma','ewc_normalize_fisher','ewc_include_fc','fisher_mode','model_name','seed','epochs','init_epoch','lrate','init_lr','batch_size','gam_grad_rho','gam_grad_norm_rho','gam_grad_gamma','resume_from_checkpoint_dir','resume_from_metrics_json','config','use_ewc','optimizer_type','phase_a_tasks','gam_task_a','gam_task_b','momentum','weight_decay']}
 rowout=dict(method=row['method'],family='old' if 'family' in row else 'unified',table_selected=('family' in row or row['method'] in ['control','factored_full','factored_s1','factored_s2']),metrics_path=str(p),metrics_sha256=sha(p),log_path=str(log),log_sha256=sha(log),config=cfg,accuracy_matrix=j['cnn']['matrices']['final'],a11=float(row['a11']),a21=float(row['a21']),a22=float(row['a22']),bwt=float(row['bwt']),delta_norm_sq=m['delta_norm'],drift_l2=math.sqrt(m['delta_norm']),task2_endpoint_fisher_mass=m['endpoint_fisher_total_mass'],task2_accumulated_fisher_mass=m['accumulated_fisher_total_mass'],inferred_prior_fisher_mass=(m['accumulated_fisher_total_mass']-m['endpoint_fisher_total_mass'])/.9,fields={k:v for k,v in m.items() if any(s in k for s in ['delta','fisher','param'])})
 rowout['current_final_checkpoint']=str(p.parent.parent/'checkpoints/final_model.pt')
 rowout['current_final_checkpoint_sha256']=sha(Path(rowout['current_final_checkpoint']))
 rowout['current_final_checkpoint_mtime_utc']=datetime.datetime.fromtimestamp(Path(rowout['current_final_checkpoint']).stat().st_mtime,datetime.timezone.utc).isoformat()
 rowout['log_final_save_records']=[z for z in txt.splitlines() if 'Saved final model to' in z or '[FinalSave]' in z]
 rowout['checkpoint_identity_binding']='same full path, tasks=1, 40-class head, present-day SHA256 and file mtime recorded; no historical hash; not cryptographically proven original artifact'
 res.append(rowout)
(OUT/'full_sources.json').write_text(json.dumps({'source_root':str(ROOT),'accuracy_field':'cnn.matrices.final','drift_field':'sqrt(mechanism.steps.1.delta_norm)','trace2_field':'mechanism.steps.1.accumulated_fisher_total_mass','note':'Table Task1 trace matches implied prior via (cumulative2-endpoint2)/0.9. Original stage1 path has been overwritten. These are historical buffer statistics, not raw sample Fisher.','rows':res},indent=2,allow_nan=True))
fields=['method','family','a11','a21','a22','bwt','delta_norm_sq','drift_l2','task2_endpoint_fisher_mass','task2_accumulated_fisher_mass','inferred_prior_fisher_mass','metrics_path','log_path']
with (OUT/'full_table_raw_fields.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(res)
print('wrote',OUT/'full_sources.json')
