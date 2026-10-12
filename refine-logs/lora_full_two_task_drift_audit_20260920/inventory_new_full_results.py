#!/usr/bin/env python3
import collections,csv,datetime,hashlib,json,math,re
from pathlib import Path
ROOT=Path('/data/140-1/users/liying/CopyFlatCV');OUT=Path(__file__).resolve().parent
bases=['outputs_logs/fullmodel_imageneta_sgd','outputs_logs/fullmodel_imageneta_aligned','outputs_logs/fullmodel_imageneta_finetune','outputs_logs/imageneta_sgd_lr_sweep','outputs_logs/imageneta_ewc_fullmodel_lambda_sweep','outputs_logs/config_FullModel','outputs_logs/verify_imagenet_r_1x1_fullmodel_ratio_twostage','outputs_logs/fishershape_imagenetr_1x1_fullmodel']
rows=[]
for base in bases:
 for p in (ROOT/base).rglob('*cl_metrics.json'):
  if 'imageneta' not in str(p) and 'fullmodel_ratio_twostage' not in str(p) and 'fishershape_imagenetr' not in str(p):continue
  try:j=json.loads(p.read_text());mat=j['cnn']['matrices']['final']
  except Exception:continue
  n=len(mat);finite=all(math.isfinite(mat[i][k]) for i in range(n) for k in range(i+1));a11=mat[0][0] if n else None;last=sum(mat[-1][:n])/n if n and finite else None;bwt=sum(mat[-1][i]-mat[i][i] for i in range(n-1))/(n-1) if n>1 and finite else None
  logs=list(p.parent.glob('*.log'));log=logs[0] if logs else None;cfg={}
  if log:
   with log.open(errors='replace') as f:txt=''.join(next(f,'') for _ in range(250))
   cfg=dict(re.findall(r'\[trainer.py\] => ([a-zA-Z0-9_]+): ([^\n]+)',txt))
  ckdir=p.parent.parent/'checkpoints';cks=[str(z) for z in ckdir.glob('*') if z.suffix in ['.pt','.pth','.pkl','.safetensors']]
  steps=j.get('mechanism',{}).get('steps',{});t1=steps.get('1',{});t0=steps.get('0',{})
  rows.append(dict(base=base,run=p.name.replace('_vit_base_patch16_224_cl_metrics.json',''),dataset=cfg.get('dataset','imageneta' if 'imageneta' in str(p) else 'imagenetr'),model=cfg.get('model_name',''),seed=cfg.get('seed',''),optimizer=cfg.get('optimizer_type',cfg.get('optimizer','')),ewc_lambda=cfg.get('ewc_lambda',''),fisher_mode=cfg.get('fisher_mode',''),epochs=cfg.get('epochs',''),lr=cfg.get('lrate',''),completed_matrix_tasks=n,finite_lower_triangle=finite,complete_10task=(n==10 and finite),a11=a11,a21=mat[1][0] if n>1 else None,a22=mat[1][1] if n>1 else None,macro_final=last,bwt=bwt,checkpoint_files=len(cks),task1_logged_drift_sq=t1.get('delta_norm'),task1_ewc_include_fc=t1.get('ewc_include_fc'),metrics_path=str(p),metrics_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),log_path=str(log) if log else '',mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),checkpoints=cks))
for r in rows:
 lp=Path(r['log_path']);examples=[]
 if lp.is_file():
  for line in lp.open(errors='replace'):
   if re.search(r'(?:Loss|loss|CE|EWC)[\w /:=.\-]*\b(?:nan|inf)\b',line,re.I): examples.append(line[:300])
 r['training_nonfinite_log_lines']=len(examples);r['training_nonfinite_examples']=examples[:2]
(OUT/'new_full_results_inventory.json').write_text(json.dumps(rows,indent=2,allow_nan=True))
fields=[k for k in rows[0] if k not in ['checkpoints','training_nonfinite_examples']]
with (OUT/'new_full_results_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
for b in bases:
 x=[v for v in rows if v['base']==b];print(b,'records',len(x),'complete10',sum(v['complete_10task'] for v in x),'weightfiles',sum(v['checkpoint_files'] for v in x))
for x in rows:
 if x['base'] in bases[:3] and x['complete_10task']: print(x['run'],x['macro_final'],x['bwt'],x['a11'],x['a21'],x['a22'],x['seed'])
