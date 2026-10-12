#!/usr/bin/env python3
import csv,datetime,gc,hashlib,json,math,re
from pathlib import Path
import torch
torch.set_num_threads(4)
ROOT=Path('/data/140-1/users/liying/CopyFlatCV'); OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs_logs/sgd_gam_s1c_fullmodel'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def norms(a,b):
 e=dict(backbone=0.,kv_weights=0.,old_head=0.,new_head_vs_zero=0.)
 old=a['fc.weight'].shape[0]
 for k,v in b.items():
  if k.startswith('fc.'):
   e['old_head']+=(v[:old]-a[k]).double().square().sum().item();e['new_head_vs_zero']+=v[old:].double().square().sum().item()
  else:
   d=(v-a[k]).double();e['backbone']+=d.square().sum().item()
   if k.endswith('attn.qkv.weight'):e['kv_weights']+=d[768:].square().sum().item()
 return {k+'_sq':v for k,v in e.items()}|{k+'_l2':math.sqrt(v) for k,v in e.items()}
rows=[];manifest=[]
for method in ['sgd','gam']:
 s1=BASE/'stage1_task0/task0_fm_lam2000_lr0p01_e50';s2=BASE/'stage2_T/T_from_task0_fm_lam2000_lr0p01_e50'
 p0=s1/f'{method}/seed0/checkpoints/task_00.pt';logs=ROOT/f'logs/sgd_gam_s1c_fullmodel/staged/stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/shell/{method}_T10_seed0.log'
 logtxt=logs.read_text(errors='replace');cfg=dict(re.findall(r'\[trainer.py\] => ([a-zA-Z0-9_]+): ([^\n]+)',logtxt))
 assert Path(cfg['resume_from_checkpoint_dir'])/'task_00.pt'==p0
 metrics=next((s2/'trainer_outputs').glob(f'logs_inc/ewcon_probe/{method}_fm_T10_lr0.01_e50/imagenetr/0/*/exp_run/20/*cl_metrics.json')); j=json.loads(metrics.read_text());mat=j['cnn']['matrices']['final'];assert len(mat)==10
 s1metrics=Path(cfg['resume_from_metrics_json']);j0=json.loads(s1metrics.read_text());assert j0['cnn']['matrices']['final'][0][0]==mat[0][0]
 x0=torch.load(p0,map_location='cpu',weights_only=False);anchor=x0['network_state']; prev=anchor
 assert x0['task_idx']==0 and x0['known_classes']==20
 assert not any(k.startswith('fc.') for k in x0['fisher'])
 assert all(torch.equal(v,anchor[k]) for k,v in x0['checkpoint_params'].items())
 amass=sum(v.double().sum().item() for v in x0['fisher'].values());assert abs(amass-j0['mechanism']['steps']['0']['accumulated_fisher_total_mass'])<1e-4
 manifest.append(dict(method=method,config={k:cfg.get(k) for k in ['dataset','seed','model_name','optimizer_type','use_ewc','ewc_lambda','ewc_gamma','fisher_mode','ncm_fisher_tau','epochs','lrate','batch_size','resume_from_checkpoint_dir','config']},metrics_path=str(metrics),metrics_sha256=sha(metrics),anchor_metrics_path=str(s1metrics),log_path=str(logs),log_sha256=sha(logs),anchor_path=str(p0),anchor_sha256=sha(p0),anchor_fisher_mass=amass,anchor_accuracy=mat[0][0],anchor_params_numel=sum(v.numel() for k,v in anchor.items() if not k.startswith('fc.')),kv_numel=12*2*768*768,checkpoints=[]))
 del x0;gc.collect()
 for t in range(1,10):
  p=s2/f'{method}/T10/seed0/checkpoints/task_{t:02}.pt';x=torch.load(p,map_location='cpu',weights_only=False);b=x['network_state'];assert x['task_idx']==t and x['known_classes']==(t+1)*20
  assert all(torch.equal(v,b[k]) for k,v in x['checkpoint_params'].items());assert not any(k.startswith('fc.') for k in x['fisher'])
  step=norms(prev,b);cum=norms(anchor,b);m=j['mechanism']['steps'][str(t)];err=step['backbone_sq']-m['delta_norm']; assert abs(err)<1e-5,(method,t,err)
  trace=sum(v.double().sum().item() for v in x['fisher'].values());assert abs(trace-m['accumulated_fisher_total_mass'])<2e-4
  save=[z for z in logtxt.splitlines() if f'Saved task {t} to' in z];assert save
  mt=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc);logdt=datetime.datetime.strptime(save[-1][:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=2)));timeerr=(logdt-mt).total_seconds()
  assert abs(timeerr)<2
  rows.append(dict(method=method,task=t,anchor_task=0,old_task_cil_accuracy=mat[t][0],anchor_accuracy=mat[0][0],old_accuracy_delta=mat[t][0]-mat[0][0],current_task_accuracy=mat[t][t],logged_step_backbone_sq=m['delta_norm'],logged_step_match_abs_error=abs(err),**{'step_'+k:v for k,v in step.items()},**{'cumulative_'+k:v for k,v in cum.items()},old_head_train_init_delta_available=True,new_head_train_init_delta_available=False))
  manifest[-1]['checkpoints'].append(dict(task=t,path=str(p),sha256=sha(p),mtime_utc=mt.isoformat(),logged_save=save[-1],save_time_delta_sec=timeerr,backbone_drift_log_error=err,fisher_trace=trace,fisher_trace_log_error=trace-m['accumulated_fisher_total_mass']))
  prev=b;del x;gc.collect();print(method,t,'backbone',cum['backbone_l2'],'KV',cum['kv_weights_l2'],'step-error',err,flush=True)
with (OUT/'new_full_checkpoint_drift.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'new_full_checkpoint_manifest.json').write_text(json.dumps({'protocol':'ImageNet-R full model SGD/GAM, ncm_empirical EWC, June22/23 historical sequence; not the earlier six-row table','rows':manifest,'new_head_distance_note':'new_head_vs_zero is zero-padding convention; actual newly initialized head snapshot was not saved, so init-to-end head movement unavailable; old head differences are exact checkpoint-to-checkpoint','identity_note':'same-run resume paths, Task0 metrics/fisher, all 18 logged step drifts, fisher traces, metadata and save timestamps matched; current SHA256 captured; no historical execution SHA'},indent=2))
print('SAVED',OUT/'new_full_checkpoint_drift.csv',flush=True)
