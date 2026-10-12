#!/usr/bin/env python3
"""CPU only. Current anchors do NOT identify the original historical anchors."""
import hashlib,json,math,datetime
from pathlib import Path
import torch
torch.set_num_threads(4)
OUT=Path(__file__).resolve().parent;ROOT=Path('/data/140-1/users/liying/CopyFlatCV')
source=json.loads((OUT/'full_sources.json').read_text()); anchors={}; info={}
def shaf(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for m in ['sgd','gam']:
 p=ROOT/f'outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_{m}_probe_lr0.01_e50/imagenetr/0/stage1_{m}_probe_e50_lam2000_s0/exp_run/checkpoints/task_00.pt'
 x=torch.load(p,map_location='cpu',weights_only=False);anchors[m]=x['network_state']
 info[m]=dict(path=str(p),sha256=shaf(p),mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),fisher_trace=sum(v.double().sum().item() for v in x['fisher'].values()),fisher_numel=sum(v.numel() for v in x['fisher'].values()))
results=[]; heads={}
for row in source['rows']:
 name=row['method']; m='sgd' if name in ['baseline','control','factored_s2'] else 'gam'
 p=Path(row['current_final_checkpoint']); b=torch.load(p,map_location='cpu',weights_only=False)['model_state_dict'];a=anchors[m]
 e={'no_fc':0,'all_head_zero_pad':0,'head_old_rows':0,'head_new_rows':0,'qv_weight':0}; count={k:0 for k in e}
 for k,v in b.items():
  ref=a[k]
  if v.shape!=ref.shape:
   padded=torch.zeros_like(v);padded[:ref.shape[0]]=ref;diff=(v-padded).double()
  else:diff=(v-ref).double()
  en=diff.square().sum().item();e['all_head_zero_pad']+=en;count['all_head_zero_pad']+=v.numel()
  if k.startswith('fc.'):
   e['head_old_rows']+=diff[:20].square().sum().item();e['head_new_rows']+=diff[20:].square().sum().item();count['head_old_rows']+=diff[:20].numel();count['head_new_rows']+=diff[20:].numel()
  else:
   e['no_fc']+=en;count['no_fc']+=v.numel()
   if k.endswith('attn.qkv.weight'):
    for z in [diff[:768],diff[1536:]]:e['qv_weight']+=z.square().sum().item();count['qv_weight']+=z.numel()
 head=torch.cat([b[k][:20].reshape(-1) for k in ['fc.weight','fc.bias']])
 if m not in heads:heads[m]=head
 old_head_equals_others=torch.equal(head,heads[m])
 inferred=row['delta_norm_sq']-e['head_new_rows']
 results.append(dict(method=name,family=row['family'],table_selected=row['table_selected'],metrics_path=row['metrics_path'],checkpoint_path=str(p),anchor_method=m,energy=e,l2={k:math.sqrt(v) for k,v in e.items()},counts=count,logged_energy=row['delta_norm_sq'],current_anchor_identity_caveat=True,old_head_bitwise_equals_all_other_same_anchor_family=old_head_equals_others,historical_backbone_energy_inferred_assuming_frozen_old_head=inferred,historical_backbone_l2_inferred_assuming_frozen_old_head=math.sqrt(inferred),new_classifier_fraction_of_logged_energy=e['head_new_rows']/row['delta_norm_sq']))
(OUT/'full_checkpoint_recompute.json').write_text(json.dumps(results,indent=2))
for m in ['gam','sgd']:
 p=next((ROOT/f'outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/stage1_factored_{m}_a_lr0.01_e50').rglob('task_00.pt')); f=torch.load(p,map_location='cpu',weights_only=False)['fisher']
 info['sibling_'+m]=dict(path=str(p),min=min(v.min().item() for v in f.values()),exact_1e5_floor_count=sum((v==torch.tensor(1e-5)).sum().item() for v in f.values()),numel=sum(v.numel() for v in f.values()),trace=sum(v.double().sum().item() for v in f.values()))
(OUT/'full_anchor_identity.json').write_text(json.dumps(info,indent=2)); print('wrote CPU audit with overwritten-anchor caveat')
