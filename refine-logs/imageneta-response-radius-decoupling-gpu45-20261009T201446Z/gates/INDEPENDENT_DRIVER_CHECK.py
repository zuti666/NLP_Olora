from pathlib import Path
import sys, json, importlib.util, copy, hashlib
from types import SimpleNamespace
NEW=Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z')
BASE=NEW.parent/'imageneta-response-four-methods-gpu567-20261009T103449Z'
sys.path.insert(0,str(NEW/'trainer'))
import response_train as new
spec=importlib.util.spec_from_file_location('independent_legacy_driver_for_review',BASE/'trainer/response_train.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/response_radius_review_unused')
checks={}
for layout in ('single200','cl10'):
 for method in ('sgd','current_g2','history_fusion','history_g2'):
  path=BASE/'configs'/f'formal_{layout}_{method}_s2.json'
  if not path.exists():
   path=Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/configs')/f'formal_{layout}_{method}_s2.json'
  raw=json.loads(path.read_text());a=new.resolve_config(copy.deepcopy(raw),cli);b=old.resolve_config(copy.deepcopy(raw),cli)
  assert a==b,(layout,method,{k:(a.get(k),b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)})
  checks[f'original_{layout}_{method}_entire_resolved_config_identical']=True
raw=json.loads((BASE/'configs/formal_cl10_current_g2_s2.json').read_text())
for label,im,om in [('FixedInner','fixed_l2','sgd_step'),('FixedOuter','sgd_step','fixed_l2'),('FixedBoth','fixed_l2','fixed_l2')]:
 d=copy.deepcopy(raw);d.update(response_radius_mode='independent_inner_outer_v1',response_probe_variant='CurrentG2-'+label,inner_probe_radius_mode=im,outer_probe_radius_mode=om,inner_probe_radius=.05 if im=='fixed_l2' else None,outer_probe_radius=.05 if om=='fixed_l2' else None)
 cfg=new.resolve_config(d,cli)
 assert cfg['history_source'] is None and cfg['history_weight']==0
 for key in ['lora_rank','init_epoch','epochs','init_lr','lrate','cosine_min_lr_ratio','class_shuffle','batch_size','init_cls','increment','ewc_lambda','ewc_skip_task_fisher','in_task_weight_merge','task_endpoint_weight_merge','attention_backend','component_scale']:
  assert cfg[key]==new.resolve_config(raw,cli)[key],(label,key)
 assert cfg['inner_probe_radius']==d['inner_probe_radius'] and cfg['outer_probe_radius']==d['outer_probe_radius']
 bad=copy.deepcopy(d);bad['outer_probe_radius']=.1 if om=='fixed_l2' else .05
 try:new.resolve_config(bad,cli);raise AssertionError('bad radius was accepted')
 except ValueError:pass
 bad=copy.deepcopy(d);bad['response_method']='history_g2';bad['history_weight']=.5;bad['history_source']='final'
 try:new.resolve_config(bad,cli);raise AssertionError('history variant accepted')
 except ValueError:pass
 checks[label+'_configuration_targets_only_geometry_and_rejects_unregistered_radius_history']=True
bad=copy.deepcopy(raw);bad.update(inner_probe_radius_mode='fixed_l2',inner_probe_radius=.05)
try:new.resolve_config(bad,cli);raise AssertionError('silently ignored independent config')
except ValueError:pass
checks['original_config_cannot_silently_ignore_new_radius_fields']=True
report={'status':'passed','checks':checks,'trainer_sha256':hashlib.sha256((NEW/'trainer/response_train.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'independent exact original/new resolved config comparison; registered variant CE/layout/optimizer/baseline and negative config acceptance checks'}
Path('/tmp/radius45_independent_driver_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
