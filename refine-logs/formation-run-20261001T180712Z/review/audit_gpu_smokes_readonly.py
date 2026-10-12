#!/usr/bin/env python3
"""Read deployment only; CPU checkpoint comparisons; writes this review directory."""
import hashlib,json,math,sys
from pathlib import Path
from datetime import datetime,timezone
import torch

DEPLOY=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z')
REVIEW=Path('/data/140-1/users/liying/O-LoRA/refine-logs/formation-run-20261001T180712Z/review')
sys.dont_write_bytecode=True
sys.path.insert(0,str(DEPLOY))
from validate_job import validate_job,receipt_matches
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=lambda p:json.loads(Path(p).read_text())
assert not torch.cuda.is_available(),'Run with CUDA_VISIBLE_DEVICES empty'
torch.set_num_threads(4)
suite=r(DEPLOY/'suite.json');pins=r(DEPLOY/'RUN_PINS.json')
assert all(h(DEPLOY/n)==v for n,v in pins['files'].items())
source_manifest=r(suite['source_manifest']);source=Path(suite['source_root'])
assert h(suite['source_manifest'])==suite['source_manifest_sha256']
assert all(h(source/n)==v for n,v in source_manifest['files'].items())
smokes=[j for j in suite['jobs'] if j['smoke']]
assert len(smokes)==12
receipts={};rows={};identities={};records=[]
for job in smokes:
 rp=DEPLOY/'runtime/jobs'/job['id']/'completed.json';receipt=r(rp);actual=validate_job(job)
 assert receipt['job_id']==job['id'] and receipt['exit']['job_id']==job['id'] and receipt['exit']['returncode']==0
 assert receipt_matches(receipt,actual)
 assert r(DEPLOY/'runtime/jobs'/job['id']/'exit.json')==receipt['exit']
 out=Path(job['output_dir']);row=r(out/'step_trace.jsonl');identity=r(out/'identity.json')
 rows[job['id']]=row;identities[job['id']]=identity;receipts[job['id']]=h(rp)
 assert row['self_spec_passed'] and row['base_sgd_exact'] and row['decomposition_spec_exact'] and row['projector_spec_exact'] and row['closure_count']==4
 records.append({'id':job['id'],'completed_at':receipt['completed_at'],'receipt_sha256':h(rp),'summary_sha256':actual['summary_sha256'],'trace_sha256':actual['step_trace_sha256'],'checkpoint_manifest_and_files_verified':len(actual['checkpoints']),
 'nominal_lr':row['lr_nominal'],'actual_step_scale':row['step_scale'],'effective_delta':row['actual_effective_delta'],'probe_roundtrip_delta':row['probe_roundtrip_delta'],'actual_match_relative_error':row.get('actual_match_relative_error')})
formation=[j for j in smokes if j['kind']=='formation' and j['seed']==2]
assert len(formation)==8
pairkeys=['initial_model_sha256','frozen_origin_sha256','class_order','task_range']
base=identities['smoke_s_s2'];base_batch=rows['smoke_s_s2']['identity']
for j in formation:
 ident=identities[j['id']]
 assert all(ident[k]==base[k] for k in pairkeys)
 assert rows[j['id']]['identity']==base_batch

def state(cp):
 aa=torch.load(cp/'lora_w_a_0.pt',map_location='cpu',weights_only=False)
 bb=torch.load(cp/'lora_w_b_0.pt',map_location='cpu',weights_only=False)
 fc=torch.load(cp/'fc_state_0.pt',map_location='cpu',weights_only=False)
 assert len(aa)==len(bb)>0
 keys=sorted(fc)
 par=[v.detach().double() for pair in zip(aa,bb) for module in pair for v in module.parameters()]+[fc[k].detach().double() for k in keys]
 eff=[b.weight.detach().double()@a.weight.detach().double() for a,b in zip(aa,bb)]+[fc[k].detach().double() for k in keys]
 assert all(bool(torch.isfinite(x).all()) for x in par+eff)
 return par,eff
initial=None
for j in formation:
 par,_=state(Path(j['output_dir'])/'checkpoints/epoch_000')
 if initial is None:initial=par
 else:assert len(par)==len(initial) and all(a.shape==b.shape and torch.equal(a,b) for a,b in zip(par,initial))
sp,se=state(DEPLOY/'outputs/smoke_sum_s2/checkpoints/epoch_001')
mp,me=state(DEPLOY/'outputs/smoke_mean_lr2_s2/checkpoints/epoch_001')
_,be=state(DEPLOY/'outputs/smoke_sum_s2/checkpoints/epoch_000')
assert len(sp)==len(mp) and len(se)==len(me)==len(be)
assert all(a.shape==b.shape==c.shape for a,b,c in zip(se,me,be))
sumdiff=math.sqrt(sum(float((a-b).square().sum()) for a,b in zip(se,me)))
sumdelta=math.sqrt(sum(float((a-b).square().sum()) for a,b in zip(se,be)))
rel=sumdiff/max(sumdelta,1e-30);assert math.isfinite(rel) and rel<=1e-3
sg=rows['smoke_g_match_s2'];ss=rows['smoke_s_s2'];assert sg['reference_target_norm']==ss['actual_effective_delta']['joint_norm']
gerr=abs(sg['actual_effective_delta']['joint_norm']-ss['actual_effective_delta']['joint_norm'])/ss['actual_effective_delta']['joint_norm'];assert gerr<=1e-3
future=[j for j in smokes if j['kind']=='future'];assert len(future)==2
fj=identities[future[0]['id']];fk=identities[future[1]['id']]
for key in ['origin_parameter_sha256','checkpoint_dir','checkpoint_files_sha256','selection_split_sha256','task1_train_samples','task1_batches_per_epoch','class_order','data_seed_rule','training_rng_seed','lr_schedule']:
 assert fj[key]==fk[key],key
assert fj['expansion']['new_rows_sha256']==fk['expansion']['new_rows_sha256']
x,y=rows[future[0]['id']],rows[future[1]['id']]
assert x['identity']==y['identity'] and x['lr_nominal']==y['lr_nominal']
assert x['training_forward_count']==y['training_forward_count']==5
for j in future:
 ev=r(Path(j['output_dir'])/'evaluations.json')
 for space in ['local','all_seen']:
  assert ev['milestones']['0']['task0']['diagnostic'][space]['ce']==rows[j['id']]['old_loss_path'][space]['baseline_k0']
 assert ev['milestones']['0']==r(Path(future[0]['output_dir'])/'evaluations.json')['milestones']['0'],'same-origin baseline differs'
 matchj=next(z for z in future if z['future_rule']=='g_match')
 sj=next(z for z in future if z['future_rule']=='s')
assert rows[matchj['id']]['norm_match']['target']==rows[sj['id']]['actual_effective_delta']['joint_norm']
ferr=abs(rows[matchj['id']]['actual_effective_delta']['joint_norm']-rows[sj['id']]['actual_effective_delta']['joint_norm'])/rows[sj['id']]['actual_effective_delta']['joint_norm'];assert ferr<=1e-3
state_now=r(DEPLOY/'runtime/queue_state.json')
assert all(state_now['jobs'][j['id']]['status']=='completed' for j in smokes)
gate_path=DEPLOY/'runtime/GPU_SMOKES_VERIFIED.json';gate=None
if gate_path.exists():
 gate=r(gate_path)
 assert gate['status']=='passed' and gate['verified_smokes']==12 and gate['suite_sha256']==h(DEPLOY/'suite.json') and gate['receipt_sha256']==receipts
 assert math.isclose(gate['Sum_vs_Mean2_relative_to_actual_update'],rel,rel_tol=1e-9,abs_tol=1e-12)
result={'status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),'deployment':str(DEPLOY),'suite_sha256':h(DEPLOY/'suite.json'),'run_pins_sha256':h(DEPLOY/'RUN_PINS.json'),
 'verified_deployment_files':len(pins['files']),'verified_source_files':len(source_manifest['files']),'verified_completed_smokes':12,'gpu_jobs_launched_by_reviewer':0,'review_device':'cpu',
 'seed2_formation_arms_initial_parameter_tensors_exact':8,'seed2_formation_arms_initial_model_and_every_batch_identity_equal':8,
 'Sum_vs_Mean2_effective_difference':sumdiff,'Sum_effective_delta_from_initial':sumdelta,'Sum_vs_Mean2_relative_to_actual_update':rel,
 'Sum_vs_Mean2_max_parameter_abs_difference':max(float((a-b).abs().max()) for a,b in zip(sp,mp)),
 'formation_Gmatch_independent_actual_S_delta_relative_error':gerr,'future_S_Gmatch_origin_head_data_LR_and_K0_pairing':'exact','future_Gmatch_independent_actual_S_delta_relative_error':ferr,
 'receipts':receipts,'smoke_records':records,'root_gpu_gate_present':gate is not None,'root_gpu_gate_hash_matches_current_suite_and_receipts':gate is not None,
 'root_gpu_gate_sha256':h(gate_path) if gate else None,'formal_job_statuses_at_audit':{k:v['status'] for k,v in state_now['jobs'].items() if not k.startswith('smoke') and v['status']!='pending'},
 'limitations':['Single real GPU update per smoke; does not establish full50epoch numerical equivalence or scientific formation effect.','start_background formal gate checks status/count only; this audit additionally binds the fresh suite/receipt hashes and pins.','This reviewer did not run root smoke checker or formal launcher and did not write deployment files.']}
(REVIEW/'GPU_SMOKES_INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','smoke_records','formal_job_statuses_at_audit']},indent=2))
