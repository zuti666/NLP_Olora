"""Verify real training/diagnostic artifacts before accepting any dependency."""
import json,math,sys
from pathlib import Path
from common import HERE as H,sha256,atomic_json,load_suite

def require(v,msg):
 if not v:raise AssertionError(msg)
def finite(v):
 if isinstance(v,float):require(math.isfinite(v),'Nonfinite artifact')
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def validate_job(j):
 out=Path(j['output_dir']);report=out/j.get('report_file','final_summary.json');s=json.loads(report.read_text());finite(s)
 if j['kind'] in ('prepare','training'):
  sys.path.insert(0,str(H/'trainer'))
  from validate_chain import validate_output,validate_pair
  checked=validate_output(j['config'],out)
  require(s['completed_steps']==j['expected_steps'],'Actual committed update budget differs')
  require(s['branch']==j['chain_branch'],'Registered branch differs')
  require(s['trainer_sha256']==sha256(H/'trainer/response_chain_train.py') and s['algorithm_sha256']==sha256(H/'trainer/chain_step.py'),'Actual trainer/kernel hash differs')
  if j['kind']=='training':
   require(s['completed_epochs']==j['expected_epochs'],'Actual epoch budget differs')
   require(s['counters']['training_ce_gradient_passes']==j['expected_gradient_reads'],'Actual gradient read budget differs')
   traces=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
   for r in traces:
    finite(r)
    require(r['natural_sgd_exact'] and r['base_sgd_max_abs_error']==0. and r['sgd_updates']==1,'Natural SGD update differs')
    require(r['class_slice']==[20,40] and r['seen_classes']==40 and r['fc_shape'][0]==40,'Native loss/classifier scope differs')
    require(all(r['state_restore_checks'].values()) and r['only_base_R1_rng_advance'],'Probe state/RNG restoration failed')
   # Every paired formal trace is certified after both branches have reached validated endpoints.
   state=json.loads((H/'runtime/queue_state.json').read_text())
   sibling=next(x for x in load_suite()['jobs'] if x['kind']=='training' and x['seed']==j['seed'] and x['smoke']==j['smoke'] and x['id']!=j['id'])
   if state['jobs'][sibling['id']]['status']=='completed':
    pair=validate_pair(out,sibling['output_dir'])
    atomic_json(H/'runtime'/f"pair_{'smoke' if j['smoke'] else 'formal'}_s{j['seed']}.json",pair)
  result=dict(summary_sha256=sha256(report),validated_checks=checked['checks'],updates=j['expected_steps'],gradient_reads=j['expected_gradient_reads'],full_formal_training=j.get('formal',False))
 elif j['kind']=='pairing':
  require(s['status']=='passed' and s['smoke_count']==2 and all(s['checks'].values()),'Actual paired smoke gate failed')
  for path,digest in s['evidence_hashes'].items():require(sha256(path)==digest,'Paired evidence changed')
  result=dict(summary_sha256=sha256(report),gate='passed',full_formal_training=False)
 elif j['kind']=='diagnostic':
  require(s['status']=='accepted' and s['submitted_training_updates']==0 and s['numerical_gate_passed'] and s['state_restoration_passed'],'Offline numerical gate failed')
  p=out/'diagnostic_results.json';require(s['final_results_sha256']==sha256(p),'Offline evidence hash differs');d=json.loads(p.read_text());finite(d)
  require(d['seed']==j['seed'] and d['smoke']==j['smoke'] and d['submitted_training_updates']==0,'Offline registered identity/purity differs')
  require(s['completed_combinations']==(1 if j['smoke'] else 12),'Offline state/batch budget differs')
  if not j['smoke']:
   require({(r['branch'],r['epoch'],r['batch_index']) for r in d['results']}=={(b,e,k) for b in ['S','R'] for e in [1,10,25] for k in [0,1]},'Incomplete/doubled formal offline combinations')
  for path,digest in s['checkpoint_sha256'].items():require(sha256(path)==digest,'Offline checkpoint changed')
  result=dict(summary_sha256=sha256(report),combinations=s['completed_combinations'],training_updates=0,full_formal_training=False)
 else:raise ValueError('Unknown kind')
 rt=H/'runtime/jobs'/j['id'];w=json.loads((rt/'cuda_witness.json').read_text());launch=json.loads((rt/'launched.json').read_text())
 require(w['status']=='passed' and w['torch']=='2.4.1+cu121' and not w['tf32_matmul'] and not w['tf32_cudnn'],'Native FP32 CUDA witness failed')
 require(launch['gpu'] in [4,5,6] and w['cuda_visible_devices']==launch['gpu_uuid'],'Witness physical GPU isolation failed')
 return result
