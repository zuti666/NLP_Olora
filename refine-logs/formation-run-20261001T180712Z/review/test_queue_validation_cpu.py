#!/usr/bin/env python3
"""Strict CPU validator regression: valid output passes; targeted corruptions fail.

Uses synthetic JSON/checkpoint bytes, never runs a model/queue/GPU. Each negative
case starts from an independently accepted valid fixture; KeyError and unrelated
validation failures are test failures. Optional --lease repeats the CPU FD test.
"""
import argparse,copy,fcntl,hashlib,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'trainer')]
sys.dont_write_bytecode=True
from validate_job import validate_job,receipt_matches
import formation_task0
import future_probe
SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,default=str)+'\n')
def codes():
 return {k:sha(ROOT/'trainer'/v) for k,v in {'trainer_sha256':'formation_task0.py','effective_step_sha256':'effective_step.py','base_trainer_sha256':'task0_trainer_base_v2.py','audit_core_sha256':'audit_core.py'}.items()}
def metrics(counts):
 n=sum(counts)
 return {'ce':1.,'accuracy':0.,'samples':n,'correct':0,'macro_accuracy':0.,'classes_present':sum(c>0 for c in counts),'per_class_counts':counts,'per_class_correct':[0]*20,'per_class_accuracy':[0.]*20}
def task_eval(spec,task):
 d=spec['task%d'%task];allcounts=[d['per_class_counts'][str(i)] for i in range(task*20,(task+1)*20)];diag=[d['diagnostic_per_class_counts'][str(i)] for i in range(task*20,(task+1)*20)]
 return {split:{space:metrics(counts) for space in ('local','all_seen')} for split,counts in [('full',allcounts),('diagnostic',diag),('reporting',[a-b for a,b in zip(allcounts,diag)])]}

class Fixture:
 def __init__(self,path,kind='formation',arm='s',rule='s',smoke=True):
  self.path=path;self.out=path/'output';self.out.mkdir(parents=True)
  self.kind,self.arm,self.rule,self.smoke=kind,arm,rule,smoke
  suite=json.loads((ROOT/'suite.json').read_text());want=('smoke_'+arm+'_s2' if kind=='formation' and smoke else 'formation_'+arm+'_s2' if kind=='formation' else ('smoke_future_s_e01_'+rule+'_s2' if smoke else 'future_s_e50_'+rule+'_s2'))
  self.job=copy.deepcopy(next(j for j in suite['jobs'] if j['id']==want));self.job['output_dir']=str(self.out)
  raw=json.loads((ROOT/'configs'/Path(self.job['config']).name).read_text());raw['outputs_root']=str(self.out)
  self.config_path=path/'input_config.json';write(self.config_path,raw)
  self.job.update(config=str(self.config_path),config_sha256=sha(self.config_path))
  steps=self.job['expected_steps'];epochs=self.job.get('expected_epochs',50)
  cli=SimpleNamespace(source=str(SOURCE),output_dir=str(self.out),max_epochs=1 if smoke else 0,max_batches=1 if smoke else 0,seed=2,device='cpu',future_rule=rule)
  self.cfg=(formation_task0.resolve_config(raw,cli) if kind=='formation' else future_probe.resolve_config(raw,cli))
  self.cfg['device']=['cuda:0']
  self.rows=[];self.spec=json.loads((ROOT/'selection_split.json').read_text());self.identity={};self.evaluations={};self.epoch_rows=[]
  command=list(self.job['command'])
  for flag,value in (('--source',str(SOURCE)),('--config',str(self.config_path)),('--output-dir',str(self.out))):command[command.index(flag)+1]=value
  self.job['command']=command
  common={'status':'completed','seed':2,'formation_arm':arm,'smoke':smoke}
  if kind=='formation':
   self.grid=[0,1] if smoke else [0,10,20,30,40,50]
   self.summary={**common,'formation_schema_version':1,'task':0,'trained_tasks':[0],'completed_epochs':epochs,'scheduler_T_max':50,
     'counters':{'training_batches':steps,'training_ce_gradient_passes':5*steps,'training_samples':128 if smoke else 1259*50,'isolated_diagnostic_ce_gradient_passes':0,'independent_sam_probe_gradient_passes':0},
     'finite_check':{'passed':True},'task_check':{'passed':True,'cur_task':0,'total_classes':20},'step_checks':{'passed':True,'steps':steps,'matched_steps':steps if arm=='g_match' else 0,'max_match_relative_error':0. if arm=='g_match' else None},
     'checkpoint_checks':{str(e):True for e in self.grid},'checkpoint_epochs':self.grid,'step_trace_steps':steps,'source_gam_sha256':sha(SOURCE/'optimer_PerturabtionType/gam.py'),**codes()}
   self.identity={'initial_model_sha256':'a'*64,'frozen_origin_sha256':'b'*64,'task_range':[0,20],'class_order':list(range(200)),'native_scheduler':'CosineAnnealingLR','scheduler_initial':{'T_max':50,'eta_min':self.cfg['init_lr']*.1}}
   for e in range(1,epochs+1):
    lr=self.cfg['init_lr']*(.1+.45*(1+math.cos(math.pi*(e-1)/50)))
    self.epoch_rows.append({'epoch':e,'task':0,'train_samples':128 if smoke else 1259,'batches':1 if smoke else 10,'eval_samples':307,'lr_used':[lr],'lr_after_scheduler':[self.cfg['init_lr']*(.1+.45*(1+math.cos(math.pi*e/50)))],'eval_accuracy':0.,'eval_ce':1.,'train_ce':1.,'train_accuracy':0.})
   self.summary['final']=copy.deepcopy(self.epoch_rows[-1]);self.summary['initial_identity']=copy.deepcopy(self.identity)
   for e in self.grid:self.make_checkpoint(self.out/'checkpoints'/('epoch_%03d'%e),e)
  else:
   cp=path/'origin_checkpoint';self.make_checkpoint(cp,self.job['formation_epoch'])
   command[command.index('--checkpoint-dir')+1]=str(cp);command[command.index('--selection-split')+1]=str(ROOT/'selection_split.json')
   source_hashes=json.loads((ROOT/'source_manifest.json').read_text())['files']
   self.identity={'schema_version':1,'seed':2,'formation_arm':arm,'formation_epoch':self.job['formation_epoch'],'future_rule':rule,'smoke':smoke,
    'source':str(SOURCE),'source_python_sha256':source_hashes,'source_gam_sha256':sha(SOURCE/'optimer_PerturabtionType/gam.py'),'trainer_sha256':sha(ROOT/'trainer/future_probe.py'),
    'audit_core_sha256':sha(ROOT/'trainer/audit_core.py'),'effective_step_sha256':sha(ROOT/'trainer/effective_step.py'),'config_sha256':sha(self.config_path),'checkpoint_dir':str(cp),
    'checkpoint_files_sha256':{p.name:sha(p) for p in cp.iterdir()},'origin_parameter_sha256':'a'*64,'selection_split_sha256':sha(ROOT/'selection_split.json'),'selection_split':self.spec,
    'expansion':{'head_seed':220000,'old_rows_exact':True,'backbone_exact':True,'new_rows_sha256':{'weight':'c'*64,'bias':'d'*64},'expanded_parameter_sha256':'e'*64},
    'task1_train_samples':997,'task1_batches_per_epoch':8,'batch_size':128,'training_old_data':False,'class_order':list(range(200)),
    'data_seed_rule':'seed*100000+10000+epoch (epoch 1-based)','training_rng_seed':230000,'lr_schedule':{'start':.02,'eta_min':.002,'T_max':50,'unit':'completed native epochs'}}
   milestones=[0,1] if smoke else [0,1,10,50,100]
   self.evaluations={'pre_expand':{'task0':task_eval(self.spec,0)},'milestones':{str(k):{'task0':task_eval(self.spec,0),'task1':task_eval(self.spec,1)} for k in milestones}}
   self.summary={**common,'schema_version':1,'task':1,'formation_epoch':self.job['formation_epoch'],'future_rule':rule,'completed_steps':steps,'requested_steps':steps,'checkpoint_unchanged':True,'source_unchanged':True,
    'completed_full_epochs':steps//8,'completed_fractional_epochs':steps/8,'counters':{'training_steps':steps,'ce_gradient_passes':5*steps,'fisher_estimation_passes':0,'weight_merge_calls':0,'task_end_hooks':0},
    'milestones_completed':milestones,'final_evaluation':copy.deepcopy(self.evaluations['milestones'][str(steps)]),'match_passed':True if rule=='g_match' else None,'max_actual_norm_match_relative_error':0.,'training_samples':128 if smoke else 12*997+4*128}
  for i in range(steps):
   epoch,batch=(1,0) if smoke else (i//(10 if kind=='formation' else 8)+1,i%(10 if kind=='formation' else 8))
   lr=self.cfg['init_lr']*(.1+.45*(1+math.cos(math.pi*(epoch-1)/50))) if kind=='formation' else .002+.009*(1+math.cos(math.pi*(epoch-1)/50))
   row={'step_index':i if kind=='formation' else i+1,'seed':2,'task':0 if kind=='formation' else 1,'formation_arm':arm,'epoch':epoch,'batch':batch,
    'self_spec_passed':True,'closure_count':4,'training_ce_gradient_passes':5,'training_forward_count':5,'base_sgd_exact':True,'decomposition_spec_exact':True,'projector_spec_exact':True,
    'actual_effective_delta':{'joint_norm':.01,'lora_norm':.008,'fc_norm':.006,'parameter_norm':.02},'probe_roundtrip_delta':{'joint_norm':1e-8,'lora_norm':1e-8,'fc_norm':0.,'parameter_norm':1e-8},
    'lr_nominal':lr,'step_scale':lr,'scale_over_nominal_lr':1.,'base_sgd_max_abs_error':0.,'fisher_gradient_passes':0,'weight_merge_calls':0,
    'identity':{'input_sha256':'1'*64,'target_sha256':'2'*64,'indices_sha256':'3'*64,'indices':list(range(128)),'data_seed':200000+(10000 if kind=='future' else 0)+epoch}}
   matched=(arm=='g_match' if kind=='formation' else rule=='g_match')
   row.update(matched=True if matched else None,actual_match_relative_error=0. if matched else None,reference_target_norm=.01 if matched else None,norm_match={'matched':True,'target':.01,'scale':lr} if matched else None)
   if kind=='future':row.update(formation_epoch=self.job['formation_epoch'],future_rule=rule,batch_samples=101 if batch==7 else 128,old_loss_path={k:{'delta_loss':0.,'I':0.,'Iplus':0.,'baseline_k0':1.,'telescoping_error':0.,'steps':i+1} for k in ('local','all_seen')},diagnostic={k:v['diagnostic'] for k,v in self.evaluations['milestones']['0'].items()})
   self.rows.append(row)
  self.save()
 def make_checkpoint(self,cp,epoch):
  cp.mkdir(parents=True)
  for n in ('lora_w_a_0.pt','lora_w_b_0.pt','lora_meta_0.json','fc_state_0.pt','training_state.pt'):(cp/n).write_bytes(b'SYNTHETIC_CPU_VALIDATOR_FIXTURE_NOT_A_MODEL')
  write(cp/'manifest.json',{'epoch':epoch,'task':0,'seed':2,'formation_arm':self.arm,'class_range':[0,20],'model_sha256':'a'*64,'source_gam_sha256':sha(SOURCE/'optimer_PerturabtionType/gam.py')})
 def save(self):
  (self.out/'step_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows));self.summary['step_trace_sha256']=sha(self.out/'step_trace.jsonl')
  write(self.out/'identity.json',self.identity);self.summary['identity_sha256']=sha(self.out/'identity.json')
  write(self.out/'effective_config.json',self.cfg)
  if self.kind=='future':
   write(self.out/'evaluations.json',self.evaluations);self.summary['evaluations_sha256']=sha(self.out/'evaluations.json');write(self.out/'summary.json',self.summary)
  else:write(self.out/'metrics.json',{'epochs':self.epoch_rows,'counters':self.summary['counters']})
  write(self.out/'final_summary.json',self.summary)

def accepted(fixture):
 result=validate_job(fixture.job)
 assert result['summary_sha256']==sha(fixture.out/'final_summary.json')
 return 'accepted'
def rejected(fixture,reason):
 try:validate_job(fixture.job)
 except ValueError as exc:
  assert reason in str(exc),('wrong rejection reason',reason,repr(exc))
  return str(exc)
 raise AssertionError('Corrupted fixture was accepted; expected '+reason)

def formation_tests(tmp):
 results={}
 for arm,smoke in [('s',True),('mean_lr2',True),('g_match',True),('s',False)]:
  label=arm+('_smoke' if smoke else '_formal');f=Fixture(tmp/label,arm=arm,smoke=smoke);results[label]=accepted(f)
 cases=[
 ('empty_checkpoint_grid',lambda f:f.summary.update(checkpoint_epochs=[],checkpoint_checks={}), 'checkpoint grid'),
 ('false_base_sgd',lambda f:f.rows[0].update(base_sgd_exact=False),'own-step verification base_sgd_exact'),
 ('false_decomposition',lambda f:f.rows[0].update(decomposition_spec_exact=False),'own-step verification decomposition_spec_exact'),
 ('false_projector',lambda f:f.rows[0].update(projector_spec_exact=False),'own-step verification projector_spec_exact'),
 ('wrong_epoch_batch',lambda f:f.rows[0].update(epoch=99,batch=99),'epoch/batch sequence'),
 ('wrong_schedule',lambda f:f.cfg.update(init_lr=100.,epochs=1,cosine_min_lr_ratio=1.),'nominal LR'),
 ('wrong_floor',lambda f:f.cfg.update(cosine_min_lr_ratio=1.),'cosine schedule'),
 ('wrong_direction',lambda f:f.cfg.update(gam_grad_beta_3=1.),'direction/radius'),
 ('false_closure_count',lambda f:f.rows[0].update(closure_count=2),'unverified GAM'),
 ('nonfinite_trace',lambda f:f.rows[0].update(observed=float('nan')),'nonfinite'),
 ('wrong_runner_hash',lambda f:f.summary.update(trainer_sha256='0'*64),'runner source'),
 ('wrong_smoke_flag',lambda f:f.summary.update(smoke=False),'smoke/formal')]
 for label,mutation,reason in cases:
  f=Fixture(tmp/label);accepted(f);mutation(f);f.save();results[label]=rejected(f,reason)
 f=Fixture(tmp/'missing_checkpoint');accepted(f);(f.out/'checkpoints/epoch_001/fc_state_0.pt').unlink();results['missing_checkpoint']=rejected(f,'native checkpoint file')
 f=Fixture(tmp/'bad_match',arm='g_match');accepted(f);f.rows[0]['actual_match_relative_error']=.02;f.save();results['bad_match']=rejected(f,'unmatched effective step')
 return results

def future_tests(tmp):
 results={}
 for rule,smoke in [('s',True),('g_match',True),('s',False),('g_match',False)]:
  label=rule+('_smoke' if smoke else '_formal');f=Fixture(tmp/label,kind='future',rule=rule,smoke=smoke);results[label]=accepted(f)
 # Expected reasons intentionally match the validator's relevant guard; never
 # count missing fields/KeyError or an unrelated earlier rejection as success.
 cases=[
 ('wrong_budget',lambda f:f.identity.update(task1_train_samples=996),'future data budget'),
 ('old_training_data',lambda f:f.identity.update(training_old_data=True),'old examples'),
 ('missing_milestone',lambda f:f.summary.update(milestones_completed=[0]),'future milestone'),
 ('false_closure',lambda f:f.rows[0].update(closure_count=3),'native probes'),
 ('nonfinite_metric',lambda f:f.evaluations['milestones']['1']['task0']['reporting']['local'].update(ce=float('nan')),'nonfinite'),
 ('false_base_sgd',lambda f:f.rows[0].update(base_sgd_exact=False),'future update verification'),
 ('wrong_epoch_batch',lambda f:f.rows[0].update(epoch=99,batch=99),'future data step sequence'),
 ('wrong_future_direction',lambda f:f.cfg.update(gam_grad_beta_1=0.,gam_grad_beta_3=1.),'future gradient/radius'),
 ('wrong_source_hash',lambda f:f.identity.update(trainer_sha256='0'*64),'future script pins'),
 ('missing_evaluation_milestone',lambda f:f.evaluations['milestones'].pop('1'),'milestone'),
 ('changed_fc_rows',lambda f:f.identity['expansion'].update(old_rows_exact=False),'changed original rows'),
 ('wrong_lr',lambda f:f.rows[0].update(lr_nominal=.04),'cosine LR'),
 ('wrong_forward_count',lambda f:f.rows[0].update(training_forward_count=4),'execution counters'),
 ('input_pin_mismatch',lambda f:f.identity.update(config_sha256='0'*64),'input pins'),
 ('fisher_pass',lambda f:f.summary['counters'].update(fisher_estimation_passes=1),'unclean future train'),
 ('final_evaluation_mismatch',lambda f:f.summary['final_evaluation']['task0']['reporting']['local'].update(ce=2.),'final evaluation differs'),
 ('evaluation_sample_count',lambda f:f.evaluations['milestones']['0']['task1']['diagnostic']['all_seen'].update(samples=83),'evaluation sample count')]
 for label,mutation,reason in cases:
  f=Fixture(tmp/label,kind='future');accepted(f);mutation(f);f.save();results[label]=rejected(f,reason)
 f=Fixture(tmp/'bad_match',kind='future',rule='g_match');accepted(f);f.rows[0]['actual_match_relative_error']=.02;f.save();results['bad_match']=rejected(f,'norm mismatch')
 f=Fixture(tmp/'identity_hash',kind='future');accepted(f);(f.out/'identity.json').write_text('{}\n');results['identity_hash']=rejected(f,'output hash mismatch')
 f=Fixture(tmp/'mutated_origin',kind='future');accepted(f);(Path(f.identity['checkpoint_dir'])/'fc_state_0.pt').write_bytes(b'changed');results['mutated_origin']=rejected(f,'checkpoint changed')
 return results

def receipt_tests():
 keys=('config_sha256','summary_sha256','step_trace_sha256','identity_sha256','evaluations_sha256','checkpoints')
 actual={k:('a'*64 if k!='checkpoints' else [{'epoch':0,'files_sha256':{'x':'a'*64}}]) for k in keys}
 assert receipt_matches(copy.deepcopy(actual),actual)
 for key in keys:
  changed=copy.deepcopy(actual);changed[key]='wrong';assert not receipt_matches(changed,actual),key
 return {'valid_receipt':True,'changed_fields_rejected':list(keys)}

def inherited_lease_test(tmp):
 lease_path=tmp/'lease.lock';ready=tmp/'child_ready';release=tmp/'release';jobpath=tmp/'job.json'
 childcode="import os,time;from pathlib import Path;p=Path("+repr(str(ready))+");p.write_text(str(os.getpid()));q=Path("+repr(str(release))+");\nwhile not q.exists():time.sleep(.02)"
 write(jobpath,{'id':'cpu_lease_test','cwd':str(tmp),'command':[sys.executable,'-c',childcode]})
 lease=open(lease_path,'a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 worker=subprocess.Popen([sys.executable,str(ROOT/'job_worker.py'),'--job',str(jobpath),'--lease-fd',str(lease.fileno())],pass_fds=(lease.fileno(),),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 lease.close();probe=open(lease_path,'a+')
 def blocked():
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(probe,fcntl.LOCK_UN);return False
  except BlockingIOError:return True
 deadline=time.monotonic()+5
 while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
 if not ready.exists():raise RuntimeError('CPU worker child failed '+repr(worker.communicate(timeout=1)))
 before=blocked();worker.kill();worker.wait(timeout=5);after=blocked();release.touch()
 deadline=time.monotonic()+5
 while blocked() and time.monotonic()<deadline:time.sleep(.02)
 freed=not blocked();probe.close();assert before and after and freed
 return {'worker_and_child_hold_lease':before,'child_keeps_lease_after_worker_killed':after,'released_after_child_exit':freed,'gpu_used':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--lease',action='store_true');parser.add_argument('--formation-only',action='store_true');args=parser.parse_args()
 result={'status':'running','reviewed_hashes':{n:sha(ROOT/n) for n in ('run_queue.py','job_worker.py','validate_job.py','prepare_manifest.py','collect_results.py')},'test_sha256':sha(__file__),'gpu_used':False}
 try:
  with tempfile.TemporaryDirectory(prefix='formation_queue_review_') as tmp:
   p=Path(tmp);result['formation']=formation_tests(p/'formation');result['receipt']=receipt_tests()
   if not args.formation_only:result['future']=future_tests(p/'future')
   if args.lease:(p/'lease').mkdir();result['lease']=inherited_lease_test(p/'lease')
  result['status']='passed'
 except BaseException as exc:
  result.update(status='failed',error=repr(exc));raise
 finally:
  print(json.dumps(result,indent=2))
  if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
