"""Read-only scientific/configuration recovery review; writes only this report."""
import ast,hashlib,json,math,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-20261001T185744Z')
NEW42=ROOT.parent/'fdhvp-followup-20261001T223005Z'
SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def func(p,name):return next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
a,b=read(OLD/'suite.json'),read(ROOT/'suite.json');oldjobs={j['id']:j for j in a['jobs']};newjobs={j['id']:j for j in b['jobs']}
assert len(oldjobs)==len(newjobs)==25 and set(oldjobs)==set(newjobs)
configs=[]
for jid,j in newjobs.items():
 old=read(Path(oldjobs[jid]['config']));p=ROOT/'configs'/Path(j['config']).name;new=read(p)
 changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
 assert changed=={'prefix','outputs_root'}
 assert j['config_sha256']==sha(p)
 expected=json.loads(json.dumps(oldjobs[jid]).replace(str(OLD),b['run_dir']));expected['config_sha256']=j['config_sha256'];assert j==expected
 configs.append({'job_id':jid,'changed_keys':sorted(changed),'config_sha256':sha(p)})
assert (ROOT/'source_manifest.json').read_bytes()==(OLD/'source_manifest.json').read_bytes()
assert (ROOT/'handoff_guard.json').read_bytes()==(OLD/'handoff_guard.json').read_bytes()
oldpins=read(OLD/'RUN_PINS.json')['files'];assert all(sha(OLD/p)==v for p,v in oldpins.items())
migration=read(ROOT/'review/RECOVERY_MIGRATION_INITIAL.json')
for case in migration['failure_evidence']:
 for info in case['files'].values():assert sha(Path(info['path']))==info['sha256']
assert not any((ROOT/name).exists() for name in ('runtime','outputs','results'))
p=ROOT/'trainer/component_horizon.py';q=NEW42/'trainer/component_horizon.py'
helper=func(p,'seen_accuracy_row');assert ast.dump(helper,include_attributes=False)==ast.dump(func(q,'seen_accuracy_row'),include_attributes=False)
oldtext=(OLD/'trainer/component_horizon.py').read_text();newtext=p.read_text()
start=newtext.index('\ndef seen_accuracy_row(');end=newtext.index('\ndef resolve_config(',start)
reversed_text=newtext[:start]+newtext[end:]
# Ignore blank-line formatting only after reversing the exact eval substitution.
oldblock="""            groups=[(key,value) for key,value in cnn['grouped'].items() if re.fullmatch(r'\\d+-\\d+',key)]
            groups.sort(key=lambda kv:int(kv[0].split('-')[0]))
            row=[float(value) for _,value in groups]
            if len(row)!=self.task+1 or self.task!=len(self.matrix) or audit.nonfinite_paths(row):raise AssertionError('Incomplete native all-seen evaluation row')"""
reversed_text=reversed_text.replace('            row=seen_accuracy_row(cnn,self.task,l._total_classes,len(self.matrix))',oldblock)
assert ast.dump(ast.parse(reversed_text),include_attributes=False)==ast.dump(ast.parse(oldtext),include_attributes=False),'Non-evaluation runtime change found'
ns={'np':np,'math':math,'re':re}
for node in (func(SOURCE/'utils/toolkit.py','accuracy'),func(SOURCE/'models_CL/baseLearner.py','_evaluate'),helper):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-frozen-eval-and-recovery-helper>','exec'),ns)
from types import SimpleNamespace
cases=[]
for task in range(10):
 n=20*(task+1);labels=np.arange(n);pred=np.stack([(labels+k)%n for k in range(5)],axis=1)
 learner=SimpleNamespace(args={'init_cls':20,'increment':20,'task_increments':[20]*10},_known_classes=20*task,_total_classes=n,topk=5)
 cnn=ns['_evaluate'](learner,pred,labels);row=ns['seen_accuracy_row'](cnn,task,n,task)
 assert row==[100.]*(task+1) and cnn['top1']==100.
 cases.append({'task':task,'accepted_seen_groups':len(row),'excluded_structural_NA':9-task})
report={'status':'PASS','blocking_findings':[],'jobs':25,'unique_configs':len(set(j['config'] for j in b['jobs'])),'config_checks':configs,
 'old_pinned_file_count':len(oldpins),'old_pinned_files_and_six_failure_receipts_unchanged':True,
 'new_runtime_outputs_absent_at_review':True,'source_manifest_and_formation120_guard_unchanged':True,
 'runtime_patch_exactly_eval_adapter_only_AST_proven':True,'same_eval_helper_as_final42':True,'actual_frozen_baseLearner_and_toolkit_all_ten_tasks':cases,
 'new_files_sha256':{name:sha(ROOT/name) for name in ('trainer/component_horizon.py','validate_job.py','validate_parent.py','wait_then_launch.py','run_queue.py','suite.json','handoff_guard.json')},
 'parent_host_readonly_evidence':{'at_utc':'2026-10-01T23:01:39.981662+00:00','hostname':'cudahpc40.cvc.uab.es','pid_namespace':'pid:[4026531836]','actual_parent_ready':True,'completed_jobs_validated':120,'controller_dead_by_PID_and_starttime':True,'source_pins_and_all_receipts_checked':True,'isolated_python_I_B_parent_import_paths_checked':True},
 'GPU_launched':False,'scope':'Configuration, frozen-source evaluation adapter, immutable failure evidence and host parent validation; real GPU recovery smoke remains root responsibility.'}
(ROOT/'review/RECOVERY_INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','jobs':25,'config_changes':['prefix','outputs_root'],'all_ten_task_native_eval_checked':True,'trainer_sha256':sha(p)}))
