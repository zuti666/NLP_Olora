"""CPU-only synthetic release-gate tests. Never invoke main/queue or GPUs."""
import json,sys,tempfile,hashlib,importlib.util,datetime,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
waiter=load('release_waiter_test',ROOT/'wait_then_launch.py');queue=load('release_queue_test',ROOT/'run_queue.py')
pins={n:sha(ROOT/n) for n in ['wait_then_launch.py','run_queue.py','validate_job.py','common.py','validate_parent.py']}
checks={}
def reject(name,fn):
 try:fn()
 except (ValueError,RuntimeError,FileNotFoundError,subprocess.CalledProcessError) as e:checks[name]={'rejected':True,'reason':str(e)}
 else:raise AssertionError('Accepted invalid release: '+name)
with tempfile.TemporaryDirectory(prefix='component-release-cpu-') as temp:
 tmp=Path(temp);parent=tmp/'parent';child=tmp/'child';source=tmp/'source';source.mkdir();(source/'x.py').write_text('# synthetic parent source\n')
 (parent).mkdir();(parent/'common.py').write_text("from pathlib import Path\nimport hashlib\nHERE=Path(__file__).resolve().parent\ndef sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n");(parent/'validate_job.py').write_text("import common\nfrom pathlib import Path\nassert common.HERE == Path(__file__).resolve().parent, 'wrong parent common context'\ndef validate_job(job): return {'synthetic_result_sha256': 'valid'}\ndef receipt_matches(receipt,actual): return receipt.get('synthetic_result_sha256') == actual['synthetic_result_sha256']\n")
 write(parent/'source_manifest.json',{'files':{'x.py':sha(source/'x.py')}})
 ps={'python':sys.executable,'run_id':'synthetic-parent','jobs':[{'id':'parent_job'}],'source_root':str(source),'source_manifest':str(parent/'source_manifest.json')};write(parent/'suite.json',ps)
 write(parent/'RUN_PINS.json',{'files':{'validate_job.py':sha(parent/'validate_job.py'),'common.py':sha(parent/'common.py')}})
 rec={'job_id':'parent_job','exit':{'returncode':0},'synthetic_result_sha256':'valid'};rp=parent/'runtime/jobs/parent_job/completed.json';write(rp,rec)
 state={'run_id':ps['run_id'],'status':'running','jobs':{'parent_job':{'status':'running'}},'controller':waiter.proc_identity(os.getpid())};sp=parent/'runtime/queue_state.json';write(sp,state)
 guard={'execution_pid_namespace':os.readlink('/proc/self/ns/pid'),'parent_dir':str(parent),'expected_parent_jobs':1,'parent_suite_sha256':sha(parent/'suite.json'),'parent_run_pins_sha256':sha(parent/'RUN_PINS.json')};write(child/'handoff_guard.json',guard)
 assert waiter.parent_ready(guard)[0] is False;checks['live_running_parent_waits']=True
 state['status']='completed';state['jobs']['parent_job']['status']='completed';write(sp,state)
 assert waiter.parent_ready(guard)[0] is False;checks['completed_but_live_parent_waits']=True
 state['controller']={'pid':999999999,'proc_starttime':'0'};write(sp,state)
 assert waiter.parent_ready(guard)[0] is True;checks['completed_exited_valid_parent_accepted']=True
 queue.HERE=child
 proof={'status':'validated','guard_sha256':sha(child/'handoff_guard.json'),'parent_jobs_validated':1,'parent_suite_sha256':guard['parent_suite_sha256'],'parent_run_pins_sha256':guard['parent_run_pins_sha256'],'parent_receipts_sha256':{'parent_job':sha(rp)}};pp=child/'runtime/PARENT_VALIDATED.json';write(pp,proof)
 queue.verify_parent_release();checks['direct_queue_verified_parent_accepted']=True
 invalid_guard={**guard,'execution_pid_namespace':'pid:[invalid]'}
 reject('wrong_pid_namespace_blocks_waiter',lambda:waiter.parent_ready(invalid_guard))
 write(child/'handoff_guard.json',invalid_guard)
 reject('wrong_pid_namespace_blocks_direct_queue',queue.verify_parent_release)
 write(child/'handoff_guard.json',guard)
 original_parent_validator=(parent/'validate_job.py').read_text()
 (parent/'validate_job.py').write_text(original_parent_validator+'# modified after proof\n')
 reject('parent_pinned_file_change_blocks_direct_queue',queue.verify_parent_release)
 (parent/'validate_job.py').write_text(original_parent_validator)
 state['jobs']['parent_job']['status']='pending';state['status']='running';write(sp,state)
 reject('parent_pending_blocks_direct_queue',queue.verify_parent_release)
 reject('exited_incomplete_parent_blocks_waiter',lambda:waiter.parent_ready(guard))
 state['jobs']['parent_job']['status']='failed';write(sp,state);reject('parent_failure_blocks_waiter',lambda:waiter.parent_ready(guard))
 state['jobs']['parent_job']['status']='completed';state['status']='completed';write(sp,state)
 write(rp,{**rec,'extra':'changed'});reject('changed_parent_receipt_blocks_direct_queue',queue.verify_parent_release);write(rp,rec)
 write(rp,{**rec,'exit':{'returncode':1}});reject('parent_exit_nonzero_blocks_waiter',lambda:waiter.parent_ready(guard));write(rp,rec)
 write(pp,{**proof,'parent_jobs_validated':0});reject('wrong_parent_count_blocks_direct_queue',queue.verify_parent_release);write(pp,proof)
 write(pp,{**proof,'parent_receipts_sha256':{}});reject('missing_parent_receipt_blocks_direct_queue',queue.verify_parent_release);write(pp,proof)
 write(pp,{**proof,'guard_sha256':'0'*64});reject('changed_guard_blocks_direct_queue',queue.verify_parent_release);write(pp,proof)
 queue.verify_parent_release()
 ss={'jobs':[{'id':'smoke_job','smoke':True}]};write(child/'suite.json',ss)
 sr=child/'runtime/jobs/smoke_job/completed.json';write(sr,{'job_id':'smoke_job','status':'completed'})
 smoke={'status':'passed','verified_smokes':1,'suite_sha256':sha(child/'suite.json'),'receipts_sha256':{'smoke_job':sha(sr)}};sg=child/'runtime/GPU_SMOKES_VERIFIED.json';write(sg,smoke)
 queue.verify_gpu_smoke_release(ss);checks['valid_current_smoke_release_accepted']=True
 write(sg,{**smoke,'suite_sha256':'0'*64});reject('stale_smoke_suite_rejected',lambda:queue.verify_gpu_smoke_release(ss));write(sg,smoke)
 write(sr,{'job_id':'smoke_job','status':'changed'});reject('changed_smoke_receipt_rejected',lambda:queue.verify_gpu_smoke_release(ss))
assert pins=={n:sha(ROOT/n) for n in pins},'Scripts changed during test'
result={'status':'PASS','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Synthetic metadata and current own PID; isolated parent subprocess proves own common context. No queue/GPU invoked. Production requires host PID namespace.','implementation_sha256':pins,'checks':checks}
(ROOT/'review/RECOVERY_RELEASE_GUARD_CPU.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks),'negative_checks':sum(isinstance(x,dict) for x in checks.values()),'implementation_sha256':pins},indent=2))
