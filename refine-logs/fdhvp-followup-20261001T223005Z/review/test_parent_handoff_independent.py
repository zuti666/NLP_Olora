#!/usr/bin/env python3
"""CPU-only real parent guard calls with isolated synthetic parent validation.

No main/launcher/deployer/worker/GPU function is called. Synthetic output hashes
test the barrier; real model completion validation has its own integration tests.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Fixture:
    def __init__(self,root,waiter,queue):
        self.parent=Path(root)/'parent';self.child=Path(root)/'child';self.source=Path(root)/'source'
        self.waiter=waiter;self.queue=queue
        for p in (self.parent,self.child,self.source):p.mkdir()
        (self.source/'frozen.py').write_text('# synthetic frozen source\n')
        (self.parent/'common.py').write_text('''import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT_MARKER='synthetic-parent-not-child'
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
''')
        (self.parent/'validate_job.py').write_text('''import json
from pathlib import Path
from common import HERE,PARENT_MARKER,sha256
assert HERE==Path(__file__).resolve().parent and PARENT_MARKER=='synthetic-parent-not-child'
def validate_job(job):
    output=Path(job['output']);value=json.loads(output.read_text())
    if value['job_id']!=job['id'] or value['status']!='completed':raise ValueError('Invalid synthetic output')
    with (HERE/'validation_calls.jsonl').open('a') as f:f.write(json.dumps({'job':job['id'],'common_root':str(HERE),'marker':PARENT_MARKER})+'\\n')
    return {'observed_sha256':sha256(output)}
def receipt_matches(receipt,actual):return receipt.get('observed_sha256')==actual['observed_sha256']
''')
        write(self.parent/'source_manifest.json',{'files':{'frozen.py':sha(self.source/'frozen.py')}})
        jobs=[]
        for i in range(25):
            jid='parent_%02d'%i;output=self.parent/'outputs'/jid/'result.json'
            write(output,{'job_id':jid,'status':'completed','fixture_value':i})
            jobs.append({'id':jid,'output':str(output)})
            write(self.parent/'runtime/jobs'/jid/'completed.json',{'job_id':jid,'exit':{'returncode':0},'observed_sha256':sha(output)})
        self.suite={'run_id':'synthetic-parent-25','jobs':jobs,'python':sys.executable,
            'source_root':str(self.source),'source_manifest':str(self.parent/'source_manifest.json')}
        write(self.parent/'suite.json',self.suite)
        write(self.parent/'RUN_PINS.json',{'files':{n:sha(self.parent/n) for n in ('common.py','validate_job.py','source_manifest.json')}})
        proc=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE)
        self.dead=waiter.proc_identity(proc.pid);assert self.dead is not None
        proc.communicate(timeout=10);assert proc.returncode==0 and not waiter.process_matches(self.dead)
        self.live=waiter.proc_identity(os.getpid());assert self.live and waiter.process_matches(self.live)
        self.state={'run_id':self.suite['run_id'],'status':'completed','controller':self.dead,
            'jobs':{j['id']:{'status':'completed'} for j in jobs}}
        self.handoff={'status':'completed','process':self.dead}
        write(self.parent/'runtime/queue_state.json',self.state)
        write(self.parent/'runtime/handoff_state.json',self.handoff)
        self.guard={'parent_dir':str(self.parent),'expected_parent_jobs':25,
            'parent_suite_sha256':sha(self.parent/'suite.json'),'parent_run_pins_sha256':sha(self.parent/'RUN_PINS.json'),
            'execution_pid_namespace':os.readlink('/proc/self/ns/pid')}
        write(self.child/'handoff_guard.json',self.guard)
        self.proof={'status':'validated','guard_sha256':sha(self.child/'handoff_guard.json'),'parent_jobs_validated':25,
            'parent_suite_sha256':self.guard['parent_suite_sha256'],'parent_run_pins_sha256':self.guard['parent_run_pins_sha256'],
            'parent_receipts_sha256':{j['id']:sha(self.parent/'runtime/jobs'/j['id']/'completed.json') for j in jobs}}
        write(self.child/'runtime/PARENT_VALIDATED.json',self.proof)
        shutil.copyfile(ROOT/'validate_parent.py',self.child/'validate_parent.py')
        (self.child/'check_gpu_smokes.py').write_text("raise RuntimeError('no verified synthetic smoke outputs')\n")
        (self.child/'child_pin.py').write_text('# synthetic child code\n')
        write(self.child/'RUN_PINS.json',{'files':{'child_pin.py':sha(self.child/'child_pin.py')}})
        self.child_source_suite={**self.suite,'source_manifest_sha256':sha(self.parent/'source_manifest.json')}
        waiter.HERE=self.child;queue.HERE=self.child
    def waiter_ready(self):return self.waiter.parent_ready(read(self.child/'handoff_guard.json'))
    def queue_ready(self):return self.queue.verify_parent_release()
    def queue_prelaunch(self):self.queue.verify_parent_release();self.queue.verify_pins(self.child_source_suite)
    def calls(self):
        p=self.parent/'validation_calls.jsonl'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def main():
    names=['run_queue.py','wait_then_launch.py','validate_parent.py','start_waiting.py','deploy_run.py',
           'common.py','validate_job.py','conditional_gate.py','check_gpu_smokes.py','collect_results.py','suite.json','FROZEN_GATES.json']
    pins={n:sha(ROOT/n) for n in names};testpin=sha(__file__)
    waiter=load('fdhvp_waiter_independent',ROOT/'wait_then_launch.py')
    queue=load('fdhvp_queue_independent',ROOT/'run_queue.py')
    # Assert the contaminated parent process really has CHILD common loaded.
    assert Path(sys.modules['common'].HERE)==ROOT
    assert not hasattr(sys.modules['common'],'PARENT_MARKER')
    checks=[];bad=[]
    def positive(name,fn):
        fn();checks.append({'case':name,'result':'passed'})
    def rejected(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError,subprocess.CalledProcessError) as exc:
            checks.append({'case':name,'result':'rejected_as_required','error':repr(exc)})
        except Exception as exc:raise AssertionError('Fixture/programming failure rather than a meaningful rejection '+name+': '+repr(exc)) from exc
        else:bad.append(name)
    def alter(path,value,fn):
        data=Path(path).read_bytes()
        try:
            if isinstance(value,bytes):Path(path).write_bytes(value)
            else:write(path,value)
            fn()
        finally:Path(path).write_bytes(data)
    with tempfile.TemporaryDirectory(prefix='fdhvp_parent25_cpu_') as temporary:
        f=Fixture(temporary,waiter,queue)
        def ready_all():
            p=f.parent/'validation_calls.jsonl'
            if p.exists():p.unlink()
            assert f.waiter_ready()[0] is True
            records=f.calls();assert len(records)==25
            assert [r['job'] for r in records]==[j['id'] for j in f.suite['jobs']]
            assert all(r['common_root']==str(f.parent) and r['marker']=='synthetic-parent-not-child' for r in records)
        positive('all_25_outputs_validated_in_fresh_parent_import_context',ready_all)
        positive('queue_accepts_current_25_receipts_after_both_real_processes_exited',f.queue_prelaunch)
        sp=f.parent/'runtime/queue_state.json';hp=f.parent/'runtime/handoff_state.json'
        def live_case(name):
            assert f.waiter_ready()[0] is False
            rejected(name+'_queue_rejects',f.queue_ready)
        alter(sp,{**f.state,'controller':f.live},lambda:positive('live_controller_waits',lambda:live_case('live_controller')))
        alter(hp,{**f.handoff,'process':f.live},lambda:positive('live_handoff_waits_even_controller_exited',lambda:live_case('live_handoff')))
        incomplete={**f.state,'status':'running','jobs':{**f.state['jobs'],'parent_24':{'status':'running'}}}
        alter(sp,{**incomplete,'controller':f.live},lambda:positive('24_of_25_alive_controller_waits',lambda:live_case('24_of_25_alive_controller')))
        alter(hp,{**f.handoff,'process':f.live},lambda:alter(sp,incomplete,
            lambda:positive('24_of_25_alive_handoff_waits',lambda:live_case('24_of_25_alive_handoff'))))
        def both(label):rejected(label+'_waiter',f.waiter_ready);rejected(label+'_queue',f.queue_ready)
        alter(sp,incomplete,lambda:both('24_of_25_both_controllers_exited'))
        alter(sp,{**f.state,'jobs':{**f.state['jobs'],'parent_24':{'status':'failed'}}},lambda:both('one_parent_failed'))
        alter(hp,{**f.handoff,'status':'blocked_failure'},lambda:both('parent_handoff_failed'))
        alter(hp,{**f.handoff,'status':'running_queue'},lambda:both('exited_handoff_not_completed'))
        guardpath=f.child/'handoff_guard.json';proofpath=f.child/'runtime/PARENT_VALIDATED.json'
        def wrong_namespace():
            proof={**f.proof,'guard_sha256':sha(guardpath)}
            hidden={'pid':999999999,'proc_starttime':'0'}
            alter(proofpath,proof,lambda:alter(sp,{**f.state,'controller':hidden},
                lambda:alter(hp,{**f.handoff,'process':hidden},lambda:both('wrong_namespace_even_inaccessible_host_pids'))))
        alter(guardpath,{**f.guard,'execution_pid_namespace':'pid:[0000000000]'},wrong_namespace)
        alter(f.parent/'suite.json',{**f.suite,'changed':True},lambda:both('parent_suite_tamper'))
        alter(f.parent/'RUN_PINS.json',{'files':{}},lambda:both('parent_pins_manifest_tamper'))
        for filename in ('common.py','validate_job.py','source_manifest.json'):
            p=f.parent/filename
            alter(p,p.read_bytes()+b'\n# tampered\n',lambda n=filename:both('parent_pinned_'+n+'_tamper'))
        p=f.source/'frozen.py'
        alter(p,b'# changed frozen source\n',lambda:(rejected('parent_source_tamper_waiter',f.waiter_ready),rejected('shared_source_tamper_queue_prelaunch',f.queue_prelaunch)))
        p=f.parent/'outputs/parent_24/result.json'
        alter(p,{'job_id':'parent_24','status':'completed','fixture_value':-1},lambda:(
            rejected('parent_observed_raw_output_tamper_before_release',f.waiter_ready),
            positive('post_release_hash_proof_does_not_claim_revalidating_historical_raw_models',f.queue_prelaunch)))
        p=f.parent/'runtime/jobs/parent_24/completed.json';receipt=read(p)
        alter(p,{**receipt,'exit':{'returncode':1}},lambda:both('parent_nonzero_exit'))
        alter(p,{**receipt,'observed_sha256':'0'*64},lambda:both('parent_receipt_tamper'))
        alter(proofpath,{**f.proof,'parent_jobs_validated':24},lambda:rejected('proof_only_24_jobs',f.queue_ready))
        alter(proofpath,{**f.proof,'guard_sha256':'0'*64},lambda:rejected('proof_wrong_guard',f.queue_ready))
        positive('restored_valid_parent_still_passes_isolated_25_output_validation',ready_all)
        # Verify each formal group binds the full exact set of CURRENT receipts.
        suite={'python':sys.executable,'smoke_groups':{'base':['base_a','base_b'],'second_hvp':['hvp_a','hvp_b']}}
        write(f.child/'suite.json',suite);write(f.child/'FROZEN_GATES.json',{'fixed_threshold':.5})
        group='second_hvp';proof={'status':'passed','group':group,'verified_smokes':2,
            'suite_sha256':sha(f.child/'suite.json'),'frozen_gates_sha256':sha(f.child/'FROZEN_GATES.json'),'receipts_sha256':{}}
        for jid in ['base_a','base_b','hvp_a','hvp_b']:
            p=f.child/'runtime/jobs'/jid/'completed.json';write(p,{'job_id':jid,'fixture':True})
            if jid.startswith('hvp_'):proof['receipts_sha256'][jid]=sha(p)
        gp=f.child/'runtime/GPU_SMOKES_second_hvp_VERIFIED.json'
        rejected('no_smoke_proof_and_no_valid_smokes_blocks_formal',lambda:queue.verify_gpu_smoke_release(suite,group))
        write(gp,proof)
        positive('correct_group_current_complete_smoke_proof_passes',lambda:queue.verify_gpu_smoke_release(suite,group))
        for key,value in [('group','base'),('verified_smokes',1),('suite_sha256','0'*64),('frozen_gates_sha256','0'*64)]:
            alter(gp,{**proof,key:value},lambda k=key:rejected('smoke_proof_wrong_'+k,lambda:queue.verify_gpu_smoke_release(suite,group)))
        swapped={jid:sha(f.child/'runtime/jobs'/jid/'completed.json') for jid in ('base_a','base_b')}
        alter(gp,{**proof,'receipts_sha256':swapped},lambda:rejected('same_count_wrong_smoke_group_receipts',lambda:queue.verify_gpu_smoke_release(suite,group)))
        p=f.child/'runtime/jobs/hvp_b/completed.json'
        alter(p,{'job_id':'hvp_b','fixture':'changed'},lambda:rejected('smoke_receipt_tamper',lambda:queue.verify_gpu_smoke_release(suite,group)))
    assert pins=={n:sha(ROOT/n) for n in names},'Implementation changed while independent test ran'
    assert testpin==sha(__file__),'Test changed while running'
    result={'status':'PASS' if not bad else 'FAIL','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'test_sha256':testpin,'implementation_sha256':pins,'checks':checks,'unexpectedly_accepted':bad,
        'pid_namespace_tested':os.readlink('/proc/self/ns/pid'),
        'scope':'CPU only; actual guard functions, fresh isolated parent validator subprocess, 25 synthetic parent outputs/receipts, real same-namespace CPU process exit; no launcher/GPU call',
        'proof_boundary':'waiter validates all parent raw outputs once; queue binds current hashes of that validated receipt set and pinned files/source without re-reading historical model outputs per launch'}
    write(ROOT/'review/PARENT_HANDOFF_CPU_INDEPENDENT.json',result)
    print(json.dumps({'status':result['status'],'checks':len(checks),'unexpectedly_accepted':bad,'test_sha256':testpin},indent=2))
    assert not bad,'Invalid handoff accepted: '+repr(bad)


if __name__=='__main__':main()
