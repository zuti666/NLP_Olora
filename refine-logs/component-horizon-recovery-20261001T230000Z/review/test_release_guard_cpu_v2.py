#!/usr/bin/env python3
"""CPU-only 120-job parent barrier, tamper and PID-namespace regression tests.

Never call waiter.main, queue.main, query_gpus, training, or a GPU process.
The synthetic validator deliberately returns only its own fixture output hash;
native completion contents are covered by the separate strict validator tests.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Fixture:
    def __init__(self,root,waiter,queue):
        self.root=Path(root);self.parent=self.root/'parent';self.child=self.root/'child';self.source=self.root/'source'
        self.waiter=waiter;self.queue=queue;self.source.mkdir();self.parent.mkdir();self.child.mkdir()
        (self.source/'x.py').write_text('# synthetic frozen parent source\n')
        (self.parent/'validate_job.py').write_text('''import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def validate_job(job):
    p=Path(job['output']);x=json.loads(p.read_text())
    if x['job_id']!=job['id'] or x['status']!='completed':raise ValueError('Invalid synthetic observed output')
    with (HERE/'validation_calls.jsonl').open('a') as f:f.write(json.dumps(job['id'])+'\\n')
    return {'observed_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def receipt_matches(receipt,actual):return receipt.get('observed_sha256')==actual['observed_sha256']
''')
        write(self.parent/'source_manifest.json',{'files':{'x.py':sha(self.source/'x.py')}})
        jobs=[]
        for i in range(120):
            jid='parent_%03d'%i;output=self.parent/'outputs'/jid/'result.json'
            write(output,{'job_id':jid,'status':'completed','fixture_value':i})
            jobs.append({'id':jid,'output':str(output)})
            write(self.parent/'runtime/jobs'/jid/'completed.json',{'job_id':jid,'exit':{'returncode':0},'observed_sha256':sha(output)})
        self.suite={'run_id':'synthetic-parent-120','jobs':jobs,'source_root':str(self.source),
                    'source_manifest':str(self.parent/'source_manifest.json')}
        write(self.parent/'suite.json',self.suite)
        write(self.parent/'RUN_PINS.json',{'files':{'validate_job.py':sha(self.parent/'validate_job.py')}})
        # Prove exit with a genuine child in THIS namespace, not a guessed host PID.
        child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE)
        self.dead_identity=waiter.proc_identity(child.pid)
        assert self.dead_identity is not None
        child.communicate(timeout=10);assert child.returncode==0 and not waiter.process_matches(self.dead_identity)
        self.state={'run_id':self.suite['run_id'],'status':'completed',
                    'jobs':{j['id']:{'status':'completed'} for j in jobs},'controller':self.dead_identity}
        write(self.parent/'runtime/queue_state.json',self.state)
        self.guard={'parent_dir':str(self.parent),'expected_parent_jobs':120,
                    'parent_suite_sha256':sha(self.parent/'suite.json'),
                    'parent_run_pins_sha256':sha(self.parent/'RUN_PINS.json'),
                    'execution_pid_namespace':os.readlink('/proc/self/ns/pid')}
        self.proof={'status':'validated','parent_jobs_validated':120,
            'parent_suite_sha256':self.guard['parent_suite_sha256'],
            'parent_run_pins_sha256':self.guard['parent_run_pins_sha256'],
            'parent_receipts_sha256':{j['id']:sha(self.parent/'runtime/jobs'/j['id']/'completed.json') for j in jobs}}
        self.write_release()
        (self.child/'child_pin.txt').write_text('synthetic child immutable pin\n')
        write(self.child/'RUN_PINS.json',{'files':{'child_pin.txt':sha(self.child/'child_pin.txt')}})
        self.child_source_suite={'source_root':str(self.source),'source_manifest':str(self.parent/'source_manifest.json'),
            'source_manifest_sha256':sha(self.parent/'source_manifest.json')}
        queue.HERE=self.child
    def write_release(self):
        write(self.child/'handoff_guard.json',self.guard)
        self.proof['guard_sha256']=sha(self.child/'handoff_guard.json')
        write(self.child/'runtime/PARENT_VALIDATED.json',self.proof)
    def calls(self):
        p=self.parent/'validation_calls.jsonl'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    def clear_calls(self):
        p=self.parent/'validation_calls.jsonl'
        if p.exists():p.unlink()
    def waiter_ready(self):return self.waiter.parent_ready(read(self.child/'handoff_guard.json'))
    def direct_ready(self):return self.queue.verify_parent_release()
    def queue_prelaunch_checks(self):
        # These two real checks are both called before run_queue can launch.
        self.queue.verify_parent_release();self.queue.verify_pins(self.child_source_suite)


def main():
    implementation=['wait_then_launch.py','run_queue.py','validate_job.py','common.py','check_gpu_smokes.py']
    pins={n:sha(ROOT/n) for n in implementation};test_pin=sha(__file__)
    waiter=load('release_cpu_waiter_v2',ROOT/'wait_then_launch.py')
    queue=load('release_cpu_queue_v2',ROOT/'run_queue.py')
    checks=[];accepted_bad=[]
    def positive(label,fn):
        fn();checks.append({'case':label,'status':'passed'})
    def rejected(label,fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError,AssertionError) as exc:
            checks.append({'case':label,'status':'rejected_as_required','error':repr(exc)})
        except Exception as exc:raise AssertionError('Unexpected fixture/programming error '+label+': '+repr(exc)) from exc
        else:accepted_bad.append(label)
    def alter(path,fn,checks_to_run):
        old=Path(path).read_bytes()
        try:fn();checks_to_run()
        finally:Path(path).write_bytes(old)
    with tempfile.TemporaryDirectory(prefix='release_guard_120_cpu_') as temp:
        f=Fixture(temp,waiter,queue)
        def all120():
            f.clear_calls();ready,_=f.waiter_ready();assert ready is True
            assert f.calls()==[j['id'] for j in f.suite['jobs']]
        positive('all_120_observed_outputs_validated_after_verified_same_namespace_exit',all120)
        positive('direct_queue_accepts_current_complete_parent_proof',f.direct_ready)
        statepath=f.parent/'runtime/queue_state.json'
        live={**f.state,'controller':waiter.proc_identity(os.getpid())}
        def live_complete():
            write(statepath,live);assert f.waiter_ready()[0] is False
            rejected('direct_queue_blocks_completed_but_live_parent',f.direct_ready)
        alter(statepath,lambda:None,lambda:positive('completed_but_live_parent_waits',live_complete))
        pending={**live,'status':'running','jobs':{**live['jobs'],'parent_119':{'status':'running'}}}
        alter(statepath,lambda:write(statepath,pending),lambda:positive('live_parent_119_of_120_waits',lambda:(_ for _ in ()).throw(AssertionError()) if f.waiter_ready()[0] else None))
        pending['controller']=f.dead_identity
        alter(statepath,lambda:write(statepath,pending),lambda:rejected('exited_parent_119_of_120_blocks',f.waiter_ready))
        failed={**f.state,'jobs':{**f.state['jobs'],'parent_119':{'status':'failed'}}}
        alter(statepath,lambda:write(statepath,failed),lambda:(rejected('parent_failure_blocks_waiter',f.waiter_ready),rejected('parent_failure_blocks_direct_queue',f.direct_ready)))
        # Even an inaccessible/absent host PID cannot pass a mismatched namespace.
        guardpath=f.child/'handoff_guard.json';proofpath=f.child/'runtime/PARENT_VALIDATED.json'
        oldproof=proofpath.read_bytes()
        def namespace_bad():
            bad={**f.guard,'execution_pid_namespace':'pid:[0000000000]'};write(guardpath,bad)
            proof={**f.proof,'guard_sha256':sha(guardpath)};write(proofpath,proof)
            hidden={**f.state,'controller':{'pid':999999999,'proc_starttime':'0'}};write(statepath,hidden)
        try:
            alter(guardpath,namespace_bad,lambda:(rejected('wrong_pid_namespace_blocks_waiter_even_absent_parent_PID',f.waiter_ready),rejected('wrong_pid_namespace_blocks_direct_queue_even_absent_parent_PID',f.direct_ready)))
        finally:proofpath.write_bytes(oldproof);write(statepath,f.state)
        def both(label):
            rejected(label+'_waiter',f.waiter_ready);rejected(label+'_queue_prelaunch',f.queue_prelaunch_checks)
        path=f.parent/'suite.json'
        alter(path,lambda:write(path,{**f.suite,'extra':'tampered'}),lambda:both('parent_suite_tamper'))
        path=f.parent/'RUN_PINS.json'
        alter(path,lambda:write(path,{'files':{},'tampered':True}),lambda:both('parent_pins_manifest_tamper'))
        path=f.parent/'validate_job.py'
        alter(path,lambda:path.write_text(path.read_text()+'\n# changed pinned implementation\n'),lambda:both('parent_pinned_file_tamper'))
        path=f.source/'x.py'
        alter(path,lambda:path.write_text('# changed frozen source\n'),lambda:both('frozen_source_tamper'))
        path=f.parent/'outputs/parent_119/result.json'
        alter(path,lambda:write(path,{'job_id':'parent_119','status':'completed','fixture_value':-1}),
              lambda:(rejected('last_parent_observed_output_tamper_before_release',f.waiter_ready),
                      positive('post_release_proof_does_not_claim_revalidation_of_historical_raw_outputs',f.queue_prelaunch_checks)))
        path=f.parent/'runtime/jobs/parent_119/completed.json'
        original=read(path)
        alter(path,lambda:write(path,{**original,'exit':{'returncode':1}}),lambda:both('parent_nonzero_exit'))
        alter(path,lambda:write(path,{**original,'observed_sha256':'0'*64}),lambda:both('parent_receipt_tamper'))
        positive('valid_parent_restored_after_all_tamper_cases',all120)
        # Formal release must have the current smoke proof; no queue is executed.
        suite={'jobs':[{'id':'smoke_%02d'%i,'smoke':True} for i in range(10)]}
        write(f.child/'suite.json',suite)
        proof={'status':'passed','verified_smokes':10,'suite_sha256':sha(f.child/'suite.json'),'receipts_sha256':{}}
        for job in suite['jobs']:
            path=f.child/'runtime/jobs'/job['id']/'completed.json';write(path,{'job_id':job['id'],'fixture':True})
            proof['receipts_sha256'][job['id']]=sha(path)
        gp=f.child/'runtime/GPU_SMOKES_VERIFIED.json'
        rejected('formal_blocks_missing_smoke_verification',lambda:queue.verify_gpu_smoke_release(suite))
        write(gp,proof);positive('current_all_ten_smoke_verification_accepted',lambda:queue.verify_gpu_smoke_release(suite))
        alter(gp,lambda:write(gp,{**proof,'verified_smokes':9}),lambda:rejected('formal_blocks_only_nine_verified_smokes',lambda:queue.verify_gpu_smoke_release(suite)))
        alter(gp,lambda:write(gp,{**proof,'suite_sha256':'0'*64}),lambda:rejected('formal_blocks_wrong_smoke_suite',lambda:queue.verify_gpu_smoke_release(suite)))
        path=f.child/'runtime/jobs/smoke_09/completed.json'
        alter(path,lambda:write(path,{'job_id':'smoke_09','fixture':'changed'}),lambda:rejected('formal_blocks_tampered_smoke_receipt',lambda:queue.verify_gpu_smoke_release(suite)))
    assert pins=={n:sha(ROOT/n) for n in implementation},'Implementation changed during test'
    assert test_pin==sha(__file__),'Test changed during execution'
    result={'status':'PASS' if not accepted_bad else 'FAIL','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'test_sha256':test_pin,'implementation_sha256':pins,'scope':'CPU only; 120 synthetic receipt/output records; actual same-namespace child exit; no queue.main/waiter.main/GPU invocation',
        'prior_tests_reused_as_design_reference':str(ROOT/'review/test_release_guard_independent.py'),
        'evidence_scope':'waiter validates all 120 raw outputs once; queue binds the resulting receipt hashes and current pinned files/source, without re-reading historical parent model outputs on every child launch',
        'pid_namespace_tested':os.readlink('/proc/self/ns/pid'),'checks':checks,'unexpectedly_accepted':accepted_bad}
    write(ROOT/'review/RELEASE_GUARD_CPU_V2.json',result)
    print(json.dumps({'status':result['status'],'checks':len(checks),'unexpectedly_accepted':accepted_bad,'implementation_sha256':pins,'test_sha256':test_pin},indent=2))
    assert result['status']=='PASS','Invalid release accepted: '+str(accepted_bad)


if __name__=='__main__':main()
