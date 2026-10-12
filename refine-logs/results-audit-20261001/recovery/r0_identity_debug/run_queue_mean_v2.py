#!/usr/bin/env python3
"""Pinned offline diagnostic queue with the same GPU leases as the formal queue."""
import argparse, fcntl, json, os, socket, subprocess, sys, time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    path=Path(args.suite).resolve(); suite=json.loads(path.read_text()); root=Path(suite['campaign']);sys.path.insert(0,str(root))
    from common import atomic_json, now, proc_identity, query_gpus, sha256, verify_source
    if socket.gethostname()!=suite['execution_hostname']: raise RuntimeError('Host mismatch')
    expected_suite=sha256(path)
    runtime=Path(suite['runtime']);runtime.mkdir(parents=True,exist_ok=True)
    def validate():
        if sha256(path)!=expected_suite: raise RuntimeError('Suite changed')
        verify_source(suite)
        for name, expected in suite['file_pins'].items():
            if sha256(name)!=expected: raise RuntimeError('File changed: '+name)
        if len(suite['jobs'])!=1 or suite['jobs'][0]['id']!='mean_epoch50_v2' or suite['jobs'][0]['max_batches']!=4 or suite.get('summarize'):raise RuntimeError('Restricted to Mean50-only 4batch recovery without collector')
        if not set(suite['allowed_gpus'])<=set(range(7)):raise RuntimeError('Unauthorized GPU')
    validate()
    if args.validate_only:
        print(json.dumps({'status':'passed','jobs':len(suite['jobs']),'suite_sha256':expected_suite}));return
    controller_lock=open(runtime/'controller.lock','a+')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (runtime/'controller.json').exists():raise RuntimeError('Existing run receipts; do not relaunch automatically')
    atomic_json(runtime/'controller.json',{**proc_identity(os.getpid()),'time':now(),'suite_sha256':expected_suite})
    pending=list(suite['jobs']);active={};complete=[];failed=[];halt=False
    locks=Path('/tmp/olora_gamfs_gpu_locks');locks.mkdir(exist_ok=True)
    def state():
        atomic_json(runtime/'state.json',{'time':now(),'status':'failed' if halt and not active else 'completed' if not pending and not active else 'running','pending':[j['id'] for j in pending],'running':[{k:v for k,v in a.items() if k not in ('process','log')} for a in active.values()],'completed':complete,'failed':failed,'paused':(runtime/'PAUSE').exists(),'suite_sha256':expected_suite})
    def result_validation(job,rc):
        errors=[];stats={}
        if rc!=0:errors.append('exit_code='+str(rc))
        try:
            result=json.loads(Path(job['output']).read_text());bs=result.get('batches',[])
            if result.get('schema_version')!=2 or result.get('audit_version')!='native_actual_probe_self_spec_v2':errors.append('audit_version')
            if result.get('diagnostic_script_sha256')!=suite['file_pins'][suite['script']] or result.get('probe_audit_sha256')!=suite['file_pins'][str(Path(suite['script']).with_name('r0_probe_audit.py'))]:errors.append('audit_code_hash')
            if [b.get('batch') for b in bs]!=list(range(job['max_batches'])):errors.append('batch_sequence')
            if result.get('status')!='completed':errors.append('result_status='+str(result.get('status')))
            if len(bs)!=job['max_batches']:errors.append('batch_count')
            if not result.get('finite_check',{}).get('ok'):errors.append('finite_check')
            if not result.get('final_restore_check') or not all(result['final_restore_check'].values()):errors.append('final_restore_check')
            for b in bs:
                if not b.get('restore_check') or not all(b['restore_check'].values()):errors.append('batch_restore')
                if not b.get('native_self_spec_passed'): errors.append('native_self_spec')
                measurement=b.get('r0_identity_sanity',{})
                if measurement.get('used_as_correctness_gate') is not False or measurement.get('reference_tolerance')!=5e-4:errors.append('r0_measurement_protocol')
                for radius,scan in b.get('radius_scan',{}).items():
                    audit=scan.get('native_self_spec',{})
                    if not audit.get('passed') or scan.get('closure_count')!=4 or scan.get('independent_self_spec_ce_passes')!=4:errors.append('native_probe_protocol:'+radius)
                    ops=audit.get('operation_checks',[]);ce=audit.get('closure_CE_checks',[])
                    if len(ops)!=7 or not all(z.get('passed') and z.get('all_exact') and z.get('finite') for z in ops):errors.append('native_operations:'+radius)
                    if len(ce)!=4 or not all(z.get('passed') and z.get('finite') and z.get('per_tensor_allclose') for z in ce):errors.append('actual_probe_CE:'+radius)
                for label in ('g1','g3','sum','half_sum'):
                    cs=[c for c in b['counterfactuals'] if c['label']==label]
                    if len(cs)!=1 or not cs[0]['match']['ok']: errors.append('core_match:'+label)
            stats={'batches':len(bs),'matched':sum(c['match']['ok'] for b in bs for c in b.get('counterfactuals',[])),'candidates':sum(len(b.get('counterfactuals',[])) for b in bs),'kl_matches':sum(b.get('radius_matched_sam',{}).get('kl',{}).get('ok',False) for b in bs),'elapsed_seconds':result.get('elapsed_seconds')}
        except Exception as exc:errors.append(repr(exc))
        return errors,stats
    try:
        while pending or active:
            for key,item in list(active.items()):
                rc=item['process'].poll()
                if rc is None:continue
                item['log'].close();job=item['job'];validate();errors,stats=result_validation(job,rc)
                receipt={'id':key,'time':now(),'returncode':rc,'validation_errors':errors,**stats,'output':job['output']}
                atomic_json(runtime/'jobs'/key/('failed.json' if errors else 'completed.json'),receipt)
                (failed if errors else complete).append(receipt);del active[key]
                if errors:halt=True
                print(json.dumps({'event':'finished',**receipt}),flush=True)
            if halt:
                state()
                if not active:break
            if not halt and not (runtime/'PAUSE').exists():
                for gpu,info in query_gpus().items():
                    if not pending or len(active)>=suite['max_parallel']:break
                    if gpu not in suite['allowed_gpus'] or info['memory_used_mb']>=500 or info['utilization_percent']>=5:continue
                    lease=open(locks/('gpu%d.lock'%gpu),'a+')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    check=query_gpus().get(gpu)
                    if not check or check['uuid']!=info['uuid'] or check['memory_used_mb']>=500 or check['utilization_percent']>=5:
                        lease.close();continue
                    try:
                        validate();job=pending.pop(0);jdir=runtime/'jobs'/job['id'];jdir.mkdir(parents=True,exist_ok=False)
                        if Path(job['output']).exists():raise RuntimeError('Output already exists '+job['output'])
                        env=dict(os.environ,**suite['environment']);env['CUDA_VISIBLE_DEVICES']=info['uuid']
                        command=[suite['python'],suite['script'],'--source',suite['source_root'],'--base-config',job['config'],'--checkpoint-dir',job['checkpoint_dir'],'--task',str(job['task']),'--output',job['output'],'--max-batches',str(job['max_batches']),'--batch-size','128','--eval-per-task','1','--eta-ref','0.02','--rho','0.05','--kl-match','--save-vectors']
                        log=open(jdir/'console.log','x');proc=subprocess.Popen(command,cwd=suite['source_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                        receipt={'id':job['id'],**proc_identity(proc.pid),'physical_gpu':gpu,'gpu_uuid':info['uuid'],'time':now(),'command':command,'job':job}
                        atomic_json(jdir/'started.json',receipt);active[job['id']]={**receipt,'process':proc,'log':log}
                        print(json.dumps({'event':'started','id':job['id'],'pid':proc.pid,'gpu':gpu}),flush=True)
                    finally:
                        # Child inherits this open description. Do not LOCK_UN.
                        lease.close()
            state()
            if active or pending:time.sleep(10)
        state()
        summarizer=root/'diagnostics'/'summarize_diagnostics.py'
        if not failed and suite.get('summarize') and summarizer.exists():
            subprocess.run([suite['python'],str(summarizer),'--results',str(root/'diagnostics'/'results'),'--output',str(root/'diagnostics'/'summary')],check=True)
        if failed:raise SystemExit(1)
    except BaseException as exc:
        atomic_json(runtime/'controller_exception.json',{'time':now(),'error':repr(exc),'active':[k for k in active],'pending':[j['id'] for j in pending]})
        raise

if __name__=='__main__':main()
