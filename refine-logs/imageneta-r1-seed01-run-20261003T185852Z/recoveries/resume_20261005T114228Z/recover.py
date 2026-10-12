"""One-time recovery of explicitly listed interrupted jobs; frozen suite is untouched."""
import argparse,copy,fcntl,hashlib,json,os,re,shlex,shutil,subprocess,sys
from collections import Counter
from pathlib import Path
D=Path(__file__).resolve().parent
PLAN=json.loads((D/'RECOVERY_PLAN.json').read_text());H=Path(PLAN['run_dir'])
sys.path.insert(0,str(H));sys.dont_write_bytecode=True
from common import atomic_json,now,read_json,sha256,process_matches,assert_host,query_gpus,load_suite
from run_queue import verify_pins
from common import verify_source
from validate_smokes import validate_smokes
from collect_results import collect

def inspect():
    suite=load_suite(H/'suite.json');assert_host(suite);verify_pins(suite);verify_source(suite)
    assert sha256(H/'suite.json')==PLAN['suite_sha256'] and sha256(H/'RUN_PINS.json')==PLAN['run_pins_sha256']
    spec=read_json(H/'environment_spec.json');canonical=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert canonical=='4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8'
    state=read_json(H/'runtime/queue_state.json');assert sha256(H/'runtime/queue_state.json')==PLAN['pre_state_sha256'],'Queue changed since recovery planning'
    assert not process_matches(state.get('controller',{})),'Original controller is alive'
    assert not (H/'runtime/PAUSE').exists(),'Explicit pause must not be silently removed'
    assert suite['allowed_gpus']==[4,5,6] and suite['max_parallel']==3
    assert {j for j,r in state['jobs'].items() if r['status']=='running'}==set(PLAN['interrupted_jobs'])
    assert Counter(r['status'] for r in state['jobs'].values())=={'completed':66,'running':3,'pending':51}
    completed={};jobs={j['id']:j for j in suite['jobs']}
    for jid,r in state['jobs'].items():
        if r['status']=='running':
            assert not process_matches(r['process']), 'Original worker still alive: '+jid
            directory=H/'runtime/jobs'/jid
            identity=read_json(directory/'worker_identity.json')
            assert not process_matches(identity['process'])
            assert not (directory/'exit.json').exists() and not (directory/'completed.json').exists()
            assert not (Path(jobs[jid]['output_dir'])/'final_summary.json').exists()
            assert not jobs[jid]['smoke']
        elif r['status']=='completed':
            receipt=H/'runtime/jobs'/jid/'completed.json';summary=Path(jobs[jid]['output_dir'])/'final_summary.json'
            proof=read_json(receipt);assert proof['passed'] and proof['status']=='completed' and proof['exit']['returncode']==0
            assert proof['summary_sha256']==sha256(summary)==r['summary_sha256']
            completed[jid]={'receipt_sha256':sha256(receipt),'summary_sha256':sha256(summary)}
        else:
            assert r['status']=='pending'
            assert not (H/'runtime/jobs'/jid).exists() and not Path(jobs[jid]['output_dir']).exists(), 'Unexpected pending artifacts '+jid
    smoke=validate_smokes(suite);results=collect(suite,write=False)
    assert smoke['passed'] and smoke['smokes']==60
    assert results['completed_new']==6 and results['completed_three_seed_conditions']==3
    return suite,state,completed,canonical

def witness(suite,gpu):
    devices=query_gpus();g=devices[gpu]
    assert g['uuid']==suite['gpu_uuids'][str(gpu)]
    assert g['memory_used_mb']<500 and g['utilization_percent']<5,'Selected GPU is busy'
    lockdir=Path(suite['gpu_lock_dir']);lockdir.mkdir(exist_ok=True);held=[]
    try:
        for name in (f'gpu{gpu}.lock',g['uuid']+'.lock'):
            handle=open(lockdir/name,'a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(handle)
        current=query_gpus()[gpu]
        assert current['uuid']==g['uuid'] and current['memory_used_mb']<500 and current['utilization_percent']<5
        env=os.environ.copy();env.update(suite['environment']);env['CUDA_VISIBLE_DEVICES']=g['uuid']
        test=read_json(H/'environment_spec.json')['smoke']['gpu_tests'][0]
        p=subprocess.run(shlex.split(test['cmd']),env=env,cwd=H,capture_output=True,text=True,check=True,pass_fds=tuple(h.fileno() for h in held))
        assert re.search(test['expect'],p.stdout,re.MULTILINE)
        return {'passed':True,'physical_gpu':gpu,'uuid':g['uuid'],'stdout':p.stdout,'optimizer_updates':0}
    finally:
        for handle in held:handle.close()

def tree_hashes(path):
    return {str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}

def main():
    ap=argparse.ArgumentParser();a=ap.add_mutually_exclusive_group(required=True);a.add_argument('--check-only',action='store_true');a.add_argument('--apply',action='store_true');ap.add_argument('--gpu-witness',type=int,choices=[4,5,6]);args=ap.parse_args()
    # Shared original controller lock prevents mutation concurrent with any scheduler.
    with open(H/'runtime/controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite,state,completed,env_hash=inspect()
        if args.check_only:
            report={'passed':True,'at':now(),'formal_completed':6,'smoke_completed':60,'restart_from_task0':PLAN['interrupted_jobs'],'unstarted':51,'remaining':54,'allowed_gpus':[4,5,6],'environment_canonical_sha256':env_hash,'frozen_files_unchanged':True,'completed_evidence':completed}
            if args.gpu_witness is not None:report['cuda_witness']=witness(suite,args.gpu_witness)
            atomic_json(D/'PREFLIGHT.json',report)
            print(json.dumps({k:v for k,v in report.items() if k!='completed_evidence'},ensure_ascii=False));return
        assert args.gpu_witness is None
        check=read_json(D/'PREFLIGHT.json');doc=read_json(D/'DOC_CHECK.json')
        assert check['passed'] and check['cuda_witness']['passed'] and doc['passed']
        assert not (D/'RECOVERY_APPLIED.json').exists(),'Recovery already applied; do not repeat'
        assert not (D/'ARCHIVE_INDEX.json').exists(),'Partial recovery exists; inspect journal before continuing'
        interrupted=D/'interrupted';interrupted.mkdir()
        # Capture immutable evidence before moves; each rename stays on the same filesystem.
        moves=[]
        for jid in PLAN['interrupted_jobs']:
            for category,source in [('runtime',H/'runtime/jobs'/jid),('output',H/'outputs'/jid)]:
                dest=interrupted/jid/category
                assert source.is_dir() and not dest.exists()
                moves.append({'job':jid,'source':str(source),'archive':str(dest),'files':tree_hashes(source),'state':'prepared'})
        backup=D/'before_recovery';backup.mkdir()
        for relative in ['runtime/queue_state.json','runtime/controller.json','controller.log','LAUNCH.json','runtime/SMOKES_PASSED.json','summary.csv','THREE_SEED_SUMMARY.csv','results.json','RESULTS.md']:
            source=H/relative
            if source.exists():
                dest=backup/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        # The unmodified controller revalidates completed jobs on restart.
        # Preserve earlier mutable validator reports before those are refreshed.
        for jid in completed:
            for name in ['validation.json','validation.console.log']:
                source=H/'runtime/jobs'/jid/name
                if source.exists():
                    dest=backup/'runtime/jobs'/jid/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        journal={'created_at':now(),'moves':moves,'completed_evidence_before':completed,'old_state_sha256':sha256(H/'runtime/queue_state.json')}
        atomic_json(D/'ARCHIVE_INDEX.json',journal)
        for move in moves:
            source=Path(move['source']);dest=Path(move['archive']);dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
            assert tree_hashes(dest)==move['files'],'Archive hash mismatch'
            move['state']='archived_and_verified';atomic_json(D/'ARCHIVE_INDEX.json',journal)
        new=copy.deepcopy(state)
        for jid in PLAN['interrupted_jobs']:
            new['jobs'][jid]={'status':'pending','attempt':2,'previous_attempt_archive':str(interrupted/jid),'recovery_reason':'host shutdown interrupted incomplete training; restart from Task0 to preserve complete native trajectory'}
        assert all(new['jobs'][jid]==old for jid,old in state['jobs'].items() if jid not in PLAN['interrupted_jobs'])
        new.update(status='recovery_prepared',updated_at=now(),counts={'completed':66,'pending':54},recovery={'directory':str(D),'at':now(),'restart_jobs':PLAN['interrupted_jobs']})
        atomic_json(H/'runtime/queue_state.json',new)
        for jid,proof in completed.items():
            j=next(j for j in suite['jobs'] if j['id']==jid)
            assert sha256(H/'runtime/jobs'/jid/'completed.json')==proof['receipt_sha256']
            assert sha256(Path(j['output_dir'])/'final_summary.json')==proof['summary_sha256']
        verify_pins(suite)
        result={'passed':True,'at':now(),'new_state_sha256':sha256(H/'runtime/queue_state.json'),'restarted_jobs':PLAN['interrupted_jobs'],'completed_jobs_preserved':66,'formal_remaining':54,'archived_files':sum(len(m['files']) for m in moves),'scientific_code_and_configs_unchanged':True}
        atomic_json(D/'RECOVERY_APPLIED.json',result);print(json.dumps(result))

if __name__=='__main__':main()
