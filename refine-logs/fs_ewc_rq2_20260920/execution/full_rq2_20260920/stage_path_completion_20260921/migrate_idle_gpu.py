"""Move only unstarted jobs to free GPUs; preserve all completed measurements."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = HERE / 'campaign'
NEW = HERE / 'campaign_idle_gpu146'
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
sys.path.insert(0, str(BASE))
import campaign_runner as scheduler

GPU_UUIDS = {'1':'GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a',
            '4':'GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5',
            '6':'GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    scheduler.write_json(Path(path), value)


def check_payload_files(receipt_path):
    receipt_path=Path(receipt_path)
    value=read(receipt_path)
    if value.get('job')=='path':
        assert value['status']=='complete'
        for key,name in [('summary_sha256','summary.json'),('manifest_sha256','manifest.json'),('states_sha256','states.jsonl')]:
            assert sha(receipt_path.parent/name)==value[key], f'Completed path payload changed: {receipt_path}/{name}'
        assert value['wrapper_sha256']==sha(HERE/'diagnostic_extension.py')


def old_snapshot():
    state, plan = read(OLD/'pipeline_state.json'), read(OLD/'plan.json')
    assert state['status'] == 'running'
    assert all(v['status'] in ('complete','queued') for v in state['jobs'].values()), 'Old jobs must all be idle or completed'
    assert scheduler.fingerprint(state['fingerprints']) == state['fingerprints'], 'Original frozen sources changed'
    completed = []
    for stage in plan['stages']:
        for job in stage['jobs']:
            row = state['jobs'][job['id']]
            if row['status'] == 'complete':
                valid, digest, error = scheduler.check_receipt(job)
                assert valid and digest == row['receipt_sha256'], (job['id'], error)
                check_payload_files(job['receipt'])
                completed.append(dict(id=job['id'],path=job['receipt'],sha256=digest,
                                      requirements=job['receipt_require']))
    assert len(completed) == 28 and state['jobs']['wait_prior']['status'] == 'complete'
    return state, plan, completed


def prepare():
    NEW.mkdir(exist_ok=True)
    assert not (NEW/'launcher.json').exists(), 'Already launched; never rebuild its plan'
    state, old_plan, completed = old_snapshot()
    paths, radii, publishes = [], [], {}
    for stage in old_plan['stages']:
        for original in stage['jobs']:
            if state['jobs'][original['id']]['status'] == 'complete':
                continue
            job = copy.deepcopy(original)
            if job['gpu'] is not None:
                is_path = job['id'].startswith('path_')
                job['gpu'] = 1 if is_path else 6
                job['gpu_pool'] = [1,4] if is_path else [6,1,4]
                job['preflight_command'][-1] = str(NEW/f"{job['id']}_env.json")
                (paths if is_path else radii).append(job)
            else:
                publishes[job['id']] = job
    assert len(paths) == 9 and len(radii) == 6 and set(publishes) == {'publish_B1','publish_B2'}
    reused = dict(old_controller=state['process_identity'], old_plan_sha256=sha(OLD/'plan.json'),
        original_fingerprints=state['fingerprints'], completed_receipts=completed,
        queued_ids=sorted(k for k,v in state['jobs'].items() if v['status']=='queued'),
        preserved_path_jobs=27, created_utc=scheduler.now())
    dump(NEW/'reused_sources.json',reused)
    guard = dict(id='verify_reused',gpu=None,command=[PYTHON,'-u',str(Path(__file__).resolve()),'--verify-reused'],
        cwd=str(HERE), receipt=str(NEW/'reuse_verified.json'),receipt_require={'status':'complete','preserved_path_jobs':27})
    plan = dict(schema='rq2_idle_gpu_migration_v1', purpose='User-authorized idle GPU1/4/6 reassignment; B1/B2 parallel',
        gpus=[1,4,6],max_parallel=3,run_dir=str(NEW),gpu_lock_dir=old_plan['gpu_lock_dir'],
        start_after={'gpu_uuids':GPU_UUIDS},
        code_files=sorted(set(old_plan['code_files']+[str(Path(__file__).resolve()),str(NEW/'reused_sources.json')])),
        stages=[dict(id='verify_reused',depends_on=[],jobs=[guard]),
                dict(id='paths',depends_on=['verify_reused'],jobs=paths),
                dict(id='radii',depends_on=['verify_reused'],jobs=radii),
                dict(id='publish_B2',depends_on=['paths'],jobs=[publishes['publish_B2']]),
                dict(id='publish_B1',depends_on=['radii'],jobs=[publishes['publish_B1']])])
    jobs = scheduler.validate_plan(plan)
    assert len(jobs)==18
    dump(NEW/'plan.json',plan)
    dump(NEW/'CPU_CHECK.json',dict(status='passed',total_jobs=18,remaining_paths=9,radius_jobs=6,
        preserved_path_jobs=27,max_parallel=3,gpus=[1,4,6],
        scientific_commands_configs_and_receipts_unchanged=True,
        fingerprints=scheduler.fingerprint(plan['code_files']+[str(NEW/'plan.json')]),
        branches_independent=True))
    print(json.dumps(dict(status='prepared',remaining_paths=9,radius_jobs=6,preserved_paths=27,gpus=[1,4,6]),indent=2))


def verify_reused():
    evidence = read(NEW/'reused_sources.json')
    assert scheduler.process_identity(evidence['old_controller']['pid']) != evidence['old_controller'], 'Old controller remains live'
    migration = read(OLD/'MIGRATED.json')
    assert migration['new_campaign']==str(NEW) and migration['old_controller']==evidence['old_controller']
    assert scheduler.fingerprint(evidence['original_fingerprints']) == evidence['original_fingerprints']
    for item in evidence['completed_receipts']:
        assert sha(item['path'])==item['sha256']
        result=read(item['path']); assert all(result.get(k)==v for k,v in item['requirements'].items())
        check_payload_files(item['path'])
    dump(NEW/'reuse_verified.json',dict(status='complete',preserved_path_jobs=27,
        verified_receipts=len(evidence['completed_receipts']),checked_utc=scheduler.now()))


def thread_children(pid):
    children=set()
    for path in Path(f'/proc/{pid}/task').glob('*/children'):
        children.update(int(p) for p in path.read_text().split())
    return children


def execute():
    assert (NEW/'plan.json').is_file()
    with (NEW/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (NEW/'launcher.json').exists(), 'Migration already launched'
        plan=read(NEW/'plan.json'); scheduler.validate_plan(plan)
        audit=read(NEW/'CPU_CHECK.json')
        assert scheduler.fingerprint(audit['fingerprints'])==audit['fingerprints']
        expected=read(NEW/'reused_sources.json')
        identity=expected['old_controller']; pid=identity['pid']
        assert scheduler.process_identity(pid)==identity, 'Controller identity changed'
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert str(HERE/'queue_extension.py').encode() in command and b'--execute' in command
        before_gpus={str(g):scheduler.gpu_status(g) for g in [1,4,6]}
        for gpu,row in before_gpus.items():
            assert row['uuid']==GPU_UUIDS[gpu]
            assert row['memory_mib']<500 and row['utilization']<5 and not row['has_compute_process'], f'GPU{gpu} is no longer idle'
        stopped=False
        try:
            os.kill(pid,signal.SIGSTOP); stopped=True
            for _ in range(50):
                statuses=[p.read_text().rsplit(')',1)[1].split()[0] for p in Path(f'/proc/{pid}/task').glob('*/stat')]
                if statuses and all(status in ('T','t') for status in statuses):break
                time.sleep(.02)
            else:raise RuntimeError('Controller did not stop for safe handoff')
            state, old_plan, completed=old_snapshot()
            assert state['process_identity']==identity
            assert not thread_children(pid), 'Live child found; refuse to stop controller'
            assert completed==expected['completed_receipts']
            assert sorted(k for k,v in state['jobs'].items() if v['status']=='queued')==expected['queued_ids']
            dump(NEW/'original_pipeline_state.json',state)
            os.kill(pid,signal.SIGTERM); os.kill(pid,signal.SIGCONT); stopped=False
            for _ in range(100):
                if scheduler.process_identity(pid)!=identity:break
                time.sleep(.05)
            else:raise RuntimeError('Old controller did not terminate; new controller not launched')
        except BaseException:
            if stopped and scheduler.process_identity(pid)==identity:os.kill(pid,signal.SIGCONT)
            raise
        with (OLD/'pipeline.lock').open('a') as old_lock:
            fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            migration=dict(status='migrated',utc=scheduler.now(),old_controller=identity,
                new_campaign=str(NEW),preserved_path_jobs=27,remaining_paths=9,radius_jobs=6,
                new_gpu_uuids=GPU_UUIDS,old_controller_stopped_without_active_children=True)
            dump(OLD/'MIGRATED.json',migration)
            state.update(status='migrated',migrated_to=str(NEW),migrated_utc=migration['utc'])
            dump(OLD/'pipeline_state.json',state)
            (OLD/'STATUS.md').write_text('# 队列已迁移至空闲 GPU1/4/6\n\n已完成27条路径保留，剩余补测在新队列执行。\n\n[查看实时状态](../campaign_idle_gpu146/STATUS.md)\n\n原始状态快照在新目录 original_pipeline_state.json；不要恢复旧队列。\n')
            dump(NEW/'migration.json',migration)
            command=[PYTHON,'-u',str(BASE/'campaign_runner.py'),'--plan',str(NEW/'plan.json'),'--execute']
            env=dict(os.environ,TMPDIR='/tmp',TMP='/tmp',TEMP='/tmp',OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            # Keep the old run locked for the lifetime of the successor and its children.
            env['RQ2_QUEUE_LOCK_FD']=str(old_lock.fileno())
            with (NEW/'supervisor.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen(command,cwd=HERE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(old_lock.fileno(),))
            launch=dict(pid=proc.pid,process_identity=scheduler.process_identity(proc.pid),
                command=command,started_utc=scheduler.now(),plan_sha256=sha(NEW/'plan.json'),
                gpu_status_at_handoff=before_gpus,old_controller=identity,detached=True)
            dump(NEW/'launcher.json',launch)
            print(json.dumps(launch,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--verify-reused',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.execute:execute()
    else:verify_reused()
