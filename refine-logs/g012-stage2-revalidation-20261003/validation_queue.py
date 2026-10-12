"""Queue zero-update diagnostics on idle GPUs; never launches training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ORIGINAL = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
sys.path.insert(0, str(ORIGINAL))
from common import atomic_json, assert_host, now, proc_identity, query_gpus, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    assert_host(manifest)
    root = Path(__file__).resolve().parent
    allowed_scripts = {str(root/'code/gradient_revalidation.py'),str(root/'code/hvp_full_revalidation.py')}
    runtime = Path(manifest['runtime'])
    if not runtime.resolve().is_relative_to(root):
        raise ValueError('Runtime must be inside the independent revalidation directory')
    runtime.mkdir(parents=True, exist_ok=True)
    controller_lock = open(runtime/'controller.lock', 'a+')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if set(manifest['allowed_gpus']) != {2,3,4,5,6}:
        raise RuntimeError('GPU0/1 remain excluded')
    state_path = runtime/'queue_state.json'
    if state_path.exists():
        raise FileExistsError('A previous diagnostic attempt exists; do not silently relaunch it')
    jobs = manifest['jobs']
    outputs = []
    for job in jobs:
        output = Path(job['output_dir']).resolve()
        if not output.is_relative_to(root) or not Path(job['report']).resolve().is_relative_to(output):
            raise ValueError('Diagnostic outputs must be isolated in this new directory')
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in outputs):
            raise ValueError('Diagnostic outputs overlap')
        outputs.append(output)
        if job['command'][0]!=manifest['python'] or job['command'][1] not in allowed_scripts:
            raise ValueError('Only reviewed zero-update diagnostics may be launched')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise RuntimeError('Duplicate job ids')
    state = {'run_id':manifest['run_id'], 'manifest_sha256':sha256(manifest_path),
             'status':'starting', 'controller':proc_identity(os.getpid()),
             'jobs':{j['id']:{'status':'pending'} for j in jobs}, 'optimizer_updates':0}
    active, leases = {}, {}
    def pins():
        if sha256(manifest_path) != state['manifest_sha256']:
            raise RuntimeError('Diagnostic manifest changed')
        for path, expected in manifest['pins'].items():
            if sha256(path) != expected:
                raise RuntimeError('Pinned artifact changed: '+path)
    pins()
    atomic_json(state_path, state)
    while True:
        for jid, child in list(active.items()):
            if child.poll() is None:
                continue
            row = state['jobs'][jid]
            job = next(j for j in jobs if j['id']==jid)
            receipt = runtime/'jobs'/jid/'exit.json'
            try:
                pins()
                exit_record = json.loads(receipt.read_text())
                report = json.loads(Path(job['report']).read_text())
                if any(report[key]!=0 for key in ('optimizer_updates','fisher_calls','weight_merge_calls')):
                    raise RuntimeError('Diagnostic performed forbidden work')
                passed = exit_record['job_id']==jid and exit_record['returncode']==0 and report['status']=='passed'
                row.update(status='completed' if passed else 'failed', completed_at=now(),
                           exit=exit_record, report_status=report['status'],
                           report_sha256=sha256(job['report']))
            except Exception as error:
                row.update(status='failed', completed_at=now(), error=repr(error))
            for handle in leases.pop(jid):
                handle.close()
            active.pop(jid)
            print(json.dumps({'event':'diagnostic_finished','job':jid,'status':row['status']}), flush=True)
            atomic_json(state_path, state)
        pending = [j for j in jobs if state['jobs'][j['id']]['status']=='pending']
        if not active and not pending:
            state.update(status='completed' if all(r['status']=='completed' for r in state['jobs'].values()) else 'completed_with_failed_gates',
                         updated_at=now())
            atomic_json(state_path, state)
            return
        if not (runtime/'PAUSE').exists():
            gpus = query_gpus()
            used = {state['jobs'][jid]['gpu'] for jid in active}
            for job in pending:
                chosen = None
                for index in manifest['allowed_gpus']:
                    gpu = gpus.get(index)
                    if not gpu or index in used or gpu['memory_used_mb']>=500 or gpu['utilization_percent']>=5:
                        continue
                    held = []
                    for name in (f'gpu{index}.lock', gpu['uuid']+'.lock'):
                        handle = open(Path(manifest['gpu_lock_dir'])/name, 'a+')
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            handle.close()
                            break
                        held.append(handle)
                    current = query_gpus().get(index)
                    if len(held)!=2 or not current or current['uuid']!=gpu['uuid'] or current['memory_used_mb']>=500 or current['utilization_percent']>=5:
                        for handle in held:
                            handle.close()
                        continue
                    chosen = (index,gpu,held)
                    break
                if chosen is None:
                    break
                pins()
                index,gpu,held = chosen
                jid = job['id']
                directory = runtime/'jobs'/jid
                directory.mkdir(parents=True, exist_ok=False)
                if Path(job['output_dir']).exists():
                    raise FileExistsError('Refusing to reuse output: '+job['output_dir'])
                record = dict(job, cwd=manifest['source_root'], cuda_witness_output=str(directory/'cuda_witness.json'))
                atomic_json(directory/'job.json',record)
                env = dict(os.environ)
                env.update(manifest['environment'])
                env['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
                with (directory/'console.log').open('xb') as log:
                    child = subprocess.Popen([manifest['python'],str(ORIGINAL/'job_worker.py'),
                        '--job',str(directory/'job.json'),'--lease-fd',str(held[0].fileno()),
                        '--uuid-lease-fd',str(held[1].fileno())], env=env,cwd=ORIGINAL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        pass_fds=tuple(h.fileno() for h in held))
                active[jid],leases[jid] = child,held
                used.add(index)
                state['jobs'][jid].update(status='running',gpu=index,gpu_uuid=gpu['uuid'],
                    process=proc_identity(child.pid),started_at=now(),command=job['command'])
                print(json.dumps({'event':'diagnostic_launched','job':jid,'gpu':index,'pid':child.pid}),flush=True)
        state.update(status='running' if active else 'waiting_for_idle_gpu',updated_at=now())
        atomic_json(state_path,state)
        time.sleep(10)


if __name__=='__main__':
    main()
