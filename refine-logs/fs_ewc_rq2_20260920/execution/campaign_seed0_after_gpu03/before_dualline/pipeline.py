"""Two GPU, stage-barrier experiment runner. No jobs run without --execute."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)
    if path.name == 'pipeline_state.json':
        lines = ['# RQ2 自动执行状态', '', f"更新时间：{data.get('updated_utc', now())}",
                 f"总状态：{data.get('status')}；当前阶段：{data.get('stage', 'initializing')}", '',
                 '| 作业 | GPU | 状态 | 日志 |', '|---|---|---|---|']
        for jid, row in data.get('jobs', {}).items():
            log = Path(row['log']).name if row.get('log') else None
            lines.append(f"| {jid} | {row.get('gpu', 'CPU')} | {row.get('status')} | "
                         + (f'[{log}]({log})' if log else '待启动') + ' |')
        if data.get('error'):
            lines += ['', '错误：' + data['error']]
        (path.parent / 'STATUS.md').write_text('\n'.join(lines) + '\n')


def fingerprint(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in paths}


def gpu_status(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    occupied = {line.strip() for line in processes.splitlines()}
    for line in output.splitlines():
        idx, uuid, mem, util = [x.strip() for x in line.split(',')]
        if int(idx) == gpu:
            return dict(index=gpu, uuid=uuid, memory_mib=int(mem), utilization=int(util),
                        has_compute_process=uuid in occupied)
    raise RuntimeError(f'GPU {gpu} missing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['gpus'] == [0, 1], 'This campaign uses authorized physical GPUs 0 and 1'
    inherited_gpu_fds = tuple(json.loads(os.environ.get('RQ2_GPU_LOCK_FDS', '{}')).values())
    jobs = [j for stage in plan['stages'] for j in stage['jobs']]
    assert len({j['id'] for j in jobs}) == len(jobs)
    for j in jobs:
        assert j['gpu'] is None or j['gpu'] in plan['gpus']
        assert Path(j['command'][0]).is_file(), j['command'][0]
    frozen = fingerprint(plan['code_files'] + [str(args.plan.resolve())])
    if not args.execute:
        print(json.dumps({'dry_run': True, 'stages': len(plan['stages']),
                          'jobs': len(jobs), 'fingerprints': frozen}, indent=2))
        return
    run = Path(plan['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = run / 'pipeline_state.json'
    if state_file.exists():
        if not args.resume:
            raise RuntimeError('Run exists. Inspect state and use --resume.')
        state = json.loads(state_file.read_text())
        if state['fingerprints'] != frozen:
            raise RuntimeError('Code/plan changed; use a new run directory to preserve provenance.')
    else:
        state = dict(created_utc=now(), fingerprints=frozen,
                     jobs={j['id']: {'status': 'queued', 'gpu': j['gpu']} for j in jobs},
                     status='initializing')
    state['pid'] = os.getpid()
    state['status'] = 'running'
    state['updated_utc'] = now()
    write_json(state_file, state)
    mutex = threading.Lock()

    def update(job_id, **fields):
        with mutex:
            state['jobs'].setdefault(job_id, {}).update(fields)
            state['updated_utc'] = now()
            write_json(state_file, state)

    def run_job(job):
        jid, gpu = job['id'], job['gpu']
        prior = state['jobs'].get(jid, {})
        if prior.get('status') == 'complete':
            receipt = Path(job['receipt'])
            if not receipt.exists() or hashlib.sha256(receipt.read_bytes()).hexdigest() != prior['receipt_sha256']:
                raise RuntimeError(f'Completed receipt changed: {jid}')
            return True
        # Dedicated lane locks prevent two jobs in this campaign using one GPU.
        lane = None
        status = {'uuid': '', 'device': 'cpu'}
        if gpu is not None:
            lane = (run / f'gpu{gpu}.lock').open('a')
            fcntl.flock(lane, fcntl.LOCK_EX)
            update(jid, status='waiting_for_gpu', gpu=gpu)
            while True:
                status = gpu_status(gpu)
                if (status['memory_mib'] < 500 and status['utilization'] < 5
                        and not status['has_compute_process']):
                    break
                time.sleep(15)
        if fingerprint(frozen) != frozen:
            raise RuntimeError('Frozen execution code changed during pipeline')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=status['uuid'], OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1')
        env.update(job.get('env', {}))
        attempt = int(prior.get('attempt', 0)) + 1
        log = run / f'{jid}.attempt{attempt}.log'
        with log.open('w') as out:
            if job.get('preflight_command'):
                update(jid, status='environment_preflight', log=str(log))
                preflight = subprocess.run(job['preflight_command'], cwd=str(run),
                                           env=env, stdout=out, stderr=subprocess.STDOUT,
                                           pass_fds=inherited_gpu_fds,
                                           timeout=180)
                if preflight.returncode:
                    update(jid, status='failed', exit_code=preflight.returncode,
                           error='CUDA environment witness failed; diagnostic not started')
                    if lane is not None:
                        lane.close()
                    return False
            p = subprocess.Popen(job['command'], cwd=job.get('cwd', str(run)),
                                 env=env, stdout=out, stderr=subprocess.STDOUT,
                                 pass_fds=inherited_gpu_fds)
            update(jid, status='running', started_utc=now(), pid=p.pid, gpu_status=status,
                   command=job['command'], log=str(log), attempt=attempt)
            print(now(), 'START', jid, 'GPU', gpu, 'PID', p.pid, flush=True)
            code = p.wait()
        receipt = Path(job['receipt'])
        valid = False
        error = None
        try:
            result = json.loads(receipt.read_text())
            valid = all(result.get(key) == value for key, value in job['receipt_require'].items())
        except Exception as exc:
            error = repr(exc)
        success = code == 0 and valid
        update(jid, status='complete' if success else 'failed', exit_code=code,
               ended_utc=now(), receipt=str(receipt), receipt_valid=valid,
               receipt_error=error,
               receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None)
        print(now(), 'END', jid, 'exit', code, 'valid', valid, flush=True)
        if lane is not None:
            fcntl.flock(lane, fcntl.LOCK_UN)
            lane.close()
        return success

    try:
        for stage in plan['stages']:
            state['stage'] = stage['id']
            state['updated_utc'] = now()
            write_json(state_file, state)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run_job, stage['jobs']))
            if not all(results):
                raise RuntimeError(f'Stage {stage["id"]} failed; later stages not launched.')
        state['status'] = 'complete'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = repr(exc)
        raise
    finally:
        state['updated_utc'] = now()
        write_json(state_file, state)


if __name__ == '__main__':
    main()
