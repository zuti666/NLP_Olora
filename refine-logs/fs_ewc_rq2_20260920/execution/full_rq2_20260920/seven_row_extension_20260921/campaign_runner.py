"""Idle-GPU DAG runner for the authorized RQ2 campaign; no launch without --execute."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading


GPU_POLL_SECONDS = 15 * 60  # User-requested idle resource polling interval.


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    """Caller must hold the runner mutex when saving shared pipeline state."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)
    if path.name == 'pipeline_state.json':
        active = ', '.join(data.get('active_stages', [])) or '无'
        lines = ['# RQ2 自动执行状态', '', f"更新时间：{data.get('updated_utc', now())}",
                 f"总状态：{data.get('status')}；活动阶段：{active}", '',
                 '| 阶段 | 依赖 | 状态 |', '|---|---|---|']
        for sid, row in data.get('stages', {}).items():
            lines.append(f"| {sid} | {', '.join(row['depends_on']) or '无'} | {row['status']} |")
        lines += ['', '| 作业 | GPU | 状态 | 日志 |', '|---|---|---|---|']
        for jid, row in data.get('jobs', {}).items():
            log = Path(row['log']).name if row.get('log') else None
            gpu = row.get('gpu')
            lines.append(f"| {jid} | {gpu if gpu is not None else 'CPU'} | {row.get('status')} | "
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


def process_identity(pid):
    """Linux PID plus start tick and boot ID, so PID reuse is not mistaken for a job."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] in ('Z', 'X'):
        return None
    return dict(pid=int(pid), start_ticks=int(fields[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def validate_plan(plan):
    gpus = plan['gpus']
    if (not isinstance(gpus, list) or len(set(gpus)) != len(gpus)
            or any(type(g) is not int or g not in range(7) for g in gpus)):
        raise ValueError('Only unique authorized physical GPUs 0 through 6 are allowed')
    maximum = plan.get('max_parallel', 4)
    if type(maximum) is not int or not 1 <= maximum <= 7:
        raise ValueError('max_parallel must be between 1 and 7')
    stages = plan['stages']
    ids = [s['id'] for s in stages]
    if not stages or len(set(ids)) != len(ids):
        raise ValueError('Stages must be nonempty and have unique IDs')
    by_id = {s['id']: s for s in stages}
    jobs = []
    for stage in stages:
        dependencies = stage.get('depends_on')
        if not isinstance(dependencies, list):
            raise ValueError(f"Explicit depends_on list required: {stage['id']}")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"Duplicate dependency: {stage['id']}")
        if any(d not in by_id for d in dependencies):
            raise ValueError(f"Unknown dependency: {stage['id']}")
        if not stage['jobs']:
            raise ValueError(f"Empty stage: {stage['id']}")
        jobs.extend(stage['jobs'])
    visiting, visited = set(), set()

    def visit(sid):
        if sid in visiting:
            raise ValueError(f'Dependency cycle at {sid}')
        if sid in visited:
            return
        visiting.add(sid)
        for dep in by_id[sid]['depends_on']:
            visit(dep)
        visiting.remove(sid)
        visited.add(sid)

    for sid in ids:
        visit(sid)
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job ID')
    for job in jobs:
        if job['gpu'] is not None and job['gpu'] not in gpus:
            raise ValueError(f"Unauthorized GPU: {job['id']}")
        if any(g not in gpus for g in job.get('gpu_pool', [])):
            raise ValueError(f"Unauthorized GPU pool: {job['id']}")
        if not job['command'] or not Path(job['command'][0]).is_file():
            raise ValueError(f"Missing executable: {job['id']}")
        if not isinstance(job.get('receipt_require'), dict) or not job['receipt_require']:
            raise ValueError(f"Nonempty receipt_require needed: {job['id']}")
    return jobs


def check_receipt(job):
    """Validate all configured gates, including sanity and path numeric witnesses."""
    path = Path(job['receipt'])
    try:
        payload = path.read_bytes()
        result = json.loads(payload)
        valid = all(result.get(key) == value for key, value in job['receipt_require'].items())
        return valid, hashlib.sha256(payload).hexdigest(), None if valid else 'Receipt requirements not satisfied'
    except Exception as exc:
        return False, None, repr(exc)


class PipelineRunner:
    ACTIVE = {'scheduled', 'waiting_for_gpu', 'environment_preflight', 'running'}

    def __init__(self, plan, frozen, state, state_file, inherited_fds):
        self.plan, self.frozen, self.state = plan, frozen, state
        self.state_file, self.run = state_file, state_file.parent
        self.inherited_fds = inherited_fds
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.state['status'] = 'running'
        self.state['pid'] = os.getpid()
        self.state['process_identity'] = process_identity(os.getpid())
        self.state.pop('stage', None)
        self.state.pop('error', None)
        with self.mutex:
            self._save_locked()

    def _refresh_stages_locked(self):
        rows = {}
        complete = {s['id'] for s in self.plan['stages']
                    if all(self.state['jobs'][j['id']]['status'] == 'complete' for j in s['jobs'])}
        active = []
        for stage in self.plan['stages']:
            statuses = {self.state['jobs'][j['id']]['status'] for j in stage['jobs']}
            if statuses & self.ACTIVE:
                active.append(stage['id'])
            if stage['id'] in complete:
                status = 'complete'
            elif 'failed' in statuses:
                status = 'failed'
            elif statuses & self.ACTIVE:
                status = 'running'
            elif self.stop.is_set():
                status = 'blocked'
            elif set(stage['depends_on']) <= complete:
                status = 'ready'
            else:
                status = 'pending'
            rows[stage['id']] = dict(status=status, depends_on=stage['depends_on'],
                                    jobs=[j['id'] for j in stage['jobs']])
        self.state['stages'] = rows
        self.state['active_stages'] = active

    def _save_locked(self):
        self._refresh_stages_locked()
        self.state['updated_utc'] = now()
        write_json(self.state_file, self.state)

    def update(self, job_id, **fields):
        with self.mutex:
            self.state['jobs'][job_id].update(fields)
            if fields.get('status') == 'failed':
                self.stop.set()
                self.state['status'] = 'draining_after_failure'
                self.state.setdefault('error', f'Job {job_id} failed; no further jobs will start')
            self._save_locked()

    def cancel_if_stopped(self, jid):
        if self.stop.is_set():
            self.update(jid, status='cancelled', ended_utc=now(),
                        error='Another job failed before this job started')
            return True
        return False

    def start_process(self, job, launch_command, env, out, phase, **fields):
        # Serialize spawn with failure publication: once a failure is known no
        # preflight or experiment process can slip through the stop check.
        with self.mutex:
            if self.stop.is_set():
                self.state['jobs'][job['id']].update(status='cancelled', ended_utc=now(),
                                                    error='Another job failed before launch')
                self._save_locked()
                return None
            p = subprocess.Popen(launch_command, cwd=job.get('cwd', str(self.run)),
                                 env=env, stdout=out, stderr=subprocess.STDOUT,
                                 pass_fds=self.inherited_fds + ((job['_gpu_lock_fd'],) if '_gpu_lock_fd' in job else ()))
            self.state['jobs'][job['id']].update(status=phase, pid=p.pid,
                                                process_identity=process_identity(p.pid), **fields)
            self._save_locked()
            return p

    def run_job(self, job):
        jid, gpu = job['id'], job['gpu']
        lane = job.pop('_gpu_lock_file', None)
        try:
            if self.cancel_if_stopped(jid):
                return False
            status = {'uuid': '', 'device': 'cpu'}
            if gpu is not None:
                lock_dir = Path(self.plan['gpu_lock_dir'])
                lock_dir.mkdir(parents=True, exist_ok=True)
                uuid = gpu_status(gpu)['uuid']
                if lane is None:
                    lane = (lock_dir / f'{uuid}.lock').open('a')
                # Nonblocking + event wait lets a sibling failure interrupt GPU waits.
                while True:
                    if self.cancel_if_stopped(jid):
                        return False
                    try:
                        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        job['_gpu_lock_fd'] = lane.fileno()
                        break
                    except BlockingIOError:
                        self.stop.wait(1)
                self.update(jid, status='waiting_for_gpu', gpu=gpu)
                while True:
                    if self.cancel_if_stopped(jid):
                        return False
                    status = gpu_status(gpu)
                    expected_uuid = self.plan.get('start_after', {}).get('gpu_uuids', {}).get(str(gpu))
                    if expected_uuid is not None and status['uuid'] != expected_uuid:
                        raise RuntimeError(f'GPU {gpu} UUID changed: {status["uuid"]}')
                    if (status['memory_mib'] < 500 and status['utilization'] < 5
                            and not status['has_compute_process']):
                        break
                    self.stop.wait(GPU_POLL_SECONDS)
            if self.cancel_if_stopped(jid):
                return False
            if fingerprint(self.frozen) != self.frozen:
                raise RuntimeError('Frozen execution code changed during pipeline')
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            env.update(job.get('env', {}))
            env['CUDA_VISIBLE_DEVICES'] = status['uuid']
            with self.mutex:
                attempt = int(self.state['jobs'][jid].get('attempt', 0)) + 1
            log = self.run / f'{jid}.attempt{attempt}.log'
            with log.open('w') as out:
                if job.get('preflight_command'):
                    preflight = self.start_process(job, job['preflight_command'], env, out,
                                                   'environment_preflight', log=str(log), attempt=attempt)
                    if preflight is None:
                        return False
                    try:
                        preflight_code = preflight.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        preflight.kill()
                        preflight.wait()
                        raise RuntimeError('CUDA environment witness timed out') from None
                    if preflight_code:
                        self.update(jid, status='failed', exit_code=preflight_code, ended_utc=now(),
                                    error='CUDA environment witness failed; diagnostic not started')
                        return False
                command = [part.replace('{gpu}', str(gpu)) for part in job['command']]
                p = self.start_process(job, command, env, out, 'running',
                                       started_utc=now(), gpu_status=status, command=command,
                                       log=str(log), attempt=attempt)
                if p is None:
                    return False
                print(now(), 'START', jid, 'GPU', gpu, 'PID', p.pid, flush=True)
                code = p.wait()
            valid, receipt_hash, error = check_receipt(job)
            success = code == 0 and valid
            self.update(jid, status='complete' if success else 'failed', exit_code=code,
                        ended_utc=now(), receipt=job['receipt'], receipt_valid=valid,
                        receipt_error=error, receipt_sha256=receipt_hash)
            print(now(), 'END', jid, 'exit', code, 'valid', valid, flush=True)
            return success
        except Exception as exc:
            self.update(jid, status='failed', ended_utc=now(), error=repr(exc))
            return False
        finally:
            if lane is not None:
                job.pop('_gpu_lock_fd', None)
                lane.close()

    def run_all(self):
        futures, occupied = {}, set()
        try:
            with ThreadPoolExecutor(max_workers=self.plan.get('max_parallel', 4)) as pool:
                while True:
                    with self.mutex:
                        self._refresh_stages_locked()
                        if not self.stop.is_set():
                            complete = {sid for sid, s in self.state['stages'].items() if s['status'] == 'complete'}
                            for stage in self.plan['stages']:
                                if not set(stage['depends_on']) <= complete:
                                    continue
                                for job in stage['jobs']:
                                    if len(futures) >= self.plan.get('max_parallel', 4):
                                        break
                                    if self.state['jobs'][job['id']]['status'] != 'queued':
                                        continue
                                    gpu = job['gpu']
                                    reserved_lane = None
                                    if gpu is not None:
                                        pool_candidates = job.get('gpu_pool', [gpu])
                                        gpu = None
                                        for candidate in pool_candidates:
                                            if candidate in occupied:
                                                continue
                                            current = gpu_status(candidate)
                                            if current['memory_mib'] < 500 and current['utilization'] < 5 and not current['has_compute_process']:
                                                lock_dir = Path(self.plan['gpu_lock_dir'])
                                                lock_dir.mkdir(parents=True, exist_ok=True)
                                                candidate_lane = (lock_dir / f"{current['uuid']}.lock").open('a')
                                                try:
                                                    fcntl.flock(candidate_lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                                except BlockingIOError:
                                                    candidate_lane.close()
                                                    continue
                                                reserved_lane = candidate_lane
                                                gpu = candidate
                                                break
                                        if gpu is None:
                                            continue
                                    launch_job = dict(job, gpu=gpu)
                                    if reserved_lane is not None:
                                        launch_job['_gpu_lock_file'] = reserved_lane
                                    self.state['jobs'][job['id']].update(status='scheduled', gpu=gpu)
                                    if gpu is not None:
                                        occupied.add(gpu)
                                    futures[pool.submit(self.run_job, launch_job)] = launch_job
                        self._save_locked()
                    if not futures:
                        if self.stop.is_set():
                            break
                        if all(s['status'] == 'complete' for s in self.state['stages'].values()):
                            break
                        ready = any(s['status'] == 'ready' for s in self.state['stages'].values())
                        if ready:
                            self.stop.wait(GPU_POLL_SECONDS)
                            continue
                        raise RuntimeError('No runnable jobs remain but dependencies are incomplete')
                    done, _ = wait(futures, timeout=GPU_POLL_SECONDS, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = futures.pop(future)
                        if job['gpu'] is not None:
                            occupied.remove(job['gpu'])
                        if not future.result():
                            self.stop.set()
                with self.mutex:
                    if self.stop.is_set():
                        for row in self.state['jobs'].values():
                            if row['status'] == 'queued':
                                row.update(status='blocked', error='Pipeline stopped after failure')
                        self.state['status'] = 'failed'
                    else:
                        self.state['status'] = 'complete'
                    self._save_locked()
            if self.stop.is_set():
                raise RuntimeError(self.state.get('error', 'Pipeline failed'))
            return self.state
        except Exception as exc:
            with self.mutex:
                self.stop.set()
                self.state['status'] = 'failed'
                self.state.setdefault('error', repr(exc))
                self._save_locked()
            raise


def execute_plan(plan, plan_path, resume=False):
    jobs = validate_plan(plan)
    frozen = fingerprint(plan['code_files'] + [str(Path(plan_path).resolve())])
    run = Path(plan['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    # The pipeline lock also reaches children. If the scheduler dies, a child
    # keeps this run locked until it exits, preventing unsafe duplicate resume.
    with (run / 'pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Pipeline or surviving child still holds pipeline.lock') from None
        state_file = run / 'pipeline_state.json'
        if state_file.exists():
            if not resume:
                raise RuntimeError('Run exists. Inspect state and use --resume.')
            state = json.loads(state_file.read_text())
            if state['fingerprints'] != frozen:
                raise RuntimeError('Code/plan changed; use a new run directory to preserve provenance.')
            # Verify every completed receipt before releasing any dependency.
            for job in jobs:
                prior = state['jobs'][job['id']]
                if prior['status'] == 'complete':
                    valid, receipt_hash, error = check_receipt(job)
                    if not valid or receipt_hash != prior.get('receipt_sha256'):
                        raise RuntimeError(f'Completed receipt changed or invalid: {job["id"]}: {error}')
                elif prior.get('pid'):
                    live = process_identity(prior['pid'])
                    if live and (not prior.get('process_identity') or live == prior['process_identity']):
                        raise RuntimeError(f'Refusing resume while prior job is alive: {job["id"]}')
            for prior in state['jobs'].values():
                if prior['status'] != 'complete':
                    prior.update(status='queued')
                    for key in ('pid', 'process_identity', 'error'):
                        prior.pop(key, None)
            state['resumed_utc'] = now()
        else:
            state = dict(created_utc=now(), fingerprints=frozen,
                         jobs={j['id']: {'status': 'queued', 'gpu': j['gpu']} for j in jobs})
        inherited_fds = list(json.loads(os.environ.get('RQ2_GPU_LOCK_FDS', '{}')).values())
        if os.environ.get('RQ2_QUEUE_LOCK_FD'):
            inherited_fds.append(int(os.environ['RQ2_QUEUE_LOCK_FD']))
        inherited_fds.append(lock.fileno())
        inherited_fds = tuple(sorted(set(int(fd) for fd in inherited_fds)))
        for fd in inherited_fds:
            os.fstat(fd)
        return PipelineRunner(plan, frozen, state, state_file, inherited_fds).run_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = validate_plan(plan)
    if not args.execute:
        print(json.dumps({'dry_run': True, 'stages': len(plan['stages']), 'jobs': len(jobs),
                          'max_parallel': plan.get('max_parallel', 4),
                          'dependencies': {s['id']: s['depends_on'] for s in plan['stages']},
                          'fingerprints': fingerprint(plan['code_files'] + [str(args.plan.resolve())])}, indent=2))
        return
    execute_plan(plan, args.plan, args.resume)


if __name__ == '__main__':
    main()
