"""Host queue: shared index/UUID leases, double idle checks, all-smoke barrier, no preemption."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import (HERE, assert_host, atomic_json, gpu_is_idle, load_suite, now, proc_identity,
                    process_matches, query_gpus, read_json, sha256, suite_counts, verify_pins, upstream_ready)
from validate_job import validate_job


def refresh_counts(state):
    state["counts"] = dict(Counter(record["status"] for record in state["jobs"].values()))
    state["updated_at"] = now()


def acquire_idle_gpu(suite, telemetry, used):
    lockdir = Path(suite["gpu_lock_dir"])
    lockdir.mkdir(parents=True, exist_ok=True)
    for index in suite["allowed_gpus"]:
        record = telemetry.get(index)
        if index in used or not gpu_is_idle(record):
            continue
        held = []
        try:
            for filename in (f"gpu{index}.lock", record["uuid"] + ".lock"):
                lease = open(lockdir / filename, "a+")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    break
                held.append(lease)
            if len(held) != 2:
                continue
            # Re-query BOTH memory/utilization and compute PIDs after acquiring both locks.
            current = query_gpus().get(index)
            if not gpu_is_idle(current) or current["uuid"] != record["uuid"]:
                continue
            result = (index, current, held)
            held = []
            return result
        finally:
            for lease in held:
                lease.close()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(HERE / "suite.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    assert_host(suite)
    verify_pins(suite)
    if args.dry_run:
        for job in suite["jobs"]:
            if not Path(job["command"][1]).is_file():
                raise FileNotFoundError(job["command"][1])
        print(json.dumps(dict(status="valid", **suite_counts(suite), allowed_gpus=suite["allowed_gpus"],
                              max_parallel=suite["max_parallel"], idle_rule="memory<500MiB/util<5%/no compute PID",
                              upstream=upstream_ready(suite))))
        return 0
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=True)
    controller_lease = open(runtime / "controller.lock", "a+")
    fcntl.flock(controller_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = runtime / "queue_state.json"
    jobs = {job["id"]: job for job in suite["jobs"]}
    state = read_json(state_path) if state_path.exists() else dict(run_id=suite["run_id"], created_at=now(),
                                jobs={job["id"]: {"status": "pending"} for job in suite["jobs"]})
    if state["run_id"] != suite["run_id"] or set(state["jobs"]) != set(jobs):
        raise RuntimeError("Persisted queue belongs to a different manifest")
    if state.get("suite_sha256", sha256(args.suite)) != sha256(args.suite):
        raise RuntimeError("Persisted queue suite hash changed")
    for job_id, record in state["jobs"].items():
        if record["status"] == "launching":
            identity_path = runtime / "jobs" / job_id / "worker_identity.json"
            if identity_path.exists():
                identity = read_json(identity_path)
                if identity["job_id"] != job_id:
                    raise RuntimeError("Recovery identity mismatch")
                record.update(status="running", process=identity["process"])
            else:
                # Never duplicate an uncertain launch; a worker/runner may still own both leases.
                record.update(status="failed", error="Interrupted launch has no worker identity; inspect host before retry")
        if record["status"] == "completed":
            checked = validate_job(jobs[job_id])
            if checked["summary_sha256"] != record["summary_sha256"]:
                raise RuntimeError("Previously completed evidence changed: " + job_id)
    state.update(controller=proc_identity(os.getpid()), status="running", suite_sha256=sha256(args.suite))
    refresh_counts(state)
    atomic_json(state_path, state)
    atomic_json(runtime / "controller.json", dict(process=proc_identity(os.getpid()), started_at=now(),
                pid_namespace=os.readlink("/proc/self/ns/pid"), command=sys.argv, suite_sha256=sha256(args.suite)))
    active = {job_id: None for job_id, record in state["jobs"].items() if record["status"] == "running"}
    leases = {}
    for key in ("TMPDIR", "TMP", "TEMP"):
        Path(suite["environment"][key]).mkdir(parents=True, exist_ok=True)
    while True:
        for job_id in list(active):
            record = state["jobs"][job_id]
            worker = active[job_id]
            if worker is not None and worker.poll() is None:
                continue
            if worker is None and process_matches(record["process"]):
                continue
            directory = runtime / "jobs" / job_id
            try:
                exit_record = read_json(directory / "exit.json")
                if exit_record["job_id"] != job_id or exit_record["returncode"] != 0:
                    raise RuntimeError("Worker failed: " + repr(exit_record))
                verify_pins(suite)
                checked = validate_job(jobs[job_id])
                atomic_json(directory / "completed.json", dict(job_id=job_id, completed_at=now(),
                            exit=exit_record, **checked))
                record.update(status="completed", completed_at=now(), summary_sha256=checked["summary_sha256"])
                print(json.dumps(dict(event="completed", job=job_id)), flush=True)
            except Exception as exc:
                record.update(status="failed", failed_at=now(), error=repr(exc))
                atomic_json(directory / "failed.json", record)
                print(json.dumps(dict(event="failed", job=job_id, error=repr(exc))), flush=True)
            finally:
                active.pop(job_id)
                for lease in leases.pop(job_id, []):
                    lease.close()
                refresh_counts(state)
                atomic_json(state_path, state)
                subprocess.run([suite["python"], str(HERE / "collect_results.py")], cwd=HERE, check=True)
        failed = any(record["status"] == "failed" for record in state["jobs"].values())
        pending = [job for job in suite["jobs"] if state["jobs"][job["id"]]["status"] == "pending"]
        if failed and not active:
            state.update(status="halted_failure")
            refresh_counts(state)
            atomic_json(state_path, state)
            return 1
        if not pending and not active:
            state.update(status="completed")
            refresh_counts(state)
            atomic_json(state_path, state)
            return 0
        paused = (runtime / "PAUSE").exists()
        telemetry_error = None
        telemetry = {}
        upstream = upstream_ready(suite)
        if not paused and not failed and upstream["ready"]:
            try:
                telemetry = query_gpus()
            except Exception as exc:
                # Missing telemetry is never evidence that a GPU is idle.
                telemetry_error = repr(exc)
            used = {state["jobs"][job_id]["gpu"] for job_id in active if state["jobs"][job_id].get("gpu") is not None}
            for job in pending:
                if any(state["jobs"][dependency]["status"] != "completed" for dependency in job["depends_on"]):
                    continue
                for dependency in job["depends_on"]:
                    dependency_job = jobs[dependency]
                    report_file = dependency_job.get("report_file", "final_summary.json")
                    report_path = Path(dependency_job["output_dir"]) / report_file
                    if sha256(report_path) != state["jobs"][dependency]["summary_sha256"]:
                        raise RuntimeError("Dependency evidence changed: " + dependency)
                verify_pins(suite)
                cpu_only = job["kind"] == "pairing"
                gpu_record = None
                held = []
                if not cpu_only:
                    if len(used) >= suite["max_parallel"] or telemetry_error:
                        break
                    selected = acquire_idle_gpu(suite, telemetry, used)
                    if selected is None:
                        break
                    index, gpu_record, held = selected
                else:
                    index = None
                job_id = job["id"]
                directory = runtime / "jobs" / job_id
                directory.mkdir(parents=True, exist_ok=False)
                output = Path(job["output_dir"])
                if output.exists() and any(output.iterdir()):
                    for lease in held:
                        lease.close()
                    raise RuntimeError("Refusing to overwrite a nonempty output: " + job_id)
                launch_job = dict(job, cwd=suite["source_root"], suite_path=str(Path(args.suite).resolve()),
                                  cuda_witness_output=str(directory / "cuda_witness.json"),
                                  assigned_gpu_uuid=None if cpu_only else gpu_record["uuid"])
                atomic_json(directory / "job.json", launch_job)
                record = state["jobs"][job_id]
                record.update(status="launching", started_at=now(), gpu=index,
                              gpu_uuid=None if cpu_only else gpu_record["uuid"], command=job["command"])
                refresh_counts(state)
                atomic_json(state_path, state)
                environment = os.environ.copy()
                environment.update(suite["environment"])
                environment["CUDA_VISIBLE_DEVICES"] = "" if cpu_only else gpu_record["uuid"]
                command = [suite["python"], str(HERE / "job_worker.py"), "--job", str(directory / "job.json")]
                if cpu_only:
                    command += ["--cpu"]
                else:
                    command += ["--lease-fd", str(held[0].fileno()), "--uuid-lease-fd", str(held[1].fileno())]
                log = open(directory / "console.log", "xb")
                try:
                    worker = subprocess.Popen(command, cwd=HERE, env=environment, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True,
                                              pass_fds=tuple(lease.fileno() for lease in held))
                finally:
                    log.close()
                record.update(status="running", process=proc_identity(worker.pid))
                active[job_id] = worker
                leases[job_id] = held
                if index is not None:
                    used.add(index)
                atomic_json(directory / "launched.json", record)
                refresh_counts(state)
                atomic_json(state_path, state)
                print(json.dumps(dict(event="launched", job=job_id, gpu=index, pid=worker.pid)), flush=True)
        ready = any(all(state["jobs"][dependency]["status"] == "completed" for dependency in job["depends_on"]) for job in pending)
        state.update(paused=paused, telemetry_error=telemetry_error,
                     upstream=upstream,
                     status="halted_failure_draining" if failed else "paused" if paused else "running" if active else
                     "waiting_for_upstream_campaign" if not upstream["ready"] else
                     "waiting_for_gpu_telemetry" if telemetry_error else "waiting_for_idle_gpu" if ready else "waiting_for_dependencies")
        refresh_counts(state)
        atomic_json(state_path, state)
        time.sleep(suite["poll_seconds"])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        (HERE / "runtime").mkdir(exist_ok=True)
        atomic_json(HERE / "runtime/controller_error.json", dict(at=now(), error=repr(exc)))
        raise
