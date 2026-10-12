#!/usr/bin/env python3
"""Pinned DAG admission on idle physical GPU 4/5; failures stop new work."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE, atomic_json, assert_host, gpu_idle, load_suite, now, proc_identity, process_matches, query_gpus, read_json, sha256, verify_pins
from validate_job import validate_job

def acquire_idle_gpu(suite, excluded):
    lockdir = Path(suite["gpu_lock_dir"])
    lockdir.mkdir(exist_ok=True)
    for index in suite["allowed_gpus"]:
        if index in excluded:
            continue
        gpu = query_gpus().get(index)
        if not gpu_idle(gpu):
            continue
        held = []
        for filename in (f"gpu{index}.lock", gpu["uuid"] + ".lock"):
            lock = open(lockdir / filename, "a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                break
            held.append(lock)
        if len(held) != 2:
            for lock in held: lock.close()
            continue
        current = query_gpus().get(index)
        if not gpu_idle(current) or current["uuid"] != gpu["uuid"]:
            for lock in held: lock.close()
            continue
        return index, current, held
    return None

def runner_is_live(directory):
    p = directory / "runner_identity.json"
    return p.exists() and process_matches(read_json(p).get("process"))

def collect(suite):
    subprocess.run([suite["python"], str(HERE / "collect_results.py")], cwd=HERE, check=True)

def verify_launch_gates(suite):
    for key in ("implementation_acceptance_file", "fresh_doc_check"):
        gate = read_json(suite[key])
        if gate.get("status") != "passed" or not gate.get("checks") or not all(gate["checks"].values()):
            raise RuntimeError("Missing successful launch acceptance: " + key)
        for path, digest in gate.get("evidence_hashes", {}).items():
            if sha256(path) != digest:
                raise RuntimeError("Launch review evidence changed: " + path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default=str(HERE / "suite.json"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    suite_path = Path(a.suite).resolve()
    suite = load_suite(suite_path)
    assert_host(suite)
    verify_pins(suite, suite_path)
    for j in suite["jobs"]:
        if not Path(j["command"][1]).is_file():
            raise FileNotFoundError(j["command"][1])
    if a.dry_run:
        print({"status": "valid", "jobs": len(suite["jobs"]), "smokes": 10, "formal": 7, "physical_gpus": [4, 5], "max_parallel": 2})
        return 0
    verify_launch_gates(suite)
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=True)
    controller_lock = open(runtime / "controller.lock", "a+")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = runtime / "queue_state.json"
    ids = {j["id"]: j for j in suite["jobs"]}
    state = read_json(state_path) if state_path.exists() else {"run_id": suite["run_id"], "created_at": now(), "jobs": {j: {"status": "pending"} for j in ids}}
    if state["run_id"] != suite["run_id"] or set(state["jobs"]) != set(ids):
        raise RuntimeError("Persisted queue does not match immutable suite")
    if state.get("suite_sha256", sha256(suite_path)) != sha256(suite_path):
        raise RuntimeError("Persisted queue suite hash changed")
    for jid, r in state["jobs"].items():
        d = runtime / "jobs" / jid
        if r["status"] == "launching":
            path = d / "worker_identity.json"
            if path.exists() and read_json(path)["job_id"] == jid:
                r.update(status="running", process=read_json(path)["process"])
            else:
                r.update(status="failed", error="Interrupted launch quarantined; no automatic retry")
        if r["status"] == "completed":
            check = validate_job(ids[jid])
            if check["summary_sha256"] != r["summary_sha256"]:
                raise RuntimeError("Completed receipt changed: " + jid)
    state.update(status="running", controller=proc_identity(os.getpid()), suite_sha256=sha256(suite_path))
    atomic_json(runtime / "controller.json", {"process": proc_identity(os.getpid()), "started_at": now(), "pid_namespace": os.readlink("/proc/self/ns/pid"), "suite_sha256": sha256(suite_path)})
    active = {jid: None for jid, r in state["jobs"].items() if r["status"] == "running"}
    leases = {}
    for key in ("TMPDIR", "TMP", "TEMP"):
        if key in suite.get("environment", {}):
            Path(suite["environment"][key]).mkdir(parents=True, exist_ok=True)
    collect(suite)
    while True:
        for jid in list(active):
            r = state["jobs"][jid]
            worker = active[jid]
            if worker is not None and worker.poll() is None:
                continue
            if worker is None and process_matches(r.get("process")):
                continue
            d = runtime / "jobs" / jid
            if runner_is_live(d):
                state["admission_error"] = "Worker identity ended while trainer remains active: " + jid
                r["orphan_runner_alive"] = True
                continue
            try:
                receipt = read_json(d / "exit.json")
                if receipt["job_id"] != jid or receipt["returncode"] != 0 or not receipt.get("source_and_execution_pins_unchanged"):
                    raise RuntimeError("Worker failed: " + repr(receipt))
                verify_pins(suite, suite_path)
                accepted = validate_job(ids[jid])
                atomic_json(d / "completed.json", {"job_id": jid, "completed_at": now(), "exit": receipt, **accepted})
                r.update(status="completed", completed_at=now(), summary_sha256=accepted["summary_sha256"])
                print({"event": "completed", "job": jid}, flush=True)
            except Exception as exc:
                r.update(status="failed", failed_at=now(), error=repr(exc))
                atomic_json(d / "failed.json", r)
                print({"event": "failed", "job": jid, "error": repr(exc)}, flush=True)
            finally:
                active.pop(jid)
                for lock in leases.pop(jid, []): lock.close()
                atomic_json(state_path, state)
                try:
                    collect(suite)
                except Exception as exc:
                    state["admission_error"] = "Collector acceptance failed: " + repr(exc)
        failed = bool(state.get("admission_error")) or any(r["status"] == "failed" for r in state["jobs"].values())
        pending = [j for j in suite["jobs"] if state["jobs"][j["id"]]["status"] == "pending"]
        if failed and not active:
            state.update(status="halted_failure", updated_at=now())
            atomic_json(state_path, state)
            return 1
        if not pending and not active:
            state.update(status="completed", updated_at=now(), counts=dict(Counter(r["status"] for r in state["jobs"].values())))
            atomic_json(state_path, state)
            collect(suite)
            return 0
        paused = (runtime / "PAUSE").exists()
        if not failed and not paused:
            used = {state["jobs"][jid].get("gpu") for jid in active}
            for job in pending:
                if len(active) >= suite["max_parallel"]:
                    break
                if any(state["jobs"][dep]["status"] != "completed" for dep in job.get("depends_on", [])):
                    continue
                held = []
                try:
                    verify_pins(suite, suite_path)
                    for dep in job.get("depends_on", []):
                        accepted = validate_job(ids[dep])
                        if accepted["summary_sha256"] != state["jobs"][dep]["summary_sha256"]:
                            raise RuntimeError("Dependency receipt changed: " + dep)
                    gpu_index, gpu_uuid = None, None
                    if job["kind"] == "training":
                        chosen = acquire_idle_gpu(suite, used)
                        if chosen is None:
                            continue
                        gpu_index, gpu, held = chosen
                        gpu_uuid = gpu["uuid"]
                    jid = job["id"]
                    d = runtime / "jobs" / jid
                    d.mkdir(parents=True, exist_ok=False)
                    output = Path(job["output_dir"])
                    if output.exists() and any(output.iterdir()):
                        raise RuntimeError("Refusing nonempty output: " + jid)
                    saved_job = {**job, "gpu": gpu_index, "gpu_uuid": gpu_uuid, "cwd": suite["source_root"], "suite_path": str(suite_path), "suite_sha256": sha256(suite_path)}
                    atomic_json(d / "job.json", saved_job)
                    r = state["jobs"][jid]
                    r.update(status="launching", started_at=now(), gpu=gpu_index, gpu_uuid=gpu_uuid, command=job["command"])
                    atomic_json(state_path, state)
                    env = os.environ.copy()
                    env.update(suite.get("environment", {}))
                    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid or ""
                    if gpu_uuid:
                        env["SINGLE200_EXPECTED_GPU"] = str(gpu_index)
                        env["SINGLE200_EXPECTED_GPU_UUID"] = gpu_uuid
                    command = [suite["python"], str(HERE / "job_worker.py"), "--job", str(d / "job.json")]
                    if held:
                        command += ["--lease-fd", str(held[0].fileno()), "--uuid-lease-fd", str(held[1].fileno())]
                    with open(d / "console.log", "xb") as log:
                        worker = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(x.fileno() for x in held))
                    r.update(status="running", process=proc_identity(worker.pid))
                    active[jid] = worker
                    leases[jid] = held
                    held = []
                    used.add(gpu_index)
                    atomic_json(d / "launched.json", r)
                    atomic_json(state_path, state)
                    print({"event": "launched", "job": jid, "gpu": gpu_index, "pid": worker.pid}, flush=True)
                except Exception as exc:
                    for lock in held: lock.close()
                    state["admission_error"] = repr(exc)
                    atomic_json(runtime / "admission_error.json", {"at": now(), "job": job["id"], "error": repr(exc)})
                    break
        failed = bool(state.get("admission_error")) or any(r["status"] == "failed" for r in state["jobs"].values())
        ready = any(all(state["jobs"][dep]["status"] == "completed" for dep in j.get("depends_on", [])) for j in pending)
        state.update(updated_at=now(), paused=paused, status="halted_failure_draining" if failed else "paused" if paused else "running" if active else "waiting_for_idle_gpu" if ready else "waiting_for_dependencies", counts=dict(Counter(r["status"] for r in state["jobs"].values())))
        atomic_json(state_path, state)
        time.sleep(max(5, min(60, suite.get("poll_seconds", 30))))

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        atomic_json(HERE / "runtime/controller_error.json", {"at": now(), "error": repr(exc)})
        raise
