"""Bounded, fail-closed queue using the authorized physical GPUs in suite.json."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import subprocess
import time

from collect_results import collect
from common import (HERE, atomic_json, bad_loss_lines, exclusive_json, load_suite, now,
                    proc_identity, process_matches, query_gpus, read_json, resolve,
                    sha256, validate, verify_config, verify_source, verify_run_pins)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--job", action="append", help="Run only these job IDs; dependencies must already be completed")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    suite = load_suite()
    provenance = validate(suite)
    run_pins_hash = verify_run_pins(suite)
    selected = set(args.job or [job["id"] for job in suite["jobs"]])
    if selected - {job["id"] for job in suite["jobs"]}:
        parser.error("Unknown --job")
    print(json.dumps({"validated_at": now(), "host": socket.gethostname(),
                      "jobs": len(suite["jobs"]), "selected_jobs": sorted(selected),
                      "source_manifest_sha256": provenance["source_manifest_sha256"],
                      "gpus": query_gpus()}, sort_keys=True), flush=True)
    if args.validate_only:
        return 0
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "jobs").mkdir(exist_ok=True)
    lock = open(runtime / "campaign.lock", "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another queue holds the campaign lock")
    suite_hash = sha256(HERE / "suite.json")
    state_path = runtime / "state.json"
    if state_path.exists() and read_json(state_path)["suite_sha256"] != suite_hash:
        raise RuntimeError("Suite changed after queue initialization; refusing restart")
    states, active, owned = {}, {}, {}
    stop_pending = False
    jobs = {job["id"]: job for job in suite["jobs"]}
    smoke_verified = False

    def fail(job_id, error):
        nonlocal stop_pending
        run = runtime / "jobs" / job_id
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "queue_failed.json").exists():
            exclusive_json(run / "queue_failed.json", {"id": job_id, "status": "failed", "time": now(),
                                                       "error": error})
        states[job_id]["status"] = "failed"
        stop_pending = True

    # A preexisting job directory is never reused. Adopt a precise living process,
    # accept an immutable completion receipt, or refuse the interrupted job.
    for job_id, job in jobs.items():
        run = runtime / "jobs" / job_id
        states[job_id] = {"status": "pending"}
        if not run.exists():
            continue
        if (run / "failed.json").exists() or (run / "queue_failed.json").exists():
            states[job_id]["status"] = "failed"
            stop_pending = True
            continue
        if (run / "completed.json").exists():
            states[job_id]["status"] = "completed"
            continue
        records = [read_json(run / name) for name in ("started.json", "launch.json") if (run / name).exists()]
        identity = next((record for record in records if "proc_starttime" in record and process_matches(record)), None)
        if identity:
            if identity.get("host") != socket.gethostname() or identity.get("gpu") not in suite["allowed_gpus"]:
                raise RuntimeError(f"Cannot safely adopt process for {job_id}")
            command = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b"\0")
            if str(HERE / "run_job.py").encode() not in command or job_id.encode() not in command:
                raise RuntimeError(f"PID command mismatch for {job_id}")
            if identity["gpu"] in {record["gpu"] for record in active.values()}:
                raise RuntimeError("Multiple active jobs on one GPU")
            active[job_id] = identity
            states[job_id] = {"status": "running", **identity}
            print(f"Adopted {job_id} PID {identity['pid']}", flush=True)
        else:
            fail(job_id, "Existing job has no completed receipt or matching live PID; refusing to rerun")
    if len(active) > suite["max_parallel"]:
        raise RuntimeError("Too many adopted jobs")

    while True:
        if verify_run_pins(suite) != run_pins_hash:
            raise RuntimeError('Run pins changed after controller initialization')
        if sha256(HERE / "suite.json") != suite_hash:
            stop_pending = True
            print("Suite changed during execution; pending jobs are blocked", flush=True)
        for job_id, identity in list(active.items()):
            run = runtime / "jobs" / job_id
            bad = bad_loss_lines(run / "console.log")
            if bad and not (run / "queue_failed.json").exists():
                # Let the current process finish, but stop all pending launches.
                fail(job_id, "Nonfinite training loss: " + bad[0])
            process = owned.get(job_id)
            alive = (process.poll() is None) if process else process_matches(identity)
            if alive:
                continue
            active.pop(job_id)
            if (run / "failed.json").exists() or (run / "queue_failed.json").exists():
                states[job_id]["status"] = "failed"
                stop_pending = True
            elif (run / "completed.json").exists() and (not process or process.returncode == 0):
                states[job_id]["status"] = "completed"
            else:
                fail(job_id, f"Process exited without successful completion receipt; returncode={getattr(process, 'returncode', None)}")
        summary = collect(suite, states)
        for row in summary["jobs"]:
            if row["status"] in {"failed", "invalid"}:
                stop_pending = True
                if row["status"] == "invalid":
                    fail(row["id"], row.get("error", "Invalid completed metrics"))
        if not stop_pending:
            busy = {identity["gpu"] for identity in active.values()}
            for job_id, job in jobs.items():
                if len(active) >= suite["max_parallel"]:
                    break
                if job_id not in selected or states[job_id]["status"] != "pending":
                    continue
                if any(states[dep]["status"] != "completed" for dep in job.get("depends_on", [])):
                    continue
                if job['phase'] == 'formal' and not smoke_verified:
                    from validate_smokes import validate_smokes
                    try:
                        validate_smokes(suite)
                        smoke_verified = True
                        print('Fresh native GPU smoke proof passed; releasing A/B full training', flush=True)
                    except Exception as exc:
                        fail(job_id, 'GPU smoke integration proof failed: ' + repr(exc))
                        break
                # A child may fail while another job's hashes are being checked.
                if any((runtime / "jobs" / key / "failed.json").exists()
                       or (runtime / "jobs" / key / "queue_failed.json").exists()
                       or bad_loss_lines(runtime / "jobs" / key / "console.log") for key in active):
                    stop_pending = True
                    break
                gpus = query_gpus()
                gpu = None
                gpu_lease = None
                lock_root = Path("/tmp/olora_gamfs_gpu_locks")
                lock_root.mkdir(exist_ok=True)
                for index in suite["allowed_gpus"]:
                    if (index in busy or index not in gpus
                            or gpus[index]["memory_used_mb"] >= 500
                            or gpus[index]["utilization_percent"] >= 5):
                        continue
                    candidate = open(lock_root / f"gpu{index}.lock", "a+")
                    try:
                        fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                        continue
                    gpu, gpu_lease = index, candidate
                    break
                if gpu is None:
                    break
                try:
                    current = verify_source(suite)
                    if current["source_manifest_sha256"] != provenance["source_manifest_sha256"]:
                        raise RuntimeError("Source manifest changed during queue execution")
                    verify_config(job)
                    # Check the exact selected physical GPU immediately before spawn.
                    refreshed = query_gpus()[gpu]
                    if (refreshed["uuid"] != gpus[gpu]["uuid"] or refreshed["memory_used_mb"] >= 500
                            or refreshed["utilization_percent"] >= 5):
                        continue
                    run = runtime / "jobs" / job_id
                    run.mkdir(exist_ok=False)
                    env = os.environ.copy()
                    env.update({str(k): str(v) for k, v in suite.get("environment", {}).items()})
                    env["CUDA_VISIBLE_DEVICES"] = gpus[gpu]["uuid"]
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    env["PYTHONUNBUFFERED"] = "1"
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    with open(run / "console.log", "xb") as log:
                        process = subprocess.Popen([suite["python"], "-u", str(HERE / "run_job.py"),
                                                    "--job", job_id, "--gpu", str(gpu), "--uuid", gpus[gpu]["uuid"],
                                                    '--run-pins-sha256', run_pins_hash],
                                                   stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env,
                                                   start_new_session=True, pass_fds=(gpu_lease.fileno(),))
                    identity = proc_identity(process.pid)
                    if not identity:
                        raise RuntimeError("Child process vanished during launch")
                    launch = {**identity, "id": job_id, "gpu": gpu, "gpu_uuid": gpus[gpu]["uuid"],
                              "host": socket.gethostname(), "launched_at": now(), "suite_sha256": suite_hash,
                              "config_sha256": job["config_sha256"], 'run_pins_sha256':run_pins_hash}
                    exclusive_json(run / "launch.json", launch)
                    active[job_id], owned[job_id] = launch, process
                    states[job_id] = {**launch, "status": "running"}
                    busy.add(gpu)
                    print(f"Started {job_id} on physical GPU {gpu}, PID {process.pid}", flush=True)
                except Exception as exc:
                    if not (runtime / "jobs" / job_id).exists():
                        (runtime / "jobs" / job_id).mkdir(exist_ok=False)
                    fail(job_id, repr(exc))
                    break
                finally:
                    # Close, do NOT LOCK_UN: the child retains this same lock
                    # descriptor until its training process exits.
                    gpu_lease.close()
        atomic_json(state_path, {"updated_at": now(), "suite_sha256": suite_hash,
                                "queue": {"host": socket.gethostname(), 'pid_namespace':os.readlink('/proc/self/ns/pid'), **proc_identity(os.getpid())},
                                "stop_pending": stop_pending, "jobs": states})
        collect(suite, states)
        pending = [job_id for job_id in selected if states[job_id]["status"] == "pending"]
        if not active and (stop_pending or not pending):
            break
        if not active and pending and all(any(states[d]["status"] != "completed" for d in jobs[j].get("depends_on", []))
                                          for j in pending):
            print("Selected jobs have unmet dependencies; no training started", flush=True)
            return 1
        time.sleep(args.poll_seconds)
    if not stop_pending and suite.get("final_report_script"):
        report = resolve(suite["final_report_script"])
        completed_report = subprocess.run([suite["python"], str(report)], cwd=HERE, check=False)
        if completed_report.returncode:
            print(f"Post-run analysis failed with exit {completed_report.returncode}; training receipts retained", flush=True)
            return completed_report.returncode
    return 1 if stop_pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
