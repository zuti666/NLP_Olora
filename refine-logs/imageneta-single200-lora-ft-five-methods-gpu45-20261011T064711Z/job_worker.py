#!/usr/bin/env python3
"""Detached worker retains physical GPU leases until the trainer exits."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import atomic_json, now, proc_identity, read_json, sha256, verify_pins

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--lease-fd", type=int)
    p.add_argument("--uuid-lease-fd", type=int)
    a = p.parse_args()
    job_path = Path(a.job)
    d = job_path.parent
    job = read_json(job_path)
    suite = read_json(job["suite_path"])
    if sha256(job["suite_path"]) != job["suite_sha256"]:
        raise RuntimeError("Suite changed after scheduling")
    if job["kind"] == "training":
        if a.lease_fd is None or a.uuid_lease_fd is None:
            raise RuntimeError("Training requires both inherited GPU leases")
        if int(job["gpu"]) not in {4, 5} or os.environ.get("CUDA_VISIBLE_DEVICES") != job["gpu_uuid"]:
            raise RuntimeError("Worker GPU assignment differs from authorized physical device")
        fds = (a.lease_fd, a.uuid_lease_fd)
    else:
        fds = ()
    identity = proc_identity(os.getpid())
    atomic_json(d / "worker_identity.json", {"job_id": job["id"], "process": identity, "started_at": now(), "gpu": job.get("gpu"), "gpu_uuid": job.get("gpu_uuid")})
    child = None
    rc = -1
    try:
        verify_pins(suite, job["suite_path"])
        child = subprocess.Popen(job["command"], cwd=job["cwd"], pass_fds=fds)
        runner = proc_identity(child.pid)
        atomic_json(d / "runner_identity.json", {"job_id": job["id"], "process": runner, "started_at": now()})
        rc = child.wait()
        verify_pins(suite, job["suite_path"])
        atomic_json(d / "exit.json", {"job_id": job["id"], "returncode": rc, "exited_at": now(), "worker": identity, "runner": runner, "source_and_execution_pins_unchanged": True})
    except BaseException as exc:
        # No preemption: if a post-launch check raises, let the active trainer finish.
        if child is not None and child.poll() is None:
            child.wait()
        atomic_json(d / "exit.json", {"job_id": job["id"], "returncode": -1, "exited_at": now(), "error": repr(exc), "worker": identity})
        raise
    return rc

if __name__ == "__main__":
    sys.exit(main())
