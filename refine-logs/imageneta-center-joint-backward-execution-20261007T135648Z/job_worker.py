"""Detached worker keeps both GPU leases until its witness and runner finish."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

from common import HERE, assert_host, atomic_json, now, proc_identity, read_json, verify_config, verify_pins


def verify_environment(suite):
    spec = read_json(suite["environment_spec"])
    software = spec["software"]
    if platform.python_version() != software["python"]:
        raise RuntimeError("Python version differs from frozen environment")
    aliases = {"torch_distribution": "torch", "torchvision_distribution": "torchvision"}
    excluded = {"python", "torch_runtime", "torchvision_runtime", "torch_cuda_compile_version"}
    observed = {}
    for key, expected in software.items():
        if key in excluded:
            continue
        package = aliases.get(key, key)
        observed[package] = importlib.metadata.version(package)
        if observed[package] != expected:
            raise RuntimeError("Environment package version differs: " + package)
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--uuid-lease-fd", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    job = read_json(args.job)
    suite = read_json(job["suite_path"])
    directory = Path(args.job).parent
    if args.cpu != (job["kind"] == "pairing"):
        raise ValueError("CPU worker is reserved for the pairing gate")
    if not args.cpu and (args.lease_fd is None or args.uuid_lease_fd is None):
        raise ValueError("Training worker requires both inherited GPU leases")
    lease_fds = () if args.cpu else (args.lease_fd, args.uuid_lease_fd)
    for fd in lease_fds:
        os.fstat(fd)
    atomic_json(directory / "worker_identity.json", dict(job_id=job["id"], process=proc_identity(os.getpid()),
                started_at=now(), hostname=socket.gethostname(), pid_namespace=os.readlink("/proc/self/ns/pid")))
    try:
        assert_host(suite)
        verify_pins(suite)
        if not args.cpu:
            verify_config(job)
        if not args.cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != job["assigned_gpu_uuid"]:
            raise RuntimeError("Worker CUDA visibility differs from leased physical UUID")
        if args.cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU pairing worker must not see any CUDA device")
        versions = verify_environment(suite)
        atomic_json(directory / "environment_observed.json", dict(status="passed", versions=versions,
                    python=platform.python_version(), executable=sys.executable))
        if not args.cpu:
            subprocess.run([sys.executable, str(HERE / "trainer/cuda_witness.py"),
                            "--output", job["cuda_witness_output"]], pass_fds=lease_fds, check=True)
            witness = read_json(job["cuda_witness_output"])
            software = read_json(suite["environment_spec"])["software"]
            if witness.get("status") != "passed" or witness["cuda_visible_devices"] != job["assigned_gpu_uuid"]:
                raise RuntimeError("CUDA witness did not validate the leased GPU")
            if witness["torch"] != software["torch_runtime"] or witness["cuda"] != software["torch_cuda_compile_version"]:
                raise RuntimeError("CUDA runtime differs from pinned environment")
        child = subprocess.Popen(job["command"], pass_fds=lease_fds, cwd=job["cwd"])
        atomic_json(directory / "runner_identity.json", dict(job_id=job["id"], process=proc_identity(child.pid),
                    command=job["command"], started_at=now()))
        returncode = child.wait()
        atomic_json(directory / "exit.json", dict(job_id=job["id"], returncode=returncode,
                    exited_at=now(), runner_pid=child.pid))
        return returncode
    except BaseException as exc:
        atomic_json(directory / "exit.json", dict(job_id=job["id"], returncode=-1,
                    exited_at=now(), error=repr(exc)))
        raise


if __name__ == "__main__":
    sys.exit(main())
