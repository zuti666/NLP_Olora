#!/usr/bin/env python3
"""CPU-only, fail-closed predecessor/GPU gate for this experiment campaign."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def process_identity(pid):
    """Linux process identity; start time and boot ID prevent PID reuse."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] in ("Z", "X"):
        return None
    return {"pid": int(pid), "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def identity_alive(identity):
    expected = {"pid": int(identity["pid"]), "start_ticks": int(identity["start_ticks"]),
                "boot_id": str(identity["boot_id"])}
    return process_identity(expected["pid"]) == expected


def snapshot_files(plan_path, plan):
    paths = {str(Path(plan_path).resolve()), str(Path(__file__).resolve())}
    paths.update(str(Path(p).resolve()) for p in plan["code_files"])
    hashes = {}
    for name in sorted(paths):
        digest = hashlib.sha256()
        with open(name, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def lock_directory(plan):
    return Path(plan.get("gpu_lock_dir", Path(__file__).resolve().parents[3] / ".cache/gpu_locks"))


def gpu_lock_path(plan, uuid):
    if not uuid.startswith("GPU-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in uuid):
        raise ValueError(f"Invalid GPU UUID: {uuid!r}")
    return lock_directory(plan) / f"{uuid}.lock"


def gpu_locks_available(plan):
    """Probe existing locks without creating files, including for --check-once."""
    blocked = []
    for gpu in plan["start_after"]["gpus"]:
        uuid = plan["start_after"]["gpu_uuids"][str(gpu)]
        path = gpu_lock_path(plan, uuid)
        try:
            # NFS requires a writable FD for LOCK_EX; r+ never creates/writes it.
            stream = path.open("r+")
        except FileNotFoundError:
            continue
        with stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                blocked.append(uuid)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return blocked


def nvidia_snapshot():
    def query(arguments):
        completed = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                                   check=True, capture_output=True, text=True, timeout=20)
        return list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))

    cards = {}
    for row in query(["--query-gpu=index,uuid,memory.used,utilization.gpu"]):
        if len(row) != 4:
            raise ValueError(f"Malformed nvidia-smi GPU row: {row!r}")
        index, uuid, memory, utilization = (v.strip() for v in row)
        memory, utilization = float(memory), float(utilization)
        if not math.isfinite(memory) or not math.isfinite(utilization):
            raise ValueError("Non-finite GPU telemetry")
        cards[uuid] = {"index": int(index), "memory_mib": memory, "utilization_pct": utilization,
                       "compute_pids": []}
    for row in query(["--query-compute-apps=gpu_uuid,pid"]):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError(f"Malformed nvidia-smi process row: {row!r}")
        uuid, pid = (v.strip() for v in row)
        if uuid not in cards:
            raise ValueError(f"Compute process GPU absent from inventory: {uuid}")
        cards[uuid]["compute_pids"].append(int(pid))
    return cards


def validate_plan(plan):
    gate = plan["start_after"]
    if not Path(plan["run_dir"]).is_absolute() or not Path(gate["state_file"]).is_absolute():
        raise ValueError("run_dir and predecessor state_file must be absolute")
    if not isinstance(plan["pipeline_command"], list) or not plan["pipeline_command"] or not all(isinstance(x, str) for x in plan["pipeline_command"]):
        raise ValueError("pipeline_command must be a nonempty argument list")
    if not isinstance(plan["code_files"], list) or not plan["code_files"]:
        raise ValueError("code_files must pin the executable sources")
    if set(gate["gpus"]) != {0, 1, 2, 3}:
        raise ValueError("This queue must wait for all physical GPUs 0-3")
    if not plan["gpus"] or not set(plan["gpus"]).issubset(set(gate["gpus"])):
        raise ValueError("Campaign GPUs must be a subset of the waited GPUs")
    uuids = [gate["gpu_uuids"][str(gpu)] for gpu in gate["gpus"]]
    if len(set(uuids)) != len(uuids):
        raise ValueError("GPU UUIDs must be unique")
    for uuid in uuids:
        gpu_lock_path(plan, uuid)
    if gate["required_status"] != "completed" or gate["required_receipt_status"] != "complete":
        raise ValueError("Predecessor must require completed state and complete receipt")
    if float(gate["idle_seconds"]) < 60 or not 0 < float(gate["poll_seconds"]) <= 60:
        raise ValueError("Require >=60 idle seconds and a polling interval in (0, 60]")
    identity = gate["supervisor_identity"]
    if int(identity["pid"]) <= 0 or int(identity["start_ticks"]) <= 0 or not identity["boot_id"]:
        raise ValueError("A complete predecessor supervisor identity is required")


def inspect_gate(plan, check_locks=True):
    """Collect all gate failures. Read/telemetry errors always keep the gate shut."""
    gate = plan["start_after"]
    result = {"checked_at": utc_now(), "ready": False, "reasons": [], "errors": []}
    try:
        predecessor = read_json(gate["state_file"])
        status = predecessor.get("status")
        result["predecessor_status"] = status
        if status != gate["required_status"]:
            result["reasons"].append(f"predecessor status is {status!r}; require {gate['required_status']!r}")
    except Exception as exc:
        result["errors"].append(f"predecessor state: {type(exc).__name__}: {exc}")
    try:
        receipt_path = Path(gate["final_receipt"])
        if not receipt_path.is_absolute():
            receipt_path = Path(gate["state_file"]).parent / receipt_path
        receipt = read_json(receipt_path)
        result["predecessor_receipt_status"] = receipt.get("status")
        if receipt.get("status") != gate["required_receipt_status"]:
            result["reasons"].append("predecessor final receipt is not complete")
    except Exception as exc:
        result["errors"].append(f"predecessor receipt: {type(exc).__name__}: {exc}")
    try:
        result["predecessor_supervisor_alive"] = identity_alive(gate["supervisor_identity"])
        if result["predecessor_supervisor_alive"]:
            result["reasons"].append("predecessor supervisor is still alive")
    except Exception as exc:
        result["errors"].append(f"supervisor identity: {type(exc).__name__}: {exc}")
    try:
        cards = nvidia_snapshot()
        result["gpus"] = {}
        for index in gate["gpus"]:
            uuid = gate["gpu_uuids"][str(index)]
            card = cards.get(uuid)
            result["gpus"][str(index)] = {"uuid": uuid, "telemetry": card}
            if card is None or card["index"] != index:
                result["errors"].append(f"GPU {index} UUID/index mapping does not match {uuid}")
            elif card["memory_mib"] >= 500 or card["utilization_pct"] >= 5 or card["compute_pids"]:
                result["reasons"].append(f"GPU {index} is busy")
    except Exception as exc:
        result["errors"].append(f"GPU telemetry: {type(exc).__name__}: {exc}")
    if check_locks:
        try:
            result["blocked_gpu_locks"] = gpu_locks_available(plan)
            if result["blocked_gpu_locks"]:
                result["reasons"].append("shared GPU locks are held: " + ", ".join(result["blocked_gpu_locks"]))
        except Exception as exc:
            result["errors"].append(f"GPU locks: {type(exc).__name__}: {exc}")
    result["ready"] = not result["reasons"] and not result["errors"]
    return result


def atomic_write(path, text):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_state(run_dir, state):
    state["updated_at"] = utc_now()
    atomic_write(run_dir / "queue_state.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Campaign queue", "", f"Status: **{state['status']}**", "",
             f"Updated: {state['updated_at']}", "", f"Plan: `{state['plan_path']}`", "",
             f"Queue PID: {state['queue_identity']['pid']}", ""]
    if state.get("child_identity"):
        lines += [f"Pipeline PID: {state['child_identity']['pid']}", ""]
    if state.get("message"):
        lines += [state["message"], ""]
    gate = state.get("gate", {})
    lines += [f"Continuous idle seconds observed: {state.get('idle_elapsed_seconds', 0):.1f}", ""]
    lines += ["- " + message for message in gate.get("reasons", []) + gate.get("errors", [])]
    if "returncode" in state:
        lines += ["", f"Pipeline return code: {state['returncode']}"]
    atomic_write(run_dir / "QUEUE_STATUS.md", "\n".join(lines) + "\n")


def acquire_gpu_locks(plan):
    """Acquire only the campaign GPUs; close every partial acquisition on failure."""
    streams = {}
    lock_directory(plan).mkdir(parents=True, exist_ok=True)
    try:
        for gpu in sorted(plan["gpus"]):
            uuid = plan["start_after"]["gpu_uuids"][str(gpu)]
            stream = gpu_lock_path(plan, uuid).open("a+")
            streams[uuid] = stream
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return streams
    except BaseException:
        for stream in streams.values():
            stream.close()
        raise


def wait_until_ready(plan_path, plan, pinned, state, run_dir):
    idle_since = None
    while True:
        if snapshot_files(plan_path, plan) != pinned:
            raise RuntimeError("Plan or pinned code changed while queued; refusing to start")
        gate = inspect_gate(plan)
        now = time.monotonic()
        if not gate["ready"]:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        elapsed = now - idle_since if idle_since is not None else 0.0
        state.update(status="blocked" if gate.get("predecessor_status") in ("failed", "error", "aborted") else "waiting",
                     gate=gate, idle_elapsed_seconds=elapsed)
        save_state(run_dir, state)
        if gate["ready"] and elapsed >= float(plan["start_after"]["idle_seconds"]):
            try:
                streams = acquire_gpu_locks(plan)
            except BlockingIOError:
                idle_since = None
            else:
                # Recheck after lock acquisition to close the acquisition race.
                verified = inspect_gate(plan, check_locks=False)
                if verified["ready"] and snapshot_files(plan_path, plan) == pinned:
                    state["gate"] = verified
                    return streams
                for stream in streams.values():
                    stream.close()
                idle_since = None
        time.sleep(float(plan["start_after"]["poll_seconds"]))


def execute(plan_path, plan):
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_lock = (run_dir / "queue.lock").open("a+")
    gpu_locks = {}
    state = None
    try:
        try:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This queue or its pipeline already holds queue.lock") from None
        previous = read_json(run_dir / "queue_state.json") if (run_dir / "queue_state.json").exists() else {}
        if previous.get("child_identity") and identity_alive(previous["child_identity"]):
            raise RuntimeError("Previously launched pipeline is still alive; refusing duplicate launch")
        if previous.get("pipeline_started_at"):
            raise RuntimeError("This queue has already launched its pipeline; inspect its state before explicitly preparing a new run directory")
        pinned = snapshot_files(plan_path, plan)
        state = {"status": "waiting", "plan_path": str(plan_path), "queued_at": previous.get("queued_at", utc_now()),
                 "queue_identity": process_identity(os.getpid()), "file_sha256": previous.get("file_sha256", pinned)}
        if state["file_sha256"] != pinned:
            raise RuntimeError("Plan or code changed since the earlier queue instance; refusing silent replacement")
        save_state(run_dir, state)
        gpu_locks = wait_until_ready(plan_path, plan, pinned, state, run_dir)
        environment = os.environ.copy()
        environment["RQ2_GPU_LOCK_FDS"] = json.dumps({uuid: stream.fileno() for uuid, stream in gpu_locks.items()})
        environment["RQ2_QUEUE_LOCK_FD"] = str(queue_lock.fileno())
        state.update(status="starting", pipeline_started_at=utc_now(), pipeline_log=str(run_dir / "pipeline.log"))
        save_state(run_dir, state)
        with (run_dir / "pipeline.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(plan["pipeline_command"], cwd=plan.get("cwd", str(Path(__file__).resolve().parents[3])),
                                     env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=False,
                                     pass_fds=tuple([queue_lock.fileno()] + [s.fileno() for s in gpu_locks.values()]))
            state.update(status="running", child_pid=child.pid, child_identity=process_identity(child.pid))
            save_state(run_dir, state)
            returncode = child.wait()
        state.update(status="completed" if returncode == 0 else "failed", returncode=returncode,
                     pipeline_finished_at=utc_now())
        save_state(run_dir, state)
        return returncode
    except BaseException as exc:
        if state is not None:
            state.update(status="failed", message=f"{type(exc).__name__}: {exc}")
            save_state(run_dir, state)
        raise
    finally:
        # Do not LOCK_UN: inherited descriptors must retain locks if child survives us.
        for stream in gpu_locks.values():
            stream.close()
        queue_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-once", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    try:
        plan = read_json(plan_path)
        validate_plan(plan)
        if args.check_once:
            print(json.dumps(inspect_gate(plan), indent=2, ensure_ascii=False))
            return 0
        return execute(plan_path, plan)
    except Exception as exc:
        print(f"queue error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
