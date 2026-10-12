"""Standard-library helpers for the isolated second-probe sign queue."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ALLOWED_GPUS = {2, 3, 4, 5, 6}
PROTOCOL = "four_probe_second_probe_sign_deterministic_exact_restore_rng_replay_v2"
VARIANTS = {"B0", "B1", "N1", "N2", "N3", "N4", "N5", "N6"}
CPU_CHECKS = {"old_b1_same_state_equivalence", "all_eight_formulas", "same_g012",
              "exact_state_rng_restoration", "four_ce_per_step",
              "selected_side_reused_opposite_only", "smoke_snapshot_hashes_exact",
              "training_compute_and_sgd_source_unchanged", "deterministic_execution_config"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def snapshot_rng_fingerprint(state):
    """Hash every saved RNG stream by content, independent of pickle storage IDs."""
    def tensor_info(value):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
    numpy_state = state[1]
    streams = dict(python=state[0],
                   numpy=dict(generator=numpy_state[0], dtype=str(numpy_state[1].dtype),
                              shape=list(numpy_state[1].shape),
                              sha256=hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
                              position=int(numpy_state[2]), has_gauss=int(numpy_state[3]), cached_gauss=float(numpy_state[4])),
                   torch_cpu=tensor_info(state[2]), torch_cuda=[tensor_info(value) for value in state[3]])
    return hashlib.sha256(json.dumps(streams, sort_keys=True).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n")


def proc_identity(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "proc_starttime": fields[19],
                "hostname": socket.gethostname(),
                "pid_namespace": os.readlink("/proc/self/ns/pid")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_matches(record):
    if not isinstance(record, dict) or not {"pid", "proc_starttime"} <= set(record):
        return False
    if record.get("hostname", socket.gethostname()) != socket.gethostname():
        return False
    if record.get("pid_namespace", os.readlink("/proc/self/ns/pid")) != os.readlink("/proc/self/ns/pid"):
        return False
    current = proc_identity(record["pid"])
    return bool(current and current["proc_starttime"] == str(record["proc_starttime"]))


def query_gpus():
    """Fail closed on unavailable telemetry; require no compute PID, not just low use."""
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    result = {}
    for row in csv.reader(io.StringIO(raw)):
        index, uuid, memory, utilization = [value.strip() for value in row]
        result[int(index)] = dict(index=int(index), uuid=uuid,
                                  memory_used_mb=int(memory), utilization_percent=int(utilization),
                                  compute_pids=[])
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    by_uuid = {record["uuid"]: record for record in result.values()}
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        if len(row) != 2:
            raise RuntimeError("Unexpected compute-process telemetry: " + repr(row))
        uuid, pid = [value.strip() for value in row]
        if uuid not in by_uuid:
            raise RuntimeError("Compute process belongs to an unobserved GPU: " + uuid)
        by_uuid[uuid]["compute_pids"].append(int(pid))
    if not result:
        raise RuntimeError("No visible GPU telemetry")
    return result


def gpu_is_idle(record):
    return bool(record and record["memory_used_mb"] < 500
                and record["utilization_percent"] < 5 and not record["compute_pids"])


def assert_host(suite):
    if socket.gethostname() != suite["execution_hostname"]:
        raise RuntimeError("Execute the queue on the pinned host")
    if os.readlink("/proc/self/ns/pid") != suite["execution_pid_namespace"]:
        raise RuntimeError("Execute the queue in the pinned host PID namespace")


def suite_counts(suite):
    jobs = suite["jobs"]
    return dict(jobs=len(jobs), smoke=sum(job["kind"] == "training" and job["smoke"] for job in jobs),
                formal=sum(job["kind"] == "training" and not job["smoke"] for job in jobs),
                pairing=sum(job["kind"] == "pairing" for job in jobs), reused=len(suite.get("reused_results", [])))


def load_suite(path=HERE / "suite.json"):
    suite = read_json(path)
    if set(suite["allowed_gpus"]) != ALLOWED_GPUS:
        raise ValueError("Only physical GPUs 2–6 are authorized; GPU0/1 excluded")
    if not 1 <= int(suite["max_parallel"]) <= len(ALLOWED_GPUS):
        raise ValueError("Invalid queue concurrency")
    if int(suite["poll_seconds"]) != 60 or suite.get("failure_policy") != "stop_on_failure":
        raise ValueError("Expected 60-second polling and stop-on-failure")
    if "CUDA_VISIBLE_DEVICES" in suite["environment"]:
        raise ValueError("CUDA visibility is owned exclusively by the queue")
    if suite.get("protocol") != PROTOCOL or suite["environment"].get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Wrong deterministic protocol/cuBLAS workspace")
    if suite.get("reused_results") != []:
        raise ValueError("Deterministic main results require all baselines retrained")
    diagnostic_accounting = dict(per_task_new_gradient_calls=1, per_task_new_forward_calls=1,
                                 selected_side_source="training_g3", selected_side_new_gradient_calls=0,
                                 opposite_side_source="isolated_opposite_probe", opposite_side_new_gradient_calls=1)
    if suite.get("diagnostic_protocol") != "selected_training_g3_reuse_opposite_only_v2" or suite.get("diagnostic_gradients_per_task") != 1 or suite.get("diagnostic_accounting") != diagnostic_accounting:
        raise ValueError("Recovery suite must account for selected training reuse/opposite-only diagnostics")
    jobs = suite["jobs"]
    identifiers = {job["id"] for job in jobs}
    if suite_counts(suite) != dict(jobs=33, smoke=16, formal=16, pairing=1, reused=0) or len(identifiers) != 33:
        raise ValueError("Expected 16 smoke, one CPU pairing, and 16 formal unique jobs; no reused endpoints")
    smokes = {job["id"] for job in jobs if job["smoke"]}
    if len(smokes) != 16:
        raise ValueError("Expected 16 smoke jobs")
    pairing = {job["id"] for job in jobs if job["kind"] == "pairing"}
    if len(pairing) != 1:
        raise ValueError("Exactly one CPU smoke-pairing gate is required")
    seen_outputs = []
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["id"]):
            raise ValueError("Invalid job identifier")
        if job["kind"] not in {"training", "pairing"} or set(job["depends_on"]) - identifiers:
            raise ValueError("Unknown job kind or dependency")
        if job["smoke"] and job["depends_on"]:
            raise ValueError("Smokes have no training dependency")
        if job["kind"] == "pairing" and set(job["depends_on"]) != smokes:
            raise ValueError("The CPU pairing gate must wait for all 16 accepted smokes")
        if job["kind"] == "training" and not job["smoke"] and set(job["depends_on"]) != pairing:
            raise ValueError("Every formal job must wait for the accepted CPU pairing gate")
        if job["kind"] == "training" and (job.get("expected_diagnostic_gradient_calls") != job["expected_tasks"] or job.get("selected_side_reused_from_training") is not True):
            raise ValueError("Wrong isolated diagnostic budget/provenance")
        out = Path(job["output_dir"]).resolve()
        owned_root = HERE / ("gates" if job["kind"] == "pairing" else "outputs")
        if not out.is_relative_to(owned_root):
            raise ValueError("Output must belong to this isolated campaign")
        if any(out == old or out.is_relative_to(old) or old.is_relative_to(out) for old in seen_outputs):
            raise ValueError("Overlapping outputs")
        seen_outputs.append(out)
    grid = {(variant, radius) for variant in VARIANTS for radius in (.25, 1.)}
    for smoke in (True, False):
        actual = [(job["variant"], job["radius"]) for job in jobs if job["kind"] == "training" and job["smoke"] is smoke]
        if set(actual) != grid or len(actual) != len(grid):
            raise ValueError("Every variant/radius needs an independent smoke and formal job")
    return suite


def verify_source(suite):
    manifest = Path(suite["source_manifest"])
    if sha256(manifest) != suite["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    root = Path(suite["source_root"]).resolve()
    files = read_json(manifest)["files"]
    if not files:
        raise RuntimeError("Empty source manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise RuntimeError("Frozen source changed: " + relative)
    return dict(source_root=str(root), source_manifest=str(manifest),
                source_manifest_sha256=sha256(manifest), source_files=files)


def verify_cpu_gate(suite):
    record = suite["cpu_gate"]
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("CPU equivalence gate changed")
    gate = read_json(path)
    if gate.get("status") != "passed" or not CPU_CHECKS <= set(gate.get("checks", {})):
        raise RuntimeError("Required CPU/equivalence gate is incomplete")
    if not all(value is True for value in gate["checks"].values()):
        raise RuntimeError("CPU/equivalence gate failed")
    if gate.get("entry_sha256") != sha256(HERE / "trainer/four_probe_train.py") or gate.get("executor_sha256") != sha256(HERE / "trainer/direction_step.py"):
        raise RuntimeError("CPU gate belongs to stale trainer code")
    return gate


def verify_config(job):
    path = Path(job["config"])
    if sha256(path) != job["config_sha256"]:
        raise RuntimeError("Frozen config changed: " + job["id"])
    config = read_json(path)
    if not Path(config["outputs_root"]).resolve().is_relative_to(Path(job["output_dir"]).resolve()):
        raise ValueError("Native output does not belong to the verified job output")
    if config.get("registered") is not True or config.get("plan_only") is not False:
        raise ValueError("The controller refuses an unregistered plan-only config")
    if config.get("protocol") != PROTOCOL or config.get("cuda_deterministic_algorithms") is not True or config.get("cuda_deterministic_warn_only") is not False:
        raise ValueError("Raw deterministic execution guard is disabled or missing")
    expected = (job["update_base"], job["second_probe_sign"], job["second_difference_sign"])
    actual = (config["update_base"], config["second_probe_sign"], config["second_difference_sign"])
    if actual != expected or config.get("first_probe_sign") != 1:
        raise ValueError("Scientific factor mismatch")
    return config


def verify_pins(suite):
    pins = read_json(HERE / "RUN_PINS.json")
    for name, expected in pins["files"].items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or sha256(path) != expected:
            raise RuntimeError("Frozen execution artifact changed: " + name)
    env = Path(suite["environment_spec"])
    if sha256(env) != suite["environment_spec_sha256"]:
        raise RuntimeError("Frozen environment spec changed")
    if pins.get("suite_sha256", sha256(HERE / "suite.json")) != sha256(HERE / "suite.json"):
        raise RuntimeError("Suite pin changed")
    verify_cpu_gate(suite)
    verify_source(suite)
    for job in suite["jobs"]:
        if job["kind"] == "training":
            verify_config(job)


NONFINITE_LOSS = re.compile(r"\bloss(?:[ _-]*(?:train|ce|ewc|flat|total))?\s*(?:[:=]\s*)?(?:tensor\(\s*)?[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z])", re.I)


def bad_loss_lines(path):
    if not Path(path).exists():
        return []
    found = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for part in line.split("\r"):
                if NONFINITE_LOSS.search(part):
                    found.append(part.strip()[-1000:])
                    if len(found) >= 10:
                        return found
    return found
