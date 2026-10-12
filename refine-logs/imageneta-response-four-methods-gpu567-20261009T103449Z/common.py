"""Small, standard-library helpers for the pinned S3 experiment queue."""
from __future__ import annotations

import ast
import csv
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
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ALLOWED_GPUS = {5,6,7}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path, base=HERE):
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve()


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


def exclusive_json(path, value):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def proc_identity(pid):
    try:
        # comm may contain spaces and parentheses; fields after its LAST ')' are stable.
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "proc_starttime": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_matches(record):
    if not isinstance(record, dict) or "pid" not in record or "proc_starttime" not in record:
        return False
    identity = proc_identity(record["pid"])
    return bool(identity and str(identity["proc_starttime"]) == str(record["proc_starttime"]))


def query_gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    result = {}
    for row in csv.reader(io.StringIO(output)):
        index, uuid, memory, utilization = [item.strip() for item in row]
        result[int(index)] = {"index": int(index), "uuid": uuid,
                              "memory_used_mb": int(memory), "utilization_percent": int(utilization)}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=30)
    pids = {}
    for line in csv.reader(io.StringIO(apps)):
        if len(line) == 2:
            uuid, pid = [v.strip() for v in line]
            pids.setdefault(uuid, []).append(int(pid))
    for gpu in result.values():
        gpu["compute_pids"] = pids.get(gpu["uuid"], [])
    return result


def assert_host(suite):
    if socket.gethostname() != suite["execution_hostname"]:
        raise RuntimeError(f"Host mismatch: {socket.gethostname()} != {suite['execution_hostname']}")
    if os.readlink('/proc/self/ns/pid') != suite['execution_pid_namespace']:
        raise RuntimeError('Execute queue in the pinned host PID namespace')


def load_suite(path=HERE / "suite.json"):
    suite = read_json(path)
    if not suite.get("jobs"):
        raise ValueError("suite.jobs must be nonempty")
    gpu_list = suite["allowed_gpus"]
    if not gpu_list or len(set(gpu_list)) != len(gpu_list) or not set(gpu_list) <= ALLOWED_GPUS:
        raise ValueError("Only physical GPUs 5,6,7 for this user-restricted campaign")
    if not 1 <= int(suite["max_parallel"]) <= len(gpu_list):
        raise ValueError("max_parallel must be between 1 and number of authorized GPUs")
    identifiers = [job["id"] for job in suite["jobs"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate job IDs")
    for job in suite["jobs"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["id"]):
            raise ValueError(f"Invalid job ID: {job['id']}")
        if set(job.get("depends_on", [])) - set(identifiers):
            raise ValueError(f"Unknown dependency for {job['id']}")
        if job["id"] in job.get("depends_on", []):
            raise ValueError("Self-dependency")
        if int(job.get("expected_tasks", suite["expected_tasks"])) < 1:
            raise ValueError("expected_tasks must be positive")
    pending = set(identifiers)
    while pending:
        ready = {j["id"] for j in suite["jobs"] if j["id"] in pending
                 and not set(j.get("depends_on", [])) & pending}
        if not ready:
            raise ValueError("Dependency cycle")
        pending -= ready
    if "CUDA_VISIBLE_DEVICES" in suite.get("environment", {}):
        raise ValueError("CUDA_VISIBLE_DEVICES is set exclusively by the queue")
    return suite


def verify_source(suite):
    root = resolve(suite["source_root"])
    manifest_path = resolve(suite["source_manifest"])
    manifest = read_json(manifest_path)
    files = manifest["files"]
    if not files:
        raise ValueError("Empty source manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    return {"source_root": str(root), "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path), "source_files": files}


def verify_config(job):
    path = resolve(job["config"])
    actual = sha256(path)
    if actual != job["config_sha256"]:
        raise RuntimeError(f"Config hash mismatch: {job['id']}")
    config = read_json(path)
    if not Path(config["outputs_root"]).is_absolute():
        raise ValueError("outputs_root must be absolute so the native trainer uses the verified location")
    output = resolve(config["outputs_root"])
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError(f"outputs_root must be a dedicated directory below campaign HERE: {output}")
    return path, config, output


def validate(suite):
    assert_host(suite)
    source = verify_source(suite)
    outputs = []
    for job in suite["jobs"]:
        _, _, output = verify_config(job)
        if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in outputs):
            raise ValueError("Job outputs_root directories overlap")
        outputs.append(output)
    if not resolve(suite["python"]).is_file():
        raise FileNotFoundError(suite["python"])
    return source


NONFINITE_LOSS = re.compile(
    r"\bloss(?:[ _-]*(?:train|ce|ewc|flat|total))?\s*(?:[:=]\s*)?"
    r"(?:tensor\(\s*)?[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z])", re.I)


def bad_loss_lines(log_path):
    if not Path(log_path).is_file():
        return []
    bad = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for fragment in line.split("\r"):
                if NONFINITE_LOSS.search(fragment):
                    bad.append(fragment.strip()[-1000:])
                    if len(bad) >= 10:
                        return bad
    return bad


def read_curve(log_path):
    curve = None
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "CNN top1 curve:" not in line:
                continue
            payload = line.split("CNN top1 curve:", 1)[1].strip()
            payload = re.sub(r"(?:np|numpy)\.float(?:16|32|64)\(([^()]*)\)", r"\1", payload)
            try:
                candidate = ast.literal_eval(payload)
                if isinstance(candidate, list):
                    curve = [float(value) for value in candidate]
            except (ValueError, SyntaxError, TypeError):
                continue
    return curve


def inspect_metrics(suite, job, *, require_complete=True):
    _, _, output = verify_config(job)
    paths = sorted(output.rglob("*_cl_metrics.json")) if output.exists() else []
    if len(paths) != 1:
        raise ValueError(f"Expected one metrics JSON, found {len(paths)}")
    data = read_json(paths[0])
    cnn = data["cnn"]
    matrices = cnn["matrices"]
    matrix = matrices.get("final")
    if matrix is None:
        if require_complete:
            raise ValueError("Missing final CNN matrix")
        steps = sorted(k for k in matrices if re.fullmatch(r"t\d+", k))
        matrix = matrices[steps[-1]] if steps else []
    expected = int(job.get("expected_tasks", suite["expected_tasks"]))
    n = len(matrix)
    if not n or (require_complete and n != expected) or n > expected:
        raise ValueError(f"Wrong task count {n}, expected {expected}")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError("CNN matrix must be square")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in row[:i+1]):
            raise ValueError(f"Nonfinite lower triangle in CNN matrix row {i}")
    # Native FWT/AFG and the first BWT entry are structurally undefined. We
    # validate observed accuracies and independently recompute reported metrics;
    # undefined native diagnostic fields are preserved as JSON null below.
    final = cnn.get("final", {})
    log = HERE / "runtime" / "jobs" / job["id"] / "console.log"
    bad = bad_loss_lines(log)
    if bad:
        raise ValueError("Nonfinite training loss: " + bad[0])
    curve = read_curve(log)
    if curve is None or (require_complete and len(curve) != expected):
        raise ValueError("Missing/incomplete sample-weighted CNN top1 curve")
    if any(not math.isfinite(x) for x in curve):
        raise ValueError("Nonfinite CNN top1 curve")
    faa = sum(matrix[-1]) / n
    aaa = sum(sum(row[:i+1]) / (i+1) for i, row in enumerate(matrix)) / n
    forgetting = (sum(max(matrix[i][j] for i in range(j, n)) - matrix[-1][j]
                      for j in range(n-1)) / (n-1)) if n > 1 else None
    bwt = (sum(matrix[-1][j] - matrix[j][j] for j in range(n-1)) / (n-1)) if n > 1 else None
    return clean_json({"metrics_path": str(paths[0]), "metrics_sha256": sha256(paths[0]),
                       "tasks_completed": n, "expected_tasks": expected,
                       "faa_task_macro": faa, "aaa_task_macro": aaa,
                       "forgetting": forgetting, "bwt": bwt,
                       "faa_sample_weighted": curve[-1],
                       "aaa_sample_weighted": sum(curve) / len(curve),
                       "cnn_top1_curve": curve, "cnn_matrix": matrix, "native_final": final})
