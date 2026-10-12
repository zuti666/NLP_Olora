"""Standard-library support for the isolated single-200-class campaign."""
from __future__ import annotations
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
METHODS = {"sgd", "sam", "mirror", "g2", "gam"}
SCOPES = {"lora", "full_ft"}

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def resolve(path, base=HERE):
    p = Path(path)
    return (p if p.is_absolute() else base / p).resolve()

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def proc_identity(pid):
    try:
        s = Path(f"/proc/{int(pid)}/stat").read_text()
        f = s[s.rfind(")") + 2:].split()
        if f[0] == "Z":
            return None
        return {"pid": int(pid), "proc_starttime": f[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None

def process_matches(record):
    if not isinstance(record, dict) or not {"pid", "proc_starttime"} <= set(record):
        return False
    current = proc_identity(record["pid"])
    return bool(current and str(current["proc_starttime"]) == str(record["proc_starttime"]))

def assert_host(suite):
    if socket.gethostname() != suite["execution_hostname"]:
        raise RuntimeError("Queue must execute on the pinned physical host")
    if os.readlink("/proc/self/ns/pid") != suite["execution_pid_namespace"]:
        raise RuntimeError("Queue must execute in the pinned host PID namespace")

def query_gpus():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=30)
    out = {}
    for row in csv.reader(io.StringIO(raw)):
        index, uuid, mem, util = [x.strip() for x in row]
        out[int(index)] = dict(index=int(index), uuid=uuid, memory_used_mb=int(mem), utilization_percent=int(util), compute_pids=[])
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=30)
    mapping = {v["uuid"]: v for v in out.values()}
    for row in csv.reader(io.StringIO(raw)):
        if len(row) == 2:
            uuid, pid = [x.strip() for x in row]
            if uuid in mapping:
                mapping[uuid]["compute_pids"].append(int(pid))
    return out

def gpu_idle(gpu):
    return bool(gpu and gpu["memory_used_mb"] < 500 and gpu["utilization_percent"] < 5 and not gpu["compute_pids"])

def verify_source(suite):
    root = resolve(suite["source_root"])
    path = resolve(suite["source_manifest"])
    if sha256(path) != suite["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    files = read_json(path)["files"]
    if len(files) != 190:
        raise RuntimeError("Expected the complete 190-file immutable source snapshot")
    for name, digest in files.items():
        p = resolve(name, root)
        if not p.is_relative_to(root) or sha256(p) != digest:
            raise RuntimeError("Frozen source changed: " + name)
    return {"passed": True, "source_manifest_sha256": sha256(path), "files_checked": len(files)}

def verify_pins(suite, suite_path=HERE / "suite.json"):
    verify_source(suite)
    pins = read_json(HERE / "RUN_PINS.json")
    if not pins["files"]:
        raise RuntimeError("Empty execution pins")
    for name, digest in pins["files"].items():
        path = resolve(name)
        if not path.is_relative_to(HERE) or sha256(path) != digest:
            raise RuntimeError("Frozen execution file changed: " + name)
    for job in suite["jobs"]:
        if job["kind"] == "training" and sha256(resolve(job["config"])) != job["config_sha256"]:
            raise RuntimeError("Config pin changed: " + job["id"])
    # suite.json must itself be covered by immutable execution pins.
    relative = str(resolve(suite_path).relative_to(HERE))
    if pins["files"].get(relative) != sha256(suite_path):
        raise RuntimeError("Suite is not frozen by RUN_PINS")
    verify_reused(suite)

def verify_reused(suite):
    gate = read_json(HERE / "gates/REUSED_RESULTS.json")
    if gate.get("status") != "passed" or not gate.get("checks") or not all(gate["checks"].values()) or not gate.get("files"):
        raise RuntimeError("Historical result reuse acceptance missing")
    for path, digest in gate["files"].items():
        if sha256(path) != digest:
            raise RuntimeError("Historical result provenance changed: " + path)
    references = suite["reused_results"]
    if len(references) != 3 or {r["method"] for r in references} != {"sgd", "sam", "g2"}:
        raise RuntimeError("Expected exactly three matched LoRA references")
    for reference in references:
        if reference["train_scope"] != "lora" or sha256(reference["summary_path"]) != reference["summary_sha256"]:
            raise RuntimeError("Historical result summary changed")
    return {"status": "passed", "files_checked": len(gate["files"]), "references": 3}

def load_suite(path=HERE / "suite.json"):
    s = read_json(path)
    if set(s["allowed_gpus"]) != {4, 5} or len(s["allowed_gpus"]) != 2 or s["max_parallel"] != 2:
        raise ValueError("Only physical GPUs 4 and 5, at most two simultaneous jobs")
    if s["gpu_lock_dir"] != "/tmp/olora_gamfs_gpu_locks":
        raise ValueError("GPU leases must share the existing host-wide lock directory")
    if "CUDA_VISIBLE_DEVICES" in s.get("environment", {}):
        raise ValueError("Queue owns CUDA_VISIBLE_DEVICES")
    ids = [j["id"] for j in s["jobs"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate jobs")
    smokes = {j["id"] for j in s["jobs"] if j["kind"] == "training" and j["smoke"]}
    pairs = {j["id"] for j in s["jobs"] if j["kind"] == "pairing"}
    if len(smokes) != 10 or len(pairs) != 1:
        raise ValueError("Require all ten method/scope smokes and one pairing gate")
    if {(j["train_scope"], j["method"]) for j in s["jobs"] if j["id"] in smokes} != {(a, b) for a in SCOPES for b in METHODS}:
        raise ValueError("Smokes do not cover the five methods in both scopes")
    formal = [j for j in s["jobs"] if j["kind"] == "training" and not j["smoke"]]
    wanted = {("lora", "mirror"), ("lora", "gam")} | {("full_ft", m) for m in METHODS}
    if len(formal) != 7 or {(j["train_scope"], j["method"]) for j in formal} != wanted:
        raise ValueError("Formal campaign must contain exactly the seven authorized conditions")
    outputs = []
    bulk_root = resolve(s.get("bulk_output_root", s.get("outputs_root", HERE / "outputs")))
    if bulk_root != HERE / "outputs" and not bulk_root.is_relative_to(Path("/data/115-2/users/liying/cache/tmp")):
        raise ValueError("Bulk outputs must remain inside the authorized task cache root")
    if bulk_root != HERE / "outputs" and HERE.name not in bulk_root.parts:
        raise ValueError("Bulk output directory must be dedicated to this campaign")
    for j in s["jobs"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", j["id"]):
            raise ValueError("Invalid job ID")
        deps = set(j.get("depends_on", []))
        if deps - set(ids) or j["id"] in deps:
            raise ValueError("Invalid dependencies")
        if j["id"] in pairs and not smokes <= deps:
            raise ValueError("Pairing requires all ten accepted smokes")
        if j in formal and not pairs <= deps:
            raise ValueError("Every formal run must wait for the accepted pairing gate")
        if j["kind"] not in {"training", "pairing"}:
            raise ValueError("Unknown job kind")
        out = resolve(j["output_dir"])
        authorized_root = bulk_root if j["kind"] == "training" else HERE / "gates"
        if not out.is_relative_to(authorized_root) or out == authorized_root or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in outputs):
            raise ValueError("Job outputs must be distinct isolated campaign directories")
        outputs.append(out)
        if not isinstance(j["command"], list) or not j["command"] or j["command"][0] != s["python"]:
            raise ValueError("Commands must use the pinned Python without a shell")
        if j["kind"] == "training":
            cfg = read_json(resolve(j["config"]))
            if sha256(resolve(j["config"])) != j["config_sha256"] or cfg["train_scope"] != j["train_scope"] or cfg["method"] != j["method"]:
                raise ValueError("Config identity differs from suite")
    pending = set(ids)
    while pending:
        ready = {j["id"] for j in s["jobs"] if j["id"] in pending and not set(j.get("depends_on", [])) & pending}
        if not ready:
            raise ValueError("Cyclic dependencies")
        pending -= ready
    return s

def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("Nonfinite metric or evidence")
    if isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite_tree(item)
