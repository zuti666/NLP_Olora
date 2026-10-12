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
PROTOCOL = "midpoint_interaction_strict_exact_restore_rng_replay_v1"
VARIANTS = {"sum12", "base02", "two_midpoint", "sum13"}

def expected_budget(config):
    direction = config.get("update_direction", config.get("direction"))
    if direction not in VARIANTS:
        raise ValueError("Unknown midpoint experiment direction")
    g3 = int(direction == "sum13")
    return dict(train_ce_per_update=3+g3, diagnostic_ce_per_update=0,
                first_plus_ce_per_update=1, first_minus_ce_per_update=0,
                g3_plus_ce_per_update=g3, g3_minus_ce_per_update=0,
                local_plus_ce_per_update=1+g3, local_minus_ce_per_update=0,
                g0_ce_per_update=1, g1_ce_per_update=1,
                g2_ce_per_update=int(direction != "two_midpoint"),
                midpoint_ce_per_update=int(direction == "two_midpoint"),
                gm_ce_per_update=int(direction == "two_midpoint"),
                g3_ce_per_update=g3)


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
    jobs=suite["jobs"]
    return dict(jobs=len(jobs), smoke=sum(j["kind"]=="training" and j["smoke"] for j in jobs),
                formal=sum(j["kind"]=="training" and not j["smoke"] for j in jobs),
                pairing=sum(j["kind"]=="pairing" for j in jobs),
                diagnostic=sum(j["kind"]=="diagnostic" for j in jobs), model_gate=sum(j["kind"]=="model_gate" for j in jobs), reused=0)


def upstream_ready(suite):
    return dict(ready=True, reason="No pending upstream campaign; existing results are frozen references only")


def load_suite(path=HERE / "suite.json"):
    suite=read_json(path)
    if set(suite["allowed_gpus"]) != ALLOWED_GPUS or not 1<=suite["max_parallel"]<=5:
        raise ValueError("Only GPUs 2–6 are authorized; 0/1 excluded")
    if suite["poll_seconds"]!=60 or suite["failure_policy"]!="stop_on_failure":raise ValueError("Bad scheduler policy")
    if suite["protocol"]!=PROTOCOL or suite["environment"].get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise ValueError("Wrong strict execution protocol")
    if "CUDA_VISIBLE_DEVICES" in suite["environment"]:raise ValueError("CUDA visibility belongs to GPU leases")
    if suite.get("reused_results"):raise ValueError("No historical endpoints counted as this new four-direction grid")
    jobs=suite["jobs"];ids={j["id"] for j in jobs}
    if len(ids)!=len(jobs) or suite_counts(suite)!=dict(jobs=12,smoke=4,formal=4,pairing=1,diagnostic=2,model_gate=1,reused=0):raise ValueError("Expected four smoke + pairing + diagnostic smoke/full + real model gate + four formal")
    smokes={j["id"] for j in jobs if j["kind"]=="training" and j["smoke"]}
    pair={j["id"] for j in jobs if j["kind"]=="pairing"}
    diag_smoke={j["id"] for j in jobs if j["kind"]=="diagnostic" and j["smoke"]}
    diag_full={j["id"] for j in jobs if j["kind"]=="diagnostic" and not j["smoke"]}
    model_gate={j["id"] for j in jobs if j["kind"]=="model_gate"}
    outputs=[]
    for j in jobs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",j["id"]):raise ValueError("Bad job ID")
        if j["kind"] not in {"training","pairing","diagnostic","model_gate"} or set(j["depends_on"])-ids:raise ValueError("Unknown dependency/kind")
        expected=smokes if j["kind"]=="pairing" else (set() if j["kind"]=="training" and j["smoke"] else (pair if j["kind"]=="diagnostic" and j["smoke"] else (diag_smoke if j["kind"]=="model_gate" else (model_gate if j["kind"]=="diagnostic" else diag_full))))
        if set(j["depends_on"])!=expected:raise ValueError("Dependency barrier mismatch: "+j["id"])
        output=Path(j["output_dir"]).resolve();owned=HERE/("gates" if j["kind"]=="pairing" else "outputs")
        if not output.is_relative_to(owned) or any(output==o or output.is_relative_to(o) or o.is_relative_to(output) for o in outputs):raise ValueError("Overlapping/unowned output")
        outputs.append(output)
        if j["kind"]=="training":
            if j["direction"] not in VARIANTS or (j["radius"],j["rho"],j["seed"],j["rank"])!=(1.,.05,2,10):raise ValueError("Scientific condition mismatch")
            if j["expected_training_ce"]!=j["expected_steps"]*expected_budget(j)["train_ce_per_update"]:raise ValueError("Wrong CE budget")
    for smoke in (True,False):
        if {j["direction"] for j in jobs if j["kind"]=="training" and j["smoke"]==smoke}!=VARIANTS:raise ValueError("Incomplete direction grid")
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
    reference=suite["cpu_gate"];path=Path(reference["path"])
    if sha256(path)!=reference["sha256"]:raise RuntimeError("CPU gate hash changed")
    gate=read_json(path)
    if gate["status"]!="passed" or not gate["checks"] or not all(v is True for v in gate["checks"].values()):raise RuntimeError("CPU gate failed")
    for child in gate["reports"]:
        if sha256(child["path"])!=child["sha256"] or not (read_json(child["path"]).get("status")=="passed" or read_json(child["path"]).get("passed") is True):raise RuntimeError("CPU child gate differs")
    for name,value in gate["code_sha256"].items():
        if sha256(HERE/name)!=value:raise RuntimeError("Gate code changed: "+name)
    return gate


def verify_config(job):
    if sha256(job["config"])!=job["config_sha256"]:raise RuntimeError("Config changed: "+job["id"])
    c=read_json(job["config"])
    if not Path(c["outputs_root"]).resolve().is_relative_to(Path(job["output_dir"]).resolve()):raise ValueError("Output ownership mismatch")
    if c.get("registered") is not True or c.get("plan_only") is not False or c.get("automatic_start") is not True:raise ValueError("Unregistered config")
    if c.get("protocol")!=PROTOCOL or c.get("cuda_deterministic_algorithms") is not True or c.get("cuda_deterministic_warn_only") is not False:raise ValueError("Strict execution guard missing")
    if (c.get("seed"),c.get("lora_rank"),c.get("gam_grad_rho"),c.get("gam_grad_norm_rho"))!=([2],10,.05,1.):raise ValueError("Scientific config mismatch")
    if c.get("required_runner")!=job["command"][1]:raise ValueError("Wrong exact execution entry")
    if job["kind"]=="training" and c["update_direction"]!=job["direction"]:raise ValueError("Direction mismatch")
    if not c.get("ewc_skip_task_fisher") or c.get("ewc_lambda")!=0. or any(c.get(k,False) for k in ("approximate_fisher_in_task","in_task_weight_merge","task_endpoint_weight_merge","as1_delta_w_penalty","use_sgv","gam_use_sgv")):raise ValueError("Pure gradient configuration violated")
    return c


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
