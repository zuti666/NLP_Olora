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
PROTOCOL = "center_joint_backward_deterministic_exact_restore_rng_replay_v3"
VARIANTS = {"CPLUS", "CMINUS", "BF", "BB"}
VARIANT_FACTORS = {"CPLUS": ("forward", "centered", 1),
                   "CMINUS": ("forward", "centered", -1),
                   "BF": ("backward_corrected", "forward", 1),
                   "BB": ("backward_corrected", "backward_corrected", 1)}
CPU_CHECKS = {"four_new_formulas", "ff_fb_same_state_regression",
              "exact_state_rng_restoration", "dynamic_ce_budget", "bb_no_positive_probes",
              "exceptions_restore", "deterministic_config"}


def expected_budget(factors):
    first = factors["first_location_estimator"]
    second = factors["second_residual_estimator"]
    if first not in ("forward", "backward_corrected") or second not in ("forward", "backward_corrected", "centered"):
        raise ValueError("Unknown explicit probe estimator")
    fplus, fminus = int(first == "forward"), int(first == "backward_corrected")
    plus, minus = int(second in ("forward", "centered")), int(second in ("backward_corrected", "centered"))
    return dict(train_ce_per_update=3+plus+minus, diagnostic_ce_per_update=0,
                first_plus_ce_per_update=fplus, first_minus_ce_per_update=fminus,
                g3_plus_ce_per_update=plus, g3_minus_ce_per_update=minus,
                local_plus_ce_per_update=fplus+plus, local_minus_ce_per_update=fminus+minus)


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


def upstream_ready(suite):
    """Preserve FIFO without changing the live upstream controller or its queue."""
    upstream = suite["upstream_campaign"]
    if sha256(upstream["suite_path"]) != upstream["suite_sha256"]:
        raise RuntimeError("Upstream immutable suite changed")
    parent_suite = read_json(upstream["suite_path"])
    state = read_json(upstream["state_path"])
    formals = [j for j in parent_suite["jobs"] if j["kind"] == "training" and not j["smoke"]]
    if len(formals) != upstream["required_formal_count"]:
        raise RuntimeError("Upstream formal grid changed")
    completed = sum(state["jobs"][j["id"]]["status"] == "completed" for j in formals)
    failed = [j["id"] for j in formals if state["jobs"][j["id"]]["status"] == "failed"]
    if failed or state["status"].startswith("halted"):
        raise RuntimeError("Upstream failed; automatic continuation is forbidden: " + repr(failed))
    ready = state["status"] == "completed" and completed == len(formals)
    for reference in suite["reused_results"]:
        for key, path in reference["evidence_paths"].items():
            if sha256(path) != reference["evidence_sha256"][key]:
                raise RuntimeError("Validated reference changed: " + reference["variant"] + "/" + key)
    return dict(ready=ready, completed_formal=completed, expected_formal=len(formals),
                upstream_status=state["status"], upstream_updated_at=state["updated_at"])


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
    if len(suite.get("reused_results", [])) != 2 or {x["variant"] for x in suite["reused_results"]} != {"FF", "FB"}:
        raise ValueError("Exactly two validated FF/FB references required")
    if suite.get("diagnostic_protocol") != "training_context_only_no_extra_gradient_v3" or suite.get("diagnostic_gradients_per_task") != 0:
        raise ValueError("No extra gradient diagnostic is allowed in v3")
    jobs = suite["jobs"]
    identifiers = {job["id"] for job in jobs}
    if suite_counts(suite) != dict(jobs=9, smoke=4, formal=4, pairing=1, reused=2) or len(identifiers) != 9:
        raise ValueError("Expected four smoke, one pairing, four formal jobs plus two reference results")
    smokes = {job["id"] for job in jobs if job["smoke"]}
    pairing = {job["id"] for job in jobs if job["kind"] == "pairing"}
    if len(smokes) != 4 or len(pairing) != 1:
        raise ValueError("Wrong smoke/pairing grid")
    upstream = suite.get("upstream_campaign", {})
    if upstream.get("required_formal_count") != 16 or not Path(upstream.get("state_path", "")).is_file():
        raise ValueError("Existing sixteen-formal campaign must remain the upstream FIFO dependency")
    seen_outputs = []
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["id"]):
            raise ValueError("Invalid job identifier")
        if job["kind"] not in {"training", "pairing"} or set(job["depends_on"]) - identifiers:
            raise ValueError("Unknown job kind or dependency")
        if job["smoke"] and job["depends_on"]:
            raise ValueError("Smokes have no within-campaign training dependency")
        if job["kind"] == "pairing" and set(job["depends_on"]) != smokes:
            raise ValueError("Pairing gate must wait for all four accepted smokes")
        if job["kind"] == "training" and not job["smoke"] and set(job["depends_on"]) != pairing:
            raise ValueError("Every formal job must wait for the accepted pairing gate")
        if job["kind"] == "training":
            traits = (job["first_location_estimator"], job["second_residual_estimator"], job["residual_coefficient"])
            if traits != VARIANT_FACTORS.get(job["variant"]):
                raise ValueError("Variant label does not match the declared mathematical condition")
            budget = expected_budget(job)
            if job.get("expected_diagnostic_gradient_calls") != 0 or job.get("expected_training_ce") != job["expected_steps"]*budget["train_ce_per_update"]:
                raise ValueError("Wrong training/diagnostic gradient budget")
            if job["radius"] != 1. or job["rho"] != .05 or job["seed"] != 2 or job["rank"] != 10 or job["update_base"] != "g0":
                raise ValueError("Scientific setting mismatch")
        out = Path(job["output_dir"]).resolve()
        owned_root = HERE / ("gates" if job["kind"] == "pairing" else "outputs")
        if not out.is_relative_to(owned_root):
            raise ValueError("Output must belong to this isolated campaign")
        if any(out == old or out.is_relative_to(old) or old.is_relative_to(out) for old in seen_outputs):
            raise ValueError("Overlapping outputs")
        seen_outputs.append(out)
    for smoke in (True, False):
        variants = [j["variant"] for j in jobs if j["kind"] == "training" and j["smoke"] is smoke]
        if set(variants) != VARIANTS or len(variants) != 4:
            raise ValueError("Each new variant needs one smoke and one formal job")
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
    keys = ("update_base", "first_location_estimator", "second_residual_estimator", "residual_coefficient")
    if any(config.get(k) != job.get(k) for k in keys):
        raise ValueError("Scientific factor mismatch")
    if config.get("gam_grad_rho") != .05 or config.get("gam_grad_norm_rho") != 1. or config.get("seed") != [2] or config.get("lora_rank") != 10:
        raise ValueError("Wrong fixed scientific protocol")
    if config.get("positive_side_diagnostic_gradients") is not False or config.get("extra_diagnostic_gradient_calls") != 0:
        raise ValueError("Hidden opposite-side diagnostics forbidden")
    if config.get("automatic_start") is not True or config.get("required_runner") != str(HERE / "trainer/four_probe_train.py"):
        raise ValueError("Wrong registered execution entry")
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
