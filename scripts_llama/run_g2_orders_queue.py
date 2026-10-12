#!/usr/bin/env python3
"""Durable seed0 pure HF-AdamW G2 queue: six Orders on physical GPU 0,1,2,3.

Prepare freezes the source and records local input hashes without allocating CUDA.
Run waits for all four selected GPUs, performs fresh CUDA/BF16 witnesses, then
admits each four-stage Order only after its own two-stage bounded smoke chain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "CurrentG2SixOrderQueue_v1"
DATA_PATHS = {"dbpedia": "TC/dbpedia", "amazon": "SC/amazon", "yahoo": "TC/yahoo", "agnews": "TC/agnews"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def import_runner(path: Path, repository: Path):
    module_spec = importlib.util.spec_from_file_location("_current_g2_shared_runner", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Unable to load the frozen shared runner.")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.REPO = repository
    return module


def input_paths(configs: list[dict]) -> list[Path]:
    paths = {REPO / "experiment_queue/nlp_multiseed_bf16_b64_ddp4_g0-3_20260922T150750Z/history_reuse_20260926/final_per_seed.csv"}
    # compute_metrics eagerly loads this tokenizer relative to its frozen source.
    # Pin its original inputs as well as the copied source files.
    metrics_tokenizer = REPO / "data/gpt2tokenizer"
    paths.update(metrics_tokenizer / name for name in (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "merges.txt", "vocab.json"))
    for cfg in configs:
        for task, relative in DATA_PATHS.items():
            for name in ("train.json", "dev.json", "test.json", "labels.json"):
                paths.add(Path(cfg["data_dir"]) / relative / name)
        model = Path(cfg["model_name_or_path"])
        # All local model/tokenizer files, including checkpoint shards, are fixed.
        paths.update(p for p in model.rglob("*") if p.is_file() and not any(
            part in {".git", ".cache"} for part in p.relative_to(model).parts))
        paths.add(Path(cfg["python_executable"]))
        paths.add(Path(cfg["instruction_file"]))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required local inputs are missing: {missing}")
    return sorted(paths)


def prepare(root: Path, config_paths: list[Path], artifact_root: Path | None = None) -> dict:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite queue directory: {root}")
    artifact_root = (artifact_root or root / "artifacts").resolve()
    if artifact_root.exists():
        raise FileExistsError(f"Refusing to reuse an existing artifact directory: {artifact_root}")
    runner = import_runner(REPO / "scripts_llama/run_g2_llama.py", REPO)
    configs = [runner.load_config(path.resolve()) for path in config_paths]
    pairs = [(cfg["model_size"], cfg["order"]) for cfg in configs]
    if len(pairs) != 6 or set(pairs) != {(size, order) for size in ("1B", "3B") for order in (1, 2, 3)}:
        raise ValueError("Provide exactly the six distinct model x Order configurations.")
    if any(cfg["method"] != "inclora_g2" or cfg["optimizer_class"] != "transformers.AdamW" for cfg in configs):
        raise ValueError("This queue admits only pure IncLoRA-G2 with HF AdamW.")
    inputs = {str(path): {"sha256": digest(path), "bytes": path.stat().st_size} for path in input_paths(configs)}
    root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=False)
    source, _ = runner.freeze_source(root)
    for name in ("run_g2_llama.py", "run_g2_orders_queue.py", "collect_g2_orders_results.py"):
        target = source / "scripts_llama" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "scripts_llama" / name, target)
    jobs = []
    for cfg in sorted(configs, key=lambda c: (c["model_size"], c["order"])):
        job_root = artifact_root / "jobs" / cfg["name"]
        cfg_path = source / "configs/g2_llama" / (cfg["name"] + ".json")
        write_json(cfg_path, cfg)
        plan = runner.build_plan(cfg, [0], job_root, source=source)
        job = {"id": cfg["name"], "model_size": cfg["model_size"], "order": cfg["order"],
               "seed": 0, "config": cfg, "plan": plan}
        jobs.append(job)
        write_json(root / "plans" / (cfg["name"] + ".json"), plan)
    source_hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    write_json(root / "source_sha256.json", source_hashes)
    manifest = {"schema": SCHEMA, "created_at": utc(), "repository": str(REPO), "queue_root": str(root),
                "source": str(source), "artifact_root": str(artifact_root), "source_sha256": source_hashes, "input_sha256": inputs,
                "gpus": "0,1,2,3", "seeds": [0], "formal_orders": 6, "formal_stages": 24,
                "smoke_orders": 6, "smoke_stages": 12, "world_size": 4, "global_batch": 64, "jobs": jobs}
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n")
    write_json(root / "runtime/queue_state.json", {"schema": SCHEMA, "status": "prepared", "prepared_at": utc(),
                "completed_stages": [], "formal_completed": 0, "smoke_completed": 0, "heartbeat_at": utc()})
    return manifest


def verify_manifest(root: Path) -> dict:
    if digest(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise RuntimeError("The prepared manifest changed.")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or Path(manifest.get("queue_root", "")).resolve() != root.resolve():
        raise RuntimeError("Queue schema or absolute root differs from the prepared manifest.")
    if manifest.get("gpus") != "0,1,2,3" or manifest.get("seeds") != [0]:
        raise RuntimeError("The prepared GPU/seed authorization changed.")
    source = Path(manifest["source"])
    current_paths = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if current_paths != set(manifest["source_sha256"]):
        raise RuntimeError("The frozen source file population changed.")
    for relative, expected in manifest["source_sha256"].items():
        if digest(source / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for absolute, expected in manifest["input_sha256"].items():
        path = Path(absolute)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen local input changed: {absolute}")
    return manifest


def witness_code() -> str:
    return '''import json, math, torch
torch.manual_seed(0)
x=torch.randn(8,8,device="cuda")
y=x@x
print("WITNESS",tuple(y.shape),torch.cuda.get_device_name(),float(y.sum()),flush=True)
assert tuple(y.shape)==(8,8) and torch.isfinite(y).all()
assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
w=torch.randn(8,8,device="cuda",dtype=torch.float32,requires_grad=True)
with torch.autocast("cuda",dtype=torch.bfloat16):
    z=(x@w).square().mean()
z.backward()
assert torch.isfinite(z) and w.grad is not None and torch.isfinite(w.grad).all()
print("BF16_WITNESS",json.dumps({"finite":True,"device":torch.cuda.get_device_name(),"loss":float(z),"grad_norm":float(w.grad.norm()),"torch":torch.__version__,"cuda":torch.version.cuda}),flush=True)
'''


def run_child(command: list[str], log: Path, env: dict, cwd: Path, handles: list[Any],
              heartbeat, *, heartbeat_seconds: float = 15.0, timeout_seconds: float | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=tuple(handle.fileno() for handle in handles))
        started = time.monotonic()
        heartbeat(process.pid)
        while True:
            try:
                returncode = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                heartbeat(process.pid)
                if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                    # Only the direct witness child created here is terminated.
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(f"Own witness exceeded {timeout_seconds}s: {log}")
        heartbeat(None)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)


def run_witnesses(manifest: dict, runner, gpus: list[dict], handles: list[Any], state: dict, save) -> None:
    root, source = Path(manifest["queue_root"]), Path(manifest["source"])
    spec = root / "runtime/witness_spec.json"
    write_json(spec, {"schema": "CurrentG2FreshCUDAWitness_v1", "seed": 0, "gpus": gpus, "timestamp": utc()})
    python = manifest["jobs"][0]["config"]["python_executable"]
    results = []
    for gpu in gpus:
        state.update(status="cuda_witness", active_gpu=gpu["index"])
        save()
        env = runner.environment(source, Path(manifest["artifact_root"]), [gpu], spec)
        log = root / "runtime" / f"cuda_witness_gpu{gpu['index']}_{time.time_ns()}.log"
        run_child([python, "-c", witness_code()], log, env, Path(manifest["repository"]), handles,
                  lambda pid: (state.update(child_pid=pid), save()), timeout_seconds=120)
        output = log.read_text()
        if "WITNESS (8, 8)" not in output or '"finite": true' not in output or "BF16_WITNESS" not in output:
            raise RuntimeError(f"Fresh CUDA/BF16 witness output failed: {log}")
        results.append({"gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"], "status": "passed", "log": str(log), "log_sha256": digest(log)})
    write_json(root / "runtime/cuda_witness.json", {"status": "passed", "timestamp": utc(), "results": results})
    state.pop("active_gpu", None)


def pause_requested(root: Path) -> bool:
    return (root / "runtime/PAUSE").exists()


def stage_key(job: dict, spec: dict) -> str:
    return f"{job['id']}:s{spec['seed']}:{'smoke' if spec['smoke'] else 'formal'}:{spec['stage']}"


def latest_trace_step(output: str | None) -> int | None:
    if not output:
        return None
    trace = Path(output) / "g2_step_trace.jsonl"
    if not trace.is_file():
        return None
    with trace.open("rb") as stream:
        stream.seek(max(0, trace.stat().st_size - 32768))
        rows = stream.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(rows):
        try:
            value = json.loads(line).get("step")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def run(root: Path, *, wait_seconds: float = 30.0) -> int:
    if wait_seconds <= 0 or wait_seconds > 60:
        raise ValueError("Queue waiting interval must be between 0 and 60 seconds.")
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "controller.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another controller already owns this queue.") from exc
    handles: list[Any] = []
    state_path = runtime / "queue_state.json"
    state = json.loads(state_path.read_text())
    if state.get("status") in {"failed", "completed", "stopped"}:
        lock.close()
        if state["status"] == "completed":
            return 0
        raise RuntimeError("Refusing automatic restart of a failed/stopped queue; preserve its artifacts.")
    state.update(controller_pid=os.getpid(), child_pid=None, started_at=utc())

    def save():
        state["heartbeat_at"] = utc()
        state["active_global_step"] = latest_trace_step(state.get("active_output"))
        state["formal_completed"] = sum(not row["smoke"] for row in state["completed_stages"])
        state["smoke_completed"] = sum(row["smoke"] for row in state["completed_stages"])
        write_json(state_path, state)

    def refresh_results(manifest):
        collect = Path(manifest["source"]) / "scripts_llama/collect_g2_orders_results.py"
        result = subprocess.run([manifest["jobs"][0]["config"]["python_executable"], str(collect), "--queue-root", str(root)],
                                cwd=Path(manifest["repository"]), text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Result collection failed: {result.stderr[-4000:]}")

    try:
        manifest = verify_manifest(root)
        runner = import_runner(Path(manifest["source"]) / "scripts_llama/run_g2_llama.py", Path(manifest["repository"]))
        if hasattr(runner, "validate_execution_profile"):
            for job in manifest["jobs"]:
                runner.validate_execution_profile(job["config"])
        completed = {row["key"] for row in state["completed_stages"]}
        # A controller crash must never silently overwrite an interrupted output.
        for job in manifest["jobs"]:
            for spec in job["plan"]["stages"]:
                if stage_key(job, spec) in completed:
                    runner.verify_stage(spec)
        gpus = None
        for job in manifest["jobs"]:
            for spec in job["plan"]["stages"]:
                key = stage_key(job, spec)
                if key in completed:
                    continue
                while pause_requested(root):
                    for handle in handles:
                        handle.close()
                    handles, gpus = [], None
                    state.update(status="paused", active_stage=key, child_pid=None)
                    save()
                    if (runtime / "STOP_AFTER_STAGE").exists():
                        state.update(status="stopped", active_stage=None)
                        save()
                        return 0
                    time.sleep(wait_seconds)
                if (runtime / "STOP_AFTER_STAGE").exists():
                    state.update(status="stopped", active_stage=None)
                    save()
                    return 0
                if gpus is None:
                    while gpus is None:
                        if (runtime / "STOP_AFTER_STAGE").exists():
                            state.update(status="stopped", active_stage=None)
                            save()
                            return 0
                        if pause_requested(root):
                            state.update(status="paused", active_stage=key)
                            save()
                            time.sleep(wait_seconds)
                            continue
                        state.update(status="waiting_gpus", active_stage=key)
                        save()
                        try:
                            gpus, handles = runner.claim_gpus("0,1,2,3")
                        except runner.GPUUnavailableError as exc:
                            state["waiting_reason"] = str(exc)
                            save()
                            time.sleep(wait_seconds)
                    state["gpu_reservation"] = gpus
                    state.pop("waiting_reason", None)
                    save()
                    run_witnesses(manifest, runner, gpus, handles, state, save)
                verify_manifest(root)
                if not spec["smoke"]:
                    smoke_done = [row for row in state["completed_stages"] if row["job"] == job["id"] and row["smoke"]]
                    if len(smoke_done) != 2:
                        raise RuntimeError("Formal admission requires this model/Order's two verified smoke stages.")
                if spec["previous_adapter"] and not Path(spec["previous_adapter"]).is_dir():
                    raise RuntimeError("The preceding stage adapter is absent.")
                output = Path(spec["output_dir"])
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite interrupted/existing stage: {output}")
                spec_path = root / "specs" / (key.replace(":", "_") + ".json")
                write_json(spec_path, spec)
                log = Path(manifest["artifact_root"]) / "logs" / (spec_path.stem + ".log")
                state.update(status="running_smoke" if spec["smoke"] else "running_formal", active_stage=key,
                             active_output=str(output), active_log=str(log), stage_started_at=utc())
                save()
                run_child(spec["command"], log, runner.environment(Path(manifest["source"]), Path(manifest["artifact_root"]), gpus, spec_path),
                          Path(manifest["repository"]), handles, lambda pid: (state.update(child_pid=pid), save()))
                receipt = runner.verify_stage(spec)
                state["completed_stages"].append({"key": key, "job": job["id"], "smoke": spec["smoke"],
                            "stage": spec["stage"], "receipt": receipt, "completed_at": utc()})
                completed.add(key)
                state.update(active_output=None, child_pid=None)
                save()
                refresh_results(manifest)
        state.update(status="completed", active_stage=None, active_output=None, active_log=None,
                     completed_at=utc(), child_pid=None)
        save()
        refresh_results(manifest)
        return 0
    except BaseException as exc:
        state.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc), "timestamp": utc()},
                     child_pid=state.get("child_pid"))
        save()
        write_json(root / "failure.json", state["failure"])
        raise
    finally:
        for handle in handles:
            handle.close()
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="Freeze the six-job manifest; no CUDA allocation.")
    parser.add_argument("--artifact-root", type=Path, help="Fresh absolute directory for checkpoints, predictions, caches and training logs.")
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    root = args.queue_root.resolve()
    if args.prepare:
        manifest = prepare(root, args.config, args.artifact_root)
        print(json.dumps({"status": "prepared", "queue_root": str(root), "formal_orders": manifest["formal_orders"],
                          "formal_stages": manifest["formal_stages"], "smoke_stages": manifest["smoke_stages"]}))
        return 0
    if args.config or args.artifact_root:
        parser.error("--config/--artifact-root are accepted only with --prepare; run uses the immutable manifest.")
    return run(root, wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
