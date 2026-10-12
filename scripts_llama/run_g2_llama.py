#!/usr/bin/env python3
"""Current-G2 migration for Orders1--3: explicit JSON, frozen source, four idle GPUs.

The dry run uses only Python's standard library, creates no files and never
queries CUDA. A real run uses a fresh output directory, reserves exactly four
idle GPUs, and runs a two-task smoke chain before each four-task formal chain.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TASK_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
}
TASK_UPDATES = {"dbpedia": 218, "amazon": 78, "yahoo": 156, "agnews": 62}
METRICS_TOKENIZER_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "merges.txt", "vocab.json",
)
SEEDS = (0, 42, 1993)
METHODS = {
    "inclora_g2": ("inclora", "g2-adamw_hf", "transformers.AdamW", 1e-6, 0.0, 0.0, 0.0),
    "inclora_g2_gambase": ("inclora", "g2-adamw_torch", "torch.optim.AdamW", 1e-8, .01, 0.0, 0.0),
}


class GPUUnavailableError(RuntimeError):
    pass


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg.get("schema_version") != 1 or cfg.get("method") not in METHODS:
        raise ValueError("Unknown G2 configuration schema or method.")
    order = cfg.get("order")
    if (cfg.get("model_size") not in ("1B", "3B") or not isinstance(order, int) or isinstance(order, bool)
            or order not in TASK_ORDERS or cfg.get("tasks") != TASK_ORDERS[order]):
        raise ValueError("This entry point fixes Llama-3.2-1B/3B and the historical task order for Orders1--3.")
    if cfg.get("seeds") != list(SEEDS) or cfg.get("default_seed") != 0:
        raise ValueError("The paired seed protocol is [0, 42, 1993], initially seed0.")
    version = cfg.get("decoder_pipeline_version")
    if version is not None and version != "decoder_token_ids_v2":
        raise ValueError("Unknown decoder token-boundary protocol.")
    if not isinstance(cfg.get("require_gradient_diagnostics", False), bool):
        raise ValueError("require_gradient_diagnostics must be boolean.")
    execution = cfg["execution"]
    if {k: execution.get(k) for k in ("world_size", "global_batch", "per_device_train_batch_size", "gradient_accumulation_steps")} != {
        "world_size": 4, "global_batch": 64, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    }:
        raise ValueError("The reference protocol requires DDP4, batch1, GA16 and global batch64.")
    if cfg.get("formal_stage_updates") != [TASK_UPDATES[task] for task in cfg["tasks"]] or cfg.get("stage_learning_rates") != [.001, .0001, .0001, .0001]:
        raise ValueError("Formal update budgets and learning rates must match the reference protocol.")
    expected = METHODS[cfg["method"]]
    training = cfg["training"]
    actual = (training.get("lora_strategy"), training.get("optimizer_type"), cfg.get("optimizer_class"),
              training.get("adam_epsilon"), training.get("weight_decay"), training.get("ewclora_lambda"), training.get("olora_lamda_1"))
    if actual != expected:
        raise ValueError(f"Method/backend/regularizer preset mismatch: {actual!r} != {expected!r}")
    fixed = {
        "train_method": "lora", "lora_dim": 8, "num_train_epochs": 1,
        "bf16": True, "fp16": False, "bf16_full_eval": False, "fp16_full_eval": False,
        "ddp_find_unused_parameters": False, "lr_scheduler_type": "constant", "warmup_steps": 0,
        "adam_beta1": .9, "adam_beta2": .999, "max_grad_norm": 1.0,
        "max_source_length": 512, "max_target_length": 50, "generation_max_length": 50,
        "g2_replay_rng_state": True, "g2_grad_reduce": "mean", "g2_require_fp32_trainable": True,
        "g2_xi": 1e-12, "olora_lamda_2": 0.0,
    }
    for key, value in fixed.items():
        if training.get(key) != value:
            raise ValueError(f"The shared protocol fixes {key}={value!r}.")
    if "g2_include_olora_regularization" in training or any("fisher" in key for key in training):
        raise ValueError("This pure-G2 launcher forbids regularization hooks and Fisher state loading.")
    if cfg.get("adapter_geometry") != {"rank": 8, "alpha": 32, "dropout": .1, "target_modules": ["q_proj", "v_proj"]}:
        raise ValueError("Adapter geometry must be rank8, alpha32, dropout0.1, Q/V.")
    for key in ("python_executable", "model_name_or_path", "data_dir", "instruction_file"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute local path.")
    validate_radii(training)
    return cfg


def validate_task_configs(cfg: dict, source: Path) -> None:
    """Check the actual on-disk train/test sets before accepting a chain."""
    for stage, task in enumerate(cfg["tasks"], start=1):
        directory = source / "configs" / f"order{cfg['order']}_configs" / task
        for split in ("train", "test"):
            data = json.loads((directory / f"{split}_tasks.json").read_text())
            datasets = [row["dataset name"] for rows in data.values() for row in rows]
            expected = [task] if split == "train" else cfg["tasks"][:stage]
            if len(datasets) != len(expected) or set(datasets) != set(expected):
                raise ValueError(f"Order{cfg['order']} stage{stage} {split} task configuration differs: {datasets!r}")


def validate_radii(training: dict) -> None:
    interval = training.get("g2_quantization_diagnostics_interval", 0)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("G2 quantization diagnostic interval must be a nonnegative integer.")
    for side in ("inner", "outer"):
        mode, radius = training.get(f"g2_{side}_radius_mode"), training.get(f"g2_{side}_radius")
        if mode not in ("sgd_step", "fixed_l2"):
            raise ValueError(f"Unknown {side} radius mode {mode!r}.")
        if mode == "sgd_step" and radius is not None:
            raise ValueError(f"The {side} sgd_step mode must not carry a fixed radius.")
        if mode == "fixed_l2" and (isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0):
            raise ValueError(f"The {side} fixed_l2 mode requires a finite positive radius.")


def validate_execution_profile(cfg: dict) -> None:
    """Legacy configs may be inspected, but cannot silently train on v2 data."""
    if cfg.get("decoder_pipeline_version") != "decoder_token_ids_v2":
        raise ValueError(
            "Legacy G2 configs are inspection-only after the decoder fix. "
            "Use an explicit decoder_token_ids_v2 profile from configs/g2_llama/corrected_v2."
        )


def make_stage(cfg: dict, seed: int, stage: int, run_root: Path, *, smoke: bool, source: Path | None = None) -> dict:
    if seed not in SEEDS or stage not in (1, 2, 3, 4):
        raise ValueError("Unsupported seed or stage.")
    tasks, order = cfg["tasks"], cfg["order"]
    task = tasks[stage - 1]
    chain = run_root / ("smoke" if smoke else "full") / f"seed_{seed}" / f"order_{order}"
    previous = chain / f"{stage - 1}-{tasks[stage - 2]}" / "adapter" if stage > 1 else None
    options = dict(cfg["training"])
    options.update(seed=seed, data_seed=seed, do_train=True, do_predict=True, predict_with_generate=True,
                   model_name_or_path=str(previous) if previous else cfg["model_name_or_path"],
                   data_dir=cfg["data_dir"], instruction_file=cfg["instruction_file"],
                   task_config_dir=str(REPO / "configs" / f"order{order}_configs" / task),
                   output_dir=str(chain / f"{stage}-{task}"),
                   per_device_train_batch_size=cfg["execution"]["per_device_train_batch_size"],
                   gradient_accumulation_steps=cfg["execution"]["gradient_accumulation_steps"],
                   learning_rate=cfg["stage_learning_rates"][stage - 1],
                   max_steps=2 if smoke else cfg["formal_stage_updates"][stage - 1],
                   logging_steps=1 if smoke else 10,
                   run_name=f"{cfg['name']}_s{seed}_{'smoke' if smoke else 'formal'}_stage{stage}")
    if smoke:
        options.update(max_train_samples=128, max_predict_samples=16)
    src = source or REPO
    if source is not None:
        options["task_config_dir"] = str(source / "configs" / f"order{order}_configs" / task)
        options["instruction_file"] = str(source / "configs/instruction_config.json")
    command = [cfg["python_executable"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=4",
               "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0", str(src / "src/run_uie_lora.py")]
    for key, value in options.items():
        if value is None:
            continue
        command.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return {"seed": seed, "stage": stage, "task": task, "smoke": smoke,
            "order": order, "model_size": cfg["model_size"], "tasks": tasks, "seen_tasks": tasks[:stage],
            "expected_updates": options["max_steps"], "expected_saved_rank": stage * 8,
            "expected_target_modules": 32 if cfg["model_size"] == "1B" else 56,
            "expected_test_samples": 16 if smoke else stage * 7600,
            "output_dir": options["output_dir"], "previous_adapter": str(previous) if previous else None,
            "optimizer_class": cfg["optimizer_class"], "options": options, "command": command,
            "decoder_pipeline_version": cfg.get("decoder_pipeline_version"),
            "require_gradient_diagnostics": cfg.get("require_gradient_diagnostics", False)}


def build_plan(cfg: dict, seeds: list[int], run_root: Path, *, max_stages: int = 4, smoke_only: bool = False,
               source: Path | None = None) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("Use distinct seeds from [0, 42, 1993].")
    if max_stages not in (1, 2, 3, 4):
        raise ValueError("max_stages must be between one and four.")
    validate_task_configs(cfg, source or REPO)
    stages = []
    for seed in seeds:
        stages.extend(make_stage(cfg, seed, stage, run_root, smoke=True, source=source) for stage in range(1, min(max_stages, 2) + 1))
        if not smoke_only:
            stages.extend(make_stage(cfg, seed, stage, run_root, smoke=False, source=source) for stage in range(1, max_stages + 1))
    return {"schema_version": 1, "method": cfg["method"], "model_size": cfg["model_size"], "order": cfg["order"],
            "tasks": cfg["tasks"],
            "run_root": str(run_root), "seeds": seeds, "world_size": 4, "global_batch": 64,
            "formal_orders": 0 if smoke_only else len(seeds),
            "formal_stages": 0 if smoke_only else len(seeds) * max_stages,
            "formal_updates_per_complete_order": sum(cfg["formal_stage_updates"]),
            "stages": stages}


def gpu_snapshot() -> list[dict]:
    snapshot = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True)
    busy = {row[0].strip() for row in csv.reader(apps.stdout.splitlines()) if row and row[0].strip().startswith("GPU-")}
    return [{"index": int(row[0]), "uuid": row[1].strip(), "memory_mib": int(row[2]), "utilization": int(row[3]),
             "has_compute_process": row[1].strip() in busy} for row in csv.reader(snapshot.stdout.splitlines())]


def idle(gpu: dict) -> bool:
    return gpu["memory_mib"] < 500 and gpu["utilization"] <= 5 and not gpu["has_compute_process"]


def validate_source_assets(source: Path) -> None:
    """Fail before GPU allocation if offline metric-tokenizer assets are absent."""
    tokenizer = source / "data/gpt2tokenizer"
    missing = [str(tokenizer / name) for name in METRICS_TOKENIZER_FILES
               if not (tokenizer / name).is_file() or (tokenizer / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required local metrics tokenizer assets: " + ", ".join(missing))


def claim_gpus(requested: str) -> tuple[list[dict], list[Any]]:
    validate_source_assets(REPO)
    values = [value.strip() for value in requested.split(",")]
    if len(values) != 4 or len(set(values)) != 4 or any(not value for value in values):
        raise ValueError("--gpus must specify exactly four distinct physical indices or UUIDs.")
    snapshots = gpu_snapshot()
    known = {str(gpu["index"]): gpu["uuid"] for gpu in snapshots}
    uuids = [known.get(value, value) for value in values]
    if len(set(uuids)) != 4 or not set(uuids) <= {gpu["uuid"] for gpu in snapshots}:
        raise ValueError("A selected GPU is absent or repeated by index and UUID.")
    selected = [next(gpu for gpu in snapshots if gpu["uuid"] == uid) for uid in uuids]
    handles = []
    try:
        for gpu in selected:
            if not idle(gpu):
                raise GPUUnavailableError(f"GPU {gpu['index']} is occupied; no job was started.")
            lock_paths = [REPO / ".cache/gpu_locks" / (gpu["uuid"] + ".lock"),
                          Path("/tmp/olora_gamfs_gpu_locks") / f"gpu{gpu['index']}.lock",
                          Path("/tmp/olora_gamfs_gpu_locks") / (gpu["uuid"] + ".lock")]
            for lock in lock_paths:
                lock.parent.mkdir(parents=True, exist_ok=True)
                handle = lock.open("a+")
                handles.append(handle)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise GPUUnavailableError(f"GPU {gpu['index']} is reserved by another experiment.") from exc
        current = {gpu["uuid"]: gpu for gpu in gpu_snapshot()}
        if any(uid not in current or not idle(current[uid]) for uid in uuids):
            raise GPUUnavailableError("A GPU became busy during reservation; no job was started.")
        for handle in handles:
            handle.write(json.dumps({"pid": os.getpid(), "method": "Current-G2-Llama", "started_at": timestamp()}) + "\n")
            handle.flush()
        return selected, handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def freeze_source(run_root: Path) -> tuple[Path, dict]:
    validate_source_assets(REPO)
    source = run_root / "source"
    shutil.copytree(REPO / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for order in TASK_ORDERS:
        shutil.copytree(REPO / "configs" / f"order{order}_configs", source / "configs" / f"order{order}_configs")
    shutil.copy2(REPO / "configs/instruction_config.json", source / "configs/instruction_config.json")
    # compute_metrics constructs this tokenizer during import, relative to
    # src/compute_metrics.py. Preserve that layout in the immutable snapshot.
    shutil.copytree(REPO / "data/gpt2tokenizer", source / "data/gpt2tokenizer",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    validate_source_assets(source)
    files = {str(path.relative_to(source)): sha256(path) for path in sorted(source.rglob("*")) if path.is_file()}
    write_json(run_root / "source_sha256.json", files)
    return source, files


def verify_source(source: Path, hashes: dict) -> None:
    for relative, digest in hashes.items():
        if sha256(source / relative) != digest:
            raise RuntimeError(f"Frozen source changed: {relative}")


def environment(source: Path, run_root: Path, gpus: list[dict], spec_file: Path) -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "NLP_MULTISEED_SPEC"} or key.startswith(
                       ("TORCHELASTIC_", "ACCELERATE_", "INCLORA_BF16_", "INCLORA_GAM_")):
            env.pop(key)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus),
               PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               NLTK_DATA=str(REPO / ".cache/nltk_data"), G2_LLAMA_SPEC=str(spec_file))
    for name, sub in {"HF_HOME": "hf", "HF_MODULES_CACHE": "modules", "HF_DATASETS_CACHE": "datasets",
                      "TRANSFORMERS_CACHE": "transformers", "UIE_DATA_CACHE_ROOT": "uie"}.items():
        cache = run_root / "cache" / sub
        cache.mkdir(parents=True, exist_ok=True)
        env[name] = str(cache)
    # Multiprocessing Unix socket names must remain below the platform limit.
    short_tmp = Path("/tmp") / ("olora_g2_" + hashlib.sha256(str(run_root).encode()).hexdigest()[:12])
    short_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(short_tmp)
    return env


def verify_adapter_tensors(spec: dict, weight: Path) -> dict:
    # This import only occurs after a real stage has exited, never in dry-run.
    import torch

    def load(path):
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            value = load_file(str(path), device="cpu")
        else:
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(path, map_location="cpu")
        if not isinstance(value, dict) or not value:
            raise RuntimeError("Adapter checkpoint is empty or not a tensor dictionary.")
        return value

    state = load(weight)
    pairs = {}
    for name, value in state.items():
        if not torch.is_tensor(value) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Saved adapter parameter is nonfinite or not FP32: {name}")
        if "loranew_" in name:
            raise RuntimeError("The exported adapter must concatenate the new rank8 branch.")
        part = "A" if ".lora_A." in name else "B" if ".lora_B." in name else None
        if part is None or value.ndim != 2 or value.shape[0 if part == "A" else 1] != spec["expected_saved_rank"]:
            raise RuntimeError(f"Saved adapter tensor geometry differs: {name}")
        prefix = name.split(f".lora_{part}.")[0]
        if part in pairs.setdefault(prefix, {}):
            raise RuntimeError("A duplicate LoRA tensor was exported.")
        pairs[prefix][part] = value
    if len(pairs) != spec["expected_target_modules"] or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise RuntimeError("Saved Q/V LoRA tensor coverage differs from the model protocol.")
    previous = spec.get("previous_adapter")
    frozen_prefix_exact = None
    if previous:
        previous_dir = Path(previous)
        old_weight = next((previous_dir / name for name in ("adapter_model.bin", "adapter_model.safetensors") if (previous_dir / name).is_file()), None)
        if old_weight is None:
            raise RuntimeError("The preceding adapter checkpoint is missing.")
        old = load(old_weight)
        if set(old) != set(state):
            raise RuntimeError("The preceding adapter's tensor keys changed.")
        for name, value in old.items():
            prefix = state[name][:value.shape[0], :] if ".lora_A." in name else state[name][:, :value.shape[1]]
            if prefix.dtype != value.dtype or not torch.equal(prefix, value):
                raise RuntimeError(f"A frozen historical adapter slice changed: {name}")
        frozen_prefix_exact = True
    return {"adapter_tensor_count": len(state), "adapter_pair_count": len(pairs), "adapter_finite_fp32": True,
            "historical_prefix_exact": frozen_prefix_exact}


def verify_runtime_audit(spec: dict, trace_hash: str, fallback_count: int) -> dict:
    audit_file = Path(spec["output_dir"]) / "g2_runtime_audit.json"
    audit = json.loads(audit_file.read_text())
    expected = {
        "schema": "CurrentG2TrainerAudit_v1", "status": "passed",
        "global_step": spec["expected_updates"], "num_records": spec["expected_updates"],
        "normal_three_read_steps": spec["expected_updates"] - fallback_count,
        "fallback_steps": fallback_count, "tail_windows": 0,
        "trainable_precision": "torch.float32", "bf16_forward": True,
        "strategy": "inclora", "fisher_state_entries": 0,
        "regularization_enabled": False, "history_enabled": False, "projection_enabled": False,
        "trace_sha256": trace_hash, "actual_micro_batch_counts": [16], "world_size": 4,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(f"The final G2 runtime audit differs: {key}={audit.get(key)!r}, expected {value!r}")
    pipeline_version = spec.get("decoder_pipeline_version")
    if pipeline_version is not None:
        pipeline_file = Path(spec["output_dir"]) / "data_pipeline_audit.json"
        pipeline = json.loads(pipeline_file.read_text())
        if (audit.get("decoder_pipeline_version") != pipeline_version
                or pipeline.get("decoder_version") != pipeline_version):
            raise RuntimeError("The corrected decoder pipeline was not actually used.")
        if audit.get("g2_quantization_diagnostics_interval") != spec["options"].get("g2_quantization_diagnostics_interval", 0):
            raise RuntimeError("The recorded G2 quantization diagnostic cadence differs.")
    if not isinstance(audit.get("trainable_elements"), int) or audit["trainable_elements"] <= 0:
        raise RuntimeError("The runtime audit did not observe trainable adapter parameters.")
    backend_prefix = "transformers." if spec["optimizer_class"] == "transformers.AdamW" else "torch.optim."
    if not audit.get("base_optimizer", "").startswith(backend_prefix) or not audit["base_optimizer"].endswith(".AdamW"):
        raise RuntimeError("The final runtime audit has the wrong actual optimizer class.")
    groups = audit.get("base_optimizer_param_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("No actual optimizer parameter groups were recorded.")
    for group in groups:
        if (group.get("betas") != [spec["options"]["adam_beta1"], spec["options"]["adam_beta2"]]
                or group.get("eps") != spec["options"]["adam_epsilon"]
                or group.get("weight_decay") != spec["options"]["weight_decay"]
                or group.get("lr") != spec["options"]["learning_rate"]):
            raise RuntimeError("Actual optimizer hyperparameters differ from the paired configuration.")
    return {"g2_runtime_audit_sha256": sha256(audit_file), "actual_base_optimizer": audit["base_optimizer"],
            "actual_base_optimizer_param_groups": groups, "trainable_elements": audit["trainable_elements"]}


def verify_stage(spec: dict) -> dict:
    output = Path(spec["output_dir"])
    state = json.loads((output / "trainer_state.json").read_text())
    if state.get("global_step") != spec["expected_updates"]:
        raise RuntimeError("Completed update count does not match the paired historical budget.")
    trace_file = output / "g2_step_trace.jsonl"
    trace = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    if len(trace) != spec["expected_updates"] or [row.get("step") for row in trace] != list(range(1, len(trace) + 1)):
        raise RuntimeError("G2 step trace does not cover every submitted update exactly once.")
    expected_backend_prefix = "transformers." if spec["optimizer_class"] == "transformers.AdamW" else "torch.optim."
    expected_reg = False
    fallback_count = 0
    for row in trace:
        checks = {
            "schema": "CurrentG2TrainerStep_v1", "world_size": 4,
            "actual_micro_batches": 16, "configured_gradient_accumulation_steps": 16,
            "effective_window_mean_denominator": 16, "tail_gradient_rescale": 1.0,
            "replay_rng_state": True, "replay_ddp_no_sync_all_micro_batches": True,
            "replay_reduction": "explicit_global_mean_once_per_probe", "clean_gradient_reduction": "native_ddp_mean",
            "history_enabled": False, "g2_history_used": False, "g2_base_optimizer_steps": 1,
            "g2_restore_base_exact": True, "g2_fisher_probe_calls": 0,
            "g2_projection_calls": 0, "g2_hvp_calls": 0,
            "g2_core_explicit_mean_reductions": 0, "g2_gradient_reduction_owner": "trainer",
            "strategy": spec["options"]["lora_strategy"], "g2_regularization_hook_used": expected_reg,
            "clean_regularization_enabled": expected_reg,
            "configured_clip_grad_norm": spec["options"]["max_grad_norm"],
            "g2_final_clip_enabled": True,
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise RuntimeError(f"G2 execution mismatch at step {row.get('step')}: {key}={row.get(key)!r}, expected {expected!r}")
        backend = row.get("base_optimizer", "")
        if not backend.startswith(expected_backend_prefix) or not backend.endswith(".AdamW"):
            raise RuntimeError(f"The actual optimizer backend differs: {backend!r}")
        if row.get("g2_final_direction") == "g2":
            if row.get("g2_ce_gradient_phases") != 3 or row.get("g2_replay_closure_calls") != 2 or row.get("g2_relocation_origin") != "base" or row.get("g2_fallback_reason") is not None:
                raise RuntimeError("A nondegenerate G2 update must use exactly three CE gradients and restore BASE.")
            if spec.get("require_gradient_diagnostics"):
                for key in ("g2_g2_minus_g0_norm", "g2_g2_minus_g0_relative_norm", "g2_response_relative_norm"):
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                        raise RuntimeError(f"Missing or invalid G2 direction diagnostic: {key}")
                cosine = row.get("g2_cos_g0_g2")
                if cosine is None:
                    if float(row.get("g2_g2_norm", float("inf"))) > spec["options"]["g2_xi"]:
                        raise RuntimeError("Nonzero g2 requires a recorded g0/g2 cosine.")
                elif not isinstance(cosine, (int, float)) or isinstance(cosine, bool) or not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
                    raise RuntimeError("Invalid G2 g0/g2 cosine.")
        elif row.get("g2_final_direction") == "g0_fallback" and row.get("g2_fallback_reason"):
            if row.get("g2_ce_gradient_phases") not in (1, 2) or row.get("g2_replay_closure_calls") != row.get("g2_ce_gradient_phases") - 1:
                raise RuntimeError("Fallback gradient-phase accounting differs.")
            fallback_count += 1
        else:
            raise RuntimeError("G2 submitted an unsupported final direction.")
        for side in ("inner", "outer"):
            if row.get(f"g2_{side}_radius_mode") != spec["options"][f"g2_{side}_radius_mode"] or row.get(f"g2_{side}_radius") != spec["options"][f"g2_{side}_radius"]:
                raise RuntimeError("Recorded probe settings do not match the supplied configuration.")
            theoretical = row.get(f"g2_{side}_radius_theoretical")
            expected_radius = row.get("g2_h_ref") if spec["options"][f"g2_{side}_radius_mode"] == "sgd_step" else spec["options"][f"g2_{side}_radius"]
            if not math.isclose(float(theoretical), float(expected_radius), rel_tol=1e-12, abs_tol=1e-15):
                raise RuntimeError("A theoretical probe radius does not match its configured rule.")
        for key in ("g2_actual_update_norm", "g2_h_ref", "g2_g0_norm", "probe_loss"):
            value = row.get(key)
            if key == "probe_loss" and value is None and row.get("g2_ce_gradient_phases") == 1 and row.get("g2_fallback_reason"):
                # A clean-gradient fallback performs no replay; no probe loss
                # exists. Other rows must carry a finite observed probe loss.
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise RuntimeError(f"G2 diagnostics are nonfinite or missing: {key}")
        if row.get("g2_lr") != spec["options"]["learning_rate"] or not math.isclose(
                row["g2_h_ref"], spec["options"]["learning_rate"] * row["g2_g0_norm"], rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("The virtual SGD probe scale differs from eta*||g0||.")
    if fallback_count == len(trace):
        raise RuntimeError("Every step fell back to g0; the smoke cannot validate G2.")
    trace_hash = sha256(trace_file)
    runtime_checks = verify_runtime_audit(spec, trace_hash, fallback_count)
    adapter = output / "adapter"
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (config.get("r") != 8 or config.get("r_sum") != spec["expected_saved_rank"] or config.get("lora_alpha") != 32
            or config.get("lora_dropout") != .1 or set(config.get("target_modules", [])) != {"q_proj", "v_proj"}):
        raise RuntimeError("Saved adapter geometry or cumulative rank does not match the four-stage protocol.")
    weight = next((adapter / name for name in ("adapter_model.bin", "adapter_model.safetensors") if (adapter / name).is_file()), None)
    if weight is None:
        raise RuntimeError("No saved adapter checkpoint was found.")
    adapter_checks = verify_adapter_tensors(spec, weight)
    if (adapter / "ewclora_fisher.pt").exists():
        raise RuntimeError("The pure G2 variant unexpectedly exported Fisher.")
    predictions = output / "predict_eval_predictions.jsonl"
    counts = {}
    seen_ids = set()
    with predictions.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            prediction = json.loads(line)
            dataset = prediction.get("Dataset")
            identity = (dataset, str(prediction.get("Instance", {}).get("id")))
            if dataset not in spec["seen_tasks"] or identity in seen_ids or "Prediction" not in prediction:
                raise RuntimeError("Prediction identities, dataset coverage or schema differ.")
            seen_ids.add(identity)
            counts[dataset] = counts.get(dataset, 0) + 1
    sample_count = sum(counts.values())
    if sample_count != spec["expected_test_samples"]:
        raise RuntimeError(f"Prediction population differs: {sample_count} != {spec['expected_test_samples']}")
    if not spec["smoke"] and counts != {task: 7600 for task in spec["seen_tasks"]}:
        raise RuntimeError("Formal evaluation must use all 7,600 examples per seen task.")
    results_file = output / "predict_results.json"
    results = json.loads(results_file.read_text())
    if not results or any(isinstance(v, float) and not math.isfinite(v) for v in results.values()):
        raise RuntimeError("Prediction results are empty or nonfinite.")
    receipt = {"status": "pass", "method": "Current-G2-Llama", "seed": spec["seed"], "stage": spec["stage"],
               "order": spec["order"], "model_size": spec["model_size"], "task": spec["task"],
               "smoke": spec["smoke"], "global_step": state["global_step"], "saved_rank": config["r_sum"],
               "prediction_samples": sample_count, "prediction_task_counts": counts,
               "optimizer_class": spec["optimizer_class"], "g2_fallback_steps": fallback_count,
               "g2_trace_sha256": trace_hash,
               "adapter_sha256": sha256(weight), "prediction_sha256": sha256(predictions),
               "prediction_results": results, **adapter_checks, **runtime_checks}
    write_json(output / "g2_script_verification.json", receipt)
    return receipt


def execute(cfg: dict, plan: dict, gpus_arg: str) -> None:
    root = Path(plan["run_root"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite or resume an existing run: {root}")
    for key in ("python_executable", "model_name_or_path", "data_dir", "instruction_file"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"Missing local input {key}: {cfg[key]}")
    validate_source_assets(REPO)
    validate_execution_profile(cfg)
    gpus, handles = claim_gpus(gpus_arg)
    try:
        root.mkdir(parents=True, exist_ok=False)
        source, hashes = freeze_source(root)
        write_json(root / "config.json", cfg)
        immutable_plan = build_plan(cfg, plan["seeds"], root,
                                    max_stages=max(spec["stage"] for spec in plan["stages"]),
                                    smoke_only=plan["formal_stages"] == 0, source=source)
        write_json(root / "plan.json", immutable_plan)
        write_json(root / "gpu_reservation.json", {"pid": os.getpid(), "gpus": gpus, "timestamp": timestamp()})
        receipts = []
        for spec in immutable_plan["stages"]:
            verify_source(source, hashes)
            if not spec["smoke"] and len([r for r in receipts if r["seed"] == spec["seed"] and r["smoke"]]) != min(2, max(s["stage"] for s in immutable_plan["stages"])):
                raise RuntimeError("Formal admission requires this seed's complete smoke chain.")
            if spec["previous_adapter"] and not Path(spec["previous_adapter"]).is_dir():
                raise RuntimeError("The preceding stage did not produce the required adapter.")
            spec_path = root / "specs" / f"s{spec['seed']}_{'smoke' if spec['smoke'] else 'formal'}_stage{spec['stage']}.json"
            write_json(spec_path, spec)
            output = Path(spec["output_dir"])
            if output.exists():
                raise FileExistsError(f"Stage output already exists: {output}")
            logfile = root / "logs" / (spec_path.stem + ".log")
            logfile.parent.mkdir(parents=True, exist_ok=True)
            print(f"Starting {spec_path.stem}: {output}", flush=True)
            with logfile.open("x") as log:
                subprocess.run(spec["command"], cwd=REPO, env=environment(source, root, gpus, spec_path),
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               pass_fds=tuple(handle.fileno() for handle in handles))
            receipts.append(verify_stage(spec))
            write_json(root / "state.json", {"status": "running", "completed_stages": receipts, "updated_at": timestamp()})
        write_json(root / "state.json", {"status": "completed", "completed_stages": receipts, "updated_at": timestamp()})
    except BaseException as exc:
        if root.exists():
            write_json(root / "failure.json", {"status": "failed", "exception": str(exc), "timestamp": timestamp()})
        raise
    finally:
        for handle in handles:
            handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, choices=SEEDS)
    seeds.add_argument("--seeds", type=int, nargs="+", choices=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", help="Exactly four physical GPU indices or UUIDs; required for a real run.")
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--max-stages", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--smoke", action="store_true", help="Only run the bounded smoke chain, without formal training.")
    for side in ("inner", "outer"):
        parser.add_argument(f"--{side}-radius-mode", choices=("sgd_step", "fixed_l2"))
        parser.add_argument(f"--{side}-radius", type=float)
    args = parser.parse_args(argv)
    cfg = load_config(args.config.resolve())
    if args.python_executable:
        cfg["python_executable"] = str(Path(args.python_executable).resolve())
    for side in ("inner", "outer"):
        mode, radius = getattr(args, f"{side}_radius_mode"), getattr(args, f"{side}_radius")
        if mode is not None:
            cfg["training"][f"g2_{side}_radius_mode"] = mode
        if radius is not None:
            cfg["training"][f"g2_{side}_radius"] = radius
        elif mode == "sgd_step":
            cfg["training"][f"g2_{side}_radius"] = None
    validate_radii(cfg["training"])
    selected_seeds = args.seeds or ([args.seed] if args.seed is not None else [cfg["default_seed"]])
    run_root = (args.run_root or REPO / "experiment_queue" / f"g2_llama_{cfg['name']}_{timestamp()}").resolve()
    plan = build_plan(cfg, selected_seeds, run_root, max_stages=args.max_stages, smoke_only=args.smoke)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if not args.gpus:
        parser.error("Real execution requires --gpus with exactly four distinct idle GPUs.")
    try:
        execute(cfg, plan, args.gpus)
    except GPUUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
