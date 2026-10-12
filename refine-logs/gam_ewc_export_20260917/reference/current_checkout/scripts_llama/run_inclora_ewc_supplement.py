#!/usr/bin/env python3
"""Supplement the missing Llama-1B EWCLoRA orders on one idle GPU.

Example (run detached for a full experiment):
  python3 scripts_llama/run_inclora_ewc_supplement.py \
    --run-dir experiment_queue/inclora_ewc_supplement_RUN --gpu auto

The default orders are 2,3. A two-task smoke precedes full training. Existing
results are never overwritten. Use --dry-run to inspect all training commands.
"""

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[1]
TASKS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    return [dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(),
                 memory_mib=int(r[3]), utilization=int(r[4]))
            for r in csv.reader(result.stdout.splitlines())]


def claim_gpu(requested):
    lock_dir = REPO / ".cache" / "gpu_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for gpu in gpu_snapshot():
        if requested != "auto" and gpu["index"] != int(requested):
            continue
        if gpu["memory_mib"] >= 500 or gpu["utilization"] > 5:
            continue
        handle = (lock_dir / (gpu["uuid"] + ".lock")).open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        # A second NVML sample narrows the race with other launchers.
        current = next(g for g in gpu_snapshot() if g["uuid"] == gpu["uuid"])
        if current["memory_mib"] >= 500 or current["utilization"] > 5:
            handle.close()
            continue
        return gpu, handle
    raise RuntimeError("No requested GPU is idle (<500 MiB and <=5% utilization).")


def training_command(args, source, stage_dir, order, stage, model_path, smoke):
    task = TASKS[order][stage - 1]
    # Legacy trainer adds EWC after dividing CE by GA. Preserve legacy
    # GA8 * lambda1 while single-GPU BS1 * GA40 preserves global batch40.
    cli_lambda = args.nominal_ewc_lambda * 8.0 / 40.0
    command = [
        args.python, str(source / "src" / "run_uie_lora.py"),
        "--seed", str(args.seed), "--data_seed", str(args.seed),
        "--do_train", "--do_predict", "--predict_with_generate",
        "--bf16", str(args.precision == "bf16").lower(), "--fp16", "false",
        "--train_method", "lora", "--lora_strategy", "ewclora",
        "--lora_dim", "8", "--optimizer_type", "adamw_hf",
        "--ewclora_lambda", str(cli_lambda), "--ewclora_gamma", "1.0",
        "--ewclora_fisher_max_batches", "2" if smoke else "0",
        "--ewclora_normalize_fisher", "false",
        "--ewclora_normalize_accumulated_fisher", "false",
        "--ewclora_fisher_damping_beta", "0.0",
        "--olora_lamda_1", "0.0", "--olora_lamda_2", "0.0",
        "--model_name_or_path", str(model_path),
        "--data_dir", str(REPO / "CL_Benchmark"),
        "--task_config_dir", str(source / "configs" / f"order{order}_configs" / task),
        "--instruction_file", str(source / "configs" / "instruction_config.json"),
        "--instruction_strategy", "single", "--output_dir", str(stage_dir),
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "40",
        "--learning_rate", "1e-3" if stage == 1 else "1e-4",
        "--num_train_epochs", "1", "--max_source_length", "512",
        "--max_target_length", "50", "--generation_max_length", "50",
        "--add_task_name", "true", "--add_dataset_name", "true",
        "--lr_scheduler_type", "constant", "--warmup_steps", "0",
        "--logging_strategy", "steps", "--logging_steps", "1" if smoke else "10",
        "--report_to", "none", "--evaluation_strategy", "no",
        "--save_strategy", "no", "--run_name", f"inclora_ewc_order{order}_stage{stage}",
    ]
    if stage > 1:
        command += ["--ewclora_fisher_path", str(model_path)]
    if smoke:
        command += ["--max_steps", "2", "--max_train_samples", "80",
                    "--max_predict_samples", "16"]
    return command


def run(args):
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and not args.dry_run:
        raise FileExistsError(f"Run directory already exists; choose a new one: {run_dir}")
    base_model = Path(args.base_model).resolve()
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"Missing local model config: {base_model}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)
    if not args.dry_run and not (Path(args.nltk_data) / "tokenizers" / "punkt").is_dir():
        raise FileNotFoundError(f"Missing NLTK punkt; set --nltk-data to a prepared data directory: {args.nltk_data}")
    source = run_dir / "source"
    plans = []
    if not args.skip_smoke:
        plans.append(("smoke", 2, 2))
    if not args.smoke_only:
        plans += [("full", order, 4) for order in args.orders]
    if args.dry_run:
        for phase, order, count in plans:
            model_path = base_model
            for stage in range(1, count + 1):
                stage_dir = run_dir / phase / f"order_{order}" / f"{stage}-{TASKS[order][stage-1]}"
                print(shlex.join(training_command(args, source, stage_dir, order,
                                                 stage, model_path, phase == "smoke")))
                model_path = stage_dir / "adapter"
        return

    gpu, gpu_lock = claim_gpu(args.gpu)
    run_dir.mkdir(parents=True)
    state = {"status": "preflight", "started_at": now(), "supervisor_pid": os.getpid(),
             "gpu": gpu, "seed": args.seed, "orders": args.orders,
             "precision": args.precision, "global_batch": 40, "per_device_batch": 1,
             "gradient_accumulation_steps": 40, "nominal_ewc_lambda": args.nominal_ewc_lambda,
             "cli_ewc_lambda": args.nominal_ewc_lambda / 5,
             "legacy_reference_ga": 8, "olora_lambdas": [0, 0], "completed": []}
    state_file = run_dir / "state.json"
    write_json(state_file, state)
    try:
        shutil.copytree(REPO / "src", source / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(REPO / "configs", source / "configs")
        # compute_metrics resolves its tokenizer relative to the source tree.
        (source / "data").symlink_to(REPO / "data", target_is_directory=True)
        shutil.copy2(__file__, source / Path(__file__).name)
        verifier = source / "verify_inclora_ewc_stage.py"
        shutil.copy2(REPO / "scripts_llama" / verifier.name, verifier)
        hash_files = (list((source / "src").rglob("*.py")) + list(source.glob("*.py"))
                      + list((source / "configs").rglob("*.json")))
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in hash_files}
        write_json(run_dir / "source_sha256.json", hashes)
        write_json(run_dir / "manifest.json", {
            **state, "base_model": str(base_model), "python": args.python,
            "training_source_snapshot": str(source),
            "comparison_notes": [
                "EWC only on fresh IncLoRA branch; OLoRA/GAM/SAM disabled.",
                "GA40 lambda0.2 preserves legacy GA8 lambda1 EWC/CE ratio.",
                "BS1 and full-data Fisher retain the legacy Fisher estimator.",
                "FP32 and single-GPU sampling differ from historical 5-GPU BF16 runs.",
            ],
        })
        env = os.environ.copy()
        for key in list(env):
            if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                       "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"} or key.startswith(("TORCHELASTIC_", "ACCELERATE_")):
                env.pop(key)
        cache = run_dir / "cache"
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu["uuid"],
            "PYTHONPATH": str(source / "src"), "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true",
            "HF_HOME": str(cache / "huggingface"),
            "HF_MODULES_CACHE": str(cache / "modules"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "TRANSFORMERS_CACHE": str(cache / "transformers"),
            "UIE_DATA_CACHE_ROOT": str(cache / "uie_data"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "NLTK_DATA": str(Path(args.nltk_data).resolve()),
        })
        for name in ["HF_HOME", "HF_MODULES_CACHE", "HF_DATASETS_CACHE",
                     "TRANSFORMERS_CACHE", "UIE_DATA_CACHE_ROOT"]:
            Path(env[name]).mkdir(parents=True, exist_ok=True)
        witness = (
            "import torch; torch.manual_seed(0); "
            "x=torch.randn(8,8,device='cuda'); y=x@x; "
            "print('WITNESS',tuple(y.shape),torch.cuda.get_device_name(),float(y.sum())); "
            "print('BF16_SUPPORTED',torch.cuda.is_bf16_supported())"
        )
        result = subprocess.run([args.python, "-c", witness], env=env, cwd=REPO,
                                check=True, text=True, capture_output=True)
        (run_dir / "preflight.log").write_text(result.stdout + result.stderr)
        if args.precision == "bf16" and "BF16_SUPPORTED True" not in result.stdout:
            raise RuntimeError("Selected GPU does not support BF16; use --precision fp32.")
        print(f"CLAIMED GPU {gpu['index']} ({gpu['uuid']}); outputs: {run_dir}", flush=True)
        (run_dir / "logs").mkdir()
        for phase, order, count in plans:
            model_path = base_model
            for stage in range(1, count + 1):
                label = f"{phase}_order{order}_{stage}-{TASKS[order][stage-1]}"
                stage_dir = run_dir / phase / f"order_{order}" / f"{stage}-{TASKS[order][stage-1]}"
                cmd = training_command(args, source, stage_dir, order, stage,
                                       model_path, phase == "smoke")
                write_json(run_dir / "commands" / f"{label}.json", cmd)
                log = run_dir / "logs" / f"{label}.log"
                print(f"START {label} at {now()}; log={log}", flush=True)
                with log.open("w") as stream:
                    child = subprocess.Popen(cmd, cwd=REPO, env=env,
                                             stdout=stream, stderr=subprocess.STDOUT)
                    state.update(status="running", phase=phase, order=order, stage=stage,
                                 worker_pid=child.pid, log=str(log), updated_at=now())
                    write_json(state_file, state)
                    code = child.wait()
                if code:
                    raise RuntimeError(f"{label} exited {code}; inspect {log}")
                verify = [args.python, str(verifier), "--current", str(stage_dir / "adapter"),
                          "--stage", str(stage), "--output", str(stage_dir / "verification.json")]
                if stage > 1:
                    verify += ["--previous", str(model_path)]
                with (run_dir / "logs" / f"{label}_verification.log").open("w") as stream:
                    subprocess.run(verify, env=env, cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                state["completed"].append(label)
                state.update(updated_at=now(), worker_pid=None)
                write_json(state_file, state)
                model_path = stage_dir / "adapter"
                print(f"VERIFIED {label} at {now()}", flush=True)
        summary = []
        for order in args.orders if not args.smoke_only else []:
            p = run_dir / "full" / f"order_{order}" / f"4-{TASKS[order][-1]}" / "predict_results.json"
            metrics = json.loads(p.read_text())
            summary.append({"order": order, "final_exact_match": metrics["predict_exact_match"],
                            "source": str(p)})
        write_json(run_dir / "final_results.json", summary)
        state.update(status="completed", finished_at=now(), worker_pid=None)
        write_json(state_file, state)
        print(f"COMPLETED {run_dir}", flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), failed_at=now())
        write_json(state_file, state)
        raise
    finally:
        gpu_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", default="auto", help="One NVML GPU index, or auto; requires idle GPU.")
    parser.add_argument("--orders", type=int, nargs="+", choices=[1, 2, 3], default=[2, 3])
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--nominal-ewc-lambda", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--python", default="/data/140-1/users/liying/conda_envs/nlora_req/bin/python")
    parser.add_argument("--base-model", default="/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B")
    parser.add_argument("--nltk-data", default=str(REPO / ".cache" / "nltk_data"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only and args.skip_smoke:
        parser.error("--smoke-only and --skip-smoke cannot be combined")
    if not (args.nominal_ewc_lambda > 0 and args.nominal_ewc_lambda < float("inf")):
        parser.error("--nominal-ewc-lambda must be positive and finite")
    if args.gpu != "auto" and not args.gpu.isdigit():
        parser.error("--gpu must be auto or a non-negative GPU index")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
