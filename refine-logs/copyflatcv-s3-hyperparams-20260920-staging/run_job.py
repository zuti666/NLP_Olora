"""Run one queue-owned job, recording immutable provenance and final status."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import traceback

from common import (HERE, assert_host, exclusive_json, inspect_metrics, load_suite,
                    now, proc_identity, query_gpus, resolve, sha256, verify_config,
                    verify_source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--uuid", required=True)
    args = parser.parse_args()
    suite = load_suite()
    job = next(job for job in suite["jobs"] if job["id"] == args.job)
    run = HERE / "runtime" / "jobs" / args.job
    record = {"id": args.job, "host": socket.gethostname(), "gpu": args.gpu,
              "gpu_uuid": args.uuid, "time": now(), **proc_identity(os.getpid())}
    try:
        assert_host(suite)
        if args.gpu not in suite["allowed_gpus"]:
            raise RuntimeError("Unauthorized GPU")
        if resolve(sys.executable) != resolve(suite["python"]):
            raise RuntimeError("Python interpreter mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.uuid:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly the assigned GPU UUID")
        gpu = query_gpus().get(args.gpu)
        if not gpu or gpu["uuid"] != args.uuid:
            raise RuntimeError("Physical GPU UUID mismatch")
        provenance = verify_source(suite)
        config_path, config, output = verify_config(job)
        record.update(provenance)
        record.update({"config": str(config_path), "config_sha256": sha256(config_path),
                       "suite_sha256": sha256(HERE / "suite.json"),
                       "runtime_sha256": {name: sha256(HERE / name) for name in
                                          ("common.py", "run_job.py", "launch_queue.py", "collect_results.py")},
                       "python": sys.executable, "python_version": platform.python_version(),
                       "environment": {key: os.environ.get(key) for key in suite.get("environment", {})}})
        with open(config_path, "rb") as src, open(run / "config.json", "xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(run / "config.json") != job["config_sha256"]:
            raise RuntimeError("Copied config hash mismatch")
        output.mkdir(parents=True, exist_ok=False)
        source = resolve(suite["source_root"])
        os.chdir(source)
        sys.path.insert(0, str(source))
        import torch
        torch.set_num_threads(8)
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"Expected exactly one visible GPU; found {torch.cuda.device_count()}")
        record.update({"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version(),
                       "visible_device_count": torch.cuda.device_count(),
                       "gpu_name": torch.cuda.get_device_name(0),
                       "native_flags_before_main": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                                                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                       "status": "started"})
        exclusive_json(run / "started.json", record)
        print(json.dumps({"runner_started": args.job, "gpu": args.gpu,
                          "uuid": args.uuid, "config_sha256": record["config_sha256"]}), flush=True)
        from src.main import main as train
        train(["--config", str(run / "config.json")])
        sys.stdout.flush()
        sys.stderr.flush()
        ending_source = verify_source(suite)
        if ending_source["source_manifest_sha256"] != provenance["source_manifest_sha256"]:
            raise RuntimeError("Source manifest changed during training")
        result = inspect_metrics(suite, job)
        exclusive_json(run / "completed.json", {**record, "status": "completed", "ended_at": now(),
                                                "result": result})
        print(json.dumps({"runner_completed": args.job, "result": result}, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        record.update({"status": "failed", "ended_at": now(), "error": repr(exc),
                       "traceback": traceback.format_exc()})
        if not (run / "failed.json").exists():
            exclusive_json(run / "failed.json", record)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
