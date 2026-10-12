#!/usr/bin/env python3
"""Pinned single-lane SAM recovery queue with full-data validation budgets."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from common import atomic_json, now, proc_identity, query_gpus, sha256, verify_source, assert_host


def checkpoint_epochs(epochs):
    return [0] + [i for i in range(1, epochs + 1) if i % 10 == 0 or i == epochs]


def inspect_completed(suite, job, rc):
    errors = []
    result = {}
    if rc != 0:
        errors.append("exit_code=" + str(rc))
    try:
        output = Path(job["output"])
        result = json.loads((output / "final_summary.json").read_text())
        n = job["expected_epochs"]
        if result.get("status") != "completed" or result.get("completed_epochs") != n:
            errors.append("completion_or_epoch_count")
        if result.get("trainer_audit_version") != 2 or result.get("trainer_sha256") != suite["trainer_core_sha256"]:
            errors.append("v2_identity")
        if result.get("seed") != 2 or result.get("direction") != "true_sam" or result.get("r") != 1.0:
            errors.append("arm_identity")
        if result.get("task") != 0 or result.get("trained_tasks") != [0] or not result.get("task_check", {}).get("passed"):
            errors.append("task_identity")
        if result.get("smoke") != job["is_validation"]:
            errors.append("validation_budget_identity")
        if not result.get("finite_check", {}).get("passed"):
            errors.append("finite_check")
        expected = checkpoint_epochs(n)
        if result.get("checkpoint_epochs") != expected:
            errors.append("checkpoint_epochs")
        for epoch in expected:
            for name in ("lora_w_a_0.pt", "lora_w_b_0.pt", "fc_state_0.pt", "lora_meta_0.json", "manifest.json", "training_state.pt"):
                if not (output / "checkpoints" / ("epoch_%03d" % epoch) / name).is_file():
                    errors.append("missing_checkpoint:%d:%s" % (epoch, name))
        count = result.get("counters", {})
        if count.get("training_samples") != 1259 * n or count.get("training_batches") != 10 * n:
            errors.append("full_training_sample_count")
        if count.get("training_ce_gradient_passes") != 20 * n:
            errors.append("true_SAM_training_pass_count")
        if count.get("isolated_diagnostic_ce_gradient_passes") != 5 * n or count.get("independent_sam_probe_gradient_passes") != n:
            errors.append("isolated_diagnostic_pass_count")
        rows = json.loads((output / "metrics.json").read_text())["epochs"]
        if len(rows) != n:
            errors.append("metric_epoch_count")
        for e, row in enumerate(rows, start=1):
            if row.get("epoch") != e or row.get("train_samples") != 1259 or row.get("eval_samples") != 307 or row.get("batches") != 10:
                errors.append("per_epoch_full_sample_count:%d" % e)
            for key in ("eval_ce", "eval_accuracy", "train_ce", "train_accuracy", "eval_macro_accuracy"):
                if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                    errors.append("nonfinite_metric:%d:%s" % (e, key))
            expected_lr = .004 + (.04 - .004) * (1 + math.cos(math.pi * (e - 1) / 50)) / 2
            lr_values = row.get("lr_used")
            if not isinstance(lr_values, list) or not lr_values or any(not isinstance(lr, (int, float)) or not math.isfinite(lr) or abs(lr - expected_lr) > 1e-10 for lr in lr_values):
                errors.append("not_original_50epoch_schedule:%d" % e)
        identity = json.loads((output / "identity.json").read_text())
        reference_identity = json.loads(Path(suite["reference_identity"]).read_text())
        for key in ("initial_model_sha256", "frozen_origin_sha256"):
            if identity.get(key) != reference_identity.get(key):
                errors.append("initial_identity:" + key)
        if identity.get("scheduler_initial", {}).get("T_max") != 50:
            errors.append("scheduler_Tmax")
        old_rows = json.loads(Path(suite["reference_metrics"]).read_text())["epochs"]
        for row, old in zip(rows, old_rows):
            if row.get("sampler_order_sha256") != old.get("sampler_order_sha256"):
                errors.append("sampler_order:" + str(row["epoch"]))
            for key in ("input_sha256", "target_sha256"):
                if row.get("first_batch", {}).get(key) != old.get("first_batch", {}).get(key):
                    errors.append("first_batch:" + str(row["epoch"]) + ":" + key)
        geometry = json.loads((output / "geometry.json").read_text())["records"]
        if len(geometry) != n or [rec.get("epoch") for rec in geometry] != list(range(1, n + 1)):
            errors.append("geometry_count")
        for rec in geometry:
            base = rec.get("base_sgd_audit", {})
            own = rec.get("self_sam_audit", {})
            if not base.get("passed") or not base.get("sam_saved_copy_restored_exactly"):
                errors.append("base_SAM_audit:" + str(rec["epoch"]))
            if not own.get("probe_parameter_check", {}).get("all_tensors_exact") or not own.get("actual_probe_gradient_check", {}).get("passed"):
                errors.append("self_SAM_probe_audit:" + str(rec["epoch"]))
            cross = rec.get("cross_algorithm_comparison", {})
            value = cross.get("sam_g1_vs_isolated_gam_g1_relative_error")
            if value is None or not math.isfinite(value) or cross.get("used_as_correctness_gate") is not False:
                errors.append("cross_algorithm_measurement:" + str(rec["epoch"]))
    except Exception as exc:
        errors.append(repr(exc))
    return errors, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    path = Path(args.suite).resolve()
    suite = json.loads(path.read_text())
    pinned_suite = sha256(path)
    runtime = Path(suite["runtime"])
    def validate():
        assert_host(suite)
        if sha256(path) != pinned_suite:
            raise RuntimeError("Suite changed")
        verify_source(suite)
        for filename, expected in suite["file_pins"].items():
            if sha256(filename) != expected:
                raise RuntimeError("Pinned file changed: " + filename)
        if suite.get("final_command"):
            raise RuntimeError("No final collector is authorized in this queue")
        if not set(suite["allowed_gpus"]) <= {5, 6} or len(suite["jobs"]) != 1:
            raise RuntimeError("Recovery queue is restricted to one job on GPU5/6")
        for job in suite["jobs"]:
            if not Path(job["output"]).is_relative_to(Path(suite["campaign"])):
                raise RuntimeError("Output escapes new recovery campaign")
            cfg = json.loads(Path(job["config"]).read_text())
            if cfg["seed"] != [2] or cfg["stage1_direction"] != "true_sam" or cfg["max_train_tasks"] != 1:
                raise RuntimeError("Wrong task0 SAM protocol")
            if job["expected_epochs"] not in (14, 50):
                raise RuntimeError("Unsupported validation/formal budget")
    validate()
    if args.validate_only:
        print(json.dumps({"status": "passed", "suite_sha256": pinned_suite,
                          "checkpoints": checkpoint_epochs(suite["jobs"][0]["expected_epochs"])}))
        return
    runtime.mkdir(parents=True, exist_ok=True)
    lock = open(runtime / "controller.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (runtime / "controller.json").exists():
        raise RuntimeError("Runtime already used")
    atomic_json(runtime / "controller.json", dict(proc_identity(os.getpid()), time=now(), suite_sha256=pinned_suite))
    job = suite["jobs"][0]
    if Path(job["output"]).exists():
        raise RuntimeError("Output already exists")
    leases = Path("/tmp/olora_gamfs_gpu_locks")
    leases.mkdir(exist_ok=True)
    process = None
    try:
        while process is None:
            atomic_json(runtime / "state.json", {"status": "waiting_for_idle_gpu", "time": now(), "pending": [job["id"]]})
            for gpu, info in query_gpus().items():
                if gpu not in suite["allowed_gpus"] or info["memory_used_mb"] >= 500 or info["utilization_percent"] >= 5:
                    continue
                lease = open(leases / ("gpu%d.lock" % gpu), "a+")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    check = query_gpus().get(gpu)
                    if not check or check["uuid"] != info["uuid"] or check["memory_used_mb"] >= 500 or check["utilization_percent"] >= 5:
                        continue
                    validate()
                    directory = runtime / "jobs" / job["id"]
                    directory.mkdir(parents=True, exist_ok=False)
                    environment = dict(os.environ, **suite["environment"])
                    environment["CUDA_VISIBLE_DEVICES"] = info["uuid"]
                    command = [suite["python"], suite["script"], "--source", suite["source_root"],
                               "--config", job["config"], "--output-dir", job["output"]] + job.get("extra_args", [])
                    log = open(directory / "console.log", "x")
                    process = subprocess.Popen(command, cwd=suite["source_root"], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    receipt = dict(proc_identity(process.pid), time=now(), physical_gpu=gpu,
                        gpu_uuid=info["uuid"], command=command, job=job, source=suite["source_root"],
                        trainer_v2_sha256=suite["trainer_core_sha256"], suite_sha256=pinned_suite)
                    atomic_json(directory / "started.json", receipt)
                    atomic_json(runtime / "state.json", {"status": "running", "time": now(), "running": [receipt]})
                    print(json.dumps({"event": "started", **receipt}), flush=True)
                    break
                finally:
                    lease.close()
            if process is None:
                time.sleep(10)
        while process.poll() is None:
            time.sleep(10)
        log.close()
        errors, result = inspect_completed(suite, job, process.returncode)
        validate()
        completion = {"time": now(), "status": "failed" if errors else "completed", "errors": errors,
                      "returncode": process.returncode, "id": job["id"], "output": job["output"],
                      "job": job, "summary": result, "started": receipt}
        atomic_json(directory / ("failed.json" if errors else "completed.json"), completion)
        atomic_json(runtime / "state.json", completion)
        print(json.dumps({"event": "finished", "status": completion["status"], "errors": errors}), flush=True)
        if errors:
            raise SystemExit(1)
    except BaseException as exc:
        atomic_json(runtime / "exception.json", {"time": now(), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
