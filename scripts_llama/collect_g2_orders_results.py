#!/usr/bin/env python3
"""Collect validated pure-G2 Orders without treating smoke/partial runs as FAA.

Exact match repeats src/compute_metrics.py's lowercase, ASCII punctuation and
whitespace normalization. AAA averages four stage seen-task means; BWT averages
the final-minus-learning-stage accuracy on the first three tasks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import string

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "experiment_queue/nlp_multiseed_bf16_b64_ddp4_g0-3_20260922T150750Z/history_reuse_20260926/final_per_seed.csv"
TASK_RELATIVE = {"dbpedia": "TC/dbpedia", "amazon": "SC/amazon", "yahoo": "TC/yahoo", "agnews": "TC/agnews"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Prediction and label must be strings.")
    return " ".join("".join(c for c in text.lower() if c not in string.punctuation).split())


def prediction_metrics(path: Path, seen_tasks: list[str], *, expected_per_task: int = 7600,
                       gold: dict | None = None) -> dict:
    counts, correct, identities, labels = {}, {}, set(), {}
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task, instance = row.get("Dataset"), row.get("Instance")
            if task not in seen_tasks or not isinstance(instance, dict) or instance.get("id") is None:
                raise ValueError(f"Unknown task or missing prediction identity on line {number}.")
            raw_id = str(instance["id"])
            if not raw_id.isdecimal() or raw_id != str(int(raw_id)) or int(raw_id) >= expected_per_task:
                raise ValueError(f"Prediction identity is not a canonical test-set index: {(task, raw_id)}")
            identity = (task, raw_id)
            if identity in identities:
                raise ValueError(f"Duplicate prediction identity: {identity}")
            identities.add(identity)
            label = instance.get("label")
            if gold is not None:
                if int(raw_id) >= len(gold[task]):
                    raise ValueError(f"Prediction identity is absent from the frozen test set: {identity}")
                expected_label = gold[task][int(raw_id)]["label"]
                if label != expected_label:
                    raise ValueError(f"Prediction label differs from the frozen test set: {identity}")
            labels[identity] = normalize(label)
            counts[task] = counts.get(task, 0) + 1
            correct[task] = correct.get(task, 0) + int(normalize(row.get("Prediction")) == labels[identity])
    if counts != {task: expected_per_task for task in seen_tasks}:
        raise ValueError(f"Prediction population differs: {counts}; expected {expected_per_task} per seen task.")
    if identities != {(task, str(index)) for task in seen_tasks for index in range(expected_per_task)}:
        raise ValueError("Prediction identities do not exactly cover the canonical frozen test-set indices.")
    accuracy = {task: 100 * correct[task] / counts[task] for task in seen_tasks}
    return {"counts": counts, "correct": correct, "accuracy": accuracy,
            "seen_task_accuracy": sum(accuracy.values()) / len(accuracy),
            "sample_accuracy": 100 * sum(correct.values()) / sum(counts.values()),
            "prediction_sha256": sha256(path), "labels": labels}


def continual_metrics(tasks: list[str], stages: dict[int, dict]) -> dict:
    if set(stages) != {1, 2, 3, 4} or len(tasks) != 4:
        raise ValueError("Final FAA/AAA/BWT require all four stage evaluations.")
    for stage in range(1, 5):
        if set(stages[stage]["accuracy"]) != set(tasks[:stage]):
            raise ValueError("The triangular evaluation matrix has incorrect task coverage.")
    final = stages[4]["accuracy"]
    return {"FAA": sum(final.values()) / 4,
            "AAA": sum(stages[stage]["seen_task_accuracy"] for stage in range(1, 5)) / 4,
            "BWT": sum(final[task] - stages[index + 1]["accuracy"][task] for index, task in enumerate(tasks[:3])) / 3,
            "test_samples": sum(stages[4]["counts"].values()),
            **{f"{task}_accuracy": final[task] for task in tasks}}


def read_baseline(path: Path, *, decoder_pipeline_version: str | None = None) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    matched = {}
    for row in rows:
        if row["method"] != "IncLoRA":
            continue
        if decoder_pipeline_version is not None and row.get("decoder_pipeline_version") != decoder_pipeline_version:
            raise ValueError("The paired baseline CSV does not declare the matching corrected decoder pipeline.")
        key = (row["model"], int(row["order"]), int(row["seed"]))
        if key in matched:
            raise ValueError(f"Duplicate paired IncLoRA baseline: {key}")
        if not math.isfinite(float(row["faa"])):
            raise ValueError(f"Paired IncLoRA FAA is nonfinite: {key}")
        matched[key] = row
    return matched


def verify_pinned_input(manifest: dict, path: Path) -> None:
    expected = manifest.get("input_sha256", {}).get(str(path.resolve()))
    if not isinstance(expected, dict) or not path.is_file() or path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
        raise ValueError(f"Frozen collector input changed or is unpinned: {path}")


def stage_cost_metrics(spec: dict, receipt: dict) -> dict:
    output = Path(spec["output_dir"])
    trace = output / "g2_step_trace.jsonl"
    trace_hash = sha256(trace)
    if trace_hash != receipt.get("g2_trace_sha256"):
        raise ValueError("Validated G2 trace changed before result collection.")
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if len(rows) != spec["expected_updates"] or [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("G2 cost diagnostics do not cover each completed update exactly once.")
    phases_total = replays_total = fallback_count = 0
    for row in rows:
        phases, replays = row.get("g2_ce_gradient_phases"), row.get("g2_replay_closure_calls")
        if isinstance(phases, bool) or not isinstance(phases, int) or phases not in (1, 2, 3) or isinstance(replays, bool) or not isinstance(replays, int) or replays != phases - 1:
            raise ValueError("G2 phase/replay cost diagnostics are invalid or nonfinite.")
        if row.get("g2_final_direction") == "g0_fallback" and row.get("g2_fallback_reason") and phases in (1, 2):
            fallback_count += 1
        elif row.get("g2_final_direction") != "g2" or phases != 3 or row.get("g2_fallback_reason") is not None:
            raise ValueError("G2 final direction and gradient-phase counts differ.")
        phases_total += phases
        replays_total += replays
    if fallback_count != receipt.get("g2_fallback_steps"):
        raise ValueError("G2 fallback count differs from its validated receipt.")
    timings = {}
    for file, key in (("train_results.json", "train_runtime"), ("predict_results.json", "predict_runtime")):
        value = json.loads((output / file).read_text()).get(key) if (output / file).is_file() else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"G2 timing diagnostic is invalid or nonfinite: {key}")
        timings[key + "_seconds"] = value if value is not None else ""
    return {"global_updates": receipt["global_step"], "ce_gradient_phases_total": phases_total,
            "replay_closure_calls_total": replays_total, "fallback_steps": fallback_count,
            "fallback_fraction": fallback_count / len(rows), "trace_path": str(trace), "trace_sha256": trace_hash, **timings}


def collect(root: Path, *, baseline_path: Path | None = None) -> dict:
    if sha256(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("The prepared collector manifest changed.")
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "runtime/queue_state.json").read_text())
    if manifest.get("schema") != "CurrentG2SixOrderQueue_v1":
        raise ValueError("Unrecognized G2 queue manifest.")
    repository = Path(manifest["repository"])
    versions = {job["config"].get("decoder_pipeline_version") for job in manifest["jobs"]}
    if len(versions) != 1:
        raise ValueError("A result table cannot mix decoder token-boundary protocols.")
    pipeline_version = next(iter(versions))
    if pipeline_version == "decoder_token_ids_v2" and baseline_path is None:
        # Historical token-boundary scores remain readable but cannot be a
        # paired comparator for corrected-pipeline training.
        baseline = {}
    else:
        if baseline_path is None:
            baseline_path = repository / BASELINE.relative_to(REPO)
        verify_pinned_input(manifest, baseline_path)
        baseline = read_baseline(baseline_path, decoder_pipeline_version=pipeline_version)
    completed = {row["key"]: row for row in state["completed_stages"]}
    order_rows, stage_rows = [], []
    for job in manifest["jobs"]:
        tasks = job["config"]["tasks"]
        for task in tasks:
            verify_pinned_input(manifest, Path(job["config"]["data_dir"]) / TASK_RELATIVE[task] / "test.json")
        gold = {task: json.loads((Path(job["config"]["data_dir"]) / TASK_RELATIVE[task] / "test.json").read_text()) for task in tasks}
        if any(len(rows) != 7600 for rows in gold.values()):
            raise ValueError("The frozen evaluation sets must each contain 7,600 examples.")
        stages, labels_by_task = {}, {}
        stage_status = []
        for spec in job["plan"]["stages"]:
            if spec["smoke"]:
                continue
            key = f"{job['id']}:s{spec['seed']}:formal:{spec['stage']}"
            if key not in completed:
                stage_status.append("running" if state.get("active_stage") == key else "pending")
                continue
            receipt_file = Path(spec["output_dir"]) / "g2_script_verification.json"
            receipt = json.loads(receipt_file.read_text())
            if receipt.get("status") != "pass" or receipt.get("smoke") is not False or receipt.get("global_step") != spec["expected_updates"]:
                raise ValueError(f"Formal stage has no matching verification receipt: {key}")
            predictions = Path(spec["output_dir"]) / "predict_eval_predictions.jsonl"
            metrics = prediction_metrics(predictions, tasks[:spec["stage"]], gold=gold)
            costs = stage_cost_metrics(spec, receipt)
            if metrics["prediction_sha256"] != receipt.get("prediction_sha256"):
                raise ValueError(f"Validated prediction output changed: {key}")
            for task in tasks[:spec["stage"]]:
                labels = {identity: label for identity, label in metrics["labels"].items() if identity[0] == task}
                if task in labels_by_task and labels_by_task[task] != labels:
                    raise ValueError("Triangular evaluations used different identities or labels.")
                labels_by_task[task] = labels
            stages[spec["stage"]] = metrics
            stage_status.append("completed")
            stage_rows.append({"model": f"Llama-3.2-{job['model_size']}", "method": "IncLoRA-G2 (HF AdamW)",
                "order": job["order"], "seed": job["seed"], "stage": spec["stage"], "learned_task": spec["task"],
                "seen_task_accuracy": metrics["seen_task_accuracy"], "test_samples": sum(metrics["counts"].values()),
                **costs,
                **{f"{task}_accuracy": metrics["accuracy"].get(task, "") for task in TASK_RELATIVE}, "output_dir": spec["output_dir"]})
        row = {"model": f"Llama-3.2-{job['model_size']}", "method": "IncLoRA-G2 (HF AdamW)",
               "order": job["order"], "seed": job["seed"], "status": "completed" if len(stages) == 4 else "partial" if stages else "pending",
               "formal_stages_completed": len(stages), "FAA": "", "AAA": "", "BWT": "", "test_samples": "",
               **{f"{task}_accuracy": "" for task in TASK_RELATIVE},
               "latest_completed_stage": max(stages) if stages else "", "latest_seen_task_accuracy": stages[max(stages)]["seen_task_accuracy"] if stages else "",
               "paired_IncLoRA_FAA": "", "delta_FAA_vs_paired_IncLoRA": "", "paired_IncLoRA_source": ""}
        paired = baseline.get((row["model"], job["order"], job["seed"]))
        if paired:
            row.update(paired_IncLoRA_FAA=float(paired["faa"]), paired_IncLoRA_source=paired["run_dir"])
        if len(stages) == 4:
            row.update(continual_metrics(tasks, stages))
            if paired:
                row["delta_FAA_vs_paired_IncLoRA"] = row["FAA"] - float(paired["faa"])
        if state.get("status") == "failed" and state.get("active_stage", "").startswith(job["id"] + ":"):
            row["status"] = "failed"
        order_rows.append(row)
    aggregate = []
    for model in ("Llama-3.2-1B", "Llama-3.2-3B"):
        rows = [row for row in order_rows if row["model"] == model and row["status"] == "completed"]
        if len(rows) == 3:
            aggregate.append({"model": model, "seed": 0, "orders": 3,
                **{metric: sum(row[metric] for row in rows) / 3 for metric in ("FAA", "AAA", "BWT")},
                "paired_IncLoRA_FAA": (sum(row["paired_IncLoRA_FAA"] for row in rows) / 3
                    if all(isinstance(row["paired_IncLoRA_FAA"], (int, float)) for row in rows) else "")})
    return {"schema": "CurrentG2SixOrderResults_v1", "queue_status": state["status"],
            "active_stage": state.get("active_stage"), "active_global_step": state.get("active_global_step"),
            "active_log": state.get("active_log"), "heartbeat_at": state.get("heartbeat_at"), "order_results": order_rows,
            "stage_results": stage_rows, "across_order_results": aggregate,
            "metric_definition": {"FAA": "Final stage mean of four task exact-match accuracies (%)",
            "AAA": "Mean of four stage seen-task means (%)", "BWT": "Mean final-minus-learning-stage accuracy on the first three tasks (percentage points)"},
            "decoder_pipeline_version": pipeline_version,
            "baseline_csv": str(baseline_path) if baseline_path is not None else None,
            "baseline_sha256": sha256(baseline_path) if baseline_path is not None else None,
            "note": ("Single seed0 only. Smoke and incomplete Orders have no final FAA/AAA/BWT; historical three-seed means are not paired controls. "
                + ("Corrected decoder_token_ids_v2: historical v1 IncLoRA is excluded; a fresh, pinned v2 baseline is required for paired deltas."
                    if pipeline_version == "decoder_token_ids_v2" else ""))}


def save_results(root: Path, result: dict) -> None:
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("orders.csv", result["order_results"]), ("stages.csv", result["stage_results"]),
                       ("across_orders.csv", result["across_order_results"])):
        destination = result_dir / name
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        temporary.replace(destination)
    destination = result_dir / "summary.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    lines = [f"Queue status: **{result['queue_status']}**. Seed **0**; pure HF-AdamW G2.", "", result["note"], "",
             "| Model | Order | Status | Stages | FAA (%) | AAA (%) | BWT (pp) | Paired IncLoRA FAA (%) | Delta FAA (pp) |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    for row in result["order_results"]:
        lines.append("| " + " | ".join(fmt(row[key]) for key in ("model", "order", "status", "formal_stages_completed", "FAA", "AAA", "BWT", "paired_IncLoRA_FAA", "delta_FAA_vs_paired_IncLoRA")) + " |")
    lines.extend(["", "FAA uses the final full 30,400-example evaluation. AAA/BWT require the full triangular matrix.",
                  ("The baseline column requires a fresh same-model/Order/seed0 corrected-pipeline CSV."
                   if result.get("decoder_pipeline_version") == "decoder_token_ids_v2" else
                   "The baseline column uses the matching model, Order and seed0 from the archived CSV."), ""])
    (result_dir / "RESULTS.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path)
    args = parser.parse_args(argv)
    result = collect(args.queue_root.resolve(), baseline_path=args.baseline_csv)
    save_results(args.queue_root.resolve(), result)
    print(json.dumps({"status": result["queue_status"], "complete_orders": sum(row["status"] == "completed" for row in result["order_results"]),
                      "complete_formal_stages": len(result["stage_results"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
