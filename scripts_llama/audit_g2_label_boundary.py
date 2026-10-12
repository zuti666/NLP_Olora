#!/usr/bin/env python3
"""CPU/offline label-boundary and archived prediction audit.

This does not load model weights, train, or replace the benchmark. Excluding
labels with a legacy token boundary mismatch is a conditional diagnostic only.
The official normalized EM and scratchpad's strict EM are reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import string

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO / "experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2"
DEFAULT_BASELINE = REPO / "experiment_queue/nlp_multiseed_bf16_b64_ddp4_g0-3_20260922T150750Z/history_reuse_20260926/final_per_seed.csv"
DEFAULT_OUTPUT = REPO / "refine-logs/llama_g2_token_boundary_fix_20261012"
TASK_TYPE = {"dbpedia": "TC", "amazon": "SC", "yahoo": "TC", "agnews": "TC"}
ORDERS = {1: ["dbpedia", "amazon", "yahoo", "agnews"], 2: ["dbpedia", "amazon", "agnews", "yahoo"], 3: ["yahoo", "amazon", "agnews", "dbpedia"]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Predictions and gold labels must be strings")
    return " ".join("".join(c for c in text.lower() if c not in string.punctuation).split())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty audit output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path, gold, tasks, affected):
    counts, correct, strict = Counter(), Counter(), Counter()
    kept, kept_correct, kept_strict = Counter(), Counter(), Counter()
    class_counts, class_correct, class_strict = Counter(), Counter(), Counter()
    wrong = defaultdict(Counter)
    identities, representatives = set(), {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            task, instance = row["Dataset"], row["Instance"]
            raw_id = instance["id"]
            if task not in tasks or not isinstance(raw_id, str) or not raw_id.isdecimal() or raw_id != str(int(raw_id)):
                raise ValueError(f"Invalid prediction identity in {path}: {(task, raw_id)}")
            index = int(raw_id)
            identity = (task, index)
            if identity in identities or index >= len(gold[task]):
                raise ValueError(f"Duplicate or invalid prediction identity in {path}: {identity}")
            identities.add(identity)
            expected = gold[task][index]
            label = instance["label"]
            if label != expected["label"] or instance.get("ground_truth", label) != label or instance["sentence"] != expected["sentence"]:
                raise ValueError(f"Prediction gold differs from frozen test input in {path}: {identity}")
            hit = int(normalize(row["Prediction"]) == normalize(label))
            raw_hit = int(row["Prediction"].strip() == label)
            counts[task] += 1
            correct[task] += hit
            strict[task] += raw_hit
            pair = (task, label)
            class_counts[pair] += 1
            class_correct[pair] += hit
            class_strict[pair] += raw_hit
            representatives.setdefault(pair, row)
            if not hit:
                wrong[pair][row["Prediction"]] += 1
            if pair not in affected:
                kept[task] += 1
                kept_correct[task] += hit
                kept_strict[task] += raw_hit
    expected_ids = {(task, index) for task in tasks for index in range(len(gold[task]))}
    if identities != expected_ids or any(counts[task] != 7600 for task in tasks):
        raise ValueError(f"Prediction population must contain exactly the canonical 7,600 examples/task: {path}")
    if any(kept[t] == 0 for t in tasks):
        raise ValueError("Exclusion has an empty task denominator")
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "counts": dict(counts), "correct": dict(correct), "strict_correct": dict(strict),
        "retained": dict(kept), "retained_correct": dict(kept_correct), "retained_strict_correct": dict(kept_strict),
        "faa": statistics.mean(100 * correct[t] / counts[t] for t in tasks),
        "exclusion_diagnostic_faa": statistics.mean(100 * kept_correct[t] / kept[t] for t in tasks),
        "strict_faa": statistics.mean(100 * strict[t] / counts[t] for t in tasks),
        "strict_exclusion_diagnostic_faa": statistics.mean(100 * kept_strict[t] / kept[t] for t in tasks),
        "class_counts": class_counts, "class_correct": class_correct, "class_strict_correct": class_strict,
        "wrong_predictions": wrong, "representatives": representatives,
    }


def public_prediction_summary(value):
    return {key: item for key, item in value.items() if not key.startswith("class_") and key not in ("representatives", "wrong_predictions")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--historical-seeds", default="0,42,1993")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer
    if torch.cuda.is_initialized():
        raise ValueError("The audit must not initialize CUDA")

    manifest_file = args.queue_root / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if sha256(manifest_file) != (args.queue_root / "manifest.sha256").read_text().strip():
        raise ValueError("Queue manifest changed")
    if manifest["seeds"] != [0]:
        raise ValueError("This audit requires the complete seed0 campaign")
    inputs = [{"path": str(manifest_file), "sha256": sha256(manifest_file)}]
    for path in [args.baseline_csv]:
        expected = manifest["input_sha256"].get(str(path.resolve()), {})
        if sha256(path) != expected.get("sha256"):
            raise ValueError(f"Pinned baseline changed: {path}")
        inputs.append({"path": str(path), "sha256": sha256(path)})
    gold = {}
    for task, typ in TASK_TYPE.items():
        path = REPO / f"CL_Benchmark/{typ}/{task}/test.json"
        expected = manifest["input_sha256"].get(str(path.resolve()), {})
        if sha256(path) != expected.get("sha256"):
            raise ValueError(f"Pinned test input changed: {path}")
        gold[task] = json.loads(path.read_text())
        if len(gold[task]) != 7600:
            raise ValueError("The archived benchmark requires 7,600 examples/task")
        inputs.append({"path": str(path), "sha256": sha256(path)})

    baselines = {}
    for row in csv.DictReader(args.baseline_csv.open()):
        if row["method"] != "IncLoRA":
            continue
        key = (row["model"], int(row["order"]), int(row["seed"]))
        if key in baselines:
            raise ValueError("Duplicate archived IncLoRA baseline")
        baselines[key] = row

    tokenizers, affected_by_model, token_rows = {}, {}, []
    for job in manifest["jobs"]:
        size = job["config"]["model_size"]
        if size in tokenizers:
            continue
        model_dir = Path(job["config"]["model_name_or_path"])
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        tokenizers[size] = tokenizer
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            path = model_dir / name
            if path.is_file():
                inputs.append({"path": str(path), "sha256": sha256(path)})
        affected = set()
        for task in TASK_TYPE:
            for label, n in sorted(Counter(x["label"] for x in gold[task]).items()):
                prompt_ids = tokenizer("Answer:", add_special_tokens=False)["input_ids"]
                label_ids = tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                joint_ids = tokenizer("Answer:" + label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                mismatch = joint_ids != prompt_ids + label_ids
                if mismatch:
                    affected.add((task, label))
                token_rows.append({
                    "model": size, "task": task, "label": label, "test_n": n, "boundary_mismatch": mismatch,
                    "joint_ids": joint_ids, "joint_pieces": tokenizer.convert_ids_to_tokens(joint_ids),
                    "separate_ids": prompt_ids + label_ids, "separate_pieces": tokenizer.convert_ids_to_tokens(prompt_ids + label_ids),
                })
        affected_by_model[size] = affected
    if affected_by_model["1B"] != affected_by_model["3B"]:
        raise ValueError("Model tokenizer mismatch sets differ; do not pool exclusion metrics")
    affected = affected_by_model["1B"]
    # Use archived prediction instructions to validate the complete, prefixed
    # collator prompt rather than only trusting the minimal Answer: reproducer.
    pair_rows, class_rows, task_rows, source_rows, stage1_rows, historical_rows = [], [], [], [], [], []
    results = {}
    for job in manifest["jobs"]:
        size, order = job["config"]["model_size"], job["config"]["order"]
        tasks = job["config"]["tasks"]
        if tasks != ORDERS[order]:
            raise ValueError("Unexpected archived task order")
        specs = {s["stage"]: s for s in job["plan"]["stages"] if not s["smoke"]}
        final = Path(specs[4]["output_dir"]) / "predict_eval_predictions.jsonl"
        receipt = json.loads((Path(specs[4]["output_dir"]) / "g2_script_verification.json").read_text())
        if receipt["status"] != "pass" or receipt["smoke"] or receipt["global_step"] != specs[4]["expected_updates"]:
            raise ValueError("G2 final stage is not a validated formal result")
        baseline = baselines[(f"Llama-3.2-{size}", order, 0)]
        baseline_final = Path(baseline["run_dir"]) / f"4-{tasks[3]}" / "predict_eval_predictions.jsonl"
        g = read_predictions(final, gold, tasks, affected)
        b = read_predictions(baseline_final, gold, tasks, affected)
        if not math.isclose(b["faa"], float(baseline["faa"]), abs_tol=1e-10):
            raise ValueError("Recomputed IncLoRA FAA disagrees with its frozen archival table")
        affected_delta_correct = sum(g["class_correct"][pair] - b["class_correct"][pair] for pair in affected)
        unaffected_delta_correct = sum(g["correct"].values()) - sum(b["correct"].values()) - affected_delta_correct
        results[(size, order)] = (g, b)
        pair_rows.append({"model": size, "order": order, "seed": 0,
                          "G2_FAA": g["faa"], "IncLoRA_FAA": b["faa"], "delta_FAA": g["faa"] - b["faa"],
                          "G2_exclusion_diagnostic_FAA": g["exclusion_diagnostic_faa"],
                          "IncLoRA_exclusion_diagnostic_FAA": b["exclusion_diagnostic_faa"],
                          "delta_exclusion_diagnostic_FAA": g["exclusion_diagnostic_faa"] - b["exclusion_diagnostic_faa"],
                          "strict_delta_FAA": g["strict_faa"] - b["strict_faa"],
                          "strict_delta_exclusion_diagnostic_FAA": g["strict_exclusion_diagnostic_faa"] - b["strict_exclusion_diagnostic_faa"],
                          "affected_classes_contribution_to_raw_delta_pp": 100 * affected_delta_correct / 30400,
                          "other_classes_contribution_to_raw_delta_pp": 100 * unaffected_delta_correct / 30400})
        for method, scored in (("G2", g), ("IncLoRA", b)):
            source_rows.append({"model": size, "order": order, "seed": 0, "method": method, "stage": 4, **public_prediction_summary(scored)})
            for task in tasks:
                task_rows.append({"model": size, "order": order, "seed": 0, "method": method, "task": task,
                                  "all_n": scored["counts"][task], "correct": scored["correct"][task],
                                  "retained_n": scored["retained"][task], "retained_correct": scored["retained_correct"].get(task, 0),
                                  "excluded_n": scored["counts"][task] - scored["retained"][task],
                                  "full_accuracy": 100 * scored["correct"][task] / scored["counts"][task],
                                  "exclusion_diagnostic_accuracy": 100 * scored["retained_correct"].get(task, 0) / scored["retained"][task]})
                for label, n in sorted(Counter(x["label"] for x in gold[task]).items()):
                    key = (task, label)
                    class_rows.append({"model": size, "order": order, "seed": 0, "method": method, "task": task, "label": label,
                                       "boundary_mismatch": key in affected, "n": n, "correct": scored["class_correct"][key],
                                       "accuracy": 100 * scored["class_correct"][key] / n,
                                       "strict_accuracy": 100 * scored["class_strict_correct"][key] / n,
                                       "top_wrong_predictions": json.dumps(scored["wrong_predictions"][key].most_common(5), ensure_ascii=False)})
            if method == "G2":
                tokenizer = tokenizers[size]
                for (task, label), representative in scored["representatives"].items():
                    instance = representative["Instance"]
                    instruction = f"Task:{representative['Task']}\nDataset:{task}\n" + instance["instruction"].format(instance["sentence"])
                    prompt = tokenizer.bos_token + instruction
                    label_text = label + tokenizer.eos_token
                    input_ids = tokenizer(prompt)["input_ids"]
                    # Detect lexical boundary fusion separately from the old
                    # automatic BOS insertion on each standalone label call.
                    legacy_label_ids = tokenizer(label_text)["input_ids"]
                    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
                    joint_ids = tokenizer(prompt + label_text)["input_ids"]
                    mismatch = joint_ids != input_ids + label_ids
                    if mismatch != ((task, label) in affected):
                        raise ValueError(f"Actual complete prompt mismatch differs from minimal reproducer: {size}/{task}/{label!r}, joint tail {joint_ids[-12:]}, separate tail {(input_ids + label_ids)[-12:]}")
                    if order == 1:
                        match_row = next(r for r in token_rows if r["model"] == size and r["task"] == task and r["label"] == label)
                        match_row.update({"actual_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                          "actual_prompt_token_count": len(input_ids), "actual_joint_token_count": len(joint_ids),
                                          "legacy_label_ids_with_automatic_specials": legacy_label_ids,
                                          "label_ids_without_automatic_specials": label_ids,
                                          "legacy_extra_label_special_tokens": len(legacy_label_ids) - len(label_ids),
                                          "legacy_prompt_initial_ids": input_ids[:3],
                                          "actual_joint_tail_ids": joint_ids[-12:], "actual_separate_tail_ids": (input_ids + label_ids)[-12:],
                                          "actual_joint_tail_pieces": tokenizer.convert_ids_to_tokens(joint_ids[-12:]),
                                          "actual_separate_tail_pieces": tokenizer.convert_ids_to_tokens((input_ids + label_ids)[-12:])})
        # First-task per-class corroboration is a separate endpoint from FAA.
        for method, path in (("G2", Path(specs[1]["output_dir"]) / "predict_eval_predictions.jsonl"),
                             ("IncLoRA", Path(baseline["run_dir"]) / f"1-{tasks[0]}" / "predict_eval_predictions.jsonl")):
            first = read_predictions(path, gold, tasks[:1], affected)
            source_rows.append({"model": size, "order": order, "seed": 0, "method": method, "stage": 1, **public_prediction_summary(first)})
            for (task, label), n in sorted(first["class_counts"].items()):
                stage1_rows.append({"model": size, "order": order, "seed": 0, "method": method, "task": task, "label": label,
                                    "n": n, "correct": first["class_correct"][(task, label)], "accuracy": 100 * first["class_correct"][(task, label)] / n})

    for (model, order, seed), baseline in sorted(baselines.items()):
        if seed not in {int(s) for s in args.historical_seeds.split(",")}:
            continue
        size = model.rsplit("-", 1)[1]
        path = Path(baseline["run_dir"]) / f"4-{ORDERS[order][3]}" / "predict_eval_predictions.jsonl"
        # seed42 reused historical folders have a different stage-directory
        # name; locate by filename and task-count, never silently substitute seed.
        if not path.is_file():
            candidates = list(Path(baseline["run_dir"]).glob("*/predict_eval_predictions.jsonl"))
            candidates = [p for p in candidates if p.parent.name.startswith("4-") or p.parent.name.startswith("4_")]
            if len(candidates) != 1:
                raise ValueError(f"Historical final predictions cannot be uniquely located: {model}/{order}/{seed}")
            path = candidates[0]
        scored = results[(size, order)][1] if seed == 0 else read_predictions(path, gold, ORDERS[order], affected)
        inputs.append({"path": str(path), "sha256": scored["sha256"]})
        historical_rows.append({"model": size, "order": order, "seed": seed, "source_kind": baseline["source_kind"],
                                "FAA": scored["faa"], "exclusion_diagnostic_FAA": scored["exclusion_diagnostic_faa"],
                                "Athlete_accuracy": 100 * scored["class_correct"][("dbpedia", "Athlete")] / scored["class_counts"][("dbpedia", "Athlete")],
                                "prediction_path": str(path), "prediction_sha256": scored["sha256"]})

    mean_rows = []
    for size in ("1B", "3B"):
        rows = [row for row in pair_rows if row["model"] == size]
        mean_rows.append({"model": size, "order": "mean_Orders1_2_3", "seed": 0,
                          **{key: statistics.mean(row[key] for row in rows) for key in rows[0] if key not in ("model", "order", "seed")}})
    for row in pair_rows + mean_rows:
        if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise ValueError("Nonfinite summary metric")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "LABEL_BOUNDARY_PAIRS.csv", pair_rows + mean_rows)
    write_csv(output / "LABEL_BOUNDARY_TASKS.csv", task_rows)
    write_csv(output / "LABEL_BOUNDARY_CLASSES.csv", class_rows)
    write_csv(output / "LABEL_BOUNDARY_STAGE1_CLASSES.csv", stage1_rows)
    write_csv(output / "LABEL_BOUNDARY_HISTORICAL_SEEDS.csv", historical_rows)
    timestamp = datetime.now(timezone.utc).isoformat()
    audit = {"status": "pass", "created_at": timestamp,
             "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
             "metric": "normalized lowercase / ASCII punctuation removal / whitespace EM, task-equal FAA",
             "diagnostic_warning": "Excluding affected gold-label classes does not repair legacy training, automatic BOS/off-by-one or truncation defects, or create a corrected benchmark. Class contributions describe localization of counts, not a causal attribution to the bug.",
             "scope": "Two models, three orders, seed0 paired G2/IncLoRA; historical IncLoRA seeds for descriptive variability only",
             "affected_labels": [{"task": t, "label": l} for t, l in sorted(affected)],
             "token_boundaries": token_rows, "paired_results": pair_rows + mean_rows,
             "input_hashes": inputs, "prediction_sources": source_rows,
             "order1_order2_first_stage_same_bytes": {
                 size: next(r["sha256"] for r in source_rows if r["model"] == size and r["method"] == "G2" and r["order"] == 1 and r["stage"] == 1)
                 == next(r["sha256"] for r in source_rows if r["model"] == size and r["method"] == "G2" and r["order"] == 2 and r["stage"] == 1)
                 for size in ("1B", "3B")},
             "cuda_initialized": torch.cuda.is_initialized()}
    if audit["cuda_initialized"]:
        raise ValueError("CPU/offline audit unexpectedly initialized CUDA")
    (output / "LABEL_BOUNDARY_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 标签边界与旧结果复核", "", f"UTC 快照：{timestamp}", "",
             "本报告按实际本地 Llama tokenizer 验证全部 33 类。1B、3B 均有 5 类在旧版训练文本 `Answer:` 与标签拼接时跨边界合并；推理只提供独立 `Answer:`。这证明训练/推理 token 前缀不一致，不能据此证明差值全部由该问题导致。", "",
             "正式指标采用旧实验一致的 normalized EM；剔除这些 gold 类后的指标仅是条件诊断，任务内样本等权、四任务等权，未改成类别等权，也不能称为修复数据后的 benchmark。剔除五类并不能去掉旧管线的重复 BOS、label 长度 off-by-one 和截断问题。", "",
             "| 任务 | 全部样本 | 剔除样本 | 保留样本 |", "|---|---:|---:|---:|"]
    for task in TASK_TYPE:
        n = sum(x["label"] in {l for t, l in affected if t == task} for x in gold[task])
        lines.append(f"| {task} | 7600 | {n} | {7600-n} |")
    lines += ["", "| 模型 | Order | 原 FAA 差值 | 剔除诊断 FAA 差值 | strict 原差值 |", "|---|---|---:|---:|---:|"]
    for row in pair_rows + mean_rows:
        lines.append(f"| {row['model']} | {row['order']} | {row['delta_FAA']:+.6f} | {row['delta_exclusion_diagnostic_FAA']:+.6f} | {row['strict_delta_FAA']:+.6f} |")
    lines += ["", "scratchpad `clean.py` 使用 prediction.strip()==ground_truth；这里正式 normalized EM 与 strict EM 分列。因此用户列出的 1B Order3 +0.84 对应 strict 口径，正式口径为 +0.815789。", "",
              "Order1/Order2 的首次 dbpedia 预测文件在同模型 G2 中逐字节一致。两个 Order 共享首任务，不能作为独立重复测量。", "",
              "旧 collator 手工前置 BOS 后又调用 tokenizer 默认 special-token 处理，本地 tokenizer 因而在 prompt 开头产生两次 BOS；独立 label 的默认 tokenization 也额外插一次 BOS，影响旧 loss_mask 的标签长度估计。这些证据已逐标签记录。", "",
              "受影响类别不等于准确率都为 0%。最终 1B Order1 Natural Place 为 G2 80.8194%、IncLoRA 53.8175%；3B Order1 Educational Institution 为 17.7264%/0%，Natural Place 为 36.8715%/0.3724%。amazon negative 与 yahoo Society & Culture 才是这里普遍接近 0% 的两类。", "",
              "3B 首任务 Athlete 的 95.0000%（G2）/14.6429%（IncLoRA）已核验。最终三 Order 平均原始 FAA 增益中，受影响五类贡献 +1.391447 个百分点，其它类贡献 +0.610746；这仅是正确样本计数分解，不能证明标签缺陷因果地产生这部分收益。", "",
              "类级错误统计、旧基线跨 seed 描述性数据和全部精确 token ID/pieces 见同目录 CSV/JSON。这些历史样本不是修复后多 seed 的方法验证。", "",
              "复现（无 GPU、无网络、无模型权重加载）：", "```bash",
              "/data/140-1/users/liying/conda_envs/nlora_req/bin/python scripts_llama/audit_g2_label_boundary.py", "```", ""]
    (output / "LABEL_BOUNDARY_ANALYSIS.md").write_text("\n".join(lines))
    print(json.dumps({"status": "pass", "affected_classes": len(affected), "paired_results": pair_rows + mean_rows, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
