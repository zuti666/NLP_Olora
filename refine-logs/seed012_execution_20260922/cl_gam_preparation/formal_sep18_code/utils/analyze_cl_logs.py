#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze continual-learning experiment logs and export summary/detail tables.

Main outputs:
  1. run_summary.csv / run_summary sheet
     Compact table requested for reports: dataset, method, optimizer, inferred task
     count, final Accuracy, log path, and config path.

  2. task_accuracy.csv / task_accuracy sheet
     One row per evaluation step. It records task_0_accuracy ... task_N_accuracy,
     so runs with different task counts can live in the same table.

  2b. comparison_summary.csv / comparison_summary sheet
      One row per comparison arm with attempted/valid seeds, success rate, and
      mean/sample-std for AIA, final/old/new accuracy, and forgetting.

  3. summary.csv / summary sheet
     Full summary table. The task count is inferred from the final Accuracy dict,
     not blindly from the config, because different datasets may have different task counts.

  4. eval_wide.csv / eval_wide sheet
     One row per evaluation after each task. Columns include total/old/new and per-task
     class-range accuracies such as acc_00-19, acc_20-39, ...

  5. eval_long.csv / eval_long sheet
     Long format: one row for each (run, evaluation step, measured task range).
     This is convenient for plotting forgetting curves.

  6. final_wide.csv / final_wide sheet
     Final evaluation only, expanded by task-range accuracy.

  7. curves_long.csv / curves_long sheet
     Parsed '(curve) top1 Acc', '(curve) top1 Acc with task', and '(curve) top1 Acc task'.

  8. prototype_eval_wide.csv / prototype_eval_long.csv, if Prototype Accuracy exists.

  9. anchor_eval_metrics.csv / mechanism_metric_summary.csv / mechanism_metrics_long.csv
     Mechanism/AnchorEval metric tables for flatness, Fisher, Hessian, empirical
     Fisher, EWC drift, and train-time mechanism statistics.

Usage examples:
  python analyze_cl_logs.py --inputs logs/
  python analyze_cl_logs.py --log-dir logs/launch --output-dir parsed_launch_logs
  python analyze_cl_logs.py --inputs "logs/launch/*.out" --output-dir parsed_launch_logs
  python analyze_cl_logs.py --inputs logs/run1.log logs/run2.out --output-dir parsed_results
  python analyze_cl_logs.py --inputs /mnt/data/cub200run_gam001-redo.log /mnt/data/seed4.log --output-dir /mnt/data/parsed_logs
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
KV_RE = re.compile(r"=>\s*([A-Za-z0-9_\-]+)\s*:\s*(.*)\s*$")
CONFIG_RE = re.compile(r"=>\s*config\s*:", re.IGNORECASE)
OUTPUTS_ROOT_RE = re.compile(r"=>\s*outputs_root\s*:", re.IGNORECASE)
PREFIX_RE = re.compile(r"=>\s*prefix\s*:", re.IGNORECASE)
ACC_RE = re.compile(r"=>\s*(?:(Prototype)\s+)?(?:Accuracy|CNN):\s*(\{.*\})", re.IGNORECASE)
CURVE_RE = re.compile(r"=>\s*\((prototype\s+)?curve\)\s*([^:]+):\s*(\[.*\])", re.IGNORECASE)
CNN_CURVE_RE = re.compile(r"=>\s*CNN\s+top1\s+curve:\s*(\[.*\])", re.IGNORECASE)
AVERAGE_CNN_RE = re.compile(r"(?:=>\s*)?Average Accuracy \(CNN\):\s*([-+0-9.eE]+)", re.IGNORECASE)
FORGETTING_CNN_RE = re.compile(r"(?:=>\s*)?Forgetting \(CNN\):\s*([-+0-9.eE]+)", re.IGNORECASE)
TASK_LEARN_RE = re.compile(r"Task\s+(\d+)\s+learning\s+on\s+class\s+([0-9]+)\s*-\s*([0-9]+)")
MECHANISM_RE = re.compile(r"(?:=>\s*)?\[[^\]]*Mechanism[^\]]*\]\s*(\{.*\})")
ANCHOR_EVAL_RE = re.compile(r"(?:=>\s*)?\[AnchorEval\]\s*(\{.*\})")
RANGE_KEY_RE = re.compile(r"^\d+\-\d+$")
NUMPY_SCALAR_RE = re.compile(
    r"\b(?:np|numpy)\.(?:float(?:16|32|64)?|int(?:8|16|32|64)?|bool_?)\(([^()]*)\)"
)
SPECIAL_FLOAT_RE = re.compile(r"(?<![A-Za-z0-9_.'\"])(?:nan|[-+]?inf)(?![A-Za-z0-9_.'\"])", re.IGNORECASE)

DEFAULT_LOG_EXTS = {".log", ".out", ".txt"}

META_KEYS = [
    "config", "prefix", "exp_name", "dataset", "data_path", "load", "seed",
    "method", "method_type", "model_name", "optimizer", "optimizer_type",
    "gam_base_optimizer", "fs_base_optimizer",
    "sessions", "total_sessions", "max_train_tasks", "init_cls", "increment",
    "init_epoch", "epochs", "batch_size",
    "rank", "lrate", "fc_lrate", "momentum", "weight_decay", "scheduler",
    "lambda", "gamma", "lambda_flat", "lambda_flat_lora", "lambda_flat_classifier",
    "lambda_ewc", "ewc_lambda", "fisher_eta", "fisher_mode",
    "gam_grad_rho", "gam_grad_norm_rho", "gam_grad_gamma",
    "gam_task_a", "gam_task_b", "phase_a_tasks", "anchor_train_mode",
    "fs_rho", "fs_alpha", "fs_alpha_mode", "fs_sam_equivalent",
    "fs_adaptive", "fs_xi", "fs_grad_reduce", "fs_task_a", "fs_task_b",
    "use_ewc", "compute_fisher", "upper_bound_type",
    "ewc_gamma", "ewc_eta", "ewc_max_batches", "ewc_include_fc",
    "ewc_normalize_fisher", "ewc_fisher_normalize_type",
    "ewc_normalize_accumulated_fisher", "ewc_fisher_damping_beta",
    "rho_sweep_anchor", "fixed_lambda",
    "mechanism_eval", "anchor_eval", "anchor_eval_task",
    "endpoint_eval", "endpoint_eval_tasks",
    "flat_eval_sharpness", "flat_eval_hessian", "flat_eval_emp_fisher_spectrum",
    "flat_eval_expected_sharpness", "flat_eval_esh_samples", "flat_eval_param_scope",
    "flat_eval_sharpness_radius", "flat_eval_loss_max_batches",
    "flat_eval_first_order_grad_batches", "flat_eval_second_order_batch_size",
    "flat_eval_batch_size", "flat_eval_hessian_batch_size",
    "flat_eval_emp_fisher_batch_size", "flat_eval_hessian_max_batches",
    "flat_eval_hessian_power_iters", "flat_eval_hessian_trace_samples",
    "flat_eval_fisher_quadratic_error", "flat_eval_fqa_radii",
    "flat_eval_fqa_samples", "flat_eval_fqa_tasks", "flat_eval_fqa_prefixes",
    "flat_eval_fqa_fisher_space", "flat_eval_factor_radius_space",
    "flat_eval_factor_fisher_delta",
    "flat_eval_unit_expected_sharpness",
    "flat_eval_unit_esh_radii", "flat_eval_unit_esh_samples",
    "fisher_topk_fractions",
    "sam_rho", "rho",
]


def cast_scalar(value: str) -> Any:
    """Best-effort scalar casting while preserving paths and complex strings."""
    v = value.strip()
    if v == "":
        return ""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in {"none", "null"}:
        return None
    # Preserve obvious paths and device/list strings as strings.
    if "/" in v or "[" in v or "]" in v or "(" in v or ")" in v:
        return v
    try:
        if re.fullmatch(r"[-+]?\d+", v):
            return int(v)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+|\d+)([eE][-+]?\d+)?", v):
            return float(v)
    except Exception:
        pass
    return v


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    # tqdm often writes carriage returns; treating them as newlines makes parsing robust.
    text = text.replace("\r", "\n")
    return text


def read_log(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    return text.splitlines()


def normalize_exts(exts: Iterable[str]) -> Set[str]:
    return {(ext if ext.startswith(".") else f".{ext}").lower() for ext in exts}


def discover_logs(inputs: Iterable[str], recursive: bool = True, exts: Optional[Set[str]] = None) -> List[Path]:
    exts = exts or DEFAULT_LOG_EXTS
    paths: List[Path] = []
    for item in inputs:
        raw = str(Path(item).expanduser())
        has_magic = glob.has_magic(raw)
        matches = glob.glob(raw, recursive=recursive)
        candidates = matches if matches else [raw]
        for candidate in candidates:
            p = Path(candidate).expanduser().resolve()
            if p.is_file():
                if not has_magic or p.suffix.lower() in exts:
                    paths.append(p)
            elif p.is_dir():
                iterator = p.rglob("*") if recursive else p.glob("*")
                paths.extend([child for child in iterator if child.is_file() and child.suffix.lower() in exts])
            else:
                print(f"[WARN] input not found: {candidate}", file=sys.stderr)
    # Stable order, unique paths.
    seen = set()
    out = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def split_runs(lines: List[str]) -> List[Tuple[int, int]]:
    # ``print_args`` writes ``config`` near the *end* of each run header, so
    # splitting on that line discards dataset/model/optimizer metadata printed
    # before it.  Prefer the first stable header key and keep ``config`` only as
    # a compatibility fallback for older logs.
    starts = [i for i, line in enumerate(lines) if OUTPUTS_ROOT_RE.search(line)]
    if not starts:
        starts = [i for i, line in enumerate(lines) if PREFIX_RE.search(line)]
    if not starts:
        starts = [i for i, line in enumerate(lines) if CONFIG_RE.search(line)]
    if not starts:
        return [(0, len(lines))]
    runs = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        runs.append((start, end))
    return runs


def safe_literal_eval(raw: str) -> Optional[Any]:
    try:
        return ast.literal_eval(raw)
    except Exception:
        # Current trainer logs numpy scalars and NaNs, e.g.
        # ``{'total': np.float64(78.03), '40-59': nan}``.  Normalize only
        # those value representations, then keep using ast.literal_eval.
        normalized = NUMPY_SCALAR_RE.sub(r"\1", raw)
        normalized = SPECIAL_FLOAT_RE.sub("None", normalized)
        try:
            return ast.literal_eval(normalized)
        except Exception:
            return None


def range_sort_key(k: str) -> Tuple[int, int, str]:
    try:
        a, b = k.split("-", 1)
        return (int(a), int(b), k)
    except Exception:
        return (10**9, 10**9, k)


def task_range_keys(acc: Dict[str, Any]) -> List[str]:
    return sorted([k for k in acc.keys() if RANGE_KEY_RE.match(str(k))], key=range_sort_key)


def observed_task_range_keys(acc: Dict[str, Any]) -> List[str]:
    """Return class ranges that contain a finite observed accuracy."""
    keys = []
    for key in task_range_keys(acc):
        value = acc.get(key)
        try:
            if value is not None and math.isfinite(float(value)):
                keys.append(key)
        except (TypeError, ValueError):
            continue
    return keys


def infer_task_count(acc: Dict[str, Any], max_train_task_seen: Optional[int] = None) -> int:
    keys = observed_task_range_keys(acc)
    if keys:
        return len(keys)
    if max_train_task_seen is not None:
        return max_train_task_seen + 1
    return 0


def common_run_id(log_path: Path, run_index: int) -> str:
    return f"{log_path.name}::run{run_index}"


def infer_metric_prefix(metrics: Dict[str, Any]) -> str:
    for prefix in ("anchor", "endpoint", "accumulated_fisher"):
        if any(str(key).startswith(f"{prefix}_") for key in metrics):
            return prefix
    return ""


def metric_task_for_prefix(metrics: Dict[str, Any], prefix: str) -> Any:
    if prefix:
        task = metrics.get(f"{prefix}_task", "")
        if task != "":
            return task
    return metrics.get("task", "")


def parse_one_run(lines: List[str], log_path: Path, run_index: int) -> Dict[str, List[Dict[str, Any]]]:
    meta: Dict[str, Any] = {k: "" for k in META_KEYS}
    timestamps: List[str] = []
    train_tasks: List[Dict[str, Any]] = []
    max_train_task_seen: Optional[int] = None

    eval_wide: List[Dict[str, Any]] = []
    eval_long: List[Dict[str, Any]] = []
    proto_wide: List[Dict[str, Any]] = []
    proto_long: List[Dict[str, Any]] = []
    curves_long: List[Dict[str, Any]] = []
    mechanism_metrics: List[Dict[str, Any]] = []
    anchor_eval_metrics: List[Dict[str, Any]] = []

    rid = common_run_id(log_path, run_index)
    base = {
        "run_id": rid,
        "run_index_in_log": run_index,
        "log_path": str(log_path),
    }

    def build_prefix() -> Dict[str, Any]:
        p = dict(base)
        p.update(meta)
        return p

    normal_eval_index = -1
    proto_eval_index = -1
    average_accuracy_cnn: Optional[float] = None
    forgetting_cnn: Optional[float] = None
    has_error = False
    has_nan_loss = False

    for line_number, line in enumerate(lines, start=1):
        ts = TIMESTAMP_RE.search(line)
        if ts:
            timestamps.append(ts.group(1))

        if re.search(r"Traceback|CUDA out of memory|RuntimeError", line, re.IGNORECASE):
            has_error = True
        if re.search(r"\bLoss\s+(?:nan|inf)\b", line, re.IGNORECASE):
            has_nan_loss = True

        avg_match = AVERAGE_CNN_RE.search(line)
        if avg_match:
            average_accuracy_cnn = float(avg_match.group(1))

        forgetting_match = FORGETTING_CNN_RE.search(line)
        if forgetting_match:
            forgetting_cnn = float(forgetting_match.group(1))

        kv = KV_RE.search(line)
        if kv:
            key = kv.group(1).strip()
            value = kv.group(2).strip()
            if key in meta and meta[key] == "":
                meta[key] = cast_scalar(value)

        tm = TASK_LEARN_RE.search(line)
        if tm:
            tid = int(tm.group(1))
            c0, c1 = int(tm.group(2)), int(tm.group(3))
            train_tasks.append({"task_id": tid, "class_start": c0, "class_end": c1})
            max_train_task_seen = tid if max_train_task_seen is None else max(max_train_task_seen, tid)

        ae = ANCHOR_EVAL_RE.search(line)
        if ae:
            metrics = safe_literal_eval(ae.group(1))
            if isinstance(metrics, dict):
                metric_prefix = infer_metric_prefix(metrics)
                row = build_prefix()
                row.update(metrics)
                row["metric_source"] = "anchor_eval"
                row["metric_prefix"] = metric_prefix
                row["metric_task"] = metric_task_for_prefix(metrics, metric_prefix)
                row["metric_time"] = ts.group(1) if ts else ""
                row["line_number"] = line_number
                row["run_start_time"] = timestamps[0] if timestamps else ""
                row["run_end_time"] = timestamps[-1] if timestamps else ""
                anchor_eval_metrics.append(row)
            continue

        mm = MECHANISM_RE.search(line)
        if mm:
            metrics = safe_literal_eval(mm.group(1))
            if isinstance(metrics, dict):
                row = build_prefix()
                row.update(metrics)
                row["metric_source"] = "mechanism"
                row["mechanism_task"] = metrics.get("task", "")
                row["metric_time"] = ts.group(1) if ts else ""
                row["line_number"] = line_number
                row["run_start_time"] = timestamps[0] if timestamps else ""
                row["run_end_time"] = timestamps[-1] if timestamps else ""
                mechanism_metrics.append(row)
            continue

        am = ACC_RE.search(line)
        if am:
            is_proto = bool(am.group(1))
            acc = safe_literal_eval(am.group(2))
            if not isinstance(acc, dict):
                continue

            keys = observed_task_range_keys(acc)
            observed_task_count = infer_task_count(acc, max_train_task_seen)
            observed_task_id = observed_task_count - 1 if observed_task_count > 0 else ""

            if is_proto:
                proto_eval_index += 1
                row = build_prefix()
                row.update({
                    "eval_index": proto_eval_index,
                    "eval_task_id": observed_task_id,
                    "observed_task_count": observed_task_count,
                    "total": acc.get("total", ""),
                    "old": acc.get("old", ""),
                    "new": acc.get("new", ""),
                })
                for k in keys:
                    row[f"acc_{k}"] = acc.get(k, "")
                proto_wide.append(row)

                for k in keys:
                    long_row = build_prefix()
                    long_row.update({
                        "eval_index": proto_eval_index,
                        "eval_task_id": observed_task_id,
                        "observed_task_count": observed_task_count,
                        "measured_task_range": k,
                        "measured_task_start": range_sort_key(k)[0],
                        "measured_task_end": range_sort_key(k)[1],
                        "accuracy": acc.get(k, ""),
                        "total": acc.get("total", ""),
                        "old": acc.get("old", ""),
                        "new": acc.get("new", ""),
                    })
                    proto_long.append(long_row)
            else:
                normal_eval_index += 1
                row = build_prefix()
                row.update({
                    "eval_index": normal_eval_index,
                    "eval_task_id": observed_task_id,
                    "observed_task_count": observed_task_count,
                    "total": acc.get("total", ""),
                    "old": acc.get("old", ""),
                    "new": acc.get("new", ""),
                })
                for k in keys:
                    row[f"acc_{k}"] = acc.get(k, "")
                eval_wide.append(row)

                for k in keys:
                    long_row = build_prefix()
                    long_row.update({
                        "eval_index": normal_eval_index,
                        "eval_task_id": observed_task_id,
                        "observed_task_count": observed_task_count,
                        "measured_task_range": k,
                        "measured_task_start": range_sort_key(k)[0],
                        "measured_task_end": range_sort_key(k)[1],
                        "accuracy": acc.get(k, ""),
                        "total": acc.get("total", ""),
                        "old": acc.get("old", ""),
                        "new": acc.get("new", ""),
                    })
                    eval_long.append(long_row)
            continue

        cm = CURVE_RE.search(line)
        if cm:
            is_proto_curve = bool(cm.group(1))
            curve_name = cm.group(2).strip()
            values = safe_literal_eval(cm.group(3))
            if not isinstance(values, list):
                continue
            associated_eval_index = proto_eval_index if is_proto_curve else normal_eval_index
            for step, value in enumerate(values):
                row = build_prefix()
                row.update({
                    "curve_type": "prototype" if is_proto_curve else "normal",
                    "curve_name": curve_name,
                    "associated_eval_index": associated_eval_index,
                    "step": step,
                    "value": value,
                })
                curves_long.append(row)
            continue

        cnn_curve = CNN_CURVE_RE.search(line)
        if cnn_curve:
            values = safe_literal_eval(cnn_curve.group(1))
            if not isinstance(values, list):
                continue
            for step, value in enumerate(values):
                row = build_prefix()
                row.update({
                    "curve_type": "normal",
                    "curve_name": "CNN top1",
                    "associated_eval_index": normal_eval_index,
                    "step": step,
                    "value": value,
                })
                curves_long.append(row)

    # Add run-level timing and train-task metadata to all parsed rows.
    run_start_time = timestamps[0] if timestamps else ""
    run_end_time = timestamps[-1] if timestamps else ""
    max_train_task = max_train_task_seen if max_train_task_seen is not None else ""

    for table in (
        eval_wide, eval_long, proto_wide, proto_long, curves_long,
        mechanism_metrics, anchor_eval_metrics,
    ):
        for row in table:
            row["run_start_time"] = run_start_time
            row["run_end_time"] = run_end_time
            row["max_train_task_seen"] = max_train_task

    # Summary row from final normal Accuracy.
    summary: List[Dict[str, Any]] = []
    final_wide: List[Dict[str, Any]] = []
    expected_tasks = (
        meta.get("max_train_tasks", "")
        or meta.get("sessions", "")
        or meta.get("total_sessions", "")
    )
    if eval_wide:
        final = dict(eval_wide[-1])
        final_task_count = int(final.get("observed_task_count") or 0)
        final_task_id = final_task_count - 1 if final_task_count > 0 else ""
        completed: Any = ""
        try:
            completed = bool(expected_tasks != "" and int(expected_tasks) == final_task_count)
        except Exception:
            completed = ""
        valid_run = bool(
            not has_error
            and not has_nan_loss
            and final_task_count > 0
            and completed is not False
        )

        summary_row = build_prefix()
        summary_row.update({
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
            "max_train_task_seen": max_train_task,
            "final_eval_index": final.get("eval_index", ""),
            "final_task_id": final_task_id,
            "task_count_inferred_from_final_accuracy": final_task_count,
            "final_total_accuracy": final.get("total", ""),
            "final_old_accuracy": final.get("old", ""),
            "final_new_accuracy": final.get("new", ""),
            "average_accuracy_cnn": average_accuracy_cnn if average_accuracy_cnn is not None else "",
            "forgetting_cnn": forgetting_cnn if forgetting_cnn is not None else "",
            "has_error": has_error,
            "has_nan_loss": has_nan_loss,
            "expected_task_count": expected_tasks,
            "valid_run": valid_run,
            "completed_by_sessions": completed,
        })
        # Compact final per-task values in summary as well.
        final_task_cols = sorted([k for k in final.keys() if k.startswith("acc_")], key=lambda x: range_sort_key(x[4:]))
        summary_row["final_task_ranges"] = ",".join([c[4:] for c in final_task_cols])
        for c in final_task_cols:
            summary_row[f"final_{c}"] = final.get(c, "")
        summary.append(summary_row)

        fw = build_prefix()
        fw.update({
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
            "final_eval_index": final.get("eval_index", ""),
            "final_task_id": final_task_id,
            "task_count": final_task_count,
            "total": final.get("total", ""),
            "old": final.get("old", ""),
            "new": final.get("new", ""),
            "average_accuracy_cnn": average_accuracy_cnn if average_accuracy_cnn is not None else "",
            "forgetting_cnn": forgetting_cnn if forgetting_cnn is not None else "",
            "has_error": has_error,
            "has_nan_loss": has_nan_loss,
            "expected_task_count": expected_tasks,
            "valid_run": valid_run,
            "completed_by_sessions": completed,
        })
        for c in sorted([k for k in final.keys() if k.startswith("acc_")], key=lambda x: range_sort_key(x[4:])):
            fw[c] = final.get(c, "")
        final_wide.append(fw)
    elif any(value not in {"", None} for value in meta.values()) or has_error or max_train_task_seen is not None:
        # Preserve failed/aborted attempts in run_summary so success-rate
        # accounting does not silently drop OOMs or pre-evaluation crashes.
        completed = False if expected_tasks != "" else ""
        summary_row = build_prefix()
        summary_row.update({
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
            "max_train_task_seen": max_train_task,
            "final_eval_index": "",
            "final_task_id": "",
            "task_count_inferred_from_final_accuracy": 0,
            "final_total_accuracy": "",
            "final_old_accuracy": "",
            "final_new_accuracy": "",
            "average_accuracy_cnn": average_accuracy_cnn if average_accuracy_cnn is not None else "",
            "forgetting_cnn": forgetting_cnn if forgetting_cnn is not None else "",
            "has_error": has_error,
            "has_nan_loss": has_nan_loss,
            "expected_task_count": expected_tasks,
            "valid_run": False,
            "completed_by_sessions": completed,
            "final_task_ranges": "",
        })
        summary.append(summary_row)

    return {
        "summary": summary,
        "eval_wide": eval_wide,
        "eval_long": eval_long,
        "final_wide": final_wide,
        "curves_long": curves_long,
        "mechanism_metrics": mechanism_metrics,
        "anchor_eval_metrics": anchor_eval_metrics,
        "prototype_eval_wide": proto_wide,
        "prototype_eval_long": proto_long,
    }


def union_fieldnames(rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> List[str]:
    preferred = preferred or []
    keys = []
    seen = set()
    for k in preferred:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    # Sort dynamic accuracy columns by numeric range while preserving non-acc order.
    fixed = [k for k in keys if not (k.startswith("acc_") or k.startswith("final_acc_"))]
    acc = [k for k in keys if k.startswith("acc_")]
    final_acc = [k for k in keys if k.startswith("final_acc_")]
    acc = sorted(acc, key=lambda x: range_sort_key(x[4:]))
    final_acc = sorted(final_acc, key=lambda x: range_sort_key(x[len("final_acc_"):]))
    return fixed + acc + final_acc


TASK_VALUE_COL_RE = re.compile(r"^task_(\d+)_(range|accuracy)$")


def task_value_sort_key(name: str) -> Tuple[int, int, str]:
    match = TASK_VALUE_COL_RE.match(name)
    if not match:
        return (10**9, 10**9, name)
    task_id = int(match.group(1))
    # Keep each range immediately before its accuracy.
    sub_order = 0 if match.group(2) == "range" else 1
    return (task_id, sub_order, name)


def requested_fieldnames(rows: List[Dict[str, Any]], preferred: List[str]) -> List[str]:
    keys = union_fieldnames(rows, preferred)
    fixed = [k for k in keys if not TASK_VALUE_COL_RE.match(k)]
    task_cols = sorted([k for k in keys if TASK_VALUE_COL_RE.match(k)], key=task_value_sort_key)
    return fixed + task_cols


SETTING_NAMES = (
    "upper_bound_joint_0_1",
    "upper_bound_task0_only",
    "upper_bound_task1_only",
    "stage2_fs_noewc",
    "stage1_fs_noewc",
    "sam_s2_noewc",
    "sam_s2_ewc",
    "factored_full",
    "factored_s1",
    "factored_s2",
    "fs_only",
    "fs_ewc",
    "baseline",
    "control",
)


def infer_setting(row: Dict[str, Any]) -> str:
    """Infer the comparison arm from an explicit upper-bound type or prefix."""
    upper = str(row.get("upper_bound_type", "") or "")
    if upper == "joint_original_classes_00_39":
        return "upper_bound_joint_0_1"
    if upper == "single_task_original_classes_00_19":
        return "upper_bound_task0_only"
    if upper == "single_task_original_classes_20_39":
        return "upper_bound_task1_only"

    prefix = str(row.get("prefix", "") or "")
    for name in SETTING_NAMES:
        if re.search(rf"(?:^|_){re.escape(name)}(?:_|$)", prefix):
            return name
    return ""


def build_run_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the compact run-level table requested by the experiment summary."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        method = row.get("method", "") or row.get("model_name", "")
        method_type = row.get("method_type", "") or row.get("optimizer_type", "")
        optimizer = row.get("optimizer_type", "") or row.get("optimizer", "")
        out.append({
            "run_id": row.get("run_id", ""),
            "setting": infer_setting(row),
            "prefix": row.get("prefix", ""),
            "dataset": row.get("dataset", ""),
            "method": method,
            "model_name": row.get("model_name", ""),
            "method_type": method_type,
            "optimizer": optimizer,
            "gam_base_optimizer": row.get("gam_base_optimizer", ""),
            "fs_base_optimizer": row.get("fs_base_optimizer", ""),
            "use_ewc": row.get("use_ewc", ""),
            "compute_fisher": row.get("compute_fisher", ""),
            "ewc_lambda": row.get("ewc_lambda", row.get("lambda_ewc", row.get("lambda", ""))),
            "fs_rho": row.get("fs_rho", ""),
            "fs_alpha": row.get("fs_alpha", ""),
            "fs_alpha_mode": row.get("fs_alpha_mode", ""),
            "fs_sam_equivalent": row.get("fs_sam_equivalent", ""),
            "ewc_fisher_damping_beta": row.get("ewc_fisher_damping_beta", ""),
            "fisher_mode": row.get("fisher_mode", ""),
            "seed": row.get("seed", ""),
            "task_count": row.get("task_count_inferred_from_final_accuracy", ""),
            "expected_task_count": row.get("expected_task_count", ""),
            "final_task_id": row.get("final_task_id", ""),
            "AIA": row.get("average_accuracy_cnn", ""),
            "Accuracy": row.get("final_total_accuracy", ""),
            "old_accuracy": row.get("final_old_accuracy", ""),
            "new_accuracy": row.get("final_new_accuracy", ""),
            "Forgetting": row.get("forgetting_cnn", ""),
            "valid_run": row.get("valid_run", ""),
            "has_nan_loss": row.get("has_nan_loss", ""),
            "has_error": row.get("has_error", ""),
            "log_path": row.get("log_path", ""),
            "config_path": row.get("config", ""),
            "run_start_time": row.get("run_start_time", ""),
            "run_end_time": row.get("run_end_time", ""),
        })
    return out


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_comparison_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate valid seeds per comparison arm, while retaining failure rate."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        setting = str(row.get("setting", "") or "")
        if not setting:
            setting = "/".join(
                str(row.get(key, "") or "")
                for key in ("model_name", "optimizer")
            )
        key = (str(row.get("dataset", "") or ""), setting)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    metric_names = ("AIA", "Accuracy", "old_accuracy", "new_accuracy", "Forgetting")
    for (dataset, setting), group_rows in sorted(groups.items()):
        valid_rows = [row for row in group_rows if row.get("valid_run") is True]
        first = group_rows[0]
        item: Dict[str, Any] = {
            "dataset": dataset,
            "setting": setting,
            "model_name": first.get("model_name", ""),
            "optimizer": first.get("optimizer", ""),
            "use_ewc": first.get("use_ewc", ""),
            "ewc_lambda": first.get("ewc_lambda", ""),
            "fs_rho": first.get("fs_rho", ""),
            "fs_alpha": first.get("fs_alpha", ""),
            "fs_alpha_mode": first.get("fs_alpha_mode", ""),
            "ewc_fisher_damping_beta": first.get("ewc_fisher_damping_beta", ""),
            "attempted_seeds": ",".join(str(row.get("seed", "")) for row in group_rows),
            "valid_seeds": ",".join(str(row.get("seed", "")) for row in valid_rows),
            "attempted_runs": len(group_rows),
            "valid_runs": len(valid_rows),
            "success_rate": len(valid_rows) / len(group_rows) if group_rows else "",
        }
        for metric in metric_names:
            values = [
                number
                for row in valid_rows
                if (number := _finite_float(row.get(metric, ""))) is not None
            ]
            item[f"{metric}_mean"] = statistics.mean(values) if values else ""
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else ""
            item[f"{metric}_n"] = len(values)
        out.append(item)
    return out


def build_task_accuracy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a task-indexed accuracy matrix from every normal Accuracy output."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        acc_cols = sorted([k for k in row if k.startswith("acc_")], key=lambda x: range_sort_key(x[4:]))
        item: Dict[str, Any] = {
            "run_id": row.get("run_id", ""),
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "method_type": row.get("method_type", ""),
            "optimizer": row.get("optimizer", ""),
            "gam_base_optimizer": row.get("gam_base_optimizer", ""),
            "fisher_mode": row.get("fisher_mode", ""),
            "seed": row.get("seed", ""),
            "eval_task_id": row.get("eval_task_id", ""),
            "observed_task_count": row.get("observed_task_count", ""),
            "total_accuracy": row.get("total", ""),
            "old_accuracy": row.get("old", ""),
            "new_accuracy": row.get("new", ""),
            "log_path": row.get("log_path", ""),
            "config_path": row.get("config", ""),
        }
        for task_id, col in enumerate(acc_cols):
            item[f"task_{task_id}_range"] = col[4:]
            item[f"task_{task_id}_accuracy"] = row.get(col, "")
        out.append(item)
    return out


SUMMARY_METRIC_PREFIXES = ("anchor", "endpoint", "accumulated_fisher")
COMPACT_COMMON_COLS = [
    "run_id", "dataset", "method", "method_type", "optimizer", "gam_base_optimizer",
    "fisher_mode", "seed", "metric_source", "mechanism_task", "metric_prefix",
    "metric_task", "task_start", "task_end", "param_space", "requested_param_scope",
    "param_count", "fisher_blocks", "fisher_numel", "fisher_total_mass",
    "fisher_mean", "fisher_max", "fisher_top0p001_mass_ratio",
    "fisher_top0p01_mass_ratio", "fisher_top0p05_mass_ratio",
    "base_loss", "grad_norm", "first_order_sharpness", "sh0_perturbed_loss",
    "sh0_max", "esh_mean", "esh_std", "hessian_batch_size", "emp_fisher_batch_size",
    "hessian_top_eigen", "hessian_trace", "emp_fisher_top_eigen",
    "emp_fisher_trace", "hessian_emp_fisher_top_eigen_ratio",
    "hessian_emp_fisher_trace_ratio", "delta_norm", "delta_norm_sq",
    "delta_l2_norm", "fisher_weighted_drift",
    "normalized_fisher_drift", "t1_fisher_top1pct_numel",
    "delta_w_energy_in_t1_top1pct", "d_f_contribution_from_t1_top1pct",
    "ewc_penalty_value", "train_ce_mean",
    "train_delta_norm_mean", "train_ewc_mean", "train_ewc_ce_ratio_mean",
    "train_ewc_penalty_value_mean", "train_fisher_weighted_drift_mean",
    "train_normalized_fisher_drift_mean", "train_cos_gam_ewc_mean",
    "train_fs_g0_norm_mean", "train_fs_delta_norm_mean", "train_fs_rf_norm_mean",
    "train_fs_curvature_ratio_mean", "train_fs_shaping_component_norm_mean",
    "train_fs_shaping_ratio_mean", "train_fs_coeff_on_delta_mean",
    "train_fs_ce_backward_count_mean", "train_fs_cos_g0_rf_mean",
    "train_fs_s0_mean", "train_fs_alpha_eff_mean", "train_fs_dir_norm_mean",
    "train_ewc_grad_norm_mean", "train_cos_fs_ewc_mean",
    "train_projection_removed_norm_mean", "train_projection_removal_ratio_mean",
    "train_projected_final_dir_norm_mean", "lambda_ewc", "log_path", "config",
    "metric_time", "line_number", "run_start_time", "run_end_time",
]


def _is_empty(value: Any) -> bool:
    return value == "" or value is None


def _first_nonempty(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key, "")
        if not _is_empty(value):
            return value
    return ""


def _prefix_present(row: Dict[str, Any], prefix: str) -> bool:
    return any(str(key).startswith(f"{prefix}_") for key in row.keys())


def _fisher_key(prefix: str, suffix: str) -> str:
    if prefix == "accumulated_fisher":
        return f"{prefix}_{suffix}"
    return f"{prefix}_fisher_{suffix}"


def _compact_metric_row(row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    task_key_prefix = "accumulated_fisher" if prefix == "accumulated_fisher" else prefix
    item: Dict[str, Any] = {
        "run_id": row.get("run_id", ""),
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "method_type": row.get("method_type", ""),
        "optimizer": row.get("optimizer", ""),
        "gam_base_optimizer": row.get("gam_base_optimizer", ""),
        "fisher_mode": row.get("fisher_mode", ""),
        "seed": row.get("seed", ""),
        "metric_source": row.get("metric_source", ""),
        "mechanism_task": row.get("mechanism_task", row.get("task", "")),
        "metric_prefix": prefix,
        "metric_task": row.get(f"{task_key_prefix}_task", row.get("mechanism_task", row.get("task", ""))),
        "task_start": row.get(f"{task_key_prefix}_task_start", ""),
        "task_end": row.get(f"{task_key_prefix}_task_end", ""),
        "param_space": row.get(f"{task_key_prefix}_param_space", ""),
        "requested_param_scope": row.get(f"{task_key_prefix}_requested_param_scope", ""),
        "param_count": row.get(f"{task_key_prefix}_param_count", ""),
        "fisher_blocks": row.get(_fisher_key(prefix, "blocks"), ""),
        "fisher_numel": row.get(_fisher_key(prefix, "numel"), ""),
        "fisher_total_mass": row.get(_fisher_key(prefix, "total_mass"), ""),
        "fisher_mean": row.get(_fisher_key(prefix, "mean"), ""),
        "fisher_max": row.get(_fisher_key(prefix, "max"), ""),
        "fisher_top0p001_mass_ratio": row.get(_fisher_key(prefix, "top0p001_mass_ratio"), ""),
        "fisher_top0p01_mass_ratio": row.get(_fisher_key(prefix, "top0p01_mass_ratio"), ""),
        "fisher_top0p05_mass_ratio": row.get(_fisher_key(prefix, "top0p05_mass_ratio"), ""),
        "base_loss": row.get(f"{task_key_prefix}_base_loss", ""),
        "grad_norm": row.get(f"{task_key_prefix}_grad_norm", ""),
        "first_order_sharpness": row.get(f"{task_key_prefix}_first_order_sharpness", ""),
        "sh0_perturbed_loss": row.get(f"{task_key_prefix}_sh0_perturbed_loss", ""),
        "sh0_max": row.get(f"{task_key_prefix}_sh0_max", ""),
        "esh_mean": row.get(f"{task_key_prefix}_esh_mean", ""),
        "esh_std": row.get(f"{task_key_prefix}_esh_std", ""),
        "hessian_batch_size": row.get(f"{task_key_prefix}_hessian_batch_size", ""),
        "emp_fisher_batch_size": row.get(f"{task_key_prefix}_emp_fisher_batch_size", ""),
        "hessian_top_eigen": _first_nonempty(
            row,
            f"{task_key_prefix}_hessian_top_eigen",
            f"{task_key_prefix}_flat_lambda_max_lanczos",
            f"{task_key_prefix}_flat_lambda_max_power",
            f"{task_key_prefix}_flat_lambda_max",
        ),
        "hessian_trace": _first_nonempty(
            row,
            f"{task_key_prefix}_hessian_trace",
            f"{task_key_prefix}_flat_hessian_trace",
        ),
        "emp_fisher_top_eigen": _first_nonempty(
            row,
            f"{task_key_prefix}_emp_fisher_top_eigen",
            f"{task_key_prefix}_flat_emp_fisher_lambda_max_lanczos",
            f"{task_key_prefix}_flat_emp_fisher_lambda_max_power",
        ),
        "emp_fisher_trace": _first_nonempty(
            row,
            f"{task_key_prefix}_emp_fisher_trace",
            f"{task_key_prefix}_flat_emp_fisher_trace",
        ),
        "hessian_emp_fisher_top_eigen_ratio": row.get(
            f"{task_key_prefix}_hessian_emp_fisher_top_eigen_ratio", ""
        ),
        "hessian_emp_fisher_trace_ratio": row.get(
            f"{task_key_prefix}_hessian_emp_fisher_trace_ratio", ""
        ),
        "delta_norm": row.get("delta_norm", ""),
        "delta_norm_sq": row.get("delta_norm_sq", row.get("delta_norm", "")),
        "delta_l2_norm": row.get("delta_l2_norm", ""),
        "fisher_weighted_drift": row.get("fisher_weighted_drift", ""),
        "normalized_fisher_drift": row.get("normalized_fisher_drift", ""),
        "t1_fisher_top1pct_numel": row.get("t1_fisher_top1pct_numel", ""),
        "delta_w_energy_in_t1_top1pct": row.get("delta_w_energy_in_t1_top1pct", ""),
        "d_f_contribution_from_t1_top1pct": row.get("d_f_contribution_from_t1_top1pct", ""),
        "ewc_penalty_value": row.get("ewc_penalty_value", ""),
        "train_ce_mean": row.get("train_ce_mean", row.get("train_clean_loss_mean", "")),
        "train_delta_norm_mean": row.get("train_delta_norm_mean", ""),
        "train_ewc_mean": row.get("train_ewc_mean", ""),
        "train_ewc_ce_ratio_mean": row.get("train_ewc_ce_ratio_mean", ""),
        "train_ewc_penalty_value_mean": row.get("train_ewc_penalty_value_mean", ""),
        "train_fisher_weighted_drift_mean": row.get("train_fisher_weighted_drift_mean", ""),
        "train_normalized_fisher_drift_mean": row.get("train_normalized_fisher_drift_mean", ""),
        "train_cos_gam_ewc_mean": row.get("train_cos_gam_ewc_mean", ""),
        "train_fs_g0_norm_mean": row.get("train_fs_g0_norm_mean", row.get("grad_fs_g0_norm_mean", "")),
        "train_fs_delta_norm_mean": row.get("train_fs_delta_norm_mean", row.get("grad_fs_delta_norm_mean", "")),
        "train_fs_rf_norm_mean": row.get("train_fs_rf_norm_mean", row.get("grad_fs_rf_norm_mean", "")),
        "train_fs_curvature_ratio_mean": row.get("train_fs_curvature_ratio_mean", row.get("grad_fs_curvature_ratio_mean", "")),
        "train_fs_shaping_component_norm_mean": row.get("train_fs_shaping_component_norm_mean", row.get("grad_fs_shaping_component_norm_mean", "")),
        "train_fs_shaping_ratio_mean": row.get("train_fs_shaping_ratio_mean", row.get("grad_fs_shaping_ratio_mean", "")),
        "train_fs_coeff_on_delta_mean": row.get("train_fs_coeff_on_delta_mean", row.get("grad_fs_coeff_on_delta_mean", "")),
        "train_fs_ce_backward_count_mean": row.get("train_fs_ce_backward_count_mean", row.get("grad_fs_ce_backward_count_mean", "")),
        "train_fs_cos_g0_rf_mean": row.get("train_fs_cos_g0_rf_mean", row.get("grad_fs_cos_g0_rf_mean", "")),
        "train_fs_s0_mean": row.get("train_fs_s0_mean", row.get("grad_fs_s0_mean", "")),
        "train_fs_alpha_eff_mean": row.get("train_fs_alpha_eff_mean", row.get("grad_fs_alpha_eff_mean", "")),
        "train_fs_dir_norm_mean": row.get("train_fs_dir_norm_mean", row.get("grad_fs_dir_norm_mean", "")),
        "train_ewc_grad_norm_mean": row.get("train_ewc_grad_norm_mean", row.get("grad_fisher_norm_mean", "")),
        "train_cos_fs_ewc_mean": row.get("train_cos_fs_ewc_mean", row.get("grad_cos_fs_fisher_mean", "")),
        "train_projection_removed_norm_mean": row.get("train_projection_removed_norm_mean", ""),
        "train_projection_removal_ratio_mean": row.get("train_projection_removal_ratio_mean", ""),
        "train_projected_final_dir_norm_mean": row.get("train_projected_final_dir_norm_mean", ""),
        "lambda_ewc": row.get("lambda_ewc", ""),
        "log_path": row.get("log_path", ""),
        "config": row.get("config", ""),
        "metric_time": row.get("metric_time", ""),
        "line_number": row.get("line_number", ""),
        "run_start_time": row.get("run_start_time", ""),
        "run_end_time": row.get("run_end_time", ""),
    }
    fqa_prefix = f"{task_key_prefix}_fqa_"
    for key, value in row.items():
        if str(key).startswith(fqa_prefix):
            item[str(key)[len(f"{task_key_prefix}_"):]] = value
        elif str(key).startswith(("grad_fs_", "grad_fisher_", "grad_cos_fs_")):
            # Preserve the learner's canonical names, including q tail
            # quantiles, while the compatibility aliases above serve older
            # plotting code.
            item[str(key)] = value
    return item


def _prefer_final_mechanism_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    final_keys = {
        (row.get("run_id", ""), row.get("mechanism_task", row.get("task", "")))
        for row in rows
        if row.get("mechanism_final") is True
        or str(row.get("mechanism_final", "")).lower() == "true"
    }
    if not final_keys:
        return rows
    return [
        row
        for row in rows
        if (
            (row.get("run_id", ""), row.get("mechanism_task", row.get("task", "")))
            not in final_keys
            or row.get("mechanism_final") is True
            or str(row.get("mechanism_final", "")).lower() == "true"
        )
    ]


def build_mechanism_metric_summary(
    mechanism_rows: List[Dict[str, Any]],
    anchor_eval_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compact one-row-per-source/prefix summary for the new mechanism metrics."""
    out: List[Dict[str, Any]] = []
    for row in anchor_eval_rows:
        prefix = row.get("metric_prefix") or infer_metric_prefix(row)
        if prefix:
            out.append(_compact_metric_row(row, str(prefix)))

    for row in _prefer_final_mechanism_rows(mechanism_rows):
        for prefix in SUMMARY_METRIC_PREFIXES:
            if _prefix_present(row, prefix):
                out.append(_compact_metric_row(row, prefix))
    return out


LONG_EXCLUDE_KEYS = set(META_KEYS) | {
    "run_id", "run_index_in_log", "log_path", "run_start_time", "run_end_time",
    "max_train_task_seen", "metric_source", "metric_prefix", "metric_task",
    "mechanism_task", "metric_time", "line_number", "task", "method_type",
}


def _long_scope_and_name(key: str) -> Tuple[str, str]:
    for prefix in ("accumulated_fisher", "anchor", "endpoint"):
        marker = f"{prefix}_"
        if key.startswith(marker):
            return prefix, key[len(marker):]
    if key.startswith("train_"):
        return "train", key[len("train_"):]
    return "mechanism", key


def build_mechanism_metrics_long(
    mechanism_rows: List[Dict[str, Any]],
    anchor_eval_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Explode mechanism/AnchorEval dictionaries into a plotting-friendly long table."""
    out: List[Dict[str, Any]] = []
    for source_row in list(anchor_eval_rows) + _prefer_final_mechanism_rows(mechanism_rows):
        for key, value in source_row.items():
            if key in LONG_EXCLUDE_KEYS or _is_empty(value):
                continue
            scope, metric_name = _long_scope_and_name(str(key))
            if metric_name in {"task", "task_start", "task_end"}:
                continue
            metric_value = value if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
            metric_text = "" if metric_value != "" else value
            task = source_row.get(f"{scope}_task", source_row.get("mechanism_task", source_row.get("task", "")))
            out.append({
                "run_id": source_row.get("run_id", ""),
                "dataset": source_row.get("dataset", ""),
                "method": source_row.get("method", ""),
                "method_type": source_row.get("method_type", ""),
                "optimizer": source_row.get("optimizer", ""),
                "gam_base_optimizer": source_row.get("gam_base_optimizer", ""),
                "fisher_mode": source_row.get("fisher_mode", ""),
                "seed": source_row.get("seed", ""),
                "metric_source": source_row.get("metric_source", ""),
                "mechanism_task": source_row.get("mechanism_task", source_row.get("task", "")),
                "metric_scope": scope,
                "metric_task": task,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_text": metric_text,
                "raw_key": key,
                "log_path": source_row.get("log_path", ""),
                "config": source_row.get("config", ""),
                "metric_time": source_row.get("metric_time", ""),
                "line_number": source_row.get("line_number", ""),
                "run_start_time": source_row.get("run_start_time", ""),
                "run_end_time": source_row.get("run_end_time", ""),
            })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = union_fieldnames(rows, preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_write_excel(path: Path, tables: Dict[str, List[Dict[str, Any]]]) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, rows in tables.items():
                if rows:
                    if name == "task_accuracy":
                        cols = requested_fieldnames(rows, [
                            "run_id", "dataset", "method", "method_type", "optimizer",
                            "gam_base_optimizer", "seed", "eval_task_id",
                            "observed_task_count", "total_accuracy", "old_accuracy",
                            "new_accuracy", "log_path", "config_path",
                        ])
                    else:
                        cols = union_fieldnames(rows)
                    pd.DataFrame(rows, columns=cols).to_excel(writer, sheet_name=name[:31], index=False)
                else:
                    pd.DataFrame().to_excel(writer, sheet_name=name[:31], index=False)
        return True
    except Exception as e:
        print(f"[WARN] failed to write Excel file: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CL experiment logs into summary/detail tables.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Log files, directories, or glob patterns. Directories are scanned recursively by default.",
    )
    parser.add_argument(
        "--log-dir",
        action="append",
        default=[],
        help="Directory to scan for logs. Can be repeated. Equivalent to adding the directory to --inputs.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(DEFAULT_LOG_EXTS),
        help="Log file extensions to include when scanning directories or globs. Default: .log .out .txt",
    )
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Scan input directories recursively. This is the default.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only scan the top level of input directories.")
    parser.add_argument("--output-dir", default="parsed_log_tables", help="Directory for generated CSV/XLSX tables.")
    args = parser.parse_args()

    inputs = list(args.inputs) + list(args.log_dir)
    if not inputs:
        parser.error("please provide --inputs and/or --log-dir")

    log_paths = discover_logs(inputs, recursive=args.recursive, exts=normalize_exts(args.extensions))
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, List[Dict[str, Any]]] = {
        "summary": [],
        "eval_wide": [],
        "eval_long": [],
        "final_wide": [],
        "curves_long": [],
        "mechanism_metrics": [],
        "anchor_eval_metrics": [],
        "prototype_eval_wide": [],
        "prototype_eval_long": [],
    }

    if not log_paths:
        print("[ERROR] no log files found.", file=sys.stderr)
        sys.exit(1)

    for log_path in log_paths:
        try:
            lines = read_log(log_path)
        except Exception as e:
            print(f"[WARN] cannot read {log_path}: {e}", file=sys.stderr)
            continue

        runs = split_runs(lines)
        for run_index, (start, end) in enumerate(runs):
            parsed = parse_one_run(lines[start:end], log_path, run_index)
            for name in tables:
                tables[name].extend(parsed[name])

    preferred_common = [
        "run_id", "run_index_in_log", "dataset", "method", "method_type", "optimizer",
        "gam_base_optimizer", "seed", "sessions", "epochs", "rank", "lrate", "fc_lrate",
        "lambda", "lambda_ewc", "lambda_flat", "lambda_flat_lora", "lambda_flat_classifier",
        "gam_grad_rho", "gam_grad_norm_rho", "gam_grad_gamma",
        "config", "log_path", "run_start_time", "run_end_time",
    ]

    run_summary = build_run_summary(tables["summary"])
    requested_tables: Dict[str, List[Dict[str, Any]]] = {
        "run_summary": run_summary,
        "comparison_summary": build_comparison_summary(run_summary),
        "task_accuracy": build_task_accuracy(tables["eval_wide"]),
        "mechanism_metric_summary": build_mechanism_metric_summary(
            tables["mechanism_metrics"],
            tables["anchor_eval_metrics"],
        ),
        "mechanism_metrics_long": build_mechanism_metrics_long(
            tables["mechanism_metrics"],
            tables["anchor_eval_metrics"],
        ),
    }
    output_tables: Dict[str, List[Dict[str, Any]]] = {}
    output_tables.update(requested_tables)
    output_tables.update(tables)

    run_summary_cols = [
        "run_id", "setting", "prefix", "dataset", "method", "model_name",
        "method_type", "optimizer", "gam_base_optimizer", "fs_base_optimizer",
        "use_ewc", "compute_fisher", "ewc_lambda", "fs_rho", "fs_alpha",
        "fs_alpha_mode", "fs_sam_equivalent", "ewc_fisher_damping_beta",
        "seed", "task_count", "expected_task_count", "final_task_id", "AIA", "Accuracy", "old_accuracy",
        "new_accuracy", "Forgetting", "valid_run", "has_nan_loss", "has_error",
        "log_path", "config_path", "run_start_time", "run_end_time",
    ]
    task_accuracy_cols = [
        "run_id", "dataset", "method", "method_type", "optimizer", "gam_base_optimizer",
        "seed", "eval_task_id", "observed_task_count", "total_accuracy", "old_accuracy",
        "new_accuracy", "log_path", "config_path",
    ]
    mechanism_cols = [
        "run_id", "dataset", "method", "method_type", "optimizer", "gam_base_optimizer",
        "seed", "metric_source", "mechanism_task", "lambda_ewc",
        "anchor_param_space", "anchor_requested_param_scope", "anchor_param_count",
        "anchor_task", "anchor_task_start", "anchor_task_end",
        "anchor_base_loss", "anchor_grad_norm",
        "anchor_first_order_sharpness", "anchor_sh0_max", "anchor_esh_mean",
        "anchor_esh_std",
        "anchor_sh0_perturbed_loss", "anchor_hessian_batch_size",
        "anchor_emp_fisher_batch_size",
        "anchor_hessian_top_eigen", "anchor_hessian_trace",
        "anchor_emp_fisher_top_eigen", "anchor_emp_fisher_trace",
        "anchor_hessian_emp_fisher_top_eigen_ratio",
        "anchor_hessian_emp_fisher_trace_ratio",
        "anchor_fisher_blocks", "anchor_fisher_numel",
        "anchor_fisher_total_mass", "anchor_fisher_mean", "anchor_fisher_max",
        "anchor_fisher_top0p001_mass_ratio", "anchor_fisher_top0p01_mass_ratio",
        "anchor_fisher_top0p05_mass_ratio",
        "endpoint_param_space", "endpoint_requested_param_scope", "endpoint_param_count",
        "endpoint_task", "endpoint_task_start", "endpoint_task_end",
        "endpoint_base_loss", "endpoint_grad_norm",
        "endpoint_first_order_sharpness", "endpoint_sh0_max", "endpoint_esh_mean",
        "endpoint_esh_std",
        "endpoint_sh0_perturbed_loss", "endpoint_hessian_batch_size",
        "endpoint_emp_fisher_batch_size",
        "endpoint_hessian_top_eigen", "endpoint_hessian_trace",
        "endpoint_emp_fisher_top_eigen", "endpoint_emp_fisher_trace",
        "endpoint_hessian_emp_fisher_top_eigen_ratio",
        "endpoint_hessian_emp_fisher_trace_ratio",
        "endpoint_fisher_blocks", "endpoint_fisher_numel",
        "endpoint_fisher_total_mass", "endpoint_fisher_mean", "endpoint_fisher_max",
        "endpoint_fisher_top0p001_mass_ratio", "endpoint_fisher_top0p01_mass_ratio",
        "endpoint_fisher_top0p05_mass_ratio",
        "accumulated_fisher_blocks", "accumulated_fisher_numel",
        "accumulated_fisher_total_mass", "accumulated_fisher_mean",
        "accumulated_fisher_max", "accumulated_fisher_top0p001_mass_ratio",
        "accumulated_fisher_top0p01_mass_ratio",
        "accumulated_fisher_top0p05_mass_ratio",
        "fisher_weighted_drift", "normalized_fisher_drift", "ewc_penalty_value",
        "t1_fisher_top1pct_numel", "delta_w_energy_in_t1_top1pct",
        "d_f_contribution_from_t1_top1pct",
        "train_ce_mean", "train_ewc_mean", "train_ewc_ce_ratio_mean",
        "train_delta_norm_mean", "train_ewc_penalty_value_mean",
        "train_fisher_weighted_drift_mean", "train_normalized_fisher_drift_mean",
        "train_cos_gam_ewc_mean", "train_projection_removal_ratio_mean",
        "log_path", "config", "metric_time", "line_number",
        "run_start_time", "run_end_time",
    ]
    anchor_eval_cols = [
        "run_id", "dataset", "method", "method_type", "optimizer", "gam_base_optimizer",
        "seed", "metric_source", "metric_prefix", "metric_task",
        "anchor_task", "anchor_task_start", "anchor_task_end",
        "anchor_param_space", "anchor_requested_param_scope", "anchor_param_count",
        "anchor_base_loss", "anchor_grad_norm", "anchor_first_order_sharpness",
        "anchor_sh0_perturbed_loss", "anchor_sh0_max", "anchor_esh_mean",
        "anchor_esh_std",
        "anchor_hessian_batch_size", "anchor_emp_fisher_batch_size",
        "anchor_hessian_top_eigen", "anchor_hessian_trace",
        "anchor_emp_fisher_top_eigen", "anchor_emp_fisher_trace",
        "anchor_hessian_emp_fisher_top_eigen_ratio",
        "anchor_hessian_emp_fisher_trace_ratio",
        "anchor_fisher_blocks", "anchor_fisher_numel", "anchor_fisher_total_mass",
        "anchor_fisher_mean", "anchor_fisher_max",
        "anchor_fisher_top0p001_mass_ratio", "anchor_fisher_top0p01_mass_ratio",
        "anchor_fisher_top0p05_mass_ratio",
        "endpoint_task", "endpoint_task_start", "endpoint_task_end",
        "endpoint_param_space", "endpoint_requested_param_scope", "endpoint_param_count",
        "endpoint_base_loss", "endpoint_grad_norm", "endpoint_first_order_sharpness",
        "endpoint_sh0_perturbed_loss", "endpoint_sh0_max", "endpoint_esh_mean",
        "endpoint_esh_std",
        "endpoint_hessian_batch_size", "endpoint_emp_fisher_batch_size",
        "endpoint_hessian_top_eigen", "endpoint_hessian_trace",
        "endpoint_emp_fisher_top_eigen", "endpoint_emp_fisher_trace",
        "endpoint_hessian_emp_fisher_top_eigen_ratio",
        "endpoint_hessian_emp_fisher_trace_ratio",
        "endpoint_fisher_blocks", "endpoint_fisher_numel", "endpoint_fisher_total_mass",
        "endpoint_fisher_mean", "endpoint_fisher_max",
        "endpoint_fisher_top0p001_mass_ratio", "endpoint_fisher_top0p01_mass_ratio",
        "endpoint_fisher_top0p05_mass_ratio",
        "log_path", "config", "metric_time", "line_number",
        "run_start_time", "run_end_time",
    ]
    mechanism_long_cols = [
        "run_id", "dataset", "method", "method_type", "optimizer", "gam_base_optimizer",
        "fisher_mode", "seed", "metric_source", "mechanism_task", "metric_scope",
        "metric_task", "metric_name", "metric_value", "metric_text", "raw_key",
        "log_path", "config", "metric_time", "line_number",
        "run_start_time", "run_end_time",
    ]

    for name, rows in output_tables.items():
        if name == "run_summary":
            write_csv(out_dir / f"{name}.csv", rows, run_summary_cols)
        elif name == "comparison_summary":
            write_csv(out_dir / f"{name}.csv", rows)
        elif name == "task_accuracy":
            path = out_dir / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=requested_fieldnames(rows, task_accuracy_cols),
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        elif name == "mechanism_metrics":
            write_csv(out_dir / f"{name}.csv", rows, mechanism_cols)
        elif name == "anchor_eval_metrics":
            write_csv(out_dir / f"{name}.csv", rows, anchor_eval_cols)
        elif name == "mechanism_metric_summary":
            write_csv(out_dir / f"{name}.csv", rows, COMPACT_COMMON_COLS)
        elif name == "mechanism_metrics_long":
            write_csv(out_dir / f"{name}.csv", rows, mechanism_long_cols)
        else:
            write_csv(out_dir / f"{name}.csv", rows, preferred_common)

    xlsx_ok = maybe_write_excel(out_dir / "log_analysis.xlsx", output_tables)

    print(f"[OK] Parsed {len(log_paths)} log file(s).")
    print(f"[OK] Detected {len(tables['summary'])} completed/non-empty run(s) with normal Accuracy output.")
    print(f"[OK] Output directory: {out_dir}")
    print("[OK] CSV files:")
    for name in output_tables:
        print(f"  - {out_dir / (name + '.csv')}  rows={len(output_tables[name])}")
    if xlsx_ok:
        print(f"[OK] Excel file: {out_dir / 'log_analysis.xlsx'}")
    else:
        print("[INFO] Excel file was not generated. Install pandas and openpyxl if needed: pip install pandas openpyxl")


if __name__ == "__main__":
    main()
