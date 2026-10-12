#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect ImageNet-R LoRA Factor-FS logs into CSV/XLSX tables.

This is intentionally narrower than ``utils/analyze_cl_logs.py``: it focuses on
the current ImageNet-R Factor-FS / SAM-equivalent / SGD-equivalent experiments
and exposes the exact hyperparameters needed for method comparison.

Example:
  python utils/collect_imagenetr_factorfs_excel.py \
    --log-dir logs_our_joost/imagenetr_lora_factorfs_dampnorm \
    --output logs_our_joost/imagenetr_lora_factorfs_dampnorm/imagenetr_factorfs_summary.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KV_RE = re.compile(r"=>\s*([A-Za-z0-9_\-]+)\s*:\s*(.*)\s*$")


CONFIG_KEYS = [
    "config",
    "prefix",
    "dataset",
    "class_shuffle",
    "init_cls",
    "increment",
    "nb_tasks",
    "total_sessions",
    "seed",
    "model_name",
    "backbone_type",
    "lora_rank",
    "optimizer",
    "optimizer_type",
    "fs_base_optimizer",
    "sam_base_optimizer",
    "batch_size",
    "init_epoch",
    "epochs",
    "init_lr",
    "lrate",
    "momentum",
    "weight_decay",
    "scheduler",
    "fs_rho",
    "sam_rho",
    "fs_alpha",
    "fs_alpha_mode",
    "fs_sam_equivalent",
    "fs_xi",
    "fs_adaptive",
    "ewc_lambda",
    "ewc_gamma",
    "ewc_eta",
    "ewc_max_batches",
    "fisher_mode",
    "ewc_true_fisher_mc_samples",
    "ewc_normalize_fisher",
    "ewc_normalize_accumulated_fisher",
    "ewc_fisher_norm_eps",
    "ewc_fisher_damping_beta",
    "ewc_skip_task_fisher",
    "mechanism_eval",
    "optimizer_tag_override",
]


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "\n")


def cast_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+|\d+)([eE][-+]?\d+)?", value):
        try:
            return float(value)
        except Exception:
            return value
    return value


def parse_np_float_dict_line(line: str, prefix: str) -> Optional[Dict[str, float]]:
    if prefix not in line:
        return None
    out: Dict[str, float] = {}
    # Matches "'total': np.float64(77.53)" and "'80-99': nan".
    for key, val, nan_val in re.findall(r"'([^']+)':\s*(?:np\.float64\(([^)]+)\)|(nan))", line):
        raw = val if val != "" else nan_val
        try:
            out[key] = float(raw)
        except Exception:
            out[key] = math.nan
    return out if out else None


def last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def infer_from_name(path: Path) -> Dict[str, Any]:
    name = path.name
    info: Dict[str, Any] = {"log_stem": path.stem}
    seed = re.search(r"_s(\d+)_gpu", name)
    gpu = re.search(r"_gpu(\d+)\.log$", name)
    lam = re.search(r"_lam([^_]+)_s\d+_gpu", name)
    if seed:
        info["seed_from_name"] = int(seed.group(1))
    if gpu:
        info["gpu_from_name"] = int(gpu.group(1))
    if lam:
        info["lambda_tag"] = lam.group(1)
    return info


def infer_method_group(path: Path, cfg: Dict[str, Any]) -> Tuple[str, str]:
    name = path.name
    model = str(cfg.get("model_name", ""))
    sameq = cfg.get("fs_sam_equivalent", "")
    alpha = cfg.get("fs_alpha", "")
    lam = float(cfg.get("ewc_lambda") or 0)

    if "sgdplain_oldlearner" in name:
        return "no-EWC" if lam == 0 else "old SGD+EWC", "old 1x SGD"
    if "samref_direct" in name:
        return "no-EWC" if lam == 0 else "direct SAM+EWC", "direct SAM"
    if sameq is True or str(sameq).lower() == "true":
        return "no-EWC" if lam == 0 else "SAM+EWC", "SAM-equivalent"
    try:
        a = float(alpha)
    except Exception:
        a = math.nan
    if not math.isnan(a) and abs(a) < 1e-12:
        return "no-EWC" if lam == 0 else "SGD+EWC", "Factor SGD"
    if model == "as1_normfisher_fishershape":
        return "no-EWC" if lam == 0 else "ratio-FS+EWC", "ratio-FS"
    return "unknown", "unknown"


def parse_log(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw)
    lines = text.splitlines()

    cfg: Dict[str, Any] = {k: "" for k in CONFIG_KEYS}
    for line in lines:
        kv = KV_RE.search(line)
        if kv:
            key, value = kv.group(1), kv.group(2)
            if key in cfg:
                cfg[key] = cast_value(value)

    row: Dict[str, Any] = {
        "log_path": str(path),
        "log_file": path.name,
        "file_size": path.stat().st_size,
    }
    row.update(infer_from_name(path))
    row.update(cfg)

    method_group, direction = infer_method_group(path, cfg)
    row["method_group"] = method_group
    row["direction"] = direction

    has_traceback = bool(re.search(r"Traceback|RuntimeError|CUDA out of memory|ValueError", text))
    has_nan_failure = bool(re.search(r"=>\s*Loss\s+nan|=>\s*Loss\s+NaN", text, flags=re.IGNORECASE))
    row["has_error"] = has_traceback
    row["has_nan_failure"] = has_nan_failure

    avg = last_float(r"Average Accuracy \(CNN\): ([0-9.]+)", text)
    forgetting = last_float(r"Forgetting \(CNN\): ([0-9.]+)", text)
    row["Avg"] = avg
    row["Forgetting"] = forgetting

    cnn_dicts = [d for line in lines if (d := parse_np_float_dict_line(line, "CNN:"))]
    nme_dicts = [d for line in lines if (d := parse_np_float_dict_line(line, "NME:"))]
    if cnn_dicts:
        final = cnn_dicts[-1]
        row["Final"] = final.get("total")
        row["Final_old"] = final.get("old")
        row["Final_new"] = final.get("new")
        for key in sorted(k for k in final if re.fullmatch(r"\d+-\d+", k)):
            row[f"Final_{key}"] = final.get(key)
    else:
        row["Final"] = None
        row["Final_old"] = None
        row["Final_new"] = None

    if nme_dicts:
        nme = nme_dicts[-1]
        row["NME_Final"] = nme.get("total")
        row["NME_old"] = nme.get("old")
        row["NME_new"] = nme.get("new")

    # Latest training progress. Supports incomplete logs.
    epoch_pat = re.compile(
        r"Task\s+(\d+),\s+Epoch\s+(\d+)/(\d+)\s+=>\s+Loss\s+([^,\s]+),\s+Train_accy\s+([^,\s]+)"
        r"(?:,\s+Test_accy\s+([^,\s]+),\s+CurTask_Test\s+([^,\s]+))?"
    )
    epoch_matches = epoch_pat.findall(text)
    if epoch_matches:
        task, epoch, total_epoch, loss, train, test, cur = epoch_matches[-1]
        row.update(
            {
                "latest_task": int(task),
                "latest_epoch": int(epoch),
                "latest_total_epoch": int(total_epoch),
                "latest_loss": cast_value(loss),
                "latest_train_accy": cast_value(train),
                "latest_test_accy": cast_value(test) if test else None,
                "latest_cur_task_test": cast_value(cur) if cur else None,
            }
        )

    if avg is not None and row.get("Final") is not None and not has_traceback and not has_nan_failure:
        row["status"] = "completed"
    elif has_traceback or has_nan_failure:
        row["status"] = "failed_or_nan"
    elif epoch_matches:
        row["status"] = "running_or_incomplete"
    else:
        row["status"] = "no_metrics"

    return row


def discover_logs(log_dir: Path) -> List[Path]:
    logs = []
    for path in sorted(log_dir.glob("*.log")):
        # Queue logs are launch wrappers, not experiment result logs.
        if "queue" in path.name or path.name.startswith("multiseed_"):
            continue
        logs.append(path)
    return logs


def write_outputs(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    completed = df[df["status"].eq("completed")].copy()
    multiseed = df[df["seed_from_name"].fillna(-1).astype(int).isin([0, 1, 2, 3, 4])].copy()
    key_cols = [
        "method_group",
        "direction",
        "seed_from_name",
        "gpu_from_name",
        "lambda_tag",
        "ewc_lambda",
        "fs_alpha",
        "fs_sam_equivalent",
        "fisher_mode",
        "init_lr",
        "lrate",
        "Avg",
        "Final",
        "Final_old",
        "Final_new",
        "Forgetting",
        "latest_task",
        "latest_epoch",
        "latest_test_accy",
        "latest_cur_task_test",
        "status",
        "log_file",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_runs", index=False)
        df[key_cols].to_excel(writer, sheet_name="compact", index=False)
        completed[key_cols].to_excel(writer, sheet_name="completed", index=False)
        multiseed[key_cols].to_excel(writer, sheet_name="multiseed_status", index=False)

        if not completed.empty:
            group_cols = ["method_group", "direction", "ewc_lambda", "fs_alpha", "fs_sam_equivalent", "fisher_mode"]
            group_cols = [c for c in group_cols if c in completed.columns]
            metrics = ["Avg", "Final", "Final_old", "Final_new", "Forgetting"]
            summary = (
                completed.groupby(group_cols, dropna=False)[metrics]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            summary.columns = [
                "_".join(str(x) for x in col if str(x) != "") if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]
            summary.to_excel(writer, sheet_name="mean_std_completed", index=False)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs_our_joost/imagenetr_lora_factorfs_dampnorm/imagenetr_factorfs_summary.xlsx"),
    )
    args = parser.parse_args(argv)

    logs = discover_logs(args.log_dir)
    rows = [parse_log(path) for path in logs]
    df = pd.DataFrame(rows)

    # Stable, report-friendly order.
    sort_cols = [c for c in ["method_group", "direction", "ewc_lambda", "fs_alpha", "seed_from_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    write_outputs(df, args.output)
    print(f"parsed_logs={len(logs)}")
    print(f"xlsx={args.output}")
    print(f"csv={args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
