#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect ImageNet-A LoRA Factor-FS logs into CSV/XLSX tables.

The parser reuses the field extraction from the ImageNet-R collector, while
adding strict continual-learning completion checks: a run is completed only
when all expected task endpoints and the final CNN metrics are present.

Example:
  python utils/collect_imageneta_factorfs_excel.py \
    --log-dir logs_our_joost/imageneta_lora_factorfs_dampnorm \
    --output logs_our_joost/imageneta_lora_factorfs_dampnorm/imageneta_factorfs_summary.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    from utils.collect_imagenetr_factorfs_excel import clean_text, discover_logs, parse_log
except ModuleNotFoundError:
    from collect_imagenetr_factorfs_excel import clean_text, discover_logs, parse_log


TASK_ENDPOINT_RE = re.compile(r"(?m)^Average Accuracy \(CNN\):\s*[-+0-9.eE]+\s*$")


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fix_imageneta_method_labels(row: Dict[str, Any]) -> None:
    """Handle old-learner filenames not covered by the shared parser."""

    name = str(row.get("log_file", ""))
    lam = float(row.get("ewc_lambda") or 0.0)
    if "sgdref_oldlearner_1x" in name:
        row["method_group"] = "old SGD+EWC" if lam > 0 else "no-EWC"
        row["direction"] = "old 1x SGD"
    elif "sgdref_gamlearner_1x" in name:
        row["method_group"] = "failed old SGD+EWC"
        row["direction"] = "GAM learner + SGD"


def parse_imageneta_log(path: Path) -> Dict[str, Any]:
    row = parse_log(path)
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    completed_tasks = len(TASK_ENDPOINT_RE.findall(text))
    expected_tasks = _as_positive_int(
        row.get("nb_tasks") or row.get("total_sessions"),
        default=10,
    )

    row["completed_tasks"] = completed_tasks
    row["expected_tasks"] = expected_tasks
    row["is_full_task_sequence"] = completed_tasks >= expected_tasks
    _fix_imageneta_method_labels(row)

    has_final_metrics = all(
        _finite_number(row.get(key)) for key in ("Avg", "Final", "Forgetting")
    )
    if bool(row.get("has_error")) or bool(row.get("has_nan_failure")):
        row["status"] = "failed_or_nan"
    elif completed_tasks >= expected_tasks and has_final_metrics:
        row["status"] = "completed"
    elif completed_tasks > 0 or row.get("latest_task") is not None:
        row["status"] = "running_or_incomplete"
    else:
        row["status"] = "no_metrics"

    return row


def _available(columns: Iterable[str], frame: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in frame.columns]


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def _format_workbook(path: Path) -> None:
    """Apply lightweight navigation formatting without changing values."""

    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            values = [str(cell.value) for cell in column_cells[:200] if cell.value is not None]
            width = min(max((len(value) for value in values), default=8) + 2, 48)
            sheet.column_dimensions[column_cells[0].column_letter].width = max(width, 10)
    workbook.save(path)


def write_outputs(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    completed = df[df["status"].eq("completed")].copy()
    seeds = pd.to_numeric(df.get("seed_from_name"), errors="coerce")
    multiseed = df[seeds.isin([0, 1, 2, 3, 4])].copy()

    compact_columns = _available(
        [
            "method_group",
            "direction",
            "seed_from_name",
            "gpu_from_name",
            "completed_tasks",
            "expected_tasks",
            "init_epoch",
            "epochs",
            "init_lr",
            "lrate",
            "fs_rho",
            "sam_rho",
            "lambda_tag",
            "ewc_lambda",
            "ewc_gamma",
            "fs_alpha",
            "fs_sam_equivalent",
            "fisher_mode",
            "ewc_true_fisher_mc_samples",
            "ewc_normalize_fisher",
            "ewc_normalize_accumulated_fisher",
            "ewc_fisher_damping_beta",
            "Avg",
            "Final",
            "Final_old",
            "Final_new",
            "Forgetting",
            "NME_Final",
            "latest_task",
            "latest_epoch",
            "latest_test_accy",
            "latest_cur_task_test",
            "has_error",
            "has_nan_failure",
            "status",
            "log_file",
        ],
        df,
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_runs", index=False)
        df[compact_columns].to_excel(writer, sheet_name="compact", index=False)
        completed[compact_columns].to_excel(writer, sheet_name="completed", index=False)
        multiseed[compact_columns].to_excel(writer, sheet_name="multiseed_status", index=False)

        if completed.empty:
            pd.DataFrame(columns=["status"]).to_excel(
                writer,
                sheet_name="mean_std_completed",
                index=False,
            )
        else:
            group_columns = _available(
                [
                    "method_group",
                    "direction",
                    "init_epoch",
                    "epochs",
                    "init_lr",
                    "lrate",
                    "fs_rho",
                    "sam_rho",
                    "ewc_lambda",
                    "ewc_gamma",
                    "fs_alpha",
                    "fs_sam_equivalent",
                    "fisher_mode",
                    "ewc_true_fisher_mc_samples",
                    "ewc_normalize_fisher",
                    "ewc_normalize_accumulated_fisher",
                    "ewc_fisher_damping_beta",
                ],
                completed,
            )
            metrics = _available(
                ["Avg", "Final", "Final_old", "Final_new", "Forgetting", "NME_Final"],
                completed,
            )
            summary = (
                completed.groupby(group_columns, dropna=False)[metrics]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            _flatten_columns(summary).to_excel(
                writer,
                sheet_name="mean_std_completed",
                index=False,
            )

    _format_workbook(output)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs_our_joost/imageneta_lora_factorfs_dampnorm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs_our_joost/imageneta_lora_factorfs_dampnorm/imageneta_factorfs_summary.xlsx"
        ),
    )
    args = parser.parse_args(argv)

    logs = discover_logs(args.log_dir)
    rows = [parse_imageneta_log(path) for path in logs]
    df = pd.DataFrame(rows)

    sort_columns = _available(
        [
            "status",
            "method_group",
            "direction",
            "init_epoch",
            "init_lr",
            "lrate",
            "ewc_lambda",
            "fisher_mode",
            "ewc_true_fisher_mc_samples",
            "fs_alpha",
            "seed_from_name",
        ],
        df,
    )
    if sort_columns:
        df = df.sort_values(sort_columns, kind="mergesort", na_position="last")

    write_outputs(df, args.output)
    status_counts = df["status"].value_counts().to_dict()
    print(f"parsed_logs={len(logs)}")
    for status in ("completed", "running_or_incomplete", "failed_or_nan", "no_metrics"):
        print(f"{status}={int(status_counts.get(status, 0))}")
    print(f"xlsx={args.output}")
    print(f"csv={args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
