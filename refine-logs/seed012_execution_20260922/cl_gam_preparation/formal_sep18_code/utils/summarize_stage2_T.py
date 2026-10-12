#!/usr/bin/env python3
"""Summarize Stage 2-T CL training results to an Excel workbook.

Scans all *_cl_metrics.json files under given input directories.
Produces sheets:
  CL_Summary       — ACA / FAA / AAA / BWT / Forget per run
  Accuracy_step    — cumulative mean accuracy at each step (prefix_mean_t)
  CA_final         — per-task accuracy after all tasks (class-incremental)
  Trajectory       — task-0 oracle vs non-oracle acc + loss delta per step
  Mechanism        — EWC drift / penalty metrics per training step

Usage:
  python utils/summarize_stage2_T.py \\
    --inputs logs/sgd_gam_s1c/staged/stage2_T/Ttraj_... \\
             logs/sgd_gam_s1c/staged/stage2_T/T_... \\
    --output result_summary/stage2_T_summary.xlsx \\
    [--fisher-raw-trace 29.876]   # trace(F_orig) from Stage 1; scales fisher_quad
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nan(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _infer_optimizer(path: str) -> str:
    lowered = path.lower().replace(os.sep, "/")
    for part in reversed(lowered.split("/")):
        if part in ("sgd",) or part.startswith("sgd_") or part.startswith("s1c_sgd_"):
            return "sgd"
        if part in ("gam",) or part.startswith("gam_") or part.startswith("s1c_gam_"):
            return "gam"
    if "/sgd_" in lowered or "_sgd_" in lowered:
        return "sgd"
    if "/gam_" in lowered or "_gam_" in lowered:
        return "gam"
    return ""


def _infer_run_tag(path: str) -> str:
    """Derive a short tag from the JSON path (parent directory of the JSON)."""
    parts = Path(path).parts
    # look for stage2_T segment and take the next part
    for i, p in enumerate(parts):
        if "stage2_T" in p and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: grandparent of the json
    return Path(path).parent.parent.name


def _find_json_files(roots: List[str]) -> List[str]:
    files = []
    for root in roots:
        pattern = os.path.join(root, "**", "*_cl_metrics.json")
        files.extend(sorted(glob.glob(pattern, recursive=True)))
    return files


# ---------------------------------------------------------------------------
# per-file extraction
# ---------------------------------------------------------------------------

def _extract_cl_summary(data: dict, path: str, opt: str, tag: str) -> Dict:
    final = data.get("cnn", {}).get("final", {})
    vecs = final.get("vectors", {})
    row: Dict = {
        "run_tag": tag,
        "optimizer": opt,
        "json_path": path,
        "ACA":          _nan(final.get("ACA")),
        "FAA":          _nan(final.get("FAA")),
        "AAA":          _nan(final.get("AAA")),
        "BWT_final_avg": _nan(final.get("BWT_final_avg")),
        "Forget_avg":    _nan(final.get("Forget_avg")),
        "ABWT":          _nan(final.get("ABWT")),
    }
    # per-step cumulative mean (prefix_mean_t) at final step
    pmt = vecs.get("prefix_mean_t", [])
    if pmt:
        row["final_prefix_mean"] = _nan(pmt[-1])
        row["n_tasks"] = len(pmt)
    # BWT_final_per_task list → spread
    for i, v in enumerate(final.get("BWT_final_per_task", [])):
        row[f"BWT_task{i}"] = _nan(v)
    return row


def _extract_accuracy_per_step(data: dict, path: str, opt: str, tag: str) -> List[Dict]:
    final = data.get("cnn", {}).get("final", {})
    pmt = final.get("vectors", {}).get("prefix_mean_t", [])
    bwt_t = final.get("BWT", [])
    rows = []
    for step, val in enumerate(pmt):
        rows.append({
            "run_tag": tag,
            "optimizer": opt,
            "step": step,
            "cumulative_mean_acc": _nan(val),
            "BWT_at_step": _nan(bwt_t[step]) if step < len(bwt_t) else float("nan"),
        })
    return rows


def _extract_ca_final(data: dict, path: str, opt: str, tag: str) -> List[Dict]:
    ca = data.get("cnn", {}).get("final", {}).get("CA", [])
    fg = data.get("cnn", {}).get("final", {}).get("vectors", {}).get("Forget_per_task", [])
    rows = []
    for task, val in enumerate(ca):
        rows.append({
            "run_tag": tag,
            "optimizer": opt,
            "task": task,
            "CA_acc": _nan(val),
            "forget": _nan(fg[task]) if task < len(fg) else float("nan"),
        })
    return rows


def _extract_trajectory(data: dict, path: str, opt: str, tag: str,
                         fisher_raw_trace: float) -> List[Dict]:
    traj = data.get("task0_trajectory", {}).get("steps", {})
    if not traj:
        return []
    rows = []
    for step_key in sorted(traj.keys(), key=int):
        s = traj[step_key]
        fq = _nan(s.get("fisher_quad", float("nan")))
        rows.append({
            "run_tag": tag,
            "optimizer": opt,
            "step": int(step_key),
            # task-oracle accuracy (logits[:, task_start:task_end])
            "train_local_acc":       _nan(s.get("train_local_acc")),
            "test_local_acc":        _nan(s.get("test_local_acc")),
            # class-incremental accuracy (full logit argmax)
            "train_full_acc":        _nan(s.get("train_full_acc")),
            "test_full_acc":         _nan(s.get("test_full_acc")),
            # loss increase vs task-0 anchor (local = task-oracle CE)
            "train_local_loss_delta": _nan(s.get("train_local_loss_delta")),
            "test_local_loss_delta":  _nan(s.get("test_local_loss_delta")),
            # loss increase (full CE, all classes)
            "train_full_loss_delta":  _nan(s.get("train_full_loss_delta")),
            "test_full_loss_delta":   _nan(s.get("test_full_loss_delta")),
            # LoRA delta-W movement
            "delta_norm":            _nan(s.get("delta_norm")),
            # EWC Fisher quadratic (with normalized F̃, trace≈1)
            "fisher_quad_normalized": fq,
            # Rescaled with raw trace: approximates 0.5 * ΔW^T F_orig ΔW
            "fisher_quad_raw":       fq * fisher_raw_trace if not math.isnan(fq) else float("nan"),
            "fisher_raw_trace_used": fisher_raw_trace,
            "normalized_fisher_drift": _nan(s.get("normalized_fisher_drift")),
            "lambda_fisher_quad":    _nan(s.get("lambda_fisher_quad")),
            "fisher_available":      int(s.get("fisher_available", 0)),
        })
    return rows


def _extract_mechanism(data: dict, path: str, opt: str, tag: str) -> List[Dict]:
    steps = data.get("mechanism", {}).get("steps", {})
    rows = []
    for step_key in sorted(steps.keys(), key=int):
        s = steps[step_key]
        rows.append({
            "run_tag": tag,
            "optimizer": opt,
            "step": int(step_key),
            "delta_norm":              _nan(s.get("delta_norm")),
            "normalized_fisher_drift": _nan(s.get("normalized_fisher_drift")),
            "ewc_penalty_value":       _nan(s.get("ewc_penalty_value")),
            "train_clean_loss_mean":   _nan(s.get("train_clean_loss_mean")),
            "train_fisher_loss_mean":  _nan(s.get("train_fisher_loss_mean")),
            "fisher_weighted_drift":   _nan(s.get("fisher_weighted_drift")),
            "high_fisher_delta_energy_ratio": _nan(s.get("high_fisher_delta_energy_ratio")),
            "lambda_ewc":              _nan(s.get("lambda_ewc")),
        })
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Root directories to scan for *_cl_metrics.json files.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output Excel (.xlsx) path.",
    )
    parser.add_argument(
        "--fisher-raw-trace", type=float, default=1.0,
        dest="fisher_raw_trace",
        help=(
            "trace(F_orig) from Stage 1 (in_task_fisher_final_raw_trace). "
            "Used to rescale fisher_quad from normalized F̃ to F_orig. "
            "Default=1.0 (no rescaling). For SGD/GAM shared Stage 1 with "
            "ewc_normalize_fisher=True, this is typically ~29.9."
        ),
    )
    args = parser.parse_args()

    json_files = _find_json_files(args.inputs)
    if not json_files:
        print(f"[WARN] No *_cl_metrics.json files found under: {args.inputs}")
        return

    cl_rows: List[Dict] = []
    acc_step_rows: List[Dict] = []
    ca_rows: List[Dict] = []
    traj_rows: List[Dict] = []
    mech_rows: List[Dict] = []

    for path in json_files:
        opt = _infer_optimizer(path)
        tag = _infer_run_tag(path)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
            continue

        cl_rows.append(_extract_cl_summary(data, path, opt, tag))
        acc_step_rows.extend(_extract_accuracy_per_step(data, path, opt, tag))
        ca_rows.extend(_extract_ca_final(data, path, opt, tag))
        traj_rows.extend(_extract_trajectory(data, path, opt, tag, args.fisher_raw_trace))
        mech_rows.extend(_extract_mechanism(data, path, opt, tag))

        has_traj = bool(data.get("task0_trajectory", {}).get("steps"))
        print(f"[OK] {opt:>4} | {tag} | trajectory={'yes' if has_traj else 'no '}")

    # build DataFrames
    df_cl   = pd.DataFrame(cl_rows).sort_values(["run_tag", "optimizer"])
    df_acc  = pd.DataFrame(acc_step_rows).sort_values(["run_tag", "optimizer", "step"])
    df_ca   = pd.DataFrame(ca_rows).sort_values(["run_tag", "optimizer", "task"])
    df_traj = pd.DataFrame(traj_rows).sort_values(["run_tag", "optimizer", "step"])
    df_mech = pd.DataFrame(mech_rows).sort_values(["run_tag", "optimizer", "step"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        df_cl.to_excel(writer,   sheet_name="CL_Summary",    index=False)
        df_acc.to_excel(writer,  sheet_name="Accuracy_step",  index=False)
        df_ca.to_excel(writer,   sheet_name="CA_final",       index=False)
        df_traj.to_excel(writer, sheet_name="Trajectory",     index=False)
        df_mech.to_excel(writer, sheet_name="Mechanism",      index=False)

        # convenience pivot: SGD vs GAM side-by-side for trajectory
        if not df_traj.empty and "step" in df_traj.columns:
            key_metrics = [
                "test_local_acc", "test_full_acc",
                "test_local_loss_delta", "test_full_loss_delta",
                "train_local_acc", "train_full_acc",
                "delta_norm", "fisher_quad_normalized", "fisher_quad_raw",
            ]
            pivot_rows = []
            for step in sorted(df_traj["step"].unique()):
                row: Dict = {"step": step}
                for opt in ["sgd", "gam"]:
                    subset = df_traj[
                        (df_traj["step"] == step) & (df_traj["optimizer"] == opt)
                    ]
                    for m in key_metrics:
                        val = subset[m].values[0] if (len(subset) > 0 and m in subset.columns) else float("nan")
                        row[f"{opt}_{m}"] = val
                # diffs GAM - SGD
                for m in key_metrics:
                    g = row.get(f"gam_{m}", float("nan"))
                    s = row.get(f"sgd_{m}", float("nan"))
                    if not (math.isnan(g) or math.isnan(s)):
                        row[f"diff_gam_minus_sgd_{m}"] = g - s
                pivot_rows.append(row)
            pd.DataFrame(pivot_rows).to_excel(
                writer, sheet_name="Traj_pivot_SGD_vs_GAM", index=False
            )

        # convenience pivot: per-step cumulative accuracy SGD vs GAM
        if not df_acc.empty:
            pivot_acc_rows = []
            for step in sorted(df_acc["step"].unique()):
                row = {"step": step}
                for opt in ["sgd", "gam"]:
                    sub = df_acc[(df_acc["step"] == step) & (df_acc["optimizer"] == opt)]
                    # take the one from the Ttraj run if multiple
                    if len(sub) > 0:
                        row[f"{opt}_cumul_acc"] = sub["cumulative_mean_acc"].values[0]
                        row[f"{opt}_BWT_at_step"] = sub["BWT_at_step"].values[0]
                pivot_acc_rows.append(row)
            pd.DataFrame(pivot_acc_rows).to_excel(
                writer, sheet_name="Acc_pivot_SGD_vs_GAM", index=False
            )

    print(f"\n[OK] Written {out_path}")
    print(f"     CL_Summary    : {len(df_cl)} rows")
    print(f"     Accuracy_step : {len(df_acc)} rows")
    print(f"     CA_final      : {len(df_ca)} rows")
    print(f"     Trajectory    : {len(df_traj)} rows")
    print(f"     Mechanism     : {len(df_mech)} rows")


if __name__ == "__main__":
    main()
