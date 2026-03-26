"""
Summarize Order-1 CL evaluation results into an Excel file.

Usage:
    python scripts_order1/summarize_order1.py --run_root <path>

Reads:
    {run_root}/{stage}/evalFlat-{task}-trainedLoRAwhole/all_results.json

Outputs:
    {run_root}/summary.xlsx  (sheets: Matrix, Metrics, PerStep)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# ─── Order-1 task sequence ────────────────────────────────────────────────────
STAGES = ["1-dbpedia", "2-amazon", "3-yahoo", "4-agnews"]
TASKS  = ["dbpedia",   "amazon",   "yahoo",   "agnews"]

# ─── CL metrics ───────────────────────────────────────────────────────────────
def compute_cl_metrics(M: np.ndarray) -> dict:
    """
    M[i, j] = accuracy of model trained up to task i, evaluated on task j.
    Only the lower-left triangle (j <= i) is meaningful for CL.

    Returns:
        ACC_T          : final model mean over all seen tasks (FAA)
        Avg_ACC        : mean of per-step ACC_t
        BWT            : backward transfer (mean of M[T-1,j] - M[j,j] for j<T-1)
        FWT            : forward transfer (mean of upper-triangle)
        Forgetting_avg : mean of (M[j,j] - M[T-1,j]) for j<T-1
        ACC_diag       : mean of diagonal (peak per-task accuracy)
    """
    M = np.asarray(M, dtype=float)
    T = M.shape[0]

    acc_t_list = [float(np.nanmean(M[t, :t + 1])) for t in range(T)]
    acc_T = acc_t_list[-1]
    avg_acc = float(np.nanmean(acc_t_list))

    if T > 1:
        diag = np.array([M[i, i] for i in range(T - 1)], dtype=float)
        bwt = float(np.nanmean(M[T - 1, :T - 1] - diag))
        forgetting_avg = float(np.nanmean(diag - M[T - 1, :T - 1]))
        fwt = float(np.nanmean(M[np.triu_indices(T, k=1)]))
    else:
        bwt = forgetting_avg = fwt = float("nan")

    acc_diag = float(np.nanmean(np.diag(M)))

    return {
        "ACC_T":          acc_T,
        "Avg_ACC":        avg_acc,
        "BWT":            bwt,
        "FWT":            fwt,
        "Forgetting_avg": forgetting_avg,
        "ACC_diag":       acc_diag,
        "ACC_t_list":     acc_t_list,
    }


# ─── Result reading ───────────────────────────────────────────────────────────
def read_exact_match(run_root: str) -> np.ndarray:
    """Build T×T accuracy matrix from evalFlat result files."""
    T = len(STAGES)
    matrix = np.full((T, T), np.nan)

    for i, stage in enumerate(STAGES):
        for j, task in enumerate(TASKS):
            if j > i:
                continue  # task not yet seen at this stage
            result_dir = os.path.join(run_root, stage,
                                      f"evalFlat-{task}-trainedLoRAwhole")
            for fname in ("all_results.json", "all_results_2.json"):
                fpath = os.path.join(result_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath) as f:
                        data = json.load(f)
                    key = f"predict_exact_match_for_{task}"
                    val = data.get(key)
                    if val is not None:
                        matrix[i, j] = float(val)
                    break
            else:
                print(f"  [warn] missing: {result_dir}")

    return matrix


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True,
                        help="Root directory of the training run")
    args = parser.parse_args()

    run_root = args.run_root
    print(f"\nSummarizing: {run_root}\n")

    matrix = read_exact_match(run_root)

    # ── Matrix sheet ──────────────────────────────────────────────────────────
    matrix_df = pd.DataFrame(matrix, index=STAGES, columns=TASKS)
    matrix_df["Stage_Avg"] = [
        float(np.nanmean(matrix[i, :i + 1])) for i in range(len(STAGES))
    ]
    matrix_df.index.name = "Stage \\ Task"

    # ── Metrics sheet ─────────────────────────────────────────────────────────
    m = compute_cl_metrics(matrix)
    metrics_df = pd.DataFrame([
        {"Metric": "ACC_T (FAA)",      "Value": m["ACC_T"]},
        {"Metric": "Avg_ACC",          "Value": m["Avg_ACC"]},
        {"Metric": "BWT",              "Value": m["BWT"]},
        {"Metric": "FWT",              "Value": m["FWT"]},
        {"Metric": "Forgetting_avg",   "Value": m["Forgetting_avg"]},
        {"Metric": "ACC_diag",         "Value": m["ACC_diag"]},
    ])

    # ── PerStep sheet ─────────────────────────────────────────────────────────
    per_step_df = pd.DataFrame({
        "Step":   list(range(1, len(STAGES) + 1)),
        "Stage":  STAGES,
        "ACC_t":  m["ACC_t_list"],
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(run_root, "summary.xlsx")
    os.makedirs(run_root, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        matrix_df.to_excel(writer, sheet_name="Matrix")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        per_step_df.to_excel(writer, index=False, sheet_name="PerStep")

    # ── Print to stdout ───────────────────────────────────────────────────────
    print("─" * 60)
    print(f"Accuracy Matrix (rows=stage, cols=task):")
    print(matrix_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("CL Metrics:")
    for _, row in metrics_df.iterrows():
        v = row["Value"]
        print(f"  {row['Metric']:<22} {v:.4f}" if not np.isnan(v) else
              f"  {row['Metric']:<22} N/A")
    print("─" * 60)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
