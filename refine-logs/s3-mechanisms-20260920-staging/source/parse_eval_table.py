"""
Parse eval results from 6 method logs and print a comprehensive table.

Accuracy metrics (from CNN Accuracy Matrix):
  a_11 = task0 acc after task0 training
  a_21 = task0 acc after task1 training  (forgetting measure)
  a_22 = task1 acc after task1 training
  FAA  = (a_21 + a_22) / 2               (Final Average Accuracy)
  BWT  = a_11 - a_21                      (Backward Transfer / Forgetting)
  AAA  = (a_11 + FAA) / 2                 (All-round Average Accuracy)

Flatness / mechanism metrics (from EWConProbe Mechanism dict at endpoint_task=1):
  endpoint_sh0_max               -- sharpness (small rho, gradient direction)
  endpoint_flat_sh0_max          -- sharpness (rho=0.05, full-param grad dir)
  endpoint_flat_Sh_param_full    -- Sh_param_full (same direction, rho=0.05)
  endpoint_grad_norm             -- gradient norm (endpoint scope)
  endpoint_first_order_sharpness -- first-order sharpness (endpoint scope)
  endpoint_flat_grad_norm        -- gradient norm (all batches)
  endpoint_flat_first_order_sharpness
  endpoint_flat_lambda_max_lanczos   -- Hessian top eigenvalue (Lanczos)
  endpoint_flat_hessian_trace
  endpoint_flat_emp_fisher_lambda_max_lanczos
  endpoint_flat_emp_fisher_trace
  delta_norm / param_delta_norm
  fisher_weighted_drift / param_fisher_weighted_drift
  normalized_fisher_drift / param_normalized_fisher_drift
  ewc_penalty_value
  d_f_contribution_from_t1_top1pct
"""

import re
import ast
import numpy as np
from pathlib import Path

BASE = Path("/data/140-1/users/liying/CopyFlatCV/logs")

METHODS = [
    ("SGD",           BASE / "verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/baseline_seed0.log"),
    ("GAM",           BASE / "verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/gam_only_seed0.log"),
    ("EWC",           BASE / "verify_imagenet_r_1x1_fullmodel_ewcon_stage2_lam10000/stage2_lam10000_from_stage1lam2000_rho0p02_norm0p2_gamma0p03_lr0p01_e50/shell/control_seed0.log"),
    ("OurMethod",     BASE / "verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_full_seed0.log"),
    ("ExpSet1",       BASE / "verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s1_seed0.log"),
    ("ExpSet2",       BASE / "verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s2_seed0.log"),
]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    result = {}

    # ------------------------------------------------------------------ #
    # 1. CNN Accuracy Matrix                                               #
    # ------------------------------------------------------------------ #
    # The log contains both CNN and NME accuracy matrices.
    # We want only the CNN one. Strategy: find the line index of
    # "Accuracy Matrix (CNN Evaluation" and then scan forward for the
    # two matrix rows: [[a_11 nan] and [a_21 a_22]].
    # Take the LAST such CNN block.
    row0_pat = re.compile(r"\[\[(\d+\.\d+)\s+nan\]")
    row1_pat = re.compile(r"\[\s*(\d+\.\d+)\s+(\d+\.\d+)\]\]")

    a_11 = a_21 = a_22 = None
    cnn_header_idx = [
        i for i, ln in enumerate(lines)
        if "Accuracy Matrix (CNN Evaluation" in ln
    ]
    for hi in reversed(cnn_header_idx):
        # scan up to 20 lines after the header to find the two matrix rows
        block = lines[hi: hi + 20]
        r0, r1 = None, None
        for ln in block:
            if r0 is None:
                m0 = row0_pat.search(ln)
                if m0:
                    r0 = m0
            else:
                m1 = row1_pat.search(ln)
                if m1:
                    r1 = m1
                    break
        if r0 is not None and r1 is not None:
            a_11 = float(r0.group(1))
            a_21 = float(r1.group(1))
            a_22 = float(r1.group(2))
            break

    if a_11 is not None:
        faa = (a_21 + a_22) / 2.0
        bwt = a_11 - a_21
        aaa = (a_11 + faa) / 2.0
        result.update(dict(a_11=a_11, a_21=a_21, a_22=a_22,
                           FAA=faa, BWT=bwt, AAA=aaa))

    # ------------------------------------------------------------------ #
    # 2. EWConProbe Mechanism dict (last occurrence)                       #
    # ------------------------------------------------------------------ #
    mech_pat = re.compile(r"\[EWConProbe\]\[Mechanism\] (\{.*\})")
    mech_dict = None
    for line in reversed(lines):
        m = mech_pat.search(line)
        if m:
            try:
                mech_dict = ast.literal_eval(m.group(1))
            except Exception:
                # fallback: use regex for each key
                mech_dict = {}
                for kv in re.finditer(r"'(\w+)':\s*([\-\d\.e\+]+)", m.group(1)):
                    try:
                        mech_dict[kv.group(1)] = float(kv.group(2))
                    except ValueError:
                        pass
            break

    if mech_dict:
        keys = [
            # accumulated Fisher
            "accumulated_fisher_total_mass",
            # weight drift
            "delta_norm", "param_delta_norm",
            "fisher_weighted_drift", "param_fisher_weighted_drift",
            "normalized_fisher_drift", "param_normalized_fisher_drift",
            "ewc_penalty_value",
            "d_f_contribution_from_t1_top1pct",
            # endpoint sharpness (small scope)
            "endpoint_sh0_max",
            "endpoint_grad_norm", "endpoint_first_order_sharpness",
            # endpoint flatness (full batches)
            "endpoint_flat_sh0_max",
            "endpoint_flat_Sh_param_full",
            "endpoint_flat_grad_norm", "endpoint_flat_first_order_sharpness",
            # Hessian
            "endpoint_flat_lambda_max_lanczos",
            "endpoint_flat_hessian_trace",
            "endpoint_flat_emp_fisher_lambda_max_lanczos",
            "endpoint_flat_emp_fisher_trace",
        ]
        for k in keys:
            if k in mech_dict:
                result[k] = mech_dict[k]

    # r_eff = sqrt(delta_norm)  (delta_norm stores ||ΔW||_F^2)
    if "delta_norm" in result:
        result["r_eff"] = result["delta_norm"] ** 0.5

    return result


# ------------------------------------------------------------------ #
# Collect all results                                                  #
# ------------------------------------------------------------------ #
all_results = {}
for name, path in METHODS:
    print(f"Parsing {name} ...")
    if not path.exists():
        print(f"  WARNING: file not found: {path}")
        all_results[name] = {}
        continue
    all_results[name] = parse_log(path)

# ------------------------------------------------------------------ #
# Print tables                                                         #
# ------------------------------------------------------------------ #

def fmt(v, fmt_str=".4f"):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return format(v, fmt_str)
    return str(v)


names = [n for n, _ in METHODS]

print("\n")
print("=" * 100)
print("TABLE 1 — Continual Learning Accuracy (CNN Evaluation, 2-Task)")
print("=" * 100)

acc_cols = [
    ("a_11",  "a_11 (%)", ".2f"),
    ("a_21",  "a_21 (%)", ".2f"),
    ("a_22",  "a_22 (%)", ".2f"),
    ("FAA",   "FAA  (%)", ".2f"),
    ("BWT",   "BWT  (%)", ".2f"),
    ("AAA",   "AAA  (%)", ".5f"),
]

header = f"{'Method':<14}" + "".join(f"{lbl:>14}" for _, lbl, _ in acc_cols)
print(header)
print("-" * len(header))
for name in names:
    r = all_results[name]
    row = f"{name:<14}"
    for key, _, fstr in acc_cols:
        row += f"{fmt(r.get(key), fstr):>14}"
    print(row)

print("\n")
print("=" * 120)
print("TABLE 2 — Weight Drift / Mechanism Metrics (at Task 1 endpoint)")
print("=" * 120)

drift_cols = [
    ("accumulated_fisher_total_mass",    "tr(F_accum)",           ".2f"),
    ("delta_norm",                       "delta_norm(=||ΔW||²)", ".4f"),
    ("r_eff",                            "r_eff=sqrt(δ_norm)",   ".4f"),
    ("fisher_weighted_drift",            "fisher_w_drift",        ".2e"),
    ("normalized_fisher_drift",          "norm_fisher_d",         ".2e"),
    ("ewc_penalty_value",                "ewc_penalty",           ".4f"),
    ("d_f_contribution_from_t1_top1pct", "ΔF@top1%",              ".4f"),
]

header2 = f"{'Method':<14}" + "".join(f"{lbl:>16}" for _, lbl, _ in drift_cols)
print(header2)
print("-" * len(header2))
for name in names:
    r = all_results[name]
    row = f"{name:<14}"
    for key, _, fstr in drift_cols:
        row += f"{fmt(r.get(key), fstr):>16}"
    print(row)

print("\n")
print("=" * 120)
print("TABLE 3 — Sharpness Metrics (at Task 1 endpoint)")
print("=" * 120)

sharp_cols = [
    ("endpoint_sh0_max",               "sh0_max(small)",      ".5f"),
    ("endpoint_grad_norm",             "grad_norm(ep)",        ".4f"),
    ("endpoint_first_order_sharpness", "Sh1(ep)",              ".5f"),
    ("endpoint_flat_sh0_max",          "sh0_max(flat)",        ".5f"),
    ("endpoint_flat_Sh_param_full",    "Sh_param_full",        ".5f"),
    ("endpoint_flat_grad_norm",        "grad_norm(flat)",      ".4f"),
    ("endpoint_flat_first_order_sharpness", "Sh1(flat)",       ".4f"),
]

header3 = f"{'Method':<14}" + "".join(f"{lbl:>18}" for _, lbl, _ in sharp_cols)
print(header3)
print("-" * len(header3))
for name in names:
    r = all_results[name]
    row = f"{name:<14}"
    for key, _, fstr in sharp_cols:
        row += f"{fmt(r.get(key), fstr):>18}"
    print(row)

print("\n")
print("=" * 100)
print("TABLE 4 — Hessian Curvature Metrics (at Task 1 endpoint, full dataset)")
print("=" * 100)

hess_cols = [
    ("endpoint_flat_lambda_max_lanczos",           "λ_max(Lanczos)",    ".2f"),
    ("endpoint_flat_hessian_trace",                "H_trace",           ".2f"),
    ("endpoint_flat_emp_fisher_lambda_max_lanczos","empF_λ_max",        ".2f"),
    ("endpoint_flat_emp_fisher_trace",             "empF_trace",        ".2f"),
]

header4 = f"{'Method':<14}" + "".join(f"{lbl:>18}" for _, lbl, _ in hess_cols)
print(header4)
print("-" * len(header4))
for name in names:
    r = all_results[name]
    row = f"{name:<14}"
    for key, _, fstr in hess_cols:
        row += f"{fmt(r.get(key), fstr):>18}"
    print(row)

print("\n")
print("=" * 160)
print("FULL TABLE — All metrics combined")
print("=" * 160)

all_cols = acc_cols + drift_cols + sharp_cols + hess_cols

# Print in two passes because terminal is wide
group1 = acc_cols + drift_cols
group2 = sharp_cols + hess_cols

for grp_cols in [group1, group2]:
    header_g = f"{'Method':<14}" + "".join(f"{lbl:>18}" for _, lbl, _ in grp_cols)
    print(header_g)
    print("-" * len(header_g))
    for name in names:
        r = all_results[name]
        row = f"{name:<14}"
        for key, _, fstr in grp_cols:
            row += f"{fmt(r.get(key), fstr):>18}"
        print(row)
    print()

# ------------------------------------------------------------------ #
# Also dump as CSV                                                     #
# ------------------------------------------------------------------ #
import csv, io

buf = io.StringIO()
writer = csv.writer(buf)

all_keys = [k for k, _, _ in all_cols]
all_labels = [lbl for _, lbl, _ in all_cols]
writer.writerow(["Method"] + all_labels)
for name in names:
    r = all_results[name]
    writer.writerow([name] + [r.get(k, "") for k in all_keys])

csv_path = Path("/data/140-1/users/liying/CopyFlatCV/eval_results_table.csv")
csv_path.write_text(buf.getvalue())
print(f"\nCSV saved to: {csv_path}")
