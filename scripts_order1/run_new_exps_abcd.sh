#!/bin/bash
# Master launch: run Exp A, C, B, D sequentially on GPU 3
# Usage: bash scripts_order1/run_new_exps_abcd.sh [gpu_id] [seed]
# Default: gpu=3, seed=1993

set -euo pipefail

gpu_id=${1:-3}
seed=${2:-1993}

log_dir="Logs_Short_Paper/Order-1_8/train_logs_abcd_seed-${seed}"
mkdir -p "${log_dir}"

echo "=== Starting Exp A: OLoRA-SAM rho=0.10 on GPU ${gpu_id} ==="
bash scripts_order1/OLoRA-SAM_rho0p10_E1_Le3.sh "${gpu_id}" "${seed}" \
    > "${log_dir}/expA_olora_sam_rho0p10.log" 2>&1
echo "=== Exp A done at $(date) ==="

echo "=== Starting Exp C: IncLoRA-RWP cfg1 on GPU ${gpu_id} ==="
bash scripts_order1/IncLoRA-RWP_cfg1_E1_Le3.sh "${gpu_id}" "${seed}" \
    > "${log_dir}/expC_inclora_rwp_cfg1.log" 2>&1
echo "=== Exp C done at $(date) ==="

echo "=== Starting Exp B: OLoRA-SAM rho=0.02 on GPU ${gpu_id} ==="
bash scripts_order1/OLoRA-SAM_rho0p02_E1_Le3.sh "${gpu_id}" "${seed}" \
    > "${log_dir}/expB_olora_sam_rho0p02.log" 2>&1
echo "=== Exp B done at $(date) ==="

echo "=== Starting Exp D: IncLoRA-SAM rho=0.01 on GPU ${gpu_id} ==="
bash scripts_order1/IncLoRA-SAM_rho0p01_E1_Le3.sh "${gpu_id}" "${seed}" \
    > "${log_dir}/expD_inclora_sam_rho0p01.log" 2>&1
echo "=== Exp D done at $(date) ==="

echo "=== All 4 experiments completed. Logs in ${log_dir} ==="
