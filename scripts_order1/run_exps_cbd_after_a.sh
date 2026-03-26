#!/bin/bash
# Run C → B → D sequentially after Exp A finishes
# Called automatically after Exp A completes

set -euo pipefail
gpu_id=${1:-3}
seed=${2:-1993}
log_dir="Logs_Short_Paper/Order-1_8/train_logs_abcd_seed-${seed}"
mkdir -p "${log_dir}"

echo "=== Exp C: IncLoRA-RWP cfg1 start at $(date) ==="
bash scripts_order1/IncLoRA-RWP_cfg1_E1_Le3.sh "${gpu_id}" "${seed}" \
    >> "${log_dir}/expC_inclora_rwp_cfg1.log" 2>&1
echo "=== Exp C done at $(date) ==="

echo "=== Exp B: OLoRA-SAM rho=0.02 start at $(date) ==="
bash scripts_order1/OLoRA-SAM_rho0p02_E1_Le3.sh "${gpu_id}" "${seed}" \
    >> "${log_dir}/expB_olora_sam_rho0p02.log" 2>&1
echo "=== Exp B done at $(date) ==="

echo "=== Exp D: IncLoRA-SAM rho=0.01 start at $(date) ==="
bash scripts_order1/IncLoRA-SAM_rho0p01_E1_Le3.sh "${gpu_id}" "${seed}" \
    >> "${log_dir}/expD_inclora_sam_rho0p01.log" 2>&1
echo "=== Exp D done at $(date) ==="

echo "=== All done at $(date). Launch eval: bash scripts_order1/Olora_Eval1/run_eval_new_exps_abcd.sh ==="
