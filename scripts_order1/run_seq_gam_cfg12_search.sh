#!/bin/bash
set -euo pipefail

# ===== GPU 分配 =====
gpu_ids_cfg3=${1:-0,1}
gpu_ids_cfg5=${2:-2,3}
gpu_ids_cfg6=${3:-4,5}
gpu_ids_cfg4=${4:-6,7}
seed=${5:-1993}

# ===== 统一日志目录 =====
log_dir=/data/140-0/users/liying/O-LoRA/Logs_Short_Paper/Order-1_8/train_logs_seed-${seed}
mkdir -p "${log_dir}"

echo "===== SeqLoRA-GAM sweep start (seed=${seed}) ====="

# ============================================================
# Sweep setting (fixed)
# beta_1=0.05, beta_2=1.0, beta_3=1.0
# adaptive=false
# sweep params: grad_rho / grad_norm_rho / gamma
# ============================================================

# =========================
# Exp-1
# =========================
exp1_name=SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p10_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00

echo "Launching Exp-1 on GPUs ${gpu_ids_cfg3}"
bash /data/140-0/users/liying/O-LoRA/scripts_order1/SeqLoRA-GAM_cfg3_E1_Le3.sh "${gpu_ids_cfg3}" "${seed}" \
  > "${log_dir}/${exp1_name}_seed-${seed}.log" 2>&1 &
pid1=$!

# =========================
# Exp-2
# =========================
exp2_name=SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00

echo "Launching Exp-2 on GPUs ${gpu_ids_cfg5}"
bash /data/140-0/users/liying/O-LoRA/scripts_order1/SeqLoRA-GAM_cfg5_E1_Le3.sh "${gpu_ids_cfg5}" "${seed}" \
  > "${log_dir}/${exp2_name}_seed-${seed}.log" 2>&1 &
pid2=$!

# =========================
# Exp-3
# =========================
exp3_name=SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00

echo "Launching Exp-3 on GPUs ${gpu_ids_cfg6}"
bash /data/140-0/users/liying/O-LoRA/scripts_order1/SeqLoRA-GAM_cfg6_E1_Le3.sh "${gpu_ids_cfg6}" "${seed}" \
  > "${log_dir}/${exp3_name}_seed-${seed}.log" 2>&1 &
pid3=$!

# =========================
# Exp-4
# =========================
exp4_name=SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p03_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00

echo "Launching Exp-4 on GPUs ${gpu_ids_cfg4}"
bash /data/140-0/users/liying/O-LoRA/scripts_order1/SeqLoRA-GAM_cfg4_E1_Le3.sh "${gpu_ids_cfg4}" "${seed}" \
  > "${log_dir}/${exp4_name}_seed-${seed}.log" 2>&1 &
pid4=$!

# ===== 等待 =====
wait ${pid1}
wait ${pid2}
wait ${pid3}
wait ${pid4}

echo "===== All experiments finished ====="