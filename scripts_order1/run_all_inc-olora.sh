#!/bin/bash
set -e



GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
# 默认 seeds；如需自定义，在 GPU_IDS 之后追加即可
if [ $# -ge 2 ]; then
    SEEDS=("${@:2}")
else
    SEEDS=(42)
fi

# ---------- 定义训练脚本执行函数 ----------
run_train_scripts() {
    local seed="$1"
    echo "=== Training Script Start at $(date) ==="

    SCRIPTS=(
    # "scripts_order1/SeqLoRA_E1_Le3.sh"
    # "scripts_order1/SeqLoRA-SAM_E1_Le3.sh"
    # "scripts_order1/SeqLoRA-GAM_E1_Le3.sh"
    #--------IncLoRA
    # "scripts_order1/IncLoRA_E1_Le3.sh"
    # "scripts_order1/IncLoRA-SAM_E1_Le3.sh"
    # "scripts_order1/IncLoRA_E1_Le3.sh"
    # "scripts_order1/IncLoRA-SAM_E1_Le3.sh"
    # "scripts_order1/IncLoRA-GAM_E1_Le3.sh"
    # "scripts_order1/OLoRA_GAM.sh"
    
    #---------Olora
    # "scripts_order1/OLoRA.sh"
    "scripts_order1/SeqLoRA-GAM_cfg3_E1_Le3.sh"
    "scripts_order1/SeqLoRA-GAM_cfg4_E1_Le3.sh"
    "scripts_order1/SeqLoRA-GAM_cfg5_E1_Le3.sh"
    "scripts_order1/SeqLoRA-GAM_cfg6_E1_Le3.sh"


    )
    LOG_BASE="/disk0/users/liying/O-LoRA/Logs_RedoOlora/Order1_liying/IncLoRA_GAM_lre-4_E1_seed-${seed}"

    for script_path in "${SCRIPTS[@]}"; do
        script_filename=$(basename "$script_path")
        script_name=${script_filename%.sh}
        LOG_DIR="${LOG_BASE}/logs_${script_name}"
        mkdir -p "$LOG_DIR"
        log_file="${LOG_DIR}/${script_name}_seed-${seed}.log"

        echo ">>> Running $script_path .Log will to $log_file"
        bash "$script_path" "$GPU_IDS" "$seed" > "$log_file" 2>&1
        echo ">>> Finished. Log saved to $log_file"
    done
}

for seed in "${SEEDS[@]}"; do
    run_train_scripts "$seed"
done
