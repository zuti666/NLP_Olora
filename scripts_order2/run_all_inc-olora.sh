#!/bin/bash
set -e




GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
# Seeds: default to 2048 and 1993, can override by passing extra args after GPU_IDS
if [ $# -ge 2 ]; then
    SEEDS=("${@:2}")
else
    SEEDS=(2048 1993)
fi

# ---------- 定义训练脚本执行函数 ----------
run_train_scripts() {
    local seed="$1"
    echo "=== Training Script Start at $(date) ==="

    SCRIPTS=(

    # "scripts_order2/SeqLoRA_E1_Le3.sh"
    # "scripts_order2/SeqLoRA-SAM_E1_Le3.sh"

    #--------IncLoRA
    "scripts_order2/IncLoRA_E1_Le3.sh"
    "scripts_order2/IncLoRA-SAM_E1_Le3.sh"

    # #---------Olora
    # "scripts_order2/OLoRA.sh"
    # "scripts_order2/OLoRA_SAM.sh"


    )
    LOG_BASE="/disk0/users/liying/O-LoRA/Logs_RedoOlora/Order2_liying/IncLoRA_SAM_lre-4_E1_seed-${seed}"

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
