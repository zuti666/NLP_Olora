#!/bin/bash
set -e



# ---------- 定义训练脚本执行函数 ----------
run_train_scripts() {
    echo "=== Training Script Start at $(date) ==="

    SCRIPTS=(

    # "scripts_order4_long/Long_Order4_SeqLoRA.sh"
    # "scripts_order4_long/Long_Order4_SeqLoRA-SAM.sh"

    # "scripts_order5_long/Long_Order5_SeqLoRA-SAM_E5_Le4.sh"
    # "scripts_order5_long/Long_Order5_SeqLora.sh"

    # "scripts_order6_long/LongOrder6_SeqLoRA_E1_Le4.sh"
    # "scripts_order6_long/LongOrder6_SeqLoRA-SAM_E1_Le4.sh"


    
    #--------IncLoRA
    # "scripts_order4_long/Long_Order4_IncLora.sh"
    # "scripts_order4_long/Long_Order4_IncLora_SAM.sh"
    # "scripts_order4_long/Long_Order4_OLoRA_SAM.sh"
    # "scripts_order4_long/Long_Order4_OLoRA.sh"


    #---------Olora
    "scripts_order6_long/LongOrder6_IncLoRA.sh"
    # "scripts_order5_long/Long_Order5_IncLora-SAM.sh"
    # "scripts_order5_long/Long_Order5_IncLora.sh"
    # "scripts_order5_long/Long_Order5_OLora-SAM.sh"
    # "scripts_order5_long/Long_Order5_OLora.sh"
    # "scripts_order6_long/LongOrder6_IncLoRA_SAM.sh"
    # "scripts_order6_long/LongOrder6_IncLoRA.sh"
    # "scripts_order6_long/LongOrder6_OLoRA_SAM.sh"
    # "scripts_order6_long/LongOrder6_OLoRA.sh"
    )
    GPU_IDS="0,1,2,3,4,5,6,7"
    LOG_BASE="/disk0/users/liying/O-LoRA/Logs_RedoOlora/Order5_liying/IncLoRA_SAM_lre-3_E1"

    for script_path in "${SCRIPTS[@]}"; do
        script_filename=$(basename "$script_path")
        script_name=${script_filename%.sh}
        LOG_DIR="${LOG_BASE}/logs_${script_name}"
        mkdir -p "$LOG_DIR"
        log_file="${LOG_DIR}/${script_name}.log"

        echo ">>> Running $script_path .Log will to $log_file"
        bash "$script_path" "$GPU_IDS" > "$log_file" 2>&1
        echo ">>> Finished. Log saved to $log_file"
    done
}
run_train_scripts
