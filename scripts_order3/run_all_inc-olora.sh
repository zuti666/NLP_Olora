#!/bin/bash
set -e

# ---------- 定义检查 GPU 是否空闲的函数 ----------
# check_gpu_free23() {
#     gpu2_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3p')
#     gpu3_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '4p')

#     echo "$(date): GPU 2 usage: ${gpu2_mem} MB, GPU 3 usage: ${gpu3_mem} MB"

#     if [[ $gpu2_mem -lt 100 && $gpu3_mem -lt 100 ]]; then
#         return 0
#     else
#         return 1
#     fi
# }


GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
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

    # "scripts_order3/SeqLoRA_E1_Le3.sh"
    "scripts_order3/SeqLoRA-SAM_E1_Le3.sh"
        
    #--------IncLoRA

    # "scripts_order3/IncLoRA_E1_Le3.sh"
    # "scripts_order3/IncLoRA-SAM_E1_Le3.sh"
   
    #---------Olora

    # "scripts_order3/OLoRA.sh"
    # "scripts_order3/OLoRA_SAM.sh"
       
    )
    LOG_BASE="/disk0/users/liying/O-LoRA/Logs_RedoOlora/Order3_liying/Train_eval_lre-3_E1_seed-${seed}"

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
# # ---------- 定时检测并执行 ----------
# echo "Start monitoring GPU 2 and 3 every 30 minutes..."
# while true; do
#     if check_gpu_free23; then
#         echo "$(date): GPU 2 and 3 appear free. Waiting 10 seconds to double-check..."
#         sleep 6
#         if check_gpu_free23; then
#             echo "$(date): GPU 2 and 3 confirmed free after 5s. Start running training scripts..."
#             run_train_scripts
#             echo "$(date): All training finished. Exit."
#             exit 0
#         else
#             echo "$(date): GPU 2 and 3 became busy during the 10s wait. Will check again after 30 minutes."
#         fi
#     else
#         echo "$(date): GPU 2 and 3 are busy. Will check again after 30 minutes."
#     fi
#     sleep 1800
# done
