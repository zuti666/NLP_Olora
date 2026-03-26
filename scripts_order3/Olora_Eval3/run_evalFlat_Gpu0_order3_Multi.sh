#!/bin/bash

# ---------- 定义检查 GPU 是否空闲的函数 ----------
check_gpu_free3() {
    gpu3_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '4p')
    if [[ $gpu3_mem -lt 5000 ]]; then
        return 0
    else
        return 1
    fi
}
check_gpu_free23() {
    gpu2_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3p')
    gpu3_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '4p')

    if [[ $gpu2_mem -lt 5000 && $gpu3_mem -lt 5000 ]]; then
        return 0
    else
        return 1
    fi
}
check_gpu_free2() {
    gpu2_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '3p')
    if [[ $gpu2_mem -lt 5000 ]]; then
        return 0
    else
        return 1
    fi
}


run_eval_scripts() {
    echo "=== Script start at $(date) ==="

    SCRIPTS=("scripts_order1/Olora_Eval1/eval_Whole.sh")

    MODELS_PATHS=(

        "Logs_Short_Paper/Order-3_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba8-1_E1_LRe3"
        "Logs_Short_Paper/Order-3_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3"
        #inclora
        # "Logs_Short_Paper/Order-3_8/IncLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3"

        # "Logs_Short_Paper/Order-3_8/IncLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3"
        #olora
        # "Logs_Short_Paper/Order-3_8/OLoRA_T5large_torch_AdamwHF_Lora8_Ba8-1_E1_LRe3"

        # "Logs_Short_Paper/Order-3_8/OLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3"
        

        )

    MODELS=( "1-yahoo" "2-amazon" "3-agnews" "4-dbpedia")
    TEST_TASKS=( "yahoo" "amazon"  "agnews" "dbpedia")

    GPU_IDS=(0 1 2 3 4 5 6 7)
    LOG_BASE="Logs_RedoOlora/Order3_liying/logs"

    MAX_PARALLEL=${#GPU_IDS[@]}  # one task per GPU
    JOB_COUNT=0

    for script_path in "${SCRIPTS[@]}"; do
        script_filename=$(basename "$script_path")
        script_name=${script_filename%.sh}

        for model_path in "${MODELS_PATHS[@]}"; do
            model_dirname=$(basename "$model_path")

            for model_name in "${MODELS[@]}"; do
                for test_task in "${TEST_TASKS[@]}"; do

                    LOG_DIR="${LOG_BASE}/Eval_whole/model-${model_name}_testdata-${test_task}"
                    mkdir -p "$LOG_DIR"
                    log_file="${LOG_DIR}/${script_name}_eval_new.log"

                    gpu_id=${GPU_IDS[$((JOB_COUNT % MAX_PARALLEL))]}
                    echo "Launching eval: model_path=$model_dirname | model=$model_name | test=$test_task | gpu=$gpu_id"
                    echo "Log: $log_file"

                    # 后台并发运行任务
                    bash "$script_path" "$test_task" "$model_path" "$model_name" "$gpu_id" > "$log_file" 2>&1 &

                    ((JOB_COUNT++))
                    if [[ $((JOB_COUNT % MAX_PARALLEL)) -eq 0 ]]; then
                        echo ">>> Waiting for current batch of $MAX_PARALLEL tasks to finish..."
                        wait
                    fi
                done
            done
        done
    done

    wait  # 等待最后一批未满 3 个的任务完成
    echo "=== All evaluation tasks finished at $(date) ==="
}

run_eval_scripts

# echo "Start monitoring GPU 2 every 30 minutes..."
# while true; do
#     if check_gpu_free2; then
#         echo "$(date): GPU 2 is free. Waiting 10 seconds to double-check..."
#         if check_gpu_free2; then
#             echo "$(date): GPU 2 is confirmed free after 10s. Start running evaluation scripts..."
#             run_eval_scripts
#             echo "$(date): All evaluation finished. Exit."
#         else
#             echo "$(date): GPU 2 became busy during the 10s wait. Will check again after 30 minutes."
#         fi
#     else
#         echo "$(date): GPU 2 is busy. Will check again after 30 minutes."
#     fi
#     sleep 1800
# done



