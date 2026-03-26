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

    SCRIPTS=(
        "scripts_order4_long/Olora_Eval4/eval_Whole1.sh"
    )

    MODELS_PATHS=(
    #    "Logs_Scripts_Long_Paper/OloraSetting_Order4_E1000/Order4_OLoRA_RWPlora-Fisher_T5large_1000_torch_AdamwHF_E1_Lora8_LR-3_a1b1_GaussStandard_N-e3"
        "Logs_Scripts_Long_Paper/Order4_train/Order4_seqLoRASAM_T5large_E1000_torch_AdamwHFSAM_E1_Lora8_LR-3_SAMRho-3"
        "Logs_Scripts_Long_Paper/Order4_train/Order4_seqLoRA_T5large_E1000_torch_AdamwHF_E1_Lora8_LRe-3"
    
    # "Logs_Scripts_Long_Paper/Order4_train/Order4_IncLoRA_T5large_E1000_torch_AdamwHF_E1_Lora8_LRe-3"
    # "Logs_Scripts_Long_Paper/Order4_train/Order4_IncLoRASAM_T5large_E1000_torch_AdamwHFSAM_E1_Lora8_LR-3_SAMRho-3"
    # "Logs_Scripts_Long_Paper/Order4_train/Order4_OLoRA_T5large_E1000_torch_AdamwHF_E1_Lora8_LRe-3"
    # "Logs_Scripts_Long_Paper/Order4_train/Order4_OLoRASAM_T5large_E1000_torch_AdamwHFSAM_E1_Lora8_LR-3_SAMRho-3"

    )

        # "1-MNLI" "2-CB" "3-WiC" "4-COPA" "5-QQP" "6-BoolQA" "7-RTE" "8-IMDB" "9-yelp"
        # "10-amazon" "11-SST-2" "12-dbpedia" "13-agnews" "14-MultiRC"

    MODELS=(
            "15-yahoo"
    )

    TEST_TASKS=("yelp" "amazon" "MNLI" "CB" "COPA" "QQP" "RTE" "IMDB" 
        "SST-2" "dbpedia" "agnews" "yahoo" "MultiRC" "BoolQA" "WiC")

    # TASK_ORDER=("${TEST_TASKS[@]}")

    GPU_IDS=(0 1 2 3 4 5 6 7)
    LOG_BASE="Logs_RedoOlora/Order4_sample_liying_eval/logs"

    MAX_PARALLEL=${#GPU_IDS[@]}  # one task per GPU
    JOB_COUNT=0

    for script_path in "${SCRIPTS[@]}"; do
        script_filename=$(basename "$script_path")
        script_name=${script_filename%.sh}

        for model_path in "${MODELS_PATHS[@]}"; do
            model_dirname=$(basename "$model_path")

            for model_name in "${MODELS[@]}"; do
                model_prefix="${model_name%%-*}"

                # if ! [[ "$model_prefix" =~ ^[0-9]+$ ]]; then
                #     echo "Skipping model ${model_name}: cannot parse prefix."
                #     continue
                # fi

                # num_tasks=$model_prefix
                # if (( num_tasks > ${#TASK_ORDER[@]} )); then
                #     num_tasks=${#TASK_ORDER[@]}
                # fi

                # TASKS_TO_RUN=("${TASK_ORDER[@]:0:num_tasks}")

                for test_task in "${TEST_TASKS[@]}"; do

                    LOG_DIR="${LOG_BASE}/Eval_whole/model-${model_name}_testdata-${test_task}"
                    mkdir -p "$LOG_DIR"
                    log_file="${LOG_DIR}/${script_name}_eval_new.log"

                    echo "Launching eval: model_path=$model_dirname | model=$model_name | test=$test_task"
                    echo "Log: $log_file"

                    gpu_id=${GPU_IDS[$((JOB_COUNT % MAX_PARALLEL))]}

                    # 后台并发运行任务
                    bash "$script_path" "$test_task" "$model_path" "$model_name" "$gpu_id" > "$log_file" 2>&1 &

                    ((JOB_COUNT++))
                    if (( JOB_COUNT % MAX_PARALLEL == 0 )); then
                        echo ">>> Waiting for current batch of $MAX_PARALLEL tasks to finish..."
                        wait
                    fi
                done
            done
        done
    done

    wait  # 等待最后一批未满 MAX_PARALLEL 的任务完成
    echo "=== All evaluation tasks finished at $(date) ==="
}

run_eval_scripts



