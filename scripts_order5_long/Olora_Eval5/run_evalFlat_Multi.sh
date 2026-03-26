#!/bin/bash

run_eval_scripts() {
    echo "=== Script start at $(date) ==="

    SCRIPTS=(
        "scripts_order5_long/Olora_Eval5/eval_Whole.sh"
    )

    MODELS_PATHS=(
    #    "Logs_Scripts_Long_Paper/OloraSetting_Order4_E1000/Order4_OLoRA_RWPlora-Fisher_T5large_1000_torch_AdamwHF_E1_Lora8_LR-3_a1b1_GaussStandard_N-e3"
        # "Logs_Scripts_Long_Paper/Order5_train/Order5_IncLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3"
        # "Logs_Scripts_Long_Paper/Order5_train/Order5_IncLoRA-SAM_T5large_torch_AdamwHF_E1_Lora8_LRe-3"
        "Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3"
        # "Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA-SAM_T5large_torch_AdamwHF_E1_Lora8_LRe-3"
    
    )

# "1-MultiRC" "2-BoolQA" "3-WiC" "4-MNLI" "5-CB" "6-COPA" "7-QQP" "8-RTE" "9-IMDB" 
#         "10-SST-2" "11-dbpedia" "12-agnews" "13-yelp" "14-amazon"

    MODELS=( "15-yahoo")
    
    TEST_TASKS=("MultiRC" "BoolQA" "WiC" "MNLI" "CB" "COPA" "QQP" "RTE" "IMDB" 
        "SST-2" "dbpedia" "agnews" "yelp" "amazon" "yahoo" )

    # TASK_ORDER=("${TEST_TASKS[@]}")

    GPU_IDS=(0 1 2 3 4 5 6 7)
    LOG_BASE="Logs_RedoOlora/Order5_sample_liying_eval/logs"

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



