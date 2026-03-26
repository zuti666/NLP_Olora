#!/bin/bash

echo "=== Script start at $(date) ==="

SCRIPTS=(
  "scripts_order6_long/Olora_Eval6/eval_Whole.sh"
  )
MODELS_PATHS=(
  # "Logs_Scripts_Long_Paper/Order6_train/Order6_SeqLoRA_GPU4_T5large_torch_AdamwHF_E5_Lora8_LRe-4_Rho002"
  # "Logs_Scripts_Long_Paper/Order6_train/Order6_SeqLoRA-SAM__T5large_torch_AdamwHF_E5_Lora8_LRe-4_Rho002"
  # "Logs_Scripts_Long_Paper/Order6_train/IncLoRASAM_T5large_torch_AdamwHFSAM_E1_Lora8_LR-3_SAMRho-3"
  "Logs_Scripts_Long_Paper/Order6_train/Order6_OLoRA_SAM_T5large_torch_AdamwHF_E1_Lora8_LR-3_Lamda-OloraSetting-SAMRho-3"
  "Logs_Scripts_Long_Paper/Order6_train/OLoRA_T5large_torch_AdamwHF_E1_Lora8_LR-3_Lamda-05-0"
  "Logs_Scripts_Long_Paper/Order6_train/Order6_IncLoRA_GPU4_T5large_torch_AdamwHF_E1_Lora8_LR-e3"

)

# 所有模型名称 
#"1-yelp"
MODELS=( "1-yelp" "2-amazon" "3-MNLI" "4-CB" "5-COPA" 
"6-QQP" "7-RTE" "8-IMDB" "9-SST-2" "10-dbpedia"
"11-agnews" "12-yahoo" "13-MultiRC" "14-BoolQA" "15-WiC"
)  


TASK_ORDER=("yelp" "amazon" "MNLI" "CB" "COPA" "QQP" "RTE" "IMDB" 
  "SST-2" "dbpedia" "agnews" "yahoo" "MultiRC" "BoolQA" "WiC") 


GPU_IDS=(0 1 2 3 4 5 6)

LOG_BASE="Logs_RedoOlora/logs"

MAX_PARALLEL=${#GPU_IDS[@]}  # one task per GPU
JOB_COUNT=0

for script_path in "${SCRIPTS[@]}"; do
  script_filename=$(basename "$script_path")
  script_name=${script_filename%.sh}

  for model_path in "${MODELS_PATHS[@]}"; do
    model_dirname=$(basename "$model_path")

    for model_name in "${MODELS[@]}"; do
      model_prefix="${model_name%%-*}"

      if ! [[ "$model_prefix" =~ ^[0-9]+$ ]]; then
        echo "Skipping model ${model_name}: cannot parse prefix."
        continue
      fi

      num_tasks=$model_prefix
      if (( num_tasks > ${#TASK_ORDER[@]} )); then
        num_tasks=${#TASK_ORDER[@]}
      fi

      TASKS_TO_RUN=("${TASK_ORDER[@]:0:num_tasks}")

      for test_task in "${TASKS_TO_RUN[@]}"; do
        LOG_DIR="${LOG_BASE}/Eval_whole/model-${model_name}_testdata-${test_task}"
        mkdir -p "$LOG_DIR"
        log_file="${LOG_DIR}/${script_name}_eval.log"

        gpu_id=${GPU_IDS[$((JOB_COUNT % MAX_PARALLEL))]}
        echo "Launching eval: model_path=$model_dirname | model=$model_name | test=$test_task | gpu=$gpu_id"
        echo "Log: $log_file"

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
wait
echo "=== All evaluation tasks finished at $(date) ==="


