#!/bin/bash
set -exo pipefail

CONDA_BASE=/data/115-2/users/liying/miniconda3_shared
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate nlora_req
set -u

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-0,1,5,6}
seed=${2:-1993}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}
run_root=logs_and_outputs_llama_1B/order_2/EWCLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA16_E1_LR1e-3_to_1e-4_lambda${lambda_ewc}_gamma${gamma_ewc}_seed-${seed}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local output_dir="$3"
  local run_name="$4"
  local learning_rate="$5"

  CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port $port src/run_uie_lora.py \
     --seed ${seed} --data_seed ${seed} \
     --do_train \
     --do_predict \
     --predict_with_generate \
     --bf16 True \
     --train_method lora \
     --lora_strategy ewclora \
     --lora_dim 8 \
     --ewclora_lambda "${lambda_ewc}" \
     --ewclora_gamma "${gamma_ewc}" \
     --optimizer_type adamw_hf \
     --model_name_or_path "${model_path}" \
     --data_dir CL_Benchmark \
     --task_config_dir "${task_config_dir}" \
     --instruction_file configs/instruction_config.json \
     --instruction_strategy single \
     --output_dir "${output_dir}" \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 4 \
     --gradient_accumulation_steps 16 \
     --learning_rate "${learning_rate}" \
     --num_train_epochs 1 \
     --run_name "${run_name}" \
     --max_source_length 512 \
     --max_target_length 50 \
     --generation_max_length 50 \
     --ddp_find_unused_parameters False \
     --add_task_name True \
     --add_dataset_name True \
     --overwrite_output_dir \
     --overwrite_cache \
     --lr_scheduler_type constant \
     --warmup_steps 0 \
     --logging_strategy steps \
     --logging_steps 10 \
     --evaluation_strategy no \
     --save_strategy no \
     --save_steps 1500
}

train_round \
  /data/140-0/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B \
  configs/order2_configs/dbpedia \
  "${run_root}/1-dbpedia" \
  order2_ewclora_round1 \
  1e-03

sleep 5

train_round \
  "${run_root}/1-dbpedia/adapter" \
  configs/order2_configs/amazon \
  "${run_root}/2-amazon" \
  order2_ewclora_round2 \
  1e-04

sleep 5

train_round \
  "${run_root}/2-amazon/adapter" \
  configs/order2_configs/agnews \
  "${run_root}/3-agnews" \
  order2_ewclora_round3 \
  1e-04

sleep 5

train_round \
  "${run_root}/3-agnews/adapter" \
  configs/order2_configs/yahoo \
  "${run_root}/4-yahoo" \
  order2_ewclora_round4 \
  1e-04
