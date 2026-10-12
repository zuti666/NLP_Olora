#!/bin/bash
set -euxo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-0,1,2,3,4,5,6,7}
seed=${2:-1993}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
train_bs=8
run_root=/data/140-0/users/liying/O-LoRA/Logs_Short_Paper/Order-1_8/EWCLoRA_T5large_torch_AdamwHF_Lora8_Ba8-1_E1_LRe3_lambda${lambda_ewc}_gamma${gamma_ewc}_seed-${seed}

# world size must match the number of visible GPUs
IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local stage_dir="$3"
  local run_name="$4"

  CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
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
    --learning_rate 1e-03 \
    --model_name_or_path "${model_path}" \
    --data_dir CL_Benchmark \
    --task_config_dir "${task_config_dir}" \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "${run_root}/${stage_dir}" \
    --per_device_train_batch_size ${train_bs} \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --run_name "${run_name}" \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
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

train_round /data/140-0/users/liying/Sharing/Models/google-t5/t5-large configs/order1_configs/dbpedia 1-dbpedia EWCLoRA_order1_round1
sleep 5
train_round "${run_root}/1-dbpedia/adapter" configs/order1_configs/amazon 2-amazon EWCLoRA_order1_round2
sleep 5
train_round "${run_root}/2-amazon/adapter" configs/order1_configs/yahoo 3-yahoo EWCLoRA_order1_round3
sleep 5
train_round "${run_root}/3-yahoo/adapter" configs/order1_configs/agnews 4-agnews EWCLoRA_order1_round4
