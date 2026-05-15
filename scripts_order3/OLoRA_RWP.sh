#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-2,3}
seed=${2:-1993}

IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}
output_root=/data/140-0/users/liying/O-LoRA/Logs_Short_Paper/Order-3_8/OLoRA-RWP_lora_Gauss_std0p05_T5small_torch_AdamwHF_Lora8_Ba8-1_E1_LRe3_seed-${seed}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local stage_dir="$3"
  local run_name="$4"
  shift 4

  CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
    --seed ${seed} --data_seed ${seed} \
    --do_train True \
    "$@" \
    --bf16 True \
    --train_method lora \
    --lora_strategy olora \
    --lora_dim 8 \
    --optimizer_type adamw_hf \
    --rwp_type lora \
    --rwp_a 0 \
    --rwp_b 1 \
    --rwp_noise_type Gauss_standard \
    --noise_std 0.05 \
    --learning_rate 1e-03 \
    --model_name_or_path "${model_path}" \
    --data_dir CL_Benchmark \
    --task_config_dir "${task_config_dir}" \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "${output_root}/${stage_dir}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
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

train_round /data/140-0/users/liying/Sharing/Models/google-t5/t5-small configs/order3_configs/yahoo 1-yahoo OLoRA_RWP_order3_round1
sleep 4
train_round "${output_root}/1-yahoo/adapter" configs/order3_configs/amazon 2-amazon OLoRA_RWP_order3_round2 --predict_with_generate
sleep 4
train_round "${output_root}/2-amazon/adapter" configs/order3_configs/agnews 3-agnews OLoRA_RWP_order3_round3
sleep 4
train_round "${output_root}/3-agnews/adapter" configs/order3_configs/dbpedia 4-dbpedia OLoRA_RWP_order3_round4
