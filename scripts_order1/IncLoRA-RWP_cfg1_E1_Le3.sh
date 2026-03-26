#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-4,5,6,7}
seed=${2:-1993}

IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}
torchrun_bin=${TORCHRUN_BIN:-/data/140-0/users/liying/conda_envs/nlora_req/bin/torchrun}

# IncLoRA + RWP cfg1 (Exp C)
# Hypothesis: gentle noise (std=0.005) + 80% clean gradient (rwp_a=0.8)
# avoids the catastrophic forgetting caused by SAM's aggressive perturbation
# optimizer stays as adamw_hf; RWP is applied inside the trainer via rwp_type
output_root=/data/140-0/users/liying/O-LoRA/Logs_Short_Paper/Order-1_8/IncLoRA-RWP_lora_Gauss_std0p005_a0p8_b0p2_T5large_r8_bs16_e1_lr1e-3_seed-${seed}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local stage_dir="$3"

  CUDA_VISIBLE_DEVICES=$gpu_ids "${torchrun_bin}" --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
    --seed ${seed} --data_seed ${seed} \
    --do_train True \
    --bf16 True \
    --train_method lora \
    --lora_strategy Inclora \
    --lora_dim 8 \
    --optimizer_type adamw_hf \
    --rwp_type lora \
    --rwp_noise_type Gauss_standard \
    --noise_std 0.005 \
    --rwp_a 0.8 \
    --rwp_b 0.2 \
    --learning_rate 1e-03 \
    --model_name_or_path "${model_path}" \
    --data_dir CL_Benchmark \
    --task_config_dir "${task_config_dir}" \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "${output_root}/${stage_dir}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --run_name IncLoRA_RWP_cfg1_order1 \
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

train_round /data/140-0/users/liying/Sharing/Models/google-t5/t5-large configs/order1_configs/dbpedia 1-dbpedia
sleep 3
train_round "${output_root}/1-dbpedia/adapter" configs/order1_configs/amazon 2-amazon
sleep 3
train_round "${output_root}/2-amazon/adapter" configs/order1_configs/yahoo 3-yahoo
sleep 3
train_round "${output_root}/3-yahoo/adapter" configs/order1_configs/agnews 4-agnews
