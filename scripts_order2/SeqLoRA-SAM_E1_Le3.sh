#!/bin/bash
set -euxo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-4,5,6,7}
seed=${2:-1993}
train_bs=16
run_root=/data/140-0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_cfgUnified_seed-${seed}

IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local stage_dir="$3"
  local run_name="$4"

  CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
    --seed ${seed} --data_seed ${seed} \
    --do_train True \
    --bf16 True \
    --train_method lora \
    --lora_strategy seqlora \
    --lora_dim 8 \
    --optimizer_type sam-adamw_hf \
    --sam_rho 0.05 \
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

train_round /data/140-0/users/liying/Sharing/Models/google-t5/t5-large configs/order2_configs/dbpedia 1-dbpedia SeqLoRA-SAM-005_order2_round1
sleep 4
train_round "${run_root}/1-dbpedia/adapter" configs/order2_configs/amazon 2-amazon SeqLoRA-SAM-005_order2_round2
sleep 4
train_round "${run_root}/2-amazon/adapter" configs/order2_configs/agnews 3-agnews SeqLoRA-SAM-005_order2_round3
sleep 4
train_round "${run_root}/3-agnews/adapter" configs/order2_configs/yahoo 4-yahoo SeqLoRA-SAM-005_order2_round4
