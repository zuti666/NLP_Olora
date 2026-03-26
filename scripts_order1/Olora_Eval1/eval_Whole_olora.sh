#!/bin/bash
# Eval a single (task, model_stage) pair for OLoRA or IncLoRA
# Args: $1=task_name  $2=model_root  $3=stage_dir  $4=gpu_id  $5=lora_strategy
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

Task_name=$1
Model_path=$2
Model_path_name=$3
GPU_id=$4
lora_strategy=${5:-olora}
torchrun_bin=${TORCHRUN_BIN:-/data/140-0/users/liying/conda_envs/nlora_req/bin/torchrun}

port=$(shuf -i25000-30000 -n1)
CUDA_VISIBLE_DEVICES=$GPU_id "${torchrun_bin}" --nproc_per_node=1 --master_port=$port src/run_uie_lora.py \
   --do_predict True \
   --predict_with_generate \
   --max_train_sample 0 \
   --bf16 True \
   --flat_part trained_whole \
   --train_method lora \
   --lora_strategy ${lora_strategy} \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path ${Model_path}/${Model_path_name}/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/lineraProbe_order1_configs/${Task_name} \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir ${Model_path}/${Model_path_name}/evalFlat-${Task_name}-trainedLoRAwhole \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 0 \
   --run_name order1_eval \
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
