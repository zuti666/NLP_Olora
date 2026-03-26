#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/disk0/users/liying/.cache/huggingface



Task_name=$1    # 例如 
Model_path=$2
Model_path_name=$3  # 例如 a
GPU_id=$4


(
port=$(shuf -i25000-26000 -n1)
CUDA_VISIBLE_DEVICES=$GPU_id torchrun --nproc_per_node=1 --master_port=$port src/run_uie_lora.py \
   --max_train_sample 0\
   --bf16 True \
   --flat_part trained_lora \
   --train_method lora \
   --lora_strategy inclora\
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-04 \
   --do_predict True\
   --predict_with_generate \
   --model_name_or_path ${Model_path}/${Model_path_name}/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_eval_configs/${Task_name} \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir ${Model_path}/${Model_path_name}/evalFlat-${Task_name}-trainedLoRAwhole \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 0 \
   --run_name order1_round1 \
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
)

echo "All jobs finished!"