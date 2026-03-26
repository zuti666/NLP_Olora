#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/disk0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-2,3}
seed=${2:-1993}
# world size must match the number of visible GPUs
IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
   --seed ${seed} --data_seed ${seed} \
   --do_train True\
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type sam-adamw_hf \
   --sam_rho 0.05 \
   --learning_rate 1e-03 \
   --model_name_or_path /disk0/users/liying/Sharing/Models/google-t5/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name IncLoRA_order2_round5 \
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

sleep 4



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
   --seed ${seed} --data_seed ${seed} \
   --do_train True\
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type sam-adamw_hf \
   --sam_rho 0.05 \
   --learning_rate 1e-03 \
   --model_name_or_path /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir  /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name IncLoRA_order2_round5 \
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

sleep 4


CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
   --seed ${seed} --data_seed ${seed} \
   --do_train True\
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type sam-adamw_hf \
   --sam_rho 0.05 \
   --learning_rate 1e-03 \
   --model_name_or_path /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/3-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name IncLoRA_order2_round5 \
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

sleep 4



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port=$port src/run_uie_lora.py \
   --seed ${seed} --data_seed ${seed} \
   --do_train True\
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type sam-adamw_hf \
   --sam_rho 0.05 \
   --learning_rate 1e-03 \
   --model_name_or_path /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/3-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir /disk0/users/liying/O-LoRA/Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}/4-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name IncLoRA_order2_round5 \
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
