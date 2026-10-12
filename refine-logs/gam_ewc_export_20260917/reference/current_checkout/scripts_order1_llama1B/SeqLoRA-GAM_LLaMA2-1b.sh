#!/bin/bash
set -euxo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-3,5,6,7}
IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}
run_root=logs_and_outputs_llama_1B/order_1/SeqLoRA-GAM-002_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4
# bash scripts_llama/order_1.sh> logs_and_outputs_llama_1B/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type gam-adamw_hf \
   --gam_grad_rho 0.02 \
   --gam_grad_norm_rho 0.2 \
   --gam_adaptive false \
   --gam_grad_beta_1 1 \
   --gam_grad_beta_2 1.0 \
   --gam_grad_beta_3 1.0 \
   --gam_grad_gamma 0.03 \
   --model_name_or_path /data/140-0/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir "${run_root}/1-dbpedia" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --run_name order1_seqlora_gam_round1 \
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
   --save_steps 1500 \


sleep 5

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type gam-adamw_hf \
   --gam_grad_rho 0.02 \
   --gam_grad_norm_rho 0.2 \
   --gam_adaptive false \
   --gam_grad_beta_1 1 \
   --gam_grad_beta_2 1.0 \
   --gam_grad_beta_3 1.0 \
   --gam_grad_gamma 0.03 \
   --model_name_or_path "${run_root}/1-dbpedia/adapter" \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir "${run_root}/2-amazon" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --run_name order1_seqlora_gam_round2 \
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
   --save_steps 1500 \

sleep 5

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type gam-adamw_hf \
   --gam_grad_rho 0.02 \
   --gam_grad_norm_rho 0.2 \
   --gam_adaptive false \
   --gam_grad_beta_1 1 \
   --gam_grad_beta_2 1.0 \
   --gam_grad_beta_3 1.0 \
   --gam_grad_gamma 0.03 \
   --model_name_or_path "${run_root}/2-amazon/adapter" \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir "${run_root}/3-yahoo" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --run_name order1_seqlora_gam_round3 \
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
   --save_steps 1500 \

sleep 5

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$num_procs --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --bf16 True \
   --train_method lora \
   --lora_strategy seqlora \
   --lora_dim 8 \
   --optimizer_type gam-adamw_hf \
   --gam_grad_rho 0.02 \
   --gam_grad_norm_rho 0.2 \
   --gam_adaptive false \
   --gam_grad_beta_1 1 \
   --gam_grad_beta_2 1.0 \
   --gam_grad_beta_3 1.0 \
   --gam_grad_gamma 0.03 \
   --model_name_or_path "${run_root}/3-yahoo/adapter" \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir "${run_root}/4-agnews" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --run_name order1_seqlora_gam_round4 \
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
   --save_steps 1500 \
