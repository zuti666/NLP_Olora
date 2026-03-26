#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/disk0/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
gpu_ids=0,1,2,3,4,5,6,7  # 直接指定使用0和1号GPU

# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path /disk0/users/liying/Sharing/Models/google-t5/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/MultiRC\
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/1-MultiRC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 


sleep 3

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/1-MultiRC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/2-BoolQA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/2-BoolQA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/3-WiC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round3 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/3-WiC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/4-MNLI \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3




CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/4-MNLI/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/5-CB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3


#--------------------------5 个任务---------------------------------

CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/5-CB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/6-COPA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/6-COPA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/7-QQP \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/7-QQP/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/8-RTE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3




CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/8-RTE/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/9-IMDB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3




CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/9-IMDB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/10-SST-2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3


#--------------------------10 个任务---------------------------------


CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/10-SST-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/11-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/11-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/12-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3


CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/12-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/13-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/13-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/14-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3



CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=8 --master_port=$port src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --train_method lora \
   --lora_strategy olora \
   --olora_lamda_1 0.5 \
   --olora_lamda_2 0 \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --model_name_or_path Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/14-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir Logs_Scripts_Long_Paper/Order5_train/Order5_OLoRA_T5large_torch_AdamwHF_E1_Lora8_LRe-3/15-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --num_train_epochs 1 \
   --run_name order5_round1 \
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
   --save_strategy no \
   --save_steps 1500 

sleep 3