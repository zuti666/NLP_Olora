#!/bin/bash
# Unified train → eval → summarize pipeline for Order-1 (4 tasks: dbpedia→amazon→yahoo→agnews)
#
# Usage:
#   bash scripts_order1/run_order1.sh <gpu_ids> <seed> <method> [sam_rho]
#
# Arguments:
#   gpu_ids  : comma-separated GPU IDs used for BOTH training and eval, e.g. "0,1,2,3"
#   seed     : random seed, e.g. 1993
#   method   : "SeqLoRA" or "SeqLoRA-SAM"
#   sam_rho  : SAM rho value (only used when method=SeqLoRA-SAM), default 0.05
#
# Typical two-experiment setup (run in two terminals):
#   bash scripts_order1/run_order1.sh 0,1,2,3 1993 SeqLoRA
#   bash scripts_order1/run_order1.sh 4,5,6,7 1993 SeqLoRA-SAM 0.05

set -euo pipefail
set -x

# ─── Arguments ──────────────────────────────────────────────────────────────
gpu_ids=${1:-"0,1,2,3"}
seed=${2:-1993}
method=${3:-"SeqLoRA"}
sam_rho=${4:-0.05}

# ─── Derived constants ───────────────────────────────────────────────────────
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/data/140-0/users/liying/.cache/huggingface
torchrun_bin=${TORCHRUN_BIN:-torchrun}

IFS=',' read -ra _gpu_array <<< "$gpu_ids"
num_procs=${#_gpu_array[@]}

# Build run_root and optimizer args from method name
case "${method}" in
    SeqLoRA)
        run_root="Logs_Short_Paper/Order-1_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}"
        optimizer_args="--optimizer_type adamw_hf"
        ;;
    SeqLoRA-SAM)
        rho_tag=$(echo "$sam_rho" | sed 's/0\./0p/')   # 0.05 → 0p05
        run_root="Logs_Short_Paper/Order-1_8/SeqLoRA-SAM-${rho_tag}_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-${seed}"
        optimizer_args="--optimizer_type sam-adamw_hf --sam_rho ${sam_rho}"
        ;;
    *)
        echo "ERROR: unknown method '${method}'. Use 'SeqLoRA' or 'SeqLoRA-SAM'." >&2
        exit 1
        ;;
esac

# Shared training/eval hyperparams
train_bs=16
lora_dim=8
lr=1e-03
base_model=/data/140-0/users/liying/Sharing/Models/google-t5/t5-large

# Order-1 task sequence
tasks=("dbpedia" "amazon" "yahoo" "agnews")
stages=("1-dbpedia" "2-amazon" "3-yahoo" "4-agnews")

echo "============================================================"
echo "  Method : ${method}"
echo "  GPUs   : ${gpu_ids}  (${num_procs} procs)"
echo "  Seed   : ${seed}"
echo "  Output : ${run_root}"
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: Sequential training
# ════════════════════════════════════════════════════════════════
echo "=== [Phase 1] Training start at $(date) ==="

prev_model="${base_model}"

for i in "${!stages[@]}"; do
    stage="${stages[$i]}"
    task="${tasks[$i]}"
    port=$(shuf -i25000-30000 -n1)

    echo "--- Training stage ${stage} (from: ${prev_model}) ---"

    CUDA_VISIBLE_DEVICES=$gpu_ids ${torchrun_bin} \
        --nproc_per_node=${num_procs} \
        --master_port=${port} \
        src/run_uie_lora.py \
        --seed ${seed} --data_seed ${seed} \
        --do_train True \
        --bf16 True \
        --train_method lora \
        --lora_strategy seqlora \
        --lora_dim ${lora_dim} \
        ${optimizer_args} \
        --learning_rate ${lr} \
        --model_name_or_path ${prev_model} \
        --data_dir CL_Benchmark \
        --task_config_dir configs/order1_configs/${task} \
        --instruction_file configs/instruction_config.json \
        --instruction_strategy single \
        --output_dir ${run_root}/${stage} \
        --per_device_train_batch_size ${train_bs} \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1 \
        --run_name ${method}_order1 \
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
        --save_strategy no

    prev_model="${run_root}/${stage}/adapter"
    sleep 4
done

echo "=== [Phase 1] Training done at $(date) ==="

# ════════════════════════════════════════════════════════════════
# PHASE 2: Parallel evaluation (1 GPU per job, round-robin across all GPUs)
# ════════════════════════════════════════════════════════════════
echo "=== [Phase 2] Evaluation start at $(date) ==="

# Build list of (stage_idx, test_task) pairs - only evaluate tasks learned so far
eval_jobs=()   # each entry: "stage_idx:test_task"
for i in "${!stages[@]}"; do
    for j in $(seq 0 $i); do
        eval_jobs+=("${i}:${tasks[$j]}")
    done
done

total_jobs=${#eval_jobs[@]}
max_parallel=${num_procs}
job_count=0

for job in "${eval_jobs[@]}"; do
    i="${job%%:*}"
    test_task="${job##*:}"
    stage="${stages[$i]}"
    model_adapter="${run_root}/${stage}/adapter"
    out_dir="${run_root}/${stage}/evalFlat-${test_task}-trainedLoRAwhole"
    gpu_id="${_gpu_array[$((job_count % max_parallel))]}"
    port=$(shuf -i25000-30000 -n1)

    echo "Launching eval: stage=${stage} test=${test_task} gpu=${gpu_id}"
    mkdir -p "${out_dir}"

    (
        CUDA_VISIBLE_DEVICES=${gpu_id} ${torchrun_bin} \
            --nproc_per_node=1 \
            --master_port=${port} \
            src/run_uie_lora.py \
            --do_predict True \
            --predict_with_generate \
            --max_train_sample 0 \
            --bf16 True \
            --flat_part trained_whole \
            --train_method lora \
            --lora_strategy seqlora \
            --lora_dim ${lora_dim} \
            --optimizer_type adamw_hf \
            --learning_rate ${lr} \
            --model_name_or_path ${model_adapter} \
            --data_dir CL_Benchmark \
            --task_config_dir configs/lineraProbe_order1_configs/${test_task} \
            --instruction_file configs/instruction_config.json \
            --instruction_strategy single \
            --output_dir ${out_dir} \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --num_train_epochs 0 \
            --run_name ${method}_order1_eval \
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
            --save_strategy no
    ) &

    job_count=$((job_count + 1))
    if [[ $((job_count % max_parallel)) -eq 0 ]]; then
        echo ">>> Waiting for eval batch (jobs $((job_count - max_parallel + 1))–${job_count} / ${total_jobs})..."
        wait
    fi
done

wait
echo "=== [Phase 2] Evaluation done at $(date) ==="

# ════════════════════════════════════════════════════════════════
# PHASE 3: Summarize results
# ════════════════════════════════════════════════════════════════
echo "=== [Phase 3] Summarizing results ==="

python scripts_order1/summarize_order1.py --run_root "${run_root}"

echo "=== Pipeline complete at $(date) ==="
echo "Results saved to: ${run_root}/summary.xlsx"
