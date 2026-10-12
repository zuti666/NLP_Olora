#!/bin/bash
set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

conda_base=${FISHERSHAPE_CONDA_BASE:-/data/115-2/users/liying/miniconda3_shared}
conda_env=${FISHERSHAPE_CONDA_ENV:-nlora_req}
if [[ ${FISHERSHAPE_SKIP_CONDA_ACTIVATE:-false} != true ]]; then
  conda_init="${conda_base}/etc/profile.d/conda.sh"
  if [[ ! -f ${conda_init} ]]; then
    echo "Conda initialization file not found: ${conda_init}" >&2
    echo "Set FISHERSHAPE_CONDA_BASE or FISHERSHAPE_SKIP_CONDA_ACTIVATE=true." >&2
    exit 2
  fi
  source "${conda_init}"
  conda activate "${conda_env}"
fi
set -u

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=${FISHERSHAPE_TRANSFORMERS_CACHE:-/data/140-0/users/liying/.cache/huggingface}

# Usage:
#   bash scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh \
#     [gpu_ids] [seed] [lambda] [gamma] [fs_alpha] [fs_rho] \
#     [normalize_fisher] [damping_beta] [include_olora]
#
# The optional tenth argument is used internally by the Order 2/3 wrappers.

port=$(shuf -i25000-30000 -n1)
gpu_ids=${1:-0,1,2,3,4,5,6,7}
seed=${2:-1993}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
fs_alpha=${5:-0.2}
fs_rho=${6:-0.05}
normalize_fisher=${7:-true}
damping_beta=${8:-0.2}
include_olora=${9:-false}
task_order=${10:-1}

case "${task_order}" in
  1)
    tasks=(dbpedia amazon yahoo agnews)
    ;;
  2)
    tasks=(dbpedia amazon agnews yahoo)
    ;;
  3)
    tasks=(yahoo amazon agnews dbpedia)
    ;;
  *)
    echo "FISHERSHAPE_ORDER must be 1, 2, or 3; got ${task_order}" >&2
    exit 2
    ;;
esac

IFS=',' read -ra _gpu_array <<< "${gpu_ids}"
num_procs=${#_gpu_array[@]}
if [[ ${num_procs} -lt 1 ]]; then
  echo "gpu_ids must contain at least one CUDA device." >&2
  exit 2
fi

train_batch_size=${FISHERSHAPE_PER_DEVICE_TRAIN_BATCH_SIZE:-1}
eval_batch_size=${FISHERSHAPE_PER_DEVICE_EVAL_BATCH_SIZE:-4}
target_global_batch=${FISHERSHAPE_GLOBAL_BATCH_SIZE:-64}
for positive_integer in "${train_batch_size}" "${eval_batch_size}" "${target_global_batch}"; do
  if [[ ! ${positive_integer} =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch-size settings must be positive integers; got ${positive_integer}." >&2
    exit 2
  fi
done
if [[ -n ${FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS:-} ]]; then
  gradient_accumulation_steps=${FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS}
elif (( target_global_batch % (num_procs * train_batch_size) == 0 )); then
  gradient_accumulation_steps=$((target_global_batch / (num_procs * train_batch_size)))
else
  echo "FISHERSHAPE_GLOBAL_BATCH_SIZE=${target_global_batch} is not divisible by" >&2
  echo "num_procs(${num_procs}) * train_batch_size(${train_batch_size})." >&2
  echo "Set FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS explicitly." >&2
  exit 2
fi
if [[ ! ${gradient_accumulation_steps} =~ ^[1-9][0-9]*$ ]]; then
  echo "Gradient accumulation steps must be a positive integer." >&2
  exit 2
fi

base_model=${FISHERSHAPE_BASE_MODEL:-/data/140-0/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B}
if [[ ${base_model} == /* && ! -e ${base_model} ]]; then
  echo "Base model path not found: ${base_model}" >&2
  echo "Set FISHERSHAPE_BASE_MODEL to an existing path or Hugging Face model ID." >&2
  exit 2
fi
output_base=${FISHERSHAPE_OUTPUT_ROOT:-logs_and_outputs_llama_1B}
run_root=${output_base}/order_${task_order}/EWCLoRA-FisherShape_Llama-3.2-1B_torch_FS-AdamwHF_Lora8_Ba${train_batch_size}-GA${gradient_accumulation_steps}_E1_LR1e-3_to_1e-4_lambda${lambda_ewc}_gamma${gamma_ewc}_ratio${fs_alpha}_rho${fs_rho}_norm${normalize_fisher}_damp${damping_beta}_olora${include_olora}_seed-${seed}

train_round() {
  local model_path="$1"
  local task_config_dir="$2"
  local output_dir="$3"
  local run_name="$4"
  local learning_rate="$5"

  CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun \
    --nproc_per_node=${num_procs} \
    --master_port=${port} \
    src/run_uie_lora.py \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --bf16 True \
    --train_method lora \
    --lora_strategy ewclora_fishershape \
    --lora_dim 8 \
    --ewclora_lambda "${lambda_ewc}" \
    --ewclora_gamma "${gamma_ewc}" \
    --ewclora_normalize_fisher "${normalize_fisher}" \
    --ewclora_normalize_accumulated_fisher false \
    --ewclora_fisher_damping_beta "${damping_beta}" \
    --optimizer_type fishershape-adamw_hf \
    --fs_alpha_mode ratio \
    --fs_alpha "${fs_alpha}" \
    --fs_rho "${fs_rho}" \
    --fs_xi 1e-12 \
    --fs_sam_equivalent false \
    --fs_adaptive false \
    --fs_grad_reduce mean \
    --fs_replay_rng_state true \
    --fs_include_olora_regularization "${include_olora}" \
    --model_name_or_path "${model_path}" \
    --data_dir CL_Benchmark \
    --task_config_dir "${task_config_dir}" \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "${output_dir}" \
    --per_device_train_batch_size "${train_batch_size}" \
    --per_device_eval_batch_size "${eval_batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --learning_rate "${learning_rate}" \
    --num_train_epochs 1 \
    --run_name "${run_name}" \
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
    --save_steps 1500
}

previous_adapter=""
for task_index in "${!tasks[@]}"; do
  round=$((task_index + 1))
  task=${tasks[$task_index]}
  stage_dir="${run_root}/${round}-${task}"
  if [[ ${round} -eq 1 ]]; then
    model_path=${base_model}
    learning_rate=1e-03
  else
    model_path=${previous_adapter}
    learning_rate=1e-04
  fi

  train_round \
    "${model_path}" \
    "configs/order${task_order}_configs/${task}" \
    "${stage_dir}" \
    "order${task_order}_ewclora_fishershape_round${round}" \
    "${learning_rate}"

  previous_adapter="${stage_dir}/adapter"
  if [[ ! -d ${previous_adapter} || ! -f ${previous_adapter}/ewclora_fisher.pt ]]; then
    echo "Stage ${round} did not produce a complete adapter/Fisher chain at:" >&2
    echo "  ${previous_adapter}" >&2
    exit 1
  fi
  if [[ ${round} -lt ${#tasks[@]} ]]; then
    sleep 5
  fi
done
