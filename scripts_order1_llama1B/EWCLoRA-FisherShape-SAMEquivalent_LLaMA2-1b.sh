#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Despite the historical filename, the referenced baseline uses Llama-3.2-1B.
#
# Usage:
#   # IncLoRA + SAM-equivalent direction, with no EWC gradient
#   bash scripts_order1_llama1B/EWCLoRA-FisherShape-SAMEquivalent_LLaMA2-1b.sh \
#     0,1,2,3,4,5,6,7 42 sam
#
#   # The same SAM-equivalent direction plus EWC from stage 2 onward
#   bash scripts_order1_llama1B/EWCLoRA-FisherShape-SAMEquivalent_LLaMA2-1b.sh \
#     0,1,2,3,4,5,6,7 42 sam_ewc 1.0 1.0
#
# Positional arguments:
#   1 gpu_ids, 2 seed, 3 mode(sam|sam_ewc), 4 EWC lambda, 5 EWC gamma
#
# A two-stage/two-step GPU smoke (two steps let the initially zero Delta-W_new
# move before checking for a non-zero EWC gradient):
#   FISHERSHAPE_MAX_STAGES=2 FISHERSHAPE_MAX_STEPS=2 \
#   FISHERSHAPE_MAX_TRAIN_SAMPLES=16 FISHERSHAPE_FISHER_MAX_BATCHES=1 \
#   FISHERSHAPE_DO_PREDICT=false FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS=1 \
#   bash scripts_order1_llama1B/EWCLoRA-FisherShape-SAMEquivalent_LLaMA2-1b.sh \
#     0 42 sam_ewc 1.0 1.0

python_executable=${FISHERSHAPE_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}
if [[ ! -x ${python_executable} ]]; then
  echo "Python executable not found or not executable: ${python_executable}" >&2
  echo "Set FISHERSHAPE_PYTHON to the nlora_req Python executable." >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cache_root=${FISHERSHAPE_CACHE_ROOT:-${repo_root}/.cache/fishershape_llama1b}
export HF_HOME=${FISHERSHAPE_HF_HOME:-${cache_root}/huggingface}
export HF_MODULES_CACHE=${FISHERSHAPE_HF_MODULES_CACHE:-${HF_HOME}/modules}
export HF_DATASETS_CACHE=${FISHERSHAPE_HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${FISHERSHAPE_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export UIE_DATA_CACHE_ROOT=${FISHERSHAPE_UIE_DATA_CACHE_ROOT:-${cache_root}/uie_data}
mkdir -p \
  "${HF_MODULES_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${UIE_DATA_CACHE_ROOT}"

gpu_ids=${1:-0,1,2,3,4,5,6,7}
# The reference IncLoRA-SAM script did not pass --seed, so its effective
# Transformers default was 42.  Keep that value as the comparison default.
seed=${2:-42}
comparison_mode=${3:-sam}
requested_lambda=${4:-1.0}
gamma_ewc=${5:-1.0}

case "${comparison_mode}" in
  sam)
    lambda_ewc=0.0
    method_tag=IncLoRA-SAM-via-FisherShape
    ;;
  sam_ewc)
    lambda_ewc=${requested_lambda}
    method_tag=EWCLoRA-SAM-via-FisherShape
    ;;
  *)
    echo "Mode must be 'sam' or 'sam_ewc'; got ${comparison_mode}." >&2
    exit 2
    ;;
esac

IFS=',' read -ra gpu_array <<< "${gpu_ids}"
num_procs=${#gpu_array[@]}
if (( num_procs < 1 )); then
  echo "gpu_ids must contain at least one CUDA device." >&2
  exit 2
fi

train_batch_size=${FISHERSHAPE_PER_DEVICE_TRAIN_BATCH_SIZE:-1}
eval_batch_size=${FISHERSHAPE_PER_DEVICE_EVAL_BATCH_SIZE:-4}
target_global_batch=${FISHERSHAPE_GLOBAL_BATCH_SIZE:-64}
for positive_integer in "${train_batch_size}" "${eval_batch_size}" "${target_global_batch}"; do
  if [[ ! ${positive_integer} =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch sizes must be positive integers; got ${positive_integer}." >&2
    exit 2
  fi
done

if [[ -n ${FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS:-} ]]; then
  gradient_accumulation_steps=${FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS}
elif (( target_global_batch % (num_procs * train_batch_size) == 0 )); then
  gradient_accumulation_steps=$((target_global_batch / (num_procs * train_batch_size)))
else
  echo "Global batch ${target_global_batch} is not divisible by" >&2
  echo "num_procs(${num_procs}) * train_batch_size(${train_batch_size})." >&2
  echo "Set FISHERSHAPE_GRADIENT_ACCUMULATION_STEPS explicitly." >&2
  exit 2
fi
if [[ ! ${gradient_accumulation_steps} =~ ^[1-9][0-9]*$ ]]; then
  echo "Gradient accumulation steps must be a positive integer." >&2
  exit 2
fi

base_model=${FISHERSHAPE_BASE_MODEL:-/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B}
if [[ ${base_model} == /* && ! -e ${base_model} ]]; then
  echo "Base model path not found: ${base_model}" >&2
  echo "Set FISHERSHAPE_BASE_MODEL to an existing model path or model ID." >&2
  exit 2
fi

output_base=${FISHERSHAPE_OUTPUT_ROOT:-${repo_root}/logs_and_outputs_llama_1B}
max_stages=${FISHERSHAPE_MAX_STAGES:-4}
sleep_seconds=${FISHERSHAPE_SLEEP_SECONDS:-5}
bf16=${FISHERSHAPE_BF16:-true}
do_predict=${FISHERSHAPE_DO_PREDICT:-true}
use_torchrun=${FISHERSHAPE_USE_TORCHRUN:-true}
dry_run=${FISHERSHAPE_DRY_RUN:-false}
normalize_fisher=${FISHERSHAPE_NORMALIZE_FISHER:-false}
normalize_accumulated_fisher=${FISHERSHAPE_NORMALIZE_ACCUMULATED_FISHER:-false}
damping_beta=${FISHERSHAPE_FISHER_DAMPING_BETA:-0.0}

if [[ ! ${max_stages} =~ ^[1-4]$ ]]; then
  echo "FISHERSHAPE_MAX_STAGES must be an integer from 1 to 4." >&2
  exit 2
fi
for boolean_value in "${bf16}" "${do_predict}" "${use_torchrun}" "${dry_run}" \
  "${normalize_fisher}" "${normalize_accumulated_fisher}"; do
  if [[ ${boolean_value} != true && ${boolean_value} != false ]]; then
    echo "Boolean settings must be true or false; got ${boolean_value}." >&2
    exit 2
  fi
done
if [[ ${use_torchrun} == false && ${num_procs} -ne 1 ]]; then
  echo "FISHERSHAPE_USE_TORCHRUN=false requires one visible GPU id." >&2
  exit 2
fi

# Full experiments use the whole train loader for post-task Fisher.  A max-step
# smoke defaults to one Fisher batch unless explicitly overridden.
if [[ -n ${FISHERSHAPE_FISHER_MAX_BATCHES:-} ]]; then
  fisher_max_batches=${FISHERSHAPE_FISHER_MAX_BATCHES}
elif [[ -n ${FISHERSHAPE_MAX_STEPS:-} ]]; then
  fisher_max_batches=1
else
  fisher_max_batches=0
fi
if [[ ! ${fisher_max_batches} =~ ^[0-9]+$ ]]; then
  echo "FISHERSHAPE_FISHER_MAX_BATCHES must be a non-negative integer." >&2
  exit 2
fi

run_root=${output_base}/order_1/${method_tag}_Llama-3.2-1B_Lora8_Ba${train_batch_size}-GA${gradient_accumulation_steps}_E1_LR1e-3_to_1e-4_rho0.05_lambda${lambda_ewc}_gamma${gamma_ewc}_norm${normalize_fisher}_damp${damping_beta}_seed-${seed}
disabled_fisher_input=${run_root}/no_fisher_input.pt

tasks=(dbpedia amazon yahoo agnews)
task_configs=(
  configs/order1_configs/dbpedia
  configs/order1_configs/amazon
  configs/order1_configs/yahoo
  configs/order1_configs/agnews
)

extra_args=()
if [[ -n ${FISHERSHAPE_MAX_STEPS:-} ]]; then
  extra_args+=(--max_steps "${FISHERSHAPE_MAX_STEPS}")
fi
if [[ -n ${FISHERSHAPE_MAX_TRAIN_SAMPLES:-} ]]; then
  extra_args+=(--max_train_samples "${FISHERSHAPE_MAX_TRAIN_SAMPLES}")
fi
if [[ -n ${FISHERSHAPE_MAX_PREDICT_SAMPLES:-} ]]; then
  extra_args+=(--max_predict_samples "${FISHERSHAPE_MAX_PREDICT_SAMPLES}")
fi

predict_args=()
if [[ ${do_predict} == true ]]; then
  predict_args+=(--do_predict --predict_with_generate)
fi

train_round() {
  local round=$1
  local model_path=$2
  local task=$3
  local task_config=$4
  local stage_dir=$5
  local learning_rate=$6
  local master_port
  master_port=$(shuf -i25000-30000 -n1)

  local -a launcher=("${python_executable}")
  if [[ ${use_torchrun} == true ]]; then
    launcher+=(
      -m torch.distributed.run
      --nproc_per_node="${num_procs}"
      --master_port="${master_port}"
    )
  fi

  local -a train_args=(
    src/run_uie_lora.py
    --seed "${seed}"
    --data_seed "${seed}"
    --do_train
    "${predict_args[@]}"
    --bf16 "${bf16}"
    --train_method lora
    --lora_strategy ewclora_fishershape
    --lora_dim 8
    --optimizer_type fishershape-adamw_hf
    --fs_rho 0.05
    --fs_alpha 0.0
    --fs_alpha_mode ratio
    --fs_sam_equivalent true
    --fs_adaptive false
    --fs_xi 1e-12
    --fs_grad_reduce mean
    --fs_replay_rng_state false
    --fs_include_olora_regularization false
    --ewclora_lambda "${lambda_ewc}"
    --ewclora_gamma "${gamma_ewc}"
    --ewclora_fisher_max_batches "${fisher_max_batches}"
    --ewclora_normalize_fisher "${normalize_fisher}"
    --ewclora_normalize_accumulated_fisher "${normalize_accumulated_fisher}"
    --ewclora_fisher_damping_beta "${damping_beta}"
    --olora_lamda_1 0.0
    --olora_lamda_2 0.0
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_epsilon 1e-6
    --weight_decay 0.0
    --model_name_or_path "${model_path}"
    --data_dir CL_Benchmark
    --task_config_dir "${task_config}"
    --instruction_file configs/instruction_config.json
    --instruction_strategy single
    --output_dir "${run_root}/${stage_dir}"
    --per_device_train_batch_size "${train_batch_size}"
    --per_device_eval_batch_size "${eval_batch_size}"
    --gradient_accumulation_steps "${gradient_accumulation_steps}"
    --learning_rate "${learning_rate}"
    --num_train_epochs 1
    --run_name "order1_${method_tag}_round${round}_${task}"
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --ddp_find_unused_parameters false
    --add_task_name true
    --add_dataset_name true
    --overwrite_output_dir
    --overwrite_cache
    --lr_scheduler_type constant
    --warmup_steps 0
    --logging_strategy steps
    --logging_steps 10
    --report_to none
    --evaluation_strategy no
    --save_strategy no
    --save_steps 1500
    "${extra_args[@]}"
  )

  if [[ ${comparison_mode} == sam ]]; then
    # Prevent the chained adapter path from auto-loading a sidecar.  lambda=0
    # already zeros the loss; this also avoids constructing the EWC graph.
    train_args+=(--ewclora_fisher_path "${disabled_fisher_input}")
  elif (( round > 1 )); then
    # Make the EWC state chain explicit instead of depending on path-name
    # auto-detection in run_uie_lora.py.
    train_args+=(--ewclora_fisher_path "${model_path}")
  fi

  echo ">>> mode=${comparison_mode} stage=${round} task=${task} lr=${learning_rate}"
  if [[ ${dry_run} == true ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q' "${gpu_ids}"
    printf ' %q' "${launcher[@]}" "${train_args[@]}"
    printf '\n'
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "${launcher[@]}" "${train_args[@]}"
}

model_path=${base_model}
for ((index = 0; index < max_stages; index++)); do
  round=$((index + 1))
  task=${tasks[index]}
  stage_dir="${round}-${task}"
  learning_rate=1e-04
  if (( round == 1 )); then
    learning_rate=1e-03
  fi

  input_model_path=${model_path}
  train_round \
    "${round}" \
    "${input_model_path}" \
    "${task}" \
    "${task_configs[index]}" \
    "${stage_dir}" \
    "${learning_rate}"

  current_adapter="${run_root}/${stage_dir}/adapter"
  if [[ ${dry_run} == true ]]; then
    model_path=${current_adapter}
    continue
  fi
  if [[ ! -d ${current_adapter} || ! -f ${current_adapter}/ewclora_fisher.pt ]]; then
    echo "Stage ${round} did not produce adapter and Fisher sidecar: ${current_adapter}" >&2
    exit 1
  fi

  verify_args=(
    "${python_executable}"
    scripts_order1_llama1B/verify_fishershape_adapter_chain.py
    --current "${current_adapter}"
    --mode "${comparison_mode}"
  )
  if (( round > 1 )); then
    verify_args+=(--previous "${input_model_path}")
  fi
  "${verify_args[@]}"

  model_path=${current_adapter}
  if (( round < max_stages )); then
    sleep "${sleep_seconds}"
  fi
done

if [[ ${dry_run} == true ]]; then
  echo "Dry run completed for mode=${comparison_mode}."
  exit 0
fi
echo "Completed ${comparison_mode}: ${run_root}"
