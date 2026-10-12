#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Shared Llama-3.2-1B launcher for two explicit strategies:
#   inclora_sam_ewc: fresh rank-8 branch per task, concat persistence.
#   seqlora_sam_ewc: one fixed rank-8 shared branch, sequential persistence.
# Both use two CE probes for the SAM direction and add EWC once at clean weights.
#
# Usage:
#   bash scripts_llama/run_sam_ewc_llama.sh \
#     <inclora_sam_ewc|seqlora_sam_ewc> <order:1|2|3> \
#     [gpu_ids] [seed] [ewc_lambda] [ewc_gamma]

if (( $# < 2 )); then
  echo "Usage: $0 <inclora_sam_ewc|seqlora_sam_ewc> <order:1|2|3> [gpu_ids] [seed] [ewc_lambda] [ewc_gamma]" >&2
  exit 2
fi

method=${1,,}
task_order=$2
gpu_ids=${3:-6}
seed=${4:-1993}
lambda_ewc=${5:-1.0}
gamma_ewc=${6:-1.0}

case "${method}" in
  inclora_sam_ewc)
    method_label=IncLoRA-SAM-EWC
    ;;
  seqlora_sam_ewc)
    method_label=SeqLoRA-SAM-EWC
    ;;
  *)
    echo "Method must be inclora_sam_ewc or seqlora_sam_ewc; got ${method}." >&2
    exit 2
    ;;
esac

case "${task_order}" in
  1) tasks=(dbpedia amazon yahoo agnews) ;;
  2) tasks=(dbpedia amazon agnews yahoo) ;;
  3) tasks=(yahoo amazon agnews dbpedia) ;;
  *)
    echo "Task order must be 1, 2, or 3; got ${task_order}." >&2
    exit 2
    ;;
esac

python_executable=${SAM_EWC_LLAMA_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}
if [[ ! -x ${python_executable} ]]; then
  echo "Python executable not found: ${python_executable}" >&2
  echo "Set SAM_EWC_LLAMA_PYTHON to the nlora_req Python executable." >&2
  exit 2
fi
if ! "${python_executable}" -c '
import math, sys
lam, gamma = map(float, sys.argv[1:])
raise SystemExit(0 if math.isfinite(lam) and lam >= 0 and math.isfinite(gamma) and gamma >= 0 else 1)
' "${lambda_ewc}" "${gamma_ewc}"; then
  echo "EWC lambda and gamma must be finite and non-negative." >&2
  exit 2
fi

base_model=${SAM_EWC_LLAMA_BASE_MODEL:-/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B}
output_base=${SAM_EWC_LLAMA_OUTPUT_ROOT:-${repo_root}/logs_and_outputs_llama_1B}
if [[ ${base_model} == /* && ! -e ${base_model} ]]; then
  echo "Base model path not found: ${base_model}" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cache_root=${SAM_EWC_LLAMA_CACHE_ROOT:-${repo_root}/.cache/sam_ewc_llama1b}
export HF_HOME=${SAM_EWC_LLAMA_HF_HOME:-${cache_root}/huggingface}
export HF_MODULES_CACHE=${SAM_EWC_LLAMA_HF_MODULES_CACHE:-${HF_HOME}/modules}
export HF_DATASETS_CACHE=${SAM_EWC_LLAMA_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${SAM_EWC_LLAMA_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export UIE_DATA_CACHE_ROOT=${SAM_EWC_LLAMA_UIE_DATA_CACHE_ROOT:-${cache_root}/uie_data}
mkdir -p "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UIE_DATA_CACHE_ROOT}"

IFS=',' read -ra gpu_array <<< "${gpu_ids}"
num_procs=${#gpu_array[@]}
if (( num_procs < 1 )); then
  echo "gpu_ids must contain at least one CUDA device." >&2
  exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
  if [[ ! ${gpu_id} =~ ^[0-9]+$ ]]; then
    echo "GPU identifiers must be non-negative integers; got ${gpu_id}." >&2
    exit 2
  fi
done

train_batch_size=${SAM_EWC_LLAMA_PER_DEVICE_TRAIN_BATCH_SIZE:-1}
eval_batch_size=${SAM_EWC_LLAMA_PER_DEVICE_EVAL_BATCH_SIZE:-4}
target_global_batch=${SAM_EWC_LLAMA_GLOBAL_BATCH_SIZE:-64}
for value in "${train_batch_size}" "${eval_batch_size}" "${target_global_batch}"; do
  if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch settings must be positive integers; got ${value}." >&2
    exit 2
  fi
done
if [[ -n ${SAM_EWC_LLAMA_GRADIENT_ACCUMULATION_STEPS:-} ]]; then
  gradient_accumulation_steps=${SAM_EWC_LLAMA_GRADIENT_ACCUMULATION_STEPS}
else
  batch_denominator=$((num_procs * train_batch_size))
  if (( target_global_batch % batch_denominator != 0 )); then
    echo "Global batch ${target_global_batch} is not divisible by ${batch_denominator}." >&2
    exit 2
  fi
  gradient_accumulation_steps=$((target_global_batch / batch_denominator))
fi
if [[ ! ${gradient_accumulation_steps} =~ ^[1-9][0-9]*$ ]]; then
  echo "Gradient accumulation must be a positive integer." >&2
  exit 2
fi

max_stages=${SAM_EWC_LLAMA_MAX_STAGES:-4}
fisher_max_batches=${SAM_EWC_LLAMA_FISHER_MAX_BATCHES:-0}
sleep_seconds=${SAM_EWC_LLAMA_SLEEP_SECONDS:-5}
bf16=${SAM_EWC_LLAMA_BF16:-true}
do_predict=${SAM_EWC_LLAMA_DO_PREDICT:-true}
dry_run=${SAM_EWC_LLAMA_DRY_RUN:-false}
force=${SAM_EWC_LLAMA_FORCE:-false}
resume=${SAM_EWC_LLAMA_RESUME:-true}
require_free_gpu=${SAM_EWC_LLAMA_REQUIRE_FREE_GPU:-true}
gpu_free_threshold_mib=${SAM_EWC_LLAMA_GPU_FREE_THRESHOLD_MIB:-500}
use_torchrun=${SAM_EWC_LLAMA_USE_TORCHRUN:-auto}
rho=${SAM_EWC_LLAMA_RHO:-0.05}

if [[ ! ${max_stages} =~ ^[1-4]$ ]]; then
  echo "SAM_EWC_LLAMA_MAX_STAGES must be 1..4." >&2
  exit 2
fi
if [[ ! ${fisher_max_batches} =~ ^[0-9]+$ ]]; then
  echo "SAM_EWC_LLAMA_FISHER_MAX_BATCHES must be non-negative." >&2
  exit 2
fi
for value in "${bf16}" "${do_predict}" "${dry_run}" "${force}" \
  "${resume}" "${require_free_gpu}"; do
  if [[ ${value} != true && ${value} != false ]]; then
    echo "Boolean settings must be true or false; got ${value}." >&2
    exit 2
  fi
done
if [[ ${use_torchrun} == auto ]]; then
  if (( num_procs > 1 )); then use_torchrun=true; else use_torchrun=false; fi
fi
if [[ ${use_torchrun} != true && ${use_torchrun} != false ]]; then
  echo "SAM_EWC_LLAMA_USE_TORCHRUN must be auto, true, or false." >&2
  exit 2
fi
if [[ ${use_torchrun} == false && ${num_procs} -ne 1 ]]; then
  echo "Disabling torchrun requires exactly one GPU." >&2
  exit 2
fi

check_selected_gpus_free() {
  local snapshot gpu_id used
  snapshot=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for gpu_id in "${gpu_array[@]}"; do
    used=$(awk -F',' -v wanted="${gpu_id}" '
      {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2)}
      $1 == wanted {print $2; exit}
    ' <<< "${snapshot}")
    if [[ -z ${used} || ${used} -ge ${gpu_free_threshold_mib} ]]; then
      echo "GPU ${gpu_id} is unavailable or busy (${used:-unknown} MiB used)." >&2
      return 1
    fi
  done
}
if [[ ${dry_run} != true && ${require_free_gpu} == true ]]; then
  check_selected_gpus_free || exit 75
fi

run_tag=${SAM_EWC_LLAMA_RUN_TAG:-${method_label}_Llama-3.2-1B_Lora8_Ba${train_batch_size}-GA${gradient_accumulation_steps}-GB${target_global_batch}_E1_LR1e-3_to_1e-4_rho${rho}_lambda${lambda_ewc}_gamma${gamma_ewc}_seed-${seed}}
run_root=${output_base}/order_${task_order}/${run_tag}

extra_args=()
if [[ -n ${SAM_EWC_LLAMA_MAX_STEPS:-} ]]; then
  extra_args+=(--max_steps "${SAM_EWC_LLAMA_MAX_STEPS}")
fi
if [[ -n ${SAM_EWC_LLAMA_MAX_TRAIN_SAMPLES:-} ]]; then
  extra_args+=(--max_train_samples "${SAM_EWC_LLAMA_MAX_TRAIN_SAMPLES}")
fi
if [[ -n ${SAM_EWC_LLAMA_MAX_PREDICT_SAMPLES:-} ]]; then
  extra_args+=(--max_predict_samples "${SAM_EWC_LLAMA_MAX_PREDICT_SAMPLES}")
fi
predict_args=()
if [[ ${do_predict} == true ]]; then
  predict_args+=(--do_predict --predict_with_generate)
fi

stage_is_complete() {
  local stage_dir=$1
  [[ -f ${stage_dir}/adapter/adapter_config.json ]] \
    && [[ -f ${stage_dir}/adapter/adapter_model.bin ]] \
    && [[ -f ${stage_dir}/adapter/ewclora_fisher.pt ]] \
    && [[ -f ${stage_dir}/adapter_chain_verification.json ]] \
    && rg -q '"status"[[:space:]]*:[[:space:]]*"pass"' \
      "${stage_dir}/adapter_chain_verification.json"
}

train_round() {
  local round=$1 task=$2 model_path=$3 stage_dir=$4 learning_rate=$5
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
    --seed "${seed}" --data_seed "${seed}"
    --do_train "${predict_args[@]}"
    --bf16 "${bf16}"
    --train_method lora
    --lora_strategy "${method}"
    --lora_dim 8
    --optimizer_type fishershape-adamw_hf
    --fs_rho "${rho}"
    --fs_alpha 0.0
    --fs_alpha_mode ratio
    --fs_sam_equivalent true
    --fs_adaptive false
    --fs_xi 1e-12
    --fs_grad_reduce mean
    --fs_replay_rng_state false
    --fs_include_olora_regularization false
    --olora_lamda_1 0.0
    --olora_lamda_2 0.0
    --ewclora_lambda "${lambda_ewc}"
    --ewclora_gamma "${gamma_ewc}"
    --ewclora_fisher_max_batches "${fisher_max_batches}"
    --ewclora_normalize_fisher false
    --ewclora_normalize_accumulated_fisher false
    --ewclora_fisher_damping_beta 0.0
    --adam_beta1 0.9 --adam_beta2 0.999
    --adam_epsilon 1e-6 --weight_decay 0.0
    --model_name_or_path "${model_path}"
    --data_dir CL_Benchmark
    --task_config_dir "configs/order${task_order}_configs/${task}"
    --instruction_file configs/instruction_config.json
    --instruction_strategy single
    --output_dir "${stage_dir}"
    --per_device_train_batch_size "${train_batch_size}"
    --per_device_eval_batch_size "${eval_batch_size}"
    --gradient_accumulation_steps "${gradient_accumulation_steps}"
    --learning_rate "${learning_rate}"
    --num_train_epochs 1
    --run_name "order${task_order}_${method_label}_round${round}_${task}"
    --max_source_length 512 --max_target_length 50 --generation_max_length 50
    --ddp_find_unused_parameters false
    --add_task_name true --add_dataset_name true
    --overwrite_output_dir --overwrite_cache
    --lr_scheduler_type constant --warmup_steps 0
    --logging_strategy steps --logging_steps 10 --report_to none
    --evaluation_strategy no --save_strategy no --save_steps 1500
    "${extra_args[@]}"
  )
  if (( round > 1 )); then
    train_args+=(--ewclora_fisher_path "${model_path}")
  fi

  echo ">>> method=${method} order=${task_order} stage=${round} task=${task} lr=${learning_rate} GPUs=${gpu_ids}"
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
  stage_dir=${run_root}/${round}-${task}
  current_adapter=${stage_dir}/adapter

  if [[ ${resume} == true ]] && stage_is_complete "${stage_dir}"; then
    echo ">>> Resuming: verified stage already complete, skipping ${stage_dir}"
    model_path=${current_adapter}
    continue
  fi
  if [[ -e ${stage_dir} && ${force} != true && ${dry_run} != true ]]; then
    echo "Refusing to overwrite existing/incomplete stage: ${stage_dir}" >&2
    echo "Use SAM_EWC_LLAMA_FORCE=true or choose SAM_EWC_LLAMA_RUN_TAG." >&2
    exit 2
  fi

  input_model_path=${model_path}
  learning_rate=1e-04
  if (( round == 1 )); then learning_rate=1e-03; fi
  train_round "${round}" "${task}" "${input_model_path}" "${stage_dir}" "${learning_rate}"

  if [[ ${dry_run} == true ]]; then
    model_path=${current_adapter}
    continue
  fi
  if [[ ! -f ${current_adapter}/adapter_config.json \
        || ! -f ${current_adapter}/adapter_model.bin \
        || ! -f ${current_adapter}/ewclora_fisher.pt ]]; then
    echo "Stage ${round} did not produce a complete adapter/Fisher chain." >&2
    exit 1
  fi
  verifier_args=(
    --current "${current_adapter}"
    --method "${method}"
    --output "${stage_dir}/adapter_chain_verification.json"
  )
  if (( round > 1 )); then verifier_args+=(--previous "${input_model_path}"); fi
  "${python_executable}" \
    scripts_order1_llama1B/verify_sam_ewc_adapter_chain.py \
    "${verifier_args[@]}"

  model_path=${current_adapter}
  if (( round < max_stages )); then sleep "${sleep_seconds}"; fi
done

if [[ ${dry_run} == true ]]; then
  echo "Dry run completed: ${method}, Order ${task_order}."
else
  echo "Completed ${method}, Order ${task_order}: ${run_root}"
fi
