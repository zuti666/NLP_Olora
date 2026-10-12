#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

python_executable=${GAMFS_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}
if ! command -v "${python_executable}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_executable}" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cache_root=${GAMFS_CACHE_ROOT:-${repo_root}/.cache/gamfishershape}
export HF_HOME=${GAMFS_HF_HOME:-${cache_root}/huggingface}
export HF_MODULES_CACHE=${GAMFS_HF_MODULES_CACHE:-${HF_HOME}/modules}
export HF_DATASETS_CACHE=${GAMFS_HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${GAMFS_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export UIE_DATA_CACHE_ROOT=${GAMFS_UIE_DATA_CACHE_ROOT:-${cache_root}/uie_data}
mkdir -p \
  "${HF_MODULES_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${UIE_DATA_CACHE_ROOT}"

# Usage:
#   bash scripts_order1/EWCLoRA-GAMFisherShape-SAMEquivalent_E1_Le3.sh \
#     [gpu_ids] [seed]
#
# Default GPU 6 preserves the previous requested device.  With one GPU the
# script automatically uses GA=8, so 1 GPU * batch 8 * GA 8 = global batch 64,
# matching scripts_order1/EWCLoRA_E1_Le3.sh.  On four GPUs GA becomes 2.
gpu_ids=${1:-6}
seed=${2:-1993}
train_bs=${GAMFS_PER_DEVICE_TRAIN_BATCH_SIZE:-8}
eval_bs=${GAMFS_PER_DEVICE_EVAL_BATCH_SIZE:-16}
global_batch=${GAMFS_GLOBAL_BATCH_SIZE:-64}
base_model=${GAMFS_BASE_MODEL:-/data/140-1/users/liying/Sharing/Models/google-t5/t5-large}
output_base=${GAMFS_OUTPUT_ROOT:-${repo_root}/Logs_Short_Paper/Order-1_8}
max_stages=${GAMFS_MAX_STAGES:-4}
sleep_seconds=${GAMFS_SLEEP_SECONDS:-5}
bf16=${GAMFS_BF16:-true}
do_predict=${GAMFS_DO_PREDICT:-true}
fisher_max_batches=${GAMFS_FISHER_MAX_BATCHES:-0}
force=${GAMFS_FORCE:-false}
experiment=${GAMFS_EXPERIMENT:-sam}
ewc_lambda=${GAMFS_EWC_LAMBDA:-0.0}
ewc_gamma=${GAMFS_EWC_GAMMA:-1.0}

case ${experiment} in
  sam)
    include_olora=false
    olora_lambda1=0.0
    olora_lambda2=0.0
    run_prefix=GAMFisherShape-SAMEquivalent
    experiment_tag=""
    method_name=GAMFisherShape-SAM
    ;;
  sam_olora)
    include_olora=true
    olora_lambda1=${GAMFS_OLORA_LAMBDA1:-0.5}
    olora_lambda2=${GAMFS_OLORA_LAMBDA2:-0.0}
    run_prefix=GAMFisherShape-SAM-OLoRA
    experiment_tag="-L1-${olora_lambda1}-L2-${olora_lambda2}"
    method_name=GAMFisherShape-SAM-OLoRA
    ;;
  *)
    echo "GAMFS_EXPERIMENT must be 'sam' or 'sam_olora'; got ${experiment}." >&2
    exit 2
    ;;
esac

if [[ ${ewc_lambda} =~ ^[+-]?0+([.]0+)?$ ]]; then
  ewc_tag=""
  ewc_label=0
  verifier_mode=gamfs_sam
else
  ewc_tag="_EWC-lambda-${ewc_lambda}-gamma-${ewc_gamma}"
  ewc_label=${ewc_lambda}
  verifier_mode=sam_ewc
fi

if [[ ${base_model} == /* && ! -e ${base_model} ]]; then
  echo "Base model path not found: ${base_model}" >&2
  exit 2
fi
if [[ ! ${max_stages} =~ ^[1-4]$ ]]; then
  echo "GAMFS_MAX_STAGES must be an integer from 1 to 4; got ${max_stages}." >&2
  exit 2
fi
if [[ ${do_predict} != true && ${do_predict} != false ]]; then
  echo "GAMFS_DO_PREDICT must be true or false; got ${do_predict}." >&2
  exit 2
fi
if [[ ${force} != true && ${force} != false ]]; then
  echo "GAMFS_FORCE must be true or false; got ${force}." >&2
  exit 2
fi

IFS=',' read -ra gpu_array <<< "${gpu_ids}"
num_procs=${#gpu_array[@]}
if (( num_procs < 1 )); then
  echo "gpu_ids must contain at least one CUDA device." >&2
  exit 2
fi

if [[ -n ${GAMFS_GRADIENT_ACCUMULATION_STEPS:-} ]]; then
  gradient_accumulation_steps=${GAMFS_GRADIENT_ACCUMULATION_STEPS}
else
  batch_denominator=$((num_procs * train_bs))
  if (( global_batch % batch_denominator != 0 )); then
    echo "Global batch ${global_batch} is not divisible by ${num_procs} GPUs * batch ${train_bs}." >&2
    exit 2
  fi
  gradient_accumulation_steps=$((global_batch / batch_denominator))
fi
if (( gradient_accumulation_steps < 1 )); then
  echo "Gradient accumulation must be positive." >&2
  exit 2
fi

use_torchrun=${GAMFS_USE_TORCHRUN:-auto}
if [[ ${use_torchrun} == auto ]]; then
  if (( num_procs > 1 )); then
    use_torchrun=true
  else
    use_torchrun=false
  fi
fi
if [[ ${use_torchrun} != true && ${use_torchrun} != false ]]; then
  echo "GAMFS_USE_TORCHRUN must be auto, true, or false; got ${use_torchrun}." >&2
  exit 2
fi
if [[ ${use_torchrun} == false && ${num_procs} -ne 1 ]]; then
  echo "GAMFS_USE_TORCHRUN=false requires exactly one GPU." >&2
  exit 2
fi

run_tag=${GAMFS_RUN_TAG:-${run_prefix}${experiment_tag}${ewc_tag}-T5large_AdamW_Lora8_Ba${train_bs}-GA${gradient_accumulation_steps}-GB${global_batch}_E1_LRe3_rho005_lambda${ewc_label}_seed-${seed}}
run_root=${output_base}/${run_tag}

extra_args=()
if [[ -n ${GAMFS_MAX_STEPS:-} ]]; then
  extra_args+=(--max_steps "${GAMFS_MAX_STEPS}")
fi
if [[ -n ${GAMFS_MAX_TRAIN_SAMPLES:-} ]]; then
  extra_args+=(--max_train_samples "${GAMFS_MAX_TRAIN_SAMPLES}")
fi
if [[ -n ${GAMFS_MAX_PREDICT_SAMPLES:-} ]]; then
  extra_args+=(--max_predict_samples "${GAMFS_MAX_PREDICT_SAMPLES}")
fi

predict_args=()
if [[ ${do_predict} == true ]]; then
  predict_args+=(--do_predict true --predict_with_generate true)
fi

tasks=(dbpedia amazon yahoo agnews)
task_configs=(
  configs/order1_configs/dbpedia
  configs/order1_configs/amazon
  configs/order1_configs/yahoo
  configs/order1_configs/agnews
)

train_round() {
  local round=$1
  local model_path=$2
  local task=$3
  local task_config=$4
  local stage_dir=$5
  local output_dir=${run_root}/${stage_dir}
  local master_port
  master_port=$(shuf -i25000-30000 -n1)

  if [[ -e ${output_dir} && ${force} != true ]]; then
    echo "Refusing to overwrite existing stage: ${output_dir}" >&2
    echo "Set GAMFS_FORCE=true or choose a different GAMFS_RUN_TAG." >&2
    exit 2
  fi

  local -a launcher=("${python_executable}")
  if [[ ${use_torchrun} == true ]]; then
    launcher+=(
      -m torch.distributed.run
      --nproc_per_node="${num_procs}"
      --master_port="${master_port}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    "${launcher[@]}" \
      src/run_uie_lora.py \
      --seed "${seed}" --data_seed "${seed}" \
      --do_train true \
      "${predict_args[@]}" \
      --bf16 "${bf16}" \
      --train_method lora \
      --lora_strategy ewclora_gamfishershape \
      --lora_dim 8 \
      --optimizer_type gamfishershape-adamw_hf \
      --gamfs_mode sam \
      --gamfs_rho 0.05 \
      --gamfs_adaptive false \
      --gamfs_xi 1e-12 \
      --gamfs_grad_reduce mean \
      --gamfs_replay_rng_state false \
      --gamfs_include_olora_regularization "${include_olora}" \
      --ewclora_lambda "${ewc_lambda}" \
      --ewclora_gamma "${ewc_gamma}" \
      --ewclora_fisher_max_batches "${fisher_max_batches}" \
      --ewclora_normalize_fisher false \
      --ewclora_normalize_accumulated_fisher false \
      --ewclora_fisher_damping_beta 0.0 \
      --olora_lamda_1 "${olora_lambda1}" \
      --olora_lamda_2 "${olora_lambda2}" \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --adam_epsilon 1e-6 \
      --weight_decay 0.0 \
      --learning_rate 1e-03 \
      --model_name_or_path "${model_path}" \
      --data_dir CL_Benchmark \
      --task_config_dir "${task_config}" \
      --instruction_file configs/instruction_config.json \
      --instruction_strategy single \
      --output_dir "${output_dir}" \
      --per_device_train_batch_size "${train_bs}" \
      --per_device_eval_batch_size "${eval_bs}" \
      --gradient_accumulation_steps "${gradient_accumulation_steps}" \
      --num_train_epochs 1 \
      --run_name "${method_name}_order1_round${round}_${task}" \
      --max_source_length 512 \
      --max_target_length 50 \
      --generation_max_length 50 \
      --add_task_name true \
      --add_dataset_name true \
      --overwrite_output_dir \
      --overwrite_cache \
      --lr_scheduler_type constant \
      --warmup_steps 0 \
      --logging_strategy steps \
      --logging_steps 10 \
      --report_to none \
      --evaluation_strategy no \
      --save_strategy no \
      --save_steps 1500 \
      --ddp_find_unused_parameters false \
      "${extra_args[@]}"
}

model_path=${base_model}
previous_adapter=""
for ((index = 0; index < max_stages; index++)); do
  round=$((index + 1))
  task=${tasks[index]}
  stage_dir=${round}-${task}
  train_round \
    "${round}" \
    "${model_path}" \
    "${task}" \
    "${task_configs[index]}" \
    "${stage_dir}"

  current_adapter=${run_root}/${stage_dir}/adapter
  verifier_args=(
    --current "${current_adapter}"
    --mode "${verifier_mode}"
    --output "${run_root}/${stage_dir}/adapter_chain_verification.json"
  )
  if [[ -n ${previous_adapter} ]]; then
    verifier_args+=(--previous "${previous_adapter}")
  fi
  "${python_executable}" \
    scripts_order1_llama1B/verify_fishershape_adapter_chain.py \
    "${verifier_args[@]}"

  previous_adapter=${current_adapter}
  model_path=${current_adapter}
  if (( round < max_stages )); then
    sleep "${sleep_seconds}"
  fi
done

echo "${method_name} run completed: ${run_root}"
