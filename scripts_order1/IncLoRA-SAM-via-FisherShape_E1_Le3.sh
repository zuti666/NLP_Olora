#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# The copied conda installation still contains /data/140-0 shebangs, so invoke
# the known-good environment Python directly.  Other machines can override it
# with FISHERSHAPE_PYTHON=python (already-active environment) or another path.
python_executable=${FISHERSHAPE_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}
if ! command -v "${python_executable}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_executable}" >&2
  echo "Set FISHERSHAPE_PYTHON to an environment containing this project's dependencies." >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cache_root=${FISHERSHAPE_CACHE_ROOT:-${repo_root}/.cache/fishershape}
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

# Usage:
#   bash scripts_order1/IncLoRA-SAM-via-FisherShape_E1_Le3.sh [gpu_ids] [seed]
#
# Full run defaults match scripts_order1/IncLoRA-SAM_E1_Le3.sh:
# T5-large, order 1, rank 8, rho .05, batch/device 16, GA 1, one epoch.
# Useful smoke overrides:
#   FISHERSHAPE_MAX_STAGES=1 FISHERSHAPE_MAX_STEPS=1 \
#   FISHERSHAPE_MAX_TRAIN_SAMPLES=64 bash ... 0 1993

gpu_ids=${1:-0,1,2,3}
seed=${2:-1993}
train_bs=${FISHERSHAPE_PER_DEVICE_TRAIN_BATCH_SIZE:-16}
eval_bs=${FISHERSHAPE_PER_DEVICE_EVAL_BATCH_SIZE:-16}
base_model=${FISHERSHAPE_BASE_MODEL:-/data/140-1/users/liying/Sharing/Models/google-t5/t5-large}
output_base=${FISHERSHAPE_OUTPUT_ROOT:-/data/140-1/users/liying/O-LoRA/Logs_Short_Paper/Order-1_8}
run_root=${output_base}/IncLoRA-SAM-via-FisherShape-T5large_AdamW_Lora8_Ba${train_bs}-GA1_E1_LRe3_rho005_lambda0_seed-${seed}
max_stages=${FISHERSHAPE_MAX_STAGES:-4}
sleep_seconds=${FISHERSHAPE_SLEEP_SECONDS:-5}
bf16=${FISHERSHAPE_BF16:-true}
fisher_max_batches=${FISHERSHAPE_FISHER_MAX_BATCHES:-1}
use_torchrun=${FISHERSHAPE_USE_TORCHRUN:-true}

if [[ ${base_model} == /* && ! -e ${base_model} ]]; then
  echo "Base model path not found: ${base_model}" >&2
  echo "Set FISHERSHAPE_BASE_MODEL to an existing model path or model ID." >&2
  exit 2
fi
if [[ ! ${max_stages} =~ ^[1-4]$ ]]; then
  echo "FISHERSHAPE_MAX_STAGES must be an integer from 1 to 4; got ${max_stages}." >&2
  exit 2
fi

IFS=',' read -ra _gpu_array <<< "${gpu_ids}"
num_procs=${#_gpu_array[@]}
if [[ ${num_procs} -lt 1 ]]; then
  echo "gpu_ids must contain at least one CUDA device." >&2
  exit 2
fi
if [[ ${use_torchrun} != true && ${use_torchrun} != false ]]; then
  echo "FISHERSHAPE_USE_TORCHRUN must be true or false; got ${use_torchrun}." >&2
  exit 2
fi
if [[ ${use_torchrun} == false && ${num_procs} -ne 1 ]]; then
  echo "FISHERSHAPE_USE_TORCHRUN=false requires exactly one gpu_id." >&2
  exit 2
fi

extra_train_args=()
if [[ -n ${FISHERSHAPE_MAX_STEPS:-} ]]; then
  extra_train_args+=(--max_steps "${FISHERSHAPE_MAX_STEPS}")
fi
if [[ -n ${FISHERSHAPE_MAX_TRAIN_SAMPLES:-} ]]; then
  extra_train_args+=(--max_train_samples "${FISHERSHAPE_MAX_TRAIN_SAMPLES}")
fi

tasks=(dbpedia amazon yahoo agnews)
task_configs=(
  configs/order1_configs/dbpedia
  configs/order1_configs/amazon
  configs/order1_configs/yahoo
  configs/order1_configs/agnews
)

# A non-existent explicit path prevents the EWCLoRA strategy from auto-loading
# the previous sidecar.  lambda=0 already makes its gradient zero; avoiding the
# load also removes needless regularization graph construction in this ablation.
disabled_fisher_input=${FISHERSHAPE_DISABLED_FISHER_INPUT:-${run_root}/no_fisher_input.pt}

train_round() {
  local round=$1
  local model_path=$2
  local task=$3
  local task_config=$4
  local stage_dir=$5
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

  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    "${launcher[@]}" \
      src/run_uie_lora.py \
      --seed "${seed}" --data_seed "${seed}" \
      --do_train true \
      --bf16 "${bf16}" \
      --train_method lora \
      --lora_strategy ewclora_fishershape \
      --lora_dim 8 \
      --optimizer_type fishershape-adamw_hf \
      --fs_rho 0.05 \
      --fs_alpha 0.0 \
      --fs_alpha_mode ratio \
      --fs_sam_equivalent true \
      --fs_adaptive false \
      --fs_xi 1e-12 \
      --fs_grad_reduce mean \
      --fs_replay_rng_state false \
      --fs_include_olora_regularization false \
      --ewclora_lambda 0.0 \
      --ewclora_gamma 1.0 \
      --ewclora_fisher_path "${disabled_fisher_input}" \
      --ewclora_fisher_max_batches "${fisher_max_batches}" \
      --ewclora_normalize_fisher false \
      --ewclora_normalize_accumulated_fisher false \
      --ewclora_fisher_damping_beta 0.0 \
      --olora_lamda_1 0.0 \
      --olora_lamda_2 0.0 \
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
      --output_dir "${run_root}/${stage_dir}" \
      --per_device_train_batch_size "${train_bs}" \
      --per_device_eval_batch_size "${eval_bs}" \
      --gradient_accumulation_steps 1 \
      --num_train_epochs 1 \
      --run_name "IncLoRA-SAM-via-FisherShape_order1_round${round}_${task}" \
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
      "${extra_train_args[@]}"
}

model_path=${base_model}
for ((index = 0; index < max_stages; index++)); do
  round=$((index + 1))
  task=${tasks[index]}
  stage_dir="${round}-${task}"
  train_round \
    "${round}" \
    "${model_path}" \
    "${task}" \
    "${task_configs[index]}" \
    "${stage_dir}"
  model_path="${run_root}/${stage_dir}/adapter"
  if (( round < max_stages )); then
    sleep "${sleep_seconds}"
  fi
done

echo "SAM-equivalence run completed: ${run_root}"
