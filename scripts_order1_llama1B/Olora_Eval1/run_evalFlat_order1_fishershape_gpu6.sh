#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
cd "${repo_root}"

# Independently reload and evaluate the adapters produced by
# EWCLoRA-FisherShape-SAMEquivalent_LLaMA2-1b.sh.  This follows the learned-only
# lower-triangular protocol of scripts_order1/Olora_Eval1, but runs serially on
# one GPU and uses a Llama-safe evaluation batch size.
#
# Full reload evaluation (2 methods x 10 stage/task cells):
#   bash scripts_order1_llama1B/Olora_Eval1/run_evalFlat_order1_fishershape_gpu6.sh 6 42
#
# Fast final-adapter smoke (2 methods x one task x 8 examples):
#   FISHERSHAPE_EVAL_STAGES=4 \
#   FISHERSHAPE_EVAL_TASKS=agnews \
#   FISHERSHAPE_EVAL_MAX_PREDICT_SAMPLES=8 \
#   bash scripts_order1_llama1B/Olora_Eval1/run_evalFlat_order1_fishershape_gpu6.sh 6 42
#
# Useful environment controls:
#   FISHERSHAPE_EVAL_METHODS=both|sam|sam_ewc
#   FISHERSHAPE_EVAL_STAGES=1,2,3,4
#   FISHERSHAPE_EVAL_TASKS=all|dbpedia,amazon,yahoo,agnews
#   FISHERSHAPE_EVAL_FORCE=true|false
#   FISHERSHAPE_EVAL_DRY_RUN=true|false

gpu_id=${1:-${FISHERSHAPE_EVAL_GPU_ID:-6}}
seed=${2:-${FISHERSHAPE_EVAL_SEED:-42}}
python_executable=${FISHERSHAPE_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}

if [[ ! -x ${python_executable} ]]; then
  echo "Python executable not found or not executable: ${python_executable}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to validate adapters and summarize metrics." >&2
  exit 2
fi
if [[ ! ${gpu_id} =~ ^[0-9]+$ ]]; then
  echo "GPU id must be a non-negative integer; got '${gpu_id}'." >&2
  exit 2
fi
if [[ ! ${seed} =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer; got '${seed}'." >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
cache_root=${FISHERSHAPE_EVAL_CACHE_ROOT:-${repo_root}/.cache/fishershape_llama1b_eval}
export HF_HOME=${FISHERSHAPE_HF_HOME:-${cache_root}/huggingface}
export HF_MODULES_CACHE=${FISHERSHAPE_HF_MODULES_CACHE:-${HF_HOME}/modules}
export HF_DATASETS_CACHE=${FISHERSHAPE_HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${FISHERSHAPE_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export UIE_DATA_CACHE_ROOT=${FISHERSHAPE_UIE_DATA_CACHE_ROOT:-${cache_root}/uie_data}

training_root=${FISHERSHAPE_EVAL_TRAINING_ROOT:-${repo_root}/logs_and_outputs_llama_1B/full_gpu6_20260811_seed42/order_1}
eval_root=${FISHERSHAPE_EVAL_OUTPUT_ROOT:-${repo_root}/logs_and_outputs_llama_1B/reload_eval_gpu6_20260811_seed42/order_1}

sam_model_root=${FISHERSHAPE_EVAL_SAM_ROOT:-${training_root}/IncLoRA-SAM-via-FisherShape_Llama-3.2-1B_Lora8_Ba1-GA64_E1_LR1e-3_to_1e-4_rho0.05_lambda0.0_gamma1.0_normfalse_damp0.0_seed-${seed}}
sam_ewc_model_root=${FISHERSHAPE_EVAL_SAM_EWC_ROOT:-${training_root}/EWCLoRA-SAM-via-FisherShape_Llama-3.2-1B_Lora8_Ba1-GA64_E1_LR1e-3_to_1e-4_rho0.05_lambda1.0_gamma1.0_normfalse_damp0.0_seed-${seed}}

method_selector=${FISHERSHAPE_EVAL_METHODS:-both}
stage_selector=${FISHERSHAPE_EVAL_STAGES:-1,2,3,4}
task_selector=${FISHERSHAPE_EVAL_TASKS:-all}
eval_batch_size=${FISHERSHAPE_EVAL_BATCH_SIZE:-4}
max_predict_samples=${FISHERSHAPE_EVAL_MAX_PREDICT_SAMPLES:-}
bf16=${FISHERSHAPE_EVAL_BF16:-true}
force=${FISHERSHAPE_EVAL_FORCE:-false}
dry_run=${FISHERSHAPE_EVAL_DRY_RUN:-false}

case "${method_selector}" in
  both)
    methods=(sam sam_ewc)
    ;;
  sam|sam_ewc)
    methods=("${method_selector}")
    ;;
  *)
    echo "FISHERSHAPE_EVAL_METHODS must be both, sam, or sam_ewc." >&2
    exit 2
    ;;
esac

if [[ ! ${eval_batch_size} =~ ^[1-9][0-9]*$ ]]; then
  echo "FISHERSHAPE_EVAL_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ -n ${max_predict_samples} && ! ${max_predict_samples} =~ ^[1-9][0-9]*$ ]]; then
  echo "FISHERSHAPE_EVAL_MAX_PREDICT_SAMPLES must be a positive integer when set." >&2
  exit 2
fi
for boolean_value in "${bf16}" "${force}" "${dry_run}"; do
  if [[ ${boolean_value} != true && ${boolean_value} != false ]]; then
    echo "Boolean controls must be true or false; got '${boolean_value}'." >&2
    exit 2
  fi
done
if [[ ${dry_run} == false ]]; then
  mkdir -p \
    "${HF_MODULES_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${UIE_DATA_CACHE_ROOT}"
fi

IFS=',' read -ra stages <<< "${stage_selector}"
if (( ${#stages[@]} == 0 )); then
  echo "FISHERSHAPE_EVAL_STAGES must select at least one stage." >&2
  exit 2
fi
for stage in "${stages[@]}"; do
  if [[ ! ${stage} =~ ^[1-4]$ ]]; then
    echo "FISHERSHAPE_EVAL_STAGES entries must be in 1..4; got '${stage}'." >&2
    exit 2
  fi
done

task_is_selected() {
  local requested_task=$1
  local candidate
  if [[ ${task_selector} == all ]]; then
    return 0
  fi
  IFS=',' read -ra selected_tasks <<< "${task_selector}"
  for candidate in "${selected_tasks[@]}"; do
    if [[ ${candidate} == "${requested_task}" ]]; then
      return 0
    fi
  done
  return 1
}

stage_name() {
  case "$1" in
    1) printf '%s\n' '1-dbpedia' ;;
    2) printf '%s\n' '2-amazon' ;;
    3) printf '%s\n' '3-yahoo' ;;
    4) printf '%s\n' '4-agnews' ;;
  esac
}

learned_tasks_for_stage() {
  case "$1" in
    1) printf '%s\n' dbpedia ;;
    2) printf '%s\n' dbpedia amazon ;;
    3) printf '%s\n' dbpedia amazon yahoo ;;
    4) printf '%s\n' dbpedia amazon yahoo agnews ;;
  esac
}

model_root_for_method() {
  case "$1" in
    sam) printf '%s\n' "${sam_model_root}" ;;
    sam_ewc) printf '%s\n' "${sam_ewc_model_root}" ;;
  esac
}

lambda_for_method() {
  case "$1" in
    sam) printf '%s\n' '0.0' ;;
    sam_ewc) printf '%s\n' '1.0' ;;
  esac
}

if [[ -n ${max_predict_samples} ]]; then
  evaluation_scope=sample-${max_predict_samples}
else
  evaluation_scope=full
fi
# Never let a small smoke result make a later full evaluation look complete.
run_output_root=${eval_root}/${evaluation_scope}
summary_file=${run_output_root}/summary.tsv
if [[ ${dry_run} == false ]]; then
  mkdir -p "${run_output_root}"
  printf 'method\tstage\ttask\tr_sum\treload_samples\treload_exact_match\ttraining_exact_match\tdelta\tresult_file\n' > "${summary_file}"
fi

append_summary() {
  local method=$1
  local stage_dir=$2
  local task=$3
  local r_sum=$4
  local result_file=$5
  local reference_file=$6
  local reload_samples reload_exact reference_exact delta

  reload_samples=$(jq -r '.predict_samples // "NA"' "${result_file}")
  reload_exact=$(jq -r '.predict_exact_match // "NA"' "${result_file}")
  reference_exact=NA
  if [[ -f ${reference_file} ]]; then
    reference_exact=$(jq -r --arg task "${task}" '. ["predict_exact_match_for_" + $task] // "NA"' "${reference_file}")
  fi
  delta=NA
  if [[ -z ${max_predict_samples} && ${reload_exact} != NA && ${reference_exact} != NA ]]; then
    delta=$(awk -v reload="${reload_exact}" -v reference="${reference_exact}" 'BEGIN {printf "%.4f", reload-reference}')
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${method}" "${stage_dir}" "${task}" "${r_sum}" \
    "${reload_samples}" "${reload_exact}" "${reference_exact}" "${delta}" \
    "${result_file}" >> "${summary_file}"
}

run_eval() {
  local method=$1
  local model_root=$2
  local lambda_ewc=$3
  local stage=$4
  local stage_dir=$5
  local task=$6
  local adapter_dir=${model_root}/${stage_dir}/adapter
  local adapter_config=${adapter_dir}/adapter_config.json
  local chain_report=${adapter_dir}/chain_verification.json
  local output_dir=${run_output_root}/${method}/${stage_dir}/test-${task}
  local result_file=${output_dir}/predict_results.json
  local reference_file=${model_root}/${stage_dir}/predict_results.json
  local expected_r_sum=$((stage * 8))
  local r_sum

  if [[ ! -f ${adapter_config} ]]; then
    echo "Missing adapter config: ${adapter_config}" >&2
    return 1
  fi
  if [[ ! -f ${adapter_dir}/adapter_model.bin && ! -f ${adapter_dir}/adapter_model.safetensors ]]; then
    echo "Missing adapter weights in ${adapter_dir}." >&2
    return 1
  fi
  r_sum=$(jq -r '.r_sum // 0' "${adapter_config}")
  if [[ ${r_sum} != "${expected_r_sum}" ]]; then
    echo "Unexpected r_sum=${r_sum} for ${adapter_dir}; expected ${expected_r_sum}." >&2
    return 1
  fi
  if [[ -f ${chain_report} && $(jq -r '.status // "missing"' "${chain_report}") != pass ]]; then
    echo "Adapter chain verification did not pass: ${chain_report}" >&2
    return 1
  fi

  if [[ ${dry_run} == false && -f ${result_file} && ${force} == false ]]; then
    echo ">>> skip existing: method=${method} stage=${stage_dir} task=${task}"
    append_summary "${method}" "${stage_dir}" "${task}" "${r_sum}" "${result_file}" "${reference_file}"
    return
  fi

  local -a extra_args=()
  if [[ -n ${max_predict_samples} ]]; then
    extra_args+=(--max_predict_samples "${max_predict_samples}")
  fi

  # Prediction only needs the serialized adapter.  Point EWCLoRA at a missing
  # sidecar path so evaluation does not load the large Fisher tensors; Fisher
  # affects training gradients, not the inference forward pass.
  local disabled_fisher_input=${run_output_root}/no_fisher_input.pt
  local -a command=(
    "${python_executable}"
    src/run_uie_lora.py
    --seed "${seed}"
    --data_seed "${seed}"
    --do_predict
    --predict_with_generate
    --bf16 "${bf16}"
    --train_method lora
    --lora_strategy ewclora_fishershape
    --lora_dim 8
    --optimizer_type fishershape-adamw_hf
    --fs_sam_equivalent true
    --fs_rho 0.05
    --fs_replay_rng_state false
    --fs_include_olora_regularization false
    --ewclora_lambda "${lambda_ewc}"
    --ewclora_gamma 1.0
    --ewclora_normalize_fisher false
    --ewclora_normalize_accumulated_fisher false
    --ewclora_fisher_damping_beta 0.0
    --ewclora_fisher_path "${disabled_fisher_input}"
    --model_name_or_path "${adapter_dir}"
    --data_dir CL_Benchmark
    --task_config_dir "configs/lineraProbe_order1_configs/${task}"
    --instruction_file configs/instruction_config.json
    --instruction_strategy single
    --output_dir "${output_dir}"
    --per_device_eval_batch_size "${eval_batch_size}"
    --gradient_accumulation_steps 1
    --num_train_epochs 0
    --run_name "order1_reload_eval_${method}_${stage_dir}_${task}"
    --max_source_length 512
    --max_target_length 50
    --generation_max_length 50
    --add_task_name true
    --add_dataset_name true
    --overwrite_output_dir
    --lr_scheduler_type constant
    --warmup_steps 0
    --logging_strategy steps
    --logging_steps 10
    --report_to none
    --evaluation_strategy no
    --save_strategy no
    "${extra_args[@]}"
  )

  echo ">>> reload-eval: method=${method} stage=${stage_dir} task=${task} gpu=${gpu_id} r_sum=${r_sum}"
  if [[ ${dry_run} == true ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q' "${gpu_id}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}" > "${output_dir}/eval.log" 2>&1
  if [[ ! -f ${result_file} ]]; then
    echo "Evaluation did not produce ${result_file}; see ${output_dir}/eval.log" >&2
    return 1
  fi
  append_summary "${method}" "${stage_dir}" "${task}" "${r_sum}" "${result_file}" "${reference_file}"
  echo ">>> completed: $(jq -c '{samples:.predict_samples, exact_match:.predict_exact_match}' "${result_file}")"
}

echo "=== FisherShape adapter reload evaluation ==="
echo "GPU=${gpu_id} seed=${seed} methods=${method_selector} stages=${stage_selector} tasks=${task_selector}"
echo "Training root: ${training_root}"
echo "Evaluation root: ${run_output_root}"

selected_cell_count=0
for method in "${methods[@]}"; do
  model_root=$(model_root_for_method "${method}")
  lambda_ewc=$(lambda_for_method "${method}")
  if [[ ! -d ${model_root} ]]; then
    echo "Missing method model root: ${model_root}" >&2
    exit 1
  fi
  for stage in "${stages[@]}"; do
    stage_dir=$(stage_name "${stage}")
    mapfile -t learned_tasks < <(learned_tasks_for_stage "${stage}")
    for task in "${learned_tasks[@]}"; do
      if ! task_is_selected "${task}"; then
        continue
      fi
      selected_cell_count=$((selected_cell_count + 1))
      run_eval "${method}" "${model_root}" "${lambda_ewc}" "${stage}" "${stage_dir}" "${task}"
    done
  done
done

if (( selected_cell_count == 0 )); then
  echo "No stage/task cells were selected." >&2
  exit 2
fi

if [[ ${dry_run} == true ]]; then
  echo "Dry run completed for ${selected_cell_count} stage/task cells."
else
  echo "Completed ${selected_cell_count} stage/task cells. Summary: ${summary_file}"
  column -t -s $'\t' "${summary_file}" 2>/dev/null || cat "${summary_file}"
fi
