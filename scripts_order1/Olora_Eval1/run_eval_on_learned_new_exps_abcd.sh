#!/bin/bash
# Evaluate Exp A, B, C, D with learned-only tasks, producing a lower-triangular matrix.
set -euo pipefail

seed=${1:-1993}
eval_script="scripts_order1/Olora_Eval1/eval_on_learned.sh"
torchrun_bin=${TORCHRUN_BIN:-/data/140-0/users/liying/conda_envs/nlora_req/bin/torchrun}
export TORCHRUN_BIN="${torchrun_bin}"

learned_tasks_for_stage() {
    local stage_name="$1"
    local stage_idx=${stage_name%%-*}
    local all_tasks=("dbpedia" "amazon" "yahoo" "agnews")
    local learned=()
    local i

    for ((i = 0; i < stage_idx && i < ${#all_tasks[@]}; i++)); do
        learned+=("${all_tasks[$i]}")
    done

    printf '%s\n' "${learned[@]}"
}

stages=("1-dbpedia" "2-amazon" "3-yahoo" "4-agnews")

declare -A exp_cfg
exp_cfg["OLoRA-SAM_T5large_r8_bs16_e1_lr1e-3_rho0p10_seed-${seed}"]="olora"
exp_cfg["OLoRA-SAM_T5large_r8_bs16_e1_lr1e-3_rho0p02_seed-${seed}"]="olora"
exp_cfg["IncLoRA-RWP_lora_Gauss_std0p005_a0p8_b0p2_T5large_r8_bs16_e1_lr1e-3_seed-${seed}"]="Inclora"
exp_cfg["IncLoRA-SAM_T5large_r8_bs16_e1_lr1e-3_rho0p01_seed-${seed}"]="Inclora"

log_base="Logs_Short_Paper/Order-1_8/eval_on_learned_logs_abcd_seed-${seed}"
gpu_id=3
job_count=0
max_parallel=4

echo "=== Eval-on-learned ABCD start at $(date) ==="

for exp_dir in "${!exp_cfg[@]}"; do
    lora_strategy="${exp_cfg[$exp_dir]}"
    model_root="Logs_Short_Paper/Order-1_8/${exp_dir}"

    if [[ ! -d "${model_root}" ]]; then
        echo "SKIP: ${exp_dir} not found"
        continue
    fi

    for stage in "${stages[@]}"; do
        mapfile -t learned_tasks < <(learned_tasks_for_stage "${stage}")

        for task in "${learned_tasks[@]}"; do
            result_path="${model_root}/${stage}/eval_on_learned-${task}-trainedLoRAwhole/predict_results.json"
            if [[ -f "${result_path}" ]]; then
                echo "SKIP (already done): ${exp_dir}/${stage}/eval_on_learned-${task}"
                continue
            fi

            log_dir="${log_base}/${exp_dir}/stage-${stage}_task-${task}"
            mkdir -p "${log_dir}"
            log_file="${log_dir}/eval.log"

            echo "Launching: ${exp_dir} | ${stage} | ${task} | gpu=${gpu_id}"
            bash "${eval_script}" "${task}" "${model_root}" "${stage}" "${gpu_id}" "${lora_strategy}" \
                > "${log_file}" 2>&1 &

            job_count=$((job_count + 1))
            if [[ $((job_count % max_parallel)) -eq 0 ]]; then
                echo ">>> Waiting for batch of ${max_parallel}..."
                wait
            fi
        done
    done
done

wait
echo "=== Eval-on-learned ABCD finished at $(date) ==="
