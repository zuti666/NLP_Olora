#!/bin/bash
set -euo pipefail

seed=${1:-1993}

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

run_eval_scripts() {
    echo "=== SeqLoRA-GAM eval_on_learned start at $(date) ==="

    script_path="/data/140-0/users/liying/O-LoRA/scripts_order1/Olora_Eval1/eval_on_learned.sh"

    model_paths=(
        "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p10_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p03_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993"
    )

    stages=("1-dbpedia" "2-amazon" "3-yahoo" "4-agnews")
    gpu_ids=(0 1 2 3)

    log_base="Logs_RedoOlora/Order1_seq_gam_cfg12/eval_on_learned_logs_seed-${seed}"
    max_parallel=${#gpu_ids[@]}
    job_count=0

    for model_path in "${model_paths[@]}"; do
        model_dirname=$(basename "$model_path")

        for stage in "${stages[@]}"; do
            mapfile -t learned_tasks < <(learned_tasks_for_stage "${stage}")

            for task in "${learned_tasks[@]}"; do
                log_dir="${log_base}/${model_dirname}/stage-${stage}_task-${task}"
                mkdir -p "${log_dir}"
                log_file="${log_dir}/eval_on_learned.log"

                gpu_id=${gpu_ids[$((job_count % max_parallel))]}
                echo "Launching eval_on_learned: model_path=${model_dirname} | stage=${stage} | task=${task} | gpu=${gpu_id}"
                echo "Log: ${log_file}"

                bash "${script_path}" "${task}" "${model_path}" "${stage}" "${gpu_id}" > "${log_file}" 2>&1 &

                job_count=$((job_count + 1))
                if [[ $((job_count % max_parallel)) -eq 0 ]]; then
                    echo ">>> Waiting for current batch of ${max_parallel} eval tasks to finish..."
                    wait
                fi
            done
        done
    done

    wait
    echo "=== SeqLoRA-GAM eval_on_learned finished at $(date) ==="
}

run_eval_scripts
