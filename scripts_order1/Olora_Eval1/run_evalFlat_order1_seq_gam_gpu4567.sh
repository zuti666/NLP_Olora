#!/bin/bash
set -euo pipefail

seed=${1:-1993}

learned_tasks_for_model() {
    local model_name="$1"

    case "${model_name}" in
        1-dbpedia)
            printf '%s\n' "dbpedia"
            ;;
        2-amazon)
            printf '%s\n' "dbpedia" "amazon"
            ;;
        3-yahoo)
            printf '%s\n' "dbpedia" "amazon" "yahoo"
            ;;
        4-agnews)
            printf '%s\n' "dbpedia" "amazon" "yahoo" "agnews"
            ;;
        *)
            echo "ERROR: unknown model stage '${model_name}'" >&2
            return 1
            ;;
    esac
}

run_eval_scripts() {
    echo "=== SeqLoRA-GAM cfg1/cfg2 learned-only eval start at $(date) ==="

    script_path="/data/140-0/users/liying/O-LoRA/scripts_order1/Olora_Eval1/eval_Whole.sh"

    model_paths=(
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg1_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_Rho0p01_NRho0p05_B10p1_B20p5_B30p9_G0p03_seed-${seed}"
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_Rho0p02_NRho0p1_B10p2_B20p5_B30p8_G0p05_seed-${seed}"
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p10_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        # "Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_cfg2_T5large_AdamwHF_r8_bs16_e1_lr1e-3_Rho0p03_NRho0p02_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993"
        "Logs_Short_Paper/Order-1_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993"
    )

    models=("1-dbpedia" "2-amazon" "3-yahoo" "4-agnews")
    gpu_ids=(4 5 6 7)

    log_base="Logs_RedoOlora/Order1_seq_gam_cfg12/eval_logs_seed-${seed}"
    max_parallel=${#gpu_ids[@]}
    job_count=0

    for model_path in "${model_paths[@]}"; do
        model_dirname=$(basename "$model_path")

        for model_name in "${models[@]}"; do
            mapfile -t test_tasks < <(learned_tasks_for_model "${model_name}")

            for test_task in "${test_tasks[@]}"; do
                log_dir="${log_base}/${model_dirname}/model-${model_name}_testdata-${test_task}"
                mkdir -p "$log_dir"
                log_file="${log_dir}/eval_Whole_eval_new.log"

                gpu_id=${gpu_ids[$((job_count % max_parallel))]}
                echo "Launching eval: model_path=${model_dirname} | model=${model_name} | test=${test_task} | gpu=${gpu_id}"
                echo "Log: ${log_file}"

                bash "${script_path}" "${test_task}" "${model_path}" "${model_name}" "${gpu_id}" > "${log_file}" 2>&1 &

                job_count=$((job_count + 1))
                if [[ $((job_count % max_parallel)) -eq 0 ]]; then
                    echo ">>> Waiting for current batch of ${max_parallel} eval tasks to finish..."
                    wait
                fi
            done
        done
    done

    wait
    echo "=== SeqLoRA-GAM cfg1/cfg2 learned-only eval finished at $(date) ==="
}

run_eval_scripts
