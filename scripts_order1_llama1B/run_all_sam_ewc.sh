#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Usage:
#   bash scripts_order1_llama1B/run_all_sam_ewc.sh \
#     [seed] [gpu_ids] [lambda] [gamma] [methods_csv] [orders_csv]
# Example smoke:
#   SAM_EWC_LLAMA_MAX_STAGES=2 SAM_EWC_LLAMA_MAX_STEPS=2 \
#   SAM_EWC_LLAMA_MAX_TRAIN_SAMPLES=16 SAM_EWC_LLAMA_FISHER_MAX_BATCHES=1 \
#   SAM_EWC_LLAMA_DO_PREDICT=false SAM_EWC_LLAMA_GRADIENT_ACCUMULATION_STEPS=1 \
#   bash scripts_order1_llama1B/run_all_sam_ewc.sh \
#     1993 6 1.0 1.0 inclora_sam_ewc,seqlora_sam_ewc 1

seed=${1:-1993}
gpu_ids=${2:-6}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
methods_csv=${5:-inclora_sam_ewc,seqlora_sam_ewc}
orders_csv=${6:-1,2,3}

IFS=',' read -ra methods <<< "${methods_csv}"
IFS=',' read -ra orders <<< "${orders_csv}"
for method in "${methods[@]}"; do
  for task_order in "${orders[@]}"; do
    echo ">>> Running ${method}, order ${task_order}"
    bash scripts_llama/run_sam_ewc_llama.sh \
      "${method}" "${task_order}" "${gpu_ids}" "${seed}" \
      "${lambda_ewc}" "${gamma_ewc}"
  done
done
