#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Usage:
#   bash scripts_order1_llama1B/run_all_fishershape_ewc.sh \
#     [seed] [gpu_ids] [lambda] [gamma] [fs_alpha] [fs_rho] \
#     [normalize_fisher] [damping_beta] [include_olora]

seed=${1:-1993}
gpu_ids=${2:-0,1,2,3,4,5,6,7}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
fs_alpha=${5:-0.2}
fs_rho=${6:-0.05}
normalize_fisher=${7:-true}
damping_beta=${8:-0.2}
include_olora=${9:-false}

scripts=(
  scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
  scripts_order2_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
  scripts_order3_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh
)

for script_path in "${scripts[@]}"; do
  echo ">>> Running ${script_path}"
  bash "${script_path}" \
    "${gpu_ids}" \
    "${seed}" \
    "${lambda_ewc}" \
    "${gamma_ewc}" \
    "${fs_alpha}" \
    "${fs_rho}" \
    "${normalize_fisher}" \
    "${damping_beta}" \
    "${include_olora}"
  echo ">>> Finished ${script_path}"
done
