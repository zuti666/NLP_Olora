#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
gpu_ids=${1:-0,1,2,3,4,5,6,7}
seed=${2:-1993}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
fs_alpha=${5:-0.2}
fs_rho=${6:-0.05}
normalize_fisher=${7:-true}
damping_beta=${8:-0.2}
include_olora=${9:-false}
exec bash "${repo_root}/scripts_order1_llama1B/EWCLoRA-FisherShape_LLaMA2-1b.sh" \
  "${gpu_ids}" "${seed}" "${lambda_ewc}" "${gamma_ewc}" \
  "${fs_alpha}" "${fs_rho}" "${normalize_fisher}" "${damping_beta}" \
  "${include_olora}" 3
