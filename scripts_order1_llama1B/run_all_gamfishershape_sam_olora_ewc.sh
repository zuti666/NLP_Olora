#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Mirrors run_all_fishershape_ewc.sh, but runs the SAM-equivalent CE probes
# plus clean-point OLoRA and EWC gradients for all three task orders.
# Usage:
#   bash scripts_order1_llama1B/run_all_gamfishershape_sam_olora_ewc.sh \
#     [seed] [gpu_ids] [ewc_lambda] [ewc_gamma] [olora_lambda1] [olora_lambda2]

seed=${1:-1993}
gpu_ids=${2:-6}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}
olora_lambda1=${5:-0.5}
olora_lambda2=${6:-0.0}
requested_orders=${GAMFS_LLAMA_ORDERS:-1,2,3}

IFS=',' read -ra orders <<< "${requested_orders}"
for order in "${orders[@]}"; do
  case "${order}" in
    1) script_path=scripts_order1_llama1B/EWCLoRA-GAMFisherShape-SAM-OLoRA-EWC_LLaMA3-1b.sh ;;
    2) script_path=scripts_order2_llama1B/EWCLoRA-GAMFisherShape-SAM-OLoRA-EWC_LLaMA3-1b.sh ;;
    3) script_path=scripts_order3_llama1B/EWCLoRA-GAMFisherShape-SAM-OLoRA-EWC_LLaMA3-1b.sh ;;
    *) echo "GAMFS_LLAMA_ORDERS must contain only 1,2,3; got ${order}." >&2; exit 2 ;;
  esac
  echo ">>> Running Llama-3.2-1B Order ${order}: ${script_path}"
  bash "${script_path}" \
    "${gpu_ids}" "${seed}" "${lambda_ewc}" "${gamma_ewc}" \
    "${olora_lambda1}" "${olora_lambda2}"
  echo ">>> Finished Llama-3.2-1B Order ${order}"
done
