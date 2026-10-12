#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher=${script_dir}/EWCLoRA-FisherShape-SAMEquivalent_LLaMA2-1b.sh

# Usage:
#   bash scripts_order1_llama1B/run_compare_fishershape_sam_ewc.sh \
#     [gpu_ids] [seed] [ewc_lambda] [ewc_gamma]

gpu_ids=${1:-0,1,2,3,4,5,6,7}
seed=${2:-42}
lambda_ewc=${3:-1.0}
gamma_ewc=${4:-1.0}

echo ">>> Baseline: IncLoRA + SAM-equivalent FisherShape, EWC off"
bash "${launcher}" "${gpu_ids}" "${seed}" sam

echo ">>> Comparison: the same SAM-equivalent FisherShape plus EWC"
bash "${launcher}" "${gpu_ids}" "${seed}" sam_ewc "${lambda_ewc}" "${gamma_ewc}"

echo ">>> Both comparison runs completed."
