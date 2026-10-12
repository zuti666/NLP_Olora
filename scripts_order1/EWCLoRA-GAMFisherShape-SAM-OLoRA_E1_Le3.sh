#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Single-variable extension of the completed SAM-equivalent baseline:
#   g_final = g_SAM(CE) + grad L_OLoRA(clean) + grad L_EWC(clean)
# This launcher keeps EWC off by default so the first comparison isolates the
# OLoRA gradient.  Set GAMFS_EWC_LAMBDA explicitly for the later joint run.
export GAMFS_EXPERIMENT=sam_olora
export GAMFS_OLORA_LAMBDA1=${GAMFS_OLORA_LAMBDA1:-0.5}
export GAMFS_OLORA_LAMBDA2=${GAMFS_OLORA_LAMBDA2:-0.0}
export GAMFS_EWC_LAMBDA=${GAMFS_EWC_LAMBDA:-0.0}
export GAMFS_EWC_GAMMA=${GAMFS_EWC_GAMMA:-1.0}

exec bash \
  "${script_dir}/EWCLoRA-GAMFisherShape-SAMEquivalent_E1_Le3.sh" \
  "$@"
