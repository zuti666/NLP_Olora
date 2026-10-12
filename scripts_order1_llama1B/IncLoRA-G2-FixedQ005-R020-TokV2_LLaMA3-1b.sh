#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
exec "${G2_LLAMA_PYTHON:-python3}" "$REPO_ROOT/scripts_llama/run_g2_llama.py" --config "$REPO_ROOT/configs/g2_llama/corrected_v2/inclora_g2_llama1b_order1_tokv2_fixed_q005_r020.json" "$@"
