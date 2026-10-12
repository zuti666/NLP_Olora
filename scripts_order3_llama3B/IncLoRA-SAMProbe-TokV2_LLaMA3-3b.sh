#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
exec "${G2_LLAMA_PYTHON:-python3}" "$REPO_ROOT/scripts_llama/run_two_point_probe_llama.py" --config "$REPO_ROOT/configs/two_point_probe_llama/inclora_probe_sam_llama3b_order3_tokv2_rho005.json" "$@"
