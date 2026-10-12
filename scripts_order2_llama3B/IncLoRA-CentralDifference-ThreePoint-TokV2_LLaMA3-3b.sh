#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
exec "${THREE_POINT_PROBE_PYTHON:-python3}" "$REPO_ROOT/scripts_llama/run_three_point_probe_llama.py" --config "$REPO_ROOT/configs/three_point_probe_llama/inclora_threepoint_central_llama3b_order2_tokv2_rho005.json" "$@"
