#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/../scripts_llama/run_sam_ewc_llama.sh" \
  inclora_sam_ewc 3 "$@"
