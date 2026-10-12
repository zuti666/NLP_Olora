#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python}"
exec "$PYTHON" "$SCRIPT_DIR/launch.py" "$@"
