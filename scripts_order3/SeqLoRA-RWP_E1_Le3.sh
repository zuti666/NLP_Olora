#!/bin/bash
set -euo pipefail

exec bash "$(dirname "$0")/SeqLoRA_E5_Le4_RWPlora-Gaussian.sh" "$@"
