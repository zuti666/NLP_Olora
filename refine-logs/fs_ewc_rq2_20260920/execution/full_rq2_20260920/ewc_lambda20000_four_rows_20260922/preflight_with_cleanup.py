#!/usr/bin/env python3
"""Run the unchanged CUDA witness, then let its context/sample window expire."""
from pathlib import Path
import subprocess
import sys
import time

result = subprocess.run([sys.executable, str(Path(__file__).with_name('env_witness.py')), *sys.argv[1:]])
if result.returncode:
    raise SystemExit(result.returncode)
# NVML GPU utilization covers a recent sampling window, including our own
# already-exited witness. The queue retains its exclusive UUID lock throughout.
# The unchanged prepare_fisher entrypoint still rechecks memory, utilization,
# foreign compute processes, GPU UUID, and inherited exclusive lock afterward.
time.sleep(3)
