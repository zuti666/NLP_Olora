"""Run all figure and table generators from any current working directory."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for name in ("gen_fig_B_neighborhood.py", "gen_fig_C_sequence.py", "gen_tables_and_captions.py"):
    subprocess.run([sys.executable, str(root / name)], check=True)
    print(f"Generated from immutable sources: {name}", flush=True)
