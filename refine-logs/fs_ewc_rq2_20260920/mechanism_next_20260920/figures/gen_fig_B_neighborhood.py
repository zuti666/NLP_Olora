"""Regenerate the current training-only Figure B from verified original data.

Canonical implementation lives in figures_redesign_20260920 and publishes both
requested PDF/PNG paths. Its original predecessor is archived in plot_revisions.
"""
from pathlib import Path
import subprocess
import sys


def main():
    script = Path(__file__).resolve().parent.parent / "figures_redesign_20260920/gen_fig_B_neighborhood_merged.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
