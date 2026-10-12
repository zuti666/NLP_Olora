"""Shared CPU-only plotting style and immutable-source provenance helpers."""
from pathlib import Path
import hashlib
import json
import os

FIG_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR / ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

EXECUTION = FIG_DIR.parent.parent / "execution"
REPORT = EXECUTION / "report_seed0"
RAW = EXECUTION / "results_seed0_recovery1"
SOURCE_DIR = FIG_DIR / "source_data"
SOURCE_DIR.mkdir(exist_ok=True)
METHODS = ("sgd", "gam")
COLORS = {"sgd": "#0072B2", "gam": "#D55E00"}
MARKERS = {"sgd": "o", "gam": "D"}
LABELS = {"sgd": "SGD + EWC", "gam": "historical GAM + EWC"}
SPLIT_STYLES = {"train": "-", "test": (0, (4, 2))}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 140, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "text.usetex": False, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

def save_fig(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", dpi=300)
    plt.close(fig)

def panel_label(ax, letter):
    ax.text(-0.18, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=11, va="bottom", ha="left", clip_on=False)

def method_handles():
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m],
                   label=LABELS[m], markersize=4) for m in METHODS]

def radius_axis(ax, radii):
    ax.set_xscale("symlog", linthresh=0.5, linscale=0.65, base=10)
    ax.set_xticks(radii, [f"{r:g}" for r in radii])
    ax.set_xlabel(r"Effective-weight radius $r$")
    ax.minorticks_off()

def ce_axis(ax):
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.65, base=10)
    ticks = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    ax.set_yticks(ticks, ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    ax.minorticks_off()

def record_provenance(name, paths, checks):
    sources = [{"path": str(p.relative_to(EXECUTION)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in paths]
    (FIG_DIR / f"{name}_provenance.json").write_text(json.dumps({
        "sources_relative_to_execution": sources, "checks": checks,
        "training_seeds": [0], "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "output": "Vector PDF and 300 dpi PNG; numerical tables retain source precision",
    }, indent=2) + "\n")

def write_markdown_table(frame, path):
    def cell(x):
        return str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(map(cell, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    path.write_text("\n".join(lines) + "\n")
