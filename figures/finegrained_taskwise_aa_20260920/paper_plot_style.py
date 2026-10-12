"""Times/line styles matched to the requested lora_taskwise_aa.pdf reference."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/finegrained_taskwise_aa_mpl')

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager

METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']
STYLES = {
    'Vanilla LoRA': {'color': '#777777', 'marker': 'o'},
    'InfLoRA': {'color': '#0072B2', 'marker': 's'},
    'OLoRA': {'color': '#56B4E9', 'marker': 'D'},
    'SDLoRA': {'color': '#D55E00', 'marker': '^'},
    'CL-LoRA': {'color': '#7B3294', 'marker': 'v'},
    'EWCLoRA': {'color': '#009E73', 'marker': 'P'},
    'GAM-EWCLoRA': {'color': '#CC3377', 'marker': '*'},
}
DATASETS = {
    'aircraft': ('Aircraft', 'aircraft'),
    'cars196': ('Cars', 'cars'),
    'cub200': ('CUB-200', 'cub200'),
    'flowers': ('Flowers', 'flowers'),
    'het5': ('5Datasets', '5datasets'),
}


def configure():
    font_dir = HERE / 'fonts'
    for path in font_dir.glob('*.TTF'):
        font_manager.fontManager.addfont(path)
    font = font_manager.findfont(
        font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    matplotlib.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 11,
        'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 8.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    })
    return font
