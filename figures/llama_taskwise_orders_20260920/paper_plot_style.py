from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/codex_llama_taskwise_mpl')
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FONT_ROOT = HERE.parent / 'lora_taskwise_20260918/aa_times/fonts'
for name in ['Times.TTF', 'Timesbd.TTF', 'Timesi.TTF', 'Timesbi.TTF']:
    if (FONT_ROOT/name).exists():
        font_manager.fontManager.addfont(str(FONT_ROOT/name))
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8.5, 'figure.dpi': 150, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'text.usetex': False, 'mathtext.fontset': 'stix',
})

# Okabe–Ito palette, held fixed across methods, orders, and models.
COLORS = {'dbpedia': '#0072B2', 'amazon': '#E69F00', 'yahoo': '#009E73', 'agnews': '#D55E00'}
LABELS = {'dbpedia': 'DBPedia', 'amazon': 'Amazon', 'yahoo': 'Yahoo', 'agnews': 'AGNews'}
METHODS = ['SeqLoRA', 'IncLoRA', 'OLoRA', 'TreeLoRA', 'EWCLoRA', 'FS-EWCLoRA']
STYLES = {
    'SeqLoRA': dict(linestyle='-', marker='o'),
    'IncLoRA': dict(linestyle=(0, (5, 2)), marker='s'),
    'OLoRA': dict(linestyle=':', marker='^'),
    'TreeLoRA': dict(linestyle='-.', marker='D'),
    'EWCLoRA': dict(linestyle=(0, (3, 1, 1, 1)), marker='P'),
    'FS-EWCLoRA': dict(linestyle=(0, (7, 1.5)), marker='*'),
}


def save_figure(fig, stem):
    for ext in ['pdf', 'svg', 'png']:
        path = HERE/f'{stem}.{ext}'
        fig.savefig(path, facecolor='white')
        print(path)
