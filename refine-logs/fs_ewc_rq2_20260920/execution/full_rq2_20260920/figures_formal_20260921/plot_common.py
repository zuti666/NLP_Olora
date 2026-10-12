"""Shared three-seed figure style and receipt checks; CPU only."""
from pathlib import Path
import hashlib
import json
import os
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import pandas as pd

METHODS = ('sgd', 'gam')
LABELS = {'sgd': 'EWCLoRA', 'gam': 'FS-EWCLoRA'}
COLORS = {'sgd': '#0072B2', 'gam': '#D55E00'}
MARKERS = {'sgd': 'o', 'gam': 'D'}
LINES = {'sgd': '-', 'gam': (0, (4, 2))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def setup_style():
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        font_manager.fontManager.addfont(Path('/home-local/liying/.fonts') / name)
    path = Path(font_manager.findfont('Times New Roman', fallback_to_default=False))
    assert font_manager.FontProperties(fname=path).get_name() == 'Times New Roman'
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.serif': ['Times New Roman'],
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold', 'mathtext.cal': 'Times New Roman:italic',
        'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman', 'text.usetex': False,
        'figure.dpi': 150, 'savefig.dpi': 300})
    return dict(path=str(path), sha256=sha(path), family='Times New Roman')


def audited_summary(seed, method, job):
    directory = BASE / f'formal_results/seed{seed}/{method}/{job}'
    summary_path, manifest_path = directory / 'summary.json', directory / 'manifest.json'
    data, meta = read(summary_path), read(manifest_path)
    assert data['status'] == 'complete'
    assert data['seed'] == meta['seed'] == seed and data['method'] == meta['method'] == method
    assert data['protocol_sha256'] == meta['protocol_sha256']
    assert canonical({k: v for k, v in meta.items() if k != 'protocol_sha256'}) == meta['protocol_sha256']
    assert meta['config']['training_seed'] == seed
    assert Path(meta['config']['data_root']).name == 'imagenet-a'
    assert meta['coordinate_numel'] == 14155776
    coverage = read(BASE / 'final_report/coverage.json')
    matching = [r for r in coverage['diagnostics'] if (r['seed'], r['method'], r['job']) == (seed, method, job)]
    assert len(matching) == 1 and matching[0]['sha256'] == sha(summary_path)
    return data, meta, [dict(path=str(p), sha256=sha(p)) for p in (summary_path, manifest_path)]


def mean_sd(frame, x, value):
    rows = []
    for (method, point), group in frame.groupby(['method', x], sort=True):
        assert sorted(group.seed.tolist()) == [0, 1, 2], (method, point)
        values = group.sort_values('seed')[value].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        rows.append(dict(method=method, **{x: point}, mean=values.mean(), sd=values.std(ddof=1), n=3))
    return pd.DataFrame(rows)


def draw_mean_sd(ax, frame, x, value, *, markers=True, marker_every=1, bands=True, errorbars=False):
    aggregate = mean_sd(frame, x, value)
    for method in METHODS:
        rows = aggregate.query('method == @method').sort_values(x)
        xx, yy, ss = rows[x].to_numpy(), rows['mean'].to_numpy(), rows.sd.to_numpy()
        if bands:
            ax.fill_between(xx, yy - ss, yy + ss, color=COLORS[method], alpha=.16, linewidth=0, zorder=1)
        ax.plot(xx, yy, color=COLORS[method], linestyle=LINES[method], linewidth=1.25,
            marker=MARKERS[method] if markers else None, markersize=3.1, markevery=marker_every,
            label=LABELS[method], zorder=3)
        if errorbars:
            ax.errorbar(xx, yy, yerr=ss, color=COLORS[method], linestyle='none', linewidth=.8,
                        capsize=2, capthick=.8, zorder=2)
    return aggregate


def limits_from_sd(ax, aggregate, include_zero=False):
    lo = float((aggregate['mean'] - aggregate.sd).min())
    hi = float((aggregate['mean'] + aggregate.sd).max())
    if include_zero:
        lo = min(lo, 0)
    pad = max((hi - lo) * .045, 1e-8)
    ax.set_ylim(lo - pad, hi + pad)
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo - pad, hi + pad)
    ax.set_yticks([x for x in ticks if lo - pad <= x <= hi + pad])
    return [lo - pad, hi + pad]


def internal_legend(ax, loc):
    return ax.legend(loc=loc, frameon=False, fontsize=8, handlelength=1.,
        handletextpad=.3, labelspacing=.25, borderpad=0, borderaxespad=.35)


def save_figure(fig, name, metadata):
    for ax in fig.axes:
        ax.tick_params(axis='both', length=3, pad=3)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= fig_box.x0 - 1 and box.x1 <= fig_box.x1 + 1, ('clipped-x', text.get_text())
        assert box.y0 >= fig_box.y0 - 1 and box.y1 <= fig_box.y1 + 1, ('clipped-y', text.get_text())
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            box, bounds = legend.get_window_extent(renderer), ax.get_window_extent(renderer)
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), 'legend outside axes'
        boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_visible()]
        assert all(not a.overlaps(b) for a,b in zip(boxes, boxes[1:])), 'overlapping x ticks'
    outputs = {}
    for suffix in ('pdf', 'png'):
        path = HERE / f'{name}.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white')
        outputs[suffix] = dict(path=str(path), sha256=sha(path))
    fonts = sorted(set(x.decode() for x in re.findall(rb'/(?:BaseFont|FontName)\s*/([^\s/>]+)', (HERE / f'{name}.pdf').read_bytes())))
    assert fonts and all('TimesNewRoman' in name for name in fonts), fonts
    metadata.update(status='passed', outputs=outputs, embedded_fonts=fonts,
                    figure_inches=list(fig.get_size_inches()),
                    statistics='3 training seeds; arithmetic mean and sample SD ddof=1; not SEM/CI',
                    no_test_neighborhood_damage_curve=True,
                    rendered_visual_review='pending root and independent inspection')
    (HERE / f'{name}_validation.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    plt.close(fig)
    return metadata
