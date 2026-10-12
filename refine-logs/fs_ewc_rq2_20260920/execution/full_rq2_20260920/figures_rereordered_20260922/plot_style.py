"""Shared compact presentation for the reordered RQ2 figures; CPU only."""
from pathlib import Path
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2-rereordered-20260922-mpl')
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'figures_titled_20260922'
SIZE = (7.6, 1.65)
WSPACE = .38
TITLE_SIZE = 10.5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = load('rq2_rereordered_base_style', BASE / 'figures_formal_20260921/plot_common.py')
helpers = load('rq2_rereordered_helpers', BASE / 'figures_unified_20260922/plot_unified.py')


def setup():
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize': 8.2, 'xtick.labelsize': 7.,
                           'ytick.labelsize': 7., 'axes.labelpad': 2.,
                           'legend.fontsize': 6.2})
    return font


def canvas():
    fig, axes = pc.plt.subplots(1, 4, figsize=SIZE)
    # Two-line descriptive titles share a reserved top strip in the same canvas.
    fig.subplots_adjust(left=.067, right=.991, bottom=.26, top=.765, wspace=WSPACE)
    return fig, axes


def draw(ax, values, x, y, *, markevery=1):
    for method in pc.METHODS:
        rows = values[values.method == method].sort_values(x)
        xx, yy = rows[x].to_numpy(), rows[y].to_numpy()
        assert pc.np.isfinite(xx).all() and pc.np.isfinite(yy).all()
        line, = ax.plot(xx, yy, color=pc.COLORS[method], linestyle=pc.LINES[method],
                        linewidth=1.15, marker=pc.MARKERS[method], markersize=1.6,
                        markeredgewidth=.35, markevery=markevery,
                        label=pc.LABELS[method], zorder=3)
        pc.np.testing.assert_array_equal(line.get_xdata(), xx)
        pc.np.testing.assert_array_equal(line.get_ydata(), yy)


def label(ax, title, ylabel, xlabel, xlim, xticks, ylim, yticks, legend):
    helpers.axis(ax, ylabel, xlabel, xlim, xticks, ylim, yticks)
    helpers.legend(ax, legend)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal', pad=3., linespacing=1.0)


def snapshot():
    return {str(p): pc.sha(p) for p in SOURCE.iterdir() if p.is_file()}


def finish(fig, name, metadata, prior, font, script):
    assert all(not ax.collections for ax in fig.axes)
    for ax in fig.axes:
        for line in ax.lines:
            assert min(line.get_ydata()) >= ax.get_ylim()[0]
            assert max(line.get_ydata()) <= ax.get_ylim()[1]
    metadata.update(font=font, figure_inches=list(SIZE), wspace=WSPACE,
                    title_fontsize=TITLE_SIZE, no_new_model_evaluations=True,
                    uncertainty_bands_drawn=False,
                    scripts=[{'path': str(p), 'sha256': pc.sha(p)}
                             for p in [Path(script), Path(__file__)]],
                    axes=[dict(title=a.get_title(), ylabel=a.get_ylabel(), xlabel=a.get_xlabel(),
                               xlim=list(a.get_xlim()), ylim=list(a.get_ylim()),
                               position=list(a.get_position().bounds)) for a in fig.axes])
    pc.HERE = HERE
    result = pc.save_figure(fig, name, metadata)
    result['statistics'] = result['statistics_scope']
    result['source_files_unchanged'] = all(pc.sha(path) == digest for path, digest in prior.items())
    assert result['source_files_unchanged']
    (HERE / f'{name}_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['outputs'], indent=2))
    return result
