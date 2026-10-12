"""B2 only: compact spacing, larger titles, exact normalized-path symbol.

CPU presentation update. Read the accepted titled figure's CSVs; never modify
the source plots, measurements, or training/diagnostic records.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2-b2-compact-20260922-mpl')
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'figures_titled_20260922'
SIZE = (7.6, 1.65)
WSPACE = .38
TITLE_SIZE = 10.5
NAME = 'Figure_B2_compact'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load('b2_compact_common', BASE / 'figures_formal_20260921/plot_common.py')
    helpers = load('b2_compact_helpers', BASE / 'figures_unified_20260922/plot_unified.py')
    font = common.setup_style()
    common.plt.rcParams.update({'axes.labelsize': 8.2, 'xtick.labelsize': 7.,
                               'ytick.labelsize': 7., 'axes.labelpad': 2.,
                               'legend.fontsize': 6.2})
    preserved = {str(p): common.sha(p) for p in SOURCE.iterdir() if p.is_file()}
    paths = [SOURCE / f'B2_{suffix}.csv' for suffix in
             ('trajectory_seed_data', 'path_seed_data', 'mean_sd')]
    trajectory, path_data, old_means = [common.pd.read_csv(p) for p in paths]
    assert len(trajectory) == 60 and len(path_data) == 54
    assert set(trajectory['T']) == set(range(1, 11))
    assert set(path_data['T']) == set(range(2, 11))
    assert (path_data['T'] == path_data.task + 1).all()
    assert (path_data.drift > 0).all()
    common.np.testing.assert_allclose(path_data.S_path,
                                    path_data.integral / path_data.drift ** 2,
                                    rtol=1e-12, atol=1e-14)
    specifications = [
        (trajectory, 'test_accuracy_percent', 'Accuracy', 'Accuracy of Task 0 (%)',
         (75, 95), [75, 80, 85, 90, 95], 'lower left'),
        (trajectory, 'signed_test_damage', 'Loss change', 'Loss change of Task 0',
         (-.05, .3), [-.05, 0, .1, .2, .3], 'upper left'),
        (trajectory, 'weight_drift', 'Weight drift', r'Weight drift $\delta_t$',
         (0, 1.6), [0, .4, .8, 1.2, 1.6], 'lower right'),
        (path_data, 'S_path', 'Average Fisher',
         r'$\overline{S}_{\mathrm{path},t}=I_t/\delta_t^2$',
         (0, .3), [0, .1, .2, .3], 'upper left'),
    ]
    fig, axes = common.plt.subplots(1, 4, figsize=SIZE)
    fig.subplots_adjust(left=.067, right=.991, bottom=.26, top=.855, wspace=WSPACE)
    tables = []
    for ax, (frame, metric, title, label, limits, ticks, loc) in zip(axes, specifications):
        values = common.mean_sd(frame, 'T', metric)
        values['metric'] = metric
        tables.append(values)
        for method in common.METHODS:
            curve = values[values.method == method].sort_values('T')
            xx, yy = curve['T'].to_numpy(), curve['mean'].to_numpy()
            line, = ax.plot(xx, yy, color=common.COLORS[method],
                            linestyle=common.LINES[method], linewidth=1.15,
                            marker=common.MARKERS[method], markersize=1.6,
                            markeredgewidth=.35, label=common.LABELS[method], zorder=3)
            common.np.testing.assert_array_equal(line.get_xdata(), xx)
            common.np.testing.assert_array_equal(line.get_ydata(), yy)
            assert yy.min() >= limits[0] and yy.max() <= limits[1]
        helpers.axis(ax, label, 'Number of Tasks', (.7, 10.3), [1, 2, 4, 6, 8, 10], limits, ticks)
        helpers.legend(ax, loc)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal', pad=3.)
        assert not ax.collections, 'No SD shading requested'
    means = common.pd.concat(tables, ignore_index=True)
    columns = ['method', 'T', 'metric', 'mean', 'sd', 'n']
    sort = ['metric', 'method', 'T']
    common.pd.testing.assert_frame_equal(
        means[columns].sort_values(sort).reset_index(drop=True),
        old_means[columns].sort_values(sort).reset_index(drop=True),
        check_dtype=False, rtol=1e-12, atol=1e-14)
    assert len(means) == 78
    common.HERE = HERE
    metadata = dict(
        source_files=[dict(path=str(p), sha256=common.sha(p)) for p in paths],
        script=dict(path=str(Path(__file__)), sha256=common.sha(Path(__file__))),
        font=font, figure_inches=list(SIZE), title_fontsize=TITLE_SIZE,
        previous_title_fontsize=8.6, wspace=WSPACE, previous_wspace=.56,
        axes=[dict(title=a.get_title(), ylabel=a.get_ylabel(), ylim=list(a.get_ylim()),
                   xlabel=a.get_xlabel(), xlim=list(a.get_xlim()),
                   position=list(a.get_position().bounds)) for a in axes],
        formula='S_path(seed,t) = integral(seed,t) / drift(seed,t)^2; plotted value = mean over seeds',
        delta_definition='delta_t = ||Delta_t||_2; Delta_t = W_t - W_0 in effective Q/V coordinates',
        task_axis='T=t+1; S_path undefined at T=1',
        path_legend='upper left', all_78_mean_sd_groups_unchanged=True,
        ratio_checked_for_all_54_paths=True, uncertainty_bands_drawn=False,
        no_new_model_evaluations=True,
    )
    result = common.save_figure(fig, NAME, metadata)
    result['statistics'] = '3 training seeds; arithmetic means shown; sample SD retained in source CSV, not drawn'
    result['source_files_unchanged'] = all(common.sha(p) == digest for p, digest in preserved.items())
    assert result['source_files_unchanged']
    (HERE / f'{NAME}_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['outputs'], indent=2))


if __name__ == '__main__':
    main()
