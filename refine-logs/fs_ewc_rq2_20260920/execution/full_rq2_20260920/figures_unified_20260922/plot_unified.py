"""CPU-only, shared compact style for measured B1 radius50 and complete B2 paths."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2-unified-20260922-mpl')
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
B1_ROOT = BASE / 'radius50_integer_seed0_halfdirs_20260921'
B1 = B1_ROOT / 'figures/replot_20260922_v3'
B2 = BASE / 'stage_path_completion_20260921/final_review_20260921'
SIZE = (7.6, 1.65)
LEGEND_SIZE = 6.2


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return dict(path=str(path), sha256=sha(path))


def canvas(pc):
    fig, axes = pc.plt.subplots(1, 4, figsize=SIZE)
    fig.subplots_adjust(left=.067, right=.991, bottom=.26, top=.935, wspace=.56)
    return fig, axes


def axis(ax, ylabel, xlabel, xlim, xticks, ylim, yticks):
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, xticks=xticks,
           ylim=ylim, yticks=yticks)
    ax.tick_params(length=2.5, pad=2)
    assert ylim[-1] == yticks[-1], 'Upper displayed limit must have a tick label'


def legend(ax, loc, *, anchor=None):
    args = dict(loc=loc, frameon=False, fontsize=LEGEND_SIZE, handlelength=.9,
                handletextpad=.24, labelspacing=.12, borderpad=0, borderaxespad=.3)
    if anchor is not None:
        args['bbox_to_anchor'] = anchor
    return ax.legend(**args)


def place_path_legend(fig, ax):
    """Find an internal gap without covering either mean curves or their SD bands."""
    for y in [.11,.10,.12,.09,.13,.08,.14,.07]:
        leg = legend(ax, 'lower right', anchor=(.999,y))
        fig.canvas.draw()
        box = leg.get_window_extent(fig.canvas.get_renderer())
        bounds = ax.get_window_extent()
        curves = [line.get_transform().transform_path(line.get_path()) for line in ax.lines]
        bands = [c.get_transform().transform_path(p) for c in ax.collections for p in c.get_paths()]
        fits = bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)
        if fits and not any(p.intersects_bbox(box, filled=False) for p in curves) and not any(p.intersects_bbox(box, filled=True) for p in bands):
            return dict(anchor=[.999,y], bounding_box_clear_of_mean_and_sd=True)
    raise AssertionError('Path legend would overlap measured curves/SD bands')


def draw(pc, ax, frame, x, y, *, sd=False, markevery=1):
    values = pc.mean_sd(frame, x, y) if sd else frame[['method', x, y]].rename(columns={y: 'mean'}).assign(sd=0.)
    for method in pc.METHODS:
        curve = values[values.method == method].sort_values(x)
        xx, yy, ss = curve[x].to_numpy(), curve['mean'].to_numpy(), curve.sd.to_numpy()
        if sd:
            ax.fill_between(xx, yy-ss, yy+ss, color=pc.COLORS[method], alpha=.16, linewidth=0, zorder=1)
        line, = ax.plot(xx, yy, color=pc.COLORS[method], linestyle=pc.LINES[method],
                       linewidth=1.15, marker=pc.MARKERS[method], markersize=1.6,
                       markeredgewidth=.35, markevery=markevery, label=pc.LABELS[method], zorder=3)
        pc.np.testing.assert_array_equal(line.get_xdata(), xx)
        pc.np.testing.assert_array_equal(line.get_ydata(), yy)
    return values


def save(pc, fig, name, metadata):
    # Standard source style checks text extents, legends inside axes, tick overlap,
    # vector font embedding and fixed output canvas size.
    pc.HERE = HERE
    metadata['axes'] = [dict(ylabel=a.get_ylabel(), ylim=list(a.get_ylim()),
                             yticks=[float(x) for x in a.get_yticks()], xlabel=a.get_xlabel(),
                             xlim=list(a.get_xlim()), xticks=[float(x) for x in a.get_xticks()],
                             legend_present=a.get_legend() is not None) for a in fig.axes]
    for ax in fig.axes:
        assert ax.get_legend() is not None
        for line in ax.lines:
            assert min(line.get_ydata()) >= ax.get_ylim()[0]
            assert max(line.get_ydata()) <= ax.get_ylim()[1]
    metadata.update(shared_style=dict(figure_inches=SIZE, font='Times New Roman',
                    colors=pc.COLORS, marker_size_points=1.6, line_width_points=1.15,
                    axis_label_fontsize=8.2, tick_fontsize=7.0, legend_fontsize=LEGEND_SIZE),
                    maximum_y_tick_equals_display_upper_limit=True,
                    no_new_model_evaluations=True, generation_code=record(Path(__file__)))
    result = pc.save_figure(fig, name, metadata)
    # The reused saver defaults to n=3. B1 is one seed and must not inherit that label.
    result['statistics'] = metadata['statistics_scope']
    (HERE / f'{name}_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def plot_b1(pc, font):
    receipt_path = B1_ROOT / 'report/radius50_complete.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['validated_jobs'] == 4
    for item in receipt['source_files'] + receipt['output_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    inputs = [B1 / 'dense_training.csv', B1 / 'fisher_training.csv', B1 / 'tolerance_bounds.csv']
    dense, fisher, old_tolerance = [pc.pd.read_csv(p) for p in inputs]
    native = B1_ROOT / 'figures/radius50_integer_seed0_dense50_fisher16'
    for frame, filename in [(dense, 'dense_training_per_seed.csv'), (fisher, 'fisher_training_per_seed.csv')]:
        pc.pd.testing.assert_frame_equal(frame, pc.pd.read_csv(native/filename))
    grids = ([i/4 for i in range(81)] + list(range(25,51)),
             [0,.5,1,2,5,10] + list(range(11,21)) + list(range(25,51)))
    for frame, expected in zip((dense, fisher), grids):
        assert set(frame.seed) == {0}
        for _, curve in frame.groupby('method'):
            assert sorted(curve.radius.tolist()) == expected
    previous = load('prior_b1_tolerance', B1/'plot_b1.py')
    tolerance = previous.measured_tolerances(pc, dense)
    pc.pd.testing.assert_frame_equal(tolerance, old_tolerance, check_dtype=False, rtol=1e-12)
    dense.to_csv(HERE/'B1_dense_seed0.csv', index=False)
    fisher.to_csv(HERE/'B1_fisher_seed0.csv', index=False)
    tolerance.to_csv(HERE/'B1_tolerance_seed0.csv', index=False)
    fig, axes = canvas(pc)
    specs = [
        (dense, 'G_train', r'Loss $\overline{[D_0]_+}$', (0,.8), [0,.2,.4,.6,.8], 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy of Task 0 (%)', (70,100), [70,80,90,100], 'lower left'),
        (fisher, 'raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', (0,25000), [0,5000,10000,15000,20000,25000], 'upper left'),
    ]
    tables = []
    for ax, (frame, metric, label, ylim, ticks, loc) in zip(axes, specs):
        marks = 1 if metric == 'raw_fisher_trace' else list(range(0,81,4))+list(range(81,107))
        values = draw(pc, ax, frame, 'radius', metric, markevery=marks)
        values['metric'] = metric
        tables.append(values)
        axis(ax, label, r'Weight drift $r$', (-1.25,51.25), [0,25,50], ylim, ticks)
        legend(ax, loc)
    values = draw(pc, axes[3], tolerance, 'epsilon', 'mean')
    axis(axes[3], r'Tolerance radius $\hat r_\epsilon$', r'Allowed damage $\epsilon$',
         (0,.104), [0,.025,.05,.075,.1], (0,30), [0,10,20,30])
    axes[3].set_xticklabels(['0','0.025','0.05','0.075','0.1'])
    legend(axes[3], 'lower right')
    pc.pd.concat(tables, ignore_index=True).to_csv(HERE/'B1_plotted_values.csv',index=False)
    assert sum(len(v) for v in tables) == 512 and len(tolerance) == 22
    return save(pc, fig, 'Figure_B1_radius50_unified', dict(font=font,
        source_receipt=record(receipt_path), sources=[record(p) for p in inputs],
        training_seeds=[0], statistics_scope='seed0 only; direction means, no cross-seed error band',
        dense_radii=grids[0], fisher_radii=grids[1], unmeasured_radii=[21,22,23,24],
        dense_examples=1259, dense_directions=50, fisher_examples=256, fisher_directions=16,
        positive_part_before_direction_mean=True, tolerance_recomputed_from_panel1=True,
        tolerance_thresholds=previous.EPSILONS, measured_curve_values=512, tolerance_values=22))


def plot_b2(pc, font):
    inputs = [B2/'figures'/f'B2_all_stages_{suffix}.csv' for suffix in
              ['trajectory_seed_data', 'path_seed_data', 'mean_sd']]
    validation = json.loads((B2/'FINAL_VALIDATION.json').read_text())
    for p in inputs:
        assert sha(p) == validation['files'][str(p.relative_to(B2))]['sha256']
    trajectory, paths, old_means = [pc.pd.read_csv(p) for p in inputs]
    assert len(trajectory) == 60 and len(paths) == 54
    assert set(trajectory['T']) == set(range(1,11)) and set(paths['T']) == set(range(2,11))
    fig, axes = canvas(pc)
    specs = [
        (trajectory, 'weight_drift', r'Weight drift $\delta$', (0,1.6), [0,.4,.8,1.2,1.6], 'lower right'),
        (trajectory, 'signed_test_damage', 'Loss change of Task 0', (-.05,.3), [-.05,0,.1,.2,.3], 'upper left'),
        (trajectory, 'test_accuracy_percent', 'Accuracy of Task 0 (%)', (75,95), [75,80,85,90,95], 'lower left'),
        (paths, 'S_path', 'Average Fisher', (0,.3), [0,.1,.2,.3], 'lower right'),
    ]
    tables = []
    for i, (ax, (frame, metric, label, ylim, ticks, loc)) in enumerate(zip(axes,specs)):
        values = draw(pc, ax, frame, 'T', metric, sd=True)
        assert (values['mean']-values.sd).min() >= ylim[0]
        assert (values['mean']+values.sd).max() <= ylim[1]
        values['metric'] = metric
        tables.append(values)
        axis(ax, label, 'Number of Tasks', (.7,10.3), [1,2,4,6,8,10], ylim, ticks)
        legend(ax, loc, anchor=(.99,.13) if i == 3 else None)
    path_legend = place_path_legend(fig, axes[3])
    means = pc.pd.concat(tables, ignore_index=True)
    cols = ['method','T','mean','sd','n','metric']
    pc.pd.testing.assert_frame_equal(means[cols].reset_index(drop=True), old_means[cols].reset_index(drop=True), check_dtype=False, rtol=1e-12)
    means.to_csv(HERE/'B2_mean_sd.csv', index=False)
    trajectory.to_csv(HERE/'B2_trajectory_seed_data.csv', index=False)
    paths.to_csv(HERE/'B2_path_seed_data.csv', index=False)
    return save(pc, fig, 'Figure_B2_stages_unified', dict(font=font,
        sources=[record(p) for p in inputs], training_seeds=[0,1,2],
        statistics_scope='three independent training seeds; mean and sample SD ddof=1, not SEM/CI',
        path_count=54, endpoints=60, T1_path_undefined=True, path_T=list(range(2,11)),
        path_legend=path_legend,
        all_existing_78_mean_sd_groups_unchanged=True, signed_test_damage_retained=True,
        last_panel_is_normalized_chord_directional_sensitivity=True))


def main():
    HERE.mkdir(exist_ok=True)
    pc = load('shared_rq2_style', BASE/'figures_formal_20260921/plot_common.py')
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize':8.2, 'xtick.labelsize':7., 'ytick.labelsize':7.,
                           'axes.labelpad':2., 'legend.fontsize':LEGEND_SIZE})
    preserved = {str(p):sha(p) for d in [B1, B2/'figures'] for p in d.iterdir() if p.is_file()}
    results = {'B1':plot_b1(pc,font), 'B2':plot_b2(pc,font)}
    assert all(sha(p) == digest for p,digest in preserved.items())
    (HERE/'GENERATION.json').write_text(json.dumps(dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='rendered; root and independent review pending', figure_inches=SIZE,
        outputs={k:v['outputs'] for k,v in results.items()}, previous_artifacts_unchanged=True,
        source_files_unchanged=preserved, script=record(Path(__file__))),indent=2)+'\n')
    print(json.dumps({k:v['outputs'] for k,v in results.items()},indent=2))


if __name__ == '__main__':
    main()
