"""B2 presentation variant: retain three-seed means, omit all SD shading."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
spec = importlib.util.spec_from_file_location('unified_source', SOURCE/'plot_unified.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def draw_means(pc, ax, frame, x, y, *, sd=False, markevery=1):
    assert sd, 'This variant keeps the original three-seed aggregation.'
    values = pc.mean_sd(frame, x, y)
    for method in pc.METHODS:
        curve = values[values.method == method].sort_values(x)
        xx, yy = curve[x].to_numpy(), curve['mean'].to_numpy()
        line, = ax.plot(xx, yy, color=pc.COLORS[method], linestyle=pc.LINES[method],
                       linewidth=1.15, marker=pc.MARKERS[method], markersize=1.6,
                       markeredgewidth=.35, markevery=markevery,
                       label=pc.LABELS[method], zorder=3)
        pc.np.testing.assert_array_equal(line.get_xdata(), xx)
        pc.np.testing.assert_array_equal(line.get_ydata(), yy)
    return values


def keep_legend(fig, ax):
    leg = plot.legend(ax, 'lower right', anchor=(.999,.09))
    fig.canvas.draw()
    box = leg.get_window_extent(fig.canvas.get_renderer())
    assert not any(line.get_transform().transform_path(line.get_path()).intersects_bbox(box,filled=False)
                   for line in ax.lines)
    return dict(anchor=[.999,.09], bounding_box_clear_of_mean_curves=True)


def main():
    original_hashes = {str(p):plot.sha(p) for p in SOURCE.iterdir() if p.is_file()}
    pc = plot.load('mean_only_shared_style', plot.BASE/'figures_formal_20260921/plot_common.py')
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize':8.2, 'xtick.labelsize':7., 'ytick.labelsize':7.,
                           'axes.labelpad':2., 'legend.fontsize':plot.LEGEND_SIZE})
    plot.HERE = HERE
    plot.draw = draw_means
    plot.place_path_legend = keep_legend
    original_save = plot.save

    def save_no_sd(pc, fig, name, metadata):
        assert all(len(ax.collections) == 0 for ax in fig.axes), 'Unexpected shaded collection'
        assert all(len(ax.lines) == 2 for ax in fig.axes)
        metadata.update(statistics_scope='three-seed arithmetic means shown; sample SD retained in CSV but not drawn',
                        uncertainty_bands_drawn=False, mean_curves_unchanged=True,
                        presentation_script=plot.record(Path(__file__)),
                        preserved_figure_size_and_axes_and_legend=True)
        return original_save(pc, fig, name, metadata)

    plot.save = save_no_sd
    result = plot.plot_b2(pc, font)
    for name in ['B2_mean_sd.csv','B2_trajectory_seed_data.csv','B2_path_seed_data.csv']:
        assert plot.sha(HERE/name) == plot.sha(SOURCE/name), name
    assert all(plot.sha(path) == digest for path,digest in original_hashes.items())
    (HERE/'GENERATION.json').write_text(json.dumps(dict(
        status='rendered; visual review pending', outputs=result['outputs'],
        standard_deviation_shading=False, mean_and_sd_csvs_identical=True,
        parent_figures_and_files_unchanged=True, parent_hashes=original_hashes,
        script=plot.record(Path(__file__))),indent=2)+'\n')
    print(json.dumps(result['outputs'],indent=2))


if __name__ == '__main__':
    main()
