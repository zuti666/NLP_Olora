"""Titled B1/B2 panels; compact positive-damage label and upper-left path legend."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'figures_unified_20260922'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    plot = load('original_unified', SOURCE/'plot_unified.py')
    means = load('mean_only', SOURCE/'b2_no_sd/plot_b2_no_sd.py')
    pc = plot.load('shared_rq2_style', plot.BASE/'figures_formal_20260921/plot_common.py')
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize':8.2, 'xtick.labelsize':7., 'ytick.labelsize':7.,
                           'axes.labelpad':2., 'legend.fontsize':plot.LEGEND_SIZE})
    previous = {str(p):plot.sha(p) for directory in [SOURCE, SOURCE/'b2_no_sd', HERE.parent/'figures_reordered_20260922']
                for p in directory.iterdir() if p.is_file()}
    plot.HERE = HERE
    original_save = plot.save
    orders = {'B1':[1,0,3,2], 'B2':[2,1,0,3]}
    names = {'B1':'Figure_B1_titled', 'B2':'Figure_B2_titled'}
    titles = {'B1':['Accuracy','Loss change','Tolerance radius','Trace of Fisher'],
              'B2':['Accuracy','Loss change','Weight drift','Average Fisher']}

    def save_reordered(pc, fig, old_name, metadata):
        kind = 'B1' if '_B1_' in old_name else 'B2'
        axes = list(fig.axes)
        boxes = [ax.get_position().frozen() for ax in axes]
        snapshots = [[(line.get_xdata().copy(),line.get_ydata().copy()) for line in ax.lines] for ax in axes]
        for position, old_index in enumerate(orders[kind]):
            box = boxes[position]
            axes[old_index].set_position([box.x0, box.y0, box.width, .87-box.y0])
        # Recreate legends after moving axes: do not retain cached anchors from
        # the pre-reorder canvas draw used by the path-legend layout check.
        locs = (['upper left','lower left','upper left','lower right'] if kind == 'B1'
                else ['lower right','upper left','lower left','upper left'])
        for i, (ax, loc) in enumerate(zip(axes, locs)):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plot.legend(ax, loc)
        for ax, old_lines in zip(axes,snapshots):
            assert len(ax.collections) == 0, 'No uncertainty shading requested'
            for line,(x,y) in zip(ax.lines,old_lines):
                pc.np.testing.assert_array_equal(line.get_xdata(),x)
                pc.np.testing.assert_array_equal(line.get_ydata(),y)
        visual = [axes[i] for i in orders[kind]]
        for ax, title in zip(visual, titles[kind]):
            ax.set_title(title, fontsize=8.6, fontweight='normal', pad=2.)
        if kind == 'B1':
            visual[1].set_ylabel(r'Loss $\overline{D_0}$')
        if kind == 'B2':
            metadata['path_legend'] = dict(location='upper left', between_curves=False)
        metadata.update(panel_source_indices=orders[kind],
                        visual_panel_labels=[ax.get_ylabel() for ax in visual],
                        visual_panel_titles=titles[kind],
                        reordered_script=plot.record(Path(__file__)),
                        uncertainty_bands_drawn=False, values_unchanged_by_reordering=True)
        if kind == 'B1':
            metadata.pop('tolerance_recomputed_from_panel1',None)
            metadata.update(tolerance_recomputed_from_loss_panel=True,
                            loss_panel_position=2, tolerance_panel_position=3)
        if kind == 'B2':
            metadata['statistics_scope']='three-seed means shown; sample SD retained in CSV, no shading'
        result = original_save(pc,fig,names[kind],metadata)
        result['axes'] = [result['axes'][i] for i in orders[kind]]
        result['axes_list_order']='visual left to right'
        for entry,title in zip(result['axes'],titles[kind]):
            entry['title']=title
        (HERE/f'{names[kind]}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result

    plot.save = save_reordered
    results = {'B1':plot.plot_b1(pc,font)}
    plot.draw = means.draw_means
    plot.place_path_legend = means.keep_legend
    results['B2'] = plot.plot_b2(pc,font)
    csv_names = ['B1_dense_seed0.csv','B1_fisher_seed0.csv','B1_tolerance_seed0.csv','B1_plotted_values.csv',
                 'B2_mean_sd.csv','B2_trajectory_seed_data.csv','B2_path_seed_data.csv']
    for name in csv_names:
        assert plot.sha(HERE/name) == plot.sha(SOURCE/name), name
    assert all(plot.sha(path)==digest for path,digest in previous.items())
    (HERE/'GENERATION.json').write_text(json.dumps(dict(
        status='rendered; root and independent review pending',
        figure_inches=list(plot.SIZE), output_files={kind:r['outputs'] for kind,r in results.items()},
        visual_orders={kind:r['visual_panel_labels'] for kind,r in results.items()},
        source_csvs_identical=True, previous_figures_unchanged=True,
        source_hashes=previous, script=plot.record(Path(__file__))),indent=2)+'\n')
    print(json.dumps({kind:r['outputs'] for kind,r in results.items()},indent=2))


if __name__ == '__main__':
    main()
