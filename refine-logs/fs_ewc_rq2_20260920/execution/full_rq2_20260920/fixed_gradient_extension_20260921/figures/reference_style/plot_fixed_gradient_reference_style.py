"""CPU-only reference-style plots from the validated six fixed-gradient jobs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STYLE = ROOT.parent / 'figures_formal_20260921'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2-fixed-gradient-reference-style-mpl')
sys.path.insert(0, str(STYLE))
import plot_common as pc
from matplotlib.ticker import FuncFormatter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt_path = ROOT / 'report/fixed_gradient_complete.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete'
    assert receipt['expected_jobs'] == receipt['validated_jobs'] == 6
    assert receipt['states'] == 558
    assert receipt['gradient_task'] == 0 and receipt['direction_sign'] == 1
    assert receipt['automatic_numeric_and_layout_checks'] == 'passed'
    data_path = ROOT / 'figures/all_per_seed.csv'
    expected = {r['path']: r['sha256'] for r in receipt['output_files']}
    assert sha(data_path) == expected[str(data_path)]
    data = pc.pd.read_csv(data_path)
    assert len(data) == 558 and not data.duplicated(['method', 'seed', 'radius']).any()
    assert (data.directions == 1).all()
    for _, group in data.groupby(['method', 'seed']):
        assert len(group) == 93 and group.radius.min() == 0 and group.radius.max() == 50
    sources = [dict(path=str(p), sha256=sha(p)) for p in
               (receipt_path, data_path, Path(__file__), STYLE / 'plot_common.py')]
    figures = []
    for zoom in (False, True):
        maximum = 2 if zoom else 50
        tag = 'near_origin' if zoom else 'radius50'
        frame = data[data.radius <= maximum].copy()
        pc.HERE = HERE
        font = pc.setup_style()
        fig, axes = pc.plt.subplots(1, 3, figsize=(7.6, 2.15))
        fig.subplots_adjust(left=.083, right=.99, bottom=.27, top=.965, wspace=.48)
        specs = [('D_train', r'Loss change $\overline{D_0(r)}$', 'upper left'),
                 ('train_accuracy_percent', 'Accuracy (%)', 'lower left'),
                 ('raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', 'upper left')]
        grid = sorted(frame.radius.unique())
        markers = list(range(len(grid))) if zoom else [
            i for i, r in enumerate(grid) if r == 0 or r >= 25 or float(r).is_integer()]
        aggregates, ranges = [], []
        for ax, (value, label, legend) in zip(axes, specs):
            agg = pc.draw_mean_sd(ax, frame, 'radius', value, marker_every=markers)
            ranges.append(pc.limits_from_sd(ax, agg, include_zero=value == 'D_train'))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda tick, pos: f'{tick:.2g}' if abs(tick) >= 1e4 or 0 < abs(tick) < 1e-3 else f'{tick:g}'))
            ax.yaxis.get_offset_text().set_visible(False)
            if value == 'D_train':
                ax.axhline(0, color='#777777', linewidth=.5, zorder=0)
            ax.set(xlabel=r'Weight drift $r$', ylabel=label,
                   xlim=(-.025 if zoom else -.5, maximum + (.025 if zoom else .5)),
                   xticks=[0, 1, 2] if zoom else [0, 25, 50])
            if not zoom and value in ('train_accuracy_percent', 'raw_fisher_trace'):
                legend = 'upper right'
            pc.internal_legend(ax, legend)
            agg['metric'] = value
            aggregates.append(agg)
        name = f'Figure_B1_fixed_task0_gradient_{tag}_reference_style'
        pc.pd.concat(aggregates, ignore_index=True).to_csv(HERE / f'{name}_data.csv', index=False)
        metadata = dict(font=font, source_files=sources, expected_jobs=6, validated_jobs=6,
            direction_sign=1, gradient_task=0, measured_radii=grid, axis_ranges=ranges,
            radius_window=[0, maximum], one_fixed_direction_per_seed_method=True,
            directions_shared_across_methods=False, loss_is_signed_ce_change=True,
            overbar_means_average_over_three_seeds=True, fisher_samples=256,
            train_loss_accuracy_samples=1259, fisher_extra_normalization_or_lambda=False)
        result = pc.save_figure(fig, name, metadata)
        figures.append(result['outputs'])
        caption = (
            'Fixed Task0 positive-gradient displacement diagnostic using formal ImageNet-A '
            'seeds 0, 1, and 2 (ViT-B/16, rank-10 Q/V LoRA). Each method/seed uses its own '
            'Task0 checkpoint and one gradient of the mean cross-entropy over all 1,259 '
            'Task0 training examples. Normalize this gradient once and evaluate theta(r) = '
            'theta0 + r*u, keeping the direction fixed across all radii. Perturbations and '
            'gradients use the 24 effective Q/V matrices; the old-task 20-class head is fixed. '
            'Left: signed training CE change relative to the corresponding anchor: '
            'for method m and seed s, D(m,s,r) = L(m,s,theta(m,s,r)) - L(m,s,theta(m,s,0)). '
            'The overbar denotes the arithmetic mean of these changes across three seeds '
            'within each method, with no positive clipping. '
            'Middle: full Task0 training accuracy (%). Right: raw empirical Fisher trace '
            'recomputed at each position on the same fixed 256 training examples, as the '
            'mean squared norm of per-example true-label CE gradients. There is no extra '
            'Fisher normalization or multiplication by lambda. Lines show seed means and '
            'shading shows one sample standard deviation (ddof=1); the independent '
            'replication unit is the training seed, not the direction or radius. '
            'Directions differ between methods. Connecting lines are visual guides between '
            'measured points. This is a fixed-direction diagnostic, not an actual future-task '
            'learning trajectory or a guarantee of monotonic loss at large radius. '
            + ('The companion displays the measured 0–2 range. ' if zoom else
               'The figure displays all 93 measured radii from 0 to 50; the separate '
               '0–2 companion resolves the initial response. ')
        )
        (HERE / f'{name}_CAPTION.md').write_text(caption + '\n')
        tex = caption.replace('%', r'\%')
        (HERE / f'{name}_include.tex').write_text(
            '\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'
            + name + '.pdf}\n\\caption{' + tex + '}\n\\end{figure*}\n')
    for item in sources:
        assert sha(item['path']) == item['sha256']
    result = dict(status='complete', generated_utc=datetime.now(timezone.utc).isoformat(),
                  source_files=sources, figures=figures, original_publisher_outputs_preserved=True,
                  rendered_visual_review='pending actual rendered-image inspection')
    (HERE / 'STYLE_PUBLICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
