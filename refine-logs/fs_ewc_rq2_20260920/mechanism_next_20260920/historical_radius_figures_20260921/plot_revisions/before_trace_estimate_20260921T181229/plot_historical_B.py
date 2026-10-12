"""Three panels from the immutable historical 81-radius training records."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplcache'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ('sgd', 'gam')
LABELS = {'sgd': 'EWCLoRA', 'gam': 'FS-EWCLoRA'}
COLORS = {'sgd': '#0072B2', 'gam': '#D55E00'}
LINES = {'sgd': '-', 'gam': (0, (4, 2))}
MARKERS = {'sgd': 'o', 'gam': 'D'}
DIMENSION = 14155776


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    manifest_path = BASE / 'radius_reanalysis_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    reanalysis_path = BASE / 'radius_damage_seed0.csv'
    reanalysis = pd.read_csv(reanalysis_path, float_precision='round_trip')
    summary, direction_rows, sources = [], [], []
    for source in manifest['sources']:
        if source['split'] != 'train':
            continue
        method = source['method']
        path = Path(source['path'])
        assert sha(path) == source['sha256']
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(records) == 81
        np.testing.assert_array_equal([r['radius'] for r in records], np.arange(81) / 4)
        for line, row in enumerate(records, 1):
            assert row['K'] == 100 and row['main_prefix'] == 'endpoint_train'
            values = row['per_u']
            loss = np.asarray(values['loss_delta'], dtype=float)
            acc = np.asarray(values['perturbed_acc'], dtype=float)
            q = np.asarray(values['endpoint_train_fisher_quad'], dtype=float)
            assert loss.shape == acc.shape == q.shape == (100,)
            assert np.isfinite(loss).all() and np.isfinite(acc).all() and np.isfinite(q).all()
            assert ((0 <= acc) & (acc <= 100)).all() and (q >= 0).all()
            radius = float(row['radius'])
            ref = reanalysis.query("method == @method and split == 'train' and radius == @radius")
            assert len(ref) == 1
            ref = ref.iloc[0]
            positive = np.maximum(loss, 0)
            np.testing.assert_allclose([positive.mean(), acc.mean()], [ref.G, ref.accuracy_mean], rtol=1e-10, atol=1e-12)
            # Saved mean is a float32 reduction; compute the plot from full per_u arrays.
            np.testing.assert_allclose(q.mean(), row['mean']['endpoint_train_fisher_quad'], rtol=3e-7, atol=1e-14)
            np.testing.assert_allclose(acc.mean(), row['mean']['perturbed_acc_mean'], rtol=3e-7, atol=1e-5)
            summary.append(dict(method=method, label=LABELS[method], dataset='ImageNet-A', seed=0,
                split='train', radius=radius, K=100, G=positive.mean(), D_mean=loss.mean(),
                accuracy=acc.mean(), Q_F_mean=q.mean(), Q_F_direction_std=q.std(ddof=1),
                Q_F_direction_rms=np.sqrt(np.mean(q*q)),
                Fisher_weighted_length_mean=np.sqrt(2*q).mean(),
                Fisher_weighted_length_rms=np.sqrt(2*q.mean()),
                Q_F_divided_by_r_squared=q.mean()/(radius*radius) if radius > 0 else np.nan,
                isotropic_trace_one_reference=radius*radius/(2*DIMENSION),
                source_path=str(path), source_line=line))
            for k in range(100):
                direction_rows.append(dict(method=method, radius=radius, direction_ordinal=k,
                    D=float(loss[k]), G_sample=float(positive[k]), accuracy=float(acc[k]),
                    Q_F=float(q[k]), source_line=line))
        sources.append(dict(path=str(path), sha256=sha(path), records=81, directions_per_radius=100))
    frame = pd.DataFrame(summary)
    assert len(frame) == 162 and set(frame.method) == set(METHODS)
    assert not frame.duplicated(['method', 'radius']).any()
    frame.to_csv(HERE / 'historical_B_plot_data.csv', index=False)
    pd.DataFrame(direction_rows).to_csv(HERE / 'historical_B_direction_data.csv', index=False)
    return frame, dict(status='passed', dataset='ImageNet-A', seed=0, split='train',
        radii_count=81, directions_per_radius=100, per_direction_rows=16200,
        point_values_recomputed_from_raw_arrays=True, loss_accuracy_match_reanalysis=True,
        q_mean_matches_float32_report_with_roundoff_tolerance=True,
        trace_normalized_fixed_anchor_Fisher=True, Fisher_recomputed_at_perturbation=False,
        Fisher_lambda_multiplier=False, Fisher_estimator='square of minibatch-mean CE gradient proxy',
        direction_pairs_across_radii=False, direction_pairing_across_methods='not verified',
        train_transform='one cached stochastic training-augmentation draw',
        effective_weight_dimension=DIMENSION, sources=sources,
        supporting_sources=[dict(path=str(p), sha256=sha(p)) for p in (manifest_path, reanalysis_path)],
        measured_values_unchanged=True, new_GPU_jobs=0)


def main():
    frame, checks = load_data()
    for font in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        font_manager.fontManager.addfont(Path('/home-local/liying/.fonts') / font)
    assert font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold', 'mathtext.cal': 'Times New Roman:italic',
        'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman',
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.073, right=.993, bottom=.24, top=.89, wspace=.46)
    columns = [('G', r'Loss $\overline{[D_t(\boldsymbol{\theta},\boldsymbol{\Delta})]_+}$'),
               ('accuracy', 'Accuracy'), ('Q_F_mean', r'$\overline{Q}_F(r)$')]
    coordinates = []
    for index, (ax, (column, label)) in enumerate(zip(axes, columns)):
        for method in METHODS:
            data = frame.query('method == @method').sort_values('radius')
            # All81 observations form each curve; markers every2.5 units aid legibility.
            line, = ax.plot(data.radius, data[column], color=COLORS[method], ls=LINES[method],
                marker=MARKERS[method], markersize=3, markevery=10,
                linewidth=2 if index == 2 and method == 'sgd' else 1.2, label=LABELS[method],
                zorder=2 if method == 'sgd' else 3)
            np.testing.assert_array_equal(line.get_xdata(), data.radius)
            np.testing.assert_array_equal(line.get_ydata(), data[column])
            coordinates.append(dict(panel=index+1, method=method, column=column,
                x=data.radius.tolist(), y=data[column].tolist()))
        ax.set_xlim(-.7, 20.7); ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlabel('Weight drift', labelpad=3); ax.set_ylabel(label, labelpad=3)
        low, high = frame[column].min(), frame[column].max()
        pad = .04*(high-low)
        limits = (0 if index == 2 else low-pad, high+pad)
        ax.set_ylim(*limits)
        ticks = MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]).tick_values(*limits)
        ax.set_yticks([x for x in ticks if limits[0] <= x <= limits[1]])
        if index == 2:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0)); ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', length=3, pad=3)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.legend(loc='lower left' if index == 1 else 'upper left', frameon=False,
            handlelength=1, handletextpad=.3, labelspacing=.25, borderpad=0, borderaxespad=.3)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for ax in axes:
        box, area = ax.get_legend().get_window_extent(renderer), ax.get_window_extent(renderer)
        assert area.contains(box.x0, box.y0) and area.contains(box.x1, box.y1)
        for line in ax.lines:
            assert not line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False)
    for obj in fig.findobj(match=Text):
        if obj.get_visible() and obj.get_text().strip():
            box = obj.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, obj.get_text()
    outputs = {}
    for suffix in ('pdf', 'png'):
        path = HERE / f'Figure_B_historical_QF.{suffix}'
        fig.savefig(path, dpi=300)
        outputs[suffix] = dict(path=str(path), sha256=sha(path))
    plt.close(fig)
    checks.update(layout='one row, three columns', font='Times New Roman', all_axes_linear=True,
        figure_inches=[7.6,1.85], markers_every_n_observations=10, all81_values_drawn=True,
        legend_inside_without_curve_overlap=True, text_inside_canvas=True,
        plotted_coordinates=coordinates, outputs=outputs,
        script_sha256=sha(Path(__file__)), rendered_visual_review='pending')
    (HERE / 'PROVENANCE.json').write_text(json.dumps(checks, indent=2) + '\n')
    caption = (
        'Historical ImageNet-A seed0 Task0 neighborhood measurements for EWCLoRA (SGD) and FS-EWCLoRA (visual GAM). '
        'From left: directional mean positive Task0 training CE damage, directional mean training accuracy (%), '
        'and directional mean fixed-anchor trace-normalized Fisher quadratic. '
        r'$\overline Q_F(r)=K^{-1}\sum_k\frac12r^2u_k^\top\widetilde F_0u_k$, where $K=100$ and '
        r'$\widetilde F_0$ is the stored diagonal anchor Fisher proxy with trace approximately one. '
        'This proxy squares minibatch-mean CE gradients, is fixed throughout the radius scan and has no EWC lambda multiplier; '
        'it is not raw per-example empirical Fisher trace at the perturbed point. '
        'All 81 radii from 0 to 20 in steps of 0.25 are plotted; markers are shown every 2.5 units. '
        'Fresh isotropic directions were drawn at each radius; cross-method direction pairing is unverified. '
        'All three quantities are read from the same historical training-split radius/direction records; '
        'train inputs use one cached draw of stochastic training augmentation, and each method retains its own fixed Task0 head. '
        r'Positive damage is $[L_0(\theta_0+ru)-L_0(\theta_0)]_+$. '
        'Weight drift is the joint Frobenius radius over 24 effective Q/V matrices (14,155,776 coordinates). '
        'The nearly overlapping third-panel means are expected for isotropic directions because the anchor traces were normalized. '
        'This historical plot uses a different measurement protocol from the deterministic raw-Fisher diagnostic; '
        'its curves are not mixed with those new measurements.\n')
    (HERE / 'CAPTION.md').write_text(caption)
    (HERE / 'include.tex').write_text('% Requires \\usepackage{graphicx,amsmath}.\n\\begin{figure*}[t]\n\\centering\n'
        '\\includegraphics[width=\\textwidth]{Figure_B_historical_QF.pdf}\n\\caption{'+caption.strip().replace('%',r'\%')+'}\n\\end{figure*}\n')
    print(json.dumps({'status':'passed', 'rows':len(frame), 'raw_direction_records':16200,
                      'figure':outputs['pdf']['path']}, indent=2))


if __name__ == '__main__':
    main()
