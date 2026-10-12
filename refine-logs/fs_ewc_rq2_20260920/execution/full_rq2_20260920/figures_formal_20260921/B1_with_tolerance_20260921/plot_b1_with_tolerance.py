"""Create a separate four-panel B1 result; leave the prior result files intact."""
from pathlib import Path
import os
import sys
import json

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))
sys.path.insert(0, str(PREVIOUS))
import plot_common as pc
from plot_b1 import load_points

np, pd, plt = pc.np, pc.pd, pc.plt
NAME = 'Figure_B1_neighborhood_with_tolerance'
SAVED_EPS = [0.001, 0.005, 0.01, 0.02, 0.05]


def tolerance_at(radii, prefix_damage, epsilon):
    """Exact inverse of the measured prefix maximum, including equality."""
    index = np.searchsorted(prefix_damage, epsilon, side='right') - 1
    assert index >= 0
    return float(radii[index]), bool(index == len(radii) - 1)


def build_tolerance(dense, stored):
    curves = {}
    thresholds = {0., *SAVED_EPS}
    for (seed, method), rows in dense.groupby(['seed', 'method']):
        rows = rows.sort_values('radius')
        radii = rows.radius.to_numpy()
        prefix = np.maximum.accumulate(rows.G_train.to_numpy())
        assert len(radii) == 81 and radii[0] == 0 and radii[-1] == 20
        curves[seed, method] = (radii, prefix)
        thresholds.update(float(v) for v in prefix if 0 <= v <= .05)
    thresholds = sorted(thresholds)
    records = []
    for (seed, method), (radii, prefix) in curves.items():
        for epsilon in thresholds:
            radius, censored = tolerance_at(radii, prefix, epsilon)
            records.append(dict(seed=seed, method=method, epsilon=epsilon,
                                tolerance_radius=radius, right_censored=censored))
        for saved in stored.query('seed == @seed and method == @method and split == "train"').to_dict('records'):
            assert saved['metric'] == 'G'
            radius, censored = tolerance_at(radii, prefix, saved['epsilon'])
            assert radius == saved['prefix_radius'] and censored == saved['right_censored']
    frame = pd.DataFrame(records)
    aggregate = pc.mean_sd(frame, 'epsilon', 'tolerance_radius')
    counts = frame.groupby(['method', 'epsilon']).right_censored.sum().rename('censored_seeds').reset_index()
    aggregate = aggregate.merge(counts, on=['method','epsilon'], validate='one_to_one')
    for method, group in aggregate.groupby('method'):
        assert np.all(np.diff(group.sort_values('epsilon')['mean']) >= 0)
    return frame, aggregate


def main():
    # Capture the previous version's files, including figures, CSVs and reports.
    old_hashes = {str(p): pc.sha(p) for p in PREVIOUS.iterdir() if p.is_file()}
    font = pc.setup_style()
    data, stored, sources, raw_count = load_points()
    dense, sparse = data.query('job == "dense"').copy(), data.query('job == "sparse_train"').copy()
    inverse, tol_stats = build_tolerance(dense, stored)
    dense.drop(columns=['raw_fisher_trace','fisher_samples']).to_csv(OUT/'dense_training_per_seed.csv', index=False)
    sparse.to_csv(OUT/'fisher_training_per_seed.csv', index=False)
    stored.to_csv(OUT/'saved_G_tolerance_train_and_test.csv', index=False)
    inverse.to_csv(OUT/'tolerance_step_per_seed.csv', index=False)
    tol_stats.to_csv(OUT/'tolerance_step_mean_sd.csv', index=False)

    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.26, top=.966, wspace=.56)
    settings = [
        (dense, 'G_train', r'Loss $\overline{[D_t(\boldsymbol{\theta},\boldsymbol{\Delta})]_+}$', 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy', 'lower left'),
        (sparse, 'raw_fisher_trace', 'Raw Fisher trace', 'upper left'),
    ]
    aggregates = []
    for ax, (frame, value, label, loc) in zip(axes[:3], settings):
        stats = pc.draw_mean_sd(ax, frame, 'radius', value,
            marker_every=4 if value != 'raw_fisher_trace' else 1)
        stats['metric'] = value; aggregates.append(stats)
        pc.limits_from_sd(ax, stats, include_zero=value == 'G_train')
        ax.set(xlabel='Weight drift', ylabel=label, xlim=(-.5,20.5), xticks=[0,10,20])
        pc.internal_legend(ax, loc)
    pd.concat(aggregates, ignore_index=True).to_csv(OUT/'first_three_mean_sd.csv', index=False)

    ax = axes[3]
    ax.axvline(.01, color='.65', linestyle=':', linewidth=.8, zorder=0)
    ax.axhline(20, color='.75', linestyle=':', linewidth=.65, zorder=0)
    for method in pc.METHODS:
        stats = tol_stats.query('method == @method').sort_values('epsilon')
        x, y, sd = (stats[c].to_numpy() for c in ['epsilon','mean','sd'])
        ax.fill_between(x, y-sd, y+sd, step='post', color=pc.COLORS[method], alpha=.16, linewidth=0)
        ax.step(x, y, where='post', color=pc.COLORS[method], linestyle=pc.LINES[method],
                linewidth=1.25, label=pc.LABELS[method], zorder=3)
        chosen = stats[stats.epsilon.isin(SAVED_EPS[:-1])]
        ax.plot(chosen.epsilon, chosen['mean'], linestyle='none', marker=pc.MARKERS[method],
                markersize=3.1, color=pc.COLORS[method], zorder=4)
        focus = stats[stats.epsilon == .01].iloc[0]
        ax.errorbar([.01],[focus['mean']],yerr=[focus['sd']], linestyle='none',
                    color=pc.COLORS[method], capsize=2, linewidth=.8, capthick=.8, zorder=5)
    # Both methods/all seeds are censored at the preset .05 threshold.
    capped = tol_stats[tol_stats.epsilon == .05]
    assert (capped['mean'] == 20).all() and (capped.censored_seeds == 3).all()
    ax.annotate('', xy=(.05,21.1), xytext=(.05,20),
                arrowprops=dict(arrowstyle='-|>', color='.35', lw=.75), zorder=6)
    ax.text(.029, 20.55, 'scan limit', fontsize=7, color='.4', va='bottom')
    ax.set(xlabel=r'Allowed damage $\epsilon$', ylabel=r'Tolerance radius $\hat r_\epsilon$',
           xlim=(-.001,.0515), ylim=(-.55,21.9), yticks=[0,5,10,15,20],
           xticks=[0,.01,.03,.05], xticklabels=['0','0.01','0.03','0.05'])
    pc.internal_legend(ax, 'lower right')

    metadata = dict(font=font, source_cohort='formal ImageNet-A training seeds 0,1,2',
        raw_states_recomputed=raw_count, source_files=sources,
        first_three_metrics_unchanged=True,
        prior_version_file_sha256=old_hashes,
        saved_thresholds=SAVED_EPS, plotted_epsilon_range=[0,.05],
        tolerance_definition='Max measured radius whose entire measured prefix has G <= epsilon; G is mean of positive CE increases over 100 directions.',
        tolerance_render='Exact right-continuous step of finite-grid inverse; no radius interpolation. Mean/sample SD after per-seed inversion.',
        exact_threshold_breakpoints=len(tol_stats.epsilon.unique()),
        right_censor='At epsilon=.05 all six train curves reach the measured scan cap r=20; a shared upward arrow denotes censoring.',
        step_upper_envelope_is_not_confidence_bound=True,
        no_test_curves=True,
        script=dict(path=str(Path(__file__).resolve()),sha256=pc.sha(__file__)),
        dependencies=[dict(path=str(PREVIOUS/n),sha256=pc.sha(PREVIOUS/n)) for n in ['plot_common.py','plot_b1.py']])
    result = pc.save_figure(fig, f'{OUT.name}/{NAME}', metadata)
    assert all(pc.sha(Path(p)) == digest for p, digest in old_hashes.items()), 'A previous-version file changed'
    result['previous_version_files_unchanged'] = True
    (OUT/f'{NAME}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    caption = r'''Training-neighborhood diagnostics on the formal three-seed ImageNet-A cohort (ViT-B/16 with rank-10 Q/V LoRA). From left to right: mean positive training CE damage, training accuracy (%), raw empirical Fisher trace, and empirical training tolerance radius. Damage and accuracy use all 1,259 Task-0 training examples, 100 paired unit directions, and 81 radii from 0 to 20 in steps of 0.25. Fisher uses a fixed 256-example training subset, the first 16 common directions, and seven measured radii (0, 0.5, 1, 2, 5, 10, 20); it is the unnormalized true-label per-example gradient-square sum, without the EWC coefficient. Segments between Fisher nodes are visual guides. The fourth panel is computed from the first panel's per-seed damage data: $\hat r_\epsilon=\max\{r_j:\max_{i\le j}G(r_i)\le\epsilon\}$, where $G(r)=100^{-1}\sum_k[L_0(\theta+ru_k)-L_0(\theta)]_+$. It is the exact finite-grid step function, with no radius interpolation or new model evaluations. Markers show the saved thresholds 0.001, 0.005, 0.01 and 0.02; the vertical dotted line and error bars highlight 0.01. At the saved threshold 0.05, all seeds of both methods reach the scan limit 20; the shared upward arrow denotes right-censoring rather than equality of their unknown boundaries beyond the scan. Some individual seeds reach the cap before 0.05; all plateaus at 20 reflect this cap. Curves and bands are means and one sample SD over training seeds 0/1/2; tolerance radii are inverted per seed before aggregation. The origin is evaluated once. The tolerance panel is a threshold summary of damage, not independent evidence or a guarantee for every direction in a ball. All four panels concern training data and do not establish improved test-neighborhood tolerance.'''
    (OUT/'CAPTION.md').write_text(caption+'\n')
    (OUT/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+NAME+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:rq2-training-neighborhood-tolerance}\n\\end{figure*}\n')
    print(json.dumps(dict(outputs=result['outputs'],previous_version_files_unchanged=True,
                         exact_threshold_breakpoints=result['exact_threshold_breakpoints'],
                         epsilon_001=tol_stats[tol_stats.epsilon==.01].to_dict('records')),indent=2))


if __name__ == '__main__':
    main()
