"""Rebuild the training-neighborhood figure from the formal three-seed raw logs."""
from collections import defaultdict
import json

from plot_common import (BASE, HERE, METHODS, audited_summary, draw_mean_sd,
    internal_legend, limits_from_sd, np, pd, plt, save_figure, setup_style, sha)


def load_points():
    rows, sources, tolerances = [], [], []
    directions, anchors = {}, {}
    checked = 0
    for seed in range(3):
        for method in METHODS:
            for job, grid, count in (
                ('dense', [i / 4 for i in range(81)], 100),
                ('sparse_train', [0, .5, 1, 2, 5, 10, 20], 16),
            ):
                summary, manifest, receipts = audited_summary(seed, method, job)
                sources.extend(receipts)
                path = BASE / f'formal_results/seed{seed}/{method}/{job}/states.jsonl'
                states = [json.loads(line) for line in path.read_text().splitlines()]
                sources.append(dict(path=str(path), sha256=sha(path)))
                assert len(states) == len({r['state_id'] for r in states}) == 1 + (len(grid)-1)*count
                assert summary['expected_states'] == summary['states_completed'] == len(states)
                assert [r['radius'] for r in summary['radius_summary']] == grid
                anchor_key = (seed, method)
                anchor = (manifest['checkpoint_sha256'], manifest['anchor_effective_sha256'],
                          manifest['head_sha256'], manifest['data_manifest_sha256'],
                          manifest['coordinate_sha256'])
                if anchor_key in anchors:
                    assert anchors[anchor_key] == anchor
                anchors[anchor_key] = anchor
                origin = next(r for r in states if r['point']['radius'] == 0)
                grouped = defaultdict(list)
                for row in states:
                    assert row['seed'] == seed and row['method'] == method
                    assert row['protocol_sha256'] == manifest['protocol_sha256']
                    assert row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method]
                    assert row['data_manifest_sha256'] == manifest['data_manifest_sha256']
                    assert row['coordinate_sha256'] == manifest['coordinate_sha256']
                    assert row['anchor_restored_exactly'] is True
                    assert row['metrics']['train']['samples'] == 1259
                    radius = row['point']['radius']
                    np.testing.assert_allclose(row['effective_weight_drift'], radius, rtol=2e-6, atol=1e-9)
                    if radius:
                        direction = row['point']['direction']
                        idx = direction['index']
                        if idx in directions:
                            assert directions[idx] == direction['sha256']
                        directions[idx] = direction['sha256']
                    grouped[radius].append(row)
                for entry in summary['radius_summary']:
                    radius = entry['radius']; values = grouped[radius]
                    assert len(values) == (1 if radius == 0 else count)
                    if radius:
                        assert {r['point']['direction']['index'] for r in values} == set(range(count))
                    losses = np.array([r['metrics']['train']['loss'] for r in values])
                    damage = losses - origin['metrics']['train']['loss']
                    G = np.maximum(damage, 0).mean()
                    accuracy = np.mean([r['metrics']['train']['accuracy'] for r in values])
                    np.testing.assert_allclose([G, accuracy], [entry['train']['G'], entry['train']['accuracy_mean']], rtol=1e-10, atol=1e-12)
                    item = dict(seed=seed, method=method, job=job, radius=radius,
                        directions=len(values), train_loss_samples=1259,
                        G_train=float(G), train_accuracy_percent=float(accuracy))
                    if job == 'sparse_train':
                        fisher = [r['fisher_by_scope']['subset'] for r in values]
                        assert all(f['samples'] == 256 and f['normalized'] is False and
                                   f['floor'] is None and f['regularizer'] is None for f in fisher)
                        trace = float(np.mean([f['raw_trace'] for f in fisher]))
                        np.testing.assert_allclose(trace, entry['raw_trace_by_scope']['subset']['mean'], rtol=1e-10)
                        item.update(raw_fisher_trace=trace, fisher_samples=256)
                    rows.append(item)
                if job == 'dense':
                    for t in summary['tolerance']:
                        if t['metric'] != 'G':
                            continue
                        radius = 0
                        for entry in summary['radius_summary']:
                            if entry[t['split']]['G'] > t['epsilon']:
                                break
                            radius = entry['radius']
                        assert radius == t['prefix_radius']
                        tolerances.append(dict(seed=seed, method=method, **t))
                checked += len(states)
    assert len(directions) == 100
    return pd.DataFrame(rows), pd.DataFrame(tolerances), sources, checked


def main():
    font = setup_style()
    data, tolerance, sources, checked = load_points()
    dense = data.query("job == 'dense'").copy()
    sparse = data.query("job == 'sparse_train'").copy()
    dense.drop(columns=['raw_fisher_trace', 'fisher_samples']).to_csv(HERE / 'B1_dense_per_seed.csv', index=False)
    sparse.to_csv(HERE / 'B1_fisher_per_seed.csv', index=False)
    tolerance.to_csv(HERE / 'B1_tolerance_per_seed.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.07, right=.99, bottom=.25, top=.965, wspace=.42)
    configurations = [
        (dense, 'G_train', r'Loss $\overline{[D_t(\boldsymbol{\theta},\boldsymbol{\Delta})]_+}$', 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy', 'lower left'),
        (sparse, 'raw_fisher_trace', 'Raw Fisher trace', 'upper left'),
    ]
    aggregates = []
    for ax, (frame, value, label, legend) in zip(axes, configurations):
        agg = draw_mean_sd(ax, frame, 'radius', value, marker_every=4 if value != 'raw_fisher_trace' else 1)
        agg['metric'] = value; aggregates.append(agg)
        ax.set_xlabel('Weight drift'); ax.set_ylabel(label)
        ax.set_xlim(-.35, 20.35); ax.set_xticks([0, 5, 10, 15, 20])
        limits_from_sd(ax, agg, include_zero=value == 'G_train')
        internal_legend(ax, legend)
    pd.concat(aggregates, ignore_index=True).to_csv(HERE / 'B1_mean_sd.csv', index=False)
    name = 'Figure_B1_neighborhood'
    metadata = dict(font=font, source_cohort='formal ImageNet-A seed0/1/2',
        raw_states_recomputed=checked, source_files=sources,
        raw_metrics_match_saved_summaries=True, shared_anchor_across_panels=True,
        shared_direction_hashes_across_seeds_methods_and_jobs=True,
        dense_grid=dict(start=0, end=20, step=.25, points=81, directions=100, training_samples=1259),
        fisher_grid=dict(radii=[0,.5,1,2,5,10,20], directions=16, training_subset_samples=256),
        notes=['One anchor observation at r=0, no duplicate directions at the origin.',
               'Each seed is averaged over directions before three-seed mean and sample SD.',
               'Dense curves use all 81 radii; markers every 1.0; Fisher markers are actual measured radii.',
               'Fisher is raw true-label per-example trace in effective Q/V weight coordinates; no EWC lambda.',
               'Fisher line segments and SD bands between measured radii are visual guides, not new evaluations.',
               'All three panels are training diagnostics; no test damage, Q95, or tolerance panel.'])
    save_figure(fig, name, metadata)
    caption = r'''# B1: Training-neighborhood diagnostics

**English caption.** Training-neighborhood diagnostics on the formal three-seed ImageNet-A cohort (ViT-B/16, rank-10 Q/V LoRA, ten 20-class tasks). EWCLoRA and FS-EWCLoRA are evaluated at their Task-0 anchors using paired unit directions in the 14,155,776-dimensional effective Q/V weight coordinates. Left: mean positive training loss increase, $G(r)=K^{-1}\sum_k[L_0(\theta+r u_k)-L_0(\theta)]_+$. Middle: Task-0 training accuracy (%). Both use all 1,259 old-task training examples, 100 common directions, and 81 radii from 0 to 20 in steps of 0.25; every measured radius is used in the line, with markers spaced at 1.0. Right: unnormalized true-label empirical Fisher trace, $N^{-1}\sum_i\|\nabla_\theta\ell_i\|_2^2$, on a fixed 256-example training subset at radii 0, 0.5, 1, 2, 5, 10, and 20, using the first 16 common directions. The origin is evaluated once. Fisher is not multiplied by the EWC coefficient. Curves show the arithmetic mean across three training seeds after averaging directions within each seed; shading is one sample standard deviation across seeds, not a confidence interval. Lines between Fisher nodes are visual guides. All panels share the same Task-0 anchors, class head, data manifest, perturbation coordinates, and direction library, but Fisher uses fewer samples/directions/radii. These are training-side neighborhood results; they do not imply improved test-side perturbation tolerance.

**中文说明。** 三图依次为训练集平均正损伤、训练准确率、训练子集 raw Fisher trace；没有测试损伤曲线、Q95 或容忍半径子图。前两图使用完整 1,259 张训练图、100 个方向和 81 个半径；第三图使用预先固定的 256 张训练子集、16 个方向和 7 个半径。三图的模型、Task-0 端点、参数坐标与公共方向库相同，但测量预算不同。Fisher 图的连接线不代表新增采样。阴影是三个独立训练 seed 的样本标准差。raw Fisher 不含 λ=2000，也未归一化；数值上千来自高维梯度平方和，不能按百分比解释。更大样本的独立一致性检查（全部 1,259 张、4 个方向、4 个半径）保存在 `sparse_train_full`，没有混入本图曲线。

复现：在本目录运行 `python plot_b1.py`。原始文件与 SHA-256 见 `Figure_B1_neighborhood_validation.json`；每个 seed 的绘图值见 `B1_dense_per_seed.csv` 和 `B1_fisher_per_seed.csv`，聚合数据见 `B1_mean_sd.csv`。原始 checkpoint 的物理文件校验沿用此前完整审核，本次重新读取日志并复算绘图数值。
'''
    (HERE / 'B1_caption.md').write_text(caption)
    (HERE / 'B1_include.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figure_B1_neighborhood.pdf}
\caption{Training-neighborhood diagnostics: mean positive loss increase, training accuracy, and raw empirical Fisher trace. Lines and shading show mean and one sample standard deviation over three training seeds. Loss and accuracy use 1,259 training examples and 100 paired directions at 81 radii; Fisher uses a fixed 256-example training subset and 16 paired directions at seven measured radii. The Fisher trace is unnormalized and does not include the EWC coefficient.}
\label{fig:rq2-formal-neighborhood}
\end{figure*}
''')
    print(json.dumps(dict(figure=name, raw_states=checked, cohort='formal seed0/1/2',
                         tolerance_epsilon_001=tolerance.query('epsilon == .01').to_dict('records')), indent=2))


if __name__ == '__main__':
    main()
