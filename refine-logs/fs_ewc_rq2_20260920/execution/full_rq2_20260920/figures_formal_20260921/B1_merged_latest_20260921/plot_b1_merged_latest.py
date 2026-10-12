"""Merge validated 16-radius Fisher measurements with the latest four-panel style.

CPU-only rendering; previous artifacts and experiment outputs are read-only inputs.
"""
from pathlib import Path
import datetime
import json
import os
import shutil
import sys

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
BASE = PARENT.parent
SOURCE = PARENT / 'B1_added_thresholds_20260921'
EXTENSION = BASE / 'stage_path_completion_20260921'
NAME = 'Figure_B1_neighborhood_merged_latest'
LEVELS = [.0025, .0075, .015, .03, .04]
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))
sys.path.insert(0, str(PARENT))
import plot_common as pc

pd, np, plt = pc.pd, pc.np, pc.plt


def checked_copy(path, hashes, target=None):
    assert pc.sha(path) == hashes[str(path)], ('source changed', str(path))
    shutil.copyfile(path, OUT / (target or path.name))
    return dict(path=str(path), sha256=pc.sha(path))


def main():
    before = {str(p): pc.sha(p) for directory in (PARENT, SOURCE)
              for p in directory.iterdir() if p.is_file()}
    prior = pc.read(SOURCE / 'ARTIFACT_MANIFEST.json')
    prior_hashes = {r['path']: r['sha256'] for r in prior['files']}
    receipt_path = EXTENSION / 'report/radius_complete.json'
    receipt = pc.read(receipt_path)
    assert receipt['status'] == 'complete'
    assert receipt['expected_jobs'] == receipt['validated_jobs'] == 6
    assert {(j['seed'], j['method']) for j in receipt['jobs']} == {
        (s, m) for s in range(3) for m in pc.METHODS}
    for job in receipt['jobs']:
        assert job['status'] == 'validated' and job['states'] == 145
        for key in ('anchor_identity_matches', 'code_data_head_coordinate_identity_matches',
                    'direction_library_matches', 'origin_repeat_matches', 'metrics_recomputed_from_raw'):
            assert job[key], (job, key)
    extension_hashes = {r['path']: r['sha256'] for r in receipt['output_files']}
    inputs = [dict(path=str(receipt_path), sha256=pc.sha(receipt_path)),
              dict(path=str(SOURCE / 'ARTIFACT_MANIFEST.json'),
                   sha256=pc.sha(SOURCE / 'ARTIFACT_MANIFEST.json')),
              dict(path=str(PARENT / 'plot_common.py'), sha256=pc.sha(PARENT / 'plot_common.py'))]
    for filename in ('dense_training_per_seed.csv', 'selected_thresholds_per_seed.csv',
                     'selected_thresholds_mean_sd.csv', 'TEN_THRESHOLDS.md'):
        inputs.append(checked_copy(SOURCE / filename, prior_hashes))
    inputs.append(checked_copy(EXTENSION / 'figures/B1_fisher_per_seed.csv', extension_hashes,
                               'fisher_training_per_seed.csv'))
    for filename in ('fisher_training_per_seed.csv', 'first_three_mean_sd.csv',
                     'fourth_panel_mean_sd.csv'):
        path = SOURCE / filename
        assert pc.sha(path) == prior_hashes[str(path)]
        inputs.append(dict(path=str(path), sha256=pc.sha(path)))

    dense = pd.read_csv(OUT / 'dense_training_per_seed.csv')
    fisher = pd.read_csv(OUT / 'fisher_training_per_seed.csv')
    old_fisher = pd.read_csv(SOURCE / 'fisher_training_per_seed.csv')
    assert len(dense) == 486 and len(fisher) == 96
    assert not fisher.duplicated(['method', 'seed', 'radius']).any()
    for _, rows in fisher.groupby(['method', 'seed']):
        assert sorted(rows.radius.tolist()) == receipt['merged_fisher_radii']
    assert (fisher.fisher_samples == 256).all()
    assert (fisher.loc[fisher.radius > 0, 'directions'] == 16).all()
    assert (fisher.loc[fisher.radius == 0, 'directions'] == 1).all()
    assert np.isfinite(fisher.raw_fisher_trace).all() and (fisher.raw_fisher_trace >= 0).all()
    original = fisher[fisher.source_partition == 'original_formal']
    order = ['method', 'seed', 'radius']
    pd.testing.assert_frame_equal(
        original[old_fisher.columns].sort_values(order).reset_index(drop=True),
        old_fisher.sort_values(order).reset_index(drop=True),
        check_exact=False, rtol=1e-12, atol=1e-12)

    # Invert the observed damage curve per seed, before cross-seed aggregation.
    # This check also confirms all ten saved training thresholds against the dense grid.
    all_seed = pd.read_csv(OUT / 'selected_thresholds_per_seed.csv')
    for row in all_seed[all_seed['split'] == 'train'].itertuples():
        curve = dense[(dense.method == row.method) & (dense.seed == row.seed)].sort_values('radius')
        np.testing.assert_allclose(curve.radius, np.arange(81) / 4, rtol=0, atol=0)
        prefix = np.maximum.accumulate(curve.G_train.to_numpy())
        safe = prefix <= row.epsilon
        radius = float(curve.radius.to_numpy()[safe][-1])
        assert radius == row.tolerance_radius
        assert bool(safe[-1]) == bool(row.right_censored)
    selected = all_seed[(all_seed['split'] == 'train') & all_seed.epsilon.isin(LEVELS)].copy()
    assert len(selected) == 30
    stats = pc.mean_sd(selected, 'epsilon', 'tolerance_radius')
    caps = selected.groupby(['method', 'epsilon']).right_censored.sum().rename('censored_seeds').reset_index()
    stats = stats.merge(caps, on=['method', 'epsilon'], validate='one_to_one').sort_values(['method', 'epsilon'])
    old_stats = pd.read_csv(SOURCE / 'fourth_panel_mean_sd.csv').sort_values(['method', 'epsilon'])
    pd.testing.assert_frame_equal(stats.reset_index(drop=True), old_stats.reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    selected.to_csv(OUT / 'fourth_panel_per_seed.csv', index=False)
    stats.to_csv(OUT / 'fourth_panel_mean_sd.csv', index=False)

    font = pc.setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.32, top=.966, wspace=.56)
    specs = [(dense, 'G_train', r'Loss change $\overline{D_t(\boldsymbol{\theta},\boldsymbol{\Delta})}$', 'upper left'),
             (dense, 'train_accuracy_percent', 'Accuracy (%)', 'lower left'),
             (fisher, 'raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', 'upper left')]
    summaries = []
    for ax, (frame, value, label, location) in zip(axes[:3], specs):
        agg = pc.draw_mean_sd(ax, frame, 'radius', value,
                              marker_every=1 if value == 'raw_fisher_trace' else 4)
        pc.limits_from_sd(ax, agg, include_zero=value == 'G_train')
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.5, 20.5), xticks=[0, 10, 20])
        pc.internal_legend(ax, location)
        agg['metric'] = value
        summaries.append(agg)
    actual = pd.concat(summaries, ignore_index=True)
    key = ['method', 'radius', 'metric']
    old = pd.read_csv(SOURCE / 'first_three_mean_sd.csv')
    pd.testing.assert_frame_equal(
        actual[actual.metric != 'raw_fisher_trace'].sort_values(key).reset_index(drop=True),
        old[old.metric != 'raw_fisher_trace'].sort_values(key).reset_index(drop=True),
        check_exact=False, rtol=1e-12, atol=1e-12)
    actual.to_csv(OUT / 'first_three_mean_sd.csv', index=False)

    ax = axes[3]
    for method in pc.METHODS:
        rows = stats.query('method == @method').sort_values('epsilon')
        ax.plot(rows.epsilon, rows['mean'], color=pc.COLORS[method], linestyle=pc.LINES[method],
                linewidth=1.25, marker=pc.MARKERS[method], markersize=3.1,
                label=pc.LABELS[method], zorder=3)
        uncensored = rows[rows.censored_seeds == 0]
        ax.errorbar(uncensored.epsilon, uncensored['mean'], yerr=uncensored.sd,
                    color=pc.COLORS[method], linestyle='none', capsize=2, linewidth=.8, capthick=.8, zorder=2)
    capped = stats[(stats.method == 'gam') & (stats.epsilon == .04)].iloc[0]
    assert capped['mean'] == 20 and capped.censored_seeds == 3
    ax.annotate(r'$\geq 20$', xy=(.04, 20), xytext=(-3, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=7, color=pc.COLORS['gam'])
    ax.set(xlabel=r'Allowed damage $\epsilon$', ylabel=r'Tolerance radius $\hat r_\epsilon$',
           xlim=(0, .0415), ylim=(-.55, 22.1), yticks=[0, 5, 10, 15, 20],
           xticks=LEVELS, xticklabels=['0.0025', '0.0075', '0.015', '0.03', '0.04'])
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    pc.internal_legend(ax, 'lower right')
    fig.align_xlabels(axes)
    encodings = []
    for ax in axes:
        mapping = {}
        for method in pc.METHODS:
            line = next(l for l in ax.lines if l.get_label() == pc.LABELS[method])
            mapping[method] = dict(color=line.get_color(), marker=line.get_marker(),
                dash=line._unscaled_dash_pattern, linewidth=line.get_linewidth(), markersize=line.get_markersize())
        encodings.append(mapping)
    assert all(x == encodings[0] for x in encodings)

    meta = dict(font=font, source_files=inputs, source_cohort='formal ImageNet-A training seeds 0/1/2',
        displayed_thresholds=LEVELS, displayed_fourth_panel_values=stats.to_dict('records'),
        merged_fisher_radii=receipt['merged_fisher_radii'], validated_extension_jobs=6,
        first_two_aggregates_unchanged=True, fourth_panel_aggregates_unchanged=True,
        original_seven_fisher_radii_unchanged=True, all_training_thresholds_recomputed_from_dense_grid=True,
        new_model_evaluations_by_this_plot_script=False,
        fisher_spike_preserved='r=17, EWCLoRA; mean and full sample-SD range retained without clipping or smoothing',
        previous_version_file_sha256=before, all_panel_method_encodings=encodings,
        script=dict(path=str(Path(__file__).resolve()), sha256=pc.sha(__file__)))
    result = pc.save_figure(fig, f'{OUT.name}/{NAME}', meta)
    assert all(pc.sha(Path(p)) == digest for p, digest in before.items())
    result['prior_versions_preserved'] = True
    (OUT / f'{NAME}_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    caption = r'''Training-neighborhood diagnostics on the formal three-seed ImageNet-A cohort (ViT-B/16, rank-10 Q/V LoRA). Left to right: mean positive cross-entropy (CE) change, training accuracy (%), empirical Fisher trace, and empirical tolerance radius. Loss and accuracy use 1,259 Task-0 training examples, 100 paired directions and 81 measured radii (0–20, step 0.25). Fisher uses the same anchors and effective Q/V coordinates, a fixed 256-example training subset and 16 paired directions at each nonzero radius, with no EWC scaling or normalization. Its 16 measured radii are 0, 0.5, 1, 2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and 20; the origin is evaluated once per seed and method. The r=17 EWCLoRA peak and its full cross-seed standard deviation are retained. All directions are paired between methods within each seed. The first-axis average-damage shorthand denotes $G(r)=K^{-1}\sum_k\max\{N^{-1}\sum_i[\ell_i(\theta+r u_k)-\ell_i(\theta)],0\}$: sample-average CE differences are clipped per direction before averaging directions. The fourth panel shows the five selected training thresholds $\epsilon\in\{0.0025,0.0075,0.015,0.03,0.04\}$ on a linear axis. Tolerance is computed independently for each seed from the 81-radius loss grid using $\hat r_\epsilon=\max\{r_j:\max_{i\le j}G(r_i)\le\epsilon\}$, then averaged across seeds. Thresholds refer to mean positive damage of sample-average CE, not the summed CE or a per-example bound. At $\epsilon=0.04$, all three FS-EWCLoRA seeds reach the scan limit 20; the plotted value is labeled $\geq20$ and does not identify a boundary beyond the scan. No zero-SD error bar is shown at this censored point. Shading in panels 1–3 and error bars at uncensored panel-4 points show one sample standard deviation across three training seeds, not confidence intervals. Fisher and tolerance connecting segments are visual guides; no new intermediate-radius measurements or interpolated tolerance estimates are implied. The full ten-threshold train/test table is supplied separately. The figure shows training diagnostics only. This rendering merges completed supplemental Fisher measurements with the previous four-panel style; it does not launch training or model evaluation.'''
    (OUT / 'CAPTION.md').write_text(caption + '\n')
    (OUT / 'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{' + NAME + '.pdf}\n\\caption{' + caption + '}\n\\label{fig:rq2-neighborhood-merged}\n\\end{figure*}\n')
    table = []
    for epsilon in LEVELS:
        vals = {m: stats[(stats.method == m) & (stats.epsilon == epsilon)].iloc[0] for m in pc.METHODS}
        gam = '≥20（扫描上限）' if vals['gam'].censored_seeds else f"{vals['gam']['mean']:.3f}"
        table.append(f"| {epsilon:g} | {vals['sgd']['mean']:.3f} | {gam} |")
    readme = f'''# B1 最新四子图合并版

[PNG]({NAME}.png) · [矢量 PDF]({NAME}.pdf) · [完整十档训练／测试阈值](TEN_THRESHOLDS.md)

保留 Times New Roman、一行四列、图内图例，以及四图一致的方法颜色、线型和标记。前三图横轴为 Weight drift $r$。旧版本均保留；本次仅使用 CPU 读取已完成结果作图。

1. 平均正 CE 损伤：1259 个 Task-0 训练样本，100 个配对方向，81 个半径（0–20，步长0.25）。
2. 训练准确率（%）：与图1相同的实测数据。
3. Fisher trace：256 个固定训练样本，16 个配对方向；合并原有7个半径与11–19九个新半径，共16个。所有6组补测已经过原点、模型/数据/坐标/方向身份与原始数值重算校验。Fisher不乘EWC的lambda，不归一化。EWCLoRA在r=17的尖峰主要来自seed0，按原值保留，纵轴覆盖均值±样本标准差。
4. 训练容忍半径：保留新增五个阈值。先对各seed的81点损伤曲线取前缀最大值，再反求最大满足阈值的已测半径，最后计算三个seed均值和样本标准差；本次重新核对了全部十档训练阈值。

| epsilon | EWCLoRA 均值 | FS-EWCLoRA 均值 |
|---:|---:|---:|
{chr(10).join(table)}

epsilon是“样本平均交叉熵相对各自锚点的正增量，再对方向平均”的阈值，不是整个数据集CE求和，也不是保证每个样本/每个方向安全。最后一点≥20表示达到扫描上限；没有恢复上限横线或平台。线段仅为视觉连接，不插值求容忍半径。

原图1、图2以及最新阈值版图4的数值均保持一致。新增Fisher不能改变基于原始损伤曲线得到的图4。阴影/误差棒为3个训练seed的样本标准差。

数据来自正式三seed ImageNet-A队列，与历史ImageNet-R诊断分开；本次没有切换数据集或模型。来源：[Fisher完成记录]({receipt_path})；详细图注见[CAPTION.md](CAPTION.md)，来源哈希与一致性检查见[{NAME}_validation.json]({NAME}_validation.json)。

复现：`/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python plot_b1_merged_latest.py`。
'''
    (OUT / 'README.md').write_text(readme)
    artifact = dict(status='numeric_and_layout_checks_passed',
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rendered_review='pending root and independent inspection', prior_versions_preserved=True,
        files=[dict(path=str(p), sha256=pc.sha(p)) for p in sorted(OUT.iterdir())
               if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'])
    (OUT / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(outputs=result['outputs'], fisher_radii=receipt['merged_fisher_radii'],
                         fourth_panel=stats.to_dict('records')), indent=2))


if __name__ == '__main__':
    main()
