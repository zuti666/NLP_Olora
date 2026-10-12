#!/usr/bin/env python3
"""CPU-only audit of the current B2 CSVs; write analysis artifacts only."""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/b2_current_analysis_mpl')
OUT = Path(__file__).resolve().parent
FIG = OUT.parent
BASE = FIG.parent
EXT = BASE / 'stage_path_completion_20260921'
spec = importlib.util.spec_from_file_location('b2_readonly_collector', EXT / 'publish.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
np, pd = collector.np, collector.pd
read, sha = collector.read, collector.sha
LABELS = {'sgd': 'EWCLoRA', 'gam': 'FS-EWCLoRA'}
PANELS = ['test_accuracy_percent', 'signed_test_damage', 'weight_drift', 'S_path']
METRIC_NAMES = {'test_accuracy_percent': 'Task0 测试准确率 (%)',
                'signed_test_damage': '有符号测试 CE 变化', 'weight_drift': '有效 Q/V 位移 δ',
                'S_path': '归一化路径 Fisher S_path', 'test_CE': '绝对测试 CE',
                'D_train': '有符号训练 CE 变化', 'integral': '路径积分 I',
                'accuracy_drop_from_anchor_pp': '相对自身Task0准确率掉点 (pp)'}


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    original_hashes = {str(p): sha(p) for p in FIG.iterdir() if p.is_file()}
    trajectory, paths, nodes, evidence = collector.collect()
    assert evidence['status'] == 'ready'
    assert (len(trajectory), len(paths), len(nodes)) == (60, 54, 502)
    comparisons = []
    for name, raw in [('B2_trajectory_seed_data.csv', trajectory), ('B2_path_seed_data.csv', paths)]:
        current = pd.read_csv(FIG / name)
        a = current.sort_values(['seed', 'method', 'T']).reset_index(drop=True)
        b = raw[a.columns].sort_values(['seed', 'method', 'T']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
        comparisons.append({'file': str(FIG / name), 'sha256': sha(FIG / name),
                            'matches_recomputed_raw_states': True, 'rows': len(current)})

    final_gates = {}
    for gate in evidence['numerical_gates']:
        key = (gate['seed'], gate['method'], gate['task'])
        if key not in final_gates or gate['nodes'] > final_gates[key]['nodes']:
            final_gates[key] = gate
    assert len(final_gates) == 54 and all(g['passed'] for g in final_gates.values())
    failed_intermediate = [g for g in evidence['numerical_gates'] if not g['passed']]
    assert len(failed_intermediate) == 2
    assert paths.measured_nodes.value_counts().to_dict() == {9: 52, 17: 2}

    # Verify the exact training runs reached by plotted checkpoint provenance.
    queue_path = BASE / 'formal_queue/pipeline_state.json'
    queue = read(queue_path)
    training, extra_sources = [], [collector.source_record(queue_path)]
    for seed in range(3):
        for method in ('sgd', 'gam'):
            for mode in ('anchor', 'sequence'):
                run = f'{mode}_{method}_seed{seed}'
                root = BASE / 'formal_training' / run
                manifest_path, receipt_path = root / 'run_manifest.json', root / 'COMPLETE.json'
                manifest, receipt = read(manifest_path), read(receipt_path)
                cfg = manifest['config']
                assert manifest['seed'] == seed and manifest['anchor_method'] == method
                assert cfg['dataset'] == 'imageneta' and cfg['ewc_lambda'] == 2000
                assert cfg['seed'] == [seed] and cfg['optimizer_type'] == method
                assert cfg['optimizer_type_by_task'] is None
                assert cfg['epochs'] == cfg['init_epoch'] == 50
                assert cfg['batch_size'] == 128 and cfg['lora_rank'] == 10
                assert receipt['status'] == 'complete'
                assert receipt['identity_sha256'] == manifest['identity_sha256']
                q = queue['jobs'][run]
                assert q['status'] == 'complete' and q['exit_code'] == 0 and q['receipt_valid']
                assert q['receipt_sha256'] == sha(receipt_path)
                assert receipt['validation']['runtime_receipts_sha256'] == sha(root / 'runtime_receipts.jsonl')
                assert receipt['validation']['rng_receipts_sha256'] == sha(root / 'rng_receipts.jsonl')
                resets = [json.loads(s) for s in (root / 'rng_receipts.jsonl').read_text().splitlines()
                          if json.loads(s)['event'] == 'task_rng_reset']
                expected_tasks = [0] if mode == 'anchor' else list(range(1, 10))
                assert sorted(r['task'] for r in resets) == expected_tasks
                assert all(r['optimizer_type'] == method for r in resets)
                plotted = trajectory[(trajectory.seed == seed) & (trajectory.method == method)
                                     & ((trajectory.task == 0) if mode == 'anchor' else (trajectory.task > 0))]
                for row in plotted.itertuples():
                    assert Path(row.checkpoint).parent.parent == root
                    checkpoint_record = receipt['validation']['checkpoint_assets'][f'checkpoints/task_{row.task:02d}.pt']
                    assert row.checkpoint_sha256 == checkpoint_record['sha256']
                if mode == 'sequence':
                    parent = BASE / 'formal_training' / f'anchor_{method}_seed{seed}' / 'checkpoints/task_00.pt'
                    assert manifest['parent']['task_00.pt']['path'] == str(parent)
                    anchor = trajectory[(trajectory.seed == seed) & (trajectory.method == method) & (trajectory.task == 0)].iloc[0]
                    assert manifest['parent']['task_00.pt']['sha256'] == anchor.checkpoint_sha256
                training.append(dict(run=run, seed=seed, method=method, mode=mode,
                    dataset=cfg['dataset'], ewc_lambda=cfg['ewc_lambda'], optimizer_type=cfg['optimizer_type'],
                    optimizer_type_by_task=None, recorded_tasks=expected_tasks,
                    completed_tasks=receipt['completed_tasks'], epoch_budget=50,
                    init_lr=cfg['init_lr'], later_lr=cfg['lrate'], batch_size=cfg['batch_size'],
                    rank=cfg['lora_rank'], parent=manifest['parent'],
                    exit_code=q['exit_code'], receipt_valid=q['receipt_valid'],
                    training_manifest=str(manifest_path), complete_receipt=str(receipt_path),
                    rng_receipts=str(root / 'rng_receipts.jsonl')))
                extra_sources.extend(collector.source_record(p) for p in
                    (manifest_path, receipt_path, root / 'rng_receipts.jsonl', root / 'runtime_receipts.jsonl'))

    # Account for the small difference between the two methods' starting accuracy.
    anchor_accuracy = trajectory[trajectory['T'] == 1].set_index(['seed', 'method']).test_accuracy_percent
    trajectory['anchor_test_accuracy_percent'] = [float(anchor_accuracy.loc[(r.seed, r.method)])
                                                 for r in trajectory.itertuples()]
    trajectory['accuracy_drop_from_anchor_pp'] = trajectory.anchor_test_accuracy_percent - trajectory.test_accuracy_percent
    # Each summary contains the three independent training seeds, never 27 pseudo-replicates.
    frames = {m: trajectory for m in PANELS[:3] + ['test_CE', 'accuracy_drop_from_anchor_pp']}
    frames.update({m: paths for m in ['S_path', 'D_train', 'integral']})
    summary, paired, pair_raw, counts, changes, stage_average = [], [], [], [], [], []
    for metric, frame in frames.items():
        assert np.isfinite(frame[metric].astype(float)).all()
        for (method, T), group in frame.groupby(['method', 'T']):
            group = group.sort_values('seed')
            assert group.seed.tolist() == [0, 1, 2]
            vals = group[metric].astype(float)
            summary.append(dict(metric=metric, method=method, T=int(T), task=int(T)-1, n=3,
                                mean=float(vals.mean()), sample_sd=float(vals.std(ddof=1)),
                                seed0=float(vals.iloc[0]), seed1=float(vals.iloc[1]), seed2=float(vals.iloc[2])))
        p = frame.pivot(index=['seed', 'T'], columns='method', values=metric).reset_index()
        p['gam_minus_sgd'] = p.gam - p.sgd
        for row in p.to_dict('records'):
            row.update(metric=metric, task=int(row['T'])-1)
            pair_raw.append(row)
        for T, group in p.groupby('T'):
            group = group.sort_values('seed')
            diff = group.gam_minus_sgd
            paired.append(dict(metric=metric, T=int(T), task=int(T)-1, n=3,
                mean_difference=float(diff.mean()), sample_sd_difference=float(diff.std(ddof=1)),
                seed0_difference=float(diff.iloc[0]), seed1_difference=float(diff.iloc[1]),
                seed2_difference=float(diff.iloc[2]), gam_lower=int((diff < 0).sum()),
                gam_higher=int((diff > 0).sum()), ties=int((diff == 0).sum())))
        for scope, p2 in [('T2-T10', p[p['T'] >= 2]), ('all_measured_stages', p)]:
            diff = p2.gam_minus_sgd
            counts.append(dict(metric=metric, scope=scope, paired_stage_seed_count=len(diff),
                independent_training_seed_count=3, gam_lower=int((diff < 0).sum()),
                gam_higher=int((diff > 0).sum()), ties=int((diff == 0).sum())))
            # Equal-weight stage mean per seed, then mean/SD over exactly three paired seeds.
            per_seed = p2.groupby('seed')[['sgd', 'gam', 'gam_minus_sgd']].mean()
            for col in ('sgd', 'gam', 'gam_minus_sgd'):
                vals = per_seed[col]
                stage_average.append(dict(metric=metric, scope=scope, quantity=col, n=3,
                    mean=float(vals.mean()), sample_sd=float(vals.std(ddof=1)),
                    seed0=float(vals.loc[0]), seed1=float(vals.loc[1]), seed2=float(vals.loc[2]),
                    aggregation='equal-weight stages within each seed, then three-seed mean/sample SD'))
        for method in ('sgd', 'gam'):
            means = frame[frame.method == method].groupby('T')[metric].mean()
            for (T0, v0), (T1, v1) in zip(list(means.items()), list(means.items())[1:]):
                changes.append(dict(metric=metric, method=method, from_T=int(T0), to_T=int(T1),
                                    before=float(v0), after=float(v1), change=float(v1-v0)))

    stats, pairs = pd.DataFrame(summary), pd.DataFrame(paired)
    plotted_means = pd.read_csv(FIG / 'B2_mean_sd.csv')
    for row in plotted_means.itertuples():
        new = stats[(stats.metric == row.metric) & (stats.method == row.method) & (stats['T'] == row.T)].iloc[0]
        assert np.isclose(new['mean'], row.mean, rtol=1e-10, atol=1e-12)
        assert np.isclose(new['sample_sd'], row.sd, rtol=1e-10, atol=1e-12) and row.n == 3
    stats.to_csv(OUT / 'B2_STAGE_MEAN_SD.csv', index=False)
    pairs.to_csv(OUT / 'B2_STAGE_PAIRED_DIFFERENCES.csv', index=False)
    pd.DataFrame(pair_raw).to_csv(OUT / 'B2_PAIRED_SEED_VALUES.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'B2_PAIR_COUNTS.csv', index=False)
    pd.DataFrame(stage_average).to_csv(OUT / 'B2_WITHIN_SEED_STAGE_AVERAGES.csv', index=False)
    pd.DataFrame(changes).to_csv(OUT / 'B2_ADJACENT_STAGE_CHANGES.csv', index=False)
    trajectory.to_csv(OUT / 'B2_ENDPOINT_EVIDENCE.csv', index=False)
    paths.to_csv(OUT / 'B2_PATH_EVIDENCE.csv', index=False)
    nodes.to_csv(OUT / 'B2_NODE_EVIDENCE.csv', index=False)
    pd.DataFrame(training).to_csv(OUT / 'B2_TRAINING_VALIDITY_LEDGER.csv', index=False)

    # Compare against the previous analysis while retaining the current figure as the target.
    previous = EXT / 'final_review_20260921/B2_STATISTICS.json'
    old = read(previous)
    for row in summary:
        if row['metric'] == 'accuracy_drop_from_anchor_pp':
            continue  # Newly added anchor-adjusted metric, absent from the old analysis.
        container = old['endpoint_statistics'] if row['metric'] in old['endpoint_statistics'] else old['path_statistics']
        old_row = next(r for r in container[row['metric']]['by_method_stage']
                       if r['method'] == row['method'] and r['T'] == row['T'])
        assert np.isclose(row['mean'], old_row['mean'], rtol=1e-10, atol=1e-12)
        assert np.isclose(row['sample_sd'], old_row['sample_sd'], rtol=1e-10, atol=1e-12)
    source_records = {r['path']: r for r in evidence['sources'] + extra_sources}
    for p in [Path(__file__), EXT / 'publish.py', previous, FIG / 'B2_caption.md',
              FIG / 'GENERATION.json', FIG / 'FINAL_MANIFEST.json',
              FIG / 'B2_mean_sd.csv', FIG / 'B2_trajectory_seed_data.csv', FIG / 'B2_path_seed_data.csv',
              FIG / 'Figure_B2_titled.pdf', FIG / 'Figure_B2_titled.png']:
        source_records[str(p)] = collector.source_record(p)
    pd.DataFrame(list(source_records.values())).to_csv(OUT / 'B2_SOURCE_LEDGER.csv', index=False)
    counterexamples = [r for r in pair_raw if r['metric'] == 'D_train' and r['gam_minus_sgd'] > 0]
    accuracy_drop_counterexamples = [r for r in pair_raw
        if r['metric'] == 'accuracy_drop_from_anchor_pp' and r['gam_minus_sgd'] > 0]
    assert len(counterexamples) == 5
    assert len(accuracy_drop_counterexamples) == 1
    assert original_hashes == {str(p): sha(p) for p in FIG.iterdir() if p.is_file()}

    validity = dict(status='passed', expected_seeds=[0, 1, 2], methods=LABELS,
        endpoints_expected=60, endpoints_validated=60, endpoints_missing=0, endpoints_duplicates=0,
        paths_expected=54, paths_validated=54, paths_missing=0, paths_duplicates=0,
        training_runs_expected=12, training_runs_validated=12, failed_training_runs=0,
        nonfinite_required_metrics=0, paths_reused=18, paths_new=36,
        nodes=502, nodes_by_path_count={'9': 52, '17': 2},
        total_gate_records=56, final_gates_passed=54, failed_intermediate_gates=failed_intermediate,
        all_intermediate_gates_passed=False,
        zero_semantics='T1 drift and signed test change are exact anchor-reference zeros; zero sample SD can be observed for discrete accuracy.',
        structural_nulls='S_path at T1 is undefined because drift=0; no T1 path record and no zero imputation.',
        checkpoint_payloads_rehashed=False,
        checkpoint_verification='Recorded checkpoint SHA matches raw states/manifests/endpoint recovery/training completion receipts; files exist. Large checkpoint payloads not reread or executed.',
        source_csv_matches_raw_recomputation=comparisons, plotted_means_and_sample_sd_match=True,
        original_figure_files_unchanged=True, previous_statistics_match=True,
        no_lambda20000_or_historical_imagenetr_cells=True)
    report = dict(status='passed', generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        target_figure=str(FIG / 'Figure_B2_titled.pdf'), panel_order=PANELS,
        cohort='ImageNet-A formal seeds 0,1,2; EWC lambda=2000', validity_ledger=validity,
        definitions=dict(T='T=t+1; task index is zero-based',
            accuracy='307 Task0 test examples; fixed original 20-class head; percent, differences in percentage points',
            accuracy_drop_from_anchor_pp='Own Task0 anchor accuracy minus current accuracy; positive means accuracy lost, negative means improvement',
            signed_test_damage='CE_test(W_t)-CE_test(W_0), each method and seed uses its own anchor; signed, no clipping',
            weight_drift='Euclidean norm of concatenated effective Q/V BA matrix displacement from own Task0 anchor',
            S_path='Simpson integral_0^1 Delta_t^T F_emp_train(W_0+s Delta_t) Delta_t ds / ||Delta_t||_2^2',
            fisher='1259 Task0 training examples; true-label per-example gradients; 24 effective Q/V matrices; no lambda multiplication, floor, regularization or trace normalization',
            normalization='Normalize each seed integral by its own drift squared, then average the three ratios',
            sample_sd='ddof=1 across 3 independent training seeds; no shading in current figure but SD retained in CSV',
            pairing='gam minus sgd at same seed and same T; 27 future-stage pairs are dependent repeated stages from three training seeds',
            paths='Separate Task0-to-endpoint straight chords; nodes are interpolation locations, not tasks or actual optimizer updates'),
        training_runs=training, statistics=summary, paired_by_stage=paired, paired_raw=pair_raw,
        pair_counts=counts, within_seed_stage_averages=stage_average, adjacent_stage_changes=changes,
        training_loss_counterexamples=counterexamples,
        accuracy_drop_counterexamples=accuracy_drop_counterexamples,
        final_gates=list(final_gates.values()), source_audit=evidence, source_ledger=list(source_records.values()),
        interpretation=dict(supported='Consistent same-stage association of better Task0 test retention, larger endpoint displacement and lower measured normalized directional path Fisher.',
            causal_limit='GAM remains enabled at Task1-9. B2 compares two complete training procedures and cannot isolate a Task0-only shaping causal effect.',
            metric_limit='Directional empirical-Fisher chord average is neither a Hessian eigenvalue nor endpoint Fisher trace; smaller S_path alone does not establish a global flatness or test-loss bound.',
            uncertainty='Only three independent training seeds. No claim of statistical significance from 27 dependent stage-seed pairs.',
            nonmonotonic='Accuracy, signed loss change and S_path are not strictly monotone; counterexamples and adjacent-stage changes are retained.',
            next_experiment='To isolate anchor shaping, fix later-task optimizer/training protocol across paired SGD/GAM anchors and preserve full-state/Fisher matching; this analysis launches no new runs.'))
    save_json('B2_ANALYSIS.json', report)

    def get(metric, method, T):
        return next(x for x in summary if (x['metric'], x['method'], x['T']) == (metric, method, T))
    def fmt(metric, method, T, digits=6):
        r = get(metric, method, T)
        return f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}"
    def pair(metric, T):
        return next(x for x in paired if (x['metric'], x['T']) == (metric, T))
    def pfmt(metric, T, digits=6):
        r = pair(metric, T)
        return f"{r['mean_difference']:+.{digits}f} ± {r['sample_sd_difference']:.{digits}f}"

    lines = [
        '# 当前 B2 四联图：逐阶段统计与来源核验', '',
        '本报告只分析 `figures_titled_20260922/Figure_B2_titled.pdf` 对应正式三 seed 数据；从原始 states.jsonl 重算，并核对当前 CSV。原图和实验数据未改动，未运行 GPU。', '',
        '**主要观察：**FS-EWCLoRA 在 T2–T10 的全部 27 个同 seed、同阶段配对中，旧任务测试准确率更高、测试损失增量更低、有效权重位移更大、归一化路径 Fisher 更低。T10 测试准确率提高 5.54 个百分点，而位移更大约 14.96%、S_path 更低约 93.66%。这描述的是两个完整训练流程的关联；GAM 在后续任务仍开启，不能把差异单独归因于 Task0 shaping。', '',
        '## 数据完整性与配置', '',
        '| 检查 | 结果 |', '|---|---|',
        '| 预期/有效端点 | 60/60 = 2 方法 × 3 seed × 10 阶段 |',
        '| 预期/有效路径 | 54/54 = 2 方法 × 3 seed × 9 后续阶段 |',
        '| 路径来源 | 18 条复用 t=1,5,9；36 条新增 t=2,3,4,6,7,8 |',
        '| 节点 | 502 = 52×9 + 2×17 |',
        '| 最终积分检查 | 54/54 通过；历史共56次检查，其中2次中间检查失败 |',
        '| anchor/sequence 训练 | 12/12 配置与完成回执一致，queue exit_code=0，receipt_valid=true |',
        '| 缺失/重复/非有限主指标 | 0/0/0 |',
        '| 与当前 CSV/mean-SD 表 | 原始端点、积分及所有已绘制均值/样本SD一致 |', '',
        '两次中间失败均为 seed2、Task8（T9）的 9 节点积分稳定性检查，分别属于 SGD/GAM；细化为17节点后通过。因此应写“所有最终检查通过”，不能写“所有中间检查均通过”。', '',
        '逐一核验三 seed 的 `anchor_sgd/gam` 与 `sequence_sgd/gam` 共12份 manifest：数据集为 ImageNet-A，EWC λ=2000，LoRA rank=10，batch size=128，Task0与后续任务各50 epochs，初始/后续学习率0.04/0.02。两臂使用同 seed 配对的 RNG 协议。`sequence_gam` 的 `optimizer_type=gam`、`optimizer_type_by_task=null`；Task1–9 的 RNG 回执每阶段均记录 GAM。λ=20000 新实验、参数扫描以及历史 ImageNet-R 结果均不在这些 CSV 的来源链中。', '',
        '来源核验覆盖 protocol hash、样本数、配置、state identity、checkpoint记录SHA、原点恢复、端点恢复、完成回执、Simpson积分及终态门槛。本次仅核对大 checkpoint 的既存SHA记录与存在性，没有重新读取约数十GB的模型负载；训练完成回执与诊断作业中记录的SHA一致。', '',
        '## 四幅图的量纲与统计口径', '',
        '从左到右是 Task0 测试准确率、有符号测试交叉熵变化、有效 Q/V 权重位移、归一化路径 Fisher。当前图只绘制三 seed 均值，省略SD阴影；CSV保存样本标准差，分母为 n−1=2。以下表中的 ± 均为样本SD，不是SEM或置信区间。配对差始终定义为同seed的 FS-EWCLoRA−EWCLoRA，再对3个差值计算均值和样本SD。', '',
        '- 横轴 T 是已学习任务数，内部零基任务编号 t=T−1。T1 是训练完Task0的原点；T10对应Task9。',
        '- 准确率与测试损失使用全部307张Task0测试图和固定Task0的20类头，不是累计200类准确率。准确率差的单位为百分点。',
        '- 测试损失变化是各方法相对自身Task0原点的有符号CE差；负数代表旧任务损失下降，不能截成0。不同方法原点CE不完全相同。',
        '- 位移是24个有效Q/V矩阵的拼接欧氏范数，Δ_t=W_t−W_0；它是原点到端点的距离，不是累计每一步更新长度。',
        '- 路径为 W_0+sΔ_t，s∈[0,1]。每个T有一条重新测量的原点到端点直弦；9或17个节点是这条弦上的位置，不是9或17个任务，也不是真实优化器trajectory。',
        '- F_emp使用1259张Task0训练图的true-label逐样本梯度外积，坐标为有效Q/V权重；不乘EWC λ，不做trace归一化，也不加floor/regularizer。', '',
        r'\[ I_{m,s,t}=\int_0^1\Delta_{m,s,t}^{\top}F^{\mathrm{emp}}_{0,\mathrm{train}}(W_{m,s,0}+u\Delta_{m,s,t})\Delta_{m,s,t}\,du,\quad S_{m,s,t}=I_{m,s,t}/\|\Delta_{m,s,t}\|_2^2,\quad \overline S_{m,t}=\tfrac13\sum_{s=0}^2 S_{m,s,t}. \]', '',
        '必须先逐seed归一化，再跨seed平均；平均I除以平均位移平方会得到不同量。T1分母为零，S_path未定义，数据中不补0。图题Average Fisher在这里指沿该弦、沿该位移方向的平均经验Fisher；它不是终点trace，也不是Hessian最大特征值。', '',
        '## T10：四项主指标', '',
        '| 指标 | EWCLoRA均值±SD | FS-EWCLoRA均值±SD | 配对差均值±SD | 同seed方向计数 |',
        '|---|---:|---:|---:|---|']
    for metric in PANELS:
        p = pair(metric, 10)
        lines.append(f"| {METRIC_NAMES[metric]} | {fmt(metric,'sgd',10)} | {fmt(metric,'gam',10)} | {pfmt(metric,10)} | 更高{p['gam_higher']}/3，更低{p['gam_lower']}/3 |")
    drift_pct = 100*(get('weight_drift','gam',10)['mean']/get('weight_drift','sgd',10)['mean']-1)
    sp_pct = 100*(1-get('S_path','gam',10)['mean']/get('S_path','sgd',10)['mean'])
    loss_pct = 100*(1-get('signed_test_damage','gam',10)['mean']/get('signed_test_damage','sgd',10)['mean'])
    lines += ['', f'末阶段方法均值相比：位移增加{drift_pct:.2f}%，S_path降低{sp_pct:.2f}%，正的测试损失增量降低{loss_pct:.2f}%。这些比例是两个方法均值的比值；不是各seed比例的平均。', '',
        f"末阶段绝对测试CE：EWCLoRA {fmt('test_CE','sgd',10)}，FS-EWCLoRA {fmt('test_CE','gam',10)}；配对差{pfmt('test_CE',10)}。这与有符号CE变化是两个不同指标。", '',
        f"控制各自Task0起点后，T10准确率掉点（自身原点准确率−T10准确率）：EWCLoRA为{fmt('accuracy_drop_from_anchor_pp','sgd',10)}个百分点，FS-EWCLoRA为{fmt('accuracy_drop_from_anchor_pp','gam',10)}个百分点，掉点的配对差为{pfmt('accuracy_drop_from_anchor_pp',10)}个百分点。Task0原点准确率均值仅相差{pair('test_accuracy_percent',1)['mean_difference']:.6f}个百分点，末端保留优势并非仅由这一起点均值差解释。", '',
        '## 全十阶段端点', '',
        '| T | EWCLoRA准确率(%) | FS-EWCLoRA准确率(%) | EWCLoRA测试CE变化 | FS-EWCLoRA测试CE变化 | EWCLoRA位移 | FS-EWCLoRA位移 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for T in range(1,11):
        vals = [fmt(metric,method,T) for metric in PANELS[:3] for method in ('sgd','gam')]
        lines.append(f'| {T} | ' + ' | '.join(vals) + ' |')
    lines += ['', '## S_path：全部九个后续阶段与同seed配对差', '',
        '| T | EWCLoRA均值±SD | FS-EWCLoRA均值±SD | 配对差均值±SD | 方法均值降低比例 |',
        '|---:|---:|---:|---:|---:|']
    for T in range(2,11):
        reduction = 100*(1-get('S_path','gam',T)['mean']/get('S_path','sgd',T)['mean'])
        lines.append(f"| {T} | {fmt('S_path','sgd',T)} | {fmt('S_path','gam',T)} | {pfmt('S_path',T)} | {reduction:.2f}% |")
    lines += ['', '## 27个后续阶段配对与反例', '',
        '| 指标 | FS更低 | FS更高 | 相等 | 比较数 |', '|---|---:|---:|---:|---:|']
    for metric in PANELS + ['D_train', 'integral', 'test_CE', 'accuracy_drop_from_anchor_pp']:
        c = next(r for r in counts if r['metric'] == metric and r['scope'] == 'T2-T10')
        lines.append(f"| {METRIC_NAMES[metric]} | {c['gam_lower']} | {c['gam_higher']} | {c['ties']} | {c['paired_stage_seed_count']} |")
    lines += ['', '**27=9阶段×3seed，只是重复测量的描述计数，独立训练重复仍是3，不能当作n=27计算显著性或置信区间。**若概括全程，本报告提供先对每seed的9阶段等权平均、再跨3seed统计的附加CSV；这也不是新的独立样本。', '',
        '计入Task0时，准确率为29/30对更高，唯一例外是seed0的Task0；原点位移与原点损失差按定义均为0。训练损失变化仅22/27对更低，5个反例为：', '',
        '| seed | T | FS−EWCLoRA训练CE变化 |', '|---:|---:|---:|']
    for row in counterexamples:
        lines.append(f"| {int(row['seed'])} | {int(row['T'])} | {row['gam_minus_sgd']:+.9f} |")
    ad = accuracy_drop_counterexamples[0]
    lines += ['', f"按相对自身原点的准确率掉点比较，FS在26/27对中更低；唯一例外为seed{int(ad['seed'])}、T{int(ad['T'])}，EWCLoRA掉点{ad['sgd']:.6f} pp、FS掉点{ad['gam']:.6f} pp（均为负数，即都提升，但前者提升更多），配对差{ad['gam_minus_sgd']:+.6f} pp。不能把原始准确率27/27更高改写成相对原点保持量也27/27更优。"]
    lines += ['', '训练损失差取1259张训练图的逐样本Fisher测量损失，与测试集CE不是同一个样本集合。已核验的数值界为训练损失变化平方 D_train²≤I（按协议数值容差检查）；它不直接约束图中的307张测试图损失，更不是测试损失的因果保证。更低S_path以及最终训练界检查通过，并不要求两方法每个训练损失差都按同一方向排序。', '',
        '## 非单调性与解释边界', '',
        '位移均值在T1–T10持续增加；准确率、测试损失变化、S_path的均值曲线均不能概括为严格单调。以下列出S_path/测试损失下降和准确率回升区段：', '',
        '| 指标 | 方法 | 相邻阶段区段 |', '|---|---|---|']
    for metric in ['test_accuracy_percent', 'signed_test_damage', 'S_path']:
        for method in ('sgd','gam'):
            wanted = [r for r in changes if r['metric'] == metric and r['method'] == method
                      and ((r['change'] > 0) if metric == 'test_accuracy_percent' else (r['change'] < 0))]
            text = '、'.join(f"{r['from_T']}→{r['to_T']}" for r in wanted) or '无'
            lines.append(f'| {METRIC_NAMES[metric]} | {LABELS[method]} | {text} |')
    lines += ['',
        '1. 观察：更大位移与更低方向敏感度同时出现，而且Task0测试保留较好。解释：这与沿较低敏感度方向适应新任务的机制相容。可支持的是该正式三seed流程的实测关联，不能仅由这些量推出全局平坦性或任意方向的鲁棒性。',
        '2. 因果边界：GAM在Task0和后续Task1–9都开启，因此同时改变原点与后续优化。若要隔离“仅Task0塑形”效应，需要同seed的不同原点配上完全相同的后续优化流程，并匹配完整权重/Fisher状态；当前报告不把已有B2解释为该对照。',
        '3. 统计边界：三seed足以报告观测一致性与种子波动，尚不足以用27个相关阶段值声称27次独立验证；本报告不报告伪独立显著性。',
        '4. 实验边界：B2关注旧任务在后续训练端点的保持，未单独量化所有新任务收益；不能替代B1的随机邻域容忍半径，也不能替代完整持续学习性能表。',
        '5. 下一步：如果论文需要Task0 shaping的因果结论，使用已有严格匹配的后续优化对照，或另行设计该对照；不要把λ=20000实验或历史ImageNet-R/256样本结果拼入当前估计。', '',
        '## 可追溯文件', '',
        '- `B2_ANALYSIS.json`：完整统计、定义、训练配置、最终与中间gate、原始来源SHA及解释边界。',
        '- `B2_STAGE_MEAN_SD.csv`：各阶段三seed均值、样本SD与逐seed值；`B2_STAGE_PAIRED_DIFFERENCES.csv`：同seed配对差均值/SD/方向计数。',
        '- `B2_PAIRED_SEED_VALUES.csv`、`B2_PAIR_COUNTS.csv`：全部逐seed差及描述计数；`B2_WITHIN_SEED_STAGE_AVERAGES.csv`：保留独立seed单位的全程概括。',
        '- `B2_ENDPOINT_EVIDENCE.csv`、`B2_PATH_EVIDENCE.csv`、`B2_NODE_EVIDENCE.csv`：60端点/54路径/502节点的原始值与来源。',
        '- `B2_TRAINING_VALIDITY_LEDGER.csv`、`B2_SOURCE_LEDGER.csv`：训练身份/exit/receipt与来源文件SHA；`B2_ADJACENT_STAGE_CHANGES.csv`：相邻阶段变化。',
        '- `analyze_b2_current.py`：CPU复算入口；其调用 `../../stage_path_completion_20260921/publish.py` 的只读collect函数，不调用绘图或训练。', '',
        '当前三个B2 CSV均与从原始states重算结果一致，所有均值/样本SD也与旧版B2_STATISTICS交叉核对一致；本报告补全当前panel顺序、T10配对差、配置因果边界，以及“中间两次失败、最终全部通过”的准确表述。']
    (OUT / 'B2_ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='passed', endpoints=60, paths=54, nodes=502,
        final_gates_passed=54, failed_intermediate_gates=2,
        t10=[r for r in paired if r['T'] == 10 and r['metric'] in PANELS],
        report=str(OUT / 'B2_ANALYSIS.md')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
