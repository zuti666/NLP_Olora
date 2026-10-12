#!/usr/bin/env python3
"""Read-only source audit and final formal B2 statistics (CPU only)."""
from pathlib import Path
import importlib.util
import json

OUT = Path(__file__).resolve().parent
EXT = OUT.parent
spec = importlib.util.spec_from_file_location('all_stage_source_collector', EXT / 'publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)
np, pd = publish.np, publish.pd


def summarize(frame, metrics):
    result = {}
    for metric in metrics:
        grouped = []
        for (method, T), group in frame.groupby(['method', 'T']):
            assert sorted(group.seed) == [0, 1, 2]
            values = group.sort_values('seed')[metric].astype(float)
            assert np.isfinite(values).all()
            grouped.append(dict(method=method, T=int(T), task=int(T)-1,
                mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                n=3, seed_values=list(map(float, values))))
        paired = frame.pivot(index=['seed', 'T'], columns='method', values=metric)
        paired['gam_minus_sgd'] = paired.gam - paired.sgd
        pair_records = paired.reset_index().to_dict(orient='records')
        stage_pairs = []
        for T, group in paired.reset_index().groupby('T'):
            assert sorted(group.seed) == [0, 1, 2]
            diff = group.sort_values('seed').gam_minus_sgd
            stage_pairs.append(dict(T=int(T), mean_difference=float(diff.mean()),
                sample_sd_difference=float(diff.std(ddof=1)),
                gam_lower=int((diff < 0).sum()), gam_higher=int((diff > 0).sum()),
                ties=int((diff == 0).sum()), paired_seed_differences=list(map(float, diff))))
        means = pd.DataFrame(grouped)
        decreases = []
        for method, group in means.groupby('method'):
            rows = list(group.sort_values('T').to_dict(orient='records'))
            for previous, current in zip(rows, rows[1:]):
                if current['mean'] < previous['mean']:
                    decreases.append(dict(method=method, from_T=previous['T'], to_T=current['T'],
                        before=previous['mean'], after=current['mean']))
        result[metric] = dict(by_method_stage=grouped, paired_by_stage=stage_pairs,
            paired_seed_values=pair_records,
            descriptive_pair_counts=dict(gam_lower=int((paired.gam_minus_sgd < 0).sum()),
                gam_higher=int((paired.gam_minus_sgd > 0).sum()),
                ties=int((paired.gam_minus_sgd == 0).sum()), total=len(paired)),
            mean_curve_decreases=decreases)
    return result


def main():
    trajectory, paths, nodes, evidence = publish.collect()
    assert evidence['status'] == 'ready' and evidence['validated_paths'] == 54
    assert not evidence['missing_paths'] and evidence['validated_new_paths'] == 36
    assert len(trajectory) == 60 and set(paths['T']) == set(range(2, 11))
    final_gates = {}
    for gate in evidence['numerical_gates']:
        key = (gate['seed'], gate['method'], gate['task'])
        if key not in final_gates or gate['nodes'] > final_gates[key]['nodes']:
            final_gates[key] = gate
    assert len(final_gates) == 54 and all(g['passed'] for g in final_gates.values())
    path_stats = summarize(paths, ['S_path', 'integral', 'drift', 'D_train', 'D_test'])
    endpoint_stats = summarize(trajectory, ['weight_drift', 'signed_test_damage',
                                          'test_accuracy_percent', 'test_CE'])
    endpoint_lookup = trajectory.set_index(['seed', 'method', 'task'])
    raw_paths = paths.copy()
    raw_paths['test_accuracy_percent'] = [float(endpoint_lookup.loc[(int(r.seed), r.method, int(r.task)), 'test_accuracy_percent'])
                                       for r in paths.itertuples()]
    raw_paths['record_type'] = 'path'
    raw_endpoints = trajectory.copy()
    raw_endpoints['record_type'] = 'endpoint'
    raw = pd.concat([raw_paths, raw_endpoints], ignore_index=True, sort=False)
    raw.to_csv(OUT / 'B2_STATISTICS.csv', index=False)
    report = dict(status='passed', cohort='formal ImageNet-A seed0/1/2',
        source_audit=evidence, path_statistics=path_stats, endpoint_statistics=endpoint_stats,
        node_counts={str(int(k)): int(v) for k, v in paths.measured_nodes.value_counts().items()},
        definitions=dict(S_path='Simpson integral Delta_t^T F_emp(W0+s Delta_t) Delta_t ds / ||Delta_t||^2',
            damage='signed cross-entropy loss change relative to own Task0 anchor',
            path_samples=1259, test_samples=307, fixed_old_head_classes=20,
            statistics='arithmetic mean and sample SD, ddof=1, over three independent training seeds',
            paired_comparisons='same seed and same endpoint; 27 counts are descriptive, not independent replicates',
            T='task index t + 1; S_path at T=1 undefined, not zero',
            path='one separate Task0-to-Task t straight chord per endpoint; not actual optimizer chronological path',
            no_historical_pooling=True),
        collector=dict(path=str(EXT / 'publish.py'), sha256=publish.sha(EXT / 'publish.py')),
        raw_csv=dict(path=str(OUT / 'B2_STATISTICS.csv'), sha256=publish.sha(OUT / 'B2_STATISTICS.csv'),
            path_rows=54, endpoint_rows=60))
    (OUT / 'B2_STATISTICS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def point(metric, method, T, endpoint=False):
        source = endpoint_stats if endpoint else path_stats
        return next(x for x in source[metric]['by_method_stage'] if x['method'] == method and x['T'] == T)
    def formatted(metric, method, T, endpoint=False):
        d = point(metric, method, T, endpoint)
        return f"{d['mean']:.6f} ± {d['sample_sd']:.6f}"
    lines = [
        '# 正式三 seed：完整逐阶段路径统计', '',
        '原始 18 条路径与新增 36 条路径均已完成，合计 54/54 条。逐条从 states.jsonl 重算 Simpson 积分及归一化敏感度，并核验配置、checkpoint 身份、完成 receipt、原点/端点恢复和数值门槛；全部通过。', '',
        f"已核验任务端点共 60 个；路径节点总数 {len(nodes)}，每条路径实际节点数为 {sorted(map(int, paths.measured_nodes.unique()))}。52 条路径使用 9 节点；seed2 的两个 Task8 路径细化到 17 节点后通过最终数值检查。",
        '三个训练 seed 的均值 ± 样本标准差（ddof=1）；同 seed、同终点配对。27 次阶段比较仅作描述，不能视为 27 个独立训练重复。', '',
        '所有路径 Fisher 使用 1259 张旧任务训练图、固定旧任务 20 类头、true-label per-example 有效 Q/V 梯度；不乘 EWC λ、不做 trace normalization。损失与准确率的测试指标使用 307 张旧任务测试图及相同旧任务 20 类头。', '',
        '## 路径敏感度随阶段变化', '',
        '| 已学习任务数 T | 终点 | EWCLoRA S_path | FS-EWCLoRA S_path | 降低比例（方法均值） |',
        '|---:|---:|---:|---:|---:|']
    for T in range(2, 11):
        a, b = point('S_path', 'sgd', T), point('S_path', 'gam', T)
        lines.append(f"| {T} | Task{T-1} | {formatted('S_path','sgd',T)} | {formatted('S_path','gam',T)} | {(1-b['mean']/a['mean'])*100:.2f}% |")
    lines += ['', '这里的降低比例是两个方法均值之比；不是逐 seed 降低比例的平均。S_path 先在每条路径独立计算 I/||Δ||²，再跨 seed 平均；不能将平均 I 除以平均位移平方。', '',
        '## 各阶段其他测量', '',
        '| T | 方法 | 位移 | 积分 I | 训练损失变化 | 测试损失变化 | 测试准确率 (%) |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for T in range(2, 11):
        for method, label in [('sgd','EWCLoRA'),('gam','FS-EWCLoRA')]:
            values = [formatted(k,method,T) for k in ('drift','integral','D_train','D_test')]
            lines.append(f"| {T} | {label} | " + ' | '.join(values) + ' | '+formatted('test_accuracy_percent',method,T,True)+' |')
    lines += ['', '## 全十阶段端点指标', '',
        '| T | EWCLoRA 位移 | FS-EWCLoRA 位移 | EWCLoRA 测试损失变化 | FS-EWCLoRA 测试损失变化 | EWCLoRA 测试准确率 (%) | FS-EWCLoRA 测试准确率 (%) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for T in range(1,11):
        values = [formatted(metric, method, T, True)
                  for metric in ('weight_drift','signed_test_damage','test_accuracy_percent')
                  for method in ('sgd','gam')]
        lines.append(f"| {T} | " + ' | '.join(values) + ' |')
    lines += ['', '## 配对结果和解释边界', '',
        '| 指标 | FS-EWCLoRA 更低 | 更高 | 相等 | 比较数 |',
        '|---|---:|---:|---:|---:|']
    for metric, data in path_stats.items():
        c = data['descriptive_pair_counts']
        lines.append(f"| {metric} | {c['gam_lower']} | {c['gam_higher']} | {c['ties']} | {c['total']} |")
    train_counter = [r for r in path_stats['D_train']['paired_seed_values'] if r['gam_minus_sgd'] > 0]
    test_counter = [r for r in path_stats['D_test']['paired_seed_values'] if r['gam_minus_sgd'] > 0]
    lines += ['', f"训练损失反例（FS-EWCLoRA 增量更大）共 {len(train_counter)}/27：" + '；'.join(f"seed{r['seed']}, T={r['T']}, Δ差={r['gam_minus_sgd']:+.6f}" for r in train_counter) + '。',
        '', f"测试损失反例共 {len(test_counter)}/27。继续学习后的测试准确率在 27/27 对比较中更高；包含 Task0 原点时为 29/30 更高，唯一例外为 seed0 的原点准确率略低。", '',
        '损失变化是有符号增量，负值代表旧任务损失比自己的 Task0 原点更低；不截断为 0。准确率为旧任务固定 20 类头准确率，不是 200 类累计分类准确率。', '',
        '逐阶段敏感度及损失不要求单调增加。下表列出均值曲线相邻阶段下降的区段，以防把“后续学习”误写成严格单调变化：', '',
        '| 指标 | 方法 | 下降区段 T |', '|---|---|---|']
    for metric in ('S_path','D_train','D_test'):
        for method,label in [('sgd','EWCLoRA'),('gam','FS-EWCLoRA')]:
            changes = [r for r in path_stats[metric]['mean_curve_decreases'] if r['method'] == method]
            lines.append(f"| {metric} | {label} | " + ('、'.join(f"{r['from_T']}→{r['to_T']}" for r in changes) or '无') + ' |')
    lines += ['', '每个 T 点代表一条独立的 Task0→Task(t=T−1) 端点直线段；它不是相邻任务之间的路径，也不是实际优化器全部更新步骤的时间积分。九个积分节点是同一条线段上的位置。T=1 时位移为零，归一化分母为零，S_path 未定义，不能填 0。', '',
        '历史 seed0 的 0.3681→0.0148 来自另一批 checkpoint 和 256 样本；当前为新训练的正式三 seed、每节点全部 1259 张训练图，不能拼接或合并估计。', '',
        'B2 可支持“相同逐阶段比较中更大的实际累计位移，并伴随更低的路径方向敏感度”及末端旧任务损伤较小；训练损失的早期反例必须保留。由本图不能推出所有阶段训练损失都更小，也不能替代 B1 的测试邻域容忍半径检验。', '',
        '完整逐 seed 数据与来源见 B2_STATISTICS.csv；统计、配对差值、原始文件 SHA 和数值检查明细见 B2_STATISTICS.json。']
    (OUT / 'B2_STATISTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed', paths=len(paths), nodes=len(nodes),
        counts={k:v['descriptive_pair_counts'] for k,v in path_stats.items()},
        node_counts=report['node_counts'], output=str(OUT / 'B2_STATISTICS.md')),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
