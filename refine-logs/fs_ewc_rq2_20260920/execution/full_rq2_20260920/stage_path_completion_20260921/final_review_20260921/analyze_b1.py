#!/usr/bin/env python3
"""Read-only audit of B1 measurements; writes only this directory's statistics."""
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics
import sys

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
BASE = EXT.parent
sys.path.insert(0, str(EXT))
import publish_radius as publisher


def summarize(values):
    return {"values": [float(v) for v in values], "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "n": len(values)}


def main():
    common, dense, sparse, new, audit = publisher.validate_all()
    paired = sparse.pivot(index=['seed', 'radius'], columns='method', values='raw_fisher_trace').reset_index()
    paired['gam_minus_sgd'] = paired['gam'] - paired['sgd']
    paired['gam_over_sgd'] = paired['gam'] / paired['sgd']
    summaries = []
    for radius, group in paired.groupby('radius'):
        sgd = summarize(group.sgd.tolist()); gam = summarize(group.gam.tolist())
        summaries.append(dict(radius=float(radius), sgd=sgd, gam=gam,
            paired_difference=summarize(group.gam_minus_sgd.tolist()),
            paired_ratio=summarize(group.gam_over_sgd.tolist()),
            ratio_of_means=gam['mean']/sgd['mean'],
            lower_seed_count=int((group.gam < group.sgd).sum())))
    new_pairs = paired[paired.radius.between(11, 19)]
    tolerance = []; spike = []; per_direction_pairs = []
    for seed in range(3):
        for method in ('sgd', 'gam'):
            directory = BASE / f'formal_results/seed{seed}/{method}/dense'
            summary = publisher.read(directory / 'summary.json')
            raw = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
            origin = next(r for r in raw if r['point']['radius'] == 0)
            grouped = defaultdict(list)
            for row in raw:
                grouped[row['point']['radius']].append(row)
            curves = {}
            for split in ('train', 'test'):
                curve = []
                for entry in summary['radius_summary']:
                    radius = entry['radius']
                    loss = math.fsum(max(r['metrics'][split]['loss'] - origin['metrics'][split]['loss'], 0)
                                     for r in grouped[radius]) / len(grouped[radius])
                    publisher.close(loss, entry[split]['G'], label='raw dense positive damage')
                    curve.append((radius, loss))
                curves[split] = curve
            for item in summary['tolerance']:
                if item['metric'] != 'G':
                    continue
                epsilon = item['epsilon']; radius = None
                for r, loss in curves[item['split']]:
                    if loss > epsilon:
                        break
                    radius = r
                publisher.require(radius == item['prefix_radius'], 'Tolerance differs from raw curve')
                tolerance.append(dict(seed=seed, method=method, **item))
            directory = EXT / f'results/seed{seed}/{method}/sparse_train'
            for row in map(json.loads, (directory / 'states.jsonl').read_text().splitlines()):
                if row['point']['radius']:
                    per_direction_pairs.append(dict(seed=seed, method=method, radius=row['point']['radius'],
                        direction=row['point']['direction']['index'], trace=row['fisher_by_scope']['subset']['raw_trace']))
                if seed == 0 and method == 'sgd' and row['point']['radius'] in (16,17,18) and row['point']['direction']['index'] == 14:
                    spike.append(row)
    raw_pairs = common.pd.DataFrame(per_direction_pairs).pivot(index=['seed','radius','direction'], columns='method', values='trace')
    tolerances = common.pd.DataFrame(tolerance)
    tol001 = tolerances[tolerances.epsilon == .01]
    tolerance_aggregate = []
    for split in ('train', 'test'):
        group = tol001[tol001.split == split].pivot(index='seed', columns='method', values='prefix_radius')
        tolerance_aggregate.append(dict(split=split, epsilon=.01, sgd=summarize(group.sgd.tolist()),
            gam=summarize(group.gam.tolist()), paired_difference=summarize((group.gam-group.sgd).tolist())))
    dense_endpoints = []
    dense_wins = {}
    for metric in ('G_train', 'train_accuracy_percent'):
        wide = dense.pivot(index=['seed','radius'], columns='method', values=metric).reset_index()
        nonzero = wide[wide.radius > 0]
        dense_wins[metric] = dict(total=len(nonzero), gam_lower=int((nonzero.gam < nonzero.sgd).sum()),
                                  gam_higher=int((nonzero.gam > nonzero.sgd).sum()))
        group = wide[wide.radius == 20]
        dense_endpoints.append(dict(radius=20, metric=metric, sgd=summarize(group.sgd.tolist()),
                                   gam=summarize(group.gam.tolist())))
    result = dict(status='passed', scope='B1 formal ImageNet-A seed0/1/2',
        validity=audit, fisher_aggregate=summaries,
        fisher_lower_counts=dict(all_seed_radius_pairs=int((paired.gam < paired.sgd).sum()),
            total_seed_radius_pairs=len(paired), new_seed_radius_pairs=int((new_pairs.gam < new_pairs.sgd).sum()),
            total_new_seed_radius_pairs=len(new_pairs),
            new_seed_radius_direction_pairs=int((raw_pairs.gam < raw_pairs.sgd).sum()),
            total_new_seed_radius_direction_pairs=len(raw_pairs)),
        dense_r20=dense_endpoints, dense_nonzero_comparisons=dense_wins,
        tolerance_epsilon_001=tolerance_aggregate, tolerance_per_seed=tolerance,
        anomaly=dict(kind='localized raw-gradient spike', seed=0, method='sgd', radius=17, direction=14,
            state='r17_u014', observations=spike,
            action='Retain observed point and full SD; no deletion or smoothing. Deterministic targeted repeat recommended.',
            caveat='Identity, restoration and summary checks pass; saved scalar logs cannot independently verify the per-example gradients.'),
        notes=['Raw Fisher is the true-label per-example CE gradient squared norm averaged over the fixed 256 training examples, with no EWC lambda or normalization.',
            'Per-seed direction means are averaged across three training seeds; SD is ddof=1, not a direction-level confidence interval.',
            'The dense loss/accuracy grid is unchanged: 81 radii, 100 directions, 1259 training examples.',
            'Fisher points have increased from 7 to 16; this does not change the dense tolerance curve.',
            'Training-side improvements do not establish a larger test protection ball.',
            'The r17 outlier precludes describing EWCLoRA Fisher as monotonic or uniformly smooth.'])
    verification_path = HERE / 'FISHER_PEAK_VERIFICATION.json'
    if verification_path.exists():
        verification = publisher.read(verification_path)
        publisher.require(verification['status'] == 'passed' and verification['identity_check']['passed'] and
                          verification['origin_check']['passed'], 'Peak verification identity/origin failed')
        publisher.require(verification['original_source_sha256'] == publisher.sha(verification['original_source']),
                          'Peak verification original source changed')
        observed = {row['point']['radius']: row['fisher_by_scope']['subset']['raw_trace'] for row in spike}
        for repeat in verification['repeated_radii']:
            publisher.require(repeat['passed'] and repeat['original_trace'] == observed[repeat['radius']] and
                              repeat['repeated_trace'] == repeat['original_trace'], 'Peak verification differs from original')
        sample_trace = verification['per_example_trace']
        publisher.require(len(sample_trace) == 256, 'Peak verification sample coverage changed')
        publisher.close(math.fsum(sample_trace)/256, observed[17], label='Per-example reconstruction of peak')
        collector = verification['single_example_collector_trace']
        direct = verification['single_example_direct_autograd_trace']
        publisher.close(collector, direct, rtol=1e-8, label='Independent autograd verification')
        publisher.require(verification['single_example_collector_matches_direct_autograd'] is True,
                          'Independent autograd verification failed')
        batch8 = verification['peak_sample_trace_batch8']
        result['anomaly'].update(status='independently_verified',
            verification_source=dict(path=str(verification_path),sha256=publisher.sha(verification_path)),
            repeated_radii=verification['repeated_radii'], peak_subset_index=verification['peak_subset_index'],
            peak_fraction_of_trace_sum=verification['peak_fraction_of_trace_sum'],
            single_example_collector_trace=collector, single_example_direct_autograd_trace=direct,
            collector_vs_direct_relative_difference=abs(collector-direct)/direct,
            batch8_peak_sample_trace=batch8, batch1_vs_batch8_relative_difference=abs(collector-batch8)/batch8,
            action='Verified original observation retained unchanged; repeat is supplementary validation, not a new formal observation.',
            caveat='Same batch-8 protocol repeats exactly. Batch-1 collector and direct autograd agree numerically; batch-1 and batch-8 values have a small floating-point difference and are not bitwise identical.')
    # One tidy CSV contains every plotted per-seed value, plus all tolerance outputs.
    output = []
    for row in sparse.to_dict('records'):
        output.append(dict(family='raw_fisher',seed=row['seed'],method=row['method'],radius=row['radius'],
            metric='raw_fisher_trace',value=row['raw_fisher_trace'],samples=256,directions=row['directions'],
            partition=row['source_partition']))
    for row in dense.to_dict('records'):
        for metric in ('G_train','train_accuracy_percent'):
            output.append(dict(family='dense_training',seed=row['seed'],method=row['method'],radius=row['radius'],
                metric=metric,value=row[metric],samples=1259,directions=row['directions'],partition='original_formal'))
    for row in tolerance:
        output.append(dict(family='tolerance',seed=row['seed'],method=row['method'],metric='prefix_radius',
            value=row['prefix_radius'],epsilon=row['epsilon'],split=row['split'],right_censored=row['right_censored'],
            partition='original_formal'))
    common.pd.DataFrame(output).to_csv(HERE / 'B1_STATISTICS.csv',index=False)
    (HERE / 'B1_STATISTICS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    display = lambda a: f"{a['mean']:.3f} ± {a['sample_sd']:.3f}"
    lines = ['# B1 正式补测核验与统计', '',
        '**6/6 作业完成并通过原始记录复算。** 新增 11–19 共 9 个半径，每个非零半径 16 个公共方向、256 张固定旧任务训练子集；3 seed × 2 方法共 864 个新增扰动状态，另有 6 个重复原点，共核验 870 条新记录。', '',
        '原有 dense 与 Fisher 原始记录另外复算 48,588 条。扩展仅改变输出目录和半径网格；原点、checkpoint/头/数据/坐标身份以及方向哈希均与同一正式批次一致，没有混入历史 seed0。物理 checkpoint 未再次全文件哈希；身份核查对照此前审计的 manifest 和日志。', '',
        '合并后 Fisher 半径为 0、0.5、1、2、5、10、11、12、13、14、15、16、17、18、19、20，共 16 个实测半径。原点仅计一次。所有表格为先在每个 seed 内对方向平均，再跨 3 seed 取均值 ± 样本标准差。', '',
        '| 半径 | EWCLoRA raw trace | FS-EWCLoRA raw trace | FS/EWC 均值比 |', '|---:|---:|---:|---:|']
    for entry in summaries:
        if entry['radius'] in (0,10,15,17,20):
            lines.append(f"| {entry['radius']:g} | {display(entry['sgd'])} | {display(entry['gam'])} | {entry['ratio_of_means']:.3f} |")
    counts=result['fisher_lower_counts']
    lines += ['',f"FS-EWCLoRA 的方向平均 trace 在新增 **{counts['new_seed_radius_pairs']}/{counts['total_new_seed_radius_pairs']}** 个 seed×半径对和合并后 **{counts['all_seed_radius_pairs']}/{counts['total_seed_radius_pairs']}** 个 seed×半径对中较低。进一步逐方向比较，新增 **{counts['new_seed_radius_direction_pairs']}/{counts['total_new_seed_radius_direction_pairs']}** 对较低；这些方向不能作为独立训练重复。",'',
        '**r=17 有实测局部尖峰，保留原值。** EWCLoRA seed0 的方向 u014 的 trace 在 r=16/17/18 分别为 3908.656、68312.656、3181.817；r=17 的 seed0 方向均值因此为 6380.277。该点有限、逐点锚点恢复和方向/模型身份检查通过，原始值与汇总吻合；对应训练损失 0.125447→0.130118→0.135190 平滑变化。不能据此声称 EWCLoRA Fisher 单调或平滑增长。不得删除该点来改善曲线。', '',
        '**损伤/准确率数据没有变化。** 前两图仍为完整 1259 张训练图、100 个方向、81 个半径（0–20 步长 0.25），不以新增 16 方向测量替换原来的 100 方向均值。', '',
        '| r=20 指标 | EWCLoRA | FS-EWCLoRA |','|---|---:|---:|']
    for entry in dense_endpoints:
        label='训练平均正损伤' if entry['metric']=='G_train' else '训练准确率（%）'
        fmt=lambda a: f"{a['mean']:.6f} ± {a['sample_sd']:.6f}"
        lines.append(f"| {label} | {fmt(entry['sgd'])} | {fmt(entry['gam'])} |")
    lines += ['', '**容忍半径没有因 Fisher 加点而改变。** 它仍从 81 点的平均正损伤 G(r) 计算：按半径递增遇到第一处 G(r)>ε 时停止，返回此前最大的已测半径；不插值。ε=0.01 时：', '',
        '| 数据范围 | EWCLoRA（seed0/1/2） | FS-EWCLoRA（seed0/1/2） | 配对平均差 |','|---|---|---|---:|']
    for entry in tolerance_aggregate:
        vals=lambda a:'/'.join(f'{v:g}' for v in a['values'])
        lines.append(f"| {'训练' if entry['split']=='train' else '测试'} | {vals(entry['sgd'])} | {vals(entry['gam'])} | {entry['paired_difference']['mean']:+.4f} |")
    lines += ['', 'raw Fisher 是固定旧任务标签下逐样本 CE 梯度平方范数的平均，不含 λ=2000、归一化或正则项。结果支持训练侧较低敏感度和更低随机扰动平均损伤；不能把训练结论外推为测试保护球扩大。有限方向、有限半径下的平均正损伤容忍半径也不是球内所有方向安全的保证。', '',
        '复现：`python analyze_b1.py`。该脚本只读原始日志与代码，写本目录 `B1_STATISTICS.{json,csv,md}`，不改动发布回执、原图、原始结果或队列状态。']
    if result['anomaly'].get('status') == 'independently_verified':
        index = next(i for i, line in enumerate(lines) if line.startswith('**r=17')) + 1
        lines[index:index] = ['', '**尖峰已通过独立复测与梯度核验。** 同一 batch=8 配置在 GPU 6 重测 r16/r17/r18，与原始 trace 逐值完全一致；r17 保存了全部 256 个样本的 trace。训练子集索引 102 的样本贡献 trace 总和的 96.4603%。对该样本单独使用 batch=1，collector 得到 16870873.60344，直接 `torch.autograd.grad` 平方和为 16870873.61554，相对差约 7.17×10⁻¹⁰。原 batch=8 中该样本 trace 为 16869008.93594，与 batch=1 存在约 0.0111% 的浮点差异，不能称不同 batch 或独立求导逐位相同。确认原尖峰可复现，原始观测、统计和图中数值均保留；复测只作验证，不添加为新的正式样本。证据见 [FISHER_PEAK_VERIFICATION.json](FISHER_PEAK_VERIFICATION.json)。']
    (HERE / 'B1_STATISTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],counts=counts,selected=[x for x in summaries if x['radius'] in (10,15,17,20)],dense_r20=dense_endpoints),indent=2))


if __name__ == '__main__':
    main()
