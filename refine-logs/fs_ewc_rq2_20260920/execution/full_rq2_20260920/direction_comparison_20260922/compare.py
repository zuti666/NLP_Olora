"""Compare completed direction diagnostics; no GPU inference or source edits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ARMS = [
    ('random', 'radius50_extension_20260921', 'radius50_complete.json', 12),
    ('task0_ascent', 'fixed_gradient_extension_20260921', 'fixed_gradient_complete.json', 6),
    ('task1_descent', 'task1_gradient_extension_20260921', 'task1_gradient_complete.json', 6),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ledger, source_files, parts = [], [], []
    for arm, directory, receipt_name, expected in ARMS:
        root = BASE / directory
        receipt_path = root / 'report' / receipt_name
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'complete'
        assert receipt['expected_jobs'] == receipt['validated_jobs'] == expected
        assert receipt['automatic_numeric_and_layout_checks'] == 'passed'
        hashes = {i['path']: i['sha256'] for i in receipt['output_files']}
        ledger.append(dict(arm=arm, validated=expected, expected=expected, failed=0, missing=0))
        source_files.append(dict(path=str(receipt_path), sha256=sha(receipt_path)))
        if arm == 'random':
            files = [
                ('figures/radius50/dense_training_per_seed.csv', 522,
                 {'G_train': 'positive_ce_damage', 'train_accuracy_percent': 'accuracy_percent'}),
                ('figures/radius50/fisher_training_per_seed.csv', 132,
                 {'raw_fisher_trace': 'raw_fisher_trace'}),
            ]
        else:
            files = [('figures/all_per_seed.csv', 558,
                      {'G_positive_train': 'positive_ce_damage',
                       'train_accuracy_percent': 'accuracy_percent',
                       'raw_fisher_trace': 'raw_fisher_trace'})]
        for relative, count, metrics in files:
            path = root / relative
            assert sha(path) == hashes[str(path)]
            frame = pd.read_csv(path)
            assert len(frame) == count and not frame.duplicated(['seed', 'method', 'radius']).any()
            if arm != 'random':
                assert (frame.D_train >= 0).all()
                np.testing.assert_allclose(frame.D_train, frame.G_positive_train, rtol=0, atol=0)
            for original, metric in metrics.items():
                assert np.isfinite(frame[original]).all()
                part = frame[['seed', 'method', 'radius', 'directions', original]].rename(columns={original: 'value'})
                part['arm'], part['metric'] = arm, metric
                parts.append(part)
            source_files.append(dict(path=str(path), sha256=sha(path)))
    data = pd.concat(parts, ignore_index=True)
    rows = []
    for (arm, metric, radius), group in data.groupby(['arm', 'metric', 'radius']):
        pivot = group.pivot(index='seed', columns='method', values='value').sort_index()
        assert pivot.index.tolist() == [0, 1, 2] and set(pivot.columns) == {'sgd', 'gam'}
        delta = pivot.gam - pivot.sgd
        improvement = delta > 0 if metric == 'accuracy_percent' else delta < 0
        rows.append(dict(arm=arm, metric=metric, radius=radius, paired_n=3,
                         EWCLoRA_mean=pivot.sgd.mean(), FS_EWCLoRA_mean=pivot.gam.mean(),
                         paired_delta_mean=delta.mean(), paired_delta_sample_sd=delta.std(ddof=1),
                         FS_better_seeds=int(improvement.sum()),
                         delta_seed0=delta.loc[0], delta_seed1=delta.loc[1], delta_seed2=delta.loc[2]))
    summary = pd.DataFrame(rows)
    data.to_csv(OUT / 'per_seed.csv', index=False)
    summary.to_csv(OUT / 'paired_summary.csv', index=False)
    selected = summary[summary.radius.isin([.1, .25, .5, 1, 2, 10, 20, 50])]
    selected.to_csv(OUT / 'selected_radii.csv', index=False)
    report = ['# 三种方向诊断的配对比较', '',
              '读取已完成数据，不运行新测量。比较对象为正式 ImageNet-A、ViT-B/16、rank-10 Q/V LoRA，seed 0/1/2。', '',
              '| 设置 | 验证/预期作业 | 失败 | 缺失 |', '|---|---:|---:|---:|']
    report += [f"| {x['arm']} | {x['validated']}/{x['expected']} | 0 | 0 |" for x in ledger]
    report += ['', '随机扩展的12作业是6组×损失/准确率及Fisher两类；合并了原正式测量、11–19 Fisher补点与25–50扩展。每个固定方向实验6组×93半径=558状态。', '',
               '三者均在各自Task0锚点的相同有效Q/V坐标中扫描 theta0+r*u，r为全坐标联合L2范数。旧任务CE/准确率使用全部1259训练图和原20类头；Fisher使用固定256旧训练图。', '',
               '| 设置 | 方向 | 每半径方向数 | 网格 |', '|---|---|---|---|',
               '| random | 固定公共随机单位方向；两方法配对 | 损失/准确率100，Fisher为前16个 | 损失87半径；Fisher22半径 |',
               '| task0_ascent | 各自原点的 +grad L0 / norm(grad L0)，只计算一次 | 1 | 93半径 |',
               '| task1_descent | 各自原点的 -grad L1 / norm(grad L1)，只计算一次 | 1 | 93半径 |', '',
               'Task1方向使用全部997新任务训练图和精确复原的未训练新20类头；测量仍用旧Task0头。两个固定方向均是纯CE梯度，不含GAM/EWC步修正。', '',
               '随机图画的是 G(r)=mean_k max(L0(theta0+r*u_k)-L0(theta0),0)，其中L0先在旧训练样本上平均。固定方向图画的是单方向有符号增量D；全部已测训练D均非负，故此处数值也等于K=1的正损伤。不能把随机G改称有符号平均损失变化：随机平均与定向单线是不同的方向采样对象。', '',
               '例如随机r=1，有符号平均增量为3.01949e-5 → 3.22450e-5（FS稍高），而平均正损伤为1.25115e-4 → 9.48268e-5（FS更低）。全部3seed×100方向中，两方法分别有135/125个负增量，取正值会影响排序。r=20和50所有原始增量为正，故这两点的G与有符号均值相等。', '',
               '额外独立原始重算见 random_raw_per_seed.csv（522行）和 random_raw_validation.json。随机有符号均值在82/86个非零半径上FS更低，在0.25、0.5、0.75、1这4个半径上FS更高；正损伤G则在全部86个非零半径均更低。前述数量描述已测网格，不是统计显著性。', '',
               'Fisher = mean_i ||grad CE_i(theta0+r*u)||^2；随机曲线再对16个方向平均。不乘lambda，无额外归一化；它不是单一方向的u^T F u。', '',
               '以下单元格均为三seed均值 EWCLoRA → FS-EWCLoRA；完整配对差值、样本标准差与逐seed差值见CSV。', '']
    for metric in ['positive_ce_damage', 'accuracy_percent', 'raw_fisher_trace']:
        report += [f'## {metric}', '', '| r | random | task0_ascent | task1_descent |', '|---:|---:|---:|---:|']
        for radius in [1, 2, 20, 50]:
            cells = []
            for arm, *_ in ARMS:
                row = summary[(summary.arm == arm) & (summary.metric == metric) & (summary.radius == radius)].iloc[0]
                cells.append(f'{row.EWCLoRA_mean:.6g} → {row.FS_EWCLoRA_mean:.6g}')
            report.append(f"| {radius} | {' | '.join(cells)} |")
        report.append('')
    report += ['## 配对统计与解释', '',
               '独立训练重复数为3；方向不是额外训练重复，跨半径计数也不是统计显著性或独立胜率。r=1的损伤配对差值定义为FS−EWC：', '',
               '| 设置 | 均值 ± 样本SD | seed0 | seed1 | seed2 | FS损伤更低的seed数 |', '|---|---:|---:|---:|---:|---:|']
    for arm, *_ in ARMS:
        row = summary[(summary.arm == arm) & (summary.metric == 'positive_ce_damage') & (summary.radius == 1)].iloc[0]
        report.append(f'| {arm} | {row.paired_delta_mean:.6g} ± {row.paired_delta_sample_sd:.6g} | {row.delta_seed0:.6g} | {row.delta_seed1:.6g} | {row.delta_seed2:.6g} | {row.FS_better_seeds}/3 |')
    report += ['',
               '1. 随机训练诊断：86个非零损失半径上三个seed均为FS损伤更低、准确率更高；21个非零Fisher半径上三个seed均为FS trace更低。这是所测随机平均结果，不是每一个随机方向都改善。',
               '2. Task0上升方向：FS在48/92个非零半径的平均损伤更低、44/92更高；例如r=1更高、r=20更低。r=50差值−0.105962但配对SD为1.920220，不能称为稳定优势。',
               '3. Task1下降方向：FS只在7个已测小半径的平均损伤更低，全部已测r>=0.5的85个半径平均损伤更高；并非每个seed在每个半径均更高。',
               '4. 高维各向同性随机方向的一阶投影通常小；Task0正梯度方向最大化原点单位位移的一阶旧损失增量（不是有限半径精确最坏方向）。Task1方向反映新目标的初始方向，不是完整训练轨迹。',
               '5. Fisher trace较低不足以保证旧损伤较低。例如Task0方向r=1，FS trace更低但损伤更高。它是新位置的局部梯度能量，不等于原点到新位置的损失差，也不必单调。',
               '6. 两方法各自的梯度方向不同，因此固定方向对比同时包含锚点性质和方向选择影响。若进一步分离两者，可预先固定同一方向并在两个锚点交叉测量；本次未启动新实验。',
               '7. 准确率含原点差异：EWCLoRA/FS原点训练准确率为97.987821%/99.152767%；应结合相对锚点损伤判断。当前比较仅是训练侧，不能据此推出测试保护区域扩大。', '',
               '**可支持的表述：FS-EWCLoRA在所测训练随机邻域中降低平均正损伤及平均Fisher trace，但该优势具有方向依赖，未普遍延伸至Task0损失上升或Task1初始下降方向。实际持续学习的结论仍以真实逐任务轨迹为依据。**', '']
    (OUT / 'COMPARISON.md').write_text('\n'.join(report))
    validation = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
                      ledger=ledger, paired_n=3, sources=source_files,
                      fixed_signed_and_positive_train_damage_identical=True,
                      rows=len(data), summary_rows=len(summary))
    (OUT / 'VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation['ledger']))
    print(OUT / 'COMPARISON.md')


if __name__ == '__main__':
    main()
