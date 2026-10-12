import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/140-1/users/liying/O-LoRA')
SRC = ROOT / 'refine-logs/imageneta-gam-cumulative-results-20261007T065249Z'
OUT = ROOT / 'refine-logs/imageneta-gradient-tables-20261007'
OUT.mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

rows = read_csv(SRC / 'ALL_UNIQUE_10TASK_RESULTS.csv')
aggregates = read_csv(SRC / 'THREE_SEED_COMPLETE_32_CONDITIONS.csv')
agg = {r['condition']: r for r in aggregates}
audits = []
for r in rows:
    raw = json.loads(Path(r['source_matrix']).read_text())
    matrix = raw['matrix'] if 'matrix' in raw else raw['cnn']['matrices']['t09']
    assert len(matrix) == 10
    matrix = [[float(v) for v in rr[:i+1]] for i, rr in enumerate(matrix)]
    assert all(len(rr) == i+1 for i, rr in enumerate(matrix))
    assert all(math.isfinite(v) and 0 <= v <= 100 for rr in matrix for v in rr)
    metrics = {
        'FAA': statistics.mean(matrix[-1]),
        'AAA': statistics.mean(statistics.mean(rr) for rr in matrix),
        'LA': statistics.mean(matrix[i][i] for i in range(10)),
        'BWT': statistics.mean(matrix[9][i]-matrix[i][i] for i in range(9)),
        'forgetting': statistics.mean(max(matrix[k][i] for k in range(i, 10))-matrix[9][i] for i in range(9)),
    }
    for key, value in metrics.items():
        assert abs(value-float(r[key])) < 1e-8, (r['job'], key, value, r[key])
    assert abs(metrics['FAA']-metrics['LA']-.9*metrics['BWT']) < 1e-8
    assert int(r['tasks']) == 10 and int(r['epochs']) == 500 and int(r['updates']) == 2550
    assert int(r['rank']) == 10 and float(r['rho']) == .05
    receipt = json.loads(Path(r['source_receipt']).read_text())
    if 'exit' in receipt:
        assert receipt['exit']['returncode'] == 0
    if 'status' in receipt:
        assert receipt['status'] == 'completed', (r['job'], receipt['status'])
    if 'passed' in receipt:
        assert receipt['passed'] is True
    hashes = {k: sha(r[k]) for k in ['source_config', 'source_matrix', 'source_summary', 'source_receipt']}
    for field, index_key in [('matrix_sha256', 'source_matrix'), ('summary_sha256', 'source_summary')]:
        if field in receipt:
            assert receipt[field] == hashes[index_key], (r['job'], field)
    r.update({k: str(v) for k, v in metrics.items()})
    config_receipt_check = 'not_declared'
    if 'config_sha256' in receipt:
        launch_config = Path(receipt.get('config', r['source_config']))
        if sha(launch_config) != receipt['config_sha256']:
            run_root = Path(r['source_receipt']).parents[3]
            candidates = [run_root/'configs'/f"{r['job']}.json", run_root/'configs'/f"{r['job']}_config.json"]
            matched = [p for p in candidates if p.exists() and sha(p) == receipt['config_sha256']]
            config_receipt_check = str(matched[0]) if matched else 'effective_config_hash_differs_from_launch_receipt; recorded, not treated as zero or metric failure'
        else:
            config_receipt_check = str(launch_config)
    audits.append({'job': r['job'], 'receipt': r['source_receipt'], 'metrics_recomputed': metrics, 'hashes': hashes, 'launch_config_receipt_check': config_receipt_check})
assert len({r['source_receipt'] for r in rows}) == len(rows) == 144

# Recompute every declared mean and sample SD from its exact seed cohort.
for a in aggregates:
    assert a['expected_seeds'] == a['completed_seeds'] == '0;1;2'
    assert int(a['n_expected']) == int(a['n_complete']) == 3
    for metric in ['FAA', 'AAA', 'LA', 'BWT', 'forgetting']:
        vals = [float(a[f'{metric}_seed{s}']) for s in range(3)]
        assert abs(statistics.mean(vals)-float(a[f'{metric}_mean'])) < 1e-8
        assert abs(statistics.stdev(vals)-float(a[f'{metric}_std'])) < 1e-8

def dump_csv(path, data):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def fmt(v):
    return f'{float(v):.3f}'

def table(data, mode='single'):
    text = '| 梯度／最终更新方向 | 方法或实验含义 | FAA ↑ | AAA ↑ | BWT ↑ |\n|---|---|---:|---:|---:|\n'
    for formula, label, value in data:
        if value is None:
            nums = ['未找到对应实验']*3
        elif mode == 'aggregate':
            nums = [f"{fmt(value[m+'_mean'])}±{fmt(value[m+'_std'])}" for m in ['FAA', 'AAA', 'BWT']]
        else:
            nums = [fmt(value[m]) for m in ['FAA', 'AAA', 'BWT']]
        text += f'| {formula} | {label} | '+ ' | '.join(nums) + ' |\n'
    return text

native = [
    (r'$(g_1+g_3)-0.1\Pi^\perp_{g_1+g_3}(g_0+g_2)$', '原版组合 GAM，含正交修正', agg['N01']),
    (r'$(g_1+g_3)-(g_0+g_2)$', '原始差分相减；同时把 γ 改为 1', None),
    (r'$(g_1+g_3)-0.1(g_0+g_2)$', '固定 γ=.1，仅删除投影', None),
    (r'$g_0$', 'SGD 方向（相同探测预算）', agg['N05']),
    (r'$g_1$', 'SAM 方向', agg['N03']),
    (r'$g_0+(g_3-g_2)$', '基点＋第二局部差分，UG', agg['N07']),
    (r'$g_1+g_3$', '组合方向；γ=0', agg['N02']),
    (r'$(g_1+g_3)/2$', '同一组合方向、幅度减半；原 LR', agg['N10']),
    (r'$g_3$', '仅外层位置的内探针梯度', agg['N04']),
    (r'$g_2$', '仅外层位置梯度', agg['N06']),
]
new3 = [
    (r'$g_0$', 'SGD 方向参考', agg['P01']),
    (r'$g_1$', '正向 SAM 方向参考', agg['P02']),
    (r'$g_2$', '正向外层位置梯度参考', agg['P04']),
    (r'$g_1+(g_3-g_2)$', 'SAM 方向＋第二局部差分', agg['G12_g1_plus_d32']),
    (r'$g_1+g_2$', '两个位置梯度求和', agg['G12_g1_plus_g2']),
    (r'$(g_1+g_2)/2$', '同方向半幅；原 LR', agg['G12_half_g1_plus_g2']),
]

def by_job(job):
    candidates = [r for r in rows if r['job'] == job and r['seed'] == '2']
    assert len(candidates) == 1, (job, len(candidates))
    return candidates[0]

def extensions(radius):
    suffix = 'r025' if radius == .25 else 'r1'
    return [
        (r'$g_1+g_2$', '求和参考', by_job(f'formal_sum12_{suffix}_k10_s2')),
        (r'$(g_1+g_2)/2$', '半和', by_job(f'formal_halfsum12_{suffix}_k10_s2')),
        (r'$g_0+g_1+g_2$', '再加入基点梯度', by_job(f'formal_sum012_{suffix}_k10_s2')),
        (r'$g_1+g_2-g_0$', '减去基点梯度', by_job(f'formal_sum12minus0_{suffix}_k10_s2')),
        (r'$g_0+\Pi^\perp_{g_0}(g_1-g_0)+\Pi^\perp_{g_0}(g_2-g_0)$', '以 g0 为参考的两残差垂直分量', by_job(f'formal_perp0_{suffix}_k10_s2')),
        (r'$g_0+[g_1+\Pi^\perp_{g_1}(g_2-g_1)]$', 'g0+B；以 g1 为投影参考，三探针', by_job(f'formal_g0B_{suffix}_k10_s2')),
        (r'$g_0+(g_2-g_1)$', 'A_P，跨位置差分', by_job('S1-07_A_P_natural_s2') if radius == 1 else next(r for r in rows if r['job'] == 'S1-07_A_P_natural_s2' and float(r['radius']) == .25)),
        (r'$g_1+(g_3-g_2)$', 'SAM 方向＋第二局部差分', by_job(f'formal_g1_d32_{suffix}_k10_s2')),
    ]

# S1-07 occurs at both radii; select explicitly rather than by job name alone.
def by_job_radius(job, radius):
    candidates = [r for r in rows if r['job'] == job and r['seed'] == '2' and float(r['radius']) == radius]
    assert len(candidates) == 1
    return candidates[0]

# Replace ambiguous accessor only for repeated S1 IDs.
old_by_job = by_job
def by_job(job):
    return by_job_radius(job, 1.0) if job.startswith('S1-') else old_by_job(job)

single_native = []
native_jobs = [
    (native[0][0], native[0][1], 's3_puregam_original_r1_k10_s2'),
    (r'$g_0$', 'SGD 方向', 'formal_g0_r1_k10_s2'),
    (r'$g_1$', 'SAM 方向：预先指定审计 S 对照', 'formal_S_r1_k10_s2'),
    (r'$g_1$', 'SAM 方向：早期物理重复，单独保留', 's3_puregam_gamma0_g1_r1_k10_s2'),
    (r'$g_0+(g_3-g_2)$', 'UG', 'formal_ug_r1_k10_s2'),
    (r'$g_1+g_3$', 'γ=0', 's3_puregam_gamma0_sum_r1_k10_s2'),
    (r'$(g_1+g_3)/2$', '原 LR', 's3_puregam_meansum_r1_k10_s2'),
    (r'$(g_1+g_3)/2$', 'LR×2，独立幅度控制', 'formal_Mean2_r1_k10_s2'),
    (r'$g_3$', '仅 g3', 's3_puregam_gamma0_g3_r1_k10_s2'),
    (r'$g_2$', '仅 g2', 'formal_g2_r1_k10_s2'),
]
single_native = [(f, l, by_job(j)) for f, l, j in native_jobs]
legacy025 = [
    (r'$g_0$', '自然步长', by_job('screen_u0_natural_r025_s2')),
    (r'$g_1$', '自然/匹配共用身份参考', by_job('screen_us_matched_r025_s2')),
    (r'$g_2$', '自然步长', by_job('screen_uo_natural_r025_s2')),
    (r'$g_3$', '自然步长', by_job('screen_ufull_natural_r025_s2')),
    (r'$g_0+(g_3-g_2)$', 'UG，自然步长', by_job('screen_ug_natural_r025_s2')),
]

export = []
for name, data, mode in [('native_r1_three_seed', native, 'aggregate'), ('new_r1_three_seed', new3, 'aggregate'), ('native_r1_seed2', single_native, 'single'), ('new_r1_seed2', extensions(1), 'single'), ('new_r025_seed2', extensions(.25), 'single'), ('recovery_r025_seed2', legacy025, 'single')]:
    for formula, label, value in data:
        item = {'table': name, 'gradient': formula.strip('$'), 'method': label, 'status': 'missing_not_zero' if value is None else 'verified', 'n': 3 if mode == 'aggregate' else 1}
        for metric in ['FAA', 'AAA', 'BWT']:
            item[metric] = '' if value is None else value[metric+'_mean' if mode == 'aggregate' else metric]
            item[metric+'_sample_SD'] = value[metric+'_std'] if value is not None and mode == 'aggregate' else ''
        item['source_config'] = value.get('source_config', '') if value is not None else ''
        item['source_matrix'] = value.get('source_matrix', '') if value is not None else ''
        item['source_receipt'] = value.get('source_receipt', '') if value is not None else ''
        export.append(item)
dump_csv(OUT / 'GRADIENT_TABLES.csv', export)
dump_csv(OUT / 'SOURCE_RESULTS_RECOMPUTED.csv', rows)

contrasts = []
for lhs, rhs in [('N01', 'N02'), ('N02', 'N10'), ('N02', 'N03'), ('N02', 'N04'), ('N07', 'N05'), ('N04', 'N03'), ('G12_g1_plus_g2', 'G12_half_g1_plus_g2'), ('G12_g1_plus_g2', 'G12_g1_plus_d32')]:
    item = {'lhs': lhs, 'rhs': rhs, 'paired_n': 3, 'seeds': [0, 1, 2], 'metrics': {}}
    for metric in ['FAA', 'AAA', 'BWT', 'LA', 'forgetting']:
        ds = [float(agg[lhs][f'{metric}_seed{s}'])-float(agg[rhs][f'{metric}_seed{s}']) for s in range(3)]
        item['metrics'][metric] = {'mean_delta': statistics.mean(ds), 'sample_SD': statistics.stdev(ds), 'deltas_by_seed': dict(zip([0, 1, 2], ds)), 'positive_count': sum(v > 0 for v in ds)}
    contrasts.append(item)
(OUT / 'PAIRED_CONTRASTS.json').write_text(json.dumps(contrasts, indent=2, ensure_ascii=False))

report = r'''# ImageNet-A 梯度方向结果表：2026-10-07

本次只整理结果，没有启动新训练。最新 g0+g1+g2 与半(g1+g2) 四项正式训练全部完成；Sum12、Mean12、D32 在 r=1 的三个 seed 也全部完成。本报告核验144/144个索引内完整十任务终点，独立重算原始矩阵；这是已完成结果索引，不表示所有后台队列或所有用户设想的方向均已完成。

统一科学设置：ImageNet-A train5981/test1519，200类固定顺序，10×20类；SeqLoRA ViT-B/16、rank10、ρ=gam_grad_rho=.05；每任务50epochs，总500epochs/2550更新，batch128；SGD momentum=0、weight decay=0，首任务LR=.04、后续LR=.02、每任务cosine最低比例.1。下文无另注时 r=gam_grad_norm_rho=1。

全部为纯方向实验：EWC/Fisher、实际权重融合、replay关闭。原 GAM 与各投影实验仍包含其公式所声明的梯度正交运算；不能把“没有 EWC/Fisher”解释为所有方法没有投影。自然步长组无最终范数匹配。

正向探针定义（公式省略实现中的数值稳定细节）：

$$g_0=g(\theta),\quad g_1=g(\theta+\rho N(g_0)),\quad g_2=g(\theta+rN(g_1-g_0)),\quad g_3=g(\theta+rN(g_1-g_0)+\rho N(g_2)).$$

$$N(v)=v/(\|v\|+\varepsilon),\qquad \Pi^\perp_u(v)=v-\frac{\langle u,v\rangle}{\|u\|^2+\varepsilon}u.$$

每组在恢复基点后，用 U 进行 θ←θ−ηU。投影是全部可训练 LoRA 与分类头参数的全局内积。

需要更正用户原表中的三个定义：

- 原 GAM 的 γ=.1，被减去的是 h− 在 h+ 正交补中的分量，不是 proj 本身，也不是未经缩放的整向量。
- 令 γ=0 得到 g1+g3；固定 γ=.1 删除投影应得到 h+−.1h−；写 h+−h− 又同时把 γ 改到1。这两个无投影方向尚未找到匹配的完整结果。
- g1 是 SAM 方向；g0+(g3−g2)是另一条实验方向，通常不相等。g1−g0测量基点附近变化，g3−g2测量外层位置附近变化，梯度和探针方向均可能不同。

FAA 为最后阶段任务等权平均；AAA 为十阶段各阶段已见任务等权平均再平均；BWT=前9任务最后准确率减各自刚学完准确率的平均，越大越好；当前均为负值，越接近0表示越少负迁移。FAA/AAA单位%，BWT单位百分点。±为seed0/1/2的样本SD（ddof=1），不表示显著性；单seed SD为N/A。投影公式忽略代码中的数值稳定项。

## 原生协议：按用户结构排列，三 seed 完整

每步原生五次 CE forward、四次梯度，按全局随机状态自然推进，原生减扰动恢复。不要与下节全状态恢复、R0重放四探针结果合并成一个严格受控排行榜。以下8个已有方向24/24完整；两个新设想无投影方向分别标缺失。

'''
report += table(native, 'aggregate')
report += r'''
这里的“组合 GAM”对应用户所称 Accelerated GAM，作为本项目表内描述名称；g0/g1只是 SGD/SAM 最终方向参考，仍执行额外探针，因此不应把其运行成本标作原生单梯度 SGD 或两梯度 SAM。

原生协议还完成了 A=g0+(g2−g1)、B=g1+Π⊥g1(g2−g1)、SAM的LR×1.5/2、半和的LR×2、g3有效步长匹配；完整三seed结果见累计来源报告。尤其半和LR×2不是本表“保持LR、只减半最终幅度”一行。

## 新协议：三个 seed 的扩展方向与参考

全状态精确恢复、相同R0随机状态重放，仅推进一次 g0/R1。三个新方向9/9实验格完整，其中6新正式训练、3复用seed2。三个参考9/9完整。参考 g0/g1/g2 的原四探针集合为 g0/g1+/g1−/g2；新方向为 g0/g1/g2/g3。未使用探针对最终状态和随机推进无影响已验证，但并非宣称两集合计算内容完全相同。

'''
report += table(new3, 'aggregate')
report += r'''
Sum12−Mean12：配对FAA +4.686±.527、AAA +1.772±.354；各3/3胜出。Sum12−D32：配对FAA +6.301±1.107、AAA +2.624±.479；各3/3胜出。同一状态下求和与半和方向完全共线，最终幅度相差2倍，原LR一致，因此更新幅度/有效步长确实影响结果；不能推出全部方向收益均由幅度解释。

Sum12−Mean12的LA +4.057±.617、BWT +.699±1.244；Sum12−D32的LA +5.332±.633、BWT +1.077±1.406。LA三seed均改善，BWT均为2/3改善，因此FAA优势不能主要归因于跨seed一致改善旧任务保持。

## 新协议 r=1：seed2 扩展（独立列出，不与三seed均值混排）

'''
report += table(extensions(1))
report += r'''
四探针求和、半和、三和、减g0、双垂直残差以及D32最终方向实验各完整。g0+B是专门三探针版本，g3已取消；探针状态与前两层相同，但计算预算不同。A_P使用四分支探针中的正向分支，最终恢复和随机状态处理相同。

对比同seed自然步长 Sum12：额外加g0使FAA −1.152、AAA −.233、BWT −.854；减g0使FAA −7.849、AAA −3.829、BWT −2.974。不能说基点梯度越多越好：当前三种系数中，g1+g2准确率最好，额外再加g0无进一步收益。

## 新协议 r=.25：seed2 对应扩展

'''
report += table(extensions(.25))
report += r'''
额外加g0使FAA −1.717、AAA −1.435；减半使FAA −1.831、AAA −.954。半径.25→1对Sum12为FAA +4.116/AAA +1.740，对Mean12仅+.657/+.643，对减g0为−3.679/−2.057；半径影响依赖最终组合与幅度。

## 原生 r=1：seed2 原始结果和重复身份

'''
report += table(single_native)
report += r'''
N03三seed统计使用冻结计划事先固定的 formal_S 审计参考（seed2 FAA44.969/AAA62.313），不是按终点择优。早期 g1 完整重复为45.132/62.166，配置只差输出名称/目录，在原生随机/GPU执行下仍产生小差异；它保留为单独物理重复，不替代既定参考、不算第四个seed。

## 早期 r=.25：seed2 分量恢复实验，单列历史协议

'''
report += table(legacy025)
report += r'''
本节epoch随机状态恢复协议独立于上面的R0重放协议，不能用本节g2的50.720替代新表g2的49.937。g1自然/匹配共用身份参考。其BWT由最终矩阵独立重算为−11.619；不能从AAA或FAA猜测BWT。

## 可以支持的结论及缺口

1. 原生协议：组合GAM与γ0双分支的配对FAA +.167±.389（仅1/3胜出）、AAA +.217±.176（3/3胜出）；BWT −.231±.662（仅1/3改善）。AAA有小幅一致增量，但当前FAA没有稳定的大幅收益证据证明原正交修正必要；这也不是固定γ只删除投影的因果对照。
2. 原生协议：g1+g3比单g1和单g3准确率更高；其中含方向与幅度两个变化。原LR半和使FAA平均降低4.438，说明幅度重要。半和LR×2的55.219±.475接近原求和54.810±.148，但不能把这些独立轨迹写成逐步完全相同。
3. 新协议：g1+g2在三个新方向中稳定最好，尤其优于SAM方向＋第二局部差分；这不说明g3在所有基点/半径/范数匹配下无用，也不直接证明Hessian机制。
4. r1 seed2：g1+g2−g0、额外加g0、仅保留以g0为参考的垂直残差均未超过g1+g2。单seed结果暂不作为跨seed确认。
5. 补齐投影问题需要同协议同γ下的 h+−.1h−；用户提出 h+−h−是另一项γ=1实验。新R0协议下 fullGAM/sum13/half13/g3/UG未找到对应完整训练；论文严格统一方向消融表应先补这5种，而不是填入原生5CE成绩。

本次只把缺口写入建议，不修改/启动队列；用户此次请求是汇总与扩展表格。

## 运行记录的一个字段歧义

最新 sum012/half12 的 effective_config 留有继承 compute_g3_probe=false，但专用FourProbeStep无条件计算g3。逐步记录、源码、2550次g3和10200CE计数确认实际四探针，g3不参与最终更新。该字段不控制此专用训练器；后续复现应以冻结代码与真实计数为准。

## 完整来源

表格和逐行来源：[GRADIENT_TABLES.csv](GRADIENT_TABLES.csv)；同seed配对差值：[PAIRED_CONTRASTS.json](PAIRED_CONTRASTS.json)；144条重算结果及配置/矩阵/收据路径：[SOURCE_RESULTS_RECOMPUTED.csv](SOURCE_RESULTS_RECOMPUTED.csv)；本次核验哈希：[AUDIT.json](AUDIT.json)。

既有累计报告：[REPORT.md](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-gam-cumulative-results-20261007T065249Z/REPORT.md)。

原生GAM冻结源码：[gam.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py:122)。

独立代理核验：[核心矩阵/配置](/data/140-1/users/liying/O-LoRA/refine-logs/GAM_TABLE_CORE_INVENTORY_20261007.json)、[最新四项正式训练](/data/140-1/users/liying/O-LoRA/refine-logs/GAM_TABLE_LATEST_AUDIT_20261007.json)、[33个三seed端点](/data/140-1/users/liying/O-LoRA/refine-logs/GAM_TABLE_THREE_SEED_AUDIT_20261007.json)。
'''
(OUT / 'REPORT.md').write_text(report)
audit = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only analysis; no training mutations or launches',
    'completed_index_metrics_verified': len(audits),
    'expected_in_completed_index': 144,
    'failed_in_completed_index': 0,
    'inference_scope': 'Tables separate execution protocols and n=1/n=3. Missing requested directions are not zeros; active incomplete jobs are not included.',
    'aggregate_conditions_verified': len(aggregates),
    'source_index_sha256': sha(SRC/'ALL_UNIQUE_10TASK_RESULTS.csv'),
    'source_aggregates_sha256': sha(SRC/'THREE_SEED_COMPLETE_32_CONDITIONS.csv'),
    'rows': audits,
    'artifacts': {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'AUDIT.json'},
}
(OUT / 'AUDIT.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
print(json.dumps({'out': str(OUT), 'rows_recomputed': len(audits), 'aggregates_recomputed': len(aggregates), 'table_rows': len(export)}, ensure_ascii=False))
