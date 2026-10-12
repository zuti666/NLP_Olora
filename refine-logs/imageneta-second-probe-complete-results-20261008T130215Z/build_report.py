"""Build a provenance-linked report and figure from accepted task matrices."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/oct08-gam-results-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/140-1/users/liying/O-LoRA')
OUT = Path(__file__).resolve().parent
with (OUT / 'ALL_20_FORMAL_RESULTS.csv').open() as f:
    all_rows = list(csv.DictReader(f))
with (OUT / 'CENTER_JOINT_SIX_CELLS.csv').open() as f:
    six = list(csv.DictReader(f))
joint = {r['variant']: r for r in six}
old = {(r['variant'], float(r['radius'])): r for r in all_rows if r['variant'] in ['B0', 'B1', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']}
assert len(all_rows) == 20 and len(six) == 6 and len(old) == 16
assert len({r['source_matrix'] for r in all_rows}) == 20

formulas = {
    'FF': r'\(g_0+g_3^{F,+}-g_2^F\)',
    'FB': r'\(g_0+g_2^F-g_3^{F,-}\)',
    'CPLUS': r'\(g_0+(g_3^{F,+}-g_3^{F,-})/2\)',
    'CMINUS': r'\(g_0-(g_3^{F,+}-g_3^{F,-})/2\)',
    'BF': r'\(g_0+g_3^{B,+}-g_2^B\)',
    'BB': r'\(g_0+g_2^B-g_3^{B,-}\)',
}
labels = {'FF': 'FF（复用）', 'FB': 'FB（复用）', 'CPLUS': 'Center+', 'CMINUS': 'Center−', 'BF': 'BF', 'BB': 'BB'}
order = ['FF', 'FB', 'CPLUS', 'CMINUS', 'BF', 'BB']
six_table = '| 条件 | 最终更新 | AAA | FAA | LA | BWT | 遗忘 |\n|---|---|---:|---:|---:|---:|---:|\n'
for v in order:
    row = joint[v]
    six_table += '| ' + ' | '.join([labels[v], formulas[v]] + [f'{float(row[k]):.3f}' for k in ['AAA', 'FAA', 'LA', 'BWT', 'forgetting']]) + ' |\n'

old_formulas = {
    'B0': r'\(g_0+(g_3^+-g_2)\)', 'B1': r'\(g_1^++(g_3^+-g_2)\)',
    'N1': r'\(g_0-(g_3^+-g_2)\)', 'N2': r'\(g_1^+-(g_3^+-g_2)\)',
    'N3': r'\(g_0+(g_3^--g_2)\)', 'N4': r'\(g_1^++(g_3^--g_2)\)',
    'N5': r'\(g_0+(g_2-g_3^-)\)', 'N6': r'\(g_1^++(g_2-g_3^-)\)',
}
old_table = '| 编号 | 最终更新 | r=.25 AAA / FAA | r=1 AAA / FAA |\n|---|---|---:|---:|\n'
for v in ['B0', 'B1', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']:
    vals = [f"{float(old[(v, r)]['AAA']):.3f} / {float(old[(v, r)]['FAA']):.3f}" for r in [.25, 1.]]
    old_table += '| ' + ' | '.join([v, old_formulas[v]] + vals) + ' |\n'

report = r'''# ImageNet-A 第二探针符号、中心响应与联合后向实验：完整结果

核验日期：2026-10-08，Europe/Madrid。上一批16项正式训练及新增4项正式训练均完成并通过验收，合计20项独立正式训练；0失败、0运行、0等待。上一批于2026-10-07 19:11:36（UTC+2）结束，新增批次于22:22:56结束。FF/FB是复用参考，不重复计入新增训练。

原始任务矩阵独立重算与汇总最大差异为2.85e-14；每项均有500条epoch记录、2550次实际SGD更新及有效完成回执。短测不进入成绩表。Center−等低分结果属于训练效果失败，流程验收通过，并非OOM、NaN或运行中断。

## 共同配置与可比范围

ImageNet-A：5981训练样本、1519测试样本，200类，固定类别顺序0…199，10×20任务；ViT-B/16、augreg2_in21k_ft_in1k预训练，冻结主干，Q/V LoRA rank10和已见类分类头训练。单seed2，内半径ρ=gam_grad_rho=.05；外半径r=gam_grad_norm_rho=.25或1，新四项仅r=1。

每任务50 epochs、batch128；SGD momentum=0、weight_decay=0，首任务LR=.04，后续LR=.02，cosine最低比例.1。FP32/native attention、TF32/AMP关闭，严格确定性算法。关闭EWC/Fisher及其计算、近似Fisher、实际权重融合、最终投影/正交修正、最终范数匹配、SGV、replay、HVP及附加漂移惩罚。探针构造仍保留单位方向。

本表采用最近一轮确定性执行协议及验收矩阵；历史其他执行协议的旧表不混入本轮统计。FF/FB复用旧批次，每步4次训练梯度，每任务另有1次隔离诊断梯度且精确恢复状态/RNG；BF/BB每步4次训练梯度、额外诊断为0；Center±每步5次训练梯度、额外诊断为0。CPU数值/状态回归及真实GPU配对门禁均通过，但Center±与单侧方法计算成本不同。

AAA为十阶段任务等权准确率的平均，FAA为最终十任务等权准确率；单位均为%。LA为刚学完各任务的准确率均值；BWT为旧九任务最终值减学习时值的均值；遗忘为学习后峰值减最终值的旧九任务均值。各条件n=1，无跨seed标准差或显著性检验。

## 方向定义

令N(v)=v/(||v||+1e−12)。第一段F为：g₁⁺=g(θ₀+ρN(g₀))，qF=g₁⁺−g₀，θ₂F=θ₀+rN(qF)。第一段B为：g₁⁻=g(θ₀−ρN(g₀))，qB=g₀−g₁⁻，θ₂B=θ₀+rN(qB)。注意B翻转了下降侧差分的符号。

在各自外点θ₂测得g₂；末端正/反探针为θ₃±=θ₂±ρN(g₂)，梯度记为g₃±。第二段F残差为g₃⁺−g₂，第二段B残差为g₂−g₃⁻。FF/FB/BF/BB均将残差加入g₀，再恢复θ₀执行SGD。后向探针不等于直接用负梯度进行最终训练更新。

## 新增中心与联合替换：r=1

''' + six_table + r'''
BB为这四个联合F/B条件中的最高观测FAA；本轮所有方向中，N6的AAA更高。因此不要将BB称为整个ImageNet-A实验史的最佳方法。

## 上一批16项：第一段固定F，只改变末端

这里所有g₂、g₃均位于第一段F所构造的外点；B0/B1的末端是正向残差，N5/N6是修正取向的后向残差，N1–N4反转了正常有向响应。

''' + old_table + r'''
## 结论及解释范围

**1. 中心响应的符号在去掉不对称项后仍决定是否正常学习。** 在同一外点定义s₂=(g₃⁺−g₃⁻)/2，e₂=(g₃⁺+g₃⁻−2g₂)/2，则正向残差为s₂+e₂、修正后向残差为s₂−e₂。Center±直接使用g₀±s₂，没有除以ρ，系数仍为1。Center+相对FF为+0.021 AAA/+0.586 FAA，相对FB为−0.188/−0.462；删掉e₂后仍保持正常表现。

Center−为2.064 AAA/0.651 FAA。Task0在线训练准确率在epoch5达到76.807%，epoch7跌破50%、epoch9跌破10%，Task0最终测试6.51%。这是首任务阶段即崩溃，不能用低BWT解释为保持良好。这增强“共同有向响应及其取向重要”的证据，但不证明所有收益只来自s₂。

e₂并非数值上可忽略：共享初始批次||e₂||/||s₂||=.309，Center+轨迹中位数=.439。删除它有效，不能推导为它总是无用或只是噪声。s₂/e₂是奇偶响应分解，不是正交投影分解。

**2. 两处局部上升侧读数可以在当前分量版本中联合替换。** BB使用g₁⁻和g₃⁻，差分分别定向为g₀−g₁⁻和g₂−g₃⁻。实测计数：2550次SGD、10200次训练梯度、两处局部正向梯度调用均为0、额外诊断梯度0。相对FF提高0.292 AAA/1.351 FAA，说明联合后向版本正常学习并保留表现。

BB仍使用梯度范数引导的外层探针，仍需四次训练梯度；它是g₀+残差的GAM分量变体，不是含原始g₁/g₃组合与正交修正的完整GAM。因此不支持“完整GAM已等价替代”或“调用次数减少”。

**3. 提升需要区分新任务学习与旧任务保持。** 十任务下FAA=LA+.9BWT是精确记账关系。BB−FF的FAA+1.351分解为LA+.536和保持项+.815；BB−FB的FAA仅+.303，却分解为LA+.809及保持项−.506，且遗忘增加.174。BB对FB的小幅优势主要来自学习更多，不能概括为遗忘更少。

第二段采用修正后向残差：FB−FF为+.209 AAA/+1.048 FAA，BB−BF为+.859/+1.596。第一段替换的影响随第二段改变：BF−FF为−.568/−.245，而BB−FB为+.083/+.303；差中之差为+.650 AAA/+.548 FAA。此处仅为单seed描述，不能做统计交互结论。

**4. 半径并非越大越好，差分符号也不能随意交换。** r=.25→1，B0的FAA−.122，B1−.619，N5+.617，N6+.197。正确取向的后向残差在两个半径均正常；反号残差在r=1均崩溃。r=.25的部分反号组能阶段性学习，例如N2 AAA39.599、N4 AAA43.107，但最终FAA仅2.213/17.364，不能把这些情况全部描述为Task0内崩溃。

**5. 当前证据还不能排除更新幅度与稳定性机制。** 所有结果采用自然步长。Center−在Task0的最大U范数约517，实际LoRA有效矩阵加分类头更新范数约160.536；Center+对应4.34/.172。Center−只有3/500次Task0更新与g₀夹角为钝角，故不能将失败简化为“每步都沿CE上升方向”。符号干预伴随梯度和有效步长放大；尚未测得或证实Hessian/HVP因果机制，也不能从单seed小差值断言优越或等价。

若后续开展验证，应优先用配对额外seed复核FF/FB/BB/Center+，并在预先固定协议下控制Center±的有效更新幅度；本次仅总结结果，没有新增训练。

## 证据与产物

- [20项独立正式训练CSV](ALL_20_FORMAL_RESULTS.csv)：原始矩阵、回执、日志及重算数值逐条对应。
- [六条件CSV](CENTER_JOINT_SIX_CELLS.csv)：明确标记FF/FB复用，附配置与文件哈希。
- [完成验收](COMPLETION_AUDIT.json)：原验证器20/20通过，源文件冻结哈希一致。
- [独立科学分析](CENTER_JOINT_ANALYSIS.json)：中心分解、轨迹指标、逐任务及配对差值。
- [图PDF](RESULT_COMPARISON.pdf) / [图PNG](RESULT_COMPARISON.png)：采用已验收结果，无误差条（n=1）。
'''
(OUT / 'RESULT_ANALYSIS.md').write_text(report, encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), layout='constrained')
short = ['FF', 'FB', 'Center+', 'Center−', 'BF', 'BB']
colors = ['#64748b', '#0f766e', '#3b82f6', '#dc2626', '#94a3b8', '#7c3aed']
for ax, metric, ceiling in [(axes[0, 0], 'AAA', 73), (axes[0, 1], 'FAA', 55)]:
    values = [float(joint[v][metric]) for v in order]
    ax.bar(short, values, color=colors, width=.68)
    for i, val in enumerate(values):
        ax.text(i, val + .75, f'{val:.3f}', ha='center', fontsize=9)
    ax.set_ylim(0, ceiling)
    ax.set_ylabel(f'{metric} (%)')
    ax.set_title(f'{metric}: centered response and joint replacement, r=1')
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
for ax, metric in [(axes[1, 0], 'AAA'), (axes[1, 1], 'FAA')]:
    for v, color, name in [('B0', '#64748b', 'g0 + forward'), ('B1', '#3b82f6', 'g1 + forward'), ('N5', '#0f766e', 'g0 + corrected backward'), ('N6', '#7c3aed', 'g1 + corrected backward')]:
        values = [float(old[(v, r)][metric]) for r in [.25, 1.]]
        ax.plot([.25, 1.], values, marker='o', color=color, label=name)
    ax.set_xticks([.25, 1.], ['0.25', '1.0'])
    ax.set_xlabel('Outer radius r')
    ax.set_ylabel(f'{metric} (%)')
    ax.set_title(f'Correctly oriented one-sided updates: {metric}')
    ax.grid(alpha=.22)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
fig.suptitle('ImageNet-A | 10 tasks | seed2 | rank10 | rho=0.05 | natural SGD step\nNo EWC/Fisher or weight fusion; FF/FB reused; single-seed results without error bars', fontsize=12)
fig.savefig(OUT / 'RESULT_COMPARISON.pdf')
fig.savefig(OUT / 'RESULT_COMPARISON.png', dpi=160)
plt.close(fig)

stamp = datetime.now(timezone.utc).isoformat()
entries = []
for file in sorted(OUT.iterdir()):
    if file.is_file() and file.name != 'ARTIFACT_MANIFEST.json':
        entries.append({'path': str(file.relative_to(ROOT)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'bytes': file.stat().st_size})
(OUT / 'ARTIFACT_MANIFEST.json').write_text(json.dumps({'created_utc': stamp, 'purpose': 'completed experiment result analysis; no training or queue mutations', 'formal_unique_runs': 20, 'new_center_joint_runs': 4, 'reused_center_joint_references': 2, 'files': entries}, ensure_ascii=False, indent=2) + '\n')
with (ROOT / 'MANIFEST.md').open('a') as f:
    for name in ['RESULT_ANALYSIS.md', 'ALL_20_FORMAL_RESULTS.csv', 'CENTER_JOINT_SIX_CELLS.csv', 'COMPLETION_AUDIT.json', 'CENTER_JOINT_ANALYSIS.json', 'STATUS_SNAPSHOT.json', 'RESULT_COMPARISON.pdf', 'ARTIFACT_MANIFEST.json']:
        f.write(f'| {stamp} | /monitor-experiment + /analyze-results | {str((OUT / name).relative_to(ROOT))} | analysis | 20/20正式训练完成；Center±/BF/BB 4项新增，FF/FB 2项复用；矩阵与回执独立验收 |\n')
print(json.dumps({'report': str(OUT / 'RESULT_ANALYSIS.md'), 'unique_formal_runs': len(all_rows), 'center_joint_rows': len(six), 'plot': str(OUT / 'RESULT_COMPARISON.pdf')}, ensure_ascii=False))
