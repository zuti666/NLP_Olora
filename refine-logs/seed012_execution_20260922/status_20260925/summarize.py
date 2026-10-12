"""Reproducible reporting only; this script never starts or modifies training."""
import csv,hashlib,json,statistics
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import Counter

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
result=json.loads((OUT/'snapshot_results.json').read_text())
state=json.loads((OUT/'snapshot_state.json').read_text())
METHODS=['Vanilla LoRA','InfLoRA','OLoRA','SDLoRA','CL-LoRA','EWCLoRA','GAM-EWCLoRA']
DATASETS=['aircraft','cars196','cub200','flowers','het5']
LABELS=['Aircraft','Cars','CUB-200','Flowers','5Datasets']
groups={(r['method'],r['dataset']):r for r in result['groups']}
def sd_text(value):return f'{value:.3f}' if 0<value<.005 else f'{value:.2f}'
def cell(r,metric,tex=False):
    if not r['n']:return '--'
    mean=r[metric+'_mean'];sd=r[metric+'_sample_sd']
    assert sd is not None
    value=(rf'\ExpRes{{{mean:.2f}}}{{{sd_text(sd)}}}'if tex else f'{mean:.2f} ± {sd_text(sd)}')
    if r['n']<3:value+=(r'$^{\dagger}$'if tex else '†')
    return value
def table(metric):
    lines=['| Method | '+' | '.join(LABELS)+' |','|---|'+'---:|'*5]
    for m in METHODS:lines.append('| '+m+' | '+' | '.join(cell(groups[m,d],metric)for d in DATASETS)+' |')
    return '\n'.join(lines)
tables='FAA（%）\n\n'+table('FAA')+'\n\nAAA_stage（%）\n\n'+table('AAA_stage')+'\n'
(OUT/'TABLES.md').write_text(tables)
tex=[r'\begin{table*}[t]',r'\centering',
 r'\caption{Fine-grained continual learning under documented configurations. Entries report mean and sample standard deviation (ddof=1) across seeds 0, 1, and 2; $^{\dagger}$ reports only successful seeds 0 and 2 because seed 1 diverged. FAA and stage-averaged AAA are percentages. 5Datasets is the cross-dataset task stream.}',
 r'\label{tab:exp_fine}',r'\scriptsize\setlength{\tabcolsep}{2.2pt}\renewcommand{\arraystretch}{1.08}',
 r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l*{10}{r}@{}}',r'\toprule',
 'Method & '+' & '.join(r'\multicolumn{2}{c}{'+x+'}'for x in LABELS)+r'\\',
 r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}',
 ' & '+' & '.join(['FAA','AAA']*5)+r'\\\midrule']
for m in METHODS:tex.append(m+' & '+' & '.join(cell(groups[m,d],q,True)for d in DATASETS for q in ['FAA','AAA_stage'])+r'\\')
tex += [r'\bottomrule',r'\end{tabular*}',r'\par\vspace{3pt}',r'\begin{minipage}{\linewidth}\scriptsize',
 r'Missing entries have no valid complete run in this campaign: SDLoRA on the four single-dataset streams ran out of memory; EWCLoRA/Cars diverged. Their dependent seeds were not started. The $^{\dagger}$ entry is incomplete and must not be described as a three-seed result. Small nonzero standard deviations use three decimals. CL-LoRA seed 0 on all datasets and GAM-EWCLoRA/5Datasets seed 0 are validated archived runs. Dataset-specific and method-specific documented settings are retained; this is not an equal-compute comparison.',
 r'\end{minipage}',r'\end{table*}']
(OUT/'table_finegrained_seed012.tex').write_text('\n'.join(tex)+'\n')
with (OUT/'raw_seed_results.csv').open('w',newline='')as f:
    keys=['method','dataset','seed','origin','FAA','AAA_stage','metric_path','metric_sha256']
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for row in result['runs']:w.writerow({k:row[k]for k in keys})
with (OUT/'group_statistics.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=result['groups'][0]);w.writeheader();w.writerows(result['groups'])
rows=[]
for d in DATASETS:
    gam=groups['GAM-EWCLoRA',d]
    for comparator in ['Vanilla LoRA','EWCLoRA']:
        base=groups[comparator,d]
        if gam['n']==3 and base['n']==3:
            rows.append(dict(dataset=d,comparator=comparator,
                faa_delta_pp=gam['FAA_mean']-base['FAA_mean'],
                faa_relative_percent=100*(gam['FAA_mean']/base['FAA_mean']-1),
                aaa_stage_delta_pp=gam['AAA_stage_mean']-base['AAA_stage_mean']))
(OUT/'descriptive_differences.json').write_text(json.dumps({'caution':'Observed differences under archived settings, not matched-protocol causal effects. Complete n3 groups only.','differences':rows},indent=2)+'\n')
fmt=lambda t:datetime.fromtimestamp(t,ZoneInfo('Europe/Madrid')).strftime('%Y-%m-%d %H:%M:%S %Z')
duration=(state['finished_unix']-state['started_unix'])/3600
receipt_hours=sum((r['finished_unix']-r['started_unix'])/3600 for p in (ROOT/'receipts').glob('*.json')for r in [json.loads(p.read_text())])
report=['# 多 seed 实验结果核查（2026-09-25）','',
 f'队列已结束，状态为 `{state["status"]}`。开始：{fmt(state["started_unix"])}；结束：{fmt(state["finished_unix"])}；经过 {duration:.2f} 小时。主机进程/GPU实时检查另见 host_check.json。','',
 '本次 99 个正式任务：83 完成、6 失败、10 因前置 seed0 失败而未启动。9 个短跑全部通过。加上6条已验证的旧seed0，共89/105条有效结果；29/35组完成seed0/1/2，1组仅seed0/2，5组无有效完整结果。','',
 '短跑只覆盖两任务，不能证明后续任务不发生OOM或数值发散。6个失败运行及10个依赖任务不进入最终均值；已有中间任务分数也不作为FAA。','',
 'FAA取最终CNN任务矩阵最后一行均值；AAA_stage取各阶段已见任务均值的平均；SD为跨seed样本标准差（ddof=1）。本报告从完整矩阵重新计算，不把旧表最佳单次结果拼接到新标准差上。','',tables,
 '† GAM-EWCLoRA/Aircraft只有成功seed0、2，seed1发生非有限梯度；所列为两个成功运行的部分统计，可能存在成功运行筛选偏差，不能当作完整三seed结果。','',
 'CL-LoRA/Flowers：FAA SD=0.004618802153514446，AAA SD=0.001552430723826036；显示为0.005和0.002，避免把四舍五入到0.00误解为完全无波动。数值精度受已保存任务准确率精度限制。','',
 '## 主要观察','',
 '1. 三seed完整组中，GAM-EWCLoRA在Cars与5Datasets的FAA/AAA均最高：Cars 53.04/56.26，5Datasets 83.24/81.19。Cars FAA比InfLoRA高17.06个百分点；5Datasets FAA比OLoRA高4.12个百分点。','',
 '2. CL-LoRA在CUB-200与Flowers领先：FAA分别78.31和97.83；但Aircraft29.35、5Datasets67.05，优势明显依赖数据集。','',
 '3. Aircraft的GAM部分均值45.97高于其他已观测均值，但seed1失败，不能据此给出完整三seed稳健性结论。','',
 '4. InfLoRA/Flowers的FAA标准差3.38最大；CL-LoRA大多数组波动较小。三seed只是初步稳定性证据，不等于显著性检验。','',
 '5. 重新运行的seed0并非所有组都数值复现旧表：GAM/Flowers旧86.747→新82.678，OLoRA/Cars旧32.855→新35.051。来源已按新冻结批次区分，因此旧单次最高值不能继续充当新三seed均值。独立复核保留了所有有效seed，包括低分运行。','',
 '## 缺口与下一步','',
 '- SDLoRA：Aircraft/Cars/CUB/Flowers的seed0显存不足，8个依赖seed未启动；需要先核对随任务增长的训练参数与显存，再决定保持数值协议的修复方案。','',
 '- EWCLoRA/Cars：seed0非有限梯度，seed1/2未启动。GAM-EWCLoRA/Aircraft：seed1非有限梯度。应保留失败记录，先诊断数值稳定性；若改变LR/精度/更新公式，应作为新协议重新跑整个seed组。','',
 '- 当前是原有归档配置下的对比；方法之间的训练轮数、调度和优化器等仍有差别，不能将上述差值直接解释成严格控制变量后的因果增益。','',
 '- 本次状态核查没有启动新训练或修改冻结训练配置。','',
 f'本次作业单进程墙钟时长合计约{receipt_hours:.2f}小时（包含失败和短跑，未包含复用历史运行）。这是单GPU作业时长近似，不是GPU实际活跃时间，也不用于跨方法等成本结论。','',
 '证据：snapshot_state.json、snapshot_results.json、raw_seed_results.csv、group_statistics.csv；独立复核与失败定位分别见INDEPENDENT_STATISTICS.md、FAILURE_ANALYSIS.md。论文表格代码见table_finegrained_seed012.tex，使用已定义的\\ExpRes。','']
(OUT/'REPORT.md').write_text('\n'.join(report))
print(tables)
