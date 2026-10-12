#!/usr/bin/env python3
import csv
import json
from pathlib import Path

P=Path(__file__).resolve().parent
T=json.loads((P/'training_results.json').read_text())
A=json.loads((P/'diagnostic_anchor_summary.json').read_text())
C=json.loads((P/'diagnostic_candidate_summary.json').read_text())
S=json.loads((P/'diagnostic_sam_summary.json').read_text())
L=json.loads((P/'VALIDITY_LEDGER.json').read_text())
byarm={r['arm']:r for r in T}
order=['original_gamma_0.1','sum_r1','mean_sum_r1','g1_r1','g3_r0','g3_r0.1','g3_r0.25','g3_r0.5','g3_r1']

lines=['# 完整十任务与同状态机制诊断审计', '',
'训练结果：9/9 有效（5 次新训练＋4 次已完成结果复用），每项 10 tasks × 50 epochs。诊断结果：9/9 有效（原队列 2 项＋恢复队列 7 项），共 33 个 probe batches，825/825 反事实有效步幅匹配成功。未改动原训练或诊断输出。', '',
'只有 seed 2。以下是该 seed 的描述性比较，没有跨 seed 标准差、置信区间或显著性结论。各诊断起点独立列出；表内均值仅对该起点的 3 或 4 个固定训练批次取平均，batch 不是训练 seed。', '',
'## 完整性与定义', '',
'- 训练完成凭据、运行配置哈希、190 个冻结源文件哈希、原始指标哈希均通过；每项找到 500 个不同的 task/epoch 记录。九项均 Fisher 跳过 10 次、历史 Fisher 为空、EWC=0、实际融合为 false。',
'- 每项原始 JSON 的 468 个 NaN 均已分类：420 个未观测矩阵上三角位置，48 个未定义的前向迁移/首任务元数据；所有已观测矩阵、训练损失和诊断数值通过有限性检查。FG 表示 forward generalization，不是 forgetting。',
'- 诊断最初 task9 请求 4 批而只有 378 个训练样本，失败没有写结果；恢复时仅将 task9 请求改为 3 批，脚本及公式不变。task0 取 4×128 样本，task4 为 128/128/128/42，task9 为 128/128/122。旧失败记录保留，未计入完成结果。',
'- 记 A[t,j] 为学完 task t 后对 task j 的准确率。Task AAA=mean_t(mean_{j≤t} A[t,j])；Task FAA=mean_j A[9,j]；acquisition=mean_j A[j,j]。BWT=mean_{j<9}(A[9,j]−A[j,j])；forgetting=mean_{j<9}(max_{t≥j} A[t,j]−A[9,j])。',
'- Sample AAA/FAA 使用真实测试样本加权的 CNN top1 曲线。它们与 task macro 的权重不同。原始 native `AAA` 字段也不是这里的 stage-wise Task AAA，不能混用。准确率与差值均用百分数/百分点。', '',
'## 十任务结果', '',
'| 臂 | Task AAA | Task FAA | Acquisition | BWT | Forgetting | Sample AAA | Sample FAA |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
metrics=['aaa_task_macro','faa_task_macro','acquisition_task_macro','bwt','forgetting','aaa_sample_weighted','faa_sample_weighted']
for arm in order:
 r=byarm[arm];lines.append('| '+arm+' | '+' | '.join(f'{r[k]:.3f}' for k in metrics)+' |')
lines += ['',
'1. only-g3 半径从 0 增至 .25 时，Task AAA 从 62.299 升至 66.230，FAA 从 45.253 升至 50.799；.25/.5/1 的 FAA 分别为 50.799/50.692/50.679，已经非常接近。r=.25 对 r=1 仅 +0.120 pp FAA，而 r=1 的 BWT 更好（−6.721 对 −7.242），forgetting 更少（7.011 对 7.366）。不能由单 seed 的微小差别宣布 .25 稳定最优。',
'2. Mean=(g1+g3)/2 的 FAA 为 49.927；Sum 为 54.726，相差 −4.799 pp。Mean 对 g1 为 +4.795 pp，对 g3(r=1) 为 −0.752 pp。因此 Sum 的长期优势对幅值敏感，这批实验不支持“仅方向组合即可解释优势”。',
'3. Sum 对 g3(r=1) 的 FAA +4.047 pp、acquisition +4.855 pp，但 BWT 更负（−7.619 对 −6.721）。收益主要伴随更强任务获取，不能简单描述为更少遗忘。',
'4. r=0 的 g3 与 g1 长期结果接近但不完全一致（FAA +0.121 pp）。局部 r0 恒等检查均通过；浮点探针往返与长期轨迹放大允许最终独立训练有小差异。',
'5. 保留 gamma=.1 的 original pure-GAM 参考在这九项中 Task AAA/FAA 最高（68.028/55.341），比 gamma=0 Sum 高 .168/.615 pp；它包含非零正交校正，不能归为本轮 gamma=0 三方向的干净对照。', '',
'## 同状态匹配有效步幅的诊断', '',
'每个 checkpoint 均从同一模型状态、同一固定增强后的训练 batch 形成方向。将候选一步更新的 concat(LoRA ΔBA, FC Δweight, FC Δbias) 范数匹配到 −.02*g1 的实际范数，然后在固定测试 batch 上测量 CE。`.02` 是虚拟诊断步长，不是该 checkpoint 的训练学习率。旧任务 CE 先按各任务样本取均值，再对旧任务等权平均。', '',
'所有实际有效步幅相对误差最大为 1.002×10⁻⁵，远小于设定 10⁻³；所有 825 个候选的范数多项式均未出现正半轴驻点，未触发多根歧义。r0 恒等最大相对梯度误差为 2.836×10⁻⁴ < 5×10⁻⁴。', '',
'下表为候选相对 g1 的**损失变化差**，负值表示该候选更有利；单位是 CE（自然对数），不是准确率百分点。每一行是独立起点。', '',
'| 起点 | n batches | g3−g1 当前 local CE | g3−g1 旧 all-seen CE | Sum−g1 当前 local CE | Sum−g1 旧 all-seen CE |',
'|---|---:|---:|---:|---:|---:|']
lookup={(r['anchor'],r['candidate']):r for r in C}
def fmt(v):return 'NA' if v is None else f'{v:+.6f}'
for a in A:
 g=lookup[(a['anchor'],'g3')];s=lookup[(a['anchor'],'sum')]
 vals=[g['current_task_local_ce_vs_g1_mean'],g['old_task_macro_all_seen_ce_vs_g1_mean'],s['current_task_local_ce_vs_g1_mean'],s['old_task_macro_all_seen_ce_vs_g1_mean']]
 lines.append('| '+a['anchor']+f" | {a['n_probe_batches']} | "+' | '.join(map(fmt,vals))+' |')
lines += ['',
'- g3 相对 g1 的当前 local CE 仅在 g1_task4 起点更好，其余八个起点更差。对有旧任务的六个起点，g3 的旧 all-seen CE 只在 g1_task4/9 更好；在 g3/sum 各自的 task4/9 checkpoint 上更差。',
'- g1_task9 展示明确取舍：g3 使当前 local CE 的一步变化由 −.003046 变为 +.002417，却使旧 all-seen CE 从 +.000548 变为 −.002656。不能只挑旧任务改善而忽略当前任务代价。',
'- g3_task4 是反例：g1 一步当前 local/旧 all-seen CE 为 −.004443/−.002443，g3 为 −.000837/+.000874，匹配步幅后 g1 在两者上都更好。',
'- Sum 与 Half-Sum 匹配后所有任务 CE 完全相同；正比例 scale-only 与 g1 的最大 CE 差仅 8.98×10⁻⁷。这是共线方向匹配控制的应有结果，不是两个独立方向的证据。',
'- 较小 r 的局部 g3 通常更接近 g1；除 g1_task4 的特定起点外，增大 r 并未给出普遍的当前损失改善。这与完整训练的半径扫描并不矛盾：末端局部一步无法代表整个学习轨迹。', '',
'## 方向几何与 SAM 半径对齐', '',
'g3=a*g1+r_perp，R_perp=||r_perp||/||g3||。下表为每个起点自己的 batch 均值；ρ_param/ρ_KL 为沿 g0 方向匹配 GAM 第四探针参数距离/输出 KL 后得到的 SAM 半径。', '',
'| 起点 | a | R_perp | cos(g1,g3) | ρ_param | ρ_KL | cos(SAM_param,g3) | cos(SAM_KL,g3) |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
sam={(r['anchor'],r['match_kind']):r for r in S}
for a in A:
 sp,sk=sam[(a['anchor'],'parameter')],sam[(a['anchor'],'kl')]
 vals=[a['a']['mean'],a['R_perp']['mean'],a['cos_g1_g3']['mean'],sp['rho']['mean'],sk['rho']['mean'],sp['cos_g3']['mean'],sk['cos_g3']['mean']]
 lines.append('| '+a['anchor']+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines += ['',
'- 33/33 参数距离匹配与 33/33 KL 匹配都成功。参数距离匹配 SAM 的半径约 1.01–1.04，约为原始 rho=.05 的 20–21 倍；KL 匹配的个别 batch 半径范围 .425–3.700，远不是原始 .05 的同半径 SAM。',
'- g3 包含明显非共线分量：R_perp 的起点均值约 .60–.88，cos(g1,g3) 约 .47–.79；但几何差异本身不说明该方向更好。a 在全部 batch 中为正。',
'- 在 g1_task4，KL 匹配 SAM 的当前 local/旧 all-seen ΔCE 为 −.007794/−.004500，均优于 g3 的 −.003174/−.000661；在 g1_task9 也是 −.005183/−.006295，对应 g3 +.002417/−.002656。这些局部结果表明“更远探针”本身是重要对照，不能把收益全部归给 GAM 专有方向。',
'- 在 g3/sum 的已训练起点，扩大半径后的 SAM 相对 g1 通常也没有更优损失变化；来源 checkpoint 的条件性很强，不能跨起点混合做统一因果结论。KL 匹配和损失评估共用当前任务的固定评估 batch，属于探索诊断。', '',
'## 可支持的结论与边界', '',
'本 seed 支持：Sum 的长期性能优势对幅值敏感；g3 半径从 0 到中等值改善完整十任务训练，但 .25/.5/1 之间没有足够证据做稳健排序；g3 与 g1 存在明显几何差异，然而末端同状态、同有效步幅诊断不支持 g3 普遍优于 g1。', '',
'不支持：从这些末端单步结果推出长期 formation/continued-learning 因果机制；把 acquisition 更高等同于 forgetting 更少；把 probe batch 当成独立 seed；把 Task0 的旧任务指标 NA 当成 0。', '',
'当前新 Task0 队列是分离训练 RNG 的独立协议，必须先按自己的完整性门槛完成，不能用这里旧十任务 Task0 填补。下一步先审该队列的独立单任务曲线与幅值对照；若要确认十任务 r=.25 对 r=1 的微小优势，需要另行授权的多 seed 配对复现，本轮不追加训练。', '',
'## 文件', '',
'- `VALIDITY_LEDGER.json`：完成、来源与结构 NA 清单。',
'- `training_summary.csv`、`training_contrasts_pp.csv`、`training_matrix_long.csv`、`training_results.json`：全表、差值、每任务矩阵及逐任务 acquisition/BWT/forgetting。',
'- `DIAGNOSTIC_LOSSES.md`：九个起点的主要候选绝对 ΔCE，当前/旧任务、all-seen/local 均列出。',
'- `diagnostic_candidate_summary.csv`：所有候选，按起点保留 batch 均值/最小/最大及相对 g1 的同 batch 差值。',
'- `diagnostic_*_batches.csv`：逐起点逐 batch 原始数值；`diagnostic_sam_summary.csv` 包含半径范围与成功状态；`diagnostic_radius_summary.csv` 包含完整 r 扫描。',
'- `audit_results.py` 可重新只读核验；`make_report.py` 重建本报告。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
detail=['# 主要候选的一步损失变化', '', '下列均值仅在各起点内部对固定 probe batches 平均，ΔCE=after−before；负值更好。完整逐 batch 数值在 CSV。Task0 的旧任务为 NA。', '']
for a in A:
 detail += ['## '+a['anchor'], '', '| 候选 | 当前 local | 当前 all-seen | 旧 task-macro local | 旧 task-macro all-seen |','|---|---:|---:|---:|---:|']
 for name in ['g1','g3','sum','half_sum','sam_parameter_matched_probe','sam_kl_matched_probe']:
  r=lookup[(a['anchor'],name)]
  keys=['current_task_local_ce_delta_mean','current_all_seen_ce_delta_mean','old_task_macro_task_local_ce_delta_mean','old_task_macro_all_seen_ce_delta_mean']
  detail.append('| '+name+' | '+' | '.join(fmt(r[k]) for k in keys)+' |')
 detail.append('')
(P/'DIAGNOSTIC_LOSSES.md').write_text('\n'.join(detail)+'\n')
print('REPORTS_WRITTEN')
