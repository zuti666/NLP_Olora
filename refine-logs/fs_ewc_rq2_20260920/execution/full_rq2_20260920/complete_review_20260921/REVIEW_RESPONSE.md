# Result-to-Claim 审查

**总判定：`claim_supported = partial`；置信度：`medium`。** 本审查为同族模型复核，`review_independence = same-family`，结论仅作 `provisional` 接受。三个独立训练 seed 足以判断多项方向一致的限域观察，但不足以稳定估计小效应或支撑跨顺序、跨数据集和因果机制主张。方向和半径是 seed 内相关探针，不能计作独立重复。

## 分项判定

| 主张 | 判定 | 审查结论 |
|---|---|---|
| RQ2 C1：塑形在 train 与 held-out test 均降低旧任务敏感度并扩大容忍半径 | `partial` | train 证据强且跨 3 seeds 一致；test 给出直接反证，不能写成 train/test 共同成立。 |
| RQ2 C2：更大有效参数移动仍可产生更小旧任务损伤，并有相容路径证据 | `partial` | t9 和所有 test 端点支持这一存在性观察；t1/t5 train 有反例，直线插值与积分界不是实际轨迹或因果识别。 |
| B3：固定后续 SGD 时，仅塑造首任务解提高旧任务保持 | `yes` | lambda=0 和 2000 两层的旧任务准确率、BWT 都改善；结论限于旧任务保持。 |
| B3/S3：EWC 在本 operating point 提供材料性保护 | `no` | B3 的 EWC 层间平均收益接近零，S3 单 seed lambda 响应也近乎平坦；这不是 EWC 普遍无效的证明。 |
| S3：factor fusion 有材料性贡献 | `no` | 预注册三 seed 的 FAA 正效应与 FAA 等价均未通过判据；结果是不确定，而非证明所有融合规则无效。 |
| S3：Fisher geometry/history/anchor 的必要性已证明 | `no` | 主要是 seed-2 探针，且预注册的 damping 确认触发后尚缺 4 个运行。 |
| S3：GAM 各子组件的必要性已证明 | `partial` | 完整 GAM 包和方向尺度与结果相关，但单组件大多仅 seed 2，SGD 对照还改变额外 forward/RNG 执行。 |

## 结果实际支持什么

RQ2 B1 在正式的 ImageNet-A、ViT-B/16、rank-10 Q/V LoRA、固定类别顺序、seeds 0/1/2 协议中，清楚支持**旧任务训练集上的局部耐受性改善**。在 80 个非零半径乘 3 seeds 的比较中，GAM 的 train 正 CE 损伤均值和 Q95 都是 240/240 低于 SGD；多个非右删失阈值下的经验容忍半径也扩大。r=20 时，train 均值从 `0.04678328±0.00185640` 降到 `0.03397111±0.00060309`。

这一结论不能延伸到 held-out test。test 正损伤均值在 225/240 个半径-seed 比较中更高，r=20 时从 `0.07993056±0.00271286` 升到 `0.08972830±0.00337397`，且 GAM 在 3/3 seeds 更差；test 容忍半径总体收缩或持平。因此“train 与 test 都更平坦”不成立。100 个方向和多个半径只提高同一训练运行内诊断的分辨率，不把有效样本量从 3 变成数百或数万。

RQ2 B2 支持一个较窄的存在性观察：在 t9，持续 GAM 的有效权重漂移更大（`1.44156` 对 `1.25391`），旧任务 train/test CE 损伤却更低，端点直线路径方向敏感度也更低（`0.01519` 对 `0.23965`）。18/18 路径数值门通过，test D0 在 t1/t5/t9 的 9/9 seed-task 配对中更低。不过，train D0 在 t1 的 3/3 seeds 和 t5 的 2/3 seeds 出现反例；端点插值不是实际优化轨迹，积分恒等式或界通过也不识别因果。B1 bridge 还显示真实学习方向的损伤在 6/6 情形超过相同半径随机方向样本最大值，所以随机邻域均值不能单独预测真实更新损伤。

RQ2 B3 对“首任务解塑形提高旧任务保持”给出直接支持。未来优化器固定为 SGD 时，B-A 的旧任务准确率为 `+4.127±0.500 pp`，D-C 为 `+4.233±0.650 pp`，BWT 分别改善约 `+3.910` 和 `+4.016 pp`。代价是两层的新任务准确率都下降 `3.413±0.992 pp`；两任务平均准确率只有 `+0.357±0.508` 和 `+0.410±0.737 pp`，后者逐 seed 为 `+0.015、+1.260、-0.045 pp`。所以可以写旧任务保持改善，不能写成稳健净平均收益。C-A 约为零、D-B 仅 `+0.053±0.092 pp`，不支持本 operating point 下 EWC 有材料性保护；这不推出 EWC 在其他协议中无效。

测得的 RQ2 raw empirical Fisher trace 在 GAM 侧普遍较低：任务 train 比较 30/30、所测 test 比较 12/12 均较低。然而 dense test 损伤经常更差，说明 trace 不是损伤的充分预测量。这里的 Fisher 是有效 Q/V 权重上逐样本真标签 CE 梯度平方和，未归一化且无 floor；它不是训练使用的 EWC 张量、Hessian 或新任务 Fisher。

## S3 结论与汇总修正

S3 必须与 RQ2 分层报告。S3 的 FAA/AAA 是 10-task task-macro 指标，native Fisher 是 batch-gradient delta-W proxy；这些指标和估计器不能与 sample-weighted accuracy、RQ2 两任务平均准确率或其他 sample_empirical/RNG-isolated 协议合并。

旧版 `summarize_mechanisms.py` 删除所有 `s3_*` override，因而丢失科学干预 `s3_fisher_location='endpoint'`，把 `s3_mech_fisher_endpoint_anchor_s2` 的 FAA `53.454` 错认成 seed-2 center，覆盖了真正的 `s3_center_s2`（FAA `53.419`）。原始运行结果未损坏，当前显式 ID 映射修复了 center 身份。

预注册主分析明确是 seeds 1/2/3。正确的 center-minus-matched-no-merge FAA 差为 `+0.089、+0.321、+0.071 pp`，均值 `+0.160333`、样本 SD `0.139432`，95% 配对 t 区间 `[-0.186035,+0.506702]`。三者虽都为正，区间包含 0，未支持正贡献；上界又略超出 `+0.5 pp`，也未建立 FAA 实用等价。AAA 差均值为 `+0.009707`、SD `0.081460`，区间 `[-0.192650,+0.212064]`，仅 AAA 满足预注册的 `±0.3 pp` 等价界。故融合的材料性贡献目前**不确定**。

最新 `SUMMARY_CORRECTION.json` 又把 seeds 2/3 标成 primary、把 seeds 1/2/3 标成 supplementary，但修正说明只解释 seed-2 center 身份，没有给出 seed 1 协议失效或排除依据。这一事后队列变化不能替代未变更的三 seed 预注册。seeds 2/3 只能作为敏感性分析：FAA `+0.196000±0.176777 pp`，95% 区间 `[-1.392276,+1.784276]`；AAA `+0.056698±0.004772 pp`，区间 `[+0.013824,+0.099571]`。

其余 S3 机制臂多数只有 seed 2。相对正确 center，whole-module off 为 `-1.269 pp`，但执行/RNG 不匹配；half GAM direction 为 `-1.585 pp`；SGD+EWC 为 `-13.296 pp`，但同时移除 GAM probe 和额外 forward，不能把差值全归因于方向质量。projection-off、rho、Fisher history/damping/anchor 及 lambda 结果不足以证明单组件必要性。center 的平均 EWC/clean 梯度比约为 `0.00144108`，alpha=1 为 `0.0000280262`，约降低 51.4 倍；按预注册“至少 2x 变化”的双向解释，这已触发 alpha0/alpha1 的 seeds 1/3 四个确认运行，但它们尚未执行。因此强 Fisher 机制归因仍有明确缺口。

因子空间 A/B 混合并不等于有效更新的精确凸组合，因为存在交叉项。sidecar 已验证交叉项恒等式和数值重构，已观测的相对间隙很小；这项代数事实仍不能证明性能因果性。配对 endpoint/retained Fisher 余弦、best-rank-10 SVD 残差和固定批次反事实 logit 差均未采集。边界 witness 只证明同一保存状态下观察器中立；跨运行逐位相等曾失败，不能写成完整训练逐位复现。

## 建议用于论文的限域表述

> 在 ImageNet-A、ViT-B/16 rank-10 Q/V LoRA、固定类别顺序和三个训练 seed 下，GAM 起点在旧任务训练集的配对随机方向上降低平均与尾部正 CE 损伤并扩大经验容忍半径；持续 GAM 链在 t9 虽产生更大的有效权重位移，仍表现出更低的旧任务 train/test 损伤和端点直线路径方向敏感度。固定后续 SGD 时，仅 Task 0 采用 GAM 也提高旧任务保持，但以约 3.41 pp 新任务准确率下降为代价，未形成稳健的两任务平均收益。held-out test 邻域扫描、较早 train 端点和 S3 三 seed 融合对照均不支持更广泛的“普遍平坦、普遍保持或融合有材料性增益”以及因果机制主张；Fisher 数值仅作为与这些限域观察一致的诊断。

## 后续实验边界

若论文要提出更强主张，需要以下预注册、定向的补证：完成已经触发的 S3 alpha0/alpha1 seeds 1/3 四个确认运行；若主张 GAM 单组件必要，增加多 seed 且匹配额外 forward/RNG 执行的组件对照；若主张 train/test 普遍更耐受，增加独立 seeds、类别顺序或数据集并预先把 held-out 指标列为共同主要终点；若要判定 S3 融合的小正效应或 FAA 等价，先固定功效与停止规则再增加配对 seeds；只有在声称 Fisher anchor 几何对齐时，才需要 endpoint/retained Fisher 配对余弦。

若论文采用上面的窄表述，本次结果已经足够，不必再启动 GPU：可以直接报告 RQ2 的限域观察和 S3 融合效应不确定；也无需为追求显著性追加运行、自动扫参，或在仅报告因子融合代数非等价时强制启动 SVD/logit 扫描。

## 证据保证范围

完成度检查显示 88/88 作业回执通过、212/212 冻结输入匹配、54/54 RQ2 诊断完成；fallback 数值预检定位了 23/23 个声明值。S3 只读审计还核验了 26/26 个原始任务矩阵、260 个 task 行均训练到 50 epoch、180 个 boundary archives 以及 factor 重构/RNG 检查，并给出 `passed_raw_integrity=true` 和空 issue 列表。标准 evidence checker 仍未解析成功，且没有独立完整性审计。这些回执、哈希、路径和重算检查提高了来源可信度，但不构成科学主张本身的证明。本报告因此保持同族、临时、`medium` 置信度。
