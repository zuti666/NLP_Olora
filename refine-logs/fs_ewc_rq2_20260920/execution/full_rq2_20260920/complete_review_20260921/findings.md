# 完整实验：结果到主张记录

判定：**partial**；置信度：**medium**。

证据支持若干严格限域的经验观察：GAM 起点在旧任务训练集局部邻域更耐受，持续 GAM 在 t9 可在更大有效参数位移下产生更小旧任务损伤，且仅塑造首任务解在相同后续 SGD 下提高旧任务保持。证据不支持把这些观察扩展为训练集与测试集普遍平坦、统一的因果 Fisher 机制、稳健的平均持续学习收益、EWC 的材料性保护，或 S3 融合和各子组件的已证明必要性。

## 各主张判定

- `rq2_c1_train_and_test_tolerance`：partial。在 ImageNet-A、ViT-B/16、rank-10 Q/V LoRA、固定类别顺序和训练 seeds 0/1/2 下，GAM 起点的旧任务训练集正 CE 损伤均值和 Q95 在 80 个非零半径乘 3 seeds 的 240/240 比较中均低于 SGD；多个非右删失阈值的经验容忍半径也扩大。
- `rq2_c2_smaller_damage_despite_larger_movement`：partial。在持续训练链 t9，GAM 的有效权重漂移更大（1.44156 对 1.25391），但旧任务 train/test CE 损伤更低，端点直线路径方向敏感度更低（0.01519 对 0.23965）；18/18 路径数值门通过。test D0 在 t1/t5/t9 的 9/9 seed-task 配对中更低。
- `b3_anchor_shaping_retention_under_same_future_sgd`：yes。固定后续优化器为 SGD 时，仅 Task 0 使用 GAM 在无 EWC 和 lambda=2000 两层均提高旧任务准确率：B-A 为 +4.127±0.500 pp，D-C 为 +4.233±0.650 pp，且 BWT 分别改善约 +3.910 和 +4.016 pp。
- `b3_ewc_material_protection`：no。在该 operating point，C-A 的平均准确率约为 0，D-B 为 +0.053±0.092 pp；S3 单 seed 的 lambda 0/200/2000/20000 响应也近乎平坦。
- `s3_factor_fusion_material_contribution`：no。按预注册 seeds 1/2/3，center-minus-matched-no-merge 的 FAA 差为 +0.089、+0.321、+0.071 pp，均值 +0.160333、样本 SD 0.139432、95% 配对 t 区间 [-0.186035,+0.506702]；区间包含 0，且略超出 ±0.5 pp 等价界。AAA 均值 +0.009707、SD 0.081460、区间 [-0.192650,+0.212064]，仅 AAA 满足 ±0.3 pp 实用等价。
- `s3_fisher_geometry_history_anchor_necessity`：no。Fisher damping、history、task normalization、endpoint anchor 与 lambda 对照多数仅 seed 2，准确率差小且探索性。alpha=1 将平均 EWC/clean 梯度比相对 center 降低约 51.4 倍，却没有相应稳定准确率效应；这说明该 operating point 下强机制归因尚未成立。
- `s3_gam_component_necessity`：partial。seed-2 中 half-direction FAA 比 center 低 1.585 pp，SGD+EWC 低 13.296 pp，说明完整 GAM 包及方向尺度与结果相关。

## 本次处理

选择收窄论文主张并保留所有反例。用户当前要求查看完整结果与结论，本次只读复核与CPU汇总，没有启动新GPU实验，也没有自动追加参数搜索。较窄经验观察可以基于现有证据报告；条件补测与更强主张的证据缺口不计入核心已完成数。

在 ImageNet-A、ViT-B/16 rank-10 Q/V LoRA、固定类别顺序和三个训练 seed 下，GAM 起点在旧任务训练集的配对随机方向上降低平均与尾部正 CE 损伤并扩大经验容忍半径；持续 GAM 链在 t9 虽产生更大的有效权重位移，仍表现出更低的旧任务 train/test 损伤和端点直线路径方向敏感度。固定后续 SGD 时，仅 Task 0 采用 GAM 也提高旧任务保持，但以约 3.41 pp 新任务准确率下降为代价，未形成稳健的两任务平均收益。held-out test 邻域扫描、较早 train 端点和 S3 三 seed 融合对照均不支持更广泛的“普遍平坦、普遍保持或融合有材料性增益”以及因果机制主张；Fisher 数值仅作为与这些限域观察一致的诊断。

## 更强主张需要的证据

- 按预注册完成 S3 alpha0/alpha1 的 seeds 1 和 3 四个确认运行；只用于检验 Fisher damping/geometry 的更强主张，不自动扩展参数搜索。
- 若声称 GAM 单组件必要，做多 seed、额外 forward 与 RNG 执行匹配的方向/组件对照。
- 若声称 train 与 test 普遍更耐受，增加预注册的独立 seeds、类别顺序或数据集，并把 held-out 指标作为共同主要终点。
- 若声称 S3 融合有小正效应或 FAA 等价，先做功效设计与固定停止规则，再增加配对 seeds。
- 仅在声称 Fisher anchor 几何对齐时，采集 endpoint/retained Fisher 配对余弦和相应匹配干预。

## 当前较窄主张无需追加

- 陈述本次严格限域的 RQ2 观察无需再启动 GPU 实验。
- 只报告‘S3 融合效应当前不确定’无需为取得显著性追加运行。
- 只报告因子融合并非精确的有效更新凸组合、且观测交叉项很小时，无需自动启动 SVD 或固定批次 logit 扫描。
- 无需基于当前结果选择更优超参数或启动无界参数搜索。

## 审读与可追溯性

- 二次审读：gpt-5.6-sol，ultra，fresh context；same-family/provisional。
- 完成与来源核验通过；完整独立实验完整性审计不可用。标准evidence_check.py未解析到，本地fallback只验证23个精确数值存在与来源哈希，不验证因果主张。
- 保留原计划S3 seeds1/2/3作为主推断。seed2中心误配已根据原始记录纠正；事后seeds2/3仅作敏感性口径。
- [完整结果](FINAL_RESULTS_AND_CONCLUSIONS.md) / [审读全文](REVIEW_RESPONSE.md) / [结构化判定](REVIEW_VERDICT.json) / [审读请求](REVIEW_REQUEST.md)
