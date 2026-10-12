# 正式三 seed 补测：完成统计与重绘

2026-09-21 核验：补测及自动汇总均已完成。迁移后的 GPU1/4/6 队列于 19:38:42 UTC 完成。本目录重新读取原始日志计算统计和绘图，不改动原始观测、配置、队列完成回执或此前图稿。

## 完成范围

| 项目 | 已完成范围 |
|---|---|
| B2 新增路径 | Task2、3、4、6、7、8 × 3 seed × 2 方法，36/36 条 |
| B2 合并路径 | 复用 Task1、5、9 的 18 条，合计 54/54 条，覆盖 T=2–10 |
| B2 端点 | 3 seed × 2 方法 × 10 阶段，60 个 |
| B2 积分节点 | 502 个；52 条路径使用 9 节点，seed2 两方法的 Task8 路径细化为 17 节点 |
| B1 加点 | 6/6 作业；半径 11–19 × 16 方向 × 3 seed × 2 方法，864 个新增非零扰动状态，另核验 6 个重复原点 |
| B1 合并 Fisher 半径 | 0、0.5、1、2、5、10、11、12、13、14、15、16、17、18、19、20，共 16 个实测半径 |

全部最终积分稳定性与数值检查通过。T=1 时位移为零，归一化路径敏感度未定义，图中留空。

## 最终图稿

- **B2 逐任务变化，1 行 4 列**：[PNG](figures/Figure_B2_sequence_stage_paths_all_stages.png) · [矢量 PDF](figures/Figure_B2_sequence_stage_paths_all_stages.pdf) · [图注与指标定义](figures/B2_all_stages_caption.md)。依次为累计权重位移、旧任务测试损失变化、旧任务测试准确率、归一化方向路径 Fisher。
- **B1 训练邻域，1 行 3 列**：[PNG](figures/Figure_B1_neighborhood_extended.png) · [矢量 PDF](figures/Figure_B1_neighborhood_extended.pdf) · [图注与测量范围](figures/B1_radius_extended_caption.md)。依次为训练平均正损伤、训练准确率、训练子集 raw Fisher；不绘制测试损伤、Q95 或容忍半径。

两图使用 Times New Roman、压缩纵向高度、线性横坐标、图内图例。B2 前三幅给出图例，第四幅沿用相同颜色和线型，避免重复图例遮挡数据。所有阴影均为三个训练 seed 的样本标准差（ddof=1），不是方向间标准差，也不是置信区间。

2026-09-22 标注更新：B2 第一个子图纵轴改为 “Weight drift $\delta$”。PNG/PDF 已重新生成；数值 CSV、其余子图数据及 B1 图稿保持不变。

## B2 统计结果

末阶段 T=10（Task9）结果，三 seed 均值 ± 样本标准差：

| 指标 | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|
| 累计权重位移 | 1.253913 ± 0.038792 | 1.441559 ± 0.010447 |
| 旧任务测试损失变化 | 0.229435 ± 0.035624 | 0.033362 ± 0.013129 |
| 旧任务测试准确率（%） | 81.107492 ± 1.174447 | 86.644951 ± 0.325733 |
| 归一化路径敏感度 | 0.239649 ± 0.037153 | 0.015193 ± 0.000273 |

九个未来阶段中，每个 seed 的 FS-EWCLoRA 路径敏感度和测试损失增量均更低、累计位移均更大、旧任务测试准确率均更高：每项均为 27/27 个阶段配对。这里仍只有 3 个独立训练 seed，不能将 27 个阶段配对当成 27 个独立重复。T=10 的路径敏感度按方法均值计算降低 93.66%。

这些结果支持本批实际后续学习中“移动更大，同时旧任务测试损伤与路径方向敏感度更低”。训练损失增量有 5/27 个反例（T=2 的三个 seed，以及 T=6 的 seed0/1），故不能写成所有阶段、所有数据划分都更好。曲线也不要求严格单调增长。

最后一幅的指标为

\[
\overline S_{\mathrm{path},t}
=\frac{\int_0^1\Delta_t^\top F_0^{\mathrm{emp}}(W_0+s\Delta_t)\Delta_t\,ds}{\|\Delta_t\|^2},\qquad \Delta_t=W_t-W_0.
\]

每个任务点来自自己的 Task0→Taskt 端点直线段积分；9/17 个节点是同一条直线上的位置，不是任务数，也不是优化器全部训练步骤的实际轨迹。先计算每个 seed 的归一化积分，再跨 seed 平均。

[全部阶段统计及解释](B2_STATISTICS.md) · [逐 seed CSV](B2_STATISTICS.csv) · [完整数值与来源 JSON](B2_STATISTICS.json)

## B1 统计结果

前两幅仍使用全部 1259 张训练图、100 个公共方向、81 个半径（0–20，步长 0.25）。第三幅使用固定 256 张训练子集、16 个公共方向及 16 个实测半径；没有把损伤扫描的 81 个点当作 Fisher 点，也没有插值生成新的测量。

FS-EWCLoRA 的方向平均 raw trace 在全部 48/48 个 seed×半径配对中更低。r=20 时，训练平均正损伤为 0.046783→0.033971，训练准确率为 96.33%→98.16%，raw trace 为 2650.23→1403.40。

**r=17 的 EWCLoRA 峰值保留。** 它主要来自 seed0、方向 u014 的一个训练样本高梯度。同配置重复测量 r=16/17/18 与原始 trace 完全一致，并对峰值样本用独立 `torch.autograd.grad` 验证梯度平方和。图中完整保留均值与标准差；没有剔除、平滑或替换观测。复测只作核验，不增加正式样本量。详见 [复测证据](FISHER_PEAK_VERIFICATION.json)。

这些结果支持训练侧随机邻域敏感度下降，不能推出测试保护区域扩大。原有密集损伤网格的容忍半径不受本次 Fisher 加点影响；ε=0.01 时训练半径三 seed 均值 10.5833→12.1667，测试为 6.9167→6.0000。有限方向的平均损伤阈值不是所有方向均安全的保证。

[B1 完整统计](B1_STATISTICS.md) · [CSV](B1_STATISTICS.csv) · [完整数值与来源 JSON](B1_STATISTICS.json)

## 数据批次与复现

本目录使用正式新训练的 **ImageNet-A 三 seed（0/1/2）**，200 类分成 10 个任务、每任务 20 类；模型为 ViT-B/16，12 个 block 的 Q/V 使用 rank-10 LoRA。展示名 EWCLoRA / FS-EWCLoRA 对应当前视觉 SGD+EWC / GAM+EWC 实现，GAM 在后续任务中继续启用。

B2 路径每个节点使用全部 1259 张旧任务训练图；图中损失与准确率使用全部 307 张旧任务测试图和固定 Task0 的 20 类分类头。Fisher 使用 true-label per-example 梯度，坐标为 24 个有效 Q/V 权重矩阵；没有乘 EWC λ，没有 trace 归一化、floor 或正则项。

历史 ImageNet-R、seed0、256 样本的 0.3681→0.0148 属于另一批 checkpoint 与测量口径，不能与本图拼接。

统计脚本：[analyze_b1.py](analyze_b1.py)、[analyze_b2.py](analyze_b2.py)。重绘脚本：[replot_final.py](replot_final.py)。使用原 Pilot_new 环境运行三个脚本可复算和重绘；这些是 CPU 统计/绘图，不会启动训练。重新生成后需要重新验图，因为原验收只对应当时文件哈希。

原始队列状态：[pipeline_state.json](../campaign_idle_gpu146/pipeline_state.json)。重绘回执：[REPLOT.json](REPLOT.json)。[主代理图面检查](ROOT_VISUAL_REVIEW.md)与[独立图稿复核](SECONDARY_FIGURE_REVIEW.md)均通过；最终文件哈希及验收汇总见 [FINAL_VALIDATION.json](FINAL_VALIDATION.json)。
