# 逐任务路径诊断已完成

2026-09-21，物理 GPU 0 和 3 分别完成 EWCLoRA / FS-EWCLoRA 的 Task1–Task8：16/16 新测量完成，合并原有两个 Task9 后为 18/18 条路径。全部使用固定的同一批 256 个 Task0 训练样本，并在 9 节点积分时通过数值收敛与同样本损失界检查。无需重训练。

[新 C 图 PDF](../Figure_C_sequence_stage_paths.pdf) · [PNG](../Figure_C_sequence_stage_paths.png) · [逐阶段结果](C_stage_path.csv) · [逐节点结果](path_nodes.csv) · [检查汇总](TREND_CHECKS.json)

## 四个子图

1. **实际累计位移**：每个阶段的模型相对本方法 Task0 anchor 的有效 Q/V 权重联合范数。这里没有把各阶段归一化到相同半径。
2. **旧任务损失变化**：固定旧 20 类分类头下的 Task0 测试交叉熵减去各自 Task0 anchor 的测试交叉熵，保留有符号值。负值表示改善。
3. **旧任务准确率**：同一 Task0 测试集、固定旧分类头的准确率。
4. **路径敏感度**：原来的 Task9 两根柱改为逐阶段曲线。每个点对应从 Task0 anchor 到该阶段 checkpoint 的独立直线段，用沿该段的真实标签、逐样本 empirical Fisher 二次型积分，再除以该段位移范数的平方。

定义 Δ_T 为当前学习阶段模型相对 Task0 anchor 的有效 Q/V 权重差，r_T = ‖Δ_T‖，则

$$\overline S_T=\frac{1}{r_T^2}\int_0^1\Delta_T^\top F_{0,256}^{\mathrm{emp}}(W_0+s\Delta_T)\Delta_T\,ds.$$

横轴 T 表示已学任务数：T=1 对应代码中的 Task0，T=10 对应 Task9。T=1 的 r=0，因此归一化敏感度未定义，不人为补零。连接线只帮助阅读离散阶段的趋势；每个标记点均有实测 checkpoint/path 记录。

## 这次结果支持的表述

在这条 **历史 ImageNet-A seed0** 序列的全部九个后续阶段，FS-EWCLoRA 相较 EWCLoRA 均表现为：累计位移更大、Task0 测试有符号损失变化更小、单位位移平方的路径平均敏感度更低。敏感度比值为基线的 4.02%–24.74%。最终阶段保持原结果：0.3680576 → 0.0147931。

这支持在所测实际后续学习序列中，“更少遗忘不能仅由更小位移解释”，并提供逐阶段路径敏感度证据。曲线保留真实起伏，不表示每阶段单调变化。路径是各阶段端点之间的直线段，未测优化器每一步的真实弯曲轨迹。一个 seed 的结果也不等于跨 seed 结论。

这里延续原 C 图的历史 ImageNet-A cohort；不把它重新标注为 ImageNet-R，也不与另行训练、使用全 1259 样本 Fisher 的三 seed cohort 混合。未来任务两条历史训练线本身分别使用 SGD+EWC 与 GAM+EWC，因而 C 图展示各自后续学习下的结果；受控 anchor 因果分解仍由 A 实验承担。

最大端点损失复现误差：2.74e-7；最大 5/9 节点积分相对差：2.93%，低于预设 5% 阈值。逐路径的配置、checkpoint 哈希、样本身份、积分节点和结果来源均在 CSV 与原始 manifest 中。
