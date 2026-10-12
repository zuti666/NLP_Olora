## 当前 B 图：训练均值三图版（2026-09-21）

当前 B 图为 Times New Roman 的一行三列：平均正损伤、训练准确率、平均 raw Fisher trace。已按用户要求删除第四幅容忍半径图。所有轴为线性，横轴为 Weight drift，仅显示训练均值。前三图来自同批历史 ImageNet-A seed0 扰动状态；loss/accuracy 使用 1259 个训练样本，Fisher 使用其中固定的 256 个样本。两处 Figure_B_neighborhood / Figure_B_neighborhood_merged 输出一致。最新图注见 B_caption.md（figures_redesign_20260920 目录），最新来源核查见 B_THREE_PANEL_SOURCES.md；下方早期记录与历史审查对应各归档版本。


# B/C 图重排与指标说明

本轮只使用原有测量值重绘，原图保留。方法显示名称统一为 EWCLoRA 和 FS-EWCLoRA；CSV内部仍保留原始sgd/gam标识，图注说明对应历史SGD+EWC与GAM+EWC。图内去掉数据集页脚、Task0起点解释和评价性文字，数据集、样本和测量协议保留在图注中。

## 图 B：三幅邻域诊断

[PNG](Figure_B_neighborhood_merged.png) / [矢量 PDF](Figure_B_neighborhood_merged.pdf)

| 子图 | 指标 | 样本范围 |
|---|---|---|
| 左 | 平均正损伤 G(r) | 1259 个 Task0 训练样本 |
| 中 | 平均训练准确率 | 同样 1259 个 Task0 训练样本 |
| 右 | 平均 raw empirical Fisher trace | 从上述训练样本固定选出的 256 个样本 |

每方法共用 97 个状态（1 个 anchor + 6 个非零半径 × 16 个方向），半径、方向、checkpoint 和固定头在三个指标间一致。两个方法共享数据、Fisher 子集及扰动方向，但各有自己的 checkpoint 和头。容忍半径数据仍保存在 source_data 中，当前主图不再绘制。

## 图C：一行四列

[PNG](Figure_C_sequence_row.png) / [矢量PDF](Figure_C_sequence_row.pdf)

新顺序为原(d)→(b)→(a)→(c)，重新标为(a)至(d)：

1. 实际累计位移 r_T。
2. 旧任务损失变化 Delta L_0(T)。
3. 旧任务准确率。
4. 最终端点之间的归一化方向敏感度柱状图。

前三幅各自放置 EWCLoRA / FS-EWCLoRA 图例，第四幅直接用两方法名称作为类别横坐标。横轴T为已经学习的任务总数，1…10；T=1对应完成Task0，T=10对应完成Task9。保持负损伤、真实波动以及每个实际端点，不做平滑或单调化。

## 最后柱状图的含义

令 Delta=W_9−W_0，r=||Delta||，u=Delta/r。柱高为

\[
\overline S_{\mathrm{path}}
=\frac{1}{\|\Delta\|^2}\int_0^1\Delta^\top F_0^{\mathrm{emp}}(W_0+s\Delta)\Delta\,ds
=\int_0^1\frac1N\sum_{i=1}^N\left[g_i(W_0+s\Delta)^\top u\right]^2ds.
\]

它衡量：从旧任务起点到最终模型的直线段上，旧任务损失对实际更新方向的平均梯度敏感程度，并消除位移长度平方的直接影响。较低的柱意味着沿这条方向的平均敏感度较低；柱高本身不是loss、accuracy、Fisher trace或位移。

现有柱值为 EWCLoRA 0.368058，FS-EWCLoRA 0.014793。该对比使用固定256个旧任务训练样本、固定Task0分类头，在0到1的9个节点做Simpson积分。它只有最终Task0→Task9这一条端点直线段测量，不能解释为沿真实优化器全部更新步骤的积分，也不能当成每个阶段均已测量的曲线。C的loss/accuracy则来自307个固定旧任务测试样本。

本轮图仍来自原有 ImageNet-A seed0 数据；图内按用户要求精简文字，图注保留这一来源。它与此前ImageNet-R主表分开。PDF为矢量图，PNG为300dpi，图注和可重现脚本一并保留。

图注：[B_caption.md](B_caption.md)、[C_caption.md](C_caption.md)；论文插入片段：[latex_includes.tex](latex_includes.tex)。复现入口：[gen_fig_B_neighborhood_merged.py](gen_fig_B_neighborhood_merged.py)、[gen_fig_C_sequence_row.py](gen_fig_C_sequence_row.py)。导出检查见 [EXPORT_CHECKS.json](EXPORT_CHECKS.json)：两张PDF均为单页矢量输出，字体嵌入，无栅格图像。未编译论文。
