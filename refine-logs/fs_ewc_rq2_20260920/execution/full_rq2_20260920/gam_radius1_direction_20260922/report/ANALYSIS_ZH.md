# 两组 GAM 训练半径的实验结论

六个 Task0 模型（两配置 × seed0/1/2）和十二条固定方向诊断全部完成，共新增 132 个位置。仅使用物理 GPU 0、1。诊断网格为 `0,0.5,...,5`；每条方向在对应新模型上计算一次，所有半径固定该方向。

两组训练半径按用户最终指令设置：`gam_grad_rho=0.05, gam_grad_norm_rho=1.0`，以及 `gam_grad_rho=1.0, gam_grad_norm_rho=1.0`。有效 `as1` 别名同步设置，实际优化器参数及配对初始化、训练数据顺序均已验证。

## 结果

下表是三 seed 均值。D 是相对各自锚点的旧 Task0 平均交叉熵增量。Task0 使用正梯度方向，Task1 使用新任务负梯度方向；两列均测旧 Task0 的损失。

| 方法及训练半径 (probe, norm) | 原点训练 CE | 原点训练准确率 % | 原点 raw Fisher | Task0 方向 D(r=1) | Task1 方向 D(r=1) |
|---|---:|---:|---:|---:|---:|
| EWCLoRA | 0.091653 | 97.988 | 1451.95 | 1.43414 | 0.99269 |
| 旧 FS-EWCLoRA (0.05,0.2) | 0.056218 | 99.153 | 634.51 | 1.86758 | 1.46447 |
| 新 FS-EWCLoRA (0.05,1) | 0.107097 | 98.173 | 1007.94 | 2.88503 | 2.15127 |
| 新 FS-EWCLoRA (1,1) | 2.998642 | 6.910 | 0.934 | 0.05852 | 0.00307 |

**只将外层半径从 0.2 增大为 1，没有改善本次两类定向扰动的耐受性。** 对 Task0、Task1 两种方向，在全部十个非零已测半径上，新 `(0.05,1)` 的三 seed 平均 D 和绝对 CE 均高于旧 FS-EWCLoRA，也高于 EWCLoRA。这是所测网格的描述，半径不是十次独立重复；三 seed 逐项数值和配对差异均保留在 CSV 中。

**两个半径均为 1 导致严重欠拟合，不能用其较低的损失增量或 Fisher 宣称获得了更好的受保护解。** 该组原点训练准确率仅 6.91%，CE≈2.999，接近 20 类均匀预测的 ln(20)≈2.996。以 Task1 方向 r=1 为例，D 虽然只有 0.00307，但绝对 CE 仍为 3.00171，准确率仅 6.78%。低增量主要需要在这个很差的起点条件下理解；这并不证明模型仍保留了旧任务能力。

Fisher 也并非在 `(1,1)` 邻域稳定较低。Task0 方向 r=3.5 时，seed1 的实测 trace 达到 76,107,577.07；另外两 seed 分别为 3.3636 和 277.3108。这个峰必须保留，均值和标准差由该 seed 主导，不能把曲线解读为一致改善。

## 方向发生了什么变化

完整计算见 [梯度方向分析](GRADIENT_ALIGNMENT.md)。将各模型 Task0 上升方向记为 u0、Task1 下降方向记为 u1。在 `(0.05,1)` 中，平均 cos(u0,u1) 从旧 GAM 的 0.1720 增至 0.2240，旧任务梯度范数也增大；沿 u1 的局部旧任务损失斜率 g0·u1 从 0.1283 增至 0.2415，约为原来的 1.88 倍。这与更大的局部旧任务敏感度一致；有限半径结果仍以实际 CE 测量为准，不能单由一阶斜率推出。

新 `(0.05,1)` 与旧 GAM 同类方向仍有较强对齐：Task0 方向余弦为 0.775–0.793，Task1 为 0.886–0.918。`(1,1)` 与旧方向的绝对余弦均小于 0.014，且训练拟合失效；其“小损伤”不能与良好拟合模型的耐扰动性等同。

## 新图与显示方式

均为 Times New Roman、一行三列、图内 legend、相同颜色和方法名称。第一列为有符号平均 loss change，第二列为旧任务 accuracy (%)，第三列为未额外归一化的 empirical Fisher。横坐标均为线性的 weight drift r。

- Task0 上升方向：[可读版 PNG](../figures_readable/task0_positive/Figure_task0_positive_training_radius_intervention_log_fisher.png) / [PDF](../figures_readable/task0_positive/Figure_task0_positive_training_radius_intervention_log_fisher.pdf)。
- Task1 下降方向：[可读版 PNG](../figures_readable/task1_negative/Figure_task1_negative_training_radius_intervention_log_fisher.png) / [PDF](../figures_readable/task1_negative/Figure_task1_negative_training_radius_intervention_log_fisher.pdf)。

可读版第三列使用对数纵轴，粗线仍为原始 Fisher 的算术均值，淡线为逐 seed 实测曲线；不做几何平均、不加 floor、不截峰，也不乘 lambda。前两列保持均值±样本标准差。第三列不绘制标准差带，因为本次巨大波动使均值减标准差为负，不能直接放在正数对数轴上。原始线性纵轴、三列均值±标准差版本完整保留在 `../figures/task0_positive/` 和 `../figures/task1_negative/`。

## 数据与配置范围

本次沿用当前正式 RQ2 的 **ImageNet-A** 三 seed 配对协议：ViT-B/16、Q/V LoRA rank10、固定类别顺序、20类/任务，Task0 训练1259张、测试307张；训练50轮。方向梯度分别用全部1259张 Task0 或997张 Task1 训练图计算；Fisher 用相同256张旧任务训练子集的 true-label per-example CE 梯度平方范数均值，没有额外归一化或 EWC lambda。

引用的 `s3_expand_r1p0_s2.json` 确实采用 `(0.05,1)`，但其完整配置还涉及不同模型分支与 in-task Fisher/weight merge。本次按已说明的受控比较保持当前正式 RQ2 其余设置固定，**不是该十任务 FAA=56.881% 的完整配置复现，也不是历史 ImageNet-R 实验**。训练半径作用于 LoRA 因子/分类头参数空间，图中 r 则是有效 Q/V 权重全局位移，不能直接混用两个尺度。

这些实验回答的是：改变 GAM 训练半径后，新锚点及其诱导的两类固定方向如何变化。它们没有重新运行随机方向邻域扫描，也不是实际未来任务训练轨迹，因此不能直接替代 B1 随机邻域证据或 C 的真实序列证据。

完整数值、样本标准差和配对比较见 [RESULTS.md](RESULTS.md)、[逐 seed 数据](../figures/all_per_seed.csv)、[均值与标准差](../figures/all_mean_sample_sd.csv)、[配对差异](../figures/paired_mean_sample_sd.csv)。
