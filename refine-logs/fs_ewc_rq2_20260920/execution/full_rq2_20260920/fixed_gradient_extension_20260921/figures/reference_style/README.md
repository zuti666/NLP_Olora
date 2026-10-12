# 固定 Task0 正梯度方向：与随机半径图一致的样式

6 组正式测量均已完成：3 seeds × 2 methods，每组 93 个半径，共 558 个测量位置；原队列于 2026-09-21 21:49:45 UTC 完成并通过发布校验。

- 全范围：`Figure_B1_fixed_task0_gradient_radius50_reference_style.pdf` / `.png`，r=0–50。
- 原点附近：`Figure_B1_fixed_task0_gradient_near_origin_reference_style.pdf` / `.png`，r=0–2。
- 重现：使用 Pilot_new 环境运行本目录 `plot_fixed_gradient_reference_style.py`，仅读取已校验结果，不进行模型训练或测量。
- 完整逐 seed 数据：`../all_per_seed.csv`；每张图的绘图均值与标准差保存在对应 `_data.csv`。

## 图的定义

每个方法/seed 在自己的 Task0 模型上，使用全部 1259 张旧任务训练图计算平均 CE 的正梯度，归一化一次并固定方向 u；每个半径直接设置 theta(r)=theta0+r*u。两种方法不共享方向，且这些点不是实际未来任务训练轨迹。

三个子图分别为：相对各自锚点的有符号训练 CE 变化、训练准确率（%）、当前扰动位置重新测量的 empirical Fisher trace。前两项覆盖全部 1259 张训练图；Fisher 使用固定的 256 张训练图，按样本平均平方梯度范数，不乘 lambda、不做额外归一化。曲线表示跨 3 个 seed 的均值，损失符号上的横线同样表示该均值；阴影为样本标准差（ddof=1），并非置信区间。标准差带完整显示，因此非负量的均值减标准差可能落在零以下；这不表示原始 Fisher 为负。

数据来自正式 ImageNet-A、ViT-B/16、rank-10 Q/V LoRA 实验，扰动坐标是 24 个有效 Q/V 权重矩阵；旧任务 20 类分类头保持固定。没有混入历史 ImageNet-R 结果。

## 显示与解读

Times New Roman、一行三列、图内图例、EWCLoRA 蓝色实线圆点、FS-EWCLoRA 橙色虚线菱形，与指定参考图一致。Fisher 保留原始线性数值及大峰值；未裁剪、平滑或按最大值归一化。

固定梯度方向下损失在小半径增加已由六组真实模型的有限差分验证。两种方法的曲线会交叉，不能将此图描述为 FS-EWCLoRA 在所有半径均具有更小损失或 Fisher。全范围图与局部放大图使用完全相同的数据。

原自动发布结果、此前随机方向图均保留；本目录是另存的样式版本。数据来源、字体和布局校验见各图 `_validation.json`，最终图片检查见 `VISUAL_REVIEW.json`。
