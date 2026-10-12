# B1 / B2：调整顺序后的整组图

本次仅调整子图位置。统一保留7.6×1.65英寸、Times New Roman、蓝橙配色、显式纵轴上限、1.6pt标记及图内图例；B2继续不绘制标准差阴影。

| 图 | 从左到右 |
|---|---|
| B1 | 准确率 → 损失变化量（平均正损伤）→ 容许损伤ε与容忍半径 → raw Fisher |
| B2 | 准确率 → 有符号损失变化 → 权重位移δ → 方向路径Fisher |

- **B1**：[PNG](Figure_B1_accuracy_loss_tolerance_fisher.png) · [PDF](Figure_B1_accuracy_loss_tolerance_fisher.pdf) · [新版图注](B1_caption.md)
- **B2**：[PNG](Figure_B2_accuracy_loss_drift_fisher.png) · [PDF](Figure_B2_accuracy_loss_drift_fisher.pdf) · [新版图注](B2_caption.md)

所有7份数值CSV均与上一版逐文件SHA一致；B1为ImageNet-A seed0的训练侧邻域均值，B2为正式三seed均值。B1平均正损伤的统计未改成带符号损失；容忍半径仍从相同损伤数据计算，ε覆盖至0.1，r=21–24缺测情况保留。B2的SD仍保存在CSV中，仅不绘制阴影；T=1路径敏感度仍未定义留空。

两图重新读取验证过的数据生成，不重新进行GPU测量；此前图稿、原始测量及统计均未改动。

绘图脚本：[plot_reordered.py](plot_reordered.py)。生成记录：[GENERATION.json](GENERATION.json)。图面复核见VISUAL_REVIEW.md。LaTeX引用片段见[latex_includes.tex](latex_includes.tex)。
