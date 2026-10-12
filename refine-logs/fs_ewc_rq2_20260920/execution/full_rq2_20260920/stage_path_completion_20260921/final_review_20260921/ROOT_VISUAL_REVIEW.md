# 主代理渲染图复核

2026-09-21，主代理在发起独立验图前，实际打开了最终重绘的两张 PNG；不以脚本检查代替图面检查。

| 最终图 | PNG SHA-256 | 图面检查 |
|---|---|---|
| B1 | `161a5937de196a2b3ffba3d4587cfe47633b0d9dfb30b0cf34f3db1d32230e14` | 1×3；标签、刻度和内部图例可读；横轴等距；16个 Fisher 实测半径及 r17 尖峰/完整样本SD保留；未绘制测试损伤或容忍半径 |
| B2 | `02997a423db09af42de4b61966927bc45f2a087c59bbef94beb293629e386c6c` | 1×4；位移、损失变化、准确率、归一化路径敏感度顺序正确；前三图十阶段，最后图T2–10且T1留空；前三图内部图例，末图无重复图例遮盖SD带 |

字体为 Times New Roman，图面纵向压缩，无 a/b/c/d 标题、标签裁切或可见文本重叠。第四图沿用同一颜色/线型，含义由共享方法编码与图注说明。

原始数据重新计算及数值核验见 B1_STATISTICS、B2_STATISTICS 和 replot_validation；r17 峰值的复测见 FISHER_PEAK_VERIFICATION.json。统计不混用历史 ImageNet-R 与本批正式 ImageNet-A，不将节点或方向作为独立训练 seed。

本记录只对应上述图稿哈希。独立第二代理的意见另见 SECONDARY_FIGURE_REVIEW.md。

## 2026-09-22 标注更新复核

B2 第一幅纵轴添加小写 $\delta$，完整显示为 “Weight drift $\delta$”。主代理实际打开更新 PNG：符号清晰、标签未裁切，四幅曲线、图例与原排版保持一致。重绘后逐文件确认所有 B2 数值 CSV、B1 图稿及原队列回执哈希保持不变。上方表格是原版验收记录；当前图稿哈希见 REPLOT.json 与 FINAL_VALIDATION.json，原文件已归档至 plot_revisions/before_delta_label_20260922。
