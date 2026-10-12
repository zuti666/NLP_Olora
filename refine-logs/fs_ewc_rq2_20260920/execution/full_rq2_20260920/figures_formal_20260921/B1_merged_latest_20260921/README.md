# B1 最新四子图合并版

[PNG](Figure_B1_neighborhood_merged_latest.png) · [矢量 PDF](Figure_B1_neighborhood_merged_latest.pdf) · [完整十档训练／测试阈值](TEN_THRESHOLDS.md)

保留 Times New Roman、一行四列、图内图例，以及四图一致的方法颜色、线型和标记。前三图横轴为 Weight drift $r$。旧版本均保留；本次仅使用 CPU 读取已完成结果作图。

1. 平均正 CE 损伤：1259 个 Task-0 训练样本，100 个配对方向，81 个半径（0–20，步长0.25）。
2. 训练准确率（%）：与图1相同的实测数据。
3. Fisher trace：256 个固定训练样本，16 个配对方向；合并原有7个半径与11–19九个新半径，共16个。所有6组补测已经过原点、模型/数据/坐标/方向身份与原始数值重算校验。Fisher不乘EWC的lambda，不归一化。EWCLoRA在r=17的尖峰主要来自seed0，按原值保留，纵轴覆盖均值±样本标准差。
4. 训练容忍半径：保留新增五个阈值。先对各seed的81点损伤曲线取前缀最大值，再反求最大满足阈值的已测半径，最后计算三个seed均值和样本标准差；本次重新核对了全部十档训练阈值。

| epsilon | EWCLoRA 均值 | FS-EWCLoRA 均值 |
|---:|---:|---:|
| 0.0025 | 5.750 | 6.583 |
| 0.0075 | 9.333 | 10.833 |
| 0.015 | 12.500 | 14.417 |
| 0.03 | 16.667 | 18.917 |
| 0.04 | 18.667 | ≥20（扫描上限） |

epsilon是“样本平均交叉熵相对各自锚点的正增量，再对方向平均”的阈值，不是整个数据集CE求和，也不是保证每个样本/每个方向安全。最后一点≥20表示达到扫描上限；没有恢复上限横线或平台。线段仅为视觉连接，不插值求容忍半径。

原图1、图2以及最新阈值版图4的数值均保持一致。新增Fisher不能改变基于原始损伤曲线得到的图4。阴影/误差棒为3个训练seed的样本标准差。

数据来自正式三seed ImageNet-A队列，与历史ImageNet-R诊断分开；本次没有切换数据集或模型。来源：[Fisher完成记录](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/report/radius_complete.json)；详细图注见[CAPTION.md](CAPTION.md)，来源哈希与一致性检查见[Figure_B1_neighborhood_merged_latest_validation.json](Figure_B1_neighborhood_merged_latest_validation.json)。

复现：`/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python plot_b1_merged_latest.py`。
