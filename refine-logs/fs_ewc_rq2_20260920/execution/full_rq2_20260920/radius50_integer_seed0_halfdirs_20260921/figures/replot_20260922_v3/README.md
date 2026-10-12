# B1 v3：扩展到 epsilon=0.1

[PNG](Figure_B1_neighborhood_seed0_radius50_20260922_v3.png) · [PDF](Figure_B1_neighborhood_seed0_radius50_20260922_v3.pdf) · [绘图脚本](plot_b1.py) · [21–24 数据检索](DATA_SEARCH.md)

第四图直接使用子图1同一个 dataframe 内全部107个实测半径的训练平均正损伤重新计算，阈值扩展为0.005以及0.01–0.1的每个0.01刻度。未插值、未增加GPU实验。

| epsilon | EWCLoRA 最大通过半径 | FS-EWCLoRA 最大通过半径 |
|---:|---:|---:|
| 0.005 | 8.25 | 9.25 |
| 0.01 | 11.00 | 12.25 |
| 0.02 | 14.50 | 16.25 |
| 0.03 | 17.00 | 19.25 |
| 0.04 | 19.00 | 20.00 |
| 0.05 | 20.00 | 20.00 |
| 0.06 | 20.00 | 25.00 |
| 0.07 | 20.00 | 26.00 |
| 0.08 | 25.00 | 27.00 |
| 0.09 | 26.00 | 28.00 |
| 0.1 | 27.00 | 29.00 |

定义：G(r)=mean_direction(max(mean_example CE(theta+r*u)-mean_example CE(theta),0))。先对每个方向的1259张训练样本CE求平均并计算相对锚点的变化，再取正部，最后对50个方向平均。epsilon表示该平均正CE增量的容許上界，不是总损失或相对百分比。

容忍半径为按r排序后前缀最大损伤不超过epsilon的最大已测半径。tolerance_bounds.csv同时记录通过点的G、首个失败半径及其G。第四图是第一图的阈值读数，不是新增独立机制测量。

已检索当前RQ2全部原始states文件、CSV、配置和日志，以及指定CopyFlatCV历史目录；没有找到21–24半径的实测记录。因此网格仍为0–20步长0.25、25–50步长1。epsilon=0.05时两方法读数同为20；epsilon=0.06/0.07时EWCLoRA仍为20。这些平台受20–25网格空缺影响，不是精确连续边界。

前三图的数据与v2逐字节一致，保留实测Fisher尖峰。只将数据点大小从2.7pt缩为1.6pt、描边缩为0.35pt；保留标签D_0、accuracy of task 0 (%)和靠近坐标轴的Weight drift r。图四保持2.7pt标记；颜色、形状和线型与前三图对应一致。

统计口径仍为当前正式ImageNet-A cohort，ViT-B/16 rank-10 Q/V LoRA，seed0单seed，固定Task-0锚点和20类旧任务头。损伤/准确率使用1259张训练样本、50个公共随机方向；Fisher使用固定256张训练样本、16个公共随机方向，为未乘lambda、未额外归一化的true-label per-example empirical Fisher trace。未显示测试曲线或跨seed误差条。

旧图全部保留。Pilot_new环境运行本目录plot_b1.py可复现PDF、PNG和逐点CSV。
