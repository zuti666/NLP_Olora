# 蓝色 / 绿色两方法四子图

本目录重新读取已完成的实验数据，保留蓝色 EWCLoRA 与绿色 FS-EWCLoRA，并新增第四列容忍半径。绿色对应本次训练参数 `gam_grad_rho=0.05, gam_grad_norm_rho=1.0`；图内图例只显示方法名。旧实验与此前图片均保留，无新增训练或 GPU 测量。

- Task0 固定上升方向：[PNG](task0_positive/Figure_task0_positive_blue_green_tolerance.png) / [PDF](task0_positive/Figure_task0_positive_blue_green_tolerance.pdf)。
- Task1 固定下降方向：[PNG](task1_negative/Figure_task1_negative_blue_green_tolerance.png) / [PDF](task1_negative/Figure_task1_negative_blue_green_tolerance.pdf)。

四列依次是旧 Task0 训练交叉熵增量、准确率、empirical Fisher trace、容忍半径。Times New Roman、一行四列、图内 legend、无 a/b/c/d 或参数标注。前三列横轴为线性的有效 Q/V 权重位移 r=0,0.5,...,5。第三列恢复线性纵轴，范围覆盖所保留两组的全部逐 seed Fisher 值，没有额外归一化、lambda、floor 或截峰。

所有主曲线均为三个训练 seed 的算术均值。前两列阴影是 sample SD；第三列淡线是各 seed 实测值；第四列按参考图采用均值曲线，逐 seed 值及 SD 另存表格。对称 SD 带可能超出物理边界，它不表示负准确率。

## 第四列的参考实现与适配

直接导入并调用参考图脚本的 `measured_tolerances`：

[参考绘图代码](../../radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v2/plot_b1.py)

每个模型、每个 seed、每个方向单独计算：

\[
G_s(r_j)=\max(D_s(r_j),0),\qquad
\widehat r_{\epsilon,s}=\max\{r_j:\max_{i\le j}G_s(r_i)\le\epsilon\}.
\]

先找到前缀最大损伤首次超过阈值的半径，返回前一个已测半径；等于阈值仍算通过。**先逐 seed 反演，再取三个半径的均值**，不对跨 seed 平均损伤曲线反演，不插值。0.8333 等数值来自三个已测网格半径的平均，不是新增测量半径。当前所选数据全部 D>=0，因而 D 与正部 G 完全相同。

第四列阈值为 **0.25、0.5、1、1.5、2、2.5、3**。这些是旧任务完整训练集上、平均每张图片交叉熵增量的允许值，不是总样本损失之和、准确率百分点或扰动半径。它们还没有乘 EWC 正则权重。

参考图的六档 0.005–0.05 在本次数据上全部得到已测网格半径0，因为每条曲线在 r=0.5 已超过0.05。原六档全部保存在 CSV 中，但主图改用当前损伤量级的七档；这不表示测过0与0.5之间的半径。

主图所有84个逐 seed 阈值结果均有通过点及下一失败点，两点间隔0.5，没有右删失。如果未来某 seed 在扫描上限仍通过，绘图均值将留空并保存下界，不能只平均未删失 seeds，也不画扫描上限横线。

这是一条固定方向上的已测网格容限，不是随机方向平均，也不是球内所有方向的保证。它是损伤曲线的阈值读数，不是独立的新机制测量；连接线只帮助阅读，不代表新增测量点。

## 三 seed 容忍半径均值

| ε | Task0 EWCLoRA | Task0 FS-EWCLoRA | Task1 EWCLoRA | Task1 FS-EWCLoRA |
|---:|---:|---:|---:|---:|
| 0.25 | 0 | 0 | 0.1667 | 0 |
| 0.5 | 0 | 0 | 0.5 | 0.1667 |
| 1 | 0.5 | 0 | 0.8333 | 0.5 |
| 1.5 | 0.8333 | 0 | 1 | 0.6667 |
| 2 | 1 | 0.5 | 1.3333 | 0.8333 |
| 2.5 | 1.3333 | 0.5 | 1.6667 | 1 |
| 3 | 1.8333 | 1 | 2.5 | 1.5 |

当前选中的绿色配置在这些阈值下没有更大的容忍半径，与前三列的损伤结果一致。

## 数据及复现

- [筛选后的132行逐 seed 数据](selected_per_seed.csv)：两方法×两方向×三 seed×11半径，逐行核验了原始 states.jsonl。
- [均值与 sample SD](selected_mean_sample_sd.csv)。
- [逐 seed 容忍半径与首次失败点](tolerance_per_seed.csv)。
- [容忍半径均值与 sample SD](tolerance_mean_sample_sd.csv)：同时含主图七档与参考六档。
- [公共绘图代码](plot_two_methods.py)：直接调用参考的容忍半径实现；核验输入哈希和此前输出未改变。
- [Task0 单图入口](plot_task0.py)、[Task1 单图入口](plot_task1.py)。

复现两张图：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/gam_radius1_direction_20260922/figures_blue_green_tolerance_20260922/plot_two_methods.py
```

数据仍是当前正式 ImageNet-A、ViT-B/16、rank10 Q/V LoRA 配对协议；CE/accuracy使用1259张旧任务训练图，Fisher使用固定256张旧任务训练子集。Task0方向由1259张旧任务训练图的正梯度得到；Task1方向由997张新任务训练图和配对未训练分类头的负梯度得到，两个图均评估旧 Task0。
