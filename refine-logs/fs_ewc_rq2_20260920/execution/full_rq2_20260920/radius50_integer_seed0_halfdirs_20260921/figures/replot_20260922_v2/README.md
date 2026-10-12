# B1 v2：六档阈值与紧凑标签

[PNG](Figure_B1_neighborhood_seed0_radius50_20260922_v2.png) · [PDF](Figure_B1_neighborhood_seed0_radius50_20260922_v2.pdf) · [绘图函数](plot_b1.py)

第四图从本版本子图1所使用的同一 dense dataframe 重新计算，并非沿用旧阈值表。包含全部107个已测半径；每个非零半径均使用 seed0 的同一50个公共方向。对于按r排序的损伤曲线G，先取前缀最大值，再取前缀值不超过 epsilon 的最大已测半径。不插值，不增加GPU测量。

| epsilon | EWCLoRA | FS-EWCLoRA |
|---:|---:|---:|
| 0.005 | 8.25 | 9.25 |
| 0.01 | 11.00 | 12.25 |
| 0.02 | 14.50 | 16.25 |
| 0.03 | 17.00 | 19.25 |
| 0.04 | 19.00 | 20.00 |
| 0.05 | 20.00 | 20.00 |

epsilon=0.05 可以从现有曲线计算：EWCLoRA 在r20的G为0.04417169，在r25为0.07853632；FS-EWCLoRA分别为0.03306670和0.05967065。两方法均在20通过、25失败，所以当前网格的读数都是20。这不等于精确连续阈值相同。

epsilon=0.03 的读数为17和19.25；相应首个失败半径为17.25和19.5。详细通过/失败损伤见 tolerance_bounds.csv。

epsilon是平均每个样本交叉熵增量的容许上界，不是1259张样本损失总和，也不是相对百分比。每个方向先计算全部训练样本平均CE相对本方法锚点的变化，取正部，再对50个方向求平均。

配置：seed0，当前正式ImageNet-A cohort，ViT-B/16 rank-10 Q/V LoRA，固定Task-0锚点和20类旧任务输出头。损伤/准确率使用1259张训练样本与50个公共方向；Fisher使用256张固定训练样本与16个公共方向，为未乘lambda、未额外归一化的true-label per-example empirical Fisher trace。单seed不展示跨seed误差带。

前两图半径为0–20步长0.25，以及25–50步长1；Fisher为0、0.5、1、2、5、10–20整数和25–50整数。21–24未测；连接线只作视觉引导。r17的EWCLoRA Fisher尖峰按实测保留。

版式修改：Loss change中的D_t改为D_0；准确率轴改为accuracy of task 0 (%)；坐标标签8.2pt，前三图刻度7.2pt，legend7.2pt，marker2.7pt。横轴标题labelpad=1.5pt，取消强制按第四图长刻度对齐，使前三图横轴标题上移。Times New Roman、一行四图、颜色线型及图内legend保留。

旧replot_20260922及自动汇总图保持不变。本目录plot_b1.py可在Pilot_new环境中复现PDF、PNG与CSV。
