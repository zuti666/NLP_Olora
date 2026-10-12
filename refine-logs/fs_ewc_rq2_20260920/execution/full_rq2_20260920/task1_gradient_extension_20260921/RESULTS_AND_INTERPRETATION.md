# Task1 固定下降方向实验结果与解释

6/6 GPU作业通过校验，558/558个测量状态完整；CPU汇总与两张图生成成功。正式完成时间约为2026-09-21 22:28 UTC。`FINAL_VERIFICATION.json`保存全量收据、代码/输出哈希和真实方向见证校验；图片经过主线程与独立审查者实际查看。

## 新图

- [r=0–50 PDF](figures/full/Figure_B1_fixed_task1_gradient_full.pdf) · [PNG](figures/full/Figure_B1_fixed_task1_gradient_full.png)
- [r=0–2 PDF](figures/near_origin/Figure_B1_fixed_task1_gradient_near_origin.pdf) · [PNG](figures/near_origin/Figure_B1_fixed_task1_gradient_near_origin.png)
- [审定图注](figures/REVIEWED_CAPTION.md)，[LaTeX插入代码](figures/reviewed_includes.tex)。自动发布的旧图注与原始输出均保留。

样式沿用参考图：Times New Roman，一行三列，EWCLoRA蓝色实线圆点，FS-EWCLoRA橙色虚线菱形，图内图例，3个训练seed均值±样本标准差。三个子图均测量旧Task0：有符号训练CE增量、训练准确率、raw empirical Fisher。数据峰值未裁剪、平滑或归一化。标准差带低于零不表示实际Fisher为负。

## 实验是什么

沿用当前正式ImageNet-A、ViT-B/16、rank-10 Q/V LoRA和seed0/1/2。每个方法/seed在自身Task0锚点上，用全部997张Task1训练图计算一次负梯度并归一化，固定该方向，按theta(r)=theta0+r*u测量93个半径。Task1未训练新20类分类头从正式初始化记录精确复原，同seed的两方法新头相同。

旧任务训练损失/准确率使用原20类头和1259张训练图；Fisher仍用固定256张旧任务训练图，按逐样本梯度平方范数平均，不乘lambda、不做额外归一化。全部6组的真实小步Task1 CE均下降，方向没有反号。没有用Task1训练后参数构造方向。

## 新结果：三seed均值

### 旧任务训练损失增量

| r | EWCLoRA | FS-EWCLoRA | FS−EWC 配对均值 |
|---:|---:|---:|---:|
| 0.1 | 0.021303 | 0.018788 | −0.002515 |
| 0.25 | 0.082585 | 0.081447 | −0.001138 |
| 0.5 | 0.276405 | 0.357613 | +0.081208 |
| 1 | 0.992690 | 1.464472 | +0.471782 |
| 2 | 2.471694 | 2.697590 | +0.225896 |
| 10 | 3.855768 | 4.802255 | +0.946487 |
| 20 | 4.834852 | 5.867995 | +1.033143 |
| 50 | 6.962059 | 7.668132 | +0.706073 |

92个已测非零半径中，FS在7个小半径的平均损伤更低（0.001、0.005、0.01、0.025、0.05、0.1、0.25），在其余85个、即所有已测r≥0.5处更高。这只是固定网格的描述，不是独立抽样胜率或显著性检验。

r=1的配对损伤变化在seed0/1/2分别为+0.573180/+0.662879/+0.179287，均值+0.471782、样本标准差0.257248；三个seed都出现反转。r=0.1三个seed均为FS损伤更低。r=20的三seed变化为+1.535989/−0.141142/+1.704581，说明单seed仍有差异。

### 准确率与Fisher

| 指标 | r | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|---:|
| 旧任务训练准确率（%） | 1 | 70.0291 | 60.4448 |
| 旧任务训练准确率（%） | 2 | 25.2846 | 16.4946 |
| raw Fisher trace | 1 | 7135.06 | 8242.15 |
| raw Fisher trace | 2 | 7866.01 | 11778.59 |
| raw Fisher trace | 20 | 45777.87 | 64368.84 |
| raw Fisher trace | 50 | 94843.01 | 4351.78 |

Fisher的比较次序也依赖位置。r=50时FS的Fisher更低，但此时两方法训练准确率仅约5.27%和3.94%，不能将当地Fisher低解释成旧任务性能得到了保护。Fisher trace是当前位置的逐样本梯度能量，而损失变化是相对原点的累计差值，两者不要求同步或单调。

## 为什么建议Task1负梯度

继续学习首先要降低新任务损失。与主动提高旧任务损失的Task0正梯度相比，Task1负梯度提供了一个与新任务目标有关的初始方向。局部旧任务变化由−r*(g0·g1)/||g1||控制；方向冲突时可能产生遗忘，对齐时可能帮助旧任务，不能预设损伤一定增加。

这次六组实际g0·u1都为正。EWCLoRA/FS-EWCLoRA的平均初始旧任务方向导数为0.140134/0.128276，旧任务梯度与新任务下降方向的平均余弦为0.137001/0.172039。新任务方向在原点确实与旧任务下降目标冲突；FS的初始旧任务损伤斜率仍稍小，但有限半径结果很快交叉。逐seed数据见`analysis/task1_old_gradient_alignment.csv`。

## 与随机、Task0正梯度结果怎样一起解释

随机扫描是在公共随机方向上求平均，固定梯度扫描则选择了有任务结构的方向，而且两种方法使用各自的方向。高维随机方向通常与旧任务梯度接近正交；固定Task0正梯度最大化原点的一阶旧任务损失增量。因此同样r的损伤量级和方法排序都可能不同。

此前Task0正梯度实验中，FS初始斜率更小，但r=0.5时其超出原点一阶预测的非线性余项更大，抵消了初始优势。该分解及完整随机对照见[分析说明](analysis/WHY_RANDOM_AND_TASK0_DIFFER.md)。余项是已有损失曲线的精确分解，不是额外独立证据或某点Hessian测量。

当前证据应表述为：**训练侧典型随机扰动响应改善，但这种改善没有普遍延伸到所测的任务梯度方向。** 新Task1实验没有恢复“更大的普遍未来保护区域”结论。原点raw Fisher较低也不保证离开原点后的整个路径Fisher更低。

本实验仍是确定性全样本CE在有效Q/V空间的固定初始方向，区别于实际LoRA因子优化器、随机增强、小批次和GAM/EWC每步修正。它不能代替Figure C的真实后续学习轨迹；C的实际结果应单独按其配置报告。

## 可复现数据

- `figures/all_per_seed.csv`：全部558个原始逐seed绘图值，含训练/测试CE、损伤、准确率、raw Fisher。
- `analysis/three_directions_paired.csv`：随机、Task0正梯度、Task1负梯度的逐半径三seed配对统计。
- `analysis/three_directions_selected.csv`：预选半径的比较表。
- `analysis/compare_random_task0_gradient.py`、`analysis/summarize_three_directions.py`：重算脚本。
- `publish_task1_gradient.py`：读取六组完整原始结果、独立验证方向/头/数据和统计并绘图的CPU入口。
