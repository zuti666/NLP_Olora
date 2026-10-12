# Figure B：半径网格、容忍半径和 Fisher 数值溯源

> 后续更新：用户已要求删除第四图；当前输出为一行三图。本文件对第四图的讨论保留为已归档版本的溯源记录。当前三图来源核查见 B_THREE_PANEL_SOURCES.md。

本次从原始记录核对并重新绘图；没有修改原始测量、插值补造测点或启动 GPU 实验。当前图仍采用历史 ImageNet-A seed0 的确定性补测数据，不应标为 ImageNet-R。

## 1. 有密集半径数据，但当前图读取的是另一次 7 点补测

| 数据组 | 已测半径 | 每半径方向数 | 能提供的量 | 与当前图的关系 |
|---|---|---:|---|---|
| 历史原始 r-sweep | 0–20，步长 0.25，共 81 点 | 100 | loss、accuracy、固定 anchor 的归一化 Fisher 二次型 | 训练图像是一次缓存的随机增强；每个半径重新采样方向，跨方法方向配对未核验；没有扰动位置的 raw Fisher trace |
| **当前 B 图**：historical checkpoint deterministic diagnostic | **0、0.5、1、2、5、10、20，共 7 点** | 非零半径 16；原点 1 | loss、accuracy、扰动位置 raw Fisher trace | 确定性图像变换；方向跨半径、跨方法固定配对；所有四图使用这一组 |
| 后续 formal dense，seed0/1/2 | 0–20，步长 0.25，共 81 点 | 100 | dense 部分是 loss、accuracy | 使用新训练的 Task0 checkpoint，不是当前历史模型；配套 raw Fisher 仍为稀疏测点 |

当前数据路径（相对于 `fs_ewc_rq2_20260920/`）：

- 原始状态：`execution/results_seed0_recovery1/{sgd,gam}/neighborhood/states.jsonl`，每方法 97 条，总计 194 条。
- 配置：`execution/diagnostic_config_recovery1.json`，`radii=[0,0.5,1,2,5,10,20]`。
- 聚合表：`execution/report_seed0/B_neighborhood_deterministic.csv`、`B_tolerance_deterministic.csv`。
- 当前绘图表：本目录 `source_data/B_train_plot_data.csv`、`B_train_tolerance.csv`。
- 历史密集表：`radius_damage_seed0.csv`（2 方法 × 2 split × 81 半径 = 324 行），`radius_reanalysis_manifest.json` 保存原始 JSON 的完整路径和哈希。原始 JSON 位于 `/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial/`。
- 新训练密集记录：`execution/full_rq2_20260920/formal_results/seed{0,1,2}/{sgd,gam}/dense/states.jsonl`，每组 8001 条。不能把这些值贴到当前历史模型的曲线上。

因此图中 15 是线性坐标轴刻度，不是被隐藏的测点；当前补测在 10 和 20 之间没有记录。前段测点密集是原始网格不均匀，不是当前坐标变换。旧密集实验与当前确定性补测的训练 anchor accuracy 分别约为 88.24%/90.31% 和 97.70%/99.05%，也说明不能直接混接两个训练测量口径。若要给当前同一口径的三条曲线补满 1–20，需要在相同历史 checkpoint、图像及方向上追加测量，特别是 raw Fisher。

## 2. 第四图为什么重合

第一图读取 `G`，即方向上的平均正损伤，而不是绝对交叉熵或有符号平均损伤：

$$D_t(\boldsymbol\theta,\boldsymbol\Delta)=L_t(\boldsymbol\theta+\boldsymbol\Delta)-L_t(\boldsymbol\theta),\qquad G(r)=K^{-1}\sum_k[D_0(\theta_0,ru_k)]_+.$$

训练集实测值：

| r | EWCLoRA G | FS-EWCLoRA G |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0.5 | 0.000070064 | 0.000046529 |
| 1 | 0.000162938 | 0.000113298 |
| 2 | 0.000420523 | 0.000294617 |
| 5 | 0.002083703 | 0.001438610 |
| 10 | 0.009386365 | 0.006566436 |
| 20 | 0.049933479 | 0.037108150 |

`execution/diagnostic.py` 的 `neighborhood()` 对半径升序遍历，一旦 `G > epsilon` 就停止，返回之前最后一个已测半径。因此

$$\widehat r_\epsilon=\max\{r_j:\max_{i\le j}G(r_i)\le\epsilon\}.$$

| 预设 epsilon | EWCLoRA | FS-EWCLoRA |
|---:|---:|---:|
| 0.001 | 2 | 2 |
| 0.005 | 5 | 5 |
| 0.01 | 10 | 10 |
| 0.02 | 10 | 10 |
| 0.05 | 20，达到扫描上限 | 20，达到扫描上限 |

epsilon=0.01 时，两方法 r=10 均小于阈值、r=20 均大于阈值。中间没有测点，代码只能给出同样的 10。重合不是实现错误，也不能据此声称两方法在所有阈值下无差别。例如只对现有表重新查询 epsilon=0.008，结果为 5/10；epsilon=0.04 则为 10/20。这两个例子是原始曲线的派生查询，不是新增实验、预设主结果或普遍更大保护区域的证据；图中保留原来的五个阈值。

第四图是第一图的 prefix maximum 曲线的离散广义反函数；当前 G 单调，因此与第一图本身直接对应。把横纵轴交换后表达的是同一损伤—位移关系，但预设阈值、离散网格和扫描截断使它不等于第一图所有点的简单互换。它没有增加独立机制证据；若正文要避免重复，可移至附录。图中连线只是测点间的视觉引导，不代表插值估计的容忍边界。

## 3. 第三图是未乘 lambda 的 raw empirical Fisher trace

$$\overline{T}(r)=\frac1K\sum_k\frac1{256}\sum_i\left\|\nabla_{\theta_{\rm eff}}\operatorname{CE}_i(\theta_0+ru_k)\right\|_2^2.$$

这里对样本、扰动方向取平均，但对有效权重坐标求和。共有 24 个 Q/V 有效权重矩阵、14,155,776 个坐标；没有除以坐标数。raw trace 不限于 [0,1]，也不是平均损失，因此千量级本身不构成异常。

代码证据在 `execution/rq2_common.py`：`PerExampleCollector.add()` 累加 `g.square().sum(dim=(1,2))`；`fisher()` 对 true-label CE 使用 `reduction='none'` 和 `losses.sum().backward()` 保留逐样本未缩放梯度，最后 `raw_trace=trace_sum/len(y)`。`execution/diagnostic.py` 再对各方向的 `raw_trace` 取均值。此计算路径没有 lambda，没有 trace normalization、floor 或 regularizer。训练中的 lambda 可能影响学到的权重，但没有作为系数乘到本图的测量值上。

| r | EWCLoRA raw trace | FS-EWCLoRA raw trace |
|---:|---:|---:|
| 0 | 1465.496927 | 761.900774 |
| 10 | 1760.348790 | 823.038780 |
| 20 | 2837.329980 | 1452.409609 |

该量也不同于旧 r-sweep 中的固定 anchor、trace-normalized Fisher 二次型，不能根据旧结果中的归一化系数解释当前 raw trace 的数值。

## 4. 本次绘图修改

第一图标记 `Loss` 及带平均线/正部的 D_t 符号，避免将原来的平均正损伤错误重标为单个有符号 D_t；本实验 t=0。第二图为 `Accuracy`，单位仍为百分比。第三图为 `Raw Fisher trace`。前三图横轴均为 `Weight drift`，即 24 个有效权重矩阵的联合 Frobenius 位移；第四图继续用 `Loss threshold` 与 `Tolerance radius`。保持一行四列、Times New Roman、训练均值曲线、图内图例和真实数值比例。
