# GAM 两组训练半径与固定方向诊断

采用用户最终更正的两组训练参数，每组 seed 0/1/2，仅使用物理 GPU 0、1：

| 新实验 | gam_grad_rho / as1_rho | gam_grad_norm_rho / as1_norm_rho |
| --- | ---: | ---: |
| probe005_norm1 | 0.05 | 1.0 |
| probe1_norm1 | 1.0 | 1.0 |

旧 FS-EWCLoRA 对照为 `(0.05, 0.2)`。此前准备的 `(1.0, 0.2)` 已取消，没有启动该配置的 GPU 训练。

## 协议与解释范围

延续当前正式 RQ2 的 ImageNet-A、ViT-B/16、Q/V LoRA rank 10、固定类别顺序、每任务 20 类；Task0 为 1259 张训练图、307 张测试图。重新训练 Task0 50 轮，batch 128，初始学习率 0.04、cosine 调度，保持其余正式协议不变。实际优化器半径在运行时另行断言并记录。

用户给出的 `CopyFlatCV/refine-logs/s3-expanded-lambda-r-run-20260921T153236Z/configs/s3_expand_r1p0_s2.json` 确为 `(0.05, 1.0)`，但该完整配置还改变模型分支、in-task Fisher 与 weight merge 等设置。本实验按此前说明的受控比较，只改变这两个半径及其有效别名；不能将本次 Task0 诊断称为该十任务 FAA=56.881% 的复现。具体差异见 `CONFIG_DIFFERENCES.json`。

训练半径作用于当前 GAM 训练参数空间；下述横轴 r 是有效 Q/V 权重的全局欧氏位移，二者不能直接当作同一个数值尺度。

## 固定方向和输出

- Task0：在各自新 Task0 模型处，用全部 1259 张旧任务训练图计算一次正交叉熵梯度，即损失上升方向。
- Task1：在各自新 Task0 模型处，按配对 RNG 构建相同初始化的新任务分类头，用 997 张 Task1 训练图计算一次负交叉熵梯度，即新任务下降方向。
- 各方向在有效 Q/V 权重坐标中全局归一化到单位长度，随后固定不变。每个测量点均从锚点构造 `theta(r)=theta0+r*u`，测量 `r=0,0.5,...,5`，不是逐点继续优化。
- 两种方向都报告旧 Task0 的损失变化、准确率和 Fisher；测量时使用各自模型固定的旧 Task0 分类头。
- Fisher 使用固定的 256 张旧任务训练样本，计算 true-label、per-example 梯度平方范数的均值；不乘 EWC lambda，不做额外 trace 归一化。
- 损失变化是有符号 CE 增量；不截断负值，也不要求大半径时单调。Task1 的下降方向只保证局部降低新任务损失。

新测量为 2 组 × 3 seeds × 2 方向 × 11 半径 = 132 个位置。绘图再复用旧 EWCLoRA 和旧 FS-EWCLoRA 同网格的 132 行，总共 264 行；复用行不属于新增实验。

两张图均使用 Times New Roman、一行三列、图内 legend、线性横轴。四条曲线为 EWCLoRA、旧 FS `(0.05,0.2)`、新 FS `(0.05,1)`、新 FS `(1,1)`，均为三 seed 均值及样本标准差。报告同时给出原点拟合和绝对 CE，以区分模型拟合变化与扰动增量变化。

## 状态与入口

- 训练状态：`training_campaign/STATUS.md`、`pipeline_state.json`。
- 接续队列：`diagnostic_campaign/queue_state.json`；全部六个训练完成且收据有效后才进入 GPU 测量。
- 诊断先完成全部 Task0 方向，再完成全部 Task1 方向，最后自动绘图，避免同一模型的共享缓存并发写入。
- 操作记录：`RUN_TRAINING.md`、`RUN_DIAGNOSTICS.md`。
- 完成判据：`report/radius_intervention_complete.json` 必须实际存在且验证通过；缺失时不得视为实验完成。
- 最终报告：`report/RESULTS.md`。
- 最终新图：`figures/task0_positive/Figure_task0_positive_training_radius_intervention.pdf` 和 `figures/task1_negative/Figure_task1_negative_training_radius_intervention.pdf`，各有 PNG。

新结果保存在本目录，原有实验数据及图片均保留。
