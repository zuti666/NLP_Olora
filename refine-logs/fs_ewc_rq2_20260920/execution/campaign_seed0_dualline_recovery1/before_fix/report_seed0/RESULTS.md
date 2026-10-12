# FS-EWC RQ2 seed0 测量结果

更新时间：2026-09-20T10:09:08.655679+00:00

本报告仅将完成的作业计入正式数值；未完成项目记为 pending。所有比较目前均为 seed0 机制案例，随机方向不是训练 seed 重复。

| 方法 | 检查/实验 | 状态 |
|---|---|---|
| sgd | sanity | pending |
| sgd | neighborhood | pending |
| sgd | bridge | pending |
| sgd | trajectory | pending |
| sgd | path | pending |
| sgd | table_a | pending |
| gam | sanity | pending |
| gam | neighborhood | pending |
| gam | bridge | pending |
| gam | trajectory | pending |
| gam | path | pending |
| gam | table_a | pending |

两方法样本ID与图像内容清单哈希一致：False。

新诊断使用确定性 Resize256/bicubic→CenterCrop224→ToTensor，不做 Normalize；训练与测试都使用这一输入处理。记录 Pillow/JPEG 解码器版本与完整 train/test 解码后张量哈希。Fisher 固定同一分层256个旧任务训练样本、24个有效 Q/V 权重矩阵、旧任务20类 head，计算逐样本 CE 梯度，不归一化、不加 floor 或正则。

当前正式测量统一使用 full FP32，并关闭 cuDNN/matmul TF32。四个历史端点可通过 TF32 patch-convolution 操作数舍入复现，这是历史精度差异的证据，不等同于取得旧运行时开关的直接记录。sanity 使用该临时 reference 核验历史端点，随即逐位恢复原始权重；reference 不用于 Fisher、邻域、路径或当前逐任务曲线。

旧半径 train 曲线来自历史缓存随机增强，方向配对未经验证，独立保存为 B_historical_damage_separate.csv 和 Historical_radius_damage_separate 图；不与新 raw trace 拼成逐点关系。旧逐任务 train loss 也保留历史口径，正文序列图使用 test。

## A：固定后续 SGD+EWC 的两任务表

| Task0 起点 | a11 | a21 | a22 | train D0 | test D0 | 位移 | raw trace T0 | raw trace T1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sgd | pending | pending | pending | pending | pending | pending | pending | pending |
| gam | pending | pending | pending | pending | pending | pending | pending | pending |

a11/a21/a22 是所有已见类别上的 CNN class-IL accuracy（%）；a11 使用当前 FP32 起点评估并与续训日志核对，历史矩阵继承的 a11_historical_inherited 另外保留在 CSV/JSON。D0 与机制 accuracy 使用固定旧任务20类 head。两个起点继承的 EWC Fisher 矩阵可不同，不能据此声称完全隔离保护矩阵。

Task1 配对随机过程核验：pending，passed=False。检查新 head、按任务/loader/iteration 分组的样本顺序及首 batch 图像/标签哈希；评估 loader 明确排除。完整细节及任何差异见 A_PAIRED_RNG_CHECK.json。未通过时不宣称严格配对，最终报告状态保持 partial。

## B：等半径邻域与真实方向

邻域正式对照 pending；已存中间状态仅用于恢复和进度，不替代完整半径方向预算。

## C：真实序列与 t9 直线路径

| 方法 | 位移 | D_subset | I | S_path | 节点数 | 求积收敛 | D_subset²≤I（2%数值容差） |
|---|---:|---:|---:|---:|---:|---|---|
| sgd | pending | pending | pending | pending | pending | pending | pending |
| gam | pending | pending | pending | pending | pending | pending | pending |

路径连接 Task0 与 Task9 的有效权重端点，固定 Task0 head，属于端点直线而非逐步训练轨迹。方向敏感度先跨全部矩阵累加每样本 g·Δ，再平方。D_subset 与积分使用同一256个样本；全 train/test 损伤不用于该数值界检查。

Figure C 前两幅曲线当前来源：Historical TF32 reference。正式交付优先使用重新评估的全10任务 FP32 曲线；历史曲线独立保存在 C_historical_task_trajectory.csv。

只有求积收敛与界检查均通过后，Figure C 才显示 S_path 对照。若 raw sensitivity 或 S_path 未下降，保留数值并收窄相应机制结论。

## 图与导出

- [Historical_radius_damage_separate.pdf](Historical_radius_damage_separate.pdf) / [Historical_radius_damage_separate.png](Historical_radius_damage_separate.png)
- [Figure_C_sequence_and_path.pdf](Figure_C_sequence_and_path.pdf) / [Figure_C_sequence_and_path.png](Figure_C_sequence_and_path.png)

数据来源与 SHA256：MANIFEST.json。所有 CSV 为显式分离的新诊断、历史曲线、历史序列或 A 表。
