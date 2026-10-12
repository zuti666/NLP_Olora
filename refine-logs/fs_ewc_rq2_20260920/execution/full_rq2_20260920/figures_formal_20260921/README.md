# 正式三 seed 的 B1 / B2 新结果图

本目录是新增的结果版本：沿用此前图形样式，以正式三 seed 数据绘制并独立保存。旧结果目录及原图继续保留；本目录的图片、绘图脚本、数据 CSV 和来源记录均独立存放。

这两张图使用 `full_rq2_20260920/formal_results` 中 seed0、seed1、seed2 的正式 **ImageNet-A** 数据。它们不属于历史 ImageNet-R 结果，也不与此前历史 seed0 图或历史半径扩展混合。显示名称 EWCLoRA / FS-EWCLoRA 分别对应配置中的 `sgd` / `gam`，后者沿用当前视觉 GAM+EWC 实现。

- [B1：训练邻域，PDF](Figure_B1_neighborhood.pdf) · [PNG](Figure_B1_neighborhood.png) · [图注](B1_caption.md)
- [B2：逐学习阶段，PDF](Figure_B2_sequence_stage_paths.pdf) · [PNG](Figure_B2_sequence_stage_paths.png) · [图注](B2_caption.md)
- [容忍半径的定义、代码、逐 seed 结果与解释](TOLERANCE_RADIUS.md)
- [完整配置和结果复核](../complete_review_20260921/CONFIG_AND_RESULTS_RECHECK.md)

两图均为 Times New Roman、7.6 × 1.85 英寸、图内图例；PDF 嵌入字体，PNG 为 300 dpi。均值和阴影/误差棒来自三个独立训练 seed，标准差使用 `ddof=1`，不是置信区间；随机方向和积分节点不作为独立训练重复。

## 图的内容和测量口径

| 图 | 内容 | 实测范围 |
|---|---|---|
| B1 左 | 训练平均正损伤 `G` | 全部 1,259 张旧任务训练图；100 个公共方向；0–20、步长 0.25，共 81 个半径 |
| B1 中 | 训练准确率 | 与 B1 左完全相同的评估状态 |
| B1 右 | raw empirical Fisher trace | 固定 256 张训练子集；公共方向库前 16 个方向；0、0.5、1、2、5、10、20 |
| B2 左 | 相对 Task-0 端点的有效权重位移 | 完成 1–10 个任务后的十个真实端点 |
| B2 第二 | 旧任务测试 CE 相对自身 Task-0 锚点的**有符号增量** | 全部 307 张测试图；固定 20 类旧任务分类头；保留负值 |
| B2 第三 | 旧任务测试准确率 | 与 B2 第二相同口径；不是已见全部类别上的准确率 |
| B2 右 | 位移平方归一化后的方向性路径 Fisher，`S_path=I/||Δ||²` | 仅完成 2、6、10 个任务时；每条锚点到端点直线弦段含 9 个积分节点，每节点全部 1,259 张训练图 |

B1 不展示测试损伤、Q95 或第四个容忍半径子图。Fisher 网格没有被补成 81 个点：只有实际节点画标记，节点之间的连线仅帮助阅读。更大训练样本一致性检查保存在 `sparse_train_full`，没有混入不同采样预算的 Fisher 曲线。

B2 的横轴是已学任务数 `T=t+1`，因此 Task-0 是 `T=1`。第二图标注为 “Loss change of Task 0”，数值为有符号损失增量，不是绝对 CE；右图 “Average Fisher” 是方向性路径敏感度，不是 trace。每个右图点是从自己的 Task-0 锚点到该阶段端点的直线积分，不是按训练步追踪出的真实优化轨迹。`T=1` 时位移为零、归一化量未定义，故不画该点；其它未测阶段也没有填值。

## 模型与数据

ViT-B/16，12 层 Q/V 的 rank-10 LoRA，冻结骨干；ImageNet-A 200 类按固定顺序划分为十个 20 类任务，无 replay。诊断参数坐标为 24 个有效 Q/V 权重矩阵，共 14,155,776 维；位移使用整体 Frobenius/欧氏范数。Task-0 及未来任务每任务训练 50 epochs；Task-0 学习率 0.04、后续 0.02，batch size 128。后续训练的 EWC 系数为 2000；图中的诊断 Fisher 是未归一化逐样本真标签梯度外积统计量，不含该系数。`gam` 分支在后续任务也继续使用 GAM，不能将 B2 的差异全部归因于 Task-0 塑形这一单独因素。精确参数以 `../configs/seed{0,1,2}.json` 为准。

## 容忍半径的结果

以平均正 CE 增量不超过 `ε=0.01` 为准，逐 seed 先反求半径，再跨 seed 平均：

| 划分 | EWCLoRA，seed0/1/2 | FS-EWCLoRA，seed0/1/2 | 半径均值变化 |
|---|---|---|---|
| 训练 | 10.75 / 10.5 / 10.5 | 12.25 / 12 / 12.25 | 10.5833 → 12.1667 |
| 测试 | 6.75 / 7 / 7 | 6 / 6.25 / 5.75 | 6.9167 → 6.0000 |

容忍半径是损伤曲线的阈值读数，没有增加独立机制证据。其实现要求此前所有已测半径均通过阈值，首次失败后停止，不做插值。它只描述指定数据、方向样本、半径网格及平均统计量；不能解释成球内每个扰动都安全。训练邻域改善与测试邻域半径缩小可以同时成立；raw Fisher 较低不是有限扰动测试损伤必然较小的充分条件。

## 复现及核验

在本目录分别运行：

```bash
python plot_b1.py
python plot_b2.py
```

这是从已有原始日志重算与绘图，不会启动训练或占用 GPU。旧图保持原数据版本。

- B1 重读 48,588 条 `states.jsonl` 记录，复算训练损伤、准确率与 raw trace，并验证 summary / manifest / coverage 哈希、配对方向和共同端点。
- B2 重读 228 条记录，复核 60 个端点、18 条路径的 162 个节点及五/九节点 Simpson 积分；保留原始有符号损伤。checkpoint 物理文件哈希沿用此前审核，本次不重新运行模型。
- `B1_*per_seed.csv`、`B1_mean_sd.csv`、`B2_*data.csv`、`B2_path_node_evidence.csv`、`B2_mean_sd.csv` 可直接追踪绘图数值。`Figure_*_validation.json` 保存来源哈希、字体与输出校验；`B2_provenance.json` 保存完整 B2 核验记录。
- `B1_include.tex` 和 `B2_include.tex` 提供 LaTeX 引用片段。详细解释图注应随图保留，避免把训练子集 Fisher、测试损伤和方向性路径 Fisher 混为同一统计量。
