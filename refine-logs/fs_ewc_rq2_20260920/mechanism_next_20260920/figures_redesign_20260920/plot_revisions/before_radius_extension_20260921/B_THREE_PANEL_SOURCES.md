# 当前 Figure B：三图来源一致性核查

已删除第四个容忍半径子图。当前为 Times New Roman、一行三列，依次显示平均正损伤、训练准确率、raw Fisher trace；横轴均为 Weight drift。

## 结论

三个指标来自每个方法的同一批扰动模型状态、同一个 Task0 训练集及相同确定性预处理。样本范围存在明确区别：

| 图 | 原始字段与聚合 | 样本 |
|---|---|---|
| Loss | 先对每方向计算 `max(metrics.train.loss - anchor.metrics.train.loss, 0)`，再取方向平均 | 全部 1259 个 Task0 训练样本 |
| Accuracy | `metrics.train.accuracy` 的方向平均，单位 % | 同样 1259 个样本 |
| Raw Fisher trace | `fisher.raw_trace` 的方向平均；raw trace 是逐样本 CE 梯度平方范数均值，未乘 lambda | 从上述样本固定选出的 256 个样本 |

不是把不同实验的 Loss/Accuracy 与另一批模型的 Fisher 拼接。两个方法各有自己的 Task0 checkpoint 和固定 20 类头；“模型与头一致”指同一方法的三个指标共享，不指两个方法权重相同。

共同配置：历史 ImageNet-A，seed0；有效权重坐标为 24 个 Q/V 矩阵，共 14,155,776 维；半径为 `[0,0.5,1,2,5,10,20]`；每个非零半径 16 个共享方向，跨半径、跨方法方向配对；原点复用一个 anchor。每方法 97 个状态，共 194 个。确定性预处理为 Resize(256,bicubic)→CenterCrop(224)→ToTensor，无输入归一化。当前图不含旧的随机增强 81 点数据，也不含新训练 formal cohort。

## 原始路径及代码证据

以下路径相对于 `fs_ewc_rq2_20260920/`：

- `execution/results_seed0_recovery1/{sgd,gam}/neighborhood/states.jsonl`：三个指标共同来自其中的状态行。
- `execution/results_seed0_recovery1/{sgd,gam}/neighborhood/manifest.json`：每方法 checkpoint、坐标、分类头和协议哈希。
- `execution/results_seed0_recovery1/{sgd,gam}/data_manifest.json`：训练图像、确定性变换、Fisher 子集的完整样本 ID 与索引。
- `execution/diagnostic.py:119–133`：`measure()` 只设置一次扰动点，然后依次 evaluate train/test 与 fisher，期间无参数更新，最后共同写入一个状态记录。
- `execution/rq2_common.py:231–247`：按类别比例选取固定子集，最终执行 `tensors['subset'] = tensors['train'][selected]`，标签同样索引。因此 Fisher 使用相同解码与变换后的训练张量子集。
- 本目录 `gen_fig_B_neighborhood_merged.py` 重新读取状态并校验报表，导出 `source_data/B_train_plot_data.csv`。Loss 从已核对的报告 G 列读取；Accuracy 与 Fisher 从相同原始状态重算。

## 独立核查结果

独立 CPU 核对 14 行 × 3 指标共 42 个绘图值，全部匹配原始记录。194 条状态的 checkpoint/data/coordinate/protocol 元数据全部匹配；两方法 96 个方向×非零半径配对全部一致。Fisher 子集的 256 个样本 ID 均逐项匹配训练清单的对应索引，索引无重复。数据清单 canonical hash 已重新计算；当前 diagnostic.py 和 rq2_common.py 文件哈希与生产 manifest 一致。

两方法共享的哈希：

```text
data manifest:
be9cf89b2bfbd5a9bc207d9755deb2d3a9bcbd6190a24b3a9c05ead97bfddecb
train decoded tensor:
1a4cdbc7b2247f779eb67387ff1e86f65eb3b4723f482707b1e6281e5bc377ec
Fisher subset tensor:
ba0cac3b272c56185152f0383967652ef36ecd12d5ddb93cac1b4655b9ac617f
```

生产记录中的 Task0 checkpoint 哈希：

```text
EWCLoRA:    b352fceeede62745a3e32db264d661e67319e35cb2b65364c64748181685c059
FS-EWCLoRA: 268cfd8575a779078a327be281ef616eac3d77f618831c284e6d4aafa769701a
```

本次重绘与核查未启动 GPU、未重新执行模型评估，也未重新哈希 checkpoint 权重文件；核对的是原始记录及生产清单。独立图形复核检查了最终 PNG 和图注，结果 PASS，无裁切或图例遮挡。移除前四图版本保留在 `plot_revisions/before_three_panels_20260921T171611/`。
