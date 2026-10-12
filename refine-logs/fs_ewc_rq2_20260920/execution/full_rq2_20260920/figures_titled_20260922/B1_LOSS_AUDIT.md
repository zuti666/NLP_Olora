# B1 损失变化核对

结论：B1 的损失曲线是 **逐方向取正部后平均** 的 `G_train`。方向级有符号损失变化确有负值；图中 `G_train` 没有负值。

范围：只统计当前 B1 前两图实际选中的 dense 数据，seed 0、每方向 1,259 个 Task-0 训练样本、50 个共享随机方向。不混入 16 方向的 Fisher 记录。每方法 107 个半径点：原点 1 次，106 个非零半径 × 50 方向 = 5,300 次；两方法合计 10,602 条。半径为 0–20（步长 0.25）及 25–50（步长 1），21–24 未测量。

## 计算顺序与代码

```text
L_train(theta) = (1/1259) * sum_i CE_i(theta)
delta_k(r) = L_train(theta + r*u_k) - L_train(theta)
G_train(r) = (1/50) * sum_k max(delta_k(r), 0)
D_mean(r) = (1/50) * sum_k delta_k(r)  # 有符号均值，当前损失图未使用
```

先对训练样本平均 CE，再减对应 anchor 的平均 CE，再对每个方向取正部，最后平均 50 个方向；原点直接为零。不同 source partition 使用自己的重复原点作为差值基准，并检查原点一致性。

- [rq2_common.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:269)：274 行累加逐样本 CE，276 行除以样本数。
- [diagnostic.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic.py:116)：继承的 `evaluate` 调用该样本平均函数；[diagnostic_full.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/diagnostic_full.py:220) 将其结果写入 `metrics`。
- [diagnostic_full.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/diagnostic_full.py:87)：88 行计算方向级 signed delta，89 行取 `clamp_min(0)`，90 行分别存储 `D_mean` 和 `G`。
- [publish_radius50.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/publish_radius50.py:205)：205–206 行重算复用半径；330–331 行重算整数扩展半径，均为 `sum(max(loss-origin,0))/len(points)`。
- [export_audit.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/source_audit_20260922/export_audit.py:107)：107、117 行保留 signed/positive 两列；146 行仅平均 positive 列得到 `G_train`。

## 实际数据

| 方法 | 非零半径方向测量 | 负 delta 数量 | delta 最小值 | delta 最大值 | 负 signed mean 半径数 | 负 G_train 点数 |
|---|---:|---:|---:|---:|---:|---:|
| EWCLoRA (`sgd`) | 5,300 | 352 (6.6415%) | -0.002283700778880543 | 0.9430690428432863 | 1 | 0 |
| FS-EWCLoRA (`gam`) | 5,300 | 315 (5.9434%) | -0.001479085579790193 | 0.756204930313689 | 0 | 0 |

合计 667 / 10,600 个非零半径方向测量的 delta 为负（6.2925%）。负 delta 表示该方向的**样本平均 CE 低于 anchor**；并非 CE 本身为负。

EWCLoRA 的最小 delta 位于 r=4.75、方向 index=7，最大值位于 r=50、index=46。FS-EWCLoRA 的最小 delta 位于 r=5、index=7，最大值位于 r=50、index=21。完整原始文件与行号见 JSON 的 `minimum_delta` / `maximum_delta`。

EWCLoRA 在 **r=0.25** 的 50 方向中有 25 个负 delta；其 `D_mean = -1.889466497793646e-6`，而实际绘图的 `G_train = 2.8899102122189878e-5`。这是唯一负的按半径 signed mean。FS-EWCLoRA 在同半径有 24 个负 delta，但 `D_mean = 6.446915890527538e-7`，`G_train = 2.259487109027014e-5`。

EWCLoRA 的图中 `G_train` 范围为 [0, 0.7467581379256526]，FS-EWCLoRA 为 [0, 0.6351337248359704]；共 214 个作图损失点无负值。

## 验证

逐条核对 10,602 条选中 dense 记录与 6 个原始 `states.jsonl` 文件，signed delta、positive delta、准确率及来源行号全部一致；`ALL_SOURCE_MEASUREMENTS.csv` 的选择子集与 `SELECTED_PLOT_MEASUREMENTS.csv` 一致。重新聚合结果与 `RECOMPUTED_BY_RADIUS.csv`、`dense_training.csv` 和 `plotted_values.csv` 的 `G_train` 在 1e-12 容差内一致，最大绝对差为 9.9136736146643312e-17。当前 `rq2_common.py` 的 SHA-256 与全部 6 个 dense manifest 一致。

仅使用标准库读取现有记录；未调用 GPU、未进行模型评估、未修改原实验或作图文件。本次新增文件仅为 `B1_LOSS_AUDIT.md` 和 `B1_LOSS_AUDIT.json`。
