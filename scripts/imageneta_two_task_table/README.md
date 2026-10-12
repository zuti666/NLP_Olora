# ImageNet-A 前两个任务：六行 LoRA 机制表

用户 2026-09-20 最新配置覆盖此前 ImageNet-R、无轨迹平均的方案。本目录只执行 ImageNet-A 固定 10×20 类划分中的前两个任务。使用当前 CopyFlatCV 连续 Q/V LoRA，不沿用旧表的 fresh K/V adapter、Adam、λ=1e7 或旧数值。

## 六行与共享模型

| variant | Task0 | Task1 | Task1 λ |
|---|---|---|---|
| vanilla | SGD | SGD | 0 |
| shaping_only | GAM | GAM | 0 |
| ewc | SGD | SGD+EWC | 2000 |
| gam_ewc | GAM | GAM+EWC | 2000 |
| task0_shaping | GAM | SGD+EWC | 2000 |
| task1_shaping | SGD | GAM+EWC | 2000 |

每 seed 只训练两个 Task0 anchor，然后派生六条 Task1。正式 seeds 0/1/2，共 6 个 anchor、18 个 Task1 训练段。所有分支均重新训练；仅复用历史模型中冻结的预训练 backbone，新 A/B、分类头重新初始化。两种 anchor 的初始完整网络已验证一致。

`config.json`：ViT-B/16、rank10、batch128；每任务 50 epochs，lr .04→.02，cosine，momentum/weight decay 均为 0。GAM 的 base optimizer 为 SGD，rho=.05、norm_rho=.2、gamma=.1、三个 beta=1、lambda_flat=1；无 EWC 投影。EWC gamma=.9、单任务及累计 trace 归一化、累计归一化后 damping alpha=.2。

原生 GAM 使用 4 次 CE 网络反传，再加 clean-point EWC 梯度；零半径极限仍有原生 `2*gCE` 尺度，不宣称相同学习率已控制相同有效步长。

## 轨迹与 Fisher

当前按**六行统一开启轨迹混合**执行；这是比较规则选择，并非旧表的 endpoint-only 复现。轨迹政策写在 `trajectory_merge_policy=all`，不能将这批数据标注为“无轨迹平均”。

1. 完成 epoch 20/40 后保存 Q/V A/B 因子、当时分类头和训练 CE。固定当前训练集、确定性预处理、eval mode；无测试集参与加权。
2. 两个训练 CE 做总体标准差标准化，再取 `softmax(-3*z)`；相同 CE 时各 .5。两个不同 CE 时较低 CE 权重约 .997527，较高者约 .002473。
3. **A、B 分别**按 `.7*final + .3*weighted_snapshot_mean` 混合；分类头和冻结 backbone 不混合。混合后的 BA 含交叉项，不是 BA 的线性平均。
4. 在混合后的模型计算 full-current-train 的 batch-gradient Fisher，batch128、真实标签 local20 CE。先单任务归一化，再 gamma 累计归一化，最后与 `I/d` 按 .2 混合。没有混合路径上的 Fisher。

质量评分与 Fisher 都显式使用 deterministic train-source 数据；这与历史默认直接迭代随机增强 train loader 有区别，记录在新协议中。训练仍使用原训练增强。默认预处理是 Resize256/CenterCrop224/ToTensor，没有额外 Normalize。

`trajectory/task_XX/epoch_20.pt`、`epoch_40.pt` 保存因子、分类头和冻结 base 来源，可配合 base 恢复快照；`pre_merge_endpoint.pt` 留存未混合端点；`merge.json` 保存 CE、权重、混合前后因子记录；`current_fisher.json` 保存混合后、归一化前的 raw trace。最终两个 `checkpoints/task_00.pt/task_01.pt` 保存完整网络。

## 表格口径

- `a11/a21/a22/BWT`：以 dataset 标签评价，a21/a22 使用全部已见 40 类分类头。
- `B1/B2`：各任务自己训练集、该任务混合后端点的未归一化 batch-gradient proxy trace，逻辑 batch128；它们不是逐样本 empirical Fisher，也不是 Task0 敏感度的两端重测。
- 另报 old-at-Task0、old-at-Task1、new-at-Task1 的**逐样本 raw empirical Fisher**，每任务固定分层 256 训练样本，排除分类头；不参与训练。
- 旧任务损伤：完整 train/test 的固定旧 local20 头 CE 相对 anchor 的增量，保留负值。
- 位移：`sqrt(sum_l ||BA_task1−BA_task0||_F²)`，有效 Q/V 坐标，排除分类头；不使用因子移动或新增头出生能量。

训练 raw B 与离线重算 B 必须通过一致性检查；EF 的 per-sample hook 还与小样本 autograd 对照。六行的 checkpoint、初始化、数据顺序及评估输入均进行配对检查。

## 执行与产物

环境：`/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`，已有 Pilot_new，FP32、TF32 关闭，无包安装。

当前作业计划：`refine-logs/imageneta_two_task_table_20260920/campaign/plan.json`。

只读检查计划（不会启动 GPU）：

```bash
python scripts/imageneta_two_task_table/pipeline.py \
  --plan refine-logs/imageneta_two_task_table_20260920/campaign/plan.json
```

调度器增加 `--execute` 才启动。仅使用已授权的物理 GPU 4、6：检测无计算进程且显存<500MiB/利用率<5%才启动；每卡一个作业，两卡可运行任一优化器。任何作业/receipt 失败即停止后续依赖，不抢占、不静默降低 batch/epochs。检查阶段失败不会放行正式任务。

先在 `smoke/runs` 执行 seed0 六行、每任务3epochs、快照1/2、EF8样本的真实 GPU 检查；它不生成论文表。通过所有六行的恢复、混合后 F、EWC 开关、CE 反传次数、实际阶段学习率及配对检查后，才进入正式50epoch任务。

正式结果根目录 `refine-logs/imageneta_two_task_table_20260920/runs/`：

- `anchors/{sgd,gam}/seedN/`：共享第一任务模型与配置。
- `{variant}/seedN/`：第二任务模型、日志、轨迹快照、真实配置；`evaluation/summary.json` 为该行原始指标。
- `comparison/comparison.tex`：六行均值±样本标准差表，数据集/优化器/轨迹说明更新。
- `comparison/legacy_layout.tex`：同布局均值版；`enhanced.tex`：损伤与逐样本 raw EF 表。
- `comparison/per_seed.csv`、`mean_sd.csv`、`summary.json`：每 seed 原值和汇总；`through_seedN/` 归档各阶段汇总。

执行状态在 `campaign/pipeline_state.json` 和 `campaign/STATUS.md`，详细 stdout/stderr 在 `campaign/*.attempt1.log`。源配置已冻结的运行不会自动接受代码变更；如需修正失败，保留旧尝试并使用新 campaign/run 根目录。正式任务未完成前，不用旧表数字填空或把检查结果当成新证据。
