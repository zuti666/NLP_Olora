# GAM 状态形成与未来学习易损性：2026-10-01 执行计划

研究假设：GAM 在训练过程中形成的模型状态，可能对后续学习的干扰更不敏感；这不同于 g3 在每一步都比 g1 好。以下设置在新训练结果出现前冻结，直接执行用户附件中的第一轮。

## 固定配置

ImageNet-A 原始 10×20 类划分，class_shuffle=false。首轮只独立训练 Task0，不缩减 DataManager 总类别数。ViT-B/16，LoRA rank10，当前 LoRA 没有 alpha/rank 缩放。单任务 50 epochs，batch128，SGD momentum=0、weight_decay=0。Task0 LR .04→.004，cosine T_max=50；未来 Task1 LR .02→.002，cosine T_max=50 epochs。内半径 rho=.05，外半径 r=1.0，G025 唯一例外 r=.25。beta2=1、gamma_GAM=0、lambda_flat=1。

无 EWC、无 Fisher 估计、无权重融合、无回放、无额外漂移惩罚、无 SGV。保留原生 GAM 四个梯度探针及额外 clean backward，使所有臂的执行路径一致。仅替换选用方向及 SGD 标量。每步验证原生分解、lambda_flat 投影器和实际 SGD 更新。

## 新增训练：8 臂 × seeds 0/1/2 = 24 次

| 配置 | 更新方向 | Task0 LR 起点/下限 | 外半径 |
|---|---|---|---|
| S | g1 | .04/.004 | 1 |
| S1.5 | g1 | .06/.006 | 1 |
| S2 | g1 | .08/.008 | 1 |
| Sum | g1+g3 | .04/.004 | 1 |
| Mean2 | (g1+g3)/2 | .08/.008 | 1 |
| Graw | g3 | .04/.004 | 1 |
| Gmatch | g3；逐步实际有效更新长度匹配独立 S | nominal .04/.004，实际 SGD 标量逐步求解 | 1 |
| G025 | g3 | .04/.004 | .25 |

Gmatch 只读取同 seed 已完成 S 的 500 步 trace；禁止用 G 当前状态的 g1 临时匹配。匹配对象为 concat(LoRA B@A, FC weight, FC bias) 的实际更新范数，基点是原生 GAM 完成 unperturb 后、SGD 前。低秩 Gram 求根后再测实际 FP32 更新，误差≤1e-3才计有效。零目标或无可行解直接失败，不伪造匹配。

每次保存 e=0/10/20/30/40/50 checkpoints；记录 CE、样本准确率、类别平均准确率、LR、逐步样本/增强 SHA、探针与参数更新统计、有效步幅、耗时和显存。Task0 每 epoch 1259 张、10 batches，共 500 步。

## 未来 S 轨迹：78 条

S/Graw/Gmatch/G025 的 6 checkpoints × 3 seeds：72 条；S1.5/S2 的 e50 × 3 seeds：6 条。各 checkpoint 独立 fork，随后统一用 S=g1 更新 Task1 一次连续 100 步。评估 K=0/1/10/50/100，K0 是扩头后的状态；另保存扩头前结果，分离扩头与后续学习的影响。

Task1 实际 997 张、每 epoch 8 batches；100 步为 12 个完整 epochs + 第 13 epoch 的 4 batches。LR 在完整 epoch 结束后推进。所有臂/形成阶段共用同 seed 的未来样本顺序、增强、分类头新增行初始化。

每步只用固定诊断集观察旧/新任务 local 与 all-seen CE/准确率，隔离 RNG、buffers、mode，旧数据不提供训练梯度。记录有符号损失增量 I 与逐步正增量和 I+；完整 I+ 仅针对逐步评估的固定诊断集。全测试集和报告集仅在里程碑评估。

另有 6 条短程未来 Gmatch 轨迹：从 S50/Gmatch50 × 3 seeds 出发，匹配独立 S50→S 的未来 100 步真实更新长度，用于 formation × continuation 对照。

## 固定分层评估与匹配规则

名单在训练前冻结于 selection_split.json，所有 seeds 共用：Task0 full307=diagnostic102+reporting205；Task1 full254=diagnostic84+reporting170。Task1 class34 仅一张，固定进入 reporting；其余类别保留双方。seed2 的完整测试集在历史探索中已查看，应标探索性；新拆分不追溯创造未接触过的 seed2 测试集。

Matched acquisition 固定 S_e50，只用诊断集从 G 的 e10/20/30/40/50 选择。abs CE差≤.01 且正确张数差≤1，按 CE差、正确张数差、距e50、epoch 顺序打破平局；无可匹配点就报告缺失，不事后放宽。

冻结 pilot 推进门槛见 FROZEN_GATES.json：Gmatch−S 的 e50/K100 旧任务 all-seen CE 增量，新增 seeds0/1 均为负，三 seed 均值≤−.01；同时要求新增 seeds 的 acquisition 匹配可行且方向一致，并检查是否只是新任务学得更慢。门槛是研究推进规则，不是统计显著性检验。

## 队列与后续阶段

12 个短测全部通过后才启动 24 个正式训练；Gmatch 还须等待同 seed S 完成。24 次形成训练全部通过后再执行 78+6 条未来轨迹。队列使用所有实际空闲 GPUs 0–6，每卡一个任务，memory<500MiB、util<5%、共享文件锁和 UUID 双重绑定；不抢占已有任务。失败停止新启动，运行中的任务可完成，保留原始失败记录。

Phase2 独立重训 SS/GS/SG/GG；Phase3 独立重训 SSS/GSS/SGS/SSG/GGG，仅在前阶段达到冻结门槛后实现、审核和运行。本轮没有自动加入这两阶段或额外 seeds3/4。旧 seed2 数据只作独立列出的探索背景，不能代替新 S 的完整步幅 trace。

部署目录：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z。suite.json 是完整队列，runtime/queue_state.json 是实际状态，results/ 为经过完成回执校验后的自动汇总。暂停新启动可创建 runtime/PAUSE；删除后继续。冻结源和历史结果不修改。
