# 本次两任务续训与历史 ImageNet-A 主链的直接核对

本次只读原始 JSON、真实训练 shell log、配置、冻结 learner 与续训 runner；没有重新训练或运行 GPU 评估。提取值、全部原始路径、真实旧配置和新 manifest 见 [HISTORICAL_COMPARISON.json](HISTORICAL_COMPARISON.json)。

结论：新实验比较“SGD Task0 → SGD Task1”与“GAM Task0 → SGD Task1”；旧主链比较“SGD Task0 → SGD 后续全部任务”与“GAM Task0 → GAM 后续全部任务”。因此新结果是对“只改变起点、后续统一 SGD”这一更强假设的独立检查，并非旧完整 GAM 方法的复现。新 seed0 没有显示平均准确率收益。单凭新旧差值不能证明关闭 GAM 是下降的唯一原因，因为随机协议、精度和诊断插入顺序也有区别。

## 原始 CNN 指标

以下旧任务 / 新任务 accuracy 均是 Task1 结束、在全部已见 40 类中分类的 CNN 指标。平均是两个任务 accuracy 的等权平均，不是样本加权 accuracy，也不是两次学习阶段的平均。

| 来源与路径 | Task0 后 a11 | Task1 后旧任务 a21 | Task1 后新任务 a22 | Task1 两任务均值 |
|---|---:|---:|---:|---:|
| 旧 SGD → SGD | 85.34 | 82.08 | 75.20 | 78.640 |
| 旧 GAM → GAM | 87.62 | 81.76 | 81.50 | 81.630 |
| 新 SGD → SGD | 85.667752（本次重测） | 84.36 | 74.80 | 79.580 |
| 新 GAM → SGD | 87.622150（本次重测） | 84.04 | 72.05 | 78.045 |

旧两行来自 `stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/trainer_outputs/**/s1c_{sgd,gam}_from_task0_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json` 第 988 行的 `cnn.matrices.t01`。新两行来自 `execution/training/A_{sgd,gam}_to_sgd_seed0/trainer_outputs/**/A_{sgd,gam}_to_sgd_seed0_vit_base_patch16_224_cl_metrics.json` 第 4 行的同名字段。

观察：旧 GAM 对 SGD 的两任务均值优势为 +2.99 pp，其中旧任务反而低 0.32 pp、新任务高 6.30 pp；新 GAM 起点对 SGD 起点均值差为 −1.535 pp，其中旧任务低 0.32 pp、新任务低 2.75 pp。故这次差距主要来自新任务学习，不应只描述成“旧知识遗忘导致下降”。旧任务全头准确率的相对方向与旧 T1 一致；新任务准确率的相对方向发生变化。

新 `cl_metrics.json` 的 a11 是继承历史前缀，仍显示 SGD 85.34，而同环境 FP32 重测为 85.667752（307 个样本中多对 1 个）。因此不能用继承值计算本次精确 forgetting。新同环境 seen-head forgetting 是 SGD 1.302932 pp、GAM 3.583062 pp。基线和端点来自两条 `task0_local_metrics.jsonl` 的第 1、2 行。

## 旧任务局部头与全部已见类别必须分开

| 本次旧任务测试指标 | SGD 起点：Task0 → Task1 | GAM 起点：Task0 → Task1 |
|---|---:|---:|
| local-head accuracy（仅旧 20 类） | 85.667752 → 86.644951 | 87.622150 → 86.319218 |
| local-head CE | 0.442423705 → 0.423177732 | 0.413620681 → 0.400232589 |
| seen-head accuracy（20 类 → 40 类） | 85.667752 → 84.364821 | 87.622150 → 84.039088 |
| seen-head CE | 0.442423705 → 0.598961180 | 0.413620681 → 0.533356956 |

两条的 local CE 都下降。GAM 起点续训后的旧任务 local CE 和 seen CE 仍较低，然而 top-1 accuracy 略低；CE 与 accuracy 的排序不同并不矛盾。新增类别带来的竞争使 seen-head accuracy 相对 local-head 更低，本次两者均相差 2.280130 pp（7/307 个样本）。这组结果不支持把全头 accuracy 下降全部等同于旧任务特征遭破坏。

旧 T1 local accuracy 是 SGD 85.016287、GAM 86.319218；对应 local CE 0.435415895、0.398130700。旧 T1 的 GAM local accuracy 较高，但 seen accuracy 较低，本来就说明两种评价口径不能替代。

## 历史十任务结论仍属另一条链

| 历史 t9 指标 | SGD 全链 | GAM 全链 |
|---|---:|---:|
| 最终十任务等权均值 | 40.994 | 52.078 |
| Task0 seen-head accuracy（200 类） | 64.169381 | 72.312704 |
| Task0 local-head accuracy（20 类） | 80.130293 | 85.342020 |
| Task0 local CE 相对本链起点增加 | 0.260342520 | 0.034413266 |

对应原始 JSON 的 `cnn.final.FAA`（第 1626 行）、`cnn.matrices.t09`（第 1376 行）及 `task0_trajectory.steps.9`。这些支持的是旧 seed0 的全程 GAM 链表现更好；不能直接证明“仅 Task0 shaping，后续统一 SGD”也更好。本次 t9 路径分析仍读取旧 GAM → GAM 链，所以它与新 A 的 GAM → SGD Task1 不能互相替代。

## 配置与代码执行的差异

| 项目 | 历史主链 | 本次 A 两任务续训 |
|---|---|---|
| 数据、顺序、架构 | ImageNet-A，固定顺序，10×20 类，ViT-B/16，LoRA rank10 | 相同；只恢复 Task0 并训练 Task1 |
| seed 标识 | 0 | 0，但新的显式 task RNG 协议 |
| Task1 训练 | 50 epoch，batch128，LR 0.02，cosine，momentum0，WD0 | 相同 |
| EWC | λ2000、gamma0.9、eta0、任务 Fisher 与累计 Fisher 归一化、最多100 batch | 相同；分别恢复各自 Task0 Fisher |
| 普通分支 Task1 | `optimizer_type=sgd` | `optimizer_type=sgd` |
| GAM 起点分支 Task1 | `optimizer_type=gam`、`lambda_flat=1` | `optimizer_type=sgd`、`lambda_flat=0` |
| GAM 参数 | rho0.05、normrho0.2、gamma0.1、beta1/2/3=1 | 配置保留这些字段，但 A 的 SGD 更新路径不用它们 |
| 随机数 | `set_random(seed)` 初始化；无 Task1 随机收据 | Task1 reset 到 20261929，新 head、采样顺序及增强输入配对核验 |
| 精度 | 缺乏整段训练的显式 TF32 witness；受控卷积精度实验可重现历史端点 | FP32，matmul TF32=false、cuDNN TF32=false |
| 额外诊断 | Task0 trajectory/Fisher 重算、full Fisher；存在额外 loader 遍历 | 关闭这些训练内诊断；新增评估保存/恢复 RNG |
| in-task Fisher、merge、projection、dual ascent | 关闭 | 关闭 |

直接证据：

- 旧实际 `.../logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/{sgd,gam}_T10_seed0.log` 第 6–54 行记录数据、rank、λ、优化器和训练参数，第 74–80 行记录真实 Task0 恢复路径与额外 full Fisher 设置。第 114–116 行记录 Fisher 成功恢复、从 Task1 继续。
- [旧启动脚本](/data/140-1/users/liying/CopyFlatCV/scripts/run_s1c_stage2_T_fullseq_from_task0.sh:210) 在 GAM 分支明确传入 `optimizer_type=gam`，SGD 分支则传 `sgd`；第 233 行从各自 Task0 恢复。
- [新 runner](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py:92) 在 continuation 模式无条件使用 SGD；第 100–125 行设置 SGD / lambda_flat0 和其余训练参数。
- [新完整恢复](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py:240) 先保留原 loader 的生命周期和 Fisher 恢复，再严格载入完整 `network_state`，350 项逐位核验；第 319–344 行实现独立 loader generator 与 task RNG reset；第 472 行显式关闭 TF32。
- [原 learner 的 SGD 分支](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py:1180) 执行 task loss + Fisher loss；旧 SGD 配置虽然 `lambda_flat=1`，这条分支并不读取该项，因此不能把该字段的 1→0 当作 SGD 结果变化原因。GAM 分支的梯度组合在第 1058 行，才使用 `lambda_flat`。
- [历史 RNG 函数](/data/140-1/users/liying/CopyFlatCV/utils/random_reproduce.py:24) 仅设置 Python、NumPy、torch seed 及 cuDNN deterministic/benchmark；没有显式 TF32 开关。新旧代码存在相同名 seed，并不保证相同新 head、增强或 batch 序列。
- `investigation/tf32_independent_comparison.json` 的四个端点 reference accuracy 全部与历史相同，CE 最大差 1.3766884e-6；这支持端点精度偏差解释，不等于重建了历史训练随机过程。

## Task0 资产同源与解释边界

已重新计算两份原 Task0 `task_00.pt` 和两份 `fisher_state_0.pt` 的 SHA256，四项均与本次 `runtime_receipts.jsonl` 的 `complete_resume` 完全一致；旧实际 shell log 指向的就是这四份文件。新旧不是换了 Task0 数据或用了别的起点。两条新分支各自恢复了自己的 normalized Fisher，因此“起点效果”同时包含权重、旧 head 和保护矩阵的差异，并未只隔离 Hessian/Fisher trace。

检查冻结 `as1_normfisher_gam_intask.py`、`seqlora.py`、`random_reproduce.py` 与当前 CopyFlatCV 对应文件的 SHA256，三项相同。历史运行日志是旧实际执行的第一手配置证据；当前源码 / 冻结源码可以说明本次实际执行分支，但不单独构成整个 6 月运行逐位复现证明。

不应与 ImageNet-R 的 ratio-FS 对照混合：那组是不同数据集、λ=1,200,000、seeds3/4/5，并非这里的 ImageNet-A/rank10/λ2000/seed0 GAM 主链。当前结论仅限本次 seed0：只保留 GAM 起点、后续统一 SGD 的 A 对照没有提高两任务平均 accuracy；全程 GAM 旧链的优势仍需按自身协议理解。要把下降归因到后续 GAM 开关，需要在同一新 RNG / 精度协议下增加 Task1 优化器的配对对照，而不是直接相减新旧实验。
