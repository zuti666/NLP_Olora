# 2026-09-25 六个失败实验的只读诊断

本报告核对原始 receipt、控制台日志、部分 CNN 任务矩阵和本次冻结源码；没有重启训练，也没有修改任何训练源码、配置或队列。当前状态为 `partial_failure`，正式实验 **83 completed、6 failed、10 held_dependency_failure**；另有 9 个完成的 smoke。任务编号以下采用 **1-based**；原日志 `Task 6` 对应第 7 个任务。epoch 编号通过已完成 epoch 的进度条与冻结循环顺序确定，失败 minibatch 下标未记录。

| 失败实验 | 首次失败位置 | 已完成任务 | 直接原因 |
|---|---|---|---|
| SDLoRA / Aircraft / seed 0 | 7/10，第 1 epoch 内 | 6/10 | CUDA_OOM |
| SDLoRA / Cars / seed 0 | 7/10，第 1 epoch 内 | 6/10 | CUDA_OOM |
| SDLoRA / CUB-200 / seed 0 | 7/10，第 1 epoch 内 | 6/10 | CUDA_OOM |
| SDLoRA / Flowers / seed 0 | 7/10，第 1 epoch 内 | 6/10 | CUDA_OOM |
| EWCLoRA / Cars / seed 0 | 10/10，第 3 epoch 内 | 9/10 | NONFINITE_SGD_GRADIENT |
| GAM-EWCLoRA / Aircraft / seed 1 | 2/10，第 1 epoch 内 | 1/10 | NONFINITE_SGD_GRADIENT |

## SDLoRA：四个 10-task 流均在第 7 个任务 OOM

**观察事实。** Aircraft/Cars/CUB-200/Flowers 的 seed 0 均正常完成原日志 `Task 5, Epoch 40/40`，进入下一任务后进度条仍为 `0/40`，在 `sdlora.py:607` 的训练 forward 失败。栈末端相同：ViT block → MLP → `timm/layers/mlp.py:42` 的 `fc1` → `F.linear`。这不是模型下载、数据读取、保存 checkpoint 或 SGD step 的报错。

四例均请求再分配 **296 MiB**，设备容量 **23.46 GiB**，仅 **72.88 MiB** 可用；进程已占 **23.38 GiB**，其中 PyTorch 已分配 **22.97 GiB**，保留但未分配仅 **166.88–167.09 MiB**。因此这是实际显存容量压力，不能只根据异常自带提示认定为碎片化或保证 `expandable_segments` 能解决。

**代码支持的机制与推断。** `models_LoRAbasedCL/sdlora.py:209–249` 每个任务替换为新建 backbone，`learn_alpha=True`，并读取之前的 A/B。`backbone/lora.py:188–224` 每次 forward 遍历此前任务的 Q/V 分支，且通过 `ParameterWrapper.forward`（430–436 行）对分支激活乘可训练标量。历史方向虽然冻结，标量梯度及向前层传播仍需要计算图；历史分支数量随任务数增长。以 batch=128、197 tokens、768 channels、float32 计算，仅每个历史任务的 Q/V 标量输入在 12 层中的张量体积合计就是 **1773 MiB**（量级说明，不是完整实测显存增量）；MLP 的 3072-channel 输出为 295.5 MiB，与本次 296 MiB 申请一致。

这是“实现中任务数增长导致激活显存增长”的有力解释，但本次没有每任务显存 profile，不能把全部 22.97 GiB 逐项归因，也不能声称每增加一个任务就常驻一整份 ViT。源码的 `base_vit = deepcopy(vit_model)`（470 行）带来固定的额外模型副本；新任务是替换 `network.backbone`，不能由此推导保留 10 个完整 backbone。历史 A/B 存储和分支计算则确实随任务增加。

**为什么 Het5 与 smoke 通过不矛盾。** Het5 只有 5 个任务，三个 seed 全完成；三个 receipt 的 `peak_cuda_bytes` 都为 21,792,056,320，约 **20.30 GiB**。两任务 smoke 和五任务 Het5 均未走到四个细粒度流的第 7 任务边界。下一次修复验证应覆盖至少第 7 任务乃至完整 10 任务。可考虑更大显存设备或严格等价的内存优化；如改 batch/累积策略/前向实现，需明确新协议并重新验证，不能直接混入当前三 seed 统计。本次未执行这些改动。

## EWCLoRA/Cars seed 0：末任务数值发散

**观察事实。** 第 9 个任务完成 40 epoch，最后显示 loss=0.014。第 10 个任务 epoch 1 loss=3.023，epoch 2 loss=40312.454；在 epoch 3 内触发 `FloatingPointError: Nonfinite SGD gradient`。栈经过 `as1_normfisher_fishershape.py:446` → `fishershape.py:331` → `run_job.py:51`，说明在交给底层 SGD 更新前，组合梯度已出现 NaN 或 Inf。

该配置是 **LR=.05、batch128、rank16、λEWC=80000、fs_direction_mode=alpha0、两次前向 wrapper、probe/fisher RNG 均保留**；它不是 `native_sgd`。wrapper 仍执行 clean/probe 两次求梯度，然后合成方向并加 EWC 梯度。`fishershape.py` 的合成代码没有对 alpha=0 短路，仍计算 `g0 + coeff*(g1-g0)`；若 probe 已非有限，零系数也不能保证数值安全。这是源码中可见的潜在传播路径，**日志不足以证明本次最早异常就在 probe 或 EWC 项**。loss 爆增是已观察的数值不稳定，不能据此单独断言 λ 是唯一根因。

只有前 9 个任务的 CNN 矩阵；最后已完成阶段的 task-macro accuracy 是 **34.2633%**，已完成阶段均值是 **42.2576%**。它们不是 10 任务 FAA/AAA，不能填入最终表。seed 1/2 被依赖门控保留未启动，不能报告为已经失败，也不能用不同历史协议结果自动补齐。

## GAM-EWCLoRA/Aircraft seed 1：第 2 个任务梯度非有限

**观察事实。** 首任务完成 40 epoch，末 epoch loss=5.194、Test_accy=10.51%；进入第 2 任务，epoch 1 尚未完成就触发相同的 finite-gradient gate。栈为 `ewclora_youyue_fitArchitechture_gam.py:519` → `gam.py:283` → `run_job.py:51`。首任务已经学习很差但仍产生有限结果；不能把 10.51% 当作最终 FAA/AAA，也不能把“准确率低”本身等同于 NaN。

该配置为 **LR=.05、λEWC=2000、GAM rho=.01、norm_rho=.1、gamma=.02**。GAM 方向与缓存的 EWC 梯度在 SGD 前组合；日志未保存失败 batch 各分支梯度，无法区分最早非有限值来自 clean CE、GAM 的哪一步或 EWC。必须将其表述为数值不稳定，不能宣称已确定唯一运算根因。

同组 seed 0 和 seed 2 完成，FAA/AAA_stage 分别为 **46.69/43.67876984** 与 **45.25/43.14815873**。如汇报这两次的均值/标准差，必须注明 **n=2 successful runs，另 1 次非有限失败**；不得标成完整 3-seed 结果或隐去失败。

## 指标与溯源边界

六个失败 JSON 都没有 `cnn.final` / `cnn.matrices.final`，只有 `t00…t05`、`t00…t08` 或 `t00`。原日志 `Average Accuracy (CNN)` 还可能采用历史总体样本 top1 曲线平均，不能直接当作本表的 task-macro stage-averaged AAA；诊断 JSON 的中间值从各阶段矩阵明确复算，并加 `NOT_FINAL` 标记。矩阵未评任务的上三角 NaN 是结构性占位，不是本次非有限训练判定依据；真实失败由底层梯度检查给出。

共有 10 个 held：四组 SDLoRA 的 seed1/2（8 个）以及 EWC/Cars 的 seed1/2（2 个）。它们尚未启动；GAM/Aircraft seed2 已完成，不能误算为 held。

精确配置、receipt/日志/部分矩阵 SHA256、原始完整 traceback、最后有限阶段、已验证源码 SHA256 均在同目录 `failure_details.json`。已重新核对相关源码文件和配置与启动时 manifest 一致。状态/进程是否仍活动由主审计另行报告；此报告不凭旧 PID 推断当前活动进程。
