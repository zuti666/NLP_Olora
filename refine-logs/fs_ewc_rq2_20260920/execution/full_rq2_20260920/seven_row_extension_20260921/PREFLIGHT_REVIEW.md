# ImageNet-A 七行扩展：只读实验前审查

## 判定

**有条件通过，可启动受依赖门控制的 controller；当前只放行两个 GPU smoke。** 未发现仍会阻塞 smoke 的静态代码、数据、父状态、CPU 恢复或队列问题。六个 50-epoch 正式续训仍必须等待两个真实 smoke 都生成有效 `SMOKE_COMPLETE.json`，且 `validate_smokes.py` 生成 `smoke/VALIDATED.json` 后才能由 DAG 自动放行。

本审查没有运行 GPU、训练或模型前向；截至审查时 smoke 尚未执行。因此这里不是正式六次训练的完成判定，也不是结果或科学主张的验证。

## 旧正式实现、数据与评估

- 冻结源码清单 [`source_frozen.json`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source_frozen.json) 的 SHA-256 为 `fe3b5ef75e89bab32010c1aa9a0581cbaeb2b068eaa2f4718f11a9cc314d0a2b`；逐一复算 199 个条目，0 个不匹配。正式旧 runner 的 SHA-256 为 `a4de971877e8bee0f7da9efbb010fb2f7c7df30a7920286ee3196cbe4dc4afe8`。
- 实际数据根为 `/data/115-2/users/liying/datasets/imagenet-a`，`train/`、`test/` 各有 200 个同序 WNID 类目录。源码 `iImageNetA` 先尝试 `imagenet-a/{train,test}`，`class_shuffle=false`，类别顺序为 identity `0..199`，任务宽度为 20。
- 资产计数为 train 5,981、test 1,519；code task0 的 train/test 为 1,259/307，code task1 的 train 为 997，0–39 类联合 test 为 561。按 ImageFolder 排序重算的路径/标签 SHA 与正式收据一致：task0 train `02b461...6911`、task0 test `884139...1860`、task1 train `8c9fb2...255`、0–39 test `a3deb9...71c`。完整值见配套 JSON。
- [`seqlora.py`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/seqlora.py) 对 code task1 用类 20–39 构造训练集，并用类 0–39 构造 `test_loader`。[`trainer.py`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/src/trainer.py) 在 `incremental_train()` 后无条件调用 `model.eval_task()`；`BaseLearner.eval_task()` 对该联合 loader 做 CNN 推理并写分任务与总准确率。因此 `endpoint_eval=false` 只关闭附加 endpoint 诊断模块，**没有关闭正式 40 类 CIL test 评估**。
- 六个正式父 anchor（SGD/GAM × seeds 0/1/2）均为 `status=complete`、`completed_tasks=1`；`task_00.pt`、`fisher_state_0.pt`、唯一 metrics JSON 均存在、非空，实算 SHA 与各自 `COMPLETE.json.validation` 相符。新 runner 还会在每次启动时重新核对同 seed、同 anchor 方法、父 identity、50 epoch、FP32/TF32 设置、源码和 runner 指纹。
- 正式配置开启 `mechanism_eval=true`、`mechanism_grad_stats=true`、任务 checkpoint 和非空 Fisher；采用 `batch_empirical`、fit loader、`fisher_split=false`、`approximate_fisher_in_task=false`。旧正式收据显示每任务 50 个训练遍历加 1 个末端 fit-Fisher 遍历。`weight_merge`、`in_task_weight_merge`、`endpoint_weight_merge`、`task0_trajectory_eval` 均关闭，无 trajectory averaging/merge。

## 新 runner 与六个配置

- 冻结候选 [`train.py`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/seven_row_extension_20260921/train.py) SHA-256 为 `4e87c9312de8f40bfa5f2f6e6df15853e6ba65c7386ea8dcb4426effe8c1b492`。两臂被硬约束为：
  - `shaping_only`: GAM task0 父状态 → GAM code task1，`ewc_lambda=0`；
  - `task2_shaping`: SGD task0 父状态 → GAM code task1，`ewc_lambda=2000`。
- runner 只允许 `resume_from_task=0`、`max_train_tasks=2`，且 patched incremental/save 只允许训练和保存 code task1。它先走原有 Fisher/lifecycle 恢复，再以父 `task_00.pt` 的完整 `network_state` 做 strict restore 和逐 tensor 相等验证；保存后又 strict reload `task_01.pt` 并逐 tensor 验证。父 checkpoint、Fisher、metrics、manifest 与 COMPLETE 的路径和 SHA 都进入实验 identity。
- 六个配置的真实 CPU 构建/完整父状态及 24 个 Fisher tensor 恢复均通过；实际 optimizer 为 GAM，lambda 分别为 0/2000。每个 seed 的两臂产生相同的新 20 类 head hash，符合共享 task RNG；CPU 检查明确未做 forward、梯度或训练。
- 机制、Fisher、连续 Q/V 位移、RNG 与 checkpoint 收据均已接入。`d_QV_cont` 由同一持续 LoRA 分支的 24 个有效 Q/V `BA` 端点差计算，排除分类头；未启用 fresh adapter 或 trajectory merge。
- completion validator 要求：2×2 `t01/final` CIL 矩阵、code task1 机制步和正确 lambda、精确父恢复、50 个训练遍历 + 1 个 fit-Fisher 遍历、最终 50/50 文件日志、全输入索引与每遍历首 batch augmentation 对正式同 seed 协议配对、task1 checkpoint/Fisher/RNG state、有限 Fisher、有效 Q/V 位移。审查期间发现的 `ns.method` 字段错误、把 51 个 loader 遍历误判为 50 个、以及错误要求文件日志包含全部 50 个 epoch 的问题，均已在上述冻结 SHA 中修正。
- CPU 合同测试 7/7 通过，覆盖六配置、臂标签拒错、smoke 隔离、六父 identity、真实 51-pass 收据与 frozen trainer 的 final-only 文件日志。

## Smoke 门与队列

- [`plan.json`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/seven_row_extension_20260921/plan.json) SHA-256 为 `f3caee8133ca463b7bb06f11ce6a5e181fe0e830eca6a1e874254ad1f0f6f208`。campaign dry-run 通过：10 stages、10 jobs、`max_parallel=6`、所有代码文件存在且带指纹。
- 两个无依赖 GPU job 是隔离的 seed0 一轮 smoke。runner 将其输出强制写入 `smoke/training/{arm}_seed0/`，仅生成 `SMOKE_COMPLETE.json`；不会生成正式 `COMPLETE.json`，也不会落入正式 `training/`。
- [`validate_smokes.py`](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/seven_row_extension_20260921/validate_smokes.py) SHA-256 为 `12192698030431971e67ac9c3e32ebd7ae162495d7c3aa31509df46287dfc1db`。它重新调用 runner completion validator，并要求真实 GAM 梯度、lambda0 无 EWC 梯度、lambda2000 有 EWC 梯度、有限机制/Fisher 值、`fisher_num_samples=997`、无 path Fisher/merge、40 类 CIL 矩阵、旧任务 307/1259 样本诊断、checkpoint/RNG 收据以及每 job 的 CUDA forward/backward witness。
- 六个正式 job 全部只依赖 `smoke_validation`；最终 CPU report 依赖六个正式 job。任一进程非零退出或 receipt 不满足条件都会阻止下游启动。GPU 0–6 仍需满足空闲、UUID 和共享锁条件，队列不会抢占其他进程。

## 放行条件与限制

controller 可按当前指纹启动并等待 GPU；正式训练的条件放行必须由以下实际文件达成，不能用本静态审查替代：

1. 两个 `SMOKE_COMPLETE.json` 均通过 runner 的实际输出校验；
2. `smoke/VALIDATED.json` 为 `status=passed, validated_smokes=2`；
3. controller 冻结指纹仍与本审查记录一致。

当前未发现 blocking finding。唯一未完成的必要证据是 GPU smoke 本身；它将检验本次 CPU/静态审查无法覆盖的 CUDA forward/backward、真实 GAM/EWC 梯度路径、训练后评估与写盘行为。
