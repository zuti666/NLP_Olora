# ImageNet-A two-task six-row implementation review

审查日期：2026-09-20。审查者未参与本次 production 文件编写；继承此前代码溯源背景，按 experiment-bridge 的独立代码审查清单进行只读审查。仅运行 CPU toy checks，未使用 GPU、未启动实验训练。

## 结论

**最终状态：可以部署当前队列。** 训练、队列与关键评估 gate 均已审查，发现的记录问题已经修复。队列可以后台等待 GPU 4/6 空闲，再执行真实 smoke；只有 smoke 通过后才允许正式训练。这不表示 smoke 或正式实验已经通过。

审查文件：`scripts/imageneta_two_task_table/{learner.py,train.py,config.json}`，以及被复用的 R runtime、GAM additive step、CopyFlatCV 训练/存取/Fisher/weight-merge 调用链。CPU 测试 `tests/test_imageneta_two_task_table.py` 的 **16 项全部通过**。

## 已核实的关键行为

- **六行协议与阶段 λ。** `train.py:18–25,58–94` 的 optimizer/保护映射正确。Task0 的两个共享 anchor 都以 λ=0 构建；Task1 的 vanilla/shaping_only 使用 0，其余四行使用 2000。原 `baseLoRA._load_auxiliary_task_state:734–770` 仅恢复 Fisher、trace mass、trace 统计，不覆盖新分支的 `_ewc_lambda` 或 `_lambda_ewc`，因此不会把受保护分支恢复成 anchor 的 λ=0。
- **初始化。** 共享 runtime `install_runtime:157–228` 仅抽取指定 ImageNet-A checkpoint 的 300 项冻结 backbone state，新 A/B 与 head 保留当前 seed 初始化；timm 被指定 `pretrained=False`，不从另一预训练别名取得权重。Task1 在原生 sidecar restore 后，又 strict 恢复完整 `network_state`，并检查全部张量相等。
- **共享文件。** `train.py:112–121` 硬链的 checkpoint 和 trajectory 都是已完成 Task0 文件。Task1 写 task_01、fisher_state_1、lora_w_a_1/b_1、fc_state_1、experiment_identity_1 和 trajectory/task_01；未发现写回 task_00 或 trajectory/task_00 的执行路径。追加的 runtime/rng receipts 使用 `copyfile`，不共享 inode；Task1 完成后再检查 Task0 checkpoint 未变。
- **种子与数据。** 共享 runtime 在每个 task 开始重新设置相同 task seed，DataLoader 使用独立 generator。new-head 初始化在重设 seed 后执行。六个 Task1 分支共享相同数据顺序/新 head RNG 协议；不是声称恢复上个进程的连续随机流。CE 质量和训练 Fisher 使用同一个确定性 current-train loader，`source='train', mode='test'`；未使用 test 标签选 snapshot。
- **训练。** 新 `_step_batch_with_log` 实际调用已审查的 additive_step；普通 SGD 一次 CE backward，GAM 原生四次 CE backward，EWC 梯度在 clean point 缓存后加到最终方向，不进入 GAM probes，也不新增第五次 CE。旧 head 行在当前任务 local CE、zero momentum/weight decay 下不更新。
- **snapshot/merge/Fisher 顺序。** epoch20/40 的 Q/V A/B 因子与 head/CE 落盘；混合仅修改 A/B，head 不参与混合。CE 权重是 population-z-score softmax，β=3；两个不同 CE 时约 0.9975/0.0025。随后 `.7*endpoint+.3*weighted_path` 分别作用于 A/B，不能解释为 BA 的线性平均。merge receipt 写入后才允许估计 current Fisher。原生 WMergeThenFisher 每任务只在混后端点估计训练 Fisher，不混路径 Fisher。
- **Fisher 口径。** 训练 estimator 是 Q/V effective-weight batch-gradient-square proxy，不是逐样本 EF。先归一化本任务 trace，γ=.9 累计并归一化，再执行 `.8*F+.2/d` 阻尼。新类继承正确的累计后 damping，没有继承 R runner 的 per-task damping。λ=0 分支也保留 Fisher 估计与 checkpoint，支持统一诊断。
- **任务数量和输出。** 保留原 ImageNet-A 10×20 类序，但 `max_train_tasks=1/2` 分别将 anchor/续训限定为前两个任务；不是把数据改成 2×100。Task1 accuracy matrix 从共享 anchor 的原 metrics 载入首行，再增加第二行。

## 首轮发现的记录问题（均已修复）

1. 已复核 `learner.py:61` 现在调用 `_record_mechanism_loss_stats`，恢复 task-level `train_clean_loss_mean` / `train_fisher_loss_mean`；同时记录 smoke 要使用的 `ewc_penalty_max`。
2. 已复核 `learner.py:32–38` 自定义 `_save_experiment_identity` 按 task 写入实际 lr/epochs；Task1 不再误写 `init_lr=.04`，而写 `.02`。

## 验证边界

运行命令：

```bash
CUDA_VISIBLE_DEVICES='' /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -m unittest discover -s tests -p test_imageneta_two_task_table.py -v
```

结果：16 tests，OK；覆盖原生 GAM/EWC 梯度、old-head/frozen-base/full-state lifecycle、精确 snapshot schedule、CE 权重、因子混合与非 BA 平均、归一化/累计/damping、merge-before-F 顺序。未替代真实 ViT 数据/GPU smoke。正式运行必须在 smoke 通过后开始。

## 队列与评估 gate 补审

审查 `pipeline.py`、`make_plan.py`、`verify_smoke.py`、`summarize_stage.py` 和实际 `campaign/plan.json`，并交叉检查 eval/summary 的关键运行 gate。

- 实际 dry-run 成功：**17 stages / 61 jobs / GPUs [4,6] / max_parallel=2**。包括 smoke 的 2 anchors + 6 branches + 6 EF/proxy evals + 1 summary + 1 numerical gate，以及正式 3 seeds 的 6 anchors + 18 branches + 18 evals + 3 prefix summaries。
- 正式 seed0 anchors 依赖 `smoke_verified`；后续 seed 依赖前一 seed 的汇总，不能越过失败 smoke/summary。smoke 3 epochs，保存 epoch1/2，EF subset=8；正式 50 epochs，epoch20/40，EF subset=256。`proportional_indices` 允许 subset=8 小于20类别，并记录部分类别分配为0；该结果只允许 sanity 输出，不允许论文表。
- GPU 4/6 必须同时满足无 compute process、显存<500 MiB、util<5%，才开始对应作业。忙时每15秒重查，不 kill、清理或抢占其他作业；按物理 UUID 绑定。每卡最多一个本队列作业。
- 任意进程非零退出或完成 receipt 不满足要求，都会停止后续 launch；已运行兄弟作业自然结束。CPU 独立合成检查已验证“exit0但receipt无效”仍导致 failed，formal dependent 为 blocked 且没有执行。证据目录 `/tmp/SYNTHETIC_A_QUEUE_REVIEW_ezu49ov4`；不含任何论文实验结果。
- 配置、plan、本目录Python、共享 runtime/additive step及核心 CopyFlatCV source 都在现有 fingerprint 检查范围；每个job启动前检查，resume还检查历史完成receipt未改变。失败重启不会悄悄改变实验预算。
- offline evaluation 要求完整 endpoint 中的因子等于 merge post factor receipt 和 current-Fisher factor receipt；head 要与 pre-merge snapshot 完全相同；训练 raw batch proxy 与离线重算需要满足 `absdiff <= 1e-6 + 1e-3*abs(training_trace)`。失败时不发布有效 COMPLETE。
- smoke numerical gate 检查两端 merge-before-F、proxy复算、阶段lr、1/4 CE backward预算和 λ=0/2000 的真实 EWC 激活。summary额外检查全部六行完成、六行配置/输入/RNG配对、共享anchor、三个seed协议一致，并拒绝 smoke/formal混合。
- 已读取另一个审查任务的 CPU 合成18run结果 `/tmp/SYNTHETIC_A_SIXROW_52aw2_xa/TEST_RESULT.json`，状态passed；这仅是schema/配对/错误拦截测试，不是实验结果。

summary provenance 指针问题已修复并复核：`summarize_stage.py:27–31` 将固定seed receipt和归档内receipt的 `summary` 均指向绝对路径 `through_seedN/summary.json`，保留与该归档内容匹配的原 hash。后续seed更新通用汇总不会改变先前seed的归档及receipt目标。

最终复核：修复后的 pipeline dry-run仍为17阶段/61作业；最新 `CPU_TESTS.log` 显示17 tests、OK。**无剩余部署 blocker，可以启动当前plan等待授权GPU空闲并执行smoke gate。** 本审查没有启动GPU作业，也没有把CPU合成验证认作研究结果。
