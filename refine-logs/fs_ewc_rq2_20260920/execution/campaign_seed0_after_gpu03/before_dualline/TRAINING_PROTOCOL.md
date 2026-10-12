# RQ2 补训练执行协议

2026-09-20。训练入口为 `train_continuation.py`。本批已按用户新授权配置接续：等待 GPU0–3 的整个 TreeLoRA 队列 `completed`，并确认四张卡连续空闲60秒后，先由主队列完成 seed0 S02、S03/S04、S05，再排 A。物理 GPU0 为 SGD 分支，GPU1 为 GAM 分支；全部科学设置不变。准备阶段的 CPU 恢复检查不表示训练完成，实际完成以每条运行目录的 `COMPLETE.json` 和 checkpoint 为准。正式 seed1/2 配置另行准备，不把配置文件视作已排期。

前驱状态：`/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`。接续入口为 `wait_then_run.py --plan campaign_plan.json --execute`；实时状态见 `campaign_seed0_after_gpu03/queue_state.json` 和 `QUEUE_STATUS.md`。实际 pipeline 启动后才生成同目录 `pipeline_state.json` 和 `STATUS.md`。此前 GPU4/6 队列停止记录和更新前文档保留于 `campaign_seed0/`，后者位于 `before_gpu03_requeue/`。

## 1. seed0 A：两条同源单 Task1 续训

保持已有两种 Task0 anchor，分别恢复各自完整 `network_state`、各自 normalized Fisher 和生命周期状态。两个分支的后续优化器都为 SGD；`optimizer_type_by_task=None`、`lambda_flat=0`，只训练 Task1，50 epoch。普通和 shaped 起点的保护矩阵不同，结果不能解释为已经隔离 Fisher 保护。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py --mode continuation --method sgd --seed 0 --gpu 0
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py --mode continuation --method gam --seed 0 --gpu 1
```

这两行由接续队列在 A 阶段分别调度到物理 GPU0、1，不能绕过前驱与前序科学检查直接启动。runner 先用 `nvidia-smi -i 0/1` 解析 GPU UUID，再隔离该 UUID；队列已有 `CUDA_VISIBLE_DEVICES=GPU-...` 时要求它与指定物理卡一致。配置中的 `device=["0"]` 是隔离后的逻辑卡编号。前驱完成、四卡连续空闲60秒以及实际作业前的空闲检查由上层队列执行。

输出分别为 `execution/training/A_sgd_to_sgd_seed0` 与 `A_gam_to_sgd_seed0`。核心文件：

- `checkpoints/task_01.pt`：完整权重；同目录保留原 A/B、FC、Fisher sidecar。
- `checkpoints/rng_state_01.pt`：checkpoint 时的 Python、NumPy、torch CPU/CUDA RNG；任务之间使用下述明确的 reset 协议。
- `trainer_outputs/**/**_cl_metrics.json`：原 trainer 的 CNN seen-class accuracy 矩阵；a21、a22 来自新训练后的评价，首行 a11 是继承的历史前缀。当前环境与历史环境端点存在细小数值差异，新的 A 表和 forgetting 不得直接使用该历史 a11。
- `task0_local_metrics.jsonl`：Task0/Task1 各自旧任务 train/test 的 deterministic local/seen CE、accuracy 和有效权重位移。保留原始 CE，D0 由自身 Task0 相减；没有截零。train 与历史缓存随机增强 train 不是同一口径。
- `runtime_receipts.jsonl`：完整恢复 hash、旧 base hash、逐任务结束事件。
- `rng_receipts.jsonl`：新 head hash、样本清单 hash、每次 train loader 遍历的全部样本索引序列 hash、首 batch 增强图像和标签 hash。
- `run_manifest.json`：冻结源码清单 SHA、runner SHA、最终配置、依赖版本、GPU UUID。
- `COMPLETE.json`：`status="complete"`；失败时写 `FAILED.json` 并返回非零。

raw trace 不在训练中近似计算；后续离线诊断使用本次保留的完整 Task1 checkpoint 与对应 Task0 anchor，采用统一固定256样本逐样本定义。历史 normalized Fisher sidecar 仅供 EWC 训练，不能当成新增 raw trace。

新表的 a11 必须使用 `task0_local_metrics.jsonl` 中 `task=0` 的 `test.seen_accuracy`，或 endpoint 诊断入口在同一环境/精度/transform 下重测的 anchor accuracy。Task0 只有20个 seen 类，该值与 local accuracy 相同。该测量发生在完整恢复后、Task1 head 创建前，已保存/恢复 RNG，不改变配对训练随机过程。汇总产物保留历史 a11、重测 a11、差值及来源；不覆盖历史指标文件。新的 backward-transfer/forgetting 同样以重测 anchor 为基准。若后续精度修复使二者一致，仍保留该可追溯规则。

## 2. 随机配对及历史兼容边界

协议名 `rq2_shared_task_rng_v1`。每次进入 Task t 的 `incremental_train` 前，对两个方法统一重置 Python、NumPy、torch CPU/CUDA RNG：

`task_seed = 20260920 + 100003 * training_seed + 1009 * task_id`。

重置发生在原 `IncrementalNet.update_fc` 之前，继续使用原 `SimpleLinear` 的 `kaiming_uniform_(nonlinearity="linear")` 初始化。新 head hash 只覆盖新增20行；旧20行保留各自起点。每个 DataLoader 使用独立 `torch.Generator`，其 seed 为 `task_seed + 10000019 + 7919 * loader_serial`，每任务序号从0开始。train sampler 与 worker 内 crop/flip RNG 因而不依赖优化器额外前反传次数。固定8 workers，关闭 persistent workers。worker 数不足1时拒绝启动，避免模型 RNG 与增强 RNG 在同一进程混用。

配对核验必须比较两个 A 输出的：Task1 `head_initialized.new_head_sha256`，train `loader_created` 的样本清单和 generator seed，前50轮 `loader_iteration.indices_sha256` 与 `first_batch_pixels_targets_sha256`。后者验证实际增广输入，不只相信相同 seed。训练后 Fisher 再遍历 train loader 时会有额外 iteration，单独保留。

历史 checkpoint 没有历史训练 RNG。因此这两行 A 是新的严格配对延续，不能声称逐位复现6月旧 Task1。seed1/2 完整序列使用相同明确协议；历史 seed0 序列与新协议有来源差别，正式汇总必须披露，不能只凭参数相同宣称完全同源三 seed。

准备阶段 CPU 检查已经通过：两 anchor 完整350 key恢复逐位相同、各自24块 Fisher 有限非负且 trace≈1、两个 Task1 新 head 的 SHA256 一致：

`05d16669f438d5ad33621d339fea510fef78c6c917131a70bc769873d592d7b2`。

另通过 seed1 GAM fresh-LoRA/head 构造及冻结目录 `src.trainer` import。记录在 `training/cpu_validation/*/CPU_VALIDATED.json`。这些检查没有做 GPU 前向或训练；正式训练前依赖统一 S02 测量恢复 gate。

## 3. 固定训练设置与最小运行时改动

配置继承冻结源码的 `base_s1c_normaccum_e50_initlr004_lr002.yaml`，另存最终 JSON。ImageNet-A；固定类别顺序；10×20类；ViT-B/16；LoRA rank10；batch128；Task0 lr=.04，后续=.02；每任务50 epoch；cosine；momentum0、weight_decay0。EWC λ2000、gamma=.9、eta=0；任务和累计 Fisher 均归一化；训练估计器显式 `batch_empirical`、fit train、最多100 batch，无 heldout split。GAM rho=.05、normrho=.2、beta1/2/3=1、gamma=.1、lambda_flat=1，仅 sequence GAM 开启。

关闭 in-task Fisher、weight merging、endpoint averaging、dual ascent、projection、delta-W penalty。原 trainer 的额外 full-Fisher、NME class means、feature drift/cache 不属于本任务，关闭以避免占用离线诊断预算；标准 CNN 评价和 EWC Fisher 完整保留。Task0 CE/accuracy 的新增前向在 eval、deterministic transform 下执行并保存/恢复主进程 RNG，不改变训练。

运行时只覆盖四类行为，原训练目标/优化器更新循环未改：

1. `timm.create_model` 强制 `pretrained=False`，从已审查 Task0 manifest 注入全部300个 frozen backbone state key，构造后逐位校验。两个历史 anchor 的这300项已证实相同。续训随后再严格载入完整 anchor 的350项。无需下载或信任 timm 可变默认别名。
2. 原 `load_task_checkpoint` 的 sidecar/lifecycle/Fisher 逻辑后，再完整载入 `network_state` 并逐位校验，避免仅恢复 adapter 而换掉 pretrained base。
3. task RNG reset、独立 loader generator 和配对审计收据。
4. 逐任务保存后增加 RNG/有限性检查、旧 head 未变检查、deterministic Task0 测量。任何异常返回非零，不以保存了部分checkpoint视作成功。

原 `CopyFlatCV` 和冻结 `execution/source` 均未被修改，patch 在单个 Python 进程运行时生效。

## 4. 正式 seed1/2 准备配置

`configs/training_plan.json` 列出所有命令、依赖及预期 receipt，当前只是准备文件。四条完整序列的入口示例：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py --mode sequence --method sgd --seed 1 --gpu 0
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py --mode sequence --method gam --seed 1 --gpu 1
```

seed2将上面 `--seed 1` 改为2。每条保存 `task_00.pt` 到 `task_09.pt`，B/C共用这一条链。正式 A 每 seed 可复用对应 SGD chain 的 Task1 行，再从 GAM chain 的 Task0 接一段 SGD Task1（`--anchor-dir`/`--anchor-metrics` 的完整命令已在计划 JSON）。由于新完整序列也用上述逐任务 reset 和同一 loader 协议，两行可以核验真实配对，无需再训普通 Task1。

准备预算共44个50-epoch任务段：seed0 A 两段，四条完整序列40段，seed1/2 shaped→SGD 两段。该数字不是已消耗的 GPU 时间。正式新seed raw neighborhood/path和dense damage诊断另计。

任务边界中断可对原命令添加 `--resume-last`：恢复该输出目录最后完整 task checkpoint 和 CNN prefix，再继续下一task。中途 epoch 不支持精确续接，重做当前未完成 task；共享 task RNG 确保同协议重启。无 `--resume-last` 时拒绝覆盖已有运行。
