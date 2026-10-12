# FS-EWC RQ2 双线队列恢复记录

更新：2026-09-20 17:30马德里时间，10/15作业complete、2个neighborhood运行中（43/97、46/97）、3个待执行，无新失败。两条trajectory和path均完成，path求积与界检查全部通过；S_path为SGD .3680576、GAM .01479309。见 [CURRENT_STATUS.md](../monitor_20260920/CURRENT_STATUS.md)。下段15:27:22 UTC状态为早前恢复快照。

恢复已启动并越过原报错位置。2026-09-20 15:27:22 UTC核验快照：新队列 PID `1341023`，pipeline于15:26:06 UTC实际启动；新版两条sanity通过，两条训练均返回 `Already complete`，两条A端点评估已完成。两条neighborhood分别已保存5/97、8/97状态，GPU2/3正在执行trajectory。以上是当时快照，最新状态以 [QUEUE_STATUS.md](QUEUE_STATUS.md)、[STATUS.md](STATUS.md) 和 [pipeline_state.json](pipeline_state.json) 为准。bridge、path和最终报告尚不能计为完成。

## 故障经过

原运行目录为 `../campaign_seed0_dualline_gpu03/`。2026-09-20时间均为UTC：

| 时间 | 事件 |
|---|---|
| 12:32:42 | 前驱与GPU空闲条件满足，双线pipeline实际启动 |
| 12:33前后 | 两条完整sanity通过，01线邻域与23线训练放行 |
| 12:33:21 | `neighborhood_sgd` 报 `TypeError: Runner.measure() got multiple values for argument 'direction'` |
| 12:39:05–12:39:06 | 已在跑的两条Task1训练各完成50 epoch并保存完成凭据 |
| 12:39:07 | 队列结束为failed，后继阶段未放行 |

`Runner.measure` 原有名为 `direction` 的扰动张量参数；邻域调用既通过位置参数传入张量，又使用 `direction=meta` 保存方向元数据，触发参数重复绑定。修复仅将函数的张量参数及 `set_point` 引用改名为 `perturbation`，`point.direction` 元数据保持原格式。

## 恢复边界

- 主计划仍为 [campaign_plan.json](../campaign_plan.json)，运行目录改为本目录。双线DAG、GPU0/1与GPU2/3分配、8阶段15作业、科学配置全部不变。
- 新配置为 [diagnostic_config_recovery1.json](../diagnostic_config_recovery1.json)，相较旧配置仅把 `output_dir` 改为 `results_seed0_recovery1/`。
- 原 `results_seed0/` 中旧版sanity和r0保留不改。诊断脚本哈希改变后重新执行sanity，后续新诊断写入新目录，不跨协议复用旧结果。
- 两条 `training/A_*_to_sgd_seed0/` 保持不动。原runner在检查到 `COMPLETE.json` 后返回 `Already complete`；恢复实际日志已确认跳过，两条训练原资产哈希未变。
- A endpoint读取既有新Task1 checkpoint和新的sanity原点，结果写入 `results_seed0_recovery1/<method>/table_a/`。B/C继续使用旧Task0–Task9 checkpoint，不能与A训练链混淆。
- 恢复子进程使用 `TMPDIR=/tmp`，避免NFS上的multiprocessing临时文件清理噪音；共享依赖环境未修改。
- 前驱整队 `completed`、最终receipt `complete`、GPU0–3连续空闲60秒以及逐作业空闲检查全部保留。
- 原 `report_seed0/` 框架已归档至 [before_fix/report_seed0/](before_fix/report_seed0/)。最终报告仍写入 `../report_seed0/`，postprocess通过新配置读取 `results_seed0_recovery1/` 下全部结果，等待两线完成与数值gate通过。

## 已完成训练的复用依据

两条训练当前生成配置与 `run_manifest.json` 完全相同，runner SHA256和冻结源清单SHA256一致。每条均有 `COMPLETE.json`（`status=complete`、`completed_tasks=2`）、Task1结束事件、完整 `task_01.pt`、Fisher/RNG/LoRA/FC sidecar，以及唯一最终accuracy矩阵。两个Task0起点来自已有资产，本次各续训一个Task1。

两条训练的55条RNG收据完全相同，覆盖Task1 reset、新head、loader创建、完整50 epoch和额外一次Fisher遍历；`paired_rng_check` 返回passed且无差异。数值与原始路径见 [training_readonly_summary.json](../monitor_20260920/training_readonly_summary.json)。该摘要为训练资产只读核验，不替代后续raw Fisher、邻域或路径测量。

| Task0起点 | 当前FP32 a11 | a21（已见类） | a22（已见类） | 有效权重位移 |
|---|---:|---:|---:|---:|
| SGD | 85.667752 | 84.36 | 74.80 | 0.374950 |
| GAM | 87.622150 | 84.04 | 72.05 | 0.349876 |

a11使用当前FP32原点测量；尤其SGD历史矩阵继承的85.34不能代替当前值。该表仅为seed0，两个起点继承的EWC保护矩阵不同，不能据此声称已隔离Fisher保护或形成多seed统计结论。

## 记录保留

修复前代码、配置、文档和报告框架保存在 [before_fix/](before_fix/)。原失败队列见 [旧STATUS.md](../campaign_seed0_dualline_gpu03/STATUS.md)。更早的单线替换凭据 `../campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json`、GPU4/6用户停止凭据 `../campaign_seed0/STOPPED_BY_USER.json` 和各历史归档继续保留。

恢复前核验见 [RECOVERY_VALIDATION.json](RECOVERY_VALIDATION.json)，调度dry-run见 [campaign_dryrun.json](campaign_dryrun.json)。报告仍需全部诊断完成、共享数据及配对核验通过、路径求积收敛与同子集数值界检查通过，不能把当前运行状态写成全部实验完成。
