# FS-EWC RQ2 文件路径索引

最新核验：2026-09-20 17:36:08马德里时间，恢复队列全部15/15作业完成并通过验证，最终报告已生成：[RESULTS.md](execution/report_seed0/RESULTS.md)。设置、实际执行、历史一致性及结果解释见 [ANALYSIS.md](execution/analysis_setting_20260920/ANALYSIS.md)，复核台账见 [VALIDITY_LEDGER.json](execution/analysis_setting_20260920/VALIDITY_LEDGER.json)。目前只有seed0，不代表多seed确认。17:30的 [CURRENT_STATUS.md](execution/monitor_20260920/CURRENT_STATUS.md) 及下文运行中描述均为历史快照。

与用户08:25:49版完整计划对照：15/15仅覆盖精简seed0队列；完整B3四格当前seed0为2/4、正式三seed为2/12，仍缺无保护分支及正式B1/B2扩展。新版表格与缺口说明见 [TABLE_REVISION.md](execution/table_revision_20260920/TABLE_REVISION.md)：含已有Fisher/位移的[当前两行LaTeX](execution/table_revision_20260920/imageneta_current_complete.tex)、[完整计划差距](execution/table_revision_20260920/FULL_PLAN_GAP_AUDIT.md)，以及在low-rank-cl-new中重新溯源的[旧ImageNet-R六行修订版](execution/table_revision_20260920/imagenetr_legacy_mean_sd.tex)。

2026-09-20：实验已实际执行。原双线 pipeline 于 12:32:42 UTC 启动，两条 sanity 通过，两条 Task1 续训各完成50 epoch；12:33:21 UTC 邻域作业遇到参数重名错误，队列在两条训练结束后于12:39:07 UTC进入 failed。旧运行记录保留于 [campaign_seed0_dualline_gpu03/STATUS.md](execution/campaign_seed0_dualline_gpu03/STATUS.md)。

已修复 `Runner.measure` 的 `direction` 参数与同名元数据冲突，恢复队列PID为 `1341023`，pipeline于15:26:06 UTC启动。15:27:22 UTC核验快照：新版两条sanity及两条A端点评估已完成，两条训练已确认跳过重训；01线邻域、23线trajectory正在运行。当前恢复目录为 `campaign_seed0_dualline_recovery1/`，最新状态以 [QUEUE_STATUS.md](execution/campaign_seed0_dualline_recovery1/QUEUE_STATUS.md) 和 [STATUS.md](execution/campaign_seed0_dualline_recovery1/STATUS.md) 为准；恢复说明见 [RECOVERY.md](execution/campaign_seed0_dualline_recovery1/RECOVERY.md)。剩余诊断和最终报告尚不能计为完成。

2026-09-20：按用户要求调整执行顺序。等待 GPU0–3 的整个 TreeLoRA 队列完成，并确认四张卡连续空闲60秒后，先在 GPU0/1 完成共同 sanity；随后 GPU0/1 执行 neighborhood→bridge，GPU2/3 执行 Task1 training→A endpoint→trajectory→path，最后统一汇总。GPU0/2 对应 SGD，GPU1/3 对应 GAM；保留8个阶段、15个作业及全部科学设置，阶段按依赖并行。排队不表示实验完成。

## 实验安排与配置

| 用途 | 绝对路径 |
|---|---|
| 完整实验计划 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md) |
| 任务清单与当前接续安排 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md) |
| 执行顺序、GPU分配与接续条件 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md) |
| 真实数据依赖、两线排程与并发写核查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md) |
| 15作业、8阶段的当前接续配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json) |
| 当前 B/C 评估配置、旧checkpoint绝对路径 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config_recovery1.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config_recovery1.json) |
| A两条Task1补训练协议 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/TRAINING_PROTOCOL.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/TRAINING_PROTOCOL.md) |
| 普通Task0 → SGD+EWC配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_sgd_to_sgd_seed0.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_sgd_to_sgd_seed0.json) |
| shaped Task0 → SGD+EWC配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_gam_to_sgd_seed0.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/configs/A_gam_to_sgd_seed0.json) |

## 此前已完成的分析

| 用途 | 绝对路径 |
|---|---|
| 跨r代码、数据口径与证据审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_RADIUS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_RADIUS.md) |
| 跨T轨迹、checkpoint与恢复逻辑审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TASK_ASSETS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TASK_ASSETS.md) |
| 两任务对照、已有结果与缺失分支审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TWO_TASK.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/AUDIT_TWO_TASK.md) |
| 旧邻域损伤重分析，324行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_damage_seed0.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_damage_seed0.csv) |
| 经验容忍半径，40行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_tolerance_seed0.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_tolerance_seed0.csv) |
| 旧逐任务结果与CPU位移核验，20行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_trajectory_recomputed.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_trajectory_recomputed.csv) |
| 原始训练配置、指标与checkpoint来源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json) |
| 旧半径结果来源与重分析参数 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_reanalysis_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/radius_reanalysis_manifest.json) |
| 相关源码/脚本来源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/source_audit_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/source_audit_manifest.json) |

## 执行与排查记录

接续入口仍为 `execution/wait_then_run.py --plan execution/campaign_plan.json --execute`；当前实时接续状态写入 `execution/campaign_seed0_dualline_recovery1/queue_state.json` 与 `QUEUE_STATUS.md`，pipeline 状态写入同目录 `pipeline_state.json` 与 `STATUS.md`。前驱状态为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，恢复仍保留整个前驱完成和四卡连续空闲60秒的 gate。

新诊断输出为 `execution/results_seed0_recovery1/`；原 `results_seed0/` 中通过的旧版 sanity 和 r0 保留不改。诊断脚本哈希变化后重跑 sanity，不静默复用旧协议结果。两条既有训练完成情况、指标与配对摘要见 [training_readonly_summary.json](execution/monitor_20260920/training_readonly_summary.json)。旧报告框架已归档至 `execution/campaign_seed0_dualline_recovery1/before_fix/report_seed0/`，最终报告仍将写入 `execution/report_seed0/`。

旧等待进程 PID `541721` 已在未启动 GPU 实验时安全停止，TreeLoRA 前驱未受影响。改双线前的计划与文档保存在 `execution/campaign_seed0_after_gpu03/before_dualline/`，替换凭据为 [SUPERSEDED_FOR_DUAL_LINE.json](execution/campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。

旧 GPU4/6 队列已于此前按用户要求停止；本次是新的接续授权。更新前的计划与文档保存在 `execution/campaign_seed0/before_gpu03_requeue/`，原停止凭据继续保留。

| 用途 | 绝对路径 |
|---|---|
| 历史 GPU4/6 队列停止凭据，保留溯源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json) |
| 历史TF32精度差异独立审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md) |
| 已完成的seed0最终汇总 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md) |
| 本次设置、执行与结果的综合分析 | [ANALYSIS.md](execution/analysis_setting_20260920/ANALYSIS.md) |
| 实际优化器、损失、恢复与随机配对审计 | [AUDIT_EXECUTION.md](execution/analysis_setting_20260920/AUDIT_EXECUTION.md) |
| 原始历史日志与当前实验对照 | [HISTORICAL_COMPARISON.md](execution/analysis_setting_20260920/HISTORICAL_COMPARISON.md) |
| A/B/C完整测量、反例及59项校验 | [MEASUREMENT_ANALYSIS.md](execution/analysis_setting_20260920/MEASUREMENT_ANALYSIS.md) |
| 已完成两条训练的只读指标与配对摘要 | [training_readonly_summary.json](execution/monitor_20260920/training_readonly_summary.json) |
| 本次故障、修复与恢复依据 | [RECOVERY.md](execution/campaign_seed0_dualline_recovery1/RECOVERY.md) |
| 首次sanity失败记录说明 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md) |
