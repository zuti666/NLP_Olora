# FS-EWC RQ2 文件路径索引

队列已按用户要求停止；以下文件保留。

## 实验安排与配置

| 用途 | 绝对路径 |
|---|---|
| 完整实验计划 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md) |
| 任务清单，现已注明停止 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md) |
| 原执行顺序、GPU分配、停止说明 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md) |
| 15作业、8阶段的队列配置，现未运行 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json) |
| B/C评估配置、旧checkpoint绝对路径 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/diagnostic_config.json) |
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

| 用途 | 绝对路径 |
|---|---|
| 停止凭据，无存活队列进程且无正式作业启动 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json) |
| 历史TF32精度差异独立审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md) |
| 新实验汇总框架；各新项目pending，不是完成结果 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md) |
| 首次sanity失败记录说明 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md) |

