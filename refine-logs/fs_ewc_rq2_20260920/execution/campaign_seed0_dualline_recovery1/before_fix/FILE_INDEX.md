# FS-EWC RQ2 文件路径索引

双线后台队列已启动，PID `600549`，当前为 `waiting`，尚未运行新实验。状态：[QUEUE_STATUS.md](execution/campaign_seed0_dualline_gpu03/QUEUE_STATUS.md)。

当前采用 GPU0/1 与 GPU2/3 两条执行线，新实验仍为 pending。新队列的启动和等待状态以 [QUEUE_STATUS.md](execution/campaign_seed0_dualline_gpu03/QUEUE_STATUS.md) 为准。

2026-09-20：按用户要求调整执行顺序。等待 GPU0–3 的整个 TreeLoRA 队列完成，并确认四张卡连续空闲60秒后，先在 GPU0/1 完成共同 sanity；随后 GPU0/1 执行 neighborhood→bridge，GPU2/3 执行 Task1 training→A endpoint→trajectory→path，最后统一汇总。GPU0/2 对应 SGD，GPU1/3 对应 GAM；保留8个阶段、15个作业及全部科学设置，阶段按依赖并行。排队不表示实验完成。

## 实验安排与配置

| 用途 | 绝对路径 |
|---|---|
| 完整实验计划 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_PLAN.md) |
| 任务清单与当前接续安排 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/EXPERIMENT_TRACKER.md) |
| 执行顺序、GPU分配与接续条件 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/EXECUTION.md) |
| 真实数据依赖、两线排程与并发写核查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/DEPENDENCIES.md) |
| 15作业、8阶段的当前接续配置 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json) |
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

接续入口为 `execution/wait_then_run.py --plan execution/campaign_plan.json --execute`；实时接续状态写入 `execution/campaign_seed0_dualline_gpu03/queue_state.json` 与 `QUEUE_STATUS.md`。实际实验 pipeline 启动后才生成同目录 `pipeline_state.json` 与 `STATUS.md`。前驱状态为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`。

旧等待进程 PID `541721` 已在未启动 GPU 实验时安全停止，TreeLoRA 前驱未受影响。改双线前的计划与文档保存在 `execution/campaign_seed0_after_gpu03/before_dualline/`，替换凭据为 [SUPERSEDED_FOR_DUAL_LINE.json](execution/campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。

旧 GPU4/6 队列已于此前按用户要求停止；本次是新的接续授权。更新前的计划与文档保存在 `execution/campaign_seed0/before_gpu03_requeue/`，原停止凭据继续保留。

| 用途 | 绝对路径 |
|---|---|
| 历史 GPU4/6 队列停止凭据，保留溯源 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0/STOPPED_BY_USER.json) |
| 历史TF32精度差异独立审查 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/investigation/INDEPENDENT_REPRODUCTION_REVIEW.md) |
| 新实验汇总框架；各新项目pending，不是完成结果 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md) |
| 首次sanity失败记录说明 | [/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/failed_attempt_01_historical_fp32_gate/README.md) |
