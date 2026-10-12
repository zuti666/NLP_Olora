# FS-EWC RQ2 seed0 自动执行

**新批次：用户已授权使用空闲GPU执行完整三seed B1/B2/B3。见 [完整队列协议](full_rq2_20260920/EXECUTION.md) 与 [实时状态](full_rq2_20260920/formal_queue/STATUS.md)。本文以下为此前15作业seed0队列的历史执行记录。**

**2026-09-20：按用户要求将8阶段、15作业改为两线并行。等待 GPU0–3 的整个 TreeLoRA 队列 `completed`，并确认四张卡连续空闲60秒后，共同 sanity 在 GPU0/1 执行；随后 GPU0/1 跑 neighborhood→bridge，GPU2/3 跑 training→table_a→trajectory→path，最后统一汇总。GPU0/2 对应 SGD，GPU1/3 对应 GAM；科学设置不变。**

旧等待进程 PID `541721` 已在未启动 GPU 实验时安全停止，TreeLoRA 前驱未受影响；改双线前计划与文档归档至 `campaign_seed0_after_gpu03/before_dualline/`，替换凭据见 [SUPERSEDED_FOR_DUAL_LINE.json](campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。首次双线运行保留在 `campaign_seed0_dualline_gpu03/`，因邻域作业参数冲突退出为 failed。当前恢复使用 `campaign_seed0_dualline_recovery1/`。此前 GPU4/6 队列的 [停止记录](campaign_seed0/STOPPED_BY_USER.json) 和 `campaign_seed0/before_gpu03_requeue/` 归档继续保留。

本次接续使用 `experiment-queue` skill；共享环境未修改，生产解释器仍是 `Pilot_new`。Pillow 11.3 仅安装到本实验候选依赖目录用于排查，正式流程未采用。

## 当前状态

最新核验快照为2026-09-20 17:30马德里时间：10/15作业complete、2个neighborhood运行中（43/97、46/97）、3个待执行，无新失败。两条trajectory与path均完成，path所有gate通过。见 [CURRENT_STATUS.md](monitor_20260920/CURRENT_STATUS.md)；下述15:27:22 UTC状态是恢复启动后的历史快照。

恢复已成功启动：队列PID `1341023`，pipeline启动于2026-09-20 15:26:06 UTC。15:27:22 UTC核验快照为：新版两条sanity通过、原训练均 `Already complete` 且资产哈希未变、两条A端点评估complete；两条neighborhood和两条trajectory正在运行。下述恢复步骤已按该快照执行，最新进度以恢复目录中的状态文件为准，不能据此视为全量完成。

首次双线 pipeline 于2026-09-20 12:32:42 UTC启动，两条 sanity 通过；12:33:21 UTC `neighborhood_sgd` 报 `TypeError: Runner.measure() got multiple values for argument 'direction'`。已经运行的两条 Task1 训练继续各完成50 epoch，队列于12:39:07 UTC进入 failed。未启动的 A endpoint、bridge、trajectory、path、report 不能视为完成。

修复将 `measure` 的扰动张量参数从 `direction` 改名为 `perturbation`，调用中保存的 `point.direction` 元数据不变。当前主计划仍为 `campaign_plan.json`，配置为 `diagnostic_config_recovery1.json`，仅将诊断输出切换到 `results_seed0_recovery1/`。诊断脚本哈希变化后重做 sanity；旧 `results_seed0/` 中通过的 sanity 和 r0 保留，不跨协议复用。两条训练的配置、源文件哈希、完整资产和配对随机过程已核验，原 runner 遇到 `COMPLETE.json` 返回 `Already complete`，无需重跑50 epoch。证据见 [training_readonly_summary.json](monitor_20260920/training_readonly_summary.json)，修复说明见 [RECOVERY.md](campaign_seed0_dualline_recovery1/RECOVERY.md)。

当前接续实时状态见 [QUEUE_STATUS.md](campaign_seed0_dualline_recovery1/QUEUE_STATUS.md) 和 [queue_state.json](campaign_seed0_dualline_recovery1/queue_state.json)。前驱状态固定为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`；恢复继续要求整个队列 `completed` 和四卡连续空闲60秒。

GPU0/1 的 kernel witness 在对应 sanity 前执行；GPU2/3 的 witness 分别作为两条 training 作业的 `preflight_command`，在 runner 确认已有完成结果之前执行。双方法新版完整 sanity 是两条线的共同前置。恢复 pipeline 状态写入 [STATUS.md](campaign_seed0_dualline_recovery1/STATUS.md) 与 [pipeline_state.json](campaign_seed0_dualline_recovery1/pipeline_state.json)。等待或预检通过均不表示剩余诊断完成。恢复子进程设置 `TMPDIR=/tmp`，避免 NFS 临时目录中的 multiprocessing 清理噪音，数值和训练参数不变。

## 自动顺序

| 阶段 | 执行位置 | SGD / GAM 作业 | 完成要求 |
|---|---|---|---|
| 共同前置 sanity | GPU0 / GPU1 | S02 完整测量检查 | 完整恢复、原生/有效权重等价、逐样本梯度与FD、端点一致、历史精度reference |
| 01线第1项 neighborhood | GPU0 / GPU1 | S03 两种 Task0 anchor 邻域 | r=0/.5/1/2/5/10/20，16公共方向，分层256 train |
| 01线第2项 bridge | GPU0 / GPU1 | S04 每个 anchor×两种真实t9方向 | r=0/.25/.5/1/1.5/2 |
| 23线第1项 training | GPU2 / GPU3 | S07 普通 / shaped Task0→SGD+EWC | 两条各50epoch已完成；GPU2/3 witness通过后runner读取COMPLETE并跳过重训 |
| 23线第2项 table_a | GPU2 / GPU3 | 新Task1端点raw Fisher等 | 同数据/坐标/head口径；a11采用当前原点复评 |
| 23线第3项 trajectory | GPU2 / GPU3 | S05 旧t0–t9完整checkpoint复评 | 当前FP32 train/test loss、accuracy、有效权重位移 |
| 23线第4项 path | GPU2 / GPU3 | S05 旧t9有效权重直线路径 | 9节点；必要时17；求积与同子集界检查均通过 |
| report | CPU，等待两线 | 一次汇总 | 双方法全部完成后输出完整报告 |

sanity 的两个作业通过后，01 与23两条线同时放行；各线内部按表中顺序执行，每张卡同一时刻只跑一个本实验作业。同一阶段的两个方法均通过后，再放行该线下一阶段；最终报告等待两线。失败会保留日志并阻止后继放行；已经运行的另一线作业保留自己的输出。不会因结果不支持假设而跳过报告。

这些阶段不是线性数据链：neighborhood、bridge、trajectory、path 均独立读取已有 checkpoint 与 sanity 原点测量；训练不读取 B/C 结果。表中线内顺序是资源安排，详见 [DEPENDENCIES.md](DEPENDENCIES.md)。旧 GAM t9 路径来自 GAM→GAM 链，新 A 为 shaped Task0→SGD Task1，两者保持独立。path 仍必须满足 `quadrature_converged=true` 和 `bound_passed_with_2pct_numeric_tolerance=true` 才能完成全量报告；不再把 path 作为 A 训练前置。

S01旧JSON重分析已经完成。未默认排入seed1/2完整序列、dense K100半径、full-model或四格；正式三seed证据仍需这些后续工作，不能把单seed称为统计结论。

## 精度排查与口径

首次sanity因直接用FP32复现历史CE失败，原始输出保存于 [失败记录](failed_attempt_01_historical_fp32_gate/README.md)。受控实验仅将patch-embedding卷积的输入与权重舍入至TF32有效精度，四个历史端点CE最大误差1.38e-6，accuracy全相同；详见 [独立审查](investigation/INDEPENDENT_REPRODUCTION_REVIEW.md)。Pillow11.3/12.1在本任务上的对照值相同，因此没有将Pillow候选当作修复。

正式新B/C/A全部使用FP32，历史TF32 reference仅用于兼容性验证，结束后精确恢复；不把历史随机增强train曲线与新确定性视图混合。

## 路径与命令

- 当前配置：[diagnostic_config_recovery1.json](diagnostic_config_recovery1.json)；旧配置与旧结果保留溯源
- 队列：[campaign_plan.json](campaign_plan.json)
- 依赖与并发核查：[DEPENDENCIES.md](DEPENDENCIES.md)
- 冻结源码：[source_frozen.json](source_frozen.json)
- 训练协议：[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)
- 最终报告预定路径：[report_seed0/RESULTS.md](report_seed0/RESULTS.md)；旧框架已归档至 `campaign_seed0_dualline_recovery1/before_fix/report_seed0/`，等待恢复诊断完成后生成。

接续入口（已有接续进程时不要重复启动）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/wait_then_run.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

如发生中断，先检查接续与 pipeline 状态、存活进程及代码/配置hash，再选择恢复入口；不要用旧 GPU4/6、旧单线或本次失败状态目录恢复新配置。训练中断的task-boundary恢复另需核对训练runner的 `--resume-last`；此次两条训练已经完成，保留原目录并跳过。队列不会静默覆盖已有训练目录。
