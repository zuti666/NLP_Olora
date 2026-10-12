# FS-EWC RQ2 seed0 自动执行

**2026-09-20：按用户要求将8阶段、15作业改为两线并行。等待 GPU0–3 的整个 TreeLoRA 队列 `completed`，并确认四张卡连续空闲60秒后，共同 sanity 在 GPU0/1 执行；随后 GPU0/1 跑 neighborhood→bridge，GPU2/3 跑 training→table_a→trajectory→path，最后统一汇总。GPU0/2 对应 SGD，GPU1/3 对应 GAM；科学设置不变。**

旧等待进程 PID `541721` 已在未启动 GPU 实验时安全停止，TreeLoRA 前驱未受影响；改双线前计划与文档归档至 `campaign_seed0_after_gpu03/before_dualline/`，替换凭据见 [SUPERSEDED_FOR_DUAL_LINE.json](campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。本次使用新的 `campaign_seed0_dualline_gpu03/` 目录记录接续与运行状态。此前 GPU4/6 队列的 [停止记录](campaign_seed0/STOPPED_BY_USER.json) 和 `campaign_seed0/before_gpu03_requeue/` 归档继续保留。

本次接续使用 `experiment-queue` skill；共享环境未修改，生产解释器仍是 `Pilot_new`。Pillow 11.3 仅安装到本实验候选依赖目录用于排查，正式流程未采用。

## 当前状态

双线后台接续队列已于 2026-09-20T10:41:00.340574+00:00 启动，PID `600549`，并确认存活、状态为 `waiting`；新实验仍为 pending。当前等待原因以 [QUEUE_STATUS.md](campaign_seed0_dualline_gpu03/QUEUE_STATUS.md) 和 [queue_state.json](campaign_seed0_dualline_gpu03/queue_state.json) 为准。前驱状态固定为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，必须是整个队列 `completed`，不能以某个进程退出或短暂空闲代替。

GPU0/1 的 kernel witness 在对应 sanity 前执行；GPU2/3 的 witness 分别作为两条 training 作业的 `preflight_command`，在训练前执行。双方法完整 sanity 是两条线的共同前置。实际 pipeline 启动后才生成 [STATUS.md](campaign_seed0_dualline_gpu03/STATUS.md) 与 [pipeline_state.json](campaign_seed0_dualline_gpu03/pipeline_state.json)。等待或预检通过均不表示正式实验完成。

## 自动顺序

| 阶段 | 执行位置 | SGD / GAM 作业 | 完成要求 |
|---|---|---|---|
| 共同前置 sanity | GPU0 / GPU1 | S02 完整测量检查 | 完整恢复、原生/有效权重等价、逐样本梯度与FD、端点一致、历史精度reference |
| 01线第1项 neighborhood | GPU0 / GPU1 | S03 两种 Task0 anchor 邻域 | r=0/.5/1/2/5/10/20，16公共方向，分层256 train |
| 01线第2项 bridge | GPU0 / GPU1 | S04 每个 anchor×两种真实t9方向 | r=0/.25/.5/1/1.5/2 |
| 23线第1项 training | GPU2 / GPU3 | S07 普通 / shaped Task0→SGD+EWC | GPU2/3 witness通过；Task1各50epoch，LR.02，λ2000，关闭后续shaping |
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

- 配置：[diagnostic_config.json](diagnostic_config.json)
- 队列：[campaign_plan.json](campaign_plan.json)
- 依赖与并发核查：[DEPENDENCIES.md](DEPENDENCIES.md)
- 冻结源码：[source_frozen.json](source_frozen.json)
- 训练协议：[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)
- 最终报告：[report_seed0/RESULTS.md](report_seed0/RESULTS.md)；运行未完时对应项为pending。

接续入口（已有接续进程时不要重复启动）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/wait_then_run.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

如发生中断，先检查接续与 pipeline 状态、存活进程及代码/配置hash，再选择恢复入口；不要用旧 GPU4/6 或旧单线状态目录恢复当前配置。训练中断的task-boundary恢复另需核对训练runner的 `--resume-last`；队列不会静默覆盖已有训练目录。
