# FS-EWC RQ2 seed0 自动执行

**2026-09-20：已按用户新授权配置接续。等待 GPU0–3 的整个 TreeLoRA 队列 `completed`，并确认四张卡连续空闲60秒后，执行当前 seed0 的8阶段、15作业。GPU0 执行 SGD 分支，GPU1 执行 GAM 分支；科学设置不变。**

此前 GPU4/6 队列的停止记录保留：[STOPPED_BY_USER.json](campaign_seed0/STOPPED_BY_USER.json)。更新前计划与文档已归档至 `campaign_seed0/before_gpu03_requeue/`；本次使用新的 `campaign_seed0_after_gpu03/` 目录记录接续与运行状态。

本次接续使用 `experiment-queue` skill；共享环境未修改，生产解释器仍是 `Pilot_new`。Pillow 11.3 仅安装到本实验候选依赖目录用于排查，正式流程未采用。

## 当前状态

后台接续队列已于 2026-09-20T10:28:32.134282+00:00 启动，PID `541721`；启动后已确认状态为 `waiting`，前驱仍为 `running`，尚未发起新GPU作业。当前等待原因以 [QUEUE_STATUS.md](campaign_seed0_after_gpu03/QUEUE_STATUS.md) 和 [queue_state.json](campaign_seed0_after_gpu03/queue_state.json) 为准。前驱状态固定为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`，必须是整个队列 `completed`，不能以某个进程退出或短暂空闲代替。

前驱完成且 GPU0–3 连续空闲60秒后，自动运行当前设备的 GPU kernel witness；各分支完整 sanity 仍是后续科学作业的必要条件。实际 pipeline 启动后才生成 [STATUS.md](campaign_seed0_after_gpu03/STATUS.md) 与 [pipeline_state.json](campaign_seed0_after_gpu03/pipeline_state.json)。等待或预检通过均不表示正式实验完成。

## 自动顺序

| 顺序 | 作业 | GPU 0 | GPU 1 | 完成要求 |
|---|---|---|---|---|
| 1 | S02 测量检查 | SGD sanity | GAM sanity | 完整恢复、原生/有效权重等价、逐样本梯度与FD、端点一致、历史精度reference |
| 2 | S03 邻域 | SGD anchor | GAM anchor | r=0/.5/1/2/5/10/20，16公共方向，分层256 train |
| 3 | S04 真实方向 | SGD anchor×2方向 | GAM anchor×2方向 | r=0/.25/.5/1/1.5/2 |
| 4 | S05 当前序列 | t0–t9复评 | t0–t9复评 | 当前FP32 train/test loss、accuracy、有效权重位移 |
| 5 | S05 t9路径 | SGD路径 | GAM路径 | 9节点；必要时17；求积与同子集界检查均通过 |
| 6 | S07 补训练 | SGD Task0→SGD+EWC | GAM Task0→SGD+EWC | Task1各50epoch，LR.02，λ2000，关闭后续shaping |
| 7 | A端点评估 | 新Task1 raw Fisher等 | 新Task1 raw Fisher等 | 同数据/坐标/head口径；a11采用当前原点复评 |
| 8 | CPU汇总 | 一次汇总 | — | 双方法全部完成后输出完整报告 |

每张卡同一时刻只跑一个本实验作业。阶段之间有屏障；失败会保留日志并停止后续阶段。不会因结果不支持假设而跳过报告。

S01旧JSON重分析已经完成。未默认排入seed1/2完整序列、dense K100半径、full-model或四格；正式三seed证据仍需这些后续工作，不能把单seed称为统计结论。

## 精度排查与口径

首次sanity因直接用FP32复现历史CE失败，原始输出保存于 [失败记录](failed_attempt_01_historical_fp32_gate/README.md)。受控实验仅将patch-embedding卷积的输入与权重舍入至TF32有效精度，四个历史端点CE最大误差1.38e-6，accuracy全相同；详见 [独立审查](investigation/INDEPENDENT_REPRODUCTION_REVIEW.md)。Pillow11.3/12.1在本任务上的对照值相同，因此没有将Pillow候选当作修复。

正式新B/C/A全部使用FP32，历史TF32 reference仅用于兼容性验证，结束后精确恢复；不把历史随机增强train曲线与新确定性视图混合。

## 路径与命令

- 配置：[diagnostic_config.json](diagnostic_config.json)
- 队列：[campaign_plan.json](campaign_plan.json)
- 冻结源码：[source_frozen.json](source_frozen.json)
- 训练协议：[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)
- 最终报告：[report_seed0/RESULTS.md](report_seed0/RESULTS.md)；运行未完时对应项为pending。

接续入口（已有接续进程时不要重复启动）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/wait_then_run.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

如发生中断，先检查接续与 pipeline 状态、存活进程及代码/配置hash，再选择恢复入口；不要用旧 GPU4/6 状态目录恢复当前配置。训练中断的task-boundary恢复另需核对训练runner的 `--resume-last`；队列不会静默覆盖已有训练目录。
