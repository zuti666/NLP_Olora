# FS-EWC RQ2 seed0 自动执行

**2026-09-20：已按用户要求停止后台队列。进程组已确认退出，正式批量诊断和补充训练均未启动；不再自动继续。**

停止凭据：[STOPPED_BY_USER.json](campaign_seed0/STOPPED_BY_USER.json)。

更新：2026-09-20T10:06:34.364374+00:00。用户已授权 GPU 4、6，先执行前三个工作包，再补实验。

使用 `run-experiment` skill；共享环境未修改，生产解释器仍是 `Pilot_new`。Pillow 11.3 仅安装到本实验候选依赖目录用于排查，正式流程未采用。

## 当前状态

后台队列已于 2026-09-20T10:12:01.618643+00:00 启动，PID 3615230；当时确认存活；现已停止。停止前两条sanity处于waiting_for_gpu，其余13个作业未启动。

以 [自动状态](campaign_seed0/STATUS.md) 和 [状态JSON](campaign_seed0/pipeline_state.json) 为准。两张卡被新 VLLM 进程占用时，队列只等待，不运行并发实验、不终止其它作业。

## 自动顺序

| 顺序 | 作业 | GPU 4 | GPU 6 | 完成要求 |
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

队列命令（已有队列时不要重复启动）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/pipeline.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_plan.json --execute
```

中断恢复使用同一命令加 `--resume`，须先检查原队列不在运行，代码/配置hash一致。训练中断的task-boundary恢复另需核对训练runner的 `--resume-last`；队列不会静默覆盖已有训练目录。
