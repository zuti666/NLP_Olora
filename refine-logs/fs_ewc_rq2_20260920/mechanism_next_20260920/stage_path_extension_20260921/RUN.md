# Historical stage-path extension: GPU 0 and 3

用户于 2026-09-21 授权使用物理 GPU 0、3 执行逐任务补测。当前状态以 [campaign/STATUS.md](campaign/STATUS.md) 为准。

- GPU 0: EWCLoRA (`sgd`), Task1 → Task8; GPU 3: FS-EWCLoRA (`gam`), Task1 → Task8. 两条队列独立推进。
- 数据：原 C 图的历史 ImageNet-A seed0；Task0 训练集固定分层 256 个样本（subset_seed=20260920）。不训练模型。
- 每个结果独立测量从固定 Task0 到对应 Taskt 的端点直线段，初始 9 节点与 5 节点比较，必要时细化至 17/33。不是逐步优化器轨迹积分。
- Task9 复用 `execution/results_seed0_recovery1/{sgd,gam}/path`，汇总时再次验证历史数据、坐标、旧 head 和 anchor 身份。
- 曲线横轴 T=已学任务数=代码 task+1；T=1 位移为零，归一化路径敏感度未定义。

复用 Pilot_new 环境，无依赖变更。GPU 队列仅在卡空闲且取得 UUID 共享锁后启动；每条线首个作业先做种子 0 FP32 CUDA forward/backward 检查。诊断中还检查端点 checkpoint 恢复、参数精确还原、积分收敛与同一 Fisher 样本集上的损失界。失败会阻止该队列继续，不自动改变样本或测量口径。

文档可复现的 CPU dry run（不运行 GPU 作业）：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/stage_path_extension_20260921/campaign/plan.json
```

执行在上述命令后添加 `--execute`。已有运行必须先检查 PID/状态；只在原进程和子进程全部退出且代码/配置未改变时使用 `--execute --resume`。共享文件锁会阻止重复启动。GPU 执行需要当前工作环境批准的沙箱外 CUDA 访问路径。

原始测量：`results/{sgd,gam}/path_tXX/`；逐节点 `states.jsonl`、身份 `manifest.json`、数值检查与指标 `summary.json`。配置：[configs/historical_seed0.json](configs/historical_seed0.json)；队列：[campaign/plan.json](campaign/plan.json)。

16 个新作业结束后用 `report_stage_paths.py` 合并 2 个旧 Task9 结果并生成逐阶段 C 图。汇总器未验证全部 18 个结果之前，不产出完整趋势图。

自动收尾：`finalize_when_complete.py` 在独立 CPU 后台进程中等待 16 个作业完成，再执行已固定代码哈希的汇总脚本。状态在 `campaign/finalizer_state.json`，日志在 `campaign/postprocess.log`。输出表格在 `report/`，新图在本目录的 `Figure_C_sequence_stage_paths.pdf` 和 `.png`。

完成记录：16/16 GPU 作业完成，18/18 路径通过汇总校验；新图和逐节点数据已生成。绘图首次导出因自动刻度超出画布而停止；只调整可见刻度并重新导出后通过，原测量未修改，首次日志和收尾 manifest 保留在 campaign/。解读见 [report/SUMMARY_CN.md](report/SUMMARY_CN.md)。
