# 启动恢复：attempt2

首次启动的 CUDA 数值自检通过，但自检退出后 NVML 短暂利用率使两条准备作业提前拒绝；未开始 Task1 训练。原始日志及失败状态保留在 campaign/。SGD0复用及GAM0原生raw Fisher准备已完成，后续按原完整校验逻辑复用。

新增 preflight_with_cleanup.py：运行原 witness，成功退出后在共享 UUID 锁仍持有时等待3秒，再进入未更改的 Fisher 准入检查。所有模型/优化器/数据/评估/报告代码、λ、900秒调度间隔不变。

活动入口改为 plan_attempt2.json，状态目录 campaign_attempt2，启动记录 LAUNCH_attempt2.json，日志 controller_attempt2.log。原 RUN.md 的科学协议仍适用。

CPU dry-run：
```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/plan_attempt2.json
```
正式命令在上述命令尾添加 --execute，由独立后台控制器启动；已有控制器时不要重复执行。
