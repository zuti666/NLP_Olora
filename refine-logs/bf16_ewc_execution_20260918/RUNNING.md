# BF16 五卡实验运行记录

已使用独立源码快照启动；两阶段 smoke 验证通过，正式实验正在运行。

- 当前运行：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z`
- Supervisor PID：`64343`；GPU：0,1,2,3,4（按 UUID 绑定）。
- 队列：Order2 → Order3 → Order1；每个 Order 从原始基座开始，四任务串行。
- 固定协议：BF16 AMP，BS1/GA8，有效 batch40，λ1/γ1，OLoRA0，seed1993。
- 两阶段 smoke：rank8→16，Fisher1→2，五个进程参数一致，历史 adapter 未改变，第二任务 EWC 惩罚与梯度均为正。
- 第一次尝试在训练前因数据缓存竞争失败，失败日志保留；已修复为主进程先构建、其他进程复用。
- 旧 Order1 缺少执行源码/模型哈希，因此重新运行 Order1；新结果不会与旧 O1 或 FP32 数字混用。

实时状态以运行目录 `state.json` 为准；每阶段日志位于 `logs/`。每完成一个完整 Order 自动更新 `final_results.json`、CSV 和 LaTeX 表；完整三个 Order 完成前 Avg 留空。任何训练或验证错误会停止队列并记录 failed。

耗时尚待首个完整阶段校准；未将正在运行写成已完成。
