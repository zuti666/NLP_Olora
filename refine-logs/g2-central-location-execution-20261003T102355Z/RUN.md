# P/C/CD：仅提交g2的执行记录

日期：2026-10-03T12:41:27.661123+02:00。当前正在完成独立验收，尚未启动GPU训练。用户授权实施三组计划，GPU0/1继续排除。

## 实验与依赖

| 内容 | 数量 | 条件与接续 |
|---|---:|---|
| 两任务短测 | 3 | P/C/CD；2tasks×1epoch×1batch |
| CPU配对检查 | 1 | 全部短测通过后，检查原始初始化、新FC行、数据/增强身份 |
| 正式训练 | 3 | 全部短测与配对检查通过后，10tasks×50epochs |
| 共同状态离线诊断 | 1 | 全部正式完成；三轨迹Task0/4/9各两个固定批次，108普通CE梯度/零更新 |

全部seed2、rank10、内层ρ=.05、外层r=.25、FP32 math，最终仅g2自然SGD。四次训练CE/一次更新；每个任务入口另做两次完整四探针零更新重放，共8诊断CE。每条正式2550更新/10200训练CE/80诊断CE；每条短测2更新/8训练CE/16诊断CE。所有模块调用、恢复与任务矩阵由原始trace验证，未完成验证不写成completed。

## 准备命令与冻结

只在所有源文件和CPU验收完成后执行，不启动训练：

```bash
cd /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z
/home-local/liying/.conda/envs/Pilot_new_local/bin/python build_suite.py --freeze
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --dry-run
```

初始materialize已完成，不要再次默认运行build_suite.py覆盖配置。--freeze保存源码、队列、配置、CPU收据、源manifest和输入资产pins；冻结后不改训练源码。独立代理原样执行ENVIRONMENT.md所述dry-run，写gates/DOC_CHECK.json。该动态文档验收收据位于pins之外，避免递归哈希。

## 调度与状态

限制GPU2–6、最多3个GPU作业；共享原`/tmp/olora_gamfs_gpu_locks`编号+UUID双锁。每次分配前双次fresh NVML检查必须<500MiB且<10%利用率，再UUID绑定。GPU0/1不使用；忙卡等候，不抢占、重试或覆盖未知进程。每个GPU worker在训练前独立执行seed2 CUDA前向/反向witness。CPU配对作业不占GPU。

宿主机控制器由root通过subprocess.Popen(start_new_session=True)启动并保存实际hostname、bootID、PIDnamespace、PID及starttime。发生失败时停止新增启动，已运行作业完成后停止；不会重用非空输出目录。

实际启动命令（由root执行，本文仅用于复现）：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-execution-20261003T102355Z/run_queue.py
```

状态：runtime/queue_state.json；控制器日志：controller.log；作业日志：runtime/jobs/<job>/console.log；完成后汇总results.json与summary.csv。runtime/PAUSE只暂停新的启动；本轮没有暂停其他队列。

上轮直接差分回加的Local-CD仍为plan_only，jobs为空。这里只运行O_P/O_C/O_CD的g2定位，不包含D/R、残差更新、额外半径/rank扫描或HVP训练。
