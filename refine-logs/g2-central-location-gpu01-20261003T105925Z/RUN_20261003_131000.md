# P/C/CD g2：GPU0/1独立执行记录

日期：2026-10-03。用户最新授权只使用物理GPU0/1、并行上限2，覆盖此前排除0/1的授权。新目录为 `/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z`，父执行目录为 `../g2-central-location-execution-20261003T102355Z`。准备阶段尚未冻结或启动新训练。

## 实验与依赖

| 内容 | 数量 | 条件与接续 |
|---|---:|---|
| 两任务短测 | 3 | P/C/CD；2tasks×1epoch×1batch；新目录重新执行 |
| CPU配对检查 | 1 | 全部新短测通过后，检查原始初始化、新FC行、数据/增强身份 |
| 正式训练 | 3 | 全部新短测与配对检查通过后，10tasks×50epochs |
| 共同状态离线诊断 | 1 | 全部正式完成；三轨迹Task0/4/9各两个固定批次，108普通CE梯度/零更新 |

全部seed2、rank10、内层ρ=.05、外层r=.25、FP32 math，最终仅g2自然SGD。四次训练CE/一次更新；每个任务入口另做两次完整四探针零更新重放，共8诊断CE。每条正式2550更新/10200训练CE/80诊断CE；每条短测2更新/8训练CE/16诊断CE。8个job ID、科学条件、依赖DAG和命令参数保持不变，仅重定向独立路径。完成状态仍由原始trace、恢复与任务矩阵验证。

## 准备命令与冻结

`trainer/` 与旧执行目录完全同字节，16+9共25项CPU通过证据原样复用，来源与哈希见 `GPU_ROUTING_CHANGE.json`。新六个入口dry-run、GPU路由差异独立审核完成后，root执行以下CPU准备命令：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/build_suite.py --freeze
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/run_queue.py --dry-run
```

新suite已从旧执行目录克隆完成。默认运行 `build_suite.py` 不执行materialize；显式 `--materialize` 也拒绝覆盖已有suite。历史ordinary原型仅用于构造器，`parent_run` 固定指向旧g2执行目录。`--freeze` 保持原门禁，保存源码、队列、配置、CPU收据、源manifest和输入资产pins；冻结后不改训练源码。独立代理须按 `ENVIRONMENT.md` 的绝对命令原样执行，写新 `gates/DOC_CHECK.json`。该动态文档收据位于pins之外，避免递归哈希。历史独立CPU审查不是新GPU路由审核，新审核见 `gates/GPU01_DELTA_REVIEW.json`。

## 调度与状态

只允许物理GPU0/1、最多2个GPU作业；CPU配对不占GPU。共享 `/tmp/olora_gamfs_gpu_locks` 编号+UUID双锁。分配前双次fresh NVML检查须<500MiB且<10%利用率，再UUID绑定。GPU0当前空闲、GPU1约12GB忙是root提供的准备阶段快照；运行时仅使用实时空闲卡。每个GPU worker训练前独立执行seed2 CUDA前向/反向witness。忙卡等候，不抢占、重试或覆盖未知进程。

旧控制器已在paused且没有活跃worker时停止；旧目录、已完成两组短测、原pins及证据保留。新目录没有继承旧runtime/outputs/LAUNCH/RUN_PINS/results/summary/DOC_CHECK，三组短测重新开始。宿主机控制器由root以 `subprocess.Popen(start_new_session=True)` 启动，保存真实hostname、bootID、PIDnamespace、PID及starttime。任一失败停止新增启动，已运行作业完成后停止；不重用非空输出目录。

实际启动命令由root在全部门禁通过后执行：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/run_queue.py
```

状态为 `runtime/queue_state.json`；控制器日志为 `controller.log`；作业日志为 `runtime/jobs/<job>/console.log`；完成后汇总 `results.json` 与 `summary.csv`。`runtime/PAUSE` 仅暂停新的启动。本轮不操作其他队列。

上轮直接差分回加Local-CD仍为plan_only，jobs为空。本队列只运行O_P/O_C/O_CD的g2定位及既定离线诊断。
