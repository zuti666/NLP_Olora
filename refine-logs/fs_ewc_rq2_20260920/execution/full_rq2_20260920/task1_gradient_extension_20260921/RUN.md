# 固定 Task1 负梯度方向：运行说明

最终方向为`u1 = -grad(mean Task1 CE) / norm(grad)`，方向只计算一次，所有半径共用。Task1使用997张训练图、新20类初始分类头；该头从正式初始化记录精确重建，不使用训练完成后的Task1参数。Task0的损失、准确率和raw Fisher继续使用原来的旧20类头。

任务：3 seeds × 2 methods × 93 radii，共6个GPU作业及1个CPU发布作业。物理GPU1/4/6，每卡一组，最多3组并行；共享UUID锁和显存/利用率/compute进程空闲检查。此前随机方向与Task0正梯度两批实验已完成，不改动其文件。

环境：温复用`Pilot_new@e94a148f`，不重建依赖。每次获得GPU锁后运行已验证的FP32 CUDA前向/反向检查，关闭TF32。每组在真实模型上验证Task1小步长下降，之后才开始93点扫描。Task0损失不设方向性门槛。

在任意目录运行以下命令仅做只读检查，不会占用GPU或启动实验：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/task1_gradient_extension_20260921/queue_task1_gradient.py
```

预期`status=passed`、`dry_run=true`、`gpu_jobs=6`、`total_jobs=7`、`gradient_task=1`、`evaluation_task=0`、`direction_sign=-1`、`states_per_job=93`。

队列通过本目录`launch.py`在宿主机后台独立启动，已有`campaign/launcher.json`则拒绝重复运行。进度见[campaign/STATUS.md](campaign/STATUS.md)。六组通过验证后自动绘制0–50与0–2两张一行三列图：旧任务训练损失变化、训练准确率、raw Fisher；保留有符号损失和全部实测值。最终收据为`report/task1_gradient_complete.json`。

预计三卡两批约15–25分钟，实际耗时以状态和日志为准。GPU进程身份必须在宿主机查询；沙箱看不到PID不能说明任务已退出。
