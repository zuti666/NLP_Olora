# r=1 四分支执行

本轮新跑12条外点相关完整十任务训练，复用4条更新与外半径无关的历史基准。先16个r1短测，全部验收后自动放行正式训练。物理GPU只允许2、3；GPU3优先、GPU2空闲后接收，max_parallel=2；不停止其他任务。

独立文档验证者读取 run-experiment skill、本 ENVIRONMENT.md 与本文件，再逐字执行下面唯一调用（真实host/GPU访问、require_escalated；不启动训练）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/documented_check.py --gpu 3
```

预期dry-run给出28 jobs、12 formal、16 smokes、allowed_gpus [3,2]、max_parallel2；后续seed2 CUDA sentinel有限。所有锁释放后才由root启动真正队列。

正式队列命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/run_queue.py
```

由root使用start_new_session后台启动，日志 controller.log，启动凭据 LAUNCH.json。现有controller存活时不要重复启动。运行状态 runtime/queue_state.json；每项 runtime/jobs/<id>/console.log、worker身份与exit/completed记录。临时暂停新启动可创建 runtime/PAUSE；当前GPU训练继续完成。失败则停止后续放行，不覆盖失败记录。

结果表 RESULTS.md、summary.csv、results.json 中未完成正式成绩留空。历史基准明确标记validated_reuse，仍保留旧r=.25trace来源。CPU/所有r1smokes不作为完整十任务成绩。报告只包含Stage1普通CE训练；不接续原HVP/机制诊断队列。
