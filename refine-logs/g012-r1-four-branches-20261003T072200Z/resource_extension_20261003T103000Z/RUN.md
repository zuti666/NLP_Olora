# 空闲GPU接续：并行上限6

用户要求查看当前空闲GPU并继续实验。2026-10-03 GPU1/2/6已有本研究三个正式训练，GPU3/4/5空闲；此前max_parallel=3阻止自动接续。本次只扩展并行上限为6，保留资源池[1,4,5,6,3,2]，GPU0不加入。训练参数、28项job集合、16项已通过短测、科学源文件、12项正式训练与既有结果全部保持原样。

父ENVIRONMENT.md的软件/spec/source190/CPU与历史基准审计保持不变。本外部manifest固定资源池和max_parallel=6，wrapper仅在内存替换原suite的allowed_gpus/max_parallel，原56冻结文件和真实worker命令不修改。只有运行控制器交接，原训练worker保持独立session和GPU index/UUID双锁，不终止训练。

独立验证者读取run-experiment skill、父ENVIRONMENT.md与本文件，逐字执行以下唯一检查命令（真实host/GPU/proc访问；dry-run加canonical seed2 tiny CUDA forward/backward，0 optimizer更新）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T103000Z/resource_queue.py check --gpu 4
```

预期28 jobs、12 formal、16 smokes、allowed_gpus=[1,4,5,6,3,2]、max_parallel=6，canonical CUDA sentinel有限。成功DOCUMENTED_WITNESS.json绑定manifest、wrapper、原RUN_PINS及实际UUID。只在资源门禁通过和原控制器身份再次核实后，root仅向原控制器PID发送SIGTERM，确认controller.lock释放，再后台启动下列入口，接管原三个worker并使用真正空闲GPU接续：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T103000Z/resource_queue.py controller
```

新HANDOFF.json记录原/新controller、完整worker身份和资源身份；runtime/queue_state.json继续记录实际GPU、UUID和PID。任何资源身份或科学文件门禁失败即保留原训练并停止新启动。原LAUNCH及资源尝试记录保留。
