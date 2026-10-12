# ImageNet-A中点与末端交互执行

状态以runtime/queue_state.json为准。本目录四项正式训练：sum12/base02/two_midpoint/sum13，固定seed2/rank10/r1/rho.05、10任务×50epochs，SGD LR.04/.02。原生3/3/3/4梯度，Fisher/EWC/权重融合/最终投影/匹配均关闭。

12项队列：4两任务短测→1同状态配对→1诊断短测→1真实模型原生/全读数回归→1旧快照完整24组合诊断→4正式训练。诊断不提交SGD；6组合FP64同坐标校准，其余18未校准。正式训练不因局部效果不好而被筛除。仅GPU2–6空闲卡，共享index/UUID租约，排除0/1，最多4项并行。

只读队列验证命令（须在cudahpc40宿主PID namespace执行）：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g12-midpoint-interaction-execution-20261008T140723Z/run_queue.py --dry-run
```

正式启动由已授权的root执行，不重复运行当前controller。

原始阶段计划：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。实时状态：[runtime/queue_state.json](runtime/queue_state.json)。正式结果：[summary.csv](summary.csv)。诊断：[outputs/diagnostic_full/diagnostics_summary.json](outputs/diagnostic_full/diagnostics_summary.json)。
