# 恢复队列：第二局部探针符号实验

客户端日期2026-10-07，时区Europe/Paris。沿用真实主机cudahpc40.cvc.uab.es及 /home-local/liying/.conda/envs/Pilot_new_local/bin/python；environment_spec.json canonical SHA256=4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8，不重建环境或改变训练参数。

原队列在首GPU短测、首次SGD前因诊断同侧重算的torch.equal断言失败，0/14正式开始。原目录 /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z 完整保留。此次仅更改隔离诊断：复用训练selected g3，只额外测opposite1次普通CE，恢复pending U/模型/R1；训练compute、点构造及apply_update源码不变，仍4训练CE/1SGD。每完整正式训练2550更新、10200训练CE、10隔离诊断CE。不能将引用复用当独立重算一致性证据。

以下只读命令须在真实主机PID命名空间执行；dry-run不认领GPU、不生成训练输出。代码和CPU gates固定后，由独立最小上下文agent逐字验收：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z/suite.json --dry-run
```

预期valid、29jobs=14smoke+1CPU pairing+14formal、2reused、allowed_gpus=[2,3,4,5,6]、max_parallel=5。GPU0/1继续排除。

独立CPU复核不重写CPU_GATE，测试仅临时目录输出：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z/tests -p test_second_probe_cpu.py -v
```

预期13项全部OK，含旧/新训练函数源码不变、八方向、完整模型/RNG恢复、旧B1逐位等价、同侧训练引用/反侧1CE来源及CPU初始probe快照。QUEUE_HELPERS_CPU.json另记录14/14 helper验收。

独立DOC_CHECK与suite/RUN_PINS哈希一致后才允许启动：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z/launch_queue.py
```

controller共享index/UUID双锁、双次idle检查(memory<500MiB/util<5%/no compute PID)，60秒轮询、最多5并行，不抢占其他任务。每worker先seed2 FP32 CUDA witness，TF32/AMP关闭，随后真实模型2tasks/1epoch/1batch smoke。14短测与实际输入/初始化/同r前三probe精确配对gate全部通过后才自动启动14正式。任一失败停止接续。CPU及kernel通过不代替真实模型短测。

本次真实主机只有GPU0–6；首次恢复查询2/3/4空闲、5/6忙碌，设备可用性以后以实时telemetry为准，不使用历史10卡编号。每smoke Task0首步保存g012和point012 CPU张量快照及SHA；任何hash差异先分析真实数值证据，不静默放松检查。两旧B1结果只在CPU等价、原validator、receipt/hash检查通过后复用。

实时状态runtime/queue_state.json，结果summary.csv，训练/诊断分别计数。没有Fisher/EWC、权重融合、投影、HVP、SGV及最终范数匹配；seed2/rank10/rho=.05/r=.25和1、十任务每任务50epochs、首LR.04/后续.02、batch128、自然SGD全部保持。
