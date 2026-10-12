# 四方法实现：复现与短测

复用 `/home-local/liying/.conda/envs/Pilot_new_local`，不安装训练依赖。实际宿主机 `cudahpc40.cvc.uab.es`，PID namespace `pid:[4026531836]`。短临时目录为 `/tmp/olora_response_20261009T090226Z`。GPU仅2–6，0/1排除；共享index/UUID双锁、空闲才接入、最多并行2项。

## CPU验证

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/trainer/test_response_step_cpu.py
```

预期23项通过，覆盖四方法公式、实际FP32位移、base/final完整缓存、退化回退、事务恢复与旧HRF等价。

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/trainer/test_response_lifecycle_cpu.py
```

预期11项通过；使用真实冻结源的DataManager和evaluator验证1×200、10×20、分类头和指标。

## 单任务配置预检（不执行GPU训练）

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/trainer/response_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/configs/formal_single200_history_g2_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/outputs/formal_single200_history_g2_s2 --dry-run
```

预期 `response_method=history_g2`、`response_layout=single200`、`history_source=final`、正常3梯度、1SGD、1诊断前向、200类/1任务/50epochs、动态半径、GAM固定半径不启用。

## 短测队列预检（必须在实际宿主机执行）

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z/run_queue.py --dry-run
```

预期9jobs、8smokes、0formal、allowed_gpus=[2,3,4,5,6]、max_parallel=2。读取并核验190项冻结源和RUN_PINS；不会占GPU或创建训练输出。sandbox PID空间不能代替宿主机验收。

## 执行边界

本次根控制器仅启动 `suite.json` 的8个实现短测及1个配对检查。单任务短测为1×1epoch×2batches，CL短测为2×1epoch×2batches。每个worker首先在分配GPU执行真实kernel witness。

八份50-epoch正式配置位于configs、索引CONFIG_INDEX.json；本短测队列没有注册正式训练，不会自动接续。计划先完成四项single200，再新增三项cl10；cl10的history_fusion复用审计后的旧HRF结果。正式启动时需在空闲GPU租约内调用相应response_train命令、去掉--dry-run且不传短测预算参数。

当前执行只启动一次，查看runtime/LAUNCH.json、controller.json及queue_state.json防止重复。短测状态由IMPLEMENTATION_TESTS.md/json汇总；短测准确率不是论文结果。源码、旧队列及旧结果不修改。
