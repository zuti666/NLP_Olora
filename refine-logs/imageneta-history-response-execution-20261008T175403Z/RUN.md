# 可复现调用与环境

临时路径固定为`/tmp/olora_hrf_20261008T175403Z`，避免multiprocessing Unix socket路径上限。已复用Pilot_new_local；不安装或更换训练依赖。实际GPU宿主机`cudahpc40.cvc.uab.es`，PID namespace `pid:[4026531836]`。仅GPU2–6，GPU0/1排除；共享GPU双锁，未空闲不启动。本文命令使用固定绝对路径；必须在实际宿主机执行GPU及队列检查，sandbox不能证明GPU/PID状态。

## CPU数值与代码检查

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z/trainer/test_history_step_cpu.py
```

## 只读队列预检（不会占GPU或创建训练输出）

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z/run_queue.py --dry-run
```

预期报告5 jobs、2 smokes、2 formal、allowed_gpus [2,3,4,5,6]、max_parallel2；检查源190项和不可变文件哈希。`--dry-run`不会启动训练。

## 单独配置预检

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z/trainer/history_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z/configs/formal_history_fused_k10_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z/outputs/formal_history_fused_k10_s2 --dry-run
```

预期direction history_fused、history_weight .5、2 normal gradients/1SGD/1diagnostic、gam_radii_active false、10tasks/50epochs。无历史版本只替换配置和输出路径为formal_future_only_k10_s2。

## 启动（当前会话仅由根控制器调用一次）

执行`run_queue.py`且没有`--dry-run`会占用空闲GPU并运行独立门控队列。不要再次启动同一run；先查看runtime/controller.json与queue_state.json。每个被分配的worker会先执行`trainer/cuda_witness.py --output runtime/jobs/<job>/cuda_witness.json`，实际kernel必须通过；两真实短测与配对通过才开始正式训练。

## 状态与产物

实时状态：runtime/queue_state.json；各作业console.log、exit.json和completed.json位于runtime/jobs。逐step trace、原矩阵、缓存checkpoint位于outputs；完整终点统计自动写入RESULTS.md、summary.csv、results.json。CPU/短测通过不代表十任务准确率已经完成。

算法与伪代码：docs/ALGORITHM.md；流程图在figures。模型和数据初始化严格复用冻结源，不修改其他队列或已有结果。
