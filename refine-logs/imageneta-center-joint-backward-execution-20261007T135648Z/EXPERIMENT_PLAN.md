# ImageNet-A：中心修正和联合后向替换执行队列

用户已授权将四组新实验加入等待队列。此目录是独立v3执行版本；原第二探针严格队列和原始结果不修改。旧规划快照位于plan_snapshot，当前执行状态以本文件、suite.json和runtime/queue_state.json为准。

## 四组新训练与两项复用

固定ImageNet-A 10×20、seed2、LoRA rank10、rho=.05、r=1、batch128、50epochs/task；SGD首LR=.04/后续=.02，momentum/WD=0，cosine最低比例.1。native注意力、严格deterministic/warnfalse、FP32、TF32/AMP关闭。没有EWC/Fisher、权重融合、投影、HVP、SGV、replay或最终更新范数匹配。

| ID | 第一定位 | 更新U | 训练CE/步 | 正式训练CE |
|---|---|---|---:|---:|
| CPLUS | qF=g1+−g0 | g0+(g3F+−g3F−)/2 | 5 | 12750 |
| CMINUS | qF=g1+−g0 | g0−(g3F+−g3F−)/2 | 5 | 12750 |
| BF | qB=g0−g1− | g0+g3B+−g2B | 4 | 10200 |
| BB | qB=g0−g1− | g0+g2B−g3B− | 4 | 10200 |

theta1±=theta0±rho N(g0)，theta2^a=theta0+r N(q_a)，theta3^(a,±)=theta2^a±rho N(g2^a)。不除以rho；最终均恢复theta0后一次theta0−lr U。中心两侧属于每步训练，不能记成每任务偶发诊断。所有新组额外梯度诊断0；BB正侧局部读数0，未观测的s/e不能填成0。

FF/FB复用原严格执行队列的B0/N5 r1完整结果65.176/47.584、65.385/48.632；两项参考不计新训练。新旧FF/FB点、梯度、U、实际SGD及RNG由CPU回归门验证；F-family真实短测的初始共同g012/点还须与旧FF/FB首步指纹一致。不同训练轨迹后续梯度不要求相同。

## 队列接续与不抢占

9作业=4真实GPU短测+1 CPU配对门+4完整训练，另2参考结果。先等待原第二探针队列16正式均验收完成且controller状态completed；不改原调度，不插队，不抢占。之后使用实际空闲GPU2–6，每卡一项，最多5并行，GPU0/1始终排除。

采用共享/tmp/olora_gamfs_gpu_locks下的index/UUID双锁；锁前、锁后查询显存<500MiB、利用率<5%、无compute PID；每60秒检查。依赖必须successful completed，失败不是放行条件。任一短测、配对或正式验收失败，停止自动接续并保存失败证据。runtime/PAUSE可暂停后续启动，不中断活跃任务。

4短测各2tasks×1epoch×1batch：中心每项10训练CE/2SGD、BF/BB每项8CE/2SGD，总36CE/8SGD；无额外诊断梯度。配对门核验输入/初始化/RNG全局一致，F与B各自共同g012/点一致，跨family只检查共同g0。全部通过才启动4次10任务正式训练，合计45900训练CE/10200SGD。

## 环境与只读复核命令

复用真实主机cudahpc40.cvc.uab.es、PID namespace pid:[4026531836]，Python /home-local/liying/.conda/envs/Pilot_new_local/bin/python。环境版本和缓存沿用environment_spec.json；未安装/修改包。source_manifest.json固定190文件只读源；RUN_PINS.json固定本执行器、配置、CPU门和suite。

CPU科学/事务/队列/入口检查（不写训练结果，也不领取GPU）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/tests -p 'test_cj_*.py' -v
```

真实host只读dry-run（需在真实PID namespace；不分配GPU）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/suite.json --dry-run
```

预期status=valid，jobs9/smoke4/formal4/pairing1/reused2、allowed_gpus=[2,3,4,5,6]；upstream.ready可为false，这表示合法等待而非未入队。GPUkernel witness由每个真正获得空闲卡的worker执行；CPU通过不称GPU短测通过。

CPU门与独立文档复核通过且最终suite/pins冻结后，唯一启动命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/launch_queue.py
```

启动后核验LAUNCH.json和runtime/controller.json中的PID/starttime/hostname/namespace，读取runtime/queue_state.json。不要用sandbox不可见PID判定后台任务消失。

## 结果验收

2550SGD、10tasks/500epochs、动态CE预算、有限观测、禁用操作0、完整receipt/源码/配置哈希、原三角矩阵重算一致后才填summary.csv/results.json。指标任务等权AAA/FAA、LA/BWT/forgetting与样本口径分列；n=1、SD=N/A，短测分数不得填完整结果。中心s2/e2与实际BA+FC步幅从现成训练梯度记录，不追加对照上升读数。

运行身份、实时等待理由与已验收作业见runtime/queue_state.json；summary.csv将新结果与复用结果明确区分。
