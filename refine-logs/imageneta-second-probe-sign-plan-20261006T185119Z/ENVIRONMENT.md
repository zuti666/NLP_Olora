# ImageNet-A 第二局部探针符号实验执行环境

沿用 cudahpc40.cvc.uab.es 上的 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，不安装或升级包。环境规格 `environment_spec.json` 的 canonical SHA256 为 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`；环境历史记录在项目 `.aris/compute/local.md`。

2026-10-06 本次真实主机查询只有七张 RTX 6000 Ada，物理编号0–6。允许使用全部非0/1的可见GPU：2、3、4、5、6，最多五项并行；启动时必须逐卡两次确认 memory.used<500MiB、utilization<5%、无compute进程，并持有共享index与UUID锁。初次查询全部忙碌，不能凭本文推断后续空闲。GPU0、1不参与本轮，不抢占其他任务。

代码冻结和CPU验收完成后，在真实主机PID命名空间逐字执行以下只读命令。它不认领GPU、不创建训练输出、不启动controller：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z/suite.json --dry-run
```

预期 `status=valid`，29个作业，14个正式训练、14个两任务短测、1个CPU配对验收，allowed_gpus=[2,3,4,5,6]，max_parallel=5。CPU_GATE.json、source_manifest.json、configs、suite.json及RUN_PINS.json必须已经生成。独立执行与代码复核记录在 gates/DOC_CHECK.json；不能用dry-run代替实际模型短测。

独立CPU复核使用以下命令，不改写已冻结的CPU_GATE.json；测试只在临时目录产生随后清理的输出：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z/tests -p test_second_probe_cpu.py -v
```

预期所有CPU测试显示OK，包括八格公式、共同前三梯度、旧B1两半径逐位等价、完整状态/随机数恢复及训练与隔离诊断计数。

独立文档验收记录与suite/RUN_PINS哈希一致后，启动器才允许执行。已获用户授权的正式调度命令为：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z/launch_queue.py
```

controller在后台等待空闲GPU。每个worker在持锁后先执行 seed2 FP32 CUDA前向/反向见证，再执行指定训练入口；每组短测2tasks×1epoch×1batch，实际2个SGD更新、8次训练CE梯度。所有14组短测和数据/初始化配对验收通过后才启动14项完整十任务训练；任一失败停止新增作业，不自动改变参数重跑。sandbox内NVML/CUDA/PID观察失败不能当真实硬件故障。

16个科学条件包含两个r切片，每切片六个新增方向、g0正向差分同协议基线和g1正向差分旧结果。B1旧结果仅在同状态数学/执行等价检查通过且原receipt/hash核验通过后复用，无重复正式训练。训练为seed2、rank10、rho=.05、r=.25/1、FP32 native attention、TF32/AMP关闭。每步四次普通CE梯度，一次恢复到原始参数后的自然SGD更新；各探针重放R0，最后仅推进至g0的R1。Fisher/EWC、权重融合、投影、HVP、SGV及最终范数匹配均关闭。

前g0/g1/g2构造不变，只有 second_probe_sign、second_difference_sign、update_base控制最终方向。数据和模型从已有冻结190文件source快照及本地离线缓存读取；运行使用suite记录的缓存和临时路径、线程数与随机环境。队列实时状态为 runtime/queue_state.json；summary.csv区分新增结果和带来源的历史复用，不以未完成任务的部分成绩填写正式结论。
