# 第二局部探针：确定性执行与队列

日期2026-10-07。真实主机cudahpc40.cvc.uab.es，Python /home-local/liying/.conda/envs/Pilot_new_local/bin/python，Python3.11.15/torch2.4.1+cu121/torchvision0.19.1+cu121/timm1.0.9；复用环境、不安装包。environment_spec.json canonical SHA256=4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8，190源文件冻结版本不变。

原首短测诊断重复同侧torch.equal失败发生在首SGD前，0正式启动。native恢复14/14短测通过，但严格g012/phi012配对FAIL，原始失败+42对数值审计保留。该数值预算没有放宽；本版启用torch.use_deterministic_algorithms(True,warn_only=False)并在每批/任务/结尾检查actualflag。CUBLAS_WORKSPACE_CONFIG=:4096:8原suite已有、不将原因归为缺变量；TF32/AMP关闭。记录deterministic_enabled=true/deterministic_warn_only=false/cublas_workspace_config。

严格执行两卡预检独立验收PASS：GPU2/3，N5/N6各2tasks/2SGD/8训练CE/2诊断CE；g012/phi012/派生q与方向均真实位等，输入/初始化/R0R1相同。这不代替本版16真实GPU短测，也不保证任何平台都位等。执行设置改变浮点轨迹，因此原B1两半径必须补跑，旧结果仅历史参考。数学四梯度/位置/U/SGD保持，direction_step与native恢复版字节相同。

最终输入冻结后，由独立最小上下文agent只读执行下述命令。host dry-run需要真实PID namespace，不认领GPU或生成训练结果：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z/suite.json --dry-run
```

预期valid，33jobs=16smoke+1pairing+16formal，0reused，GPU=[2,3,4,5,6]，max_parallel5。GPU0/1仍排除。

只读CPU验收（不重写CPU_GATE）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z/tests -p test_second_probe_cpu.py -v
```

预期16项全部OK，涵盖八格/共同g012/旧B1数学等价/完整状态恢复/4CE与1SGD/诊断来源及1CE/快照/新flag配置拒绝与严格运行时见证。CPU_GATE与TRAINER_HASHES在生成时冻结；helper另验收33队列、签名/错flag/真实快照/严格位等失败和完整结果排除partial。

DOC_CHECK的suite/RUN_PINS SHA一致且status=passed后，唯一授权启动命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z/launch_queue.py
```

先16真实短测和严格输入/初始状态/同r g012/phi012配对门，再16正式。每smoke2SGD/8训练CE/2隔离诊断CE；每formal10tasks/500epochs/2550SGD/10200训练CE/10隔离诊断CE。诊断selected g3复用，仅新增反侧1CE/task并完整恢复pending U/模型/R1，不能将引用复用当作独立重复测量。

共享index/UUID双锁、锁后二次空闲检查(memory<500MiB/util<5%/no compute PID)，每60秒检测空闲GPU2–6、max_parallel5、无抢占。真实设备0–6均RTX6000Ada；GPU5/6当前其他任务占用，使用情况以实时telemetry为准。所有失败停止接续；GPU0/1继续排除。

seed2/rank10/rho=.05/r=1和.25、ImageNet-A10×20固定类顺序、50epochs/task、batch128、首LR.04/后续.02、cosine、自然SGD保持不变。Fisher/EWC/实际权重融合/投影/HVP/SGV/replay/最终梯度范数匹配执行计数必须0。实时runtime/queue_state.json，results.json/summary.csv只收纳完整验收结果。没有新完整FAA/AAA时不得用短测分数填正式表。

[PyTorch2.4确定性设置说明](https://docs.pytorch.org/docs/2.4/generated/torch.use_deterministic_algorithms.html)；实际正确性证据是CPU/GPU gates和完整日志，文档不能替代验收。
