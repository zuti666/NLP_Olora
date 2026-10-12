# 三层证据链：复现与运行

执行目录 `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z`。

沿用已验证环境 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，torch2.4.1+cu121/timm1.0.9，原生FP32、TF32关闭，不安装包。环境规范hash `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`；`environment_spec.json`是原样复用的历史环境/模型规范，具体数据预算与GPU政策由本轮`EXPERIMENT_PLAN.md`、配置和`suite.json`覆盖。

冻结源 `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`，190文件；manifest SHA `e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2`。冻结执行文件清单在`RUN_PINS.json`。实际host `cudahpc40.cvc.uab.es`、PID namespace `pid:[4026531836]`。host的CUDA/进程核验必须在真实host执行，sandbox的NVML失败不作硬件故障证据。

仅物理GPU4/5/6，max_parallel3；共享`/tmp/olora_gamfs_gpu_locks`的index+UUID锁。队列只使用显存<500MiB、利用率<5%、无computePID的GPU，不抢占其他任务。每个worker用绑定UUID的CUDA_VISIBLE_DEVICES，先执行seed2 FP32 forward/backward witness。

## 文档核验用CPU命令（不会启动实验）

在上述执行目录，原样执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/response_chain_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config configs/formal_R_s0.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-response-state-chain-gpu456-20261011T195426Z/outputs/formal_R_s0 --dry-run
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python diagnostics/offline_diagnostics.py --help
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python collect_results.py
```

第一条应打印seed0、branchR、rolebranch的有效配置，且不创建正式输出目录。第二条应展示`--run-root`、可重复`--checkpoint`、`--seed`、`--output-dir`、`--smoke`。第三条只生成空/已验收结果表，不创建训练输出、不使用GPU。

host上的CPU队列预检：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json --dry-run
```

应登记16jobs、6formal_new_training、2training_smokes加pairing及diagnostic_smoke共4smokejobs、3prepare、3offline；onlyGPUs[4,5,6]、max3。dryrun不执行witness、不占GPU、不创建任何job输出。历史父模型3个已完成50轮，复用不是新增训练。

## 正式启动（由主代理执行一次；文档核验代理不执行）

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json
```

控制器需以detached进程启动；CPU、独立代码审查和fresh-doc门控必须全部通过才允许进入队列。先3seed各准备一次expanded40共同状态，再seed2两个2epoch×2batch短测（e1SAM/S_F→e2SGD，cosine仍Tmax50）、配对、真实HVP短测，最后6条50epoch分支和3组离线诊断。失败依赖不放行；没有自动重试或改变算法参数。

正式分支：e1–25S用SAM(.05)，R用S_F g2(θ−ηg0，外.25)，e26–50共同SGD，continuouslycosine(.02→.002)，不重置。所有Task1为906训练/91dev/官方旧307与新254测试；无EWC/Fisher/merge/projection/replay/history/finalmatching，HVP仅离线。

检查点`outputs/formal_S_s2/checkpoints/epoch_001.pt`等，每个seed离线读取S/R的e1/e10/e25和2固定训练batch128。数学注意力/microbatchHVP须与原生一阶结果一致且双侧差分2相邻距离验证后才报告H²解释。候选匹配只离线使用e10/e25，同点SAM自然步幅作为目标；采用保存checkpoint实际next_lr（同时报告epoch_lr_used）。

## 进度与产物

`runtime/queue_state.json`是实际状态，`runtime/controller.json`记录PID/starttime，`runtime/jobs/JOB/console.log`和`exit.json`保存执行结果。检查真实host PID身份和NVML，不能仅凭running标志认定正在训练。

完成须exit0且`validate_job.py`接受完整训练/诊断工件；每个seed正式S/R完整400个增强batch序列必须配对；三个seed是独立重复单位。只验收完成的分支进入`FORMAL_RESULTS.csv`、`RESULTS.md`；`PAIRED_SUMMARY.json`为R−S配对均值±样本SD，`TAIL_CHANGES.csv`为e50−e25，`FIT_MATCHED_COMPARISONS.csv`按预登记cleanCE分箱首进入比较，`LORA_FACTOR_ALL_LAYERS.csv`不挑层。离线诊断在outputs/offline_s2等，保存有效率/退化/失败边界。

每轮LoRA/FC及少量完整状态保存于独占`/data/115-2/users/liying/cache/tmp/imageneta-response-state-chain-gpu456-20261011T195426Z/outputs`，本目录outputs为其符号链接。TMPDIR使用短`/tmp/olora_chain_20261011T195426Z`，避免multiprocessing UNIX socket路径过长。

`touch runtime/PAUSE`只暂停后续接续，正在运行的任务继续；移除PAUSE恢复。不得改sealed配置/代码；修复须保存原失败证据、停止本控制器并排空自己的worker后，写版本化修复及重新门控。
