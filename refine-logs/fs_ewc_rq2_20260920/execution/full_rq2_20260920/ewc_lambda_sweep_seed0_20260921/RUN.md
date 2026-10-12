# Raw Fisher EWC sweep：运行与核验

最终授权范围和网格见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。seed0、SGD→SGD、复用Task0完整模型并实际重算raw Fisher，关闭单任务和累计trace归一化；14个λ均重新训练Task1。完整计划18作业。

环境沿用 Pilot_new@e94a148f，无依赖修改。解释器 /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python，Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。环境spec在上两级environment_spec.json，记录在项目.aris/compute/local.md。FP32、两种TF32均关闭；每个GPU作业在已取得空闲卡后执行seeded CUDA前向/反向witness。

## 只读文档核验命令

在项目根目录原样执行，仅读取计划和指纹，不运行训练、不调用GPU、不创建campaign状态：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/plan.json
```

预期dry_run=true、stages=18、jobs=18、max_parallel=6。依赖为prepare_raw_fisher→smoke_sgd_lambda60→validate_smoke→14个lambda_*→sweep_report。检查plan的all_gpu_start_gate.required=true，gpus=[0,1,2,3,4,5,6]，poll_seconds=900。

## 后台入口

由助手在宿主机启动一次，将stdout/stderr保存到campaign/controller.log，并把PID、启动ticks、boot_id和计划SHA记到LAUNCH.json。不要重复执行已有活动队列。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_sweep_seed0_20260921/plan.json --execute
```

所有GPU0–6必须同时空闲才释放第一次准入；每15分钟检查，繁忙期间只等待。准入后各作业仍需要空闲卡和共享UUID锁，最多6并发；不会每波等待自己的作业占用卡清空。禁止在控制器运行时编辑冻结源文件和计划。恢复前用宿主机PID身份及共享锁核验无控制器/子进程存活，再以相同冻结入口增加--resume；sandbox /proc不可用于断定宿主机进程死亡。

## 评估和收据

endpoint_eval=false只是关闭额外endpoint诊断模块，正式eval_task的CIL矩阵仍开启；mechanism_eval、mechanism_grad_stats、旧任务loss/accuracy/位移、任务末Fisher、checkpoint完整重载和RNG配对均验收。短测与正式结果分目录，短测不能产出COMPLETE.json。raw Fisher准备失败或短测验收失败时不开始正式扫描。

每个正式点TRIAL_RESULT.json的status=observed可能是outcome=complete或diverged，必须区分。只有完整50轮、评估/检查点均通过的点有正式准确率。数值发散是边界观测；其他错误使队列失败，不改变科学参数自动重跑。

状态以 [campaign/STATUS.md](campaign/STATUS.md)、[campaign/pipeline_state.json](campaign/pipeline_state.json) 为准。CPU部分汇总可运行report.py --allow-partial；只读已生成收据，未完成不会产生完整报告收据。自动最终汇总必须14点均已观测且λ=0完整完成。
