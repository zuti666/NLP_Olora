# 自动接续诊断和绘图

前置：training_campaign全部6个Task0训练完成且完成收据通过。等待阶段仅CPU轮询，每30秒检查一次；诊断阶段仍仅物理GPU0/1、最大并行2、共享UUID锁、逐作业CUDA witness。

先运行所有6个Task0正梯度诊断，再运行所有6个Task1负梯度诊断，最后CPU绘图。这样同一模型的两个诊断不会并发写共用数据缓存。每个方向在新锚点仅计算一次，然后测0,.5,...,5共11个位置。新测量总计132个位置。复用旧EWCLoRA和旧FS-EWCLoRA同网格的132行用于绘图，总计264行，不能全部称为新结果。

准备后可逐字执行以下只读命令，不分配GPU：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/gam_radius1_direction_20260922/wait_and_run_diagnostics.py
```

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/gam_radius1_direction_20260922/publish.py --validate-only
```

训练未完成时publisher必须显示ready_awaiting_results、missing_jobs，不生成正式图或完成收据。实际一次性接续启动由launch_diagnostics.py记录进程身份和plan hash；查看diagnostic_campaign/queue_state.json，实际进入诊断后再查看pipeline_state.json和STATUS.md。

最终图：

- figures/task0_positive/Figure_task0_positive_training_radius_intervention.pdf及PNG。
- figures/task1_negative/Figure_task1_negative_training_radius_intervention.pdf及PNG。

每图一行三列：有符号旧任务训练CE增量、旧任务训练accuracy%、未额外归一化的旧任务empirical Fisher trace。四条曲线为旧EWCLoRA、旧FS(.05,.2)、新FS(.05,1)、新FS(1,1)；括号为训练的(probe rho,norm radius)，横轴为另一个定义明确的有效Q/V位移r。

最终入口report/radius_intervention_complete.json与report/RESULTS.md。完整完成后还要实际查看渲染图；自动输出的“待视觉审查”状态不得当作人工/独立图审查已经通过。
