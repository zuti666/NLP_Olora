# GPU1优先的空闲GPU接续

GPU3、4的先前精确文档调用均因卡已被其他作业占用而未执行CUDA witness；原失败记录保持不变。GPU3–6当前均忙；本研究此前GPU1 Mean2已完成验收并释放卡。沿用用户明确授权GPU1十任务实验及使用所有空闲GPU，本资源尝试优先GPU1，其余GPU4/5/6/3/2实际空闲后接续，最大并行3。GPU0不进入本轮池，不停止其他作业。

父ENVIRONMENT.md的软件/spec/source190/CPU与历史基准审计保持不变；本资源声明覆盖其GPU池一节。外部manifest固定资源池[1,4,5,6,3,2]及max3，wrapper仅在内存替换原suite的allowed_gpus/max_parallel，原56冻结文件、原配置/真实worker命令、numerics/validator均不修改。先16个GPU短测，全部验收后才放行12条正式训练。

独立验证者读取run-experiment skill、父ENVIRONMENT.md与本文件，逐字执行下列唯一调用（require_escalated真实host/GPU/proc访问，仅dry-run+canonicalseed2 tinyCUDA forward/backward，0optimizer更新）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T074000Z/resource_queue.py check --gpu 1
```

预期28jobs、12formal、16smokes、allowed_gpus=[1,4,5,6,3,2]、max_parallel3，canonicalCUDA sentinel有限。成功报告DOCUMENTED_WITNESS.json绑定manifest、wrapper、原RUN_PINS及实际UUID；释放共享index/UUID双锁后root后台启动：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T074000Z/resource_queue.py controller
```

唯一controller持父runtime/controller.lock；不要启动旧文档入口。LAUNCH.json记录实际控制器与资源身份，runtime/queue_state.json记录实际各job GPU、UUID与PID，RESULTS.md自动更新完整结果。此前两个busy资源尝试不会自动启动，只保留失败的文档检查；没有训练进程需要停止。
