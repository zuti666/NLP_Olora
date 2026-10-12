# 四方向seed0/1补充实验执行说明

日期：2026-10-12。运行目录 `/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-four-seed01-20261012T104100Z`。只允许物理GPU4/5/6，max_parallel=3。用户在准备期间更改GPU范围；未发生旧GPU启动，证据见gates/GPU_SCOPE_CHANGE.json。

科学设置、公式和完整预算见EXPERIMENT_PLAN.md。8新正式+8短测+1按seed配对作业，共17jobs；父seed2四项只读复用。

配置已物化，不能重复运行materialize。冻结前需要trainer的数学/seed CPU门、16个真实入口dry-run、队列CPU门、父复用验证及独立review全部通过。root随后执行：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-four-seed01-20261012T104100Z/build_suite.py --freeze
```

独立代理按ENVIRONMENT.md原样执行CPU队列dry-run，另写gates/DOC_CHECK.json，不能把准备门当GPU运行通过。root使用以下入口作为后台控制器，stdout/stderr写runtime/controller.log：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-four-seed01-20261012T104100Z/run_queue.py
```

每次启动前后检查空闲卡与共享双锁，worker/trainer继承锁；只有4/5/6可入队。每个worker执行自身seed的CUDA witness。8短测全部通过后执行一次CPU配对：同seed四方向共同初始读数、数据增强和两任务FC身份配对；不同seed分开。配对通过后8正式自动接续；任何失败停止新增启动，不自动重试、降低精度或改超参。

runtime/PAUSE只阻止后续新任务，不终止当前训练。runtime/queue_state.json是权威进度；runtime/jobs/<id>/console.log、exit.json和completed.json是逐任务凭据。outputs/<id>/保存实际配置、step_trace、任务矩阵和检查点。

collector周期生成新8任务summary.csv、RESULTS.md；THREE_SEED_SUMMARY.csv只在每方向0/1/2齐全时计算mean/sampleSD。所有历史引用保留路径和SHA。旧目录、旧队列与其他实验不改动，不发送外部通知。
