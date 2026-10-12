# 细化搜索：GPU2–6动态队列

2026-09-22用户明确覆盖旧全卡等待规则：优先GPU3、4、6，GPU2、5空闲后可承接待执行任务；排除GPU0、1。按每卡空闲状态和UUID共享锁准入，无须全部同时空闲；最多5并行，定时资源检查900秒，已有任务结束时也会调度下一项。

只修改资源字段与队列状态目录。六个λ、seed0、原始raw Fisher、训练代码、50轮、评估及汇总命令完全不变，结果仍在training/与report/。Pilot_new@e94a148f环境不变。每项训练前仍执行原CUDA witness。

## 只读核验

项目根目录原样运行：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/plan_gpu23456.json
```

预期dry_run=true，stages=7，jobs=7，max_parallel=5；六训练后接一个combined_report。CLI输出依赖和234项文件指纹。资源政策须另读plan_gpu23456.json核验：gpus和每GPU job的gpu_pool均为[3,4,6,2,5]，all_gpu_start_gate.required=false，gpu_poll_seconds=900。只读命令不访问GPU、不启动作业。

## 实际执行

先在宿主机核对旧LAUNCH.json控制器身份、所有7job仍queued且没有子进程，结束仅等待的旧controller并保存STOP_FOR_GPU_POLICY.json；若已有训练则不能按此迁移。原plan/campaign及科学文件保留。然后启动：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda_refinement_round1_20260922/plan_gpu23456.json --execute
```

保存新启动身份LAUNCH_GPU23456.json与宿主机核验LAUNCH_GPU23456_VERIFIED.json。当前队列状态在campaign_gpu23456/STATUS.md；旧campaign/是已停止的历史等待状态，不能用来判断新队列。禁止两个controller同时管理同一六个trial。
