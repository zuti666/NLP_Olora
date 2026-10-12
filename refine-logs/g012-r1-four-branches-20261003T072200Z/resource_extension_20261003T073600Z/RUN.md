# 空闲GPU资源扩展

原GPU3在启动前被其他训练占用，精确原文档调用dry-run通过、空闲门失败，证据保留于父目录gates/DOC_CHECK.json。原GPU2独立S匹配实验继续运行。沿用会话用户“使用所有空闲GPU”及原四分支实验GPU2–6授权，本资源扩展优先使用空闲GPU4/5/6；GPU2/3空闲后可接收任务，最多并行3。GPU0/1不进入本轮池。

冻结科学代码、配置、CPU proof、历史基准审计与RUN_PINS全部保持原bytes。资源wrapper只在内存改变allowed_gpus/max_parallel，并继续调用原科学contract检查原suite。worker和真实训练仍执行原被冻结的入口，无数值hook替换、无新方向或任务、无阈值放宽。

环境参照父目录ENVIRONMENT.md：软件/spec/原生source190未变，resource声明覆盖其中GPU2/3一节。资源验证者读取 run-experiment skill、父ENVIRONMENT.md与本文件，逐字执行下列调用（真实host GPU访问，require_escalated；只dry-run和seed2小CUDA forward/backward，0optimizer更新）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T073600Z/resource_queue.py check --gpu 4
```

预期dry-run：28jobs、12formal、16smokes，allowed_gpus=[4,5,6,3,2]，max_parallel=3；canonicalCUDA sentinel有限。成功记录DOCUMENTED_WITNESS.json，释放index/UUID双共享锁。不得声称原GPU3验证已通过。

通过后root以start_new_session后台执行真正队列入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T073600Z/resource_queue.py controller
```

资源manifest绑定本wrapper SHA与原suite/RUN_PINS。根目录LAUNCH.json记录实际controller PID/starttime、资源manifest与wrapper hashes；实际每job的GPU UUID、worker/runner PID和退出凭据由原队列保存。动态状态和结果仍在父目录runtime/queue_state.json、RESULTS.md，不创建重复队列或恢复旧Stage2。原RUN.md单纯GPU2/3入口已由本资源说明替代，不要两入口同时启动。
