# 独立文档调用检查

日期：2026-10-03 UTC。独立 agent 仅读取 run-experiment 技能、父目录 ENVIRONMENT.md 和本资源扩展 RUN.md，按文档原样通过 require_escalated 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/resource_extension_20261003T073600Z/resource_queue.py check --gpu 4
```

退出码：**1**。工具结果 chunk：`32fb6a`。

dry-run 为 `valid`，实际 28 jobs / 12 formal / 16 smokes，`allowed_gpus=[4,5,6,3,2]`、`max_parallel=3`，与文档预期完全一致。

随后在 `resource_queue.py:102` 报错：`RuntimeError: Selected witness GPU is busy; no preemption`。GPU4 在实际调用时未通过空闲门，因此 canonical seed2 CUDA forward/backward witness 未执行，未观察到优化器更新。本次输出不能证明成功 CUDA witness 或共享 index/UUID 锁释放；未另行读取或检查这些状态。

结论：资源 dry-run 通过，真实 GPU4 空闲门失败，文档调用整体未通过。没有训练启动、代码修改，也没有覆盖父目录冻结 RUN_PINS 或原 GPU3 DOCfail。本记录仅写入本资源扩展的新 DOC_CHECK.json / DOC_CHECK.md。
