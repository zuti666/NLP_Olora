# FS-EWC RQ2 核验与补实验安排

**2026-09-20：已按用户要求停止后台队列。进程组已确认退出，正式批量诊断和补充训练均未启动；不再自动继续。**

停止凭据：[STOPPED_BY_USER.json](execution/campaign_seed0/STOPPED_BY_USER.json)。

2026-09-20。完成原始结果重分析、源码审查和20个checkpoint的CPU张量核验。用户随后授权使用GPU4、6；GPU数值排查已执行，新诊断与补训练按自动队列推进。

- [最终补实验计划](EXPERIMENT_PLAN.md)：两个主张、A/B/C三块、一张表两张图；具体配置、代码复用点、排期与预算。
- [执行清单](EXPERIMENT_TRACKER.md)：区分已完成CPU工作、待实现测量和缺失训练分支。
- [跨r审查](AUDIT_RADIUS.md)、[跨T/模型资产审查](AUDIT_TASK_ASSETS.md)、[两任务审查](AUDIT_TWO_TASK.md)：逐项来源、脚本和实现逻辑。
- [旧损伤曲线数据](radius_damage_seed0.csv)、[经验半径](radius_tolerance_seed0.csv)、[逐任务结果](task_trajectory_recomputed.csv)。
- [半径来源清单](radius_reanalysis_manifest.json)、[checkpoint/配置清单](task_assets_manifest.json)。

关键结论：seed0两种方法Task0–Task9模型均在，B/C可优先离线补；旧train容忍半径改善、test不一致；旧Fisher trace已归一化，必须重测raw逐样本指标。A主配置缺GAM Task0→SGD+EWC Task1，最少补一段；严格随机配对无法复原时，从两个现有anchor各续一段Task1。

仅复现本轮CPU分析：

```bash
python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/reanalyze_radius.py
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/audit_task_assets_cpu.py
```

这两个入口分别读取旧JSON和CPU checkpoint，不是训练/离线Fisher评估入口。新的raw Fisher/path evaluator和两条Task1续训入口已实现；见 [执行说明](execution/EXECUTION.md)、[实时状态](execution/campaign_seed0/STATUS.md)、[汇总报告](execution/report_seed0/RESULTS.md)。只有完成凭据通过的项目才计为新结果。
