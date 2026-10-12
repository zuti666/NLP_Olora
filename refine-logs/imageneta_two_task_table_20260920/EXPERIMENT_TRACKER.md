# ImageNet-A 两任务表执行清单

实际状态由 [队列状态](campaign/STATUS.md) 与 [原始状态JSON](campaign/pipeline_state.json) 自动更新。本文件记录启动时的安排，不能代替实时状态。

| 阶段 | 工作 | 启动时状态 |
|---|---|---|
| CPU | 17项科学测试、真实ViT初始化、EF/proxy校验、汇总合成输入检查 | PASS |
| GPU sanity | seed0六行、3epochs/task、快照1/2、EF8；失败不放行正式任务 | WAITING_GPU |
| 正式seed0 | 2 anchors + 6 Task1分支 + 6评估 + 首张单seed表 | QUEUED |
| 正式seed1 | 同配置补第2seed并更新汇总 | QUEUED |
| 正式seed2 | 同配置补第3seed，生成均值±样本标准差表 | QUEUED |

后台调度器 PID3811571，物理GPU4/6。启动时两卡被其他作业占用，未抢占；当前没有本批正式训练结果。完整配置位于 scripts/imageneta_two_task_table/config.json。

六行均统一启用CE加权因子轨迹混合。原表ImageNet-R/Adam/fresh-KV/无轨迹平均的说明与旧数值不用于新表。
