# 实验追踪

本轮计划已注册，正式结果尚未产生。实时状态以[runtime/queue_state.json](runtime/queue_state.json)为准，已验收结果见[results/summary.csv](results/summary.csv)。

- `smoke_parent_sgd_s2`：pending；依赖 CPU/文档验收
- `smoke_task1_B0_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_task1_R25_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_task1_S_F_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_task1_F_S_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_task1_S_S_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_task1_R100_s2`：pending；依赖 smoke_parent_sgd_s2
- `smoke_local_probe_audit_s2`：pending；依赖 smoke_task1_B0_s2
- `fork_smoke_pairing`：pending；依赖 smoke_parent_sgd_s2, smoke_task1_B0_s2, smoke_task1_R25_s2, smoke_task1_S_F_s2, smoke_task1_F_S_s2, smoke_task1_S_S_s2, smoke_task1_R100_s2, smoke_local_probe_audit_s2
- `formal_parent_sgd_s2`：pending；依赖 fork_smoke_pairing
- `formal_task1_B0_s2`：pending；依赖 formal_parent_sgd_s2
- `formal_task1_R25_s2`：pending；依赖 formal_parent_sgd_s2
- `formal_task1_S_F_s2`：pending；依赖 formal_parent_sgd_s2
- `formal_task1_F_S_s2`：pending；依赖 formal_parent_sgd_s2
- `formal_task1_S_S_s2`：pending；依赖 formal_parent_sgd_s2
- `formal_task1_R100_s2`：pending；依赖 formal_parent_sgd_s2
- `local_probe_audit_s2`：pending；依赖 formal_task1_B0_s2

## 已验证启动 2026-10-09T22:56:33.471976+00:00

GPU6正式共享SGD父模型已开始；已观察27/450更新、第3轮。7个训练短测、1个离线短测与配对验收全通过。六条Task1分支及正式离线审计等待成功依赖。后台监控每30秒读取本队列和GPU4/5/6；不会抢占其他任务。此处仅为启动快照，最新状态见runtime/queue_state.json、runtime/monitor_snapshot.json和results/STATUS.md。

## 最终完成核验 2026-10-10T07:21:17.039787+00:00

全队列17/17通过；正式训练7/7、正式诊断1/1；running0/failed0/pending0。结束于2026-10-10 02:09:25 CEST。旧pending/启动信息是历史快照，当前以结果核验和durablequeue为准。详见[完成与结果](results/status_20261010/STATUS_AND_RESULTS.md)、[正式结果CSV](results/comparison.csv)。
