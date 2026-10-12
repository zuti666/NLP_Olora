# RQ2 自动执行状态

更新时间：2026-09-21T17:01:08.441078+00:00
总状态：running；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| smoke_shaping_only_seed0 | 无 | ready |
| smoke_task2_shaping_seed0 | 无 | ready |
| smoke_validation | smoke_shaping_only_seed0, smoke_task2_shaping_seed0 | pending |
| shaping_only_seed0 | smoke_validation | pending |
| task2_shaping_seed0 | smoke_validation | pending |
| shaping_only_seed1 | smoke_validation | pending |
| task2_shaping_seed1 | smoke_validation | pending |
| shaping_only_seed2 | smoke_validation | pending |
| task2_shaping_seed2 | smoke_validation | pending |
| seven_row_report | shaping_only_seed0, task2_shaping_seed0, shaping_only_seed1, task2_shaping_seed1, shaping_only_seed2, task2_shaping_seed2 | pending |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| smoke_shaping_only_seed0 | 0 | queued | 待启动 |
| smoke_task2_shaping_seed0 | 0 | queued | 待启动 |
| smoke_validation | CPU | queued | 待启动 |
| shaping_only_seed0 | 0 | queued | 待启动 |
| task2_shaping_seed0 | 0 | queued | 待启动 |
| shaping_only_seed1 | 0 | queued | 待启动 |
| task2_shaping_seed1 | 0 | queued | 待启动 |
| shaping_only_seed2 | 0 | queued | 待启动 |
| task2_shaping_seed2 | 0 | queued | 待启动 |
| seven_row_report | CPU | queued | 待启动 |
