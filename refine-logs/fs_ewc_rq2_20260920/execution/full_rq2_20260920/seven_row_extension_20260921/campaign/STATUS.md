# RQ2 自动执行状态

更新时间：2026-09-21T18:49:58.878273+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| smoke_shaping_only_seed0 | 无 | complete |
| smoke_task2_shaping_seed0 | 无 | complete |
| smoke_validation | smoke_shaping_only_seed0, smoke_task2_shaping_seed0 | complete |
| shaping_only_seed0 | smoke_validation | complete |
| task2_shaping_seed0 | smoke_validation | complete |
| shaping_only_seed1 | smoke_validation | complete |
| task2_shaping_seed1 | smoke_validation | complete |
| shaping_only_seed2 | smoke_validation | complete |
| task2_shaping_seed2 | smoke_validation | complete |
| seven_row_report | shaping_only_seed0, task2_shaping_seed0, shaping_only_seed1, task2_shaping_seed1, shaping_only_seed2, task2_shaping_seed2 | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| smoke_shaping_only_seed0 | 1 | complete | [smoke_shaping_only_seed0.attempt1.log](smoke_shaping_only_seed0.attempt1.log) |
| smoke_task2_shaping_seed0 | 4 | complete | [smoke_task2_shaping_seed0.attempt1.log](smoke_task2_shaping_seed0.attempt1.log) |
| smoke_validation | CPU | complete | [smoke_validation.attempt1.log](smoke_validation.attempt1.log) |
| shaping_only_seed0 | 1 | complete | [shaping_only_seed0.attempt1.log](shaping_only_seed0.attempt1.log) |
| task2_shaping_seed0 | 4 | complete | [task2_shaping_seed0.attempt1.log](task2_shaping_seed0.attempt1.log) |
| shaping_only_seed1 | 2 | complete | [shaping_only_seed1.attempt1.log](shaping_only_seed1.attempt1.log) |
| task2_shaping_seed1 | 5 | complete | [task2_shaping_seed1.attempt1.log](task2_shaping_seed1.attempt1.log) |
| shaping_only_seed2 | 4 | complete | [shaping_only_seed2.attempt1.log](shaping_only_seed2.attempt1.log) |
| task2_shaping_seed2 | 1 | complete | [task2_shaping_seed2.attempt1.log](task2_shaping_seed2.attempt1.log) |
| seven_row_report | CPU | complete | [seven_row_report.attempt1.log](seven_row_report.attempt1.log) |
