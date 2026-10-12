# RQ2 自动执行状态

更新时间：2026-09-20T12:39:07.574565+00:00
总状态：failed；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| sanity | 无 | complete |
| neighborhood | sanity | failed |
| training | sanity | complete |
| bridge | neighborhood | blocked |
| table_a | training | blocked |
| trajectory | table_a | blocked |
| path | trajectory | blocked |
| report | bridge, path | blocked |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| sanity_sgd | 0 | complete | [sanity_sgd.attempt1.log](sanity_sgd.attempt1.log) |
| sanity_gam | 1 | complete | [sanity_gam.attempt1.log](sanity_gam.attempt1.log) |
| neighborhood_sgd | 0 | failed | [neighborhood_sgd.attempt1.log](neighborhood_sgd.attempt1.log) |
| neighborhood_gam | 1 | cancelled | 待启动 |
| training_sgd | 2 | complete | [training_sgd.attempt1.log](training_sgd.attempt1.log) |
| training_gam | 3 | complete | [training_gam.attempt1.log](training_gam.attempt1.log) |
| bridge_sgd | 0 | blocked | 待启动 |
| bridge_gam | 1 | blocked | 待启动 |
| table_a_sgd | 2 | blocked | 待启动 |
| table_a_gam | 3 | blocked | 待启动 |
| trajectory_sgd | 2 | blocked | 待启动 |
| trajectory_gam | 3 | blocked | 待启动 |
| path_sgd | 2 | blocked | 待启动 |
| path_gam | 3 | blocked | 待启动 |
| report_seed0 | CPU | blocked | 待启动 |

错误：Job neighborhood_sgd failed; no further jobs will start
