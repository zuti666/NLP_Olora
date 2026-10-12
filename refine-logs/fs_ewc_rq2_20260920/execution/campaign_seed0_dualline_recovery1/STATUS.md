# RQ2 自动执行状态

更新时间：2026-09-20T15:36:08.384510+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| sanity | 无 | complete |
| neighborhood | sanity | complete |
| training | sanity | complete |
| bridge | neighborhood | complete |
| table_a | training | complete |
| trajectory | table_a | complete |
| path | trajectory | complete |
| report | bridge, path | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| sanity_sgd | 0 | complete | [sanity_sgd.attempt1.log](sanity_sgd.attempt1.log) |
| sanity_gam | 1 | complete | [sanity_gam.attempt1.log](sanity_gam.attempt1.log) |
| neighborhood_sgd | 0 | complete | [neighborhood_sgd.attempt1.log](neighborhood_sgd.attempt1.log) |
| neighborhood_gam | 1 | complete | [neighborhood_gam.attempt1.log](neighborhood_gam.attempt1.log) |
| training_sgd | 2 | complete | [training_sgd.attempt1.log](training_sgd.attempt1.log) |
| training_gam | 3 | complete | [training_gam.attempt1.log](training_gam.attempt1.log) |
| bridge_sgd | 0 | complete | [bridge_sgd.attempt1.log](bridge_sgd.attempt1.log) |
| bridge_gam | 1 | complete | [bridge_gam.attempt1.log](bridge_gam.attempt1.log) |
| table_a_sgd | 2 | complete | [table_a_sgd.attempt1.log](table_a_sgd.attempt1.log) |
| table_a_gam | 3 | complete | [table_a_gam.attempt1.log](table_a_gam.attempt1.log) |
| trajectory_sgd | 2 | complete | [trajectory_sgd.attempt1.log](trajectory_sgd.attempt1.log) |
| trajectory_gam | 3 | complete | [trajectory_gam.attempt1.log](trajectory_gam.attempt1.log) |
| path_sgd | 2 | complete | [path_sgd.attempt1.log](path_sgd.attempt1.log) |
| path_gam | 3 | complete | [path_gam.attempt1.log](path_gam.attempt1.log) |
| report_seed0 | CPU | complete | [report_seed0.attempt1.log](report_seed0.attempt1.log) |
