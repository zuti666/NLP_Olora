# RQ2 自动执行状态

更新时间：2026-09-22T09:26:32.450037+00:00
总状态：failed；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| prepare_sgd_seed0 | 无 | complete |
| prepare_sgd_seed1 | 无 | blocked |
| prepare_sgd_seed2 | 无 | failed |
| prepare_gam_seed0 | 无 | complete |
| prepare_gam_seed1 | 无 | blocked |
| prepare_gam_seed2 | 无 | failed |
| smoke_ewclora_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | blocked |
| smoke_fs_ewclora_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | blocked |
| smoke_task1_shaping_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | blocked |
| smoke_task2_shaping_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | blocked |
| validate_smokes | smoke_ewclora_seed0, smoke_fs_ewclora_seed0, smoke_task1_shaping_seed0, smoke_task2_shaping_seed0 | blocked |
| ewclora_seed0 | validate_smokes | blocked |
| fs_ewclora_seed0 | validate_smokes | blocked |
| task1_shaping_seed0 | validate_smokes | blocked |
| task2_shaping_seed0 | validate_smokes | blocked |
| ewclora_seed1 | validate_smokes, ewclora_seed0 | blocked |
| fs_ewclora_seed1 | validate_smokes, fs_ewclora_seed0 | blocked |
| task1_shaping_seed1 | validate_smokes, task1_shaping_seed0 | blocked |
| task2_shaping_seed1 | validate_smokes, task2_shaping_seed0 | blocked |
| ewclora_seed2 | validate_smokes, ewclora_seed1 | blocked |
| fs_ewclora_seed2 | validate_smokes, fs_ewclora_seed1 | blocked |
| task1_shaping_seed2 | validate_smokes, task1_shaping_seed1 | blocked |
| task2_shaping_seed2 | validate_smokes, task2_shaping_seed1 | blocked |
| report | ewclora_seed0, ewclora_seed1, ewclora_seed2, fs_ewclora_seed0, fs_ewclora_seed1, fs_ewclora_seed2, task1_shaping_seed0, task1_shaping_seed1, task1_shaping_seed2, task2_shaping_seed0, task2_shaping_seed1, task2_shaping_seed2 | blocked |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| prepare_sgd_seed0 | CPU | complete | [prepare_sgd_seed0.attempt1.log](prepare_sgd_seed0.attempt1.log) |
| prepare_sgd_seed1 | 0 | cancelled | [prepare_sgd_seed1.attempt1.log](prepare_sgd_seed1.attempt1.log) |
| prepare_sgd_seed2 | 1 | failed | [prepare_sgd_seed2.attempt1.log](prepare_sgd_seed2.attempt1.log) |
| prepare_gam_seed0 | 2 | complete | [prepare_gam_seed0.attempt1.log](prepare_gam_seed0.attempt1.log) |
| prepare_gam_seed1 | 5 | cancelled | [prepare_gam_seed1.attempt1.log](prepare_gam_seed1.attempt1.log) |
| prepare_gam_seed2 | 4 | failed | [prepare_gam_seed2.attempt1.log](prepare_gam_seed2.attempt1.log) |
| smoke_ewclora_seed0 | 0 | blocked | 待启动 |
| smoke_fs_ewclora_seed0 | 1 | blocked | 待启动 |
| smoke_task1_shaping_seed0 | 2 | blocked | 待启动 |
| smoke_task2_shaping_seed0 | 5 | blocked | 待启动 |
| validate_smokes | CPU | blocked | 待启动 |
| ewclora_seed0 | 0 | blocked | 待启动 |
| fs_ewclora_seed0 | 1 | blocked | 待启动 |
| task1_shaping_seed0 | 2 | blocked | 待启动 |
| task2_shaping_seed0 | 5 | blocked | 待启动 |
| ewclora_seed1 | 0 | blocked | 待启动 |
| fs_ewclora_seed1 | 1 | blocked | 待启动 |
| task1_shaping_seed1 | 2 | blocked | 待启动 |
| task2_shaping_seed1 | 5 | blocked | 待启动 |
| ewclora_seed2 | 0 | blocked | 待启动 |
| fs_ewclora_seed2 | 1 | blocked | 待启动 |
| task1_shaping_seed2 | 2 | blocked | 待启动 |
| task2_shaping_seed2 | 5 | blocked | 待启动 |
| report | CPU | blocked | 待启动 |

错误：Job prepare_gam_seed2 failed; no further jobs will start
