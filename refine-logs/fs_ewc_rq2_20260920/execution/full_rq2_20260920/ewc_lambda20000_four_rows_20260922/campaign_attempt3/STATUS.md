# RQ2 自动执行状态

更新时间：2026-09-22T10:48:10.448467+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| prepare_sgd_seed0 | 无 | complete |
| prepare_sgd_seed1 | 无 | complete |
| prepare_sgd_seed2 | 无 | complete |
| prepare_gam_seed0 | 无 | complete |
| prepare_gam_seed1 | 无 | complete |
| prepare_gam_seed2 | 无 | complete |
| smoke_ewclora_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | complete |
| smoke_fs_ewclora_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | complete |
| smoke_task1_shaping_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | complete |
| smoke_task2_shaping_seed0 | prepare_sgd_seed0, prepare_sgd_seed1, prepare_sgd_seed2, prepare_gam_seed0, prepare_gam_seed1, prepare_gam_seed2 | complete |
| validate_smokes | smoke_ewclora_seed0, smoke_fs_ewclora_seed0, smoke_task1_shaping_seed0, smoke_task2_shaping_seed0 | complete |
| ewclora_seed0 | validate_smokes | complete |
| fs_ewclora_seed0 | validate_smokes | complete |
| task1_shaping_seed0 | validate_smokes | complete |
| task2_shaping_seed0 | validate_smokes | complete |
| ewclora_seed1 | validate_smokes, ewclora_seed0 | complete |
| fs_ewclora_seed1 | validate_smokes, fs_ewclora_seed0 | complete |
| task1_shaping_seed1 | validate_smokes, task1_shaping_seed0 | complete |
| task2_shaping_seed1 | validate_smokes, task2_shaping_seed0 | complete |
| ewclora_seed2 | validate_smokes, ewclora_seed1 | complete |
| fs_ewclora_seed2 | validate_smokes, fs_ewclora_seed1 | complete |
| task1_shaping_seed2 | validate_smokes, task1_shaping_seed1 | complete |
| task2_shaping_seed2 | validate_smokes, task2_shaping_seed1 | complete |
| report | ewclora_seed0, ewclora_seed1, ewclora_seed2, fs_ewclora_seed0, fs_ewclora_seed1, fs_ewclora_seed2, task1_shaping_seed0, task1_shaping_seed1, task1_shaping_seed2, task2_shaping_seed0, task2_shaping_seed1, task2_shaping_seed2 | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| prepare_sgd_seed0 | CPU | complete | [prepare_sgd_seed0.attempt1.log](prepare_sgd_seed0.attempt1.log) |
| prepare_sgd_seed1 | 0 | complete | [prepare_sgd_seed1.attempt1.log](prepare_sgd_seed1.attempt1.log) |
| prepare_sgd_seed2 | 1 | complete | [prepare_sgd_seed2.attempt1.log](prepare_sgd_seed2.attempt1.log) |
| prepare_gam_seed0 | 2 | complete | [prepare_gam_seed0.attempt1.log](prepare_gam_seed0.attempt1.log) |
| prepare_gam_seed1 | 5 | complete | [prepare_gam_seed1.attempt1.log](prepare_gam_seed1.attempt1.log) |
| prepare_gam_seed2 | 4 | complete | [prepare_gam_seed2.attempt1.log](prepare_gam_seed2.attempt1.log) |
| smoke_ewclora_seed0 | 0 | complete | [smoke_ewclora_seed0.attempt1.log](smoke_ewclora_seed0.attempt1.log) |
| smoke_fs_ewclora_seed0 | 1 | complete | [smoke_fs_ewclora_seed0.attempt1.log](smoke_fs_ewclora_seed0.attempt1.log) |
| smoke_task1_shaping_seed0 | 2 | complete | [smoke_task1_shaping_seed0.attempt1.log](smoke_task1_shaping_seed0.attempt1.log) |
| smoke_task2_shaping_seed0 | 5 | complete | [smoke_task2_shaping_seed0.attempt1.log](smoke_task2_shaping_seed0.attempt1.log) |
| validate_smokes | CPU | complete | [validate_smokes.attempt1.log](validate_smokes.attempt1.log) |
| ewclora_seed0 | 0 | complete | [ewclora_seed0.attempt1.log](ewclora_seed0.attempt1.log) |
| fs_ewclora_seed0 | 1 | complete | [fs_ewclora_seed0.attempt1.log](fs_ewclora_seed0.attempt1.log) |
| task1_shaping_seed0 | 2 | complete | [task1_shaping_seed0.attempt1.log](task1_shaping_seed0.attempt1.log) |
| task2_shaping_seed0 | 5 | complete | [task2_shaping_seed0.attempt1.log](task2_shaping_seed0.attempt1.log) |
| ewclora_seed1 | 4 | complete | [ewclora_seed1.attempt1.log](ewclora_seed1.attempt1.log) |
| fs_ewclora_seed1 | 0 | complete | [fs_ewclora_seed1.attempt1.log](fs_ewclora_seed1.attempt1.log) |
| task1_shaping_seed1 | 0 | complete | [task1_shaping_seed1.attempt1.log](task1_shaping_seed1.attempt1.log) |
| task2_shaping_seed1 | 1 | complete | [task2_shaping_seed1.attempt1.log](task2_shaping_seed1.attempt1.log) |
| ewclora_seed2 | 2 | complete | [ewclora_seed2.attempt1.log](ewclora_seed2.attempt1.log) |
| fs_ewclora_seed2 | 2 | complete | [fs_ewclora_seed2.attempt1.log](fs_ewclora_seed2.attempt1.log) |
| task1_shaping_seed2 | 4 | complete | [task1_shaping_seed2.attempt1.log](task1_shaping_seed2.attempt1.log) |
| task2_shaping_seed2 | 5 | complete | [task2_shaping_seed2.attempt1.log](task2_shaping_seed2.attempt1.log) |
| report | CPU | complete | [report.attempt1.log](report.attempt1.log) |
