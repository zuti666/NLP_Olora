# RQ2 自动执行状态

更新时间：2026-09-21T19:38:42.720580+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| verify_reused | 无 | complete |
| paths | verify_reused | complete |
| radii | verify_reused | complete |
| publish_B2 | paths | complete |
| publish_B1 | radii | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| verify_reused | CPU | complete | [verify_reused.attempt1.log](verify_reused.attempt1.log) |
| path_seed0_sgd_t03 | 1 | complete | [path_seed0_sgd_t03.attempt1.log](path_seed0_sgd_t03.attempt1.log) |
| path_seed2_gam_t02 | 4 | complete | [path_seed2_gam_t02.attempt1.log](path_seed2_gam_t02.attempt1.log) |
| path_seed2_gam_t04 | 1 | complete | [path_seed2_gam_t04.attempt1.log](path_seed2_gam_t04.attempt1.log) |
| path_seed2_sgd_t06 | 4 | complete | [path_seed2_sgd_t06.attempt1.log](path_seed2_sgd_t06.attempt1.log) |
| path_seed2_gam_t06 | 4 | complete | [path_seed2_gam_t06.attempt1.log](path_seed2_gam_t06.attempt1.log) |
| path_seed2_sgd_t07 | 4 | complete | [path_seed2_sgd_t07.attempt1.log](path_seed2_sgd_t07.attempt1.log) |
| path_seed2_gam_t07 | 1 | complete | [path_seed2_gam_t07.attempt1.log](path_seed2_gam_t07.attempt1.log) |
| path_seed2_sgd_t08 | 4 | complete | [path_seed2_sgd_t08.attempt1.log](path_seed2_sgd_t08.attempt1.log) |
| path_seed2_gam_t08 | 1 | complete | [path_seed2_gam_t08.attempt1.log](path_seed2_gam_t08.attempt1.log) |
| radius_seed0_sgd | 6 | complete | [radius_seed0_sgd.attempt1.log](radius_seed0_sgd.attempt1.log) |
| radius_seed0_gam | 4 | complete | [radius_seed0_gam.attempt1.log](radius_seed0_gam.attempt1.log) |
| radius_seed1_sgd | 1 | complete | [radius_seed1_sgd.attempt1.log](radius_seed1_sgd.attempt1.log) |
| radius_seed1_gam | 4 | complete | [radius_seed1_gam.attempt1.log](radius_seed1_gam.attempt1.log) |
| radius_seed2_sgd | 6 | complete | [radius_seed2_sgd.attempt1.log](radius_seed2_sgd.attempt1.log) |
| radius_seed2_gam | 1 | complete | [radius_seed2_gam.attempt1.log](radius_seed2_gam.attempt1.log) |
| publish_B2 | CPU | complete | [publish_B2.attempt1.log](publish_B2.attempt1.log) |
| publish_B1 | CPU | complete | [publish_B1.attempt1.log](publish_B1.attempt1.log) |
