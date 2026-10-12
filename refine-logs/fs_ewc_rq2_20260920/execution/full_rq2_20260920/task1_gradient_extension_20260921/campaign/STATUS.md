# RQ2 自动执行状态

更新时间：2026-09-21T22:28:24.130451+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| task1_gradient | 无 | complete |
| publish | task1_gradient | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| task1_gradient_seed0_sgd | 1 | complete | [task1_gradient_seed0_sgd.attempt1.log](task1_gradient_seed0_sgd.attempt1.log) |
| task1_gradient_seed0_gam | 4 | complete | [task1_gradient_seed0_gam.attempt1.log](task1_gradient_seed0_gam.attempt1.log) |
| task1_gradient_seed1_sgd | 6 | complete | [task1_gradient_seed1_sgd.attempt1.log](task1_gradient_seed1_sgd.attempt1.log) |
| task1_gradient_seed1_gam | 1 | complete | [task1_gradient_seed1_gam.attempt1.log](task1_gradient_seed1_gam.attempt1.log) |
| task1_gradient_seed2_sgd | 4 | complete | [task1_gradient_seed2_sgd.attempt1.log](task1_gradient_seed2_sgd.attempt1.log) |
| task1_gradient_seed2_gam | 6 | complete | [task1_gradient_seed2_gam.attempt1.log](task1_gradient_seed2_gam.attempt1.log) |
| publish_task1_gradient | CPU | complete | [publish_task1_gradient.attempt1.log](publish_task1_gradient.attempt1.log) |
