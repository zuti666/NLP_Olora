# RQ2 自动执行状态

更新时间：2026-09-21T21:49:45.597728+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| wait_radius50 | 无 | complete |
| fixed_gradient | wait_radius50 | complete |
| publish | fixed_gradient | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| wait_random_radius50 | CPU | complete | [wait_random_radius50.attempt1.log](wait_random_radius50.attempt1.log) |
| gradient_seed0_sgd | 1 | complete | [gradient_seed0_sgd.attempt1.log](gradient_seed0_sgd.attempt1.log) |
| gradient_seed0_gam | 4 | complete | [gradient_seed0_gam.attempt1.log](gradient_seed0_gam.attempt1.log) |
| gradient_seed1_sgd | 6 | complete | [gradient_seed1_sgd.attempt1.log](gradient_seed1_sgd.attempt1.log) |
| gradient_seed1_gam | 1 | complete | [gradient_seed1_gam.attempt1.log](gradient_seed1_gam.attempt1.log) |
| gradient_seed2_sgd | 4 | complete | [gradient_seed2_sgd.attempt1.log](gradient_seed2_sgd.attempt1.log) |
| gradient_seed2_gam | 6 | complete | [gradient_seed2_gam.attempt1.log](gradient_seed2_gam.attempt1.log) |
| publish_fixed_gradient | CPU | complete | [publish_fixed_gradient.attempt1.log](publish_fixed_gradient.attempt1.log) |
