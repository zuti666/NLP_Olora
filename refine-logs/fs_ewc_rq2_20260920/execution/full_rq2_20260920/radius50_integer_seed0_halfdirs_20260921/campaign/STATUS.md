# RQ2 自动执行状态

更新时间：2026-09-21T22:53:29.690908+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| measure | 无 | complete |
| publish | measure | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| dense_seed0_sgd | 0 | complete | [dense_seed0_sgd.attempt1.log](dense_seed0_sgd.attempt1.log) |
| dense_seed0_gam | 2 | complete | [dense_seed0_gam.attempt1.log](dense_seed0_gam.attempt1.log) |
| sparse_train_seed0_sgd | 3 | complete | [sparse_train_seed0_sgd.attempt1.log](sparse_train_seed0_sgd.attempt1.log) |
| sparse_train_seed0_gam | 5 | complete | [sparse_train_seed0_gam.attempt1.log](sparse_train_seed0_gam.attempt1.log) |
| publish_radius50 | CPU | complete | [publish_radius50.attempt1.log](publish_radius50.attempt1.log) |
