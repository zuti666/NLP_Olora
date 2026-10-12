# RQ2 自动执行状态

更新时间：2026-09-21T18:19:12.622493+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| sgd_sanity | 无 | complete |
| gam_sanity | 无 | complete |
| sgd_neighborhood | sgd_sanity | complete |
| gam_neighborhood | gam_sanity | complete |
| publish_B | sgd_neighborhood, gam_neighborhood | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| sgd_sanity | 0 | complete | [sgd_sanity.attempt1.log](sgd_sanity.attempt1.log) |
| gam_sanity | 3 | complete | [gam_sanity.attempt1.log](gam_sanity.attempt1.log) |
| sgd_neighborhood | 0 | complete | [sgd_neighborhood.attempt1.log](sgd_neighborhood.attempt1.log) |
| gam_neighborhood | 3 | complete | [gam_neighborhood.attempt1.log](gam_neighborhood.attempt1.log) |
| publish_B | CPU | complete | [publish_B.attempt1.log](publish_B.attempt1.log) |
