# RQ2 自动执行状态

更新时间：2026-09-21T23:11:07.907033+00:00
总状态：complete；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| prepare_raw_fisher | 无 | complete |
| smoke_sgd_lambda60 | prepare_raw_fisher | complete |
| validate_smoke | smoke_sgd_lambda60 | complete |
| lambda_0_seed0 | validate_smoke | complete |
| lambda_1_seed0 | validate_smoke | complete |
| lambda_10_seed0 | validate_smoke | complete |
| lambda_30_seed0 | validate_smoke | complete |
| lambda_60_seed0 | validate_smoke | complete |
| lambda_100_seed0 | validate_smoke | complete |
| lambda_300_seed0 | validate_smoke | complete |
| lambda_1000_seed0 | validate_smoke | complete |
| lambda_3000_seed0 | validate_smoke | complete |
| lambda_10000_seed0 | validate_smoke | complete |
| lambda_30000_seed0 | validate_smoke | complete |
| lambda_100000_seed0 | validate_smoke | complete |
| lambda_300000_seed0 | validate_smoke | complete |
| lambda_1000000_seed0 | validate_smoke | complete |
| sweep_report | lambda_0_seed0, lambda_1_seed0, lambda_10_seed0, lambda_30_seed0, lambda_60_seed0, lambda_100_seed0, lambda_300_seed0, lambda_1000_seed0, lambda_3000_seed0, lambda_10000_seed0, lambda_30000_seed0, lambda_100000_seed0, lambda_300000_seed0, lambda_1000000_seed0 | complete |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| prepare_raw_fisher | 0 | complete | [prepare_raw_fisher.attempt1.log](prepare_raw_fisher.attempt1.log) |
| smoke_sgd_lambda60 | 1 | complete | [smoke_sgd_lambda60.attempt1.log](smoke_sgd_lambda60.attempt1.log) |
| validate_smoke | CPU | complete | [validate_smoke.attempt1.log](validate_smoke.attempt1.log) |
| lambda_0_seed0 | 0 | complete | [lambda_0_seed0.attempt1.log](lambda_0_seed0.attempt1.log) |
| lambda_1_seed0 | 1 | complete | [lambda_1_seed0.attempt1.log](lambda_1_seed0.attempt1.log) |
| lambda_10_seed0 | 2 | complete | [lambda_10_seed0.attempt1.log](lambda_10_seed0.attempt1.log) |
| lambda_30_seed0 | 3 | complete | [lambda_30_seed0.attempt1.log](lambda_30_seed0.attempt1.log) |
| lambda_60_seed0 | 4 | complete | [lambda_60_seed0.attempt1.log](lambda_60_seed0.attempt1.log) |
| lambda_100_seed0 | 5 | complete | [lambda_100_seed0.attempt1.log](lambda_100_seed0.attempt1.log) |
| lambda_300_seed0 | 4 | complete | [lambda_300_seed0.attempt1.log](lambda_300_seed0.attempt1.log) |
| lambda_1000_seed0 | 6 | complete | [lambda_1000_seed0.attempt1.log](lambda_1000_seed0.attempt1.log) |
| lambda_3000_seed0 | 0 | complete | [lambda_3000_seed0.attempt1.log](lambda_3000_seed0.attempt1.log) |
| lambda_10000_seed0 | 1 | complete | [lambda_10000_seed0.attempt1.log](lambda_10000_seed0.attempt1.log) |
| lambda_30000_seed0 | 2 | complete | [lambda_30000_seed0.attempt1.log](lambda_30000_seed0.attempt1.log) |
| lambda_100000_seed0 | 3 | complete | [lambda_100000_seed0.attempt1.log](lambda_100000_seed0.attempt1.log) |
| lambda_300000_seed0 | 5 | complete | [lambda_300000_seed0.attempt1.log](lambda_300000_seed0.attempt1.log) |
| lambda_1000000_seed0 | 4 | complete | [lambda_1000000_seed0.attempt1.log](lambda_1000000_seed0.attempt1.log) |
| sweep_report | CPU | complete | [sweep_report.attempt1.log](sweep_report.attempt1.log) |

全卡启动准入：admitted；阻塞 GPU：[]
最近检查：2026-09-21T22:57:35.835244+00:00；下次检查：None

| GPU | UUID | 显存 MiB | 利用率 % | Compute 进程 | 阻塞原因 |
|---|---|---|---|---|---|
| 0 | GPU-fd7e2955-3733-e30a-dff1-5da992423839 | 2 | 0 | False | 无 |
| 1 | GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a | 2 | 0 | False | 无 |
| 2 | GPU-fe4ca05a-b7fd-efa7-8192-201e24573825 | 2 | 0 | False | 无 |
| 3 | GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26 | 2 | 0 | False | 无 |
| 4 | GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5 | 2 | 0 | False | 无 |
| 5 | GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e | 2 | 0 | False | 无 |
| 6 | GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a | 2 | 0 | False | 无 |
