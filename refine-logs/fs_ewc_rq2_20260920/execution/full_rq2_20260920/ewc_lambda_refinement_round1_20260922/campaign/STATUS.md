# RQ2 自动执行状态

更新时间：2026-09-22T06:13:56.231186+00:00
总状态：waiting_for_all_gpus；活动阶段：无

| 阶段 | 依赖 | 状态 |
|---|---|---|
| refine1_01_seed0 | 无 | ready |
| refine1_02_seed0 | 无 | ready |
| refine1_03_seed0 | 无 | ready |
| refine1_04_seed0 | 无 | ready |
| refine1_05_seed0 | 无 | ready |
| refine1_06_seed0 | 无 | ready |
| combined_report | refine1_01_seed0, refine1_02_seed0, refine1_03_seed0, refine1_04_seed0, refine1_05_seed0, refine1_06_seed0 | pending |

| 作业 | GPU | 状态 | 日志 |
|---|---|---|---|
| refine1_01_seed0 | 0 | queued | 待启动 |
| refine1_02_seed0 | 0 | queued | 待启动 |
| refine1_03_seed0 | 0 | queued | 待启动 |
| refine1_04_seed0 | 0 | queued | 待启动 |
| refine1_05_seed0 | 0 | queued | 待启动 |
| refine1_06_seed0 | 0 | queued | 待启动 |
| combined_report | CPU | queued | 待启动 |

全卡启动准入：waiting_for_all_gpus；阻塞 GPU：[2, 5]
最近检查：2026-09-22T06:13:56.231123+00:00；下次检查：2026-09-22T06:28:56.231123+00:00

| GPU | UUID | 显存 MiB | 利用率 % | Compute 进程 | 阻塞原因 |
|---|---|---|---|---|---|
| 0 | GPU-fd7e2955-3733-e30a-dff1-5da992423839 | 2 | 0 | False | 无 |
| 1 | GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a | 2 | 0 | False | 无 |
| 2 | GPU-fe4ca05a-b7fd-efa7-8192-201e24573825 | 12406 | 100 | True | memory_mib_at_least_500, utilization_at_least_5, compute_process_present |
| 3 | GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26 | 2 | 0 | False | 无 |
| 4 | GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5 | 2 | 0 | False | 无 |
| 5 | GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e | 12368 | 100 | True | memory_mib_at_least_500, utilization_at_least_5, compute_process_present |
| 6 | GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a | 2 | 0 | False | 无 |
