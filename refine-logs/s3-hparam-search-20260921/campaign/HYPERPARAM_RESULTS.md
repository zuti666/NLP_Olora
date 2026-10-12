# S3 seed2 单轴搜索及已有seed2/3统计

更新：2026-09-21T09:44:52.293352+00:00。已复用9个完整配置；新增完成 0/7。

新增仅seed2；已有seed2/3完整结果均纳入。每点标明实际seed数；n=1时样本标准差为N/A，配对增量使用同seed中心。

| 轴 | 值 | seed2 FAA | seed3 FAA | 完成seed | FAA均值 | 样本SD | 同seed配对Δ均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ewc_lambda | 0 | 53.500 | — | 1/1 | 53.500 | N/A | 0.081 |
| ewc_lambda | 200 | 53.405 | — | 1/1 | 53.405 | N/A | -0.014 |
| ewc_lambda | 2000 | 53.419 | 52.006 | 2/2 | 52.713 | 0.999 | 0.000 |
| ewc_lambda | 20000 | 53.426 | — | 1/1 | 53.426 | N/A | 0.007 |
| gam_grad_norm_rho | 0.1 | 52.462 | — | 1/1 | 52.462 | N/A | -0.957 |
| gam_grad_norm_rho | 0.2 | 53.419 | 52.006 | 2/2 | 52.713 | 0.999 | 0.000 |
| gam_grad_norm_rho | 0.5 | 54.978 | — | 1/1 | 54.978 | N/A | 1.559 |
| ewc_fisher_damping_alpha | 0.0 | 53.431 | — | 1/1 | 53.431 | N/A | 0.012 |
| ewc_fisher_damping_alpha | 0.1 | — | — | 0/1 | N/A | N/A | N/A |
| ewc_fisher_damping_alpha | 0.2 | 53.419 | 52.006 | 2/2 | 52.713 | 0.999 | 0.000 |
| ewc_fisher_damping_alpha | 0.5 | — | — | 0/1 | N/A | N/A | N/A |
| ewc_fisher_damping_alpha | 1.0 | 53.353 | — | 1/1 | 53.353 | N/A | -0.066 |
| in_task_fisher_omega | 0.05 | — | — | 0/1 | N/A | N/A | N/A |
| in_task_fisher_omega | 0.1 | — | — | 0/1 | N/A | N/A | N/A |
| in_task_fisher_omega | 0.2 | — | — | 0/1 | N/A | N/A | N/A |
| in_task_fisher_omega | 0.3 | 53.419 | 52.006 | 2/2 | 52.713 | 0.999 | 0.000 |
| in_task_fisher_omega | 0.4 | — | — | 0/1 | N/A | N/A | N/A |
| in_task_fisher_omega | 0.5 | — | — | 0/1 | N/A | N/A | N/A |

原日志中心FAA：seed2=53.311、seed3=51.968；本表采用同冻结源码的新执行中心53.419/52.006，不将两次执行的数值混用。

weightmerge系数为正值扫描；已有保留快照CE的matched-no-merge单独作为零融合控制，不伪装成原生omega=0。

详细完成状态与任务进度见 EXPERIMENT_TRACKER.md；所有GPU0–6均可分配，仍被占用的GPU等待释放。
