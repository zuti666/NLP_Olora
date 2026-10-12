# 现有 S3 两个 campaign 超参点核验

核验时间：2026-09-21T09:35:03.430595+00:00

只读核对源码清单、配置、启动/完成 receipt、控制台 500 个 task/epoch 对、十任务原始矩阵及指标。没有使用 GPU，没有修改队列。

FAA=最后一行任务准确率均值；AAA=十个阶段的已见任务均值再平均；遗忘=各旧任务历次最高成绩（包括最终阶段）减最终成绩后平均。sample 为控制台最终 CNN top1。未来任务格子的 NaN 是结构性空值。

## 核验状态

{"status_counts": {"completed": 22, "running": 4}, "all_row_checks_passed": true, "target_completed_core_rows": ["s3_center_s2", "s3_lambda0_s2", "s3_lambda200_s2", "s3_lambda20000_s2", "s3_radius01_s2", "s3_radius05_s2", "s3_mech_a_nomerge_matched_s2", "s3_mech_a_off_s2", "s3_mech_center_s3", "s3_mech_a_nomerge_matched_s3", "s3_mech_a_uniform_s2", "s3_mech_a_latest_s2", "s3_mech_fisher_alpha0_s2", "s3_mech_fisher_alpha1_s2"], "target_running_core_rows": []}

## seed2/3 可复用的核心轴点

| job | seed | 轴/设置 | FAA | sample | AAA | 遗忘 |
|---|---:|---|---:|---:|---:|---:|
| s3_center_s2 | 2 | center | 53.419 | 60.700 | 66.746 | 6.287 |
| s3_lambda0_s2 | 2 | {"ewc_lambda": [2000, 0]} | 53.500 | 60.900 | 66.761 | 6.240 |
| s3_lambda200_s2 | 2 | {"ewc_lambda": [2000, 200]} | 53.405 | 60.570 | 66.715 | 6.153 |
| s3_lambda20000_s2 | 2 | {"ewc_lambda": [2000, 20000]} | 53.426 | 60.700 | 66.684 | 6.286 |
| s3_radius01_s2 | 2 | {"gam_grad_norm_rho": [0.2, 0.1]} | 52.462 | 59.580 | 66.052 | 6.358 |
| s3_radius05_s2 | 2 | {"gam_grad_norm_rho": [0.2, 0.5]} | 54.978 | 62.340 | 67.886 | 6.240 |
| s3_mech_a_nomerge_matched_s2 | 2 | {"in_task_weight_merge_start_task": [0, 10]} | 53.098 | 60.630 | 66.692 | 6.643 |
| s3_mech_a_off_s2 | 2 | {"in_task_weight_merge": [true, false]} | 52.150 | 59.510 | 67.164 | 8.203 |
| s3_mech_center_s3 | 3 | center | 52.006 | 59.970 | 67.511 | 8.332 |
| s3_mech_a_nomerge_matched_s3 | 3 | {"in_task_weight_merge_start_task": [0, 10]} | 51.935 | 59.970 | 67.451 | 8.167 |
| s3_mech_a_uniform_s2 | 2 | {"in_task_path_beta": [3.0, 0.0]} | 53.591 | 60.830 | 66.718 | 5.814 |
| s3_mech_a_latest_s2 | 2 | {"in_task_path_start_ratio": [0.0, 0.6]} | 53.298 | 60.430 | 66.644 | 6.164 |
| s3_mech_fisher_alpha0_s2 | 2 | {"ewc_fisher_damping_alpha": [0.2, 0.0]} | 53.431 | 60.630 | 66.774 | 6.317 |
| s3_mech_fisher_alpha1_s2 | 2 | {"ewc_fisher_damping_alpha": [0.2, 1.0]} | 53.353 | 60.760 | 66.793 | 6.327 |

## 源码与观测边界

两个冻结源码树共有 190 个父文件逐字节相同；机制 campaign 只增加 `utils/s3_mechanism_controls.py`。每次运行的 started/completed source manifest 均重新核对。

机制 campaign 开启 s3_boundary_capture；retained 分支调用原生 Fisher 一次，边界侧存档只复制因子，已有原生同状态 6 类等价性 witness 及每个已完成任务 RNG 检查通过。可与 native sweep 作科学配置复用，但不是历史原日志的逐位重现。此审计没有再次加载大体积 boundary .pt；其检查状态引用已完成 receipt。

原历史 seed2/3 中心 FAA 为 53.311/51.968；这里重跑中心为 53.419/52.006。保留原中心作历史复现参考；本轮新单轴变化优先相对本轮中心计算，不能把两次执行当成同一条结果。

## 需要保留的区别

- λ 是 ewc_lambda；lambda_flat 另属 GAM 方向强度，不能合并。
- 训练 r 对应 gam_grad_norm_rho；gam_grad_rho=.025 的在跑机制实验只改内层 ρ，不补外层 r 的缺点。
- α=0/1 是 ewc_fisher_damping_alpha 的单轴控制；γ=0/1 是历史 Fisher 累积，不能标为 damping α。
- 原生正 ω 可用 in_task_weight_merge_omega 单轴改变权重融合；这里两 campaign 尚无正 ω≠.3 的完成点。
- matched_no_merge 保留快照 CE；native ω=0 会跳过快照/CE。Aoff 可作为模块关闭对照，但本报告没有把其配置改名为执行过的 ω=0。
- 均匀 β=0 与后期 τ=.6 是权重路径构造设置，不是 ω 扫描。
- 旧 seed0 同时融合 Fisher 与权重的 ω 扫描不在这两个 campaign 内，不能当本中心纯权重 ω 扫描复用。

## 未完成记录

- s3_mech_fisher_gamma1_s2: running, 已见曲线阶段 6, 最后 task/epoch=(6, 30); 无最终指标。
- s3_mech_fisher_tasknorm_off_s2: running, 已见曲线阶段 2, 最后 task/epoch=(2, 47); 无最终指标。
- s3_mech_gam_projection_off_s2: running, 已见曲线阶段 1, 最后 task/epoch=(1, 47); 无最终指标。
- s3_mech_gam_rho025_s2: running, 已见曲线阶段 0, 最后 task/epoch=(0, 49); 无最终指标。
