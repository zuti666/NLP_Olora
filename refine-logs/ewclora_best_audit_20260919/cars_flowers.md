# Cars / Flowers EWCLoRA 最优值审计（2026-09-19）

结论：当前表中的 Cars **29.54/39.74**、Flowers **48.45/62.42** 是历史 seed0、固定 λ=2000 的 α=0 wrapper 参考值，**不是全部运行最优，也不是同一历史 sweep 的 FAA 最优**。这里的“最好”仅指本次可定位完整有限的 CNN 结果；没有穷尽所有机器、所有配置。AAA 使用每阶段任务宏平均的阶段均值，不能将遗忘率、样本加权平均或 NME 混入。

## 同一历史 sweep / seed0 比较

| 数据集 | 选择依据 | λ | FAA | 同一运行 AAA |
|---|---|---:|---:|---:|
| Cars | 当前表，固定参考 | 2000 | 29.543 | 39.735614 |
| Cars | FAA 与 AAA 均最高 | 80000 | 30.697 | 40.681111 |
| Flowers | 当前表，固定参考 | 2000 | 48.454 | 62.420013 |
| Flowers | FAA 最高 | 10 | 48.652 | 62.408981 |
| Flowers | AAA 最高 | 200 | 48.576 | 62.422186 |

Flowers 的 AAA 最大点与当前点仅相差 0.002173 个百分点，显示两位小数均为 62.42。不能因为显示精度相同就认定它们原始值完全相等，也不能把 λ10 的 FAA 和 λ200 的 AAA 拼成一条实验。

共同骨架为 ViT-B/16，rank16，batch128，40 epochs/task，SGD（momentum=0，weight decay=0）+cosine，无回放、自然类顺序。Cars 为 16+9×20 类、LR=.05；Flowers 为 12+9×10 类、LR=.0025。约束 Q/V 有效 ΔW，Fisher gamma=1、eta=0、batch_empirical、max_batches=100、每任务 trace 归一化、累积后不归一化、damping=0。历史 sweep 未记录 probe/Fisher RNG 隔离开关；可比较为同归档 sweep，但不宣称运行源码/权重/数据逐字节同一性。

## 放宽到已发现的其他实现族和 seeds

| 数据集 | 最高指标 | 实现族 | λ | seed | FAA | 同一运行 AAA |
|---|---|---|---:|---:|---:|---:|
| Cars | FAA 与 AAA | α=0，probe/Fisher RNG 均隔离 | 80000 | 1 | 34.555 | 43.942118 |
| Flowers | FAA | α=0，probe/Fisher RNG 均隔离 | 10 | 1 | 53.434 | 65.449362 |
| Flowers | AAA | α=0，仅记录 Fisher RNG 隔离 | 20000 | 0 | 52.199 | 65.924706 |

这些是跨配置族/seed 的观察最大值，不能静默替换进“相同 seed0/统一实现”的表中。既然参数/seed 选择看过测试分数，应标为历史探索性 test-selected best，而不能作为独立验证选参的无偏最终结果。更合适的是保留固定参考主表，并另列同 family 的多 seed 均值与探索性 sweep 表。

严格 native one-pass：本次找到 Cars λ80000 seed0 一项，但训练非有限，排除；未找到 Flowers 完整 native metrics。因此这里更高的有效点全部属于 α=0 wrapper，并非 native one-pass 新结果。

## 搜索范围和完成数

本次重查 CopyFlatCV 的 `outputs_logs`、`logs_inc_lora`、`refine-logs`：共有 **79** 个 Cars/Flowers metrics 文件；原审计命名过滤命中 34 个。对全部 79 个存在的伴随日志扩大核查后，没有找到 26 条清单之外、同时满足正 λ、α=0、非 SAM 的额外 native/wrapper 记录。79 个中另有 GAM、FS、SAM、λ0 和 CL-LoRA 等，不计入 EWCLoRA。

| 数据集/族 | 参数范围 | 已有记录 | 完整有限 |
|---|---|---:|---:|
| Cars 历史 sweep | seed0；λ∈{200,2000,20000,40000,80000,160000} | 6 | 5 |
| Flowers 历史 sweep | seed0；λ∈{1,10,50,200,2000,20000} | 6 | 6 |
| Cars 双 RNG 隔离 α0 | λ80000；seeds1,2 | 2 | 2 |
| Flowers 双 RNG 隔离 α0 | λ10；seeds1,2 | 2 | 2 |
| Flowers Fisher RNG 隔离 α0 | λ∈{10,20000}×seeds0,1,2；damping0 | 6 | 6 |
| Flowers Fisher RNG 隔离 α0+damping | λ10×seeds0,1,2；damping.2 | 3 | 3 |
| Cars native one-pass | λ80000；seed0 | 1 | 0 |
| 合计 | Cars 9 条、Flowers 17 条 | 26 | 24 |

排除的两条是 Cars 历史 λ160000 和 native λ80000，均在训练日志出现非有限 loss；即使它们写出最终矩阵，也不能列为有效完成。检索到的配置不代表笛卡尔积里所有 λ×seed 都跑过。

另外检查 Flatness_CV 的 Cars/Flowers EWC 命名结果，发现的是 GAM-EWCLoRA 与 `ewcon` 全模型 EWC 家族；未发现额外 plain-EWCLoRA/α0 候选。全模型 EWC 不并入本 LoRA 表。

## 独立验证

- 全部 26 个原始 metrics 文件 SHA256 与昨日 per_run 清单一致。
- 从原始 CNN final task matrix 重新计算 FAA、AAA-stage，全部与 per_run 一致。
- 从原始训练日志重新核查 λ、seed、10任务数与非有限状态，26条全部一致。
- 未启动、修改或重跑实验；不替换主表。

## 最高点的原始来源

- cars196 / alpha0_historical_sweep / best faa: λ=80000, seed=0, FAA=30.697000, AAA=40.681111；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam80000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam80000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000_vit_base_patch16_224.log)。
- cars196 / alpha0_historical_sweep / best aaa_stage: λ=80000, seed=0, FAA=30.697000, AAA=40.681111；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam80000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam80000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam80000_vit_base_patch16_224.log)。
- cars196 / ALL / best faa: λ=80000, seed=1, FAA=34.555000, AAA=43.942118；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)。
- cars196 / ALL / best aaa_stage: λ=80000, seed=1, FAA=34.555000, AAA=43.942118；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)。
- flowers / alpha0_historical_sweep / best faa: λ=10, seed=0, FAA=48.652000, AAA=62.408981；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round3/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam10/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam10/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam10_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round3/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam10/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam10/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam10_vit_base_patch16_224.log)。
- flowers / alpha0_historical_sweep / best aaa_stage: λ=200, seed=0, FAA=48.576000, AAA=62.422186；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam200/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam200_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam200/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam200_vit_base_patch16_224.log)。
- flowers / ALL / best faa: λ=10, seed=1, FAA=53.434000, AAA=65.449362；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)。
- flowers / ALL / best aaa_stage: λ=20000, seed=0, FAA=52.199000, AAA=65.924706；[原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224_cl_metrics.json)；[日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224.log)。

## 26 条明细

| 数据集 | family | λ | seed | damping | FAA | AAA-stage | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| cars196 | alpha0_both_rng_preserved | 80000 | 1 | 0.0 | 34.555000 | 43.942118 | verified |
| cars196 | alpha0_both_rng_preserved | 80000 | 2 | 0.0 | 32.144000 | 42.481264 | verified |
| flowers | alpha0_both_rng_preserved | 10 | 1 | 0.0 | 53.434000 | 65.449362 | verified |
| flowers | alpha0_both_rng_preserved | 10 | 2 | 0.0 | 51.258000 | 65.023630 | verified |
| flowers | alpha0_fisher_rng_preserved | 10 | 0 | 0.0 | 52.182000 | 65.913043 | verified |
| flowers | alpha0_fisher_rng_preserved | 10 | 1 | 0.0 | 49.084000 | 63.971602 | verified |
| flowers | alpha0_fisher_rng_preserved | 10 | 2 | 0.0 | 47.992000 | 64.069940 | verified |
| flowers | alpha0_fisher_rng_preserved | 20000 | 0 | 0.0 | 52.199000 | 65.924706 | verified |
| flowers | alpha0_fisher_rng_preserved | 20000 | 1 | 0.0 | 49.027000 | 63.956758 | verified |
| flowers | alpha0_fisher_rng_preserved | 20000 | 2 | 0.0 | 47.972000 | 64.066371 | verified |
| flowers | alpha0_fisher_rng_preserved_damping02 | 10 | 0 | 0.2 | 52.182000 | 65.910293 | verified |
| flowers | alpha0_fisher_rng_preserved_damping02 | 10 | 1 | 0.2 | 49.056000 | 63.968802 | verified |
| flowers | alpha0_fisher_rng_preserved_damping02 | 10 | 2 | 0.2 | 47.906000 | 64.064229 | verified |
| cars196 | alpha0_historical_sweep | 200 | 0 | 0.0 | 28.875000 | 39.263955 | verified |
| cars196 | alpha0_historical_sweep | 2000 | 0 | 0.0 | 29.543000 | 39.735614 | verified |
| cars196 | alpha0_historical_sweep | 20000 | 0 | 0.0 | 30.094000 | 40.167788 | verified |
| cars196 | alpha0_historical_sweep | 40000 | 0 | 0.0 | 30.505000 | 40.607637 | verified |
| cars196 | alpha0_historical_sweep | 80000 | 0 | 0.0 | 30.697000 | 40.681111 | verified |
| cars196 | alpha0_historical_sweep | 160000 | 0 | 0.0 | -- | -- | nonfinite_training |
| flowers | alpha0_historical_sweep | 1 | 0 | 0.0 | 48.479000 | 62.406945 | verified |
| flowers | alpha0_historical_sweep | 10 | 0 | 0.0 | 48.652000 | 62.408981 | verified |
| flowers | alpha0_historical_sweep | 50 | 0 | 0.0 | 48.494000 | 62.415000 | verified |
| flowers | alpha0_historical_sweep | 200 | 0 | 0.0 | 48.576000 | 62.422186 | verified |
| flowers | alpha0_historical_sweep | 2000 | 0 | 0.0 | 48.454000 | 62.420013 | verified |
| flowers | alpha0_historical_sweep | 20000 | 0 | 0.0 | 48.465000 | 62.420940 | verified |
| cars196 | native_one_pass | 80000 | 0 | 0.0 | -- | -- | nonfinite_training |

[原 per_run 清单](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/ewc_cars_flowers_wiki_20260918/per_run.json)；[此前 Cars/Flowers 来源审计](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/EWC_CARS_FLOWERS_WIKI_AUDIT_20260918.md)。
