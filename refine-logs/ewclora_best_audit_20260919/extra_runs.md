# EWCLoRA 跨目录补充发现（2026-09-19）

当前表不是本地所有配置、seed、wrapper 版本中的数值最大值。下列是探索性候选，不能把不同seed或配置分别挑出的FAA与AAA拼成一条真实结果。

## 搜索与核验范围

只读扫描两个项目全目录下 3009 份 `_cl_metrics.json`：CopyFlatCV 1636，Flatness_CV 1373。跳过 GAM-EWC/evidence 中191份归档副本；原始候选须有伴随日志。日志实际dataset落入六数据集的共有625份。实际参数识别出77个正λ EWC候选，其中47个未在既有vision_inventory/selected、Cars/Flowers专项或当前comparison来源集合中。

检查CNN完整任务矩阵、下三角有限值、日志最后CNN分任务准确率逐项一致、最后任务final epoch、训练loss非有限标记；明确排除GAM-EWC、正alpha的FS-EWC、SAM+EWC和全参数ewcon。仅以CNN为准。历史任务缺乏统一成功receipt，不把这些检查冒充完整run provenance。

## 探索性数值上界：每项保留来自同一次运行的一对值

| 数据集 | FAA最大候选：FAA / AAA | 实现/seed/λ | AAA最大候选：FAA / AAA | 实现/seed/λ |
|---|---:|---|---:|---|
| cub200 | 60.406000 / 69.399929 | alpha0_wrapper; s0; λ200000 | 60.160000 / 70.647066 | alpha0_wrapper; s1; λ200000 |
| cars196 | 34.555000 / 43.942118 | alpha0_wrapper; s1; λ80000 | 34.555000 / 43.942118 | alpha0_wrapper; s1; λ80000 |
| aircraft | 41.832000 / 39.695065 | alpha0_wrapper; s0; λ200 | 41.321000 / 39.757967 | alpha0_wrapper; s0; λ2000 |
| flowers | 53.434000 / 65.449362 | alpha0_wrapper; s1; λ10 | 52.199000 / 65.924706 | alpha0_wrapper; s0; λ20000 |
| pets | 66.425556 / 73.712628 | alpha0_wrapper; s0; λ200000 | 66.375556 / 73.739711 | alpha0_wrapper; s0; λ20000 |
| het5 | 73.868000 / 73.067933 | native; s2; λ50000 | 73.072000 / 73.074567 | alpha0_wrapper; s1; λ50000 |

这里的“最大”只限定上述已发现且未见已知失效问题的本地候选；不是独立验证选择的无偏最终性能，更不是全局最优。两列可指向不同run，主表不能只取两列最大分数拼接。

## Het5 纠正与风险隔离

Het5 λ100000/200000 的现存JSON虽然完整且最后CNN日志一致，但既有审计证明并发共享输出路径，无法恢复唯一运行绑定。因此两项均隔离，73.892/73.3039与75.288/74.1696不能作为可信best反例。扫描没有发现这两个高λ的唯一路径复跑。其余Het5最高FAA为已有native seed2 λ50000的73.868/73.067933；α0 seed1 λ50000的AAA=73.074567略高，但FAA=73.072更低，不能用来宣称整体改善。

- [既有RACE证据](/data/140-1/users/liying/CopyFlatCV/newmethod_design_wiki/FISHERSHAPE_5DATASET_HET5_AUDIT_AND_PLAN.md:42)

CUB/ Aircraft的旧lambda sweep与Pets的RNG隔离诊断是更高单次值的主要来源。Pets λ200与2000的旧sweep存在bit-identical审计提示，未作为本次最大候选；Pets较高的新诊断在Fisher RNG/阻尼等字段有变化，不是native单因素复现。

## 上界候选来源与关键差异

### cub200 — alpha0_wrapper / seed0 / λ200000

FAA=60.406000，AAA(stage)=69.399929；rank16、batch128、40 epochs/task、LR=0.01。日志时间 2026-08-11 07:13:39,611 至 2026-08-11 11:03:38,750。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200000/cub200/0/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000/exp_run/20/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200000/cub200/0/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000/exp_run/20/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep_round2/cub200_sgd_ewc_lam200000.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "200000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "mechanism_eval": "False"
}
```

### cub200 — alpha0_wrapper / seed1 / λ200000

FAA=60.160000，AAA(stage)=70.647066；rank16、batch128、40 epochs/task、LR=0.01。日志时间 2026-08-27 02:31:47,285 至 2026-08-27 04:16:54,959。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam200000_beta000/cub200/1/cub200_g0_ewc_c000_rho005_lam200000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cub200_g0_ewc_c000_rho005_lam200000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam200000_beta000/cub200/1/cub200_g0_ewc_c000_rho005_lam200000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cub200_g0_ewc_c000_rho005_lam200000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/cub200_g0_ewc_c000_rho005_lam200000_beta000_s1.yaml)

```json
{
  "fs_direction_mode": "alpha0",
  "fs_alpha": "0.0",
  "ewc_lambda": "200000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "fs_preserve_probe_rng": "True",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### cars196 — alpha0_wrapper / seed1 / λ80000

FAA=34.555000，AAA(stage)=43.942118；rank16、batch128、40 epochs/task、LR=0.05。日志时间 2026-08-27 13:00:29,298 至 2026-08-27 15:00:03,753。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1.yaml)

```json
{
  "fs_direction_mode": "alpha0",
  "fs_alpha": "0.0",
  "ewc_lambda": "80000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "fs_preserve_probe_rng": "True",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### aircraft — alpha0_wrapper / seed0 / λ200

FAA=41.832000，AAA(stage)=39.695065；rank16、batch128、40 epochs/task、LR=0.05。日志时间 2026-08-11 14:13:59,300 至 2026-08-11 17:10:45,942。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep/aircraft_sgd_ewc_lam200.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "200",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "mechanism_eval": "False"
}
```

### aircraft — alpha0_wrapper / seed0 / λ2000

FAA=41.321000，AAA(stage)=39.757967；rank16、batch128、40 epochs/task、LR=0.05。日志时间 2026-08-11 17:10:49,819 至 2026-08-11 20:06:00,827。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam2000/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam2000_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam2000/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam2000_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep/aircraft_sgd_ewc_lam2000.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "mechanism_eval": "False"
}
```

### flowers — alpha0_wrapper / seed1 / λ10

FAA=53.434000，AAA(stage)=65.449362；rank16、batch128、40 epochs/task、LR=0.0025。日志时间 2026-08-27 16:37:54,492 至 2026-08-27 17:26:33,905。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1.yaml)

```json
{
  "fs_direction_mode": "alpha0",
  "fs_alpha": "0.0",
  "ewc_lambda": "10.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "fs_preserve_probe_rng": "True",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### flowers — alpha0_wrapper / seed0 / λ20000

FAA=52.199000，AAA(stage)=65.924706；rank16、batch128、40 epochs/task、LR=0.0025。日志时间 2026-08-25 05:24:16,941 至 2026-08-25 06:12:57,148。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_rngctl_diagnostics/flowers102_sgd_ewc_rngctl_fisheron_lam20000.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "20000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### pets — alpha0_wrapper / seed0 / λ200000

FAA=66.425556，AAA(stage)=73.712628；rank16、batch128、40 epochs/task、LR=0.0025。日志时间 2026-08-25 06:13:56,296 至 2026-08-25 07:05:23,675。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam200000_s012/pets/0/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam200000_s012/exp_run/4/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam200000_s012_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam200000_s012/pets/0/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam200000_s012/exp_run/4/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam200000_s012_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_rngctl_diagnostics/oxford_pet_sgd_ewc_rngctl_fisheron_lam200000.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "200000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### pets — alpha0_wrapper / seed0 / λ20000

FAA=66.375556，AAA(stage)=73.739711；rank16、batch128、40 epochs/task、LR=0.0025。日志时间 2026-08-25 05:23:59,822 至 2026-08-25 06:15:31,990。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/pets/0/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/4/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_rngctl_diagnostics_20260825/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_rngctl_mech_fisheron_lam20000_s012/pets/0/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012/exp_run/4/as1fs_oxford_pet_sgdeq_r16_lr00025_e40_rngctl_mech_fisheron_lam20000_s012_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_rngctl_diagnostics/oxford_pet_sgd_ewc_rngctl_fisheron_lam20000.yaml)

```json
{
  "fs_alpha": "0.0",
  "ewc_lambda": "20000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### het5 — native / seed2 / λ50000

FAA=73.868000，AAA(stage)=73.067933；rank16、batch128、30 epochs/task、LR=0.02。日志时间 2026-08-30 23:30:39,506 至 2026-08-31 02:20:55,549。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/het5_native_ewc_rho005_lam50000_beta000_s2.yaml)

```json
{
  "fs_direction_mode": "native_sgd",
  "fs_alpha": "0.0",
  "ewc_lambda": "50000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "fs_preserve_probe_rng": "True",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

### het5 — alpha0_wrapper / seed1 / λ50000

FAA=73.072000，AAA(stage)=73.074567；rank16、batch128、30 epochs/task、LR=0.02。日志时间 2026-08-26 18:31:25,005 至 2026-08-26 22:46:54,991。

- [CNN原始矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam50000_beta000/het5/1/het5_g0_ewc_c000_rho005_lam50000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/1/het5_g0_ewc_c000_rho005_lam50000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam50000_beta000/het5/1/het5_g0_ewc_c000_rho005_lam50000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/1/het5_g0_ewc_c000_rho005_lam50000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log)
- [配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/het5_g0_ewc_c000_rho005_lam50000_beta000_s1.yaml)

```json
{
  "fs_direction_mode": "alpha0",
  "fs_alpha": "0.0",
  "ewc_lambda": "50000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_skip_task_fisher": "False",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "fs_preserve_probe_rng": "True",
  "ewc_preserve_fisher_rng": "True",
  "mechanism_eval": "True"
}
```

## 机器可读索引与限制

- [全部77个候选、参数、状态、SHA与路径](/data/140-1/users/liying/O-LoRA/refine-logs/ewclora_best_audit_20260919/extra_candidates.json)
- 不读取远端/其他项目、已删除输出、非标准命名指标或只有console没有matrix的运行；因此不声称穷尽所有历史实验。
- 不复跑模型；日志缺失的非有限事件、其他未记录的输出竞态、训练时源码/数据绑定不能靠当前文件完全排除。
- α0-wrapper与native方向等价但执行路径、RNG消耗不同；更高wrapper分数只说明当前表不是所有EWC变体/设置的最大值，不能默认为相同实现的提升。
- 本次未修改任何主表或训练配置，没有启动或停止任务。
