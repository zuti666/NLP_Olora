# EWCLoRA / GAM-EWCLoRA 六数据集来源与协议核验

核验时间（UTC）：2026-09-18T19:31:59.545782+00:00。本文件只复核现有源文件，不训练、不修改 CopyFlatCV。

## 可填入表格的结果

FAA = 最终任务行的任务等权平均；AAA = 各阶段已见任务平均值再取阶段平均。以下 11 个完成单元格均直接从原始 cnn.matrices 复算，与来源表在 1e-8 内一致，完整下三角均为有限数。所有原始 metrics SHA 与已归档审计一致；9 份 legacy log SHA 同时一致。Cars/Flowers wrapper 日志已重新读取并记下本轮 SHA。每行均为单次运行，没有跨 seed 标准差。

| 方法 | 数据集 | FAA | AAA | 实现 | seed | λ | epochs/task | LR |
|---|---|---:|---:|---|---:|---:|---:|---:|
| EWCLoRA | cub200 | 60.21 | 69.97 | native one-pass | 0 | 200000.0 | 40 | 0.01 |
| EWCLoRA | cars196 | 29.54 | 39.74 | alpha-zero two-probe wrapper | 0 | 2000 | 40 | 0.05 |
| EWCLoRA | aircraft | 40.21 | 38.94 | native one-pass | 0 | 200.0 | 40 | 0.05 |
| EWCLoRA | flowers | 48.45 | 62.42 | alpha-zero two-probe wrapper | 0 | 2000 | 40 | 0.0025 |
| EWCLoRA | pets | 64.17 | 71.09 | native one-pass | 2 | 200000.0 | 40 | 0.0025 |
| EWCLoRA | het5 | 73.87 | 73.07 | native one-pass | 2 | 50000.0 | 30 | 0.02 |
| GAM-EWCLoRA | cub200 | 74.22 | 76.47 | historical GAM | 0 | 2000 | 40 | 0.01 |
| GAM-EWCLoRA | cars196 | 51.74 | 55.44 | historical GAM | 0 | 2000 | 40 | 0.05 |
| GAM-EWCLoRA | aircraft | 46.66 | 43.65 | historical GAM | 0 | 2000 | 40 | 0.05 |
| GAM-EWCLoRA | flowers | 86.75 | 91.76 | historical GAM | 0 | 2000 | 40 | 0.0025 |
| GAM-EWCLoRA | pets | 78.82 | 83.00 | historical GAM | 0 | 2000 | 40 | 0.0025 |
| GAM-EWCLoRA | het5 | -- | -- | new GAM, running | 0 | 2000 | 30 | .02 |

## 比较结论与必须披露的差异

这些记录可作有来源的 archived configurations 方法比较；不能描述为全表统一实现、统一 seed、完全受控或多 seed 的比较。当前数值不构成 GAM 对 EWC 的严格单因素消融。

- EWC 的 CUB/Aircraft/Pets/Het5 来自 native 单次 CE backward 分支；Cars/Flowers 来自 α=0 wrapper。后者方向中 shaping 系数为零，但仍执行 clean/perturbed 两次 probe，其额外 RNG 消耗和扰动/恢复操作可能改变轨迹。不能把整行统称 native EWC。
- EWC λ 按 CUB/Cars/Aircraft/Flowers/Pets/Het5 为 200000/2000/200/2000/200000/50000；GAM 六项 λ=2000。Pets、Het5 的 EWC seed=2，其余完成记录 seed=0。Cars/Flowers 同 λ、同 seed 仍不能消除实现、RNG、数据/版本归档差异。
- 五个独立数据集均40 epochs/task；EWC、GAM 的 Het5 为30。其他 LoRA 方法中 InfLoRA/OLoRA/SDLoRA 的 Het5 是40，因此整张7方法表的 Het5 训练预算仍不同。
- EWC native 日志明确 batch_empirical、ΔW Fisher、per-task trace normalize=true、accumulated normalize=false、damping=0、gamma=1、eta=0、max_batches=100、preserve_fisher_rng=true。它是 batch-gradient-square proxy，不能称为逐样本 Fisher 或 model-label true Fisher。
- 两项历史 EWC wrapper 记录同样明确 batch_empirical、gamma=1、eta=0、max_batches=100、每任务trace归一化、累积不归一化和零阻尼；没有记录 probe/Fisher RNG 隔离开关，不能借用后续配置补称已隔离。
- 历史 GAM 日志明确 λ=2000、gamma=1、eta=0、max_batches=100、ewc_normalize_fisher=true、lambda_flat=1，无 merge。旧日志未打印 fisher_mode、accumulated-normalization/damping/RNG 开关，当前源码默认值不能冒充历史运行见证。
- 历史 GAM 的 (grad_rho, grad_norm_rho, grad_gamma) 在 CUB 为(.2,.2,.1)，Cars为(.05,.2,.03)，Aircraft为(.01,.1,.02)，三者beta1/2/3=1。Flowers/Pets 日志及对应已归档配置缺少这些字段，数值应标未记录，不用当前默认值填充。
- 所有11项已读取日志都记录同一 checkpoint hub ID：timm/vit_base_patch16_224.augreg2_in21k_ft_in1k；rank16、batch128、SGD(momentum=0, weight_decay=0)、cosine、class_shuffle=false、无回放。相同 hub ID 仍不能证明历史 revision/文件SHA完全相同。
- 历史代码/数据清单未全部与运行不可变绑定；新 CL-LoRA/GAM-Het5 用 /data/115-2/users/liying/datasets 和锁定图片清单/预训练SHA。当前副本不能独自证明与历史逐图片一致，尤其 Flowers loader 从目录索引改为官方 image ID 映射。
- 原 native Cars λ80000 运行最后任务出现非有限训练，仍排除。表内复用 λ2000 wrapper，不能表述为已修复并重跑 native Cars。Cars λ80000、Flowers λ10 的较高 wrapper 点源于测试 sweep，不替换固定 λ2000格来选最高分。

## GAM-EWC / Het5 当前状态

截至 2026-09-18T19:31:59.545782+00:00，state 和 receipt 均为 running，记录 PID 2922262、GPU8。控制台最近已完成 Task 1（第2/5任务）Epoch 19/30，Loss=0.699，Train_accy=83.55。控制台 mtime=2026-09-18T19:28:04.461782+00:00。

目前 metrics 只包含 t00，t00=[[87.19]]。这只是第一任务结果，不是完整五任务 FAA/AAA；待5阶段完成并成功 receipt 后才可填表。

本工具子进程处于仅3个进程可见的隔离 /proc，本地 ps 看不到宿主PID，不能因此判断训练已停止；运行判断依据宿主运行记录与持续更新日志。

- [实时 receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/receipts/gam_ewc_het5_s0.json)
- [配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/gam_ewc_het5_s0.json)
- [进度日志](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/logs/gam_ewc_het5_s0.log)
- [当前部分矩阵](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/ewclora_normfisher_gam/fine_gapfill_20260918/het5/0/gam_ewc_het5_s0/exp_run/100/gam_ewc_het5_s0_vit_base_patch16_224_cl_metrics.json)

新 Het5 固定 rho=.2/norm_rho=.2/gamma=.1/beta1/2/3=1；Fisher batch_empirical、λ2000、gamma1、eta0、max_batches100、每任务trace归一化、累积不归一化、damping0、无merge；ewc_preserve_fisher_rng=true、ewc_isolate_clean_rng=true。它是新seed0协议，不能冒充历史seed0复现。

## 逐格实际来源与配置

### EWCLoRA / cub200

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200000_beta000/cub200/0/cub200_native_ewc_rho005_lam200000_beta000_s0_fivearm_native_v1_20260826/exp_run/20/cub200_native_ewc_rho005_lam200000_beta000_s0_fivearm_native_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200000_beta000/cub200/0/cub200_native_ewc_rho005_lam200000_beta000_s0_fivearm_native_v1_20260826/exp_run/20/cub200_native_ewc_rho005_lam200000_beta000_s0_fivearm_native_v1_20260826_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/cub200_native_ewc_rho005_lam200000_beta000_s0.yaml)
- metrics SHA256: `f570c5dafd51744ca41637f1b0930f728c0e7cfb4e4d8fbd4611e26ec81bf6be`
- log SHA256: `1cd4985077d0f596fb029988836a243b84bf59d84e7b08752ed67997466d8d70`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_direction_mode": "native_sgd",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.01",
  "ewc_lambda": "200000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "fs_preserve_probe_rng": "True"
}
```

### EWCLoRA / cars196

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam2000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam2000_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/cars196/0/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam2000/exp_run/20/as1fs_cars196_sgdeq_r16_lr005_e40_ewc_lam2000_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep/cars196_sgd_ewc_lam2000.yaml)
- metrics SHA256: `fa5be60fead0a7f4a990054950717e379304a58cb78948a167a77701b32b7e34`
- log SHA256: `82283cd798af793565e9806119ca4a6788f548ff252c3ff54a38dce5e749c6be`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.05",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0"
}
```

### EWCLoRA / aircraft

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200_beta000/aircraft/0/aircraft_native_ewc_rho005_lam200_beta000_s0_fivearm_native_v1_20260826/exp_run/10/aircraft_native_ewc_rho005_lam200_beta000_s0_fivearm_native_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200_beta000/aircraft/0/aircraft_native_ewc_rho005_lam200_beta000_s0_fivearm_native_v1_20260826/exp_run/10/aircraft_native_ewc_rho005_lam200_beta000_s0_fivearm_native_v1_20260826_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/aircraft_native_ewc_rho005_lam200_beta000_s0.yaml)
- metrics SHA256: `2166cdab4c6ced37bc56f900e3a158775591b410f63c45eb242dec33f3f1d95d`
- log SHA256: `a44329c456a4ff6d7f12c66e9e51dbd389cde054b4d7eae41bffb4db1b383034`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_direction_mode": "native_sgd",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.05",
  "ewc_lambda": "200.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "fs_preserve_probe_rng": "True"
}
```

### EWCLoRA / flowers

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam2000/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam2000_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam2000/flowers/0/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam2000/exp_run/10/as1fs_flowers102_sgdeq_r16_lr00025_e40_ewc_lam2000_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep/flowers102_sgd_ewc_lam2000.yaml)
- metrics SHA256: `b2b26c8b616a8986f2cf2d61decc98b044209c4c16d5495794a11e4d1f12fa6f`
- log SHA256: `e5702c1024b63f9368e6ee115402af9e4173b8612ef56cbf1b3e194dd85b7c2f`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.0025",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0"
}
```

### EWCLoRA / pets

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200000_beta000/pets/2/oxford_pet_native_ewc_rho005_lam200000_beta000_s2_fivearm_native_v1_20260826/exp_run/4/oxford_pet_native_ewc_rho005_lam200000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam200000_beta000/pets/2/oxford_pet_native_ewc_rho005_lam200000_beta000_s2_fivearm_native_v1_20260826/exp_run/4/oxford_pet_native_ewc_rho005_lam200000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/oxford_pet_native_ewc_rho005_lam200000_beta000_s2.yaml)
- metrics SHA256: `9d35e2e7e97f168dc9c2de7057a77856fb2eb75711738ff672a83861d3179aab`
- log SHA256: `0663bbdf0e1707c99f64d8057d3eda363cc7d048ca1de39a292e5fc25ebb4e84`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_direction_mode": "native_sgd",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "2",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.0025",
  "ewc_lambda": "200000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "fs_preserve_probe_rng": "True"
}
```

### EWCLoRA / het5

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/het5_native_ewc_rho005_lam50000_beta000_s2.yaml)
- metrics SHA256: `3d19fa0b0daaeef68a0959a9ed1db66a9bdb68767b2d2054d663ddee07e295ca`
- log SHA256: `afa87ddb36e20fe92d8f212183dd996beb08bd14e755f3a544b579b856cfe975`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "as1_normfisher_fishershape",
  "optimizer": "sgd",
  "optimizer_type": "fishershape",
  "fs_direction_mode": "native_sgd",
  "fs_alpha": "0.0",
  "fs_sam_equivalent": "False",
  "seed": "2",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "30",
  "lrate": "0.02",
  "ewc_lambda": "50000.0",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "fisher_mode": "batch_empirical",
  "ewc_normalize_fisher": "True",
  "ewc_normalize_accumulated_fisher": "False",
  "ewc_fisher_damping_beta": "0.0",
  "ewc_preserve_fisher_rng": "True",
  "fs_preserve_probe_rng": "True"
}
```

### GAM-EWCLoRA / cub200

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cub200/0/paperA_as1normfisher_lam2000_cub200_t20/exp_run/20/paperA_as1normfisher_lam2000_cub200_t20_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cub200/0/paperA_as1normfisher_lam2000_cub200_t20/exp_run/20/paperA_as1normfisher_lam2000_cub200_t20_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp_paperA_newmethod/as1_normfisher_lam2000_cub200_t20.yaml)
- metrics SHA256: `aa2cfb4d1b7c68af6d569fa68b734e51804f93f45e35d5d8e20a0650f724c80c`
- log SHA256: `5448f71b6c4c4a86f1951459ef94600c038519f404530d2ab498d6bb79880536`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "ewclora_normfisher_gam",
  "optimizer": "sgd",
  "optimizer_type": "gam",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.01",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "ewc_normalize_fisher": "True",
  "gam_grad_rho": "0.2",
  "gam_grad_norm_rho": "0.2",
  "gam_grad_gamma": "0.1",
  "gam_grad_beta_1": "1.0",
  "gam_grad_beta_2": "1.0",
  "gam_grad_beta_3": "1.0"
}
```

### GAM-EWCLoRA / cars196

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cars196/0/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0/exp_run/20/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cars196/0/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0/exp_run/20/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_cars196_t10c20_r16.yaml)
- metrics SHA256: `c61f7c415f63eccb1209da4e844dc3bb13ea0bfecbbdcbb8b20dd4ac1b0f0247`
- log SHA256: `3077c35a4a9cb58f18ac398c1ed8a695e867a45e23240bf433b1b2d210aaba8a`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "ewclora_normfisher_gam",
  "optimizer": "sgd",
  "optimizer_type": "gam",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.05",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "ewc_normalize_fisher": "True",
  "gam_grad_rho": "0.05",
  "gam_grad_norm_rho": "0.2",
  "gam_grad_gamma": "0.03",
  "gam_grad_beta_1": "1",
  "gam_grad_beta_2": "1.0",
  "gam_grad_beta_3": "1.0"
}
```

### GAM-EWCLoRA / aircraft

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/aircraft/0/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/aircraft/0/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_aircraft_t10c10_r16.yaml)
- metrics SHA256: `b624e99fb2d1bbd08118bd1cfd1616e759e50eb18f4641c74a9ed54c38d20280`
- log SHA256: `44b7a209173a038deae9e53675673c47d86cc7a32777f9d118dc554d63b8e393`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "ewclora_normfisher_gam",
  "optimizer": "sgd",
  "optimizer_type": "gam",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.05",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "ewc_normalize_fisher": "True",
  "gam_grad_rho": "0.01",
  "gam_grad_norm_rho": "0.1",
  "gam_grad_gamma": "0.02",
  "gam_grad_beta_1": "1",
  "gam_grad_beta_2": "1.0",
  "gam_grad_beta_3": "1.0"
}
```

### GAM-EWCLoRA / flowers

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/flowers/0/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/flowers/0/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_flowers_t10c10_r16.yaml)
- metrics SHA256: `fd031037c95bd89b137e6926d95f83bf591aa87dac95e70fa98a7bb2878f6f92`
- log SHA256: `7b0665b2a3d808bc3a0664437eb659674152e4de34c712c82387f97d60f5500c`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "ewclora_normfisher_gam",
  "optimizer": "sgd",
  "optimizer_type": "gam",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.0025",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "ewc_normalize_fisher": "True"
}
```

### GAM-EWCLoRA / pets

- [原始任务矩阵](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/pets/0/paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0/exp_run/4/paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0_vit_base_patch16_224_cl_metrics.json)
- [实际日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/pets/0/paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0/exp_run/4/paperA_as1normfisher_lam2000_oxfordpet_t9c4_r16_s0_vit_base_patch16_224.log)
- [对应现存配置](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_oxfordpet_t9c4_r16.yaml)
- metrics SHA256: `11d7cad65557282ecebef86e2d94ce23b55a3440dba62320d6aa873519afc9d4`
- log SHA256: `445d2a58da2ae5f486d53b9eef491686d178ebfd430e396672af78bcafe9f878`

日志参数（未出现字段不补默认值）：

```json
{
  "model_name": "ewclora_normfisher_gam",
  "optimizer": "sgd",
  "optimizer_type": "gam",
  "seed": "0",
  "lora_rank": "16",
  "batch_size": "128",
  "epochs": "40",
  "lrate": "0.0025",
  "ewc_lambda": "2000",
  "ewc_gamma": "1.0",
  "ewc_eta": "0.0",
  "ewc_max_batches": "100",
  "ewc_normalize_fisher": "True"
}
```

## 上游证据

- [FINE_NLP_TABLE_AUDIT_20260918.md](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/FINE_NLP_TABLE_AUDIT_20260918.md)
- [EWC_CARS_FLOWERS_WIKI_AUDIT_20260918.md](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/EWC_CARS_FLOWERS_WIKI_AUDIT_20260918.md)
- [PROTOCOL_ALIGNMENT_AND_SEED0_PLAN_20260918.md](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/PROTOCOL_ALIGNMENT_AND_SEED0_PLAN_20260918.md)
- [FINE_GAPFILL_EXECUTION_20260918.md](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/FINE_GAPFILL_EXECUTION_20260918.md)
