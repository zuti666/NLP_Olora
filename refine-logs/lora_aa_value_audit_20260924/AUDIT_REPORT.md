# 当前 LoRA AA 图数值核查（2026-09-24）

核查对象：`lora_taskwise_aa_compact_tighty.pdf` 的 InfLoRA 和 FS-EWC 曲线。只检查结果来源、统计口径和绘图数据，没有重新运行模型评估，也没有修改原图或原始日志。

## 结论

1. 图中 InfLoRA 四个数据集的曲线与指定 seed0 原始日志一致。原始日志、导出 CSV 和实际 SVG 坐标相互吻合。两种方法共核查 70 个绘图点，SVG 反算误差不超过 3.60e-7 个百分点（坐标序列化舍入）。
2. 当前图的每个点是当前阶段已见任务的等权平均准确率。它不是日志 global top1，也不是累计 AAA 曲线。两方法在本图内部使用相同计算定义。
3. FS-EWC 是 `GAM-EWC+Merge` 的显示别名。ImageNet-R/A 加载 seed 0–5 六次完整运行；实际日志/机制记录对应历史 WMergeThenFisher 变体，不能仅凭标签把它当作其他 FS-EWCLoRA 实现。
4. InfLoRA 是 seed0 单次结果，细带为固定 ±0.2 pp 示意带；FS 使用六 seed 均值±样本标准差。重复次数和训练配置未对齐，数值差值只是这批历史运行之间的差值。
5. 当前图、同一批日志的样本加权 top1、以及 sources.json 中的旧表快照，三者的分数和方法差值不完全一致。原因包括统计口径与运行集合不同；没有为匹配旧表而调整曲线。

## 指标定义

- 当前图：`AA_t = mean(a[t, 1:t])`；本报告任务等权 FAA=`AA_T`；AAA=`mean(AA_1,...,AA_T)`。
- 日志 global top1：各阶段所有测试样本汇总的分类准确率；FAA 为最后阶段 top1；AAA 为各阶段 global top1 的均值。
- 百分比单位；差值是百分点。多 seed 先在每次运行内计算，再汇总均值与样本标准差（ddof=1）。
- 部分 cl_metrics.json 的同名 AAA 是整个下三角矩阵元素的均值；不能直接代替本报告按阶段平均的 AAA。

## 当前图对应数值（任务等权）

| 数据集 | 方法 | n | FAA=AA_T | AAA=mean(AA_t) |
|---|---|---:|---:|---:|
| ImageNet-R | InfLoRA | 1 | 76.793000 | 82.624822 |
| ImageNet-R | GAM-EWC+Merge | 6 | 77.898000 ± 0.307256 | 82.657505 ± 0.129110 |
| ImageNet-A | InfLoRA | 1 | 47.566000 | 59.226844 |
| ImageNet-A | GAM-EWC+Merge | 6 | 51.882667 ± 1.004920 | 66.941663 ± 0.455878 |
| DomainNet | InfLoRA | 1 | 72.342000 | 78.972067 |
| DomainNet | GAM-EWC+Merge | 6 | 73.859000 ± 0.052440 | 79.483883 ± 0.033443 |
| CIFAR-100 | InfLoRA | 1 | 86.440000 | 91.825679 |
| CIFAR-100 | GAM-EWC+Merge | 4 | 87.872500 ± 0.204512 | 91.966021 ± 0.211700 |

## 同一批日志的 global top1 口径

| 数据集 | 方法 | n | FAA | AAA |
|---|---|---:|---:|---:|
| ImageNet-R | InfLoRA | 1 | 76.770000 | 82.567000 |
| ImageNet-R | GAM-EWC+Merge | 6 | 78.033333 ± 0.301971 | 82.736000 ± 0.128623 |
| ImageNet-A | InfLoRA | 1 | 48.330000 | 59.351000 |
| ImageNet-A | GAM-EWC+Merge | 6 | 59.150000 ± 1.170504 | 70.466167 ± 0.489613 |

ImageNet-A 的六 seed FS：任务等权 FAA=51.882667，但按样本汇总 FAA=59.150000。这是同一组日志的不同统计量，不是画图时缩放分数。任务分组测试样本数不相同，任务等权与样本加权不保证一致。

## FS-EWC 减 InfLoRA 的差值

| 数据集 | 统计口径/来源 | ΔFAA (pp) | ΔAAA (pp) |
|---|---|---:|---:|
| ImageNet-R | 当前图任务等权 | +1.105000 | +0.032683 |
| ImageNet-R | 同批日志样本加权 | +1.263333 | +0.169000 |
| ImageNet-R | 旧表快照 | +3.760000 | +2.430000 |
| ImageNet-A | 当前图任务等权 | +4.316667 | +7.714819 |
| ImageNet-A | 同批日志样本加权 | +10.820000 | +11.115167 |
| ImageNet-A | 旧表快照 | +9.310000 | +6.320000 |

ImageNet-R 当前图中两者 AAA 仅相差约 0.033 pp，不能据此描述为明显改善，更不能在 InfLoRA 只有单 seed 的情况下声称统计显著。

## 旧表快照与当前来源

| 数据集 | 方法 | 旧表 FAA | 旧表 AAA |
|---|---|---:|---:|
| ImageNet-R | InfLoRA | 74.41 ± 0.63 | 80.31 ± 0.60 |
| ImageNet-R | GAM-EWC+Merge | 78.17 ± 0.27 | 82.74 ± 0.14 |
| ImageNet-A | InfLoRA | 50.75 ± 1.33 | 64.36 ± 1.01 |
| ImageNet-A | GAM-EWC+Merge | 60.06 ± 0.60 | 70.68 ± 0.40 |

以上仅复述 `sources.json.reported_table` 已有快照；本次没有重新追溯旧表每个数字的运行集合。InfLoRA 当前 manifest 明确写明 `reference_only_unbound_to_table`；FS 当前使用全部完整六 seed，与旧表集合不等同。

## 是否有插值/平滑

`plot_lora_aa.py` 将整数任务编号直接传入 `ax.plot`，纵值为原始任务分组准确率重算值。导出 SVG 每条 ImageNet-R/A/CIFAR 曲线只有 10 个测量点，DomainNet 只有 5 个；均与相应 AA 精确匹配到 SVG 坐标舍入精度。相邻点使用直线连接，没有额外生成插值测量点、平滑或端点修正。

## ImageNet-R 逐阶段图中数值

| 已学任务数 | InfLoRA | FS-EWC 均值 | FS−InfLoRA (pp) |
|---:|---:|---:|---:|
| 1 | 94.010000 | 90.775000 | -3.235000 |
| 2 | 89.295000 | 88.431667 | -0.863333 |
| 3 | 86.030000 | 85.690556 | -0.339444 |
| 4 | 82.955000 | 84.567083 | +1.612083 |
| 5 | 82.250000 | 82.537000 | +0.287000 |
| 6 | 81.370000 | 81.488611 | +0.118611 |
| 7 | 79.832857 | 79.406905 | -0.425952 |
| 8 | 77.301250 | 78.083750 | +0.782500 |
| 9 | 76.411111 | 77.696481 | +1.285370 |
| 10 | 76.793000 | 77.898000 | +1.105000 |

## ImageNet-A 逐阶段图中数值

| 已学任务数 | InfLoRA | FS-EWC 均值 | FS−InfLoRA (pp) |
|---:|---:|---:|---:|
| 1 | 75.000000 | 87.080000 | +12.080000 |
| 2 | 66.805000 | 81.306667 | +14.501667 |
| 3 | 66.266667 | 77.265556 | +10.998889 |
| 4 | 62.447500 | 75.669167 | +13.221667 |
| 5 | 59.030000 | 69.670667 | +10.640667 |
| 6 | 58.586667 | 63.425278 | +4.838611 |
| 7 | 54.862857 | 57.386905 | +2.524048 |
| 8 | 51.983750 | 54.187500 | +2.203750 |
| 9 | 49.720000 | 51.542222 | +1.822222 |
| 10 | 47.566000 | 51.882667 | +4.316667 |

## 配置和来源限制

- ImageNet-R/A 均为 10 个任务、每任务 20 类，日志声明不打乱类别、LoRA rank=10、batch size=128。
- InfLoRA：Adam、lr=0.0005，ImageNet-R 50 epochs，ImageNet-A 20 epochs。FS：GAM/SGD、50 epochs，R init/lr=0.02/0.02，A init/lr=0.04/0.02。
- InfLoRA 日志 `load=vit_base_patch16_224_in21k`；FS 日志显示 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`。当前材料不能证明两者实际 checkpoint 相同，Inf 日志没有权重哈希，不能进一步断言权重字节是否相同。
- ImageNet-A FS seed0 日志包含两套完整运行；绘图取最后完整的一套，且与最终 `cl_metrics.json` 矩阵一致。其他来源核查细节见两份独立报告。

## 每个源文件

- CIFAR-100 / InfLoRA / seed0: [最后阶段原始日志](/data/115-2/users/liying/low-rank-cl-new/logs/inflora/cifar100/t10/seed0.log:152)；任务等权 FAA=86.440000，AAA=91.825679。
- DomainNet / InfLoRA / seed0: [最后阶段原始日志](/data/115-2/users/liying/low-rank-cl-new/logs/inflora/domainnet/t5/seed0.log:87)；任务等权 FAA=72.342000，AAA=78.972067。
- ImageNet-A / GAM-EWC+Merge / seed0: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:1414)；任务等权 FAA=52.316000，AAA=67.074384。
- ImageNet-A / GAM-EWC+Merge / seed1: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/1/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1_vit_base_patch16_224.log:673)；任务等权 FAA=50.585000，AAA=67.029258。
- ImageNet-A / GAM-EWC+Merge / seed2: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/2/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224.log:673)；任务等权 FAA=53.311000，AAA=66.700178。
- ImageNet-A / GAM-EWC+Merge / seed3: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/3/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3_vit_base_patch16_224.log:672)；任务等权 FAA=51.968000，AAA=67.528266。
- ImageNet-A / GAM-EWC+Merge / seed4: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/4/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4_vit_base_patch16_224.log:672)；任务等权 FAA=52.236000，AAA=67.134107。
- ImageNet-A / GAM-EWC+Merge / seed5: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/5/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5_vit_base_patch16_224.log:673)；任务等权 FAA=50.880000，AAA=66.183783。
- ImageNet-A / InfLoRA / seed0: [最后阶段原始日志](/data/115-2/users/liying/low-rank-cl-new/logs/inflora/imagenet-a/t10/seed0.log:152)；任务等权 FAA=47.566000，AAA=59.226844。
- ImageNet-R / GAM-EWC+Merge / seed0: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224.log:673)；任务等权 FAA=78.031000，AAA=82.567130。
- ImageNet-R / GAM-EWC+Merge / seed1: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/1/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s1/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s1_vit_base_patch16_224.log:673)；任务等权 FAA=77.559000，AAA=82.525667。
- ImageNet-R / GAM-EWC+Merge / seed2: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/2/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s2_vit_base_patch16_224.log:673)；任务等权 FAA=78.247000，AAA=82.861067。
- ImageNet-R / GAM-EWC+Merge / seed3: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/3/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s3/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s3_vit_base_patch16_224.log:672)；任务等权 FAA=77.652000，AAA=82.767330。
- ImageNet-R / GAM-EWC+Merge / seed4: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/4/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s4/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s4_vit_base_patch16_224.log:672)；任务等权 FAA=77.674000，AAA=82.620348。
- ImageNet-R / GAM-EWC+Merge / seed5: [最后阶段原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/5/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s5/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s5_vit_base_patch16_224.log:672)；任务等权 FAA=78.225000，AAA=82.603490。
- ImageNet-R / InfLoRA / seed0: [最后阶段原始日志](/data/115-2/users/liying/low-rank-cl-new/logs/inflora/imagenet-r/t10/seed0.log:152)；任务等权 FAA=76.793000，AAA=82.624822。

## 复算与独立核验

- [结构化结果与完整来源](value_audit.json)
- [每 seed 指标 CSV](per_seed_metrics.csv)
- [InfLoRA 独立原始日志复算](inflora_independent_check.md)
- [FS 原始日志与机制记录复算](fs_independent_check.md)
- [复算和 SVG 坐标验证脚本](check_figure_values.py)

```bash
MPLCONFIGDIR=/tmp/lora-aa-reorder-mpl python refine-logs/lora_aa_value_audit_20260924/check_figure_values.py
```

后续若用于同一张论文主结果表，应先固定任务/样本统计口径，并让图和表使用同一来源运行集合。若要比较受控条件下的方法收益，还需对齐预训练权重、训练预算和重复 seed；本次未启动新实验。
