# FS-EWC 图中 ImageNet-R / ImageNet-A 独立数值核查

核查日期：2026-09-24。只读原始文件；本报告未导入或复用绘图 parser。使用独立正则读取每条 CNN grouped accuracy，按有效任务数确定阶段，对重复阶段取最后一次；另外核对每个 run 的 cl_metrics.json 最终任务矩阵。

## 核查结论

- 两个数据集均取 manifest 中 include=true 的 seeds 0–5，共 12 份日志；12 个 SHA-256 均与 manifest 一致。
- 每份日志最终 10 阶段的 CNN 任务精度与 cl_metrics.json 的 cnn.matrices.final 下三角逐项完全一致。
- 所有 12 个最终 run 的真实 variant 都是 `as1_normfisher_gam_intask_damp_weightmergethenfisher`，启用 loss-weighted 参数合并、合并后计算 Fisher、damping 和 GAM。不能仅凭 `model_name` 别名将其等同于其他 FS-EWCLoRA 公式或其他历史 variant。
- ImageNet-A seed 0 文件有两次完整 run；本图读取后一次（2026-06-29），不是开头 2026-06-25 的 run。后一次与最终 JSON 矩阵一致。

## 指标定义

令 A[t,j] 为完成阶段 t 后对任务 j 的 CNN 分类精度。图中 AA[t] = (1/t) sum_j A[t,j]；FAA = AA[10]；AAA = (1/10) sum_t AA[t]。这是任务等权。日志 CNN total/top1 是全体已见测试样本加权，其 FAA/AAA 单列为 global 指标。所有 ± 为跨六 seed 样本标准差（ddof=1），不是标准误。

| 数据集 | task FAA | task AAA | global top1 FAA | global top1 AAA |
|---|---:|---:|---:|---:|
| imagenetr | 77.898000000 ± 0.307256245 | 82.657505291 ± 0.129109561 | 78.033333333 ± 0.301971301 | 82.736000000 ± 0.128623482 |
| imageneta | 51.882666667 ± 1.004919831 | 66.941662698 ± 0.455877964 | 59.150000000 ± 1.170504165 | 70.466166667 ± 0.489613487 |

单位：百分比（差值单位为百分点）。

## 逐 seed 结果

| 数据集 | seed | task FAA | task AAA | global FAA | global AAA | JSON 内名为 AAA 的字段 |
|---|---:|---:|---:|---:|---:|---:|
| imagenetr | 0 | 78.031000000 | 82.567130159 | 78.170000000 | 82.643000000 | 80.322000000 |
| imagenetr | 1 | 77.559000000 | 82.525666667 | 77.720000000 | 82.621000000 | 80.278727273 |
| imagenetr | 2 | 78.247000000 | 82.861067460 | 78.380000000 | 82.950000000 | 80.731272727 |
| imagenetr | 3 | 77.652000000 | 82.767329762 | 77.780000000 | 82.824000000 | 80.522545455 |
| imagenetr | 4 | 77.674000000 | 82.620347619 | 77.800000000 | 82.727000000 | 80.393454545 |
| imagenetr | 5 | 78.225000000 | 82.603490079 | 78.350000000 | 82.651000000 | 80.300545455 |
| imageneta | 0 | 52.316000000 | 67.074383730 | 59.120000000 | 70.485000000 | 60.771272727 |
| imageneta | 1 | 50.585000000 | 67.029258333 | 57.800000000 | 70.641000000 | 60.446727273 |
| imageneta | 2 | 53.311000000 | 66.700178175 | 60.700000000 | 70.426000000 | 60.629818182 |
| imageneta | 3 | 51.968000000 | 67.528266270 | 59.970000000 | 71.147000000 | 61.224727273 |
| imageneta | 4 | 52.236000000 | 67.134107143 | 59.510000000 | 70.470000000 | 60.766545455 |
| imageneta | 5 | 50.880000000 | 66.183782540 | 57.800000000 | 69.628000000 | 59.541272727 |

## 每阶段 AA（图中任务等权口径）

| 数据集 | seed | t1 → t10 |
|---|---:|---|
| imagenetr | 0 | 91.510000, 87.880000, 85.370000, 84.570000, 82.198000, 81.150000, 79.402857, 78.005000, 77.554444, 78.031000 |
| imagenetr | 1 | 90.410000, 88.170000, 85.756667, 84.495000, 82.986000, 81.300000, 79.070000, 78.070000, 77.440000, 77.559000 |
| imagenetr | 2 | 91.190000, 88.340000, 85.763333, 84.330000, 81.978000, 81.568333, 80.094286, 78.587500, 78.512222, 78.247000 |
| imagenetr | 3 | 90.410000, 89.185000, 85.956667, 84.595000, 82.346000, 81.593333, 79.665714, 78.186250, 78.083333, 77.652000 |
| imagenetr | 4 | 89.940000, 88.505000, 85.913333, 84.827500, 83.034000, 81.573333, 79.137143, 78.062500, 77.536667, 77.674000 |
| imagenetr | 5 | 91.190000, 88.510000, 85.383333, 84.585000, 82.680000, 81.746667, 79.071429, 77.591250, 77.052222, 78.225000 |
| imageneta | 0 | 87.300000, 80.150000, 77.706667, 76.587500, 69.750000, 63.083333, 57.377143, 54.958750, 51.514444, 52.316000 |
| imageneta | 1 | 86.640000, 82.605000, 78.146667, 75.310000, 69.738000, 63.630000, 58.670000, 54.681250, 50.286667, 50.585000 |
| imageneta | 2 | 87.300000, 80.705000, 76.103333, 74.977500, 67.726000, 62.498333, 56.907143, 54.631250, 52.842222, 53.311000 |
| imageneta | 3 | 87.620000, 81.235000, 78.006667, 75.792500, 70.300000, 64.230000, 58.182857, 55.288750, 52.658889, 51.968000 |
| imageneta | 4 | 87.300000, 81.035000, 77.013333, 76.320000, 70.564000, 64.235000, 57.398571, 53.062500, 52.176667, 52.236000 |
| imageneta | 5 | 86.320000, 82.110000, 76.616667, 75.027500, 69.946000, 62.875000, 55.785714, 52.502500, 49.774444, 50.880000 |

## 来源与精确行号

manifest：[sources.json](/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/sources.json:255) 中 ImageNet-R run 条目自 255 行开始，ImageNet-A 自 489 行开始。所有 source_group 分别为 LoRA_merge_imagenetr / LoRA_merge_imageneta；paper_method 是 FS-EWCLoRA，内部 method 是 GAM-EWC+Merge。

| 数据集 | seed | 原始日志（链接指向最后阶段） | CNN 阶段行号 | 最终 run 开始行 | 真实 variant 声明行 |
|---|---:|---|---|---:|---:|
| imagenetr | 0 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |
| imagenetr | 1 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/1/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s1/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s1_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |
| imagenetr | 2 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/2/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s2_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |
| imagenetr | 3 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/3/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s3/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s3_vit_base_patch16_224.log:672) | 448, 472, 497, 522, 547, 572, 597, 622, 647, 672 | 1 | 88 |
| imagenetr | 4 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/4/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s4/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s4_vit_base_patch16_224.log:672) | 448, 472, 497, 522, 547, 572, 597, 622, 647, 672 | 1 | 88 |
| imagenetr | 5 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/5/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s5/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s5_vit_base_patch16_224.log:672) | 448, 472, 497, 522, 547, 572, 597, 622, 647, 672 | 1 | 88 |
| imageneta | 0 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:1414) | 1190, 1214, 1239, 1264, 1289, 1314, 1339, 1364, 1389, 1414 | 743 | 830 |
| imageneta | 1 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/1/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |
| imageneta | 2 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/2/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |
| imageneta | 3 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/3/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3_vit_base_patch16_224.log:672) | 448, 472, 497, 522, 547, 572, 597, 622, 647, 672 | 1 | 88 |
| imageneta | 4 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/4/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4_vit_base_patch16_224.log:672) | 448, 472, 497, 522, 547, 572, 597, 622, 647, 672 | 1 | 88 |
| imageneta | 5 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/5/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5_vit_base_patch16_224.log:673) | 449, 473, 498, 523, 548, 573, 598, 623, 648, 673 | 1 | 89 |

ImageNet-R 来源目录：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/`。

ImageNet-A 来源目录：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/`。

## 实际训练配置

12 份 manifest.logged_config 的每一个字段都与各自最后一次 run 的对应日志字段一致。以下为数据集内共同配置（六 seed 一致）：

| 字段 | ImageNet-R | ImageNet-A |
|---|---|---|
| `model_name` | ewclora_normfisher_gam_intask_damp | ewclora_normfisher_gam_intask_damp |
| `lora_rank` | 10 | 10 |
| `init_cls` | 20 | 20 |
| `increment` | 20 | 20 |
| `nb_tasks` | 10 | 10 |
| `class_shuffle` | False | False |
| `init_epoch` | 50 | 50 |
| `epochs` | 50 | 50 |
| `init_lr` | 0.02 | 0.04 |
| `lrate` | 0.02 | 0.02 |
| `batch_size` | 128 | 128 |
| `optimizer` | sgd | sgd |
| `optimizer_type` | gam | gam |
| `ewc_lambda` | 2000 | 2000 |
| `ewc_gamma` | 0.9 | 0.9 |
| `ewc_fisher_damping_alpha` | 0.2 | 0.2 |
| `in_task_fisher_interval` | 20 | 20 |
| `in_task_fisher_omega` | 0.3 | 0.3 |
| `in_task_weight_merge` | True | True |
| `in_task_path_weighted` | True | True |
| `in_task_path_beta` | 3.0 | 3.0 |
| `gam_grad_rho` | 0.05 | 0.05 |
| `gam_grad_norm_rho` | 0.1 | 0.2 |

实际 variant 声明补充 `weight_checkpoint_interval=20, weight_merge_omega=0.3, fisher_snapshots=False, fisher_source=post_weight_merge`。见 [R seed0:89](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224.log:89) 和 [A seed0:830](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:830)。

两者预训练 checkpoint 均明确为 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`。见 [R seed0:82](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224.log:82)、[A seed0 最后 run:824](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:824)。这只是 FS 侧证据，不单独证明与 InfLoRA 同 checkpoint。

## ImageNet-A seed 0 重复 run 专项

- 首 run：[日志:1](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:1)，2026-06-25；配置路径 `logs_try/logs_s3_damp_vs_endpoint_merge/tmp_s3_damp_c3_alpha02_int20.yaml`（78 行）。
- 最后 run：[日志:743](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log:743)，2026-06-29；配置路径 `config_save/s3_damp_c3_alpha02_int20_s0.yaml`（820 行）。
- 首尾 trainer 配置字段（不计 CNN/NME 结果）仅 `config` 文件路径发生变化；checkpoint、epochs、学习率、GAM 参数、damping、merge 的可见配置均一致。manifest 的全部 logged_config 与最后一次 run 一致。
- 但运行时代码不同：首段只声明 NormFisher-Damp（88 行）；后段明确激活 WMergeThenFisher（830 行）。不能用首段初始化行替代实际绘制 run 的实现证据。
- 首段最后 CNN global total=60.37；绘图所用最后 run 的 total=59.12，task FAA=52.316。最终矩阵匹配后段各行，未混合两个 run。

## cl_metrics.json 的 AAA 名称陷阱

JSON 的 `cnn.final.AAA` 恰等于整个有效下三角的全部 55 个元素均值，即 `sum_t t*AA[t] / 55`。它让后期阶段权重更大，和本图 `sum_t AA[t] / 10` 不同。12 份 JSON 均独立验证此关系；其 `FAA` 则与任务等权 FAA 一致。

- R seed0：JSON FAA=78.031，JSON AAA=80.322，而本图 AAA=82.567130159。[JSON FAA/AAA](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json:647)；[prefix_mean_t（各阶段AA）](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr002_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imagenetr/0/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imagenetr_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json:725)。
- A seed0：JSON FAA=52.316，JSON AAA=60.771272727，而本图 AAA=67.074383730。[JSON FAA/AAA](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json:647)；[prefix_mean_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json:725)。

两份 seed0 JSON 的 CNN 最终矩阵从 519 行开始；真实 variant 位于 1568 行、`current_fisher_source=post_weight_merge` 位于 1580 行、`weight_merge_then_fisher=true` 位于 1602 行。它们的 Fisher estimator 标记 `batch_gradient_square_proxy` 在 1563 行；不能直接称为逐样本 empirical Fisher。

## 与 manifest 早期 reported_table 的关系

manifest 的 reported_table 中 FS-EWCLoRA 旧表值为 R FAA/AAA=78.17±0.27 / 82.74±0.14，A=60.06±0.60 / 70.68±0.40。当前图使用上述完整六 seed cohort 和任务等权口径，既不是复制旧表数字，也不能把旧表数字当作当前曲线终点。manifest 本身 binding 已注明 historical implementation lineage，available seed cohort differs from paper。
