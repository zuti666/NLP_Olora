# CIFAR-100 全参数 EWC-GAM：多seed与参数差异核查

核查日期：2026-09-19。方法是日志中的 ewc_gam_wmergethenfisher（GAM + EWC + Merge）。

**CIFAR-100 未找到 seed1/2 等多seed执行证据。** 在 CopyFlatCV 源运行目录检出的29份 CIFAR EWC-GAM日志（15份 outputs_logs、14份 logs_run）中，所有启动日志的实际seed均为0。排除证据副本，输出与终端日志不重复计数。

此前“7套完整实验”指7套不同参数配置、每套最后一次完整运行；其中一套有2次完整运行，因此7套配置共保留8次完整attempt，仍只有seed0。

共同设置：ViT-B/16全参数（初始All params=Trainable params=85,798,656），CIFAR-100，10×10类，20 epochs/任务，batch128，基础SGD，初始/后续LR0.01，cosine，momentum/weight_decay=0；EWC λ40000，每任务Fisher归一化=True，Fisher damping=0.2，max_batches100；每5 epochs存储/合并快照，path_beta=3.0。

7组记录的 config 字段指向同一个基础配置文件，但运行时打印的参数存在覆盖，不能仅根据配置文件名判断参数一致。以下数值来自最后一次启动对应的实际参数及最后10任务CNN曲线。

| 编号 | GAM ρ | GAM normρ | GAM γ | Merge ω | EWC γ | 累计Fisher归一化 | Final (%) | 十阶段均值 (%) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.005 | 0.01 | 0.01 | 0.3 | 0.9 | 关 | 77.32 | 84.696 |
| 2 | 0.001 | 0.005 | 0.03 | 0.3 | 1.0 | 开 | 75.78 | 84.159 |
| 3 | 0.005 | 0.02 | 0.01 | 0.5 | 1.0 | 开 | 75.18 | 84.043 |
| 4 | 0.005 | 0.01 | 0.03 | 0.3 | 1.0 | 开 | 74.91 | 84.091 |
| 5 | 0.005 | 0.01 | 0.01 | 0.5 | 1.0 | 开 | 74.30 | 82.741 |
| 6 | 0.005 | 0.05 | 0.01 | 0.5 | 1.0 | 开 | 72.90 | 79.642 |
| 7 | 0.01 | 0.2 | 0.03 | 0.3 | 1.0 | 开 | 11.60 | 11.596 |

Merge ω变化时，日志中的 in_task_fisher_omega 同步变化。GAM γ与EWC γ是两个不同参数。十阶段均值是10个CNN sample accuracy的算术平均，不混用结果JSON的final.AAA字段。

- 第3/5/6组的主要训练设置一致，GAM normρ分别为0.02/0.01/0.05，对应Final75.18/74.30/72.90；结果存在差异，但目前只有seed0，不能据此估计跨seed稳定性。
- 第1组相对第5组，Merge/Fisher ω、EWC γ、累计Fisher归一化同时变化；Final提高3.02个百分点，不能单独归因于某一个改动。
- 最高Final77.32；其余非极低性能配置最低72.90，相差4.42个百分点。第7组完成10任务但仅11.60，该完整低性能结果也保留在表中。

同配置同seed重复运行：

| 配置编号 | 启动时间（原日志） | seed | 完成任务 | Final (%) | 十阶段均值 (%) | 原始曲线 |
|---|---|---:|---:|---:|---:|---|
| 5 | 2026-07-14 09:05:03,217 | 0 | 10 | 76.28 | 84.059 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho001_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0_vit_base_patch16_224.log:768) |
| 5 | 2026-07-14 19:32:43,776 | 0 | 10 | 74.30 | 82.741 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho001_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0_vit_base_patch16_224.log:1282) |

这两次启动日志打印的参数字段完全一致，Final差1.98个百分点；仅凭日志无法确认差异来自代码版本、环境或非确定性。它们不能作为两个独立seed，也不能用7套配置计算多seed均值±标准差。

每组原始日志：

- 配置1：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0_vit_base_patch16_224.log:485)
- 配置2：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_task0_lr001_e20_lam40000_g10_rho0001_gamma003_normrho0005_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_task0_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0001_gamma003_normrho0005_int5_omega030_alpha02_s0/exp_run/10/cifar100_task0_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0001_gamma003_normrho0005_int5_omega030_alpha02_s0_vit_base_patch16_224.log:1068)
- 配置3：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho002_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho002_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho002_omega050_s0_vit_base_patch16_224.log:492)
- 配置4：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lr001_e20_lam40000_g10_rho0005_gamma003_normrho001_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0005_gamma003_normrho001_int5_omega030_alpha02_s0/exp_run/10/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0005_gamma003_normrho001_int5_omega030_alpha02_s0_vit_base_patch16_224.log:769)
- 配置5：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho001_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0_vit_base_patch16_224.log:1282)
- 配置6：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho005_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho005_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho005_omega050_s0_vit_base_patch16_224.log:492)
- 配置7：[最终曲线](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lr001_e20_lam40000_g10_rho001_gamma003_normrho02_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho001_gamma003_int5_omega030_alpha02_s0/exp_run/10/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho001_gamma003_int5_omega030_alpha02_s0_vit_base_patch16_224.log:491)

其他数据集的全参数 EWC-GAM+Merge 多seed交叉核查：

| 数据集／同配置组 | 实际seed | 完成情况 |
|---|---|---|
| DomainNet λ100000、ρ0.05、GAM γ0.03、interval1 | 0、1、2 | 均完成5任务，无训练NaN |
| ImageNet-R λ80000、ρ0.05、GAM γ0.03、interval10 | 0、1、2、4 | 均完成10任务，无训练NaN；seed2表现退化 |
| ImageNet-A λ2000、ρ0.01、GAM γ0.05、interval20 | 0、1、2 | 均完成10任务，无训练NaN；seed2明显较差 |
| ImageNet-A λ200000，其余同上 | 0、1、2 | 均输出10任务；seed2训练Loss NaN，属于失败运行 |

原始证据示例：[DomainNet seed2完整曲线](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_domainnet_gam_wmergethenfisher/ewc_gam_wmergethenfisher_domainnet_e5_lam100000_g10_rho005_int1_s2_gpu6.log:875)；[ImageNet-A λ200000 seed2 NaN](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s2_gpu6.log:441)。

ImageNet-R 文件名 s2_rerun 的实际seed为4；另一个 rho001...s2 文件实际seed为0，且rho变成0.01，不能并入rho0.05组作为seed2。[实际seed日志](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imagenetr_aligned/ewc_gam_wmergethenfisher_imagenetr_fullmodel_e30_lr001_lam80000_rho001_gamma003_int10_s2_gpu1.log:14)。
