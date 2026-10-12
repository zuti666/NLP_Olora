# FullModel：CIFAR-100 GAM+EWC 与 EWC/GAM 单独实验核查

核查日期：2026-09-17。源仓库 CopyFlatCV 只读核查；没有修改训练配置或启动实验。本报告与证据索引保存在当前 O-LoRA 工作区。

结论：CIFAR-100 有7个完整10任务的 GAM+EWC+Merge 配方，全部 seed0；四数据集都有 EWC-only 的三seed尝试；已确认的 GAM-only 结果是 ImageNet-R 两任务 seed0，未在本次搜索范围找到四数据集完整全任务 GAM-only。

Final 是最终 CNN sample accuracy，AIA 是各阶段该准确率的算术平均，单位%。SD 使用样本标准差 ddof=1。不同配置、重复启动和同seed的不同attempt不计为多seed。

**CIFAR-100：完整 GAM+EWC+Merge 记录**

共同设置：ViT-B/16 全参数，10×10类，20 epochs/任务，batch128，SGD LR .01/.01，EWC λ40000，damping .2，每5 epochs合并。初始日志 All params=Trainable params=85,798,656；cifar224 是 CIFAR-100。

| GAM ρ / normρ / γ | Merge ω | EWC γ / 累积归一化 | seed | Final | AIA | 原始日志 |
|---|---:|---|---:|---:|---:|---|
| 0.005 / 0.01 / 0.01 | 0.3 | 0.9 / False | 0 | 77.32 | 84.696 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_normrho001_g001_omega030_ewcg09_noaccnorm_s0_vit_base_patch16_224.log) |
| 0.001 / 0.005 / 0.03 | 0.3 | 1.0 / True | 0 | 75.78 | 84.159 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_task0_lr001_e20_lam40000_g10_rho0001_gamma003_normrho0005_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_task0_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0001_gamma003_normrho0005_int5_omega030_alpha02_s0/exp_run/10/cifar100_task0_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0001_gamma003_normrho0005_int5_omega030_alpha02_s0_vit_base_patch16_224.log) |
| 0.005 / 0.02 / 0.01 | 0.5 | 1.0 / True | 0 | 75.18 | 84.043 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho002_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho002_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho002_omega050_s0_vit_base_patch16_224.log) |
| 0.005 / 0.01 / 0.03 | 0.3 | 1.0 / True | 0 | 74.91 | 84.091 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lr001_e20_lam40000_g10_rho0005_gamma003_normrho001_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0005_gamma003_normrho001_int5_omega030_alpha02_s0/exp_run/10/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho0005_gamma003_normrho001_int5_omega030_alpha02_s0_vit_base_patch16_224.log) |
| 0.005 / 0.01 / 0.01 | 0.5 | 1.0 / True | 0 | 74.30 | 82.741 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho001_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho001_omega050_s0_vit_base_patch16_224.log) |
| 0.005 / 0.05 / 0.01 | 0.5 | 1.0 / True | 0 | 72.90 | 79.642 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lam40000_rho0005_g001_normrho005_omega050_s0/cifar224/0/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho005_omega050_s0/exp_run/10/cifar100_full_wmergethenfisher_lam40000_rho0005_g001_normrho005_omega050_s0_vit_base_patch16_224.log) |
| 0.01 / 0.2 / 0.03 | 0.3 | 1.0 / True | 0 | 11.60 | 11.596 | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_cifar100_gam_wmergethenfisher/logs_inc/ewc_gam_wmergethenfisher/gam_full_wmergethenfisher_cifar100_lr001_e20_lam40000_g10_rho001_gamma003_normrho02_f100_int5_wmerge_omega030_alpha02_s0/cifar224/0/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho001_gamma003_int5_omega030_alpha02_s0/exp_run/10/cifar100_full_ewc_gam_wmergethenfisher_e20_lr001_lam40000_g10_rho001_gamma003_int5_omega030_alpha02_s0_vit_base_patch16_224.log) |

7条均未发现训练非有限数值。11.60 是完整但低性能的结果，保留在表中。另有2条多任务未完成、4条仅首任务、2条启动probe，不并入完整结果。名称带task0的75.78记录实际上完成10任务。无Merge的 projected 配置存在，但本次未找到对应完整运行产物。

**EWC-only：完整任务序列**

这里 EWC-only 指当前任务用 SGD+CE，旧任务用 EWC 正则；没有 GAM 或轨迹 Merge。12条日志均核对 model_name=ewcon_probe、optimizer_type=sgd、use_ewc=True，All params与Trainable params相等。

| 数据集 | epochs / λ | 有效seeds / 尝试 | Final mean±SD | AIA mean±SD |
|---|---|---|---:|---:|
| ImageNet-R | 30 / 80000 | 0,1,2 / 3 | 64.167±0.920 | 75.278±0.545 |
| DomainNet | 5 / 100000 | 0,1,2 / 3 | 58.853±1.711 | 72.129±0.582 |
| CIFAR-100 | 20 / 40000 | 0,1,2 / 3 | 75.060±3.485 | 85.721±1.040 |
| ImageNet-A | 20 / 200000 | 0,1 / 3 | 21.300±4.794 | 45.769±0.439 |

ImageNet-A 仅成功子集统计；seed2训练NaN，最终1.25、AIA16.969，不能删除失败记录。

| 数据集 | seed | Final | AIA | 状态 | 原始日志 |
|---|---:|---:|---:|---|---|
| ImageNet-R | 0 | 63.18 | 74.923 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e30_f100/imagenetr/0/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s0/exp_run/20/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s0_vit_base_patch16_224.log) |
| ImageNet-R | 1 | 64.32 | 75.906 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e30_f100/imagenetr/1/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s1/exp_run/20/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s1_vit_base_patch16_224.log) |
| ImageNet-R | 2 | 65.00 | 75.006 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e30_f100/imagenetr/2/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s2/exp_run/20/imagenetr_10t_ewc_on_e30_lam80000_gamma1_s2_vit_base_patch16_224.log) |
| DomainNet | 0 | 58.78 | 72.234 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e5_f100/domainnet/0/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s0/exp_run/69/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s0_vit_base_patch16_224.log) |
| DomainNet | 1 | 60.60 | 72.652 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e5_f100/domainnet/1/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s1/exp_run/69/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s1_vit_base_patch16_224.log) |
| DomainNet | 2 | 57.18 | 71.502 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e5_f100/domainnet/2/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s2/exp_run/69/domainnet_5t_ewc_on_e5_lam100000_gamma1_nofullfisher_s2_vit_base_patch16_224.log) |
| CIFAR-100 | 0 | 75.11 | 85.126 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e20_f100/cifar224/0/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s0/exp_run/10/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s0_vit_base_patch16_224.log) |
| CIFAR-100 | 1 | 71.55 | 85.115 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e20_f100/cifar224/1/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s1/exp_run/10/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s1_vit_base_patch16_224.log) |
| CIFAR-100 | 2 | 78.52 | 86.921 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_FullModel/logs_inc/ewcon_probe/ewcon_probe_sgd_lr001_e20_f100/cifar224/2/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s2/exp_run/10/cifar100_10t_ewc_on_e20_lam40000_batch_empirical_noaccnorm_s2_vit_base_patch16_224.log) |
| ImageNet-A | 0 | 24.69 | 46.079 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_imageneta_aligned/logs_inc/ewcon_probe/ewcon_probe_sgd_full_lr001_e20_f100_norm_lam2000/imageneta/0/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s0/exp_run/20/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s0_vit_base_patch16_224.log) |
| ImageNet-A | 1 | 17.91 | 45.458 | valid_complete | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_imageneta_aligned/logs_inc/ewcon_probe/ewcon_probe_sgd_full_lr001_e20_f100_norm_lam2000/imageneta/1/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s1/exp_run/20/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s1_vit_base_patch16_224.log) |
| ImageNet-A | 2 | 1.25 | 16.969 | training_nonfinite_failure | [log](/data/140-1/users/liying/CopyFlatCV/outputs_logs/fullmodel_imageneta_aligned/logs_inc/ewcon_probe/ewcon_probe_sgd_full_lr001_e20_f100_norm_lam2000/imageneta/2/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s2/exp_run/20/imageneta_full_ewc_norm_e20_lr001_lam200000_gamma1_noaccnorm_s2_vit_base_patch16_224.log) |

**GAM-only：已确认是 ImageNet-R 两任务实验**

本次搜索 outputs_logs 的1318个日志、logs_run 的91个日志以及 config_FullModel 的57份配置、config_newData 的34份配置。找到的纯GAM候选全部属于ImageNet-R、实际seed0、最多2任务；没有找到 CIFAR-100/ImageNet-A/ImageNet-R 的完整10任务或DomainNet完整5任务纯GAM产物。这是限定搜索范围的结论。

统一Stage-1比较中的纯GAM结果：seed0，Final=88.25，sample AIA=90.745，曲线[93.24, 88.25]。实际开关为optimizer_type=gam、use_ewc=False、ewc_lambda=0。[原始日志](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage2_gam_only_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/imagenetr/0/fullmodel_stage2_gam_only_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/exp_run/20/fullmodel_stage2_gam_only_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0_vit_base_patch16_224.log)。

同一两任务协议的 EWC-only Final=88.25，GAM+EWC Final=90.91；组合相对两者均为+2.66个百分点（相对+3.01%）。这是一组两任务单seed结果，不是四数据集完整任务序列的消融结论。[两任务完整复盘](/data/140-1/users/liying/CopyFlatCV/newmethod_design_wiki/ImageNetR-FullModel-GAM-EWC-TwoStage_experiment_recap.md)。

旧复盘的纯GAM AIA-2=90.570采用任务宏平均；本报告sample AIA=90.745，二者口径不同。其他同seed纯GAM尝试不计为多seed。名称含gam_fm_T10的71.75记录实际use_ewc=True、λ2000，已排除出纯GAM。

**比较边界**

- CIFAR的7条增强GAM是单seed参数探索，EWC表是三seed；不能把所观察最佳77.32与EWC三seed均值75.06的差值当作GAM的净贡献。
- ImageNet-R、DomainNet、CIFAR的EWC基线与增强GAM参考主要epochs/LR/λ匹配，但归一化、阻尼和Merge等也有变化；只能比较整套配方。
- ImageNet-A这组EWC λ200000，而之前增强GAM参考λ2000，不能视为λ匹配消融。
- 若需完整序列的GAM/EWC组件消融，现有证据缺口是匹配配方的纯GAM，以及CIFAR增强GAM共同多seed。

完整逐运行参数、原始路径、任务曲线、SHA及失败/未完成记录：[证据JSON](FULLMODEL_CIFAR_EWC_GAM_AUDIT_20260917.json)。
