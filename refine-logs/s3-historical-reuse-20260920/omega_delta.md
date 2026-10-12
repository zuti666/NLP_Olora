# 历史 omega 与 ΔW 惩罚复用核验

指定两目录的10次完成运行 + 找回的 omega=.10 完成运行 + R_delta=.5 的外部配对锚点；共12个完整来源。

**本组可复用 11 次旧协议结果，外加 1 次边界配对锚点；直接替代当前同协议机制实验 0 次。**

全部 12 次均 seed0、ImageNet-A 10×20 类、rank10、batch128、每任务 50 epochs、LR .04/.02、λEWC=2000、GAM ρ=.05/r=.2。日志覆盖每任务 epoch1–50；10×10 原始 CNN 下三角 55 项有限并逐行吻合控制台；无非有限训练 Loss/CE/GAM/EWC/DWpen 或 traceback。未见 checkpoint 恢复，`fromgam` 指从 GAM ρ 读取附加边界。

## 共同 omega 六点

| omega | FAA | AAA | 样本加权最终 CNN | 完成 |
|---|---:|---:|---:|---|
| 0.05 | 53.036 | 66.994 | 59.78 | complete_verified |
| 0.10 | 53.128 | 67.063 | 59.91 | complete_verified |
| 0.20 | 53.112 | 67.063 | 59.97 | complete_verified |
| 0.30 | 53.179 | 67.110 | 60.04 | complete_verified |
| 0.40 | 52.741 | 67.035 | 59.78 | complete_verified |
| 0.50 | 52.531 | 67.153 | 59.58 | complete_verified |

六点除 omega 和输出/设备元数据外，日志训练字段一致。omega=.10 的完整控制台是 `logs_step2d_weightmerge_sameomega/s2d_wmerge_sameomega_omega010_gpu6.log`，同目录其它 gpu1 的 .10 仅训练到 Task0 epoch2；不能把中断日志计作独立完成复现。该目录 omega=.15 的 gpu7 记录为 CUDA 不可用失败，gpu2 只到 Task0 epoch2，不能算第七个完整点。

原 YAML 明示同一 omega 控制 `F=(1−ω)N(F_final)+ωN(mean(F20,F40))` 与 `θ=(1−ω)θ_final+ω mean(θ20,θ40)`。每条日志的10任务均有路径 Fisher 使用和48个 A/B 因子的实际融合。现存基础类代码吻合此执行证据。当前 WMergeThenFisher 则先以 CE 权重融合 A/B，之后只在保留权重估计一次 Fisher，并带 alpha=.2 阻尼。故旧曲线不能称为当前方案的纯权重 omega 扫描，也不能替换当前均匀融合/关闭融合控制。FAA最优旧点为 omega=.3，AAA最优旧点为 omega=.5；这是单seed探索，不据此改当前中心。

## 额外 ΔW 软惩罚

训练额外目标为 `λΔ max(||ΔW−ΔWref||−κRΔ,0)²`；κ=0 时精确退化为无阈值 L2。五点的 RΔ=.05 来自固定 GAM ρ=.05；GAM 外半径 r=.2 未变。Task0不启用惩罚，Task1–9保留非零惩罚和非零惩罚梯度记录。

| RΔ | κ | λΔ | 有效阈值 κRΔ | FAA | AAA | 样本最终 CNN | Task1–9 激活比例范围 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 2 | 20 | 0.10 | 49.734 | 65.839 | 57.14 | 0.907–0.983 |
| 0.05 | 2 | 50 | 0.10 | 49.972 | 65.777 | 57.47 | 0.907–0.983 |
| 0.05 | 2 | 100 | 0.10 | 49.716 | 65.755 | 57.21 | 0.880–0.975 |
| 0.05 | 1 | 50 | 0.05 | 49.807 | 65.536 | 57.54 | 0.973–0.993 |
| 0.05 | 0 | 50 | 0.00 | 50.516 | 65.617 | 58.20 | 1.000–1.000 |
| 0.50 | 1 | 50 | 0.50 | 52.815 | 66.989 | 59.58 | 0.000–0.662 |

可复用 κ=2 下 λΔ=20/50/100 三点，以及 λΔ=50 下 κ=0/1/2 三点。外部 RΔ=.5、κ=1、λΔ=50 对照 RΔ=.05 的字段受控比较给出 FAA 52.815→49.807（−3.008pp），但属于附加约束，不能填入 GAM ρ/r 曲线。

## 来源和限制

原始日志、YAML、metrics JSON及现存源码 SHA256、逐任务机制字段、全量配置、矩阵/训练预算检查保存在 `omega_delta.json`。原始JSON的FWT/AFG及上三角NaN为未定义结构项，不作为训练失败；报告JSON只保留有限派生结果。历史执行源码和预训练文件未逐运行冻结，因此不声称其与当前 frozen source 逐字一致。

| ID | 原始日志 | 原始矩阵 |
|---|---|---|
| s2d_wmerge_sameomega_omega005_gpu1 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345/s2d_wmerge_sameomega_omega005_gpu1.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.05/imageneta/0/s2d_wmerge_sameomega_omega005_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega005_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| s2d_wmerge_sameomega_omega020_gpu2 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345/s2d_wmerge_sameomega_omega020_gpu2.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.20/imageneta/0/s2d_wmerge_sameomega_omega020_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega020_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| s2d_wmerge_sameomega_omega030_gpu3 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345/s2d_wmerge_sameomega_omega030_gpu3.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30/imageneta/0/s2d_wmerge_sameomega_omega030_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega030_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| s2d_wmerge_sameomega_omega040_gpu4 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345/s2d_wmerge_sameomega_omega040_gpu4.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.40/imageneta/0/s2d_wmerge_sameomega_omega040_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega040_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| s2d_wmerge_sameomega_omega050_gpu5 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345/s2d_wmerge_sameomega_omega050_gpu5.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.50/imageneta/0/s2d_wmerge_sameomega_omega050_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega050_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| s2d_wmerge_sameomega_omega010_gpu6 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_weightmerge_sameomega/s2d_wmerge_sameomega_omega010_gpu6.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.10/imageneta/0/s2d_wmerge_sameomega_omega010_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_sameomega_omega010_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| f1_fromgam_k200_lam20_gpu1 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s2d_delta_r005_ablation/f1_fromgam_k200_lam20_gpu1.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_fromgam_k200_lam20/imageneta/0/s2d_wmerge_omega030_delta_fromgam_k200_lam20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_fromgam_k200_lam20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| f2_fromgam_k200_lam50_gpu2 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s2d_delta_r005_ablation/f2_fromgam_k200_lam50_gpu2.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_fromgam_k200_lam50/imageneta/0/s2d_wmerge_omega030_delta_fromgam_k200_lam50_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_fromgam_k200_lam50_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| f3_fromgam_k200_lam100_gpu3 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s2d_delta_r005_ablation/f3_fromgam_k200_lam100_gpu3.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_fromgam_k200_lam100/imageneta/0/s2d_wmerge_omega030_delta_fromgam_k200_lam100_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_fromgam_k200_lam100_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| f4_fromgam_k100_lam50_gpu4 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s2d_delta_r005_ablation/f4_fromgam_k100_lam50_gpu4.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_fromgam_k100_lam50/imageneta/0/s2d_wmerge_omega030_delta_fromgam_k100_lam50_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_fromgam_k100_lam50_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| f5_fromgam_k000_lam50_gpu5 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s2d_delta_r005_ablation/f5_fromgam_k000_lam50_gpu5.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_fromgam_k000_lam50/imageneta/0/s2d_wmerge_omega030_delta_fromgam_k000_lam50_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_fromgam_k000_lam50_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
| e2_warnk100_r05_lam50_gpu2 | [log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_delta_penalty_ablation/e2_warnk100_r05_lam50_gpu2.log) | [metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation_euc/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega0.30_deltawarnk100_r05_lam50/imageneta/0/s2d_wmerge_omega030_delta_warnk100_r05_lam50_imageneta_t10c20_r10_sgd_s0/exp_run/20/s2d_wmerge_omega030_delta_warnk100_r05_lam50_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json) |
