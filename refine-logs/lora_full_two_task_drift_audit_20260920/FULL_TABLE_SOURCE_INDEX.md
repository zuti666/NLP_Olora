# Full finetune原六行表：原始数据索引

2026-09-20。只读核验；没有运行训练或GPU评估。下列均为用户原full表的来源，不是LoRA，也不是另一套新full序列。

## 找到了什么

六行各自的指标JSON、原始shell日志和final_model.pt均存在；18个文件SHA与此前记录一致。准确率、BWT、Task2累计Fisher和drift均匹配用户原表。

**Task1 trace有直接来源缺口：五份JSON没有step0；FineTune唯一带step0的字段为869.0516968，与表869.80不同。869.80/858.59只能从同一Task2记录的累计式反推至显示精度，不能说已找回直接Task1原测量。原Task1模型路径被覆盖。**

另限定检查了 logs/verify_imagenet_r_1x1_fullmodel* 下32份stage1相关/shell日志，仍未找到与旧表匹配的直接Task0记录。同路径现存stage1日志已是6月9日运行：accuracy为92.77/93.08、trace为58.0808258/6.3359461；其他邻近版本也不匹配，未用它们替代原始锚点。

## 六行原始数值

| 行 | a11 | a21 | a22 | BWT | Task2累计trace | 原delta_norm平方 | 开方后的drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| FineTune | 92.61 | 78.46 | 92.47 | -14.15 | 1650.36 | 24.4091972443 | 4.940566 |
| FisherShape | 93.24 | 81.92 | 93.44 | -11.32 | 1631.30 | 26.3250126006 | 5.130791 |
| EWC | 92.61 | 85.53 | 90.66 | -7.08 | 1655.95 | 22.5312992719 | 4.746715 |
| FS-EWC | 93.24 | 90.25 | 90.79 | -2.99 | 1631.73 | 21.8330962667 | 4.672590 |
| EWC + Task-1 shaping | 93.24 | 87.89 | 87.31 | -5.35 | 1638.97 | 22.7878817377 | 4.773665 |
| EWC + Task-2 shaping | 92.61 | 82.08 | 93.86 | -10.53 | 1642.00 | 21.1143645776 | 4.595037 |

## 可直接点击的原始文件

| 表格行 | 原始指标 | 原始训练日志 | 最终模型 |
|---|---|---|---|
| FineTune | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/baseline_lr0.01_e50/imagenetr/0/fullmodel_baseline_e50_lam2000_s0/exp_run/20/fullmodel_baseline_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/baseline_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/baseline_lr0.01_e50/imagenetr/0/fullmodel_baseline_e50_lam2000_s0/exp_run/checkpoints/final_model.pt) |
| FisherShape | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/gam_only_lr0.01_e50/imagenetr/0/fullmodel_gam_only_e50_lam2000_s0/exp_run/20/fullmodel_gam_only_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/gam_only_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/gam_only_lr0.01_e50/imagenetr/0/fullmodel_gam_only_e50_lam2000_s0/exp_run/checkpoints/final_model.pt) |
| EWC | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage2_control_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/imagenetr/0/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/exp_run/20/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon_stage2_lam10000/stage2_lam10000_from_stage1lam2000_rho0p02_norm0p2_gamma0p03_lr0p01_e50/shell/control_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage2_control_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/imagenetr/0/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/exp_run/checkpoints/final_model.pt) |
| FS-EWC | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_full_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_full_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_full_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_full_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_full_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_full_e50_lam10000_s0/exp_run/checkpoints/final_model.pt) |
| EWC + Task-1 shaping | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s1_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s1_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_s1_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s1_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s1_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s1_e50_lam10000_s0/exp_run/checkpoints/final_model.pt) |
| EWC + Task-2 shaping | [metrics JSON](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s2_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s2_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_s2_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json) | [shell log](/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s2_seed0.log) | [final_model.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s2_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s2_e50_lam10000_s0/exp_run/checkpoints/final_model.pt) |

## 字段和配置

- a11/a21/a22：cnn.matrices.final 的 [0][0]、[1][0]、[1][1]。
- BWT：a21-a11。
- drift：sqrt(mechanism.steps[1].delta_norm)。旧字段是平方范数。
- Task2表中trace：mechanism.steps[1].accumulated_fisher_total_mass。
- Task2当前任务buffer另在endpoint_fisher_total_mass；两个字段不能互换。
- Task1 trace反推：(accumulated_fisher_total_mass - endpoint_fisher_total_mass)/0.9。
- 配置应以各原始日志打印的参数为依据；当前同名YAML可能变化，不能作为历史实际覆盖参数的唯一依据。完整逐行配置已随本次JSON保存。

## 逐行完整绝对路径

### 1. FineTune（old / baseline）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/baseline_lr0.01_e50/imagenetr/0/fullmodel_baseline_e50_lam2000_s0/exp_run/20/fullmodel_baseline_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/baseline_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/baseline_lr0.01_e50/imagenetr/0/fullmodel_baseline_e50_lam2000_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe
- optimizer_type: sgd
- ewc_lambda: 0
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: /data/140-1/users/liying/CopyFlatCV/config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/20/stage1_sgd_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

### 2. FisherShape（old / gam_only）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/gam_only_lr0.01_e50/imagenetr/0/fullmodel_gam_only_e50_lam2000_s0/exp_run/20/fullmodel_gam_only_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/fullmodel_lam2000_rho0p02_lr0p01_e50/shell/gam_only_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/gam_only_lr0.01_e50/imagenetr/0/fullmodel_gam_only_e50_lam2000_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe
- optimizer_type: gam
- ewc_lambda: 0
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: /data/140-1/users/liying/CopyFlatCV/config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/20/stage1_gam_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

### 3. EWC（unified / control）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage2_control_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/imagenetr/0/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/exp_run/20/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon_stage2_lam10000/stage2_lam10000_from_stage1lam2000_rho0p02_norm0p2_gamma0p03_lr0p01_e50/shell/control_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage2_control_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/imagenetr/0/fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe
- optimizer_type: sgd
- ewc_lambda: 10000
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: /data/140-1/users/liying/CopyFlatCV/config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: /data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/20/stage1_sgd_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

### 4. FS-EWC（unified / factored_full）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_full_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_full_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_full_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_full_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_full_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_full_e50_lam10000_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe_gam_factored
- optimizer_type: gam
- ewc_lambda: 10000
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/20/stage1_gam_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

### 5. EWC + Task-1 shaping（unified / factored_s1）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s1_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s1_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_s1_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s1_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s1_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s1_e50_lam10000_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe_gam_factored
- optimizer_type: gam
- ewc_lambda: 10000
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_gam_probe_lr0.01_e50/imagenetr/0/stage1_gam_probe_e50_lam2000_s0/exp_run/20/stage1_gam_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

### 6. EWC + Task-2 shaping（unified / factored_s2）

原始指标：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s2_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s2_e50_lam10000_s0/exp_run/20/fullmodel_unified_factored_s2_e50_lam10000_s0_vit_base_patch16_224_cl_metrics.json

原始日志：
/data/140-1/users/liying/CopyFlatCV/logs/verify_imagenet_r_1x1_fullmodel_ewcon/unified_stage1_lam10000_selected/shell/factored_s2_seed0.log

最终模型：
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_newData/base/logs_inc/ewcon_probe_gam_factored/unified_factored_s2_lr0.01_e50/imagenetr/0/fullmodel_unified_factored_s2_e50_lam10000_s0/exp_run/checkpoints/final_model.pt

历史日志配置（原值）：

- model_name: ewcon_probe_gam_factored
- optimizer_type: gam
- ewc_lambda: 10000
- ewc_gamma: 0.9
- fisher_mode: batch_empirical
- epochs: 50
- lrate: 0.01
- batch_size: 128
- config: config_newData/base_as1_normfisher_lam2000_imagenetr_t10c20_r10_sgd_lr001_e50_rho005_f100.yaml
- resume_from_checkpoint_dir: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/checkpoints
- resume_from_metrics_json: outputs_logs/config_newData/base/logs_inc/ewcon_probe/stage1_sgd_probe_lr0.01_e50/imagenetr/0/stage1_sgd_probe_e50_lam2000_s0/exp_run/20/stage1_sgd_probe_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json

## 不应混用的来源和含义

FineTune与FisherShape来自较早old运行族；其他四行来自后续unified族。已有汇总中还包含另一组FineTune=79.56/93.17、FisherShape=82.08/93.72，不能拿来冒充原表78.46/92.47与81.92/93.44。

原表drift包含零扩展分类头，不能直接解释为纯backbone位移。Task2 trace是处理后的累计保护buffer，不是raw逐样本旧任务Fisher。来源存在与指标含义正确是两回事，详见 [FULL_PROVENANCE.md](FULL_PROVENANCE.md)。

六个final都只有当前模型，不含旧backbone快照；因此找到final并不能弥补被覆盖的原Task1 anchor。详见 [FULL_HISTORY_RECOVERY.md](FULL_HISTORY_RECOVERY.md)。

本次仅保留原表六行、按表格顺序提取：[CSV](full_table_selected_original_data.csv)、[配置/字段/哈希JSON](full_table_selected_original_data.json)。已有全来源（含额外分支）：[full_sources.json](full_sources.json)。
