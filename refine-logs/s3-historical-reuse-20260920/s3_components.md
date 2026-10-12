# 指定 S3 组件、融合与 S1h 阻尼历史结果核验

8/8 均完成10任务×50epoch，55个已见任务精度均有限；7条原始JSON与控制台逐格一致，1条同名JSON已被后续运行覆盖，已从原控制台恢复。FAA、AAA均按任务宏平均计算，Sample Final为样本加权精度。

| 原始日志 | seed | FAA | AAA | Sample Final | 复用范围 |
|---|---:|---:|---:|---:|---|
| s3_damp_c3_alpha02_int20_gpu3 | 0 | 52.397 | 67.901 | 60.37 | supporting_different_protocol_original_console_recovered |
| s3_endpoint_wmerge_omega030_gpu4 | 0 | 49.637 | 65.978 | 57.47 | supporting_different_protocol |
| component_ewc_gpu1 | 2 | 40.025 | 60.152 | 46.74 | supporting_missing_factorial_corner |
| component_gam_gpu2 | 2 | 52.293 | 67.098 | 59.78 | historical_same_scientific_protocol_component |
| component_merge_gpu6 | 2 | 40.127 | 60.850 | 47.00 | historical_same_scientific_protocol_component |
| s1h_endpoint_damp_alpha01_lam4e4_gpu5 | 0 | 50.562 | 66.771 | 58.39 | supporting_different_protocol |
| s1h_endpoint_damp_alpha02_lam4e4_gpu6 | 0 | 50.490 | 66.669 | 58.26 | supporting_different_protocol |
| s1h_endpoint_equiv_lam4e4_gpu4 | 0 | 50.546 | 66.700 | 58.39 | supporting_different_protocol |

## 能复用的核心结论

- July4历史组件：EWC为40.025，GAM+EWC为52.293，SGD+EWC+A为40.127 FAA。对应GAM增量+12.268个百分点；在SGD背景下打开A模块仅+0.102。二者都保留EWC，不是GAM-only或merge-only。
- `component_gam_gpu2`在科学配置上对应当前`s3_mech_a_off_s2`，`component_merge_gpu6`对应`s3_mech_sgd_ewc_s2`。可作为历史同协议组件结果复用；缺少历史启动源代码/环境哈希，不能伪装为新冻结队列产物，也不能提供新边界sidecars。
- A模块关闭会跳过快照CE前向，改变历史原生RNG消费；因此不能替代当前三个seed的`a_nomerge_matched`，后者保留快照和CE，仅跳过融合。
- S1h是λ=40000、A关闭、seed0的阻尼切片：α=0/0.1/0.2 FAA分别50.546/50.562/50.490，增量+0.016/−0.056。可以复用为这一背景下的阴性探索证据，不能拼到当前λ=2000、A打开、seed2阻尼曲线上。

## 必须修正的来源混淆

- June25 `s3_damp_c3_alpha02_int20_gpu3.log`对应旧父类，原日志明确记录epoch20/40同时保存raw Fisher与LoRA权重。其输出目录被June29的WMergeThenFisher seed0重跑复用；当前JSON的FAA52.316属于June29，并非June25。原日志恢复结果见上表；该条checkpoint也不能再归属于June25。
- `s3_endpoint_wmerge_omega030_gpu4`是每个任务结束执行`0.7*θ_end+0.3*θ_start`，运行日志证实10次各48个LoRA因子tensor有非零shift。它不是“在原路径融合上把Fisher估计位置移到premerge端点”；两组还同时改变阻尼、Fisher快照与额外前向。
- 同名`ewclora_normfisher_gam_intask_damp`在June24/25映射Damp父类，June26起映射WeightMergeThenFisher；不能用当前factory反向认定所有旧日志实现。

## 源码证据的边界

July3 commit `0a7ac7f`中GAM、LoRA forward、Intask与Damp四个核心文件与当前冻结源逐字一致。WMergeThenFisher差异仅在本批旧组件默认关闭的anchor/future诊断class_slice修复。更低层变动含默认关闭的Fisher RNG隔离与来源收据、当前双归一化开启时等价的归一化重构。以上支持科学配置复用，不构成历史执行代码完整哈希证明。详细哈希、55格CNN矩阵、500个epoch覆盖、原配置与指标来源、每条运行机制统计见`s3_components.json`。

## s3_damp_c3_alpha02_int20_gpu3

- 原日志：[s3_damp_c3_alpha02_int20_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_vs_endpoint_merge/s3_damp_c3_alpha02_int20_gpu3.log)
- 配置：[tmp_s3_damp_c3_alpha02_int20.yaml](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_vs_endpoint_merge/tmp_s3_damp_c3_alpha02_int20.yaml)；SHA256 `127dc0b15befa0fa5ef863c181d8a3b6c90dea3cc7a9ef853bd91c371dd6d5be`
- 有效结果来源：original_console_CNN_per_task_rows_recovered；原始JSON路径：拒绝使用：同名输出已覆盖
- 当前队列科学配置对应：无
- Original June25 learner is Damp parent with trajectory Fisher snapshots AND weight fusion; current alias now maps to WeightMergeThenFisher with weight snapshots only.
- Current JSON and checkpoints were overwritten by June29 seed0 reference rerun; never attribute those files to June25 result.
- Seed0 differs from current seed2 center.

## s3_endpoint_wmerge_omega030_gpu4

- 原日志：[s3_endpoint_wmerge_omega030_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_vs_endpoint_merge/s3_endpoint_wmerge_omega030_gpu4.log)
- 配置：[tmp_s3_endpoint_wmerge_omega030.yaml](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_vs_endpoint_merge/tmp_s3_endpoint_wmerge_omega030.yaml)；SHA256 `1f1698db112d7535a94534a148842f51570e7cda8902fb9662b9785e4e0daed1`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_endpoint_wmerge_omega030/imageneta/0/s3_endpoint_wmerge_omega030_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_endpoint_wmerge_omega030_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：无
- Factor interpolation is 0.7*task_end + 0.3*task_start, not epochs20/40 path fusion.
- No trajectory Fisher snapshots, no damping, seed0.
- The endpoint name denotes weight-fusion endpoints; Fisher runs after the interpolation and is not the current premerge Fisher-location ablation.

## component_ewc_gpu1

- 原日志：[component_ewc_gpu1.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_ewc_gpu1.log)
- 配置：[s3_c3_component_ewc_s2.yaml](/data/140-1/users/liying/CopyFlatCV/config_save/s3_c3_component_ewc_s2.yaml)；SHA256 `956b1dded6a125cb34e5c926dda1acb4eec95d54338fdd39dcf8fb210f9a0abc`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/component_ewc_sgd_initlr004_lr002_e50_rho005_f100_nomerge_alpha02/imageneta/2/s3_c3_component_ewc_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_c3_component_ewc_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：无
- No immutable historical source/environment/RNG receipt; source-level equivalence is bounded by dated git inspection, not bitwise provenance.
- No new task-boundary endpoint/path sidecars.
- Whole A-module removal also removes snapshot CE forwards and changes native RNG consumption; cannot replace matched_no_merge_start_task10.
- approximate_fisher_in_task=false differs textually from current Aoff config but is a legacy no-op in WeightMergeThenFisher (Fisher snapshots disabled in either case).

## component_gam_gpu2

- 原日志：[component_gam_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_gam_gpu2.log)
- 配置：[s3_c3_component_gam_s2.yaml](/data/140-1/users/liying/CopyFlatCV/config_save/s3_c3_component_gam_s2.yaml)；SHA256 `973301ecce60f099d33ab44b972bfa03f48d8e86da9d3578584474492127e0df`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/component_gam_sgd_initlr004_lr002_e50_rho005_f100_nomerge_alpha02/imageneta/2/s3_c3_component_gam_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_c3_component_gam_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：s3_mech_a_off_s2
- No immutable historical source/environment/RNG receipt; source-level equivalence is bounded by dated git inspection, not bitwise provenance.
- No new task-boundary endpoint/path sidecars.
- Whole A-module removal also removes snapshot CE forwards and changes native RNG consumption; cannot replace matched_no_merge_start_task10.
- approximate_fisher_in_task=false differs textually from current Aoff config but is a legacy no-op in WeightMergeThenFisher (Fisher snapshots disabled in either case).

## component_merge_gpu6

- 原日志：[component_merge_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_merge_gpu6.log)
- 配置：[s3_c3_component_merge_s2.yaml](/data/140-1/users/liying/CopyFlatCV/config_save/s3_c3_component_merge_s2.yaml)；SHA256 `79d921840b009cb46f445d906696957c9bba3c4e3c72e6158efd0f2e845a2971`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/component_merge_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_alpha02_wpathbeta30/imageneta/2/s3_c3_component_merge_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_c3_component_merge_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：s3_mech_sgd_ewc_s2
- No immutable historical source/environment/RNG receipt; source-level equivalence is bounded by dated git inspection, not bitwise provenance.
- No new task-boundary endpoint/path sidecars.

## s1h_endpoint_damp_alpha01_lam4e4_gpu5

- 原日志：[s1h_endpoint_damp_alpha01_lam4e4_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha01_lam4e4_gpu5.log)
- 配置：[s1h_endpoint_damp_alpha01_lam4e4_e50_initlr004_lr002.yaml](/data/140-1/users/liying/CopyFlatCV/config_Expand/step_ablation/s1h_endpoint_damp_alpha01_lam4e4_e50_initlr004_lr002.yaml)；SHA256 `0b178269a519dae8137ad08587c0d1086b4e838629d067c5f5df98061c92fbc3`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_lam4e4_endpoint_damp_alpha01/imageneta/0/s1h_endpoint_damp_alpha01_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0/exp_run/20/s1h_endpoint_damp_alpha01_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：无
- Seed0, EWC lambda40000, weight fusion off; current damping axis uses seed2/lambda2000/Aon.
- No path snapshots; interval10/omega0.5/path settings are inactive, not tested fusion hyperparameters.

## s1h_endpoint_damp_alpha02_lam4e4_gpu6

- 原日志：[s1h_endpoint_damp_alpha02_lam4e4_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha02_lam4e4_gpu6.log)
- 配置：[s1h_endpoint_damp_alpha02_lam4e4_e50_initlr004_lr002.yaml](/data/140-1/users/liying/CopyFlatCV/config_Expand/step_ablation/s1h_endpoint_damp_alpha02_lam4e4_e50_initlr004_lr002.yaml)；SHA256 `e0b0f580df76b9bef6242c21353b5393622872d8e903c0731d90bd61f3cbb74f`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_lam4e4_endpoint_damp_alpha02/imageneta/0/s1h_endpoint_damp_alpha02_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0/exp_run/20/s1h_endpoint_damp_alpha02_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：无
- Seed0, EWC lambda40000, weight fusion off; current damping axis uses seed2/lambda2000/Aon.
- No path snapshots; interval10/omega0.5/path settings are inactive, not tested fusion hyperparameters.

## s1h_endpoint_equiv_lam4e4_gpu4

- 原日志：[s1h_endpoint_equiv_lam4e4_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_equiv_lam4e4_gpu4.log)
- 配置：[s1h_endpoint_equiv_lam4e4_e50_initlr004_lr002.yaml](/data/140-1/users/liying/CopyFlatCV/config_Expand/step_ablation/s1h_endpoint_equiv_lam4e4_e50_initlr004_lr002.yaml)；SHA256 `905c14e931307bd7d5464892494d77f97a34404c371623add586a93ea08b092d`
- 有效结果来源：raw_metrics_and_crosschecked_console；原始JSON路径：/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask/gam_sgd_initlr004_lr002_e50_rho005_f100_lam4e4_endpoint_equiv/imageneta/0/s1h_endpoint_equiv_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0/exp_run/20/s1h_endpoint_equiv_lam4e4_e50_initlr004_lr002_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json
- 当前队列科学配置对应：无
- Seed0, EWC lambda40000, weight fusion off; current damping axis uses seed2/lambda2000/Aon.
- No path snapshots; interval10/omega0.5/path settings are inactive, not tested fusion hyperparameters.
