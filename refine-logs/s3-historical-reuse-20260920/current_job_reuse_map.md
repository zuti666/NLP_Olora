# 当前18项实验与历史结果逐项复用映射

在205条历史索引中，核验到 **4项同seed、同科学配置的已有结果**：center seed1/3、GAM+EWC且A模块关闭seed2、SGD+EWC+A seed2。这里的“已有”指历史结果可复用，**不是新增4项队列完成，也未修改或停止当前队列**。

| 当前任务 | 历史同配置结果FAA | 复用结论 |
|---|---:|---|
| `s3_mech_a_nomerge_matched_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_gam_half_direction_s2` | — | 未见匹配的历史控制 |
| `s3_mech_sgd_ewc_s2` | 40.127 | 已有同seed科学配置 |
| `s3_mech_center_s1` | 50.585 | 已有同seed科学配置 |
| `s3_mech_a_nomerge_matched_s1` | — | 仅有不同控制的相关证据 |
| `s3_mech_a_off_s2` | 52.293 | 已有同seed科学配置 |
| `s3_mech_center_s3` | 51.968 | 已有同seed科学配置 |
| `s3_mech_a_nomerge_matched_s3` | — | 未见匹配的历史控制 |
| `s3_mech_fisher_endpoint_anchor_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_a_uniform_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_a_latest_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_fisher_alpha0_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_fisher_alpha1_s2` | — | 仅有不同控制的相关证据 |
| `s3_mech_fisher_gamma0_s2` | — | 未见匹配的历史控制 |
| `s3_mech_fisher_gamma1_s2` | — | 未见匹配的历史控制 |
| `s3_mech_fisher_tasknorm_off_s2` | — | 未见匹配的历史控制 |
| `s3_mech_gam_projection_off_s2` | — | 未见匹配的历史控制 |
| `s3_mech_gam_rho025_s2` | — | 未见匹配的历史控制 |

历史GAM×A模块四格12次（seed0/1/2）可完整复用，适用于历史组件表。它不替代本轮保持快照CE执行和RNG消费的matched-no-merge三seed对照。

四项已有同配置实验如果仅为补齐性能表，属于重复训练；若用于生成新冻结源/环境绑定证据以及premerge endpoint、epoch20/40路径因子，则仍有明确新增产物。历史checkpoint仅保存融合后状态，不能反推出这些边界状态。

源码默认核验：train/eval workers=8、batch_empirical Fisher、fisher_split=false、Fisher RNG未隔离、任务末Fisher未跳过、NME均开启；历史数据根目录默认/data/140-1/datasets。ignore approximate_fisher_in_task仅限runtime已证实WMergeThenFisher的候选。无历史全代码/环境/数据收据，故不宣称位级复现。

## s3_mech_a_nomerge_matched_s2

未见in_task_weight_merge_start_task=10且仍保存两次快照、执行质量CE的历史训练。Aoff跳过CE并改变RNG，不能替代。
- 相关但不同控制：[component_gam_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_gam_gpu2.log)；与当前不同字段：in_task_weight_merge, in_task_weight_merge_start_task。

## s3_mech_gam_half_direction_s2

索引中完成ImageNet-A实验的GAM β1/β3均为1，未见完整GAM方向减半而EWC保持不变的实验。

## s3_mech_sgd_ewc_s2

同seed、相同科学配置与运行分支的历史结果已经存在；可复用性能及已有机制统计。新运行的增量是冻结源/环境收据与任务边界endpoint/path状态，不是首次覆盖该科学配置。
- 原日志：[component_merge_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_merge_gpu6.log)；FAA 40.127，AAA 60.850，Sample Final 47.00。

## s3_mech_center_s1

同seed、相同科学配置与运行分支的历史结果已经存在；可复用性能及已有机制统计。新运行的增量是冻结源/环境收据与任务边界endpoint/path状态，不是首次覆盖该科学配置。
- 原日志：[s3_damp_c3_alpha02_int20_s1_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s1_gpu2.log)；FAA 50.585，AAA 67.029，Sample Final 57.80。

## s3_mech_a_nomerge_matched_s1

未见in_task_weight_merge_start_task=10且仍保存两次快照、执行质量CE的历史训练。Aoff跳过CE并改变RNG，不能替代。
- 相关但不同控制：[component_gam_s1_smigpu7.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_multiseed_verification/component_gam_s1_smigpu7.log)；与当前不同字段：in_task_weight_merge, in_task_weight_merge_start_task。

## s3_mech_a_off_s2

同seed、相同科学配置与运行分支的历史结果已经存在；可复用性能及已有机制统计。新运行的增量是冻结源/环境收据与任务边界endpoint/path状态，不是首次覆盖该科学配置。
- 原日志：[component_gam_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_component_ablation_seed2/component_gam_gpu2.log)；FAA 52.293，AAA 67.098，Sample Final 59.78。

## s3_mech_center_s3

同seed、相同科学配置与运行分支的历史结果已经存在；可复用性能及已有机制统计。新运行的增量是冻结源/环境收据与任务边界endpoint/path状态，不是首次覆盖该科学配置。
- 原日志：[s3_damp_c3_alpha02_int20_s3_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s3_gpu4.log)；FAA 51.968，AAA 67.528，Sample Final 59.97。

## s3_mech_a_nomerge_matched_s3

未见in_task_weight_merge_start_task=10且仍保存两次快照、执行质量CE的历史训练。Aoff跳过CE并改变RNG，不能替代。

## s3_mech_fisher_endpoint_anchor_s2

旧endpoint权重融合改变task-start/end权重，不是固定融合模型下只将Fisher估计移到premerge端点。
- 相关但不同控制：[s1h_endpoint_damp_alpha02_lam4e4_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha02_lam4e4_gpu6.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, in_task_weight_merge, in_task_weight_merge_start_task, s3_fisher_location。
- 相关但不同控制：[s1h_endpoint_damp_alpha01_lam4e4_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha01_lam4e4_gpu5.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task, s3_fisher_location。
- 相关但不同控制：[s1h_endpoint_equiv_lam4e4_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_equiv_lam4e4_gpu4.log)；与当前不同字段：seed, model_name, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_start_ratio, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, in_task_path_loss_max_batches, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task, s3_fisher_location。
- 相关但不同控制：[s3_endpoint_wmerge_omega030_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_vs_endpoint_merge/s3_endpoint_wmerge_omega030_gpu4.log)；与当前不同字段：seed, model_name, in_task_path_start_ratio, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, in_task_path_loss_max_batches, ewc_fisher_damping_alpha, in_task_weight_merge, s3_fisher_location。

## s3_mech_a_uniform_s2

历史快照加权/窗口有相关结果，但seed、omega、间隔、Fisher版本或额外CE执行预算不同。
- 相关但不同控制：[s6_a2_wmergef_omega05_gpu1.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s6_ablation/s6_a2_wmergef_omega05_gpu1.log)；与当前不同字段：seed, in_task_fisher_interval, in_task_fisher_omega, in_task_path_weighted, in_task_path_normalize_snapshots。
- 相关但不同控制：[s2d_wmerge_damp_batchF_b0_sanity_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_damp_batchfisher_ablation/s2d_wmerge_damp_batchF_b0_sanity_gpu2.log)；与当前不同字段：seed, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha。
- 相关但不同控制：[s4a_B1_allpath_sr00_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s4a_wmergethenfisher_AB/s4a_B1_allpath_sr00_gpu2.log)；与当前不同字段：seed, in_task_fisher_interval, in_task_fisher_omega, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha。
- 相关但不同控制：[s4a_B3_late_sr06_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s4a_wmergethenfisher_AB/s4a_B3_late_sr06_gpu4.log)；与当前不同字段：seed, in_task_fisher_interval, in_task_fisher_omega, in_task_path_start_ratio, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha。
- 相关但不同控制：[s4a_B2_midlate_sr04_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s4a_wmergethenfisher_AB/s4a_B2_midlate_sr04_gpu3.log)；与当前不同字段：seed, in_task_fisher_interval, in_task_fisher_omega, in_task_path_start_ratio, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha。

## s3_mech_a_latest_s2

历史快照加权/窗口有相关结果，但seed、omega、间隔、Fisher版本或额外CE执行预算不同。
- 相关但不同控制：[s2d_wmerge_damp_batchF_c2_alpha02_int10_sr06_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_damp_batchfisher_ablation/s2d_wmerge_damp_batchF_c2_alpha02_int10_sr06_gpu4.log)；与当前不同字段：seed, in_task_fisher_interval。
- 相关但不同控制：[s4a_B3_late_sr06_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s4a_wmergethenfisher_AB/s4a_B3_late_sr06_gpu4.log)；与当前不同字段：seed, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha。

## s3_mech_fisher_alpha0_s2

已有旧背景阻尼结果，但不能替代当前seed2、lambda2000、Aon的单轴控制。
- 相关但不同控制：[s2d_wmerge_damp_batchF_b0_sanity_gpu2.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_step2d_damp_batchfisher_ablation/s2d_wmerge_damp_batchF_b0_sanity_gpu2.log)；与当前不同字段：seed, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots。
- 相关但不同控制：[mechanism_baseline_s2_v2_smigpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_mechanism/mechanism_baseline_s2_v2_smigpu5.log)；与当前不同字段：in_task_weight_merge。
- 相关但不同控制：[s1h_endpoint_damp_alpha02_lam4e4_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha02_lam4e4_gpu6.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。
- 相关但不同控制：[s1h_endpoint_damp_alpha01_lam4e4_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha01_lam4e4_gpu5.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。
- 相关但不同控制：[s1h_endpoint_equiv_lam4e4_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_equiv_lam4e4_gpu4.log)；与当前不同字段：seed, model_name, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_start_ratio, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, in_task_path_loss_max_batches, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。

## s3_mech_fisher_alpha1_s2

未见alpha=1的trace匹配各向同性Fisher控制；旧alpha0/0.1/0.2只能提供相关背景。
- 相关但不同控制：[s1h_endpoint_damp_alpha02_lam4e4_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha02_lam4e4_gpu6.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。
- 相关但不同控制：[s1h_endpoint_damp_alpha01_lam4e4_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_damp_alpha01_lam4e4_gpu5.log)；与当前不同字段：seed, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。
- 相关但不同控制：[s1h_endpoint_equiv_lam4e4_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s1h_endpoint_damp_ablation/s1h_endpoint_equiv_lam4e4_gpu4.log)；与当前不同字段：seed, model_name, ewc_lambda, in_task_fisher_interval, in_task_fisher_omega, in_task_path_start_ratio, in_task_path_beta, in_task_path_weighted, in_task_path_normalize_snapshots, in_task_path_loss_max_batches, ewc_fisher_damping_alpha, in_task_weight_merge, in_task_weight_merge_start_task。

## s3_mech_fisher_gamma0_s2

完成的相关EWC实验gamma均为0.9，未见gamma=0或1的相应历史消融。

## s3_mech_fisher_gamma1_s2

完成的相关EWC实验gamma均为0.9，未见gamma=0或1的相应历史消融。

## s3_mech_fisher_tasknorm_off_s2

未见同一S3分支中仅关闭任务Fisher归一化的完整实验；NCL/KFAC的不归一化配置是另一方法。

## s3_mech_gam_projection_off_s2

索引中完成ImageNet-A实验的GAM gamma均为0.1，未见gamma=0投影消融。

## s3_mech_gam_rho025_s2

索引中完成ImageNet-A实验的rho为0.05或其他方法的0.02，未见当前rho=0.025。
