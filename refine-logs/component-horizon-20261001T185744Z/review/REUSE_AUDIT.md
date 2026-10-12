# 10-task component reuse audit

审计时间：2026-10-01T19:09:20.931809+00:00。原始结果只读；无训练、无 GPU 调用、无旧队列改动。

## 结论

Stage I 应安排 **9 条 fresh seed2 轨迹**：matched {U0, US, UG, UO, UFull} 与 natural {U0, UG, UO, UFull}。fresh matchedUS 只有在标量 c=η exact、四探针与恢复/SGD路径完全相同时，才可同时作为 naturalUS。旧 g3 r=.25 十任务可以保留为历史 practical 结果；为了比较 natural 与 matched 的尺度效应，仍需要新 cohort 的 naturalUFull。旧 g1 r=1 只作历史辅助，不充当 r=.25 的 bitwise 对照。

截至本次所搜范围，未找到匹配 50 epochs、rank10、seed2、no Fisher/no EWC/no fusion 的完整十任务 g0，也未找到 r=.25 的完整纯 g1。找到了更早 clean-SGD 十任务，但预算、seed、learner 均不同，不能直接填进新对照表。

新附件的 matched target 是候选自身当前状态的虚拟 g1 更新范数；当前 formation campaign 的独立 S 轨迹 reference 不符合这一新定义。旧 natural 结果也不构成新 matched 结果。

新阶段需等待当前 campaign 的 120 项（12 smoke + 24 formation + 84 future）完成并经 receipt/hash 验证。Stage II/III 及 r=1 robustness 仅保留为条件计划，不自动创建或启动。

## 完整纯 GAM 历史结果

以下全部为 ImageNet-A 20+20×9、固定类序、ViT-B/16、LoRA rank10、seed2、batch128；Task0 与后续任务都是 50 epochs。init_lr=.04、lrate=.02；cosine T_max=50，floor 分别 .004/.002；momentum=weight_decay=0，rho=.05、beta2=1、lambda_flat=1。Fisher skip=true、EWC lambda=0、normalization/damping/in-task Fisher/weight merge/endpoint merge 关闭。

所有数值从原始 CNN 10×10 matrix 的 55 个已观测有限元素独立重算。AAA 是 10 个“阶段已见任务均值”的平均；FAA 是最终 10 任务均值；LA 是对角均值；BWT/F 只平均旧9任务。原文件 native AAA 是三角55格均值，不用于这里的 AAA 排序。sample AAA/FAA 另列，来自 native CNN top1 曲线。

| arm | r | β1 / β3 / γ | AAA | FAA | LA | BWT | F | sample AAA / FAA |
|---|---:|---|---:|---:|---:|---:|---:|---|
| mean_sum_r1 | 1.0 | 0.5 / 0.5 / 0.0 | 65.950948 | 49.927 | 57.172 | -8.050000 | 8.305556 | 69.861 / 58.06 |
| g3_r0 | 0.0 | 0.0 / 1.0 / 0.0 | 62.299385 | 45.253 | 53.731 | -9.420000 | 9.587778 | 65.936 / 51.94 |
| g3_r0.1 | 0.1 | 0.0 / 1.0 / 0.0 | 64.651266 | 49.384 | 56.103 | -7.465556 | 7.465556 | 68.504 / 57.01 |
| g3_r0.25 | 0.25 | 0.0 / 1.0 / 0.0 | 66.229609 | 50.799 | 57.317 | -7.242222 | 7.365556 | 70.226 / 58.92 |
| g3_r0.5 | 0.5 | 0.0 / 1.0 / 0.0 | 66.132657 | 50.692 | 57.662 | -7.744444 | 8.114444 | 70.011 / 58.85 |
| original_gamma_0.1 | 1.0 | 1.0 / 1.0 / 0.1 | 68.028517 | 55.341 | 61.719 | -7.086667 | 7.086667 | 71.253 / 62.08 |
| sum_r1 | 1.0 | 1.0 / 1.0 / 0.0 | 67.860863 | 54.726 | 61.583 | -7.618889 | 7.618889 | 71.103 / 61.49 |
| g3_r1 | 1.0 | 0.0 / 1.0 / 0.0 | 65.993172 | 50.679 | 56.728 | -6.721111 | 7.011111 | 69.998 / 58.85 |
| g1_r1 | 1.0 | 1.0 / 0.0 / 0.0 | 62.165531 | 45.132 | 53.436 | -9.226667 | 9.482222 | 65.842 / 51.88 |

这九条 receipt 的 config、runtime config、started id/config hash、suite、runtime scripts、raw metrics hash 均与现存文件一致；均无失败 receipt；每条 console 中存在完整 10×50=500 组 epoch 记录。每条10次 TaskStart 均 has_fisher=False、10次 task-end Fisher skip，机制记录10任务 EWC/Fisher范数为0、snapshot为0、merge applied=false。类名/固定标签中的 Fisher/WMergeThenFisher 不代表实际执行。物理行号保存在 JSON。没有将结构性的上三角 null 或不可用的 FWT/首任务 BWT 当作训练失败。

同一份 190 个 native source 文件已重新逐文件校验，九条相同，并与当前 formation deployment 的 source manifest 全部一致。此结论只涵盖 native source，不代表不同 wrapper 的数据/RNG路径等价。已保存 receipt 的 native_flags_before_main；该字段不是全过程 flags 观测。native seed helper 将 cudnn deterministic=True、benchmark=False。

## 配对与结论边界

- 旧 g1 的 r=1 与新 r=.25 数学上选择同一个 g1，但 native 仍计算 g2/g3，随后执行原位减法恢复；FP32 roundtrip 会随 outer probe 改变。不得宣称数值轨迹或 bitwise 相同。
- 旧 g3 r=.25 与新模型骨架、50-epoch LR schedule、rank、rho、纯度可对齐。旧训练使用 native 全局数据 RNG；新 wrapper 按 epoch 显式种子配对和隔离诊断 RNG。旧 seed2 结果不能被标作新 cohort 的严格 paired control，旧新差值不能单独解释为步长机制。
- r=.25/.5/1 的 FAA 为 50.799/50.692/50.679，AAA 为 66.229609/66.132657/65.993172。可说这一 seed2 历史 natural-scale 扫描较接近；不能宣称多seed等效、最优或已证明平台。
- 标为 original_gamma_0.1 的 pure GAM 是带投影的 g1+g3 组合，不能算 UFull=g3。r=0 的 g3 也不能不经说明替换 g1。
- 本次是旧日志/指标/receipt/source审计，未读取所有旧 checkpoint tensor；若要 warm-start 或未来重放旧 checkpoint，须另做 checkpoint 文件与初始/data身份检查。

## 原始配置、指标和完成凭据

- **mean_sum_r1**，完成 2026-09-30T23:39:19.688226+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_meansum_r1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_meansum_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_meansum_r1_k10_s2/exp_run/20/s3_puregam_meansum_r1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_meansum_r1_k10_s2/completed.json)。
- **g3_r0**，完成 2026-09-30T23:38:35.057039+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_onlyg3_r0_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_onlyg3_r0_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_onlyg3_r0_k10_s2/exp_run/20/s3_puregam_onlyg3_r0_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_onlyg3_r0_k10_s2/completed.json)。
- **g3_r0.1**，完成 2026-09-30T23:42:12.517448+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_onlyg3_r0p1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_onlyg3_r0p1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_onlyg3_r0p1_k10_s2/exp_run/20/s3_puregam_onlyg3_r0p1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_onlyg3_r0p1_k10_s2/completed.json)。
- **g3_r0.25**，完成 2026-09-30T23:41:53.089648+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_onlyg3_r0p25_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_onlyg3_r0p25_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_onlyg3_r0p25_k10_s2/exp_run/20/s3_puregam_onlyg3_r0p25_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_onlyg3_r0p25_k10_s2/completed.json)。
- **g3_r0.5**，完成 2026-09-30T23:35:28.086458+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_onlyg3_r0p5_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_onlyg3_r0p5_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_onlyg3_r0p5_k10_s2/exp_run/20/s3_puregam_onlyg3_r0p5_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_onlyg3_r0p5_k10_s2/completed.json)。
- **original_gamma_0.1**，完成 2026-09-30T20:37:52.139908+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_original_r1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_original_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_original_r1_k10_s2/exp_run/20/s3_puregam_original_r1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_original_r1_k10_s2/completed.json)。
- **sum_r1**，完成 2026-09-30T20:37:44.515663+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_sum_r1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_sum_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_sum_r1_k10_s2/exp_run/20/s3_puregam_gamma0_sum_r1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_sum_r1_k10_s2/completed.json)。
- **g3_r1**，完成 2026-09-30T20:40:38.700995+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_g3_r1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g3_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g3_r1_k10_s2/exp_run/20/s3_puregam_gamma0_g3_r1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_g3_r1_k10_s2/completed.json)。
- **g1_r1**，完成 2026-09-30T20:40:10.357134+00:00；[config](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_g1_r1_k10_s2.json)；[raw metrics](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g1_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g1_r1_k10_s2/exp_run/20/s3_puregam_gamma0_g1_r1_k10_s2_vit_base_patch16_224_cl_metrics.json)；[completed receipt](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_g1_r1_k10_s2/completed.json)。

## 更早的 clean-SGD：存在但协议不匹配

| 旧实验 | seed | init/task epochs | init_lr / lrate | AAA | FAA | LA | BWT | F |
|---|---:|---|---|---:|---:|---:|---:|---:|
| lora_sgdplain_oldlearner_1x_noewc_imageneta_t10c20_r10_e20_initlr002_lr002_lam0_s0 | 0 | 20 / 20 | 0.02 / 0.02 | 59.487334 | 41.573 | 47.162 | -6.210000 | 6.645556 |
| lora_sgdplain_oldlearner_1x_noewc_imageneta_t10c20_r10_e20_initlr003_lr003_lam0_s0 | 0 | 20 / 20 | 0.03 / 0.03 | 61.068782 | 42.278 | 50.516 | -9.153333 | 9.400000 |
| lora_sgdplain_oldlearner_1x_noewc_imageneta_t10c20_r10_e10_initlr004_lr002_lam0_s0 | 0 | 10 / 10 | 0.04 / 0.02 | 53.846946 | 36.279 | 39.354 | -3.416667 | 3.952222 |
| lora_sgdplain_oldlearner_1x_noewc_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0 | 0 | 20 / 20 | 0.04 / 0.02 | 60.466615 | 40.630 | 47.322 | -7.435556 | 8.102222 |

这些均为 rank10、SGD、no EWC、skip Fisher、momentum0/weight_decay0 的旧 learner `ewclora_youyue_fitarchitechture`；rho/r/gamma 对普通 SGD 更新不生效。原始 log 确认每任务最终 epoch 与完整 matrix，未找到 hash-bound completed receipt，不能假称有 receipt。log 中有效参数可能由命令行覆盖，不能把引用的 YAML 模板当实际配置。原 metrics/log 路径、hash、名义配置路径完整保存在 JSON。另在历史索引见到 seed2 的 FisherShape/no-EWC 点，但使用10/20epochs和momentum .9，也不能填当前 g0。

## 容易误认的 λ=0 控制

| id | 指标口径 | AAA | FAA | 不可复用原因 |
|---|---|---:|---:|---|
| s3f_diagonal_sgd_s2_l0 | official test | 60.729855 | 40.748000 | task-end Fisher active; in-task weight merge active; different native source cohort |
| s3f_kronecker_sgd_s2_l0 | official test | 60.783322 | 40.777000 | task-end Fisher active; in-task weight merge active; different native source cohort |
| coarse__diagonal__sgd__seed42__lambda0 | train holdout 10% dev | 60.905600 | 39.134689 | seed42; train_holdout_10pct development evaluation; not official test; batched Fisher-history learner/passes |
| coarse__kronecker__sgd__seed42__lambda0 | train holdout 10% dev | 60.905600 | 39.134689 | seed42; train_holdout_10pct development evaluation; not official test; batched Fisher-history learner/passes |

以上四条原始 matrix、receipt和metrics hash重新核验；seed42 dev行与官方测试表完全分开。λ=0 本身不保证 Fisher passes/融合关闭。独立 Task0 clean CE（262/307=85.342%）只有一个任务，不能当十任务 g0。

## 搜索范围与可复查文件

无日期限制地枚举 CopyFlatCV/refine-logs、logs_try、GAM-EWC 的 JSON/YAML/log（包括被 gitignore 的文件，排除 checkpoint payload/source树），解析 ImageNet-A 配置；同时检查 GAM-EWC 历史 all_metrics_inventory 的426条记录并回到关联原日志/metrics。既有原始路径位于 logs_our_joost/outputs_logs 时按索引定点追溯，没有泛扫 checkpoint。未找到的结论限定在这些实际搜索范围。

[结构化审计](REUSE_AUDIT.json)、[JSON配置搜索索引](SEARCH_CONFIG_INDEX.json)、[YAML索引](SEARCH_YAML_INDEX.json)、[旧十任务候选索引](SEARCH_HISTORICAL_CANDIDATES.json)、[只读重算脚本](audit_reuse_readonly.py)。
