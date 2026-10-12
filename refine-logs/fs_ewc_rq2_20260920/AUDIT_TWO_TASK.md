# Two-task 与 full-model 资产审查

日期：2026-09-20。只读核查本地日志、JSON、脚本与 checkpoint 文件存在性；未运行训练、模型评估或 GPU 作业。没有找到适用的 AGENTS.md。下文路径均以 `/data/140-1/users/liying/CopyFlatCV/` 为根，另有说明除外。

更新说明（2026-09-20，后续扩展检索）：本文下方“19.885→10.946、87.95→91.82未溯源”仅适用于当时的 CopyFlatCV 搜索范围。现已在 `/data/115-2/users/liying/low-rank-cl-new` 找到用户六行 ImageNet-R 表的18份原日志和配置，42个数字均复算通过；其中91.82确为该旧LoRA cohort的Task-1 shaping结果。见 [旧表精确来源](/data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/LORA_PROVENANCE.md) 和 [本轮核验](execution/table_revision_20260920/USER_TABLE_PROVENANCE.md)。它使用Adam、fresh K/V adapter、λ=1e7，仍不能代替本次ImageNet-A/SGD/QV/λ2000的同设置对照。下文保留原审查记录。

## 结论

主实验 ImageNet-A、LoRA rank10、λ=2000 的 **普通 Task0 → SGD+EWC Task1** 已有资产；**shaped Task0 → SGD+EWC Task1** 未找到。现有主序列是 SGD→SGD 与 GAM→GAM，后者不能作为关闭后续 shaping 的第二行。

邻近目录确有完整 `control` / `factored_s1` 对照，属于 **ImageNet-R 20+20、ratio-FS、λ=1,200,000、seeds3/4/5**，只能作为独立背景实验，不能代替主配置。该组相对自身起点的平均 accuracy 损伤没有改善，因此不能把其存在解释成新主张已获支持。

用户引用的 `trace 19.885 → 10.946` 与配对 `old accuracy 87.95 → 91.82` 未核验到对应原始 LoRA 两行，不能引用为已经复现的结果。`91.82` 确实出现在 full-model seed1 的普通 Task0 训练结果中，但不构成这组配对证据。

## 1. 指定目录实际有什么

| 目录 | 已核验内容 | 可用范围 |
|---|---|---|
| `outputs_logs/verify_imagenet_r_1x1_lora_fixedfs_upper_bounds` | task0-only、task1-only、joint40 上界；native seeds0/1/2 与另一组 seeds3/4/5；各实验只有一个训练阶段 | 上界背景，不能当两任务遗忘对照 |
| `outputs_logs/verify_imagenet_r_1x1_fullmodel_ratio_twostage` | 只有 seed9、1 epoch smoke 的两个 Task0 anchor 和 baseline/control Task1 metrics；control 的 a11/a21/a22=60.69/7.39/0.00 | 调试记录；不能当正式 FS 效果或 full-model 成功复验 |
| `outputs_logs/fishershape_imagenetr_1x1_fullmodel` | 普通 Task0 seeds0/1/2 accuracy=91.67/91.82/91.67，以及 seed0 task0-only 上界92.45；FS Task0 与 joint 日志没有完成 metrics | 有普通 full-model anchor，未核验出完整 shaped/full-model pair |

Full-model smoke 的协议在 `scripts/run_verify_imagenet_r_1x1_fullmodel_fishershape_staged_v2.sh:126` 显式将 epochs 设为1；目录中也只有 `smoke_ratiofs_fullmodel_full10_c0p20_lam1200000_s9_v1`。不能由 launcher 支持 full10 推定 full10 已完成。

## 2. 主实验 A 的最小补足

现成主配置日志：

- `logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/sgd_T10_seed0.log`：17行 rank10，19行 λ2000，23行 Fisher 归一化，36行 optimizer_type=sgd，74行恢复 SGD Task0。
- `outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/sgd/T10/seed0/checkpoints/task_01.pt` 存在，文件大小 688,097,974 bytes。
- 对应 trainer metrics：`outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/sgd_from_task0_T10_lr0.02_e50/imageneta/0/s1c_sgd_from_task0_T10_e50_lam2000_s0/exp_run/20/s1c_sgd_from_task0_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json`，`cnn.matrices.final` 前两行给出 a11=85.34、a21=82.08、a22=75.20。
- GAM Task0 anchor：`outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/gam/seed0/checkpoints/`。现成 GAM 全序列第二阶段仍 optimizer_type=gam；其 a11/a21/a22=87.62/81.76/81.50 不能填入“后续同为 SGD+EWC”的 shaped 行。

补足顺序：先验证旧 SGD Task1 的源码、加载与 loss 口径；从已有 GAM Task0 只补 **一个 Task1 的 native SGD+EWC**，关闭后续 shaping。若当前源码或环境无法与旧 SGD 分支公平对齐，则从两个已存 Task0 anchor 同源重跑两条单 Task1；不重训 anchor，不默认四格。

新增离线测量应在两个 anchor 与两个 Task1 endpoint 上固定旧任务数据、旧分类头和 eval 变换，输出 old loss 增量、有效权重 L2 位移、未归一化逐样本 empirical Fisher trace。旧 trainer accuracy 可能使用随 seen classes 扩展的分类头；报告表应同时注明原生 CIL accuracy 与固定旧头 loss 的口径。

搜索范围：扫描本仓库 `logs/` 与 `outputs_logs/` 的全部 2082 个 `.log` 文件前181行配置，提取 dataset/model/optimizer/resume 及分阶段开关。ImageNet-A LoRA 恢复记录共36份（含 shell/trainer 重复）只出现 `gam anchor → gam` 和 `sgd anchor → sgd`；没有 `gam anchor → optimizer_type=sgd`，也没有 `phase_a_tasks`、`gam_task_b`、`fs_task_b` 的混合候选。这是本次已检查范围内的未发现，不声称其他文件系统一定不存在。

## 3. 已找到的独立 ImageNet-R ratio-FS pair

实验根：`outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage`；tag：`ratiofs_l12c20_full_twostage_s345_v1`。

设置 metadata 在 `result_summary/verify_imagenet_r_1x1_lora_ratio_full_twostage/ratiofs_l12c20_full_twostage_s345_v1/setting_metadata.csv:3` 和 `:7`：

- control：native_sgd → native_sgd + EWC。
- factored_s1：ratio → native_sgd + EWC。

这里 `factored_s1` 表示只在第一阶段做 shaping，**不是** Fisher 测量使用 LoRA factor 坐标。训练 shaping 在 factor 空间，EWC 保护/漂移在有效 delta-W 空间。

| seed | anchor → future | a11 | a21 | a22 | a11−a21 (pp) | 原 delta_norm 字段=平方位移 |
|---|---|---:|---:|---:|---:|---:|
| 3 | native → SGD+EWC | 89.47 | 84.43 | 89.40 | 5.04 | 0.18372286 |
| 3 | ratio → SGD+EWC | 89.47 | 84.75 | 89.26 | 4.72 | 0.19942320 |
| 4 | native → SGD+EWC | 88.84 | 85.85 | 88.01 | 2.99 | 0.18140464 |
| 4 | ratio → SGD+EWC | 88.84 | 86.01 | 88.15 | 2.83 | 0.20073381 |
| 5 | native → SGD+EWC | 89.78 | 87.42 | 86.05 | 2.36 | 0.18228763 |
| 5 | ratio → SGD+EWC | 90.09 | 87.11 | 86.05 | 2.98 | 0.20286462 |
| mean | native → SGD+EWC | 89.363 | 85.900 | 87.820 | 3.463 | 0.18247171 |
| mean | ratio → SGD+EWC | 89.467 | 85.957 | 87.820 | 3.510 | 0.20100721 |

从原 metrics JSON 提取；没有用日志最后一个 `CurTask_Test` 代替 a22（其分类头口径不同）。没有进行显著性检验。旧 loss 增量、anchor/endpoint empirical Fisher trace 在 summary 中为空，本次不能补成结果。

具体原始证据：

- `logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/ratiofs_l12c20_full_twostage_s345_v1/stage2/control_seed3.log:73` 为 native_sgd；`:80` 恢复 native_sgd anchor；`:471` 是最终 old/new accuracy。
- 同目录 `factored_s1_seed3.log:73` 为 native_sgd；`:80` 恢复 ratio anchor；`:82` 确认旧优化器 FisherShape；`:100` 记录本阶段 direction=native_sgd；`:469` 是最终 old/new accuracy。
- 同目录两个日志 `:107` / `:109` 附近明确恢复各自 Fisher state，trace=1。这是两张分别归一化的保护矩阵，**不等于矩阵相同**。
- `scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh:218` 的公共配置：rank10、20+20、batch128、cosine、50 epochs、lr0.02、momentum0、wd0、batch_empirical、归一化 Fisher、dampingβ0.2。λ1.2e6、FS c0.20、ρ0.05。
- `models_LoRAbasedCL/as1_normfisher_fishershape.py:379` 的 native_sgd 分支是一次 CE backward 加 EWC 后普通 optimizer.step()，不进行 shaping。
- `models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:353` 至 `:356` 累计 update_sq；`:383` 把它命名为 delta_norm。因此表中的 0.18/0.20 是 **L2²**，不能直接标作扰动半径。

Checkpoint：六个 Task0 anchor 的目录均有 `task_00.pt`、`lora_w_a_0.pt`、`lora_w_b_0.pt`、`lora_meta_0.json`、`fc_state_0.pt`、`fisher_state_0.pt`、`experiment_identity_0.json` 和 `parent_artifacts.sha256`。六个 Task1 endpoint 均有 `lora_w_a_1.pt`、`lora_w_b_1.pt`、`lora_meta_1.json`、`fc_state_1.pt`、CLs weight/bias；配置 `save_final_model=False`，不存在 final_model 不应误判训练缺失。这里只检查存在性，没有加载验证端点复原。

## 4. 其它候选与测量注意

`verify_imagenet_r_1x1_lora_fixedfs/fixedfs_exploratory_s012_a005_lam4e5_rho005_v1` 中 seed0 control 已完成，a11/a21/a22=90.25/85.69/88.70；factored_s1 日志止于 Task1 epoch40，未找到其最终 metrics/checkpoint，不能拼出完整 pair。

较早 `verify_imagenet_r_1x1_lora_fishershape/lora_fs_1x1_s012_v1` 的 λ8e5 seeds0/1/2 也有 `control/factored_s1` 表，但未来端是 alpha=0 FisherShape wrapper；不如后续 native_sgd 实现严格，且与主配置仍不同。应保留为历史记录，不据此宣称新主实验完成。

保护矩阵是 trace-normalized batch-gradient Fisher proxy；`fisher_mode=batch_empirical` 不代表逐样本 empirical Fisher。现行 `as1_normfisher_gam.py:177` 的 trace normalization 能记录 raw_trace，但先前日志中的 trace=1 只反映归一化。诊断必须明确 estimator、样本、坐标，不将 normalized protection trace 当作 raw sensitivity。

用户指定 `scripts/run_imagenet_r_task1_fqa_lora_factor_adam_gam.sh:10` 默认半径0到1、K100；`:15` 委托另一 wrapper，`:16` 启用 `--factor-fqa`，`:18` 读取 `_endpoint`。这属于旧 factor FQA 入口，不等于本文所需固定有效权重坐标的邻域 raw Fisher 或最终路径 probe。
