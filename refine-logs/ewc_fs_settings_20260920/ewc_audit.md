# 当前最高 FAA EWCLoRA 五数据集参数与 Fisher 核查

只使用当前 `ewc_best_selection.json` 的既定五条来源；未重选 run、未训练、未改表。日志/YAML明确值与当前可读源码推导分开说明，历史源码未完整锁定。

## 已记录的论文参数

| 数据集（表格顺序） | λ | LoRA r | 初始/后续任务 LR | epochs/task | seed | 执行路径 |
|---|---:|---:|---:|---:|---:|---|
| aircraft | 200 | 16 | 0.05 / 0.05 | 40 | 0 | α=0 two-probe wrapper |
| cars196 | 80000 | 16 | 0.05 / 0.05 | 40 | 1 | α=0 two-probe wrapper |
| cub200 | 200000 | 16 | 0.01 / 0.01 | 40 | 0 | α=0 two-probe wrapper |
| flowers | 10 | 16 | 0.0025 / 0.0025 | 40 | 1 | α=0 two-probe wrapper |
| het5 | 50000 | 16 | 0.02 / 0.02 | 30 | 2 | native one-pass |

共同明确值：batch128，SGD，momentum=weight_decay=0，cosine；`ewc_gamma=1`、`ewc_eta=0`、`ewc_max_batches=100`、`fisher_mode=batch_empirical`、`ewc_normalize_fisher=true`、`ewc_normalize_accumulated_fisher=false`、`ewc_fisher_norm_eps=1e-12`、`ewc_fisher_damping_beta=0`、`ewc_dual_ascent=false`、`ewc_skip_task_fisher=false`、`compute_full_fisher=false`。

各运行 λ、γ、Fisher normalization/damping、r、batch、Fisher批次上限在任务间固定；λ按数据集变化，不是全表同一 λ。`dual_ascent=false` 禁用 λ 的自适应更新；当前代码更新 λ 的唯一相关分支受该开关控制。初始 LR在各任务间相同，但任务内使用cosine调度，不能称每个SGD step学习率恒定。

五配置均写 `fs_rho=.05`、`fs_alpha=0`、`fs_alpha_mode=ratio`、`fs_xi=1e-12`、`fs_sam_equivalent=false`、`fs_adaptive=false`。四单数据集仍执行两次CE探测，α=0使最终额外FS方向为0；Het5 `fs_direction_mode=native_sgd`，只做一次CE backward，rho/alpha探测配置不参与该分支的更新。Cars/Flowers显式 `fs_direction_mode=alpha0`、probe/Fisher RNG均隔离；Aircraft/CUB旧日志没有记录这两个RNG开关，不应把现代码默认false倒推为历史已证实值。Het5明确preserve Fisher RNG=true，probe开关虽true但native分支无probe。

符号注意：这里能确认的“β”是 **Fisher isotropic damping β=0**；“ρ=.05”是wrapper探测半径。若论文的δ、β另有所指，不能直接替换。日志初始化中还会打印 `dual_delta=.0001`，但dual_ascent=false，这不是实际启用的遗忘约束预算。

## Fisher估计、数量与累计

当前实现使用Q/V的有效更新 ΔW=BA 坐标，共24个delta tensors；Fisher不在A/B参数坐标或classifier坐标计算。EWC惩罚为

`(λ/2) Σ_k F_past,k · (ΔW_k − ΔW_ref,k)^2`，其中η=0。

`batch_empirical` 实际是 **batch-gradient-square proxy**：先对当前batch的CE取mean，再对该mean的ΔW梯度平方，而不是逐样本梯度平方后平均。当前代码表达为

`F_raw,t = [Σ_b |B_b| · (∇_ΔW meanCE(B_b))²] / [Σ_b |B_b|]`。

随后按全部delta坐标的总trace归一化：`F_t = F_raw,t / (trace(F_raw,t)+1e-12)`；本表β=0，所以不做isotropic damping。累计为 `F_past,t = γ F_past,t−1 + F_t`，γ=1，不再归一化累计结果。因此任务级超参数恒定不意味着累计Fisher总质量恒定，也不是每任务重新丢弃旧Fisher。

最多读 **100个mini-batch**，训练batch128，因此至多12800个样本；小任务提前遍历完。不能在论文中写“固定100样本”“所有任务固定12800样本”或“每任务正好100 batch”。没有配置独立held-out Fisher split；当前相同配置路径会回退使用该任务训练loader，保留train-time图像增强，Fisher forward时model.eval()。不使用旧类replay和测试图像估Fisher。

历史log中可直接确认的实际Fisher样本数N_t及按batch128推算的实际批次数：

| 数据集 | 日志记录N_t（任务顺序） | 对应ceil(N_t/128)，均低于100 |
|---|---|---|
| Aircraft | 旧日志未记录实际N_t | 不编造逐任务值，仅确认配置上限100 |
| Cars | 652, 832, 809, 822, 849, 836, 839, 829, 825, 851 | 6, 7, 7, 7, 7, 7, 7, 7, 7, 7 |
| CUB | 旧日志未记录实际N_t | 不编造逐任务值，仅确认配置上限100 |
| Flowers | 240, 200, 200, 200, 200, 200, 200, 200, 200, 200 | 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 |
| Het5 | 9430, 6667, 8144, 2040, 3680 | 74, 53, 64, 16, 29 |

若论文将K_t定义为Fisher采样数，应先说明单位：batch数时是 `min(100, len(current_task_loader))`；样本数时是表中N_t（或最多12800的上限），不是把两个量混用。Cars/Flowers/Het5上述数值来自历史FinalMechanism而非当前目录重建；Aircraft/CUB仅配置及24-tensor更新日志有直接证据。

## retained anchor / merge的实现边界

训练使用persistent rank16 LoRA backbone：首任务建立一次，后续任务继续同一个adapter；每个任务开始复制当时的BA作为本任务 `ΔW_ref`。任务结束在已经训练完成的现有模型上估Fisher并累计，之后清空本任务reference；下一任务开始再次snapshot retained BA。可以表述“Fisher is estimated at the retained end-of-task model”。

**当前被审计的继承路径没有任务末 merge→rank-truncation/SVD→再估Fisher 操作。** 因此不能把当前EWCLoRA写成“完成低秩合并压缩后，在压缩后的anchor上计算Fisher”。这里只有持续更新的固定rank adapter，以及任务边界BA reference。若论文的retained anchor专指某个显式压缩/重构算子，需要另行对齐，不能由这个表的来源替代证明。

## 源码定位（当前源码，不是历史锁定版本）

- [EWC λ/γ/η、max_batches与RNG开关](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:72)
- [Q/V BA与task-start reference snapshot](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:228)
- [ΔW二次惩罚及1/2系数](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:270)
- [累计γF_old+F_new](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:309)
- [任务末Fisher→累计→清空reference](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:325)
- [训练前snapshot](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:376)
- [Fisher loader选择及训练loader回退](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:135)
- [batch-gradient-square proxy的定义](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:48)
- [task Fisher trace normalization](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:177)
- [累计不归一化时走父类求和](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:243)
- [Fisher当前任务batch循环/100上限/model.eval](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:381)
- [batch大小加权梯度平方及总样本数归一](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:348)
- [dual_ascent条件下才更新λ](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_gam.py:599)
- [Fisher damping β=0直接返回](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:179)
- [Het5 native单次CE分支](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:379)
- [干净参数处独立EWC梯度，加入CE方向之后](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/as1_normfisher_fishershape.py:423)
- [rho探测半径与epsilon](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/fishershape.py:155)
- [α=0方向组合](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/fishershape.py:182)
- [wrapper两次CE与回到干净参数](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/fishershape.py:287)
- [persistent backbone仅首任务创建](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:1512)
- [SeqLoRA after_task不执行merge](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:281)
- [无独立Fisher split时用train dataset/loader](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:392)

## 当前选中来源的参数/样本数证据

- **aircraft**：[原配置参数](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep/aircraft_sgd_ewc_lam200.yaml:22)；[原日志参数](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200_vit_base_patch16_224.log:18)；[首任务Fisher更新](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200_vit_base_patch16_224.log:469)、[末任务Fisher更新](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200/aircraft/0/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200/exp_run/10/as1fs_aircraft_sgdeq_r16_lr005_e40_ewc_lam200_vit_base_patch16_224.log:577)，实际N_t未记录。
- **cars196**：[原配置参数](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1.yaml:22)；[原日志参数](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:18)；[首任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:475)、[末任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam80000_beta000/cars196/1/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826/exp_run/20/cars196_g0_ewc_c000_rho005_lam80000_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:601)，各任务FinalMechanism行号 `[475, 489, 503, 517, 531, 545, 559, 573, 587, 601]`。
- **cub200**：[原配置参数](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_sgd_ewc_lambda_sweep_round2/cub200_sgd_ewc_lam200000.yaml:22)；[原日志参数](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200000/cub200/0/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000/exp_run/20/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000_vit_base_patch16_224.log:18)；[首任务Fisher更新](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200000/cub200/0/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000/exp_run/20/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000_vit_base_patch16_224.log:468)、[末任务Fisher更新](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_sgd_ewc_lambda_sweep_round2/logs_inc/as1_normfisher_fishershape/as1fs_sgdeq_ewc_lam200000/cub200/0/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000/exp_run/20/as1fs_cub200_sgdeq_r16_lr001_e40_ewc_lam200000_vit_base_patch16_224.log:576)，实际N_t未记录。
- **flowers**：[原配置参数](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1.yaml:22)；[原日志参数](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:18)；[首任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:475)、[末任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_ratiofs_ewc_paper/ratiofs_paper_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_g0_ewc_c000_rho005_lam10_beta000/flowers/1/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826/exp_run/10/flowers102_g0_ewc_c000_rho005_lam10_beta000_s1_ratiofs_paper_v1_20260826_vit_base_patch16_224.log:601)，各任务FinalMechanism行号 `[475, 489, 503, 517, 531, 545, 559, 573, 587, 601]`。
- **het5**：[原配置参数](/data/140-1/users/liying/CopyFlatCV/config_5datasets/as1_fivearm_native_controls/fivearm_native_v1_20260826/het5_native_ewc_rho005_lam50000_beta000_s2.yaml:22)；[原日志参数](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log:18)；[首任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log:475)、[末任务实际N_t](/data/140-1/users/liying/CopyFlatCV/outputs_logs/5datasets_as1_fivearm_native_controls/fivearm_native_v1_20260826/logs_inc/as1_normfisher_fishershape/as1fs_native_ewc_rho005_lam50000_beta000/het5/2/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826/exp_run/1/het5_native_ewc_rho005_lam50000_beta000_s2_fivearm_native_v1_20260826_vit_base_patch16_224.log:531)，各任务FinalMechanism行号 `[475, 489, 503, 517, 531]`。

选择来源本身属于已审计历史候选中的最高测试FAA运行；不是固定预验证参数得到的无选择偏差seed均值。当前参数段应保留该选择限制、五列不同λ/seed和Het5 30轮、native/wrapper差异。
