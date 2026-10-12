# ImageNet-A r=1 seed0/1 重复实验追踪

记录：2026-10-03 18:21 CEST。全部状态为计划草案；无controller、无自动接续、无训练进程。

|Run ID|区块|协议|设置|seed|前置正式任务|GPU池|状态|
|---|---|---|---|---:|---|---|---|
|N01_r1_k10_s0|B1|native_global_rng_5CE|original_GAM|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N01_r1_k10_s1|B1|native_global_rng_5CE|original_GAM|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N02_r1_k10_s0|B1|native_global_rng_5CE|Sum|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N02_r1_k10_s1|B1|native_global_rng_5CE|Sum|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N03_r1_k10_s0|B1|native_global_rng_5CE|g1 / S（审计基准）|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N03_r1_k10_s1|B1|native_global_rng_5CE|g1 / S（审计基准）|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N04_r1_k10_s0|B1|native_global_rng_5CE|g3|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N04_r1_k10_s1|B1|native_global_rng_5CE|g3|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N05_r1_k10_s0|B1|native_global_rng_5CE|g0|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N05_r1_k10_s1|B1|native_global_rng_5CE|g0|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N06_r1_k10_s0|B1|native_global_rng_5CE|g2|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N06_r1_k10_s1|B1|native_global_rng_5CE|g2|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N07_r1_k10_s0|B1|native_global_rng_5CE|ug|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N07_r1_k10_s1|B1|native_global_rng_5CE|ug|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N08_r1_k10_s0|B1|native_global_rng_5CE|A|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N08_r1_k10_s1|B1|native_global_rng_5CE|A|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N09_r1_k10_s0|B1|native_global_rng_5CE|B|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N09_r1_k10_s1|B1|native_global_rng_5CE|B|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N10_r1_k10_s0|B2|native_global_rng_5CE|Mean|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N10_r1_k10_s1|B2|native_global_rng_5CE|Mean|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N11_r1_k10_s0|B2|native_global_rng_5CE|S1.5|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N11_r1_k10_s1|B2|native_global_rng_5CE|S1.5|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N12_r1_k10_s0|B2|native_global_rng_5CE|S2|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N12_r1_k10_s1|B2|native_global_rng_5CE|S2|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N13_r1_k10_s0|B2|native_global_rng_5CE|Mean2|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|N13_r1_k10_s1|B2|native_global_rng_5CE|Mean2|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|N14_r1_k10_s0|B2|native_global_rng_5CE|Gmatch|0|N03_r1_k10_s0|4/5/6|PLANNED_NOT_REGISTERED|
|N14_r1_k10_s1|B2|native_global_rng_5CE|Gmatch|1|N03_r1_k10_s1|4/5/6|PLANNED_NOT_REGISTERED|
|P01_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-01_CE_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P01_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-01_CE_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P02_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-02_Splus_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P02_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-02_Splus_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P03_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-03_Sminus_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P03_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-03_Sminus_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P04_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-04_O_P_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P04_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-04_O_P_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P05_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-05_O_D_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P05_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-05_O_D_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P06_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-06_O_C_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P06_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-06_O_C_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P07_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-07_A_P_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P07_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-07_A_P_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P08_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-08_A_D_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P08_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-08_A_D_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P09_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-09_A_C_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P09_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-09_A_C_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P10_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-10_O_R_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P10_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-10_O_R_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P11_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-11_A_R_natural|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P11_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-11_A_R_natural|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P12_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-12_A_P_matched|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P12_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-12_A_P_matched|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P13_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-13_A_D_matched|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P13_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-13_A_D_matched|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P14_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-14_A_C_matched|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P14_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-14_A_C_matched|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P15_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-15_A_R_matched|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P15_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-15_A_R_matched|1|—|4/5/6|PLANNED_NOT_REGISTERED|
|P16_r1_k10_s0|B3|four_probe_exact_restore_rng_replay|S1-16_CE_matched|0|—|4/5/6|PLANNED_NOT_REGISTERED|
|P16_r1_k10_s1|B3|four_probe_exact_restore_rng_replay|S1-16_CE_matched|1|—|4/5/6|PLANNED_NOT_REGISTERED|
