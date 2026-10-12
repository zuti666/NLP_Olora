# ImageNet-A r=1：seed0/1 纯 GAM 重复实验

更新：2026-10-04T01:06:57.510433+00:00

正式训练验收 6/60；已有 seed2 复用30项；三seed齐全 3/30 条件。
AAA/FAA均按任务等权计算；仅纳入通过验收的完整十任务结果。std为seed间样本标准差（ddof=1），不足两个seed留空。
原生5次CE协议与精确恢复4探针协议分开统计；同名梯度不跨协议合并。

| 条件 | 协议 | 已完成seed | AAA均值 ± std | FAA均值 ± std | 状态 |
|---|---|---|---:|---:|---|
| N01 original_GAM | 原生5CE | 0;1;2 | 68.333 ± 0.281 | 54.977 ± 0.351 | complete_3_seeds |
| N02 Sum | 原生5CE | 0;1;2 | 68.117 ± 0.347 | 54.810 ± 0.148 | complete_3_seeds |
| N03 g1 / S（审计基准） | 原生5CE | 0;1;2 | 63.204 ± 0.885 | 45.559 ± 0.511 | complete_3_seeds |
| N04 g3 | 原生5CE | 2 | 65.993 ± — | 50.679 ± — | partial_seeds |
| N05 g0 | 原生5CE | 2 | 59.453 ± — | 39.338 ± — | partial_seeds |
| N06 g2 | 原生5CE | 2 | 63.838 ± — | 49.989 ± — | partial_seeds |
| N07 ug | 原生5CE | 2 | 64.102 ± — | 47.870 ± — | partial_seeds |
| N08 A | 原生5CE | 2 | 64.595 ± — | 49.807 ± — | partial_seeds |
| N09 B | 原生5CE | 2 | 64.826 ± — | 47.590 ± — | partial_seeds |
| N10 Mean | 原生5CE | 2 | 65.951 ± — | 49.927 ± — | partial_seeds |
| N11 S1.5 | 原生5CE | 2 | 62.636 ± — | 46.419 ± — | partial_seeds |
| N12 S2 | 原生5CE | 2 | 62.899 ± — | 46.589 ± — | partial_seeds |
| N13 Mean2 | 原生5CE | 2 | 68.002 ± — | 54.730 ± — | partial_seeds |
| N14 Gmatch | 原生5CE | 2 | 64.683 ± — | 47.877 ± — | partial_seeds |
| P01 S1-01_CE_natural | 恢复4探针 | 2 | 60.337 ± — | 40.140 ± — | partial_seeds |
| P02 S1-02_Splus_natural | 恢复4探针 | 2 | 63.234 ± — | 44.578 ± — | partial_seeds |
| P03 S1-03_Sminus_natural | 恢复4探针 | 2 | 51.090 ± — | 25.368 ± — | partial_seeds |
| P04 S1-04_O_P_natural | 恢复4探针 | 2 | 65.735 ± — | 50.250 ± — | partial_seeds |
| P05 S1-05_O_D_natural | 恢复4探针 | 2 | 37.359 ± — | 23.187 ± — | partial_seeds |
| P06 S1-06_O_C_natural | 恢复4探针 | 2 | 65.774 ± — | 49.192 ± — | partial_seeds |
| P07 S1-07_A_P_natural | 恢复4探针 | 2 | 65.842 ± — | 49.325 ± — | partial_seeds |
| P08 S1-08_A_D_natural | 恢复4探针 | 2 | 52.065 ± — | 35.434 ± — | partial_seeds |
| P09 S1-09_A_C_natural | 恢复4探针 | 2 | 64.438 ± — | 47.185 ± — | partial_seeds |
| P10 S1-10_O_R_natural | 恢复4探针 | 2 | 38.561 ± — | 24.355 ± — | partial_seeds |
| P11 S1-11_A_R_natural | 恢复4探针 | 2 | 2.006 ± — | 0.882 ± — | partial_seeds |
| P12 S1-12_A_P_matched | 恢复4探针 | 2 | 63.325 ± — | 44.015 ± — | partial_seeds |
| P13 S1-13_A_D_matched | 恢复4探针 | 2 | 58.756 ± — | 43.944 ± — | partial_seeds |
| P14 S1-14_A_C_matched | 恢复4探针 | 2 | 63.632 ± — | 45.062 ± — | partial_seeds |
| P15 S1-15_A_R_matched | 恢复4探针 | 2 | 2.318 ± — | 1.111 ± — | partial_seeds |
| P16 S1-16_CE_matched | 恢复4探针 | 2 | 59.955 ± — | 39.565 ± — | partial_seeds |
