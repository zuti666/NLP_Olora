# ImageNet-A：r=1 正反探针与差分符号

更新：2026-10-03T14:43:41.669183+00:00

seed2、rank10、rho=.05，10任务×50 epochs；新完整训练12项，半径无关基准复用4项。
匹配参考为候选自身当前状态的raw g1+实际BA＋全部已见FC位移。
未验收完整训练的成绩留空；历史基准的r=.25外点诊断不能用于r=1机制统计。

| 条件 | 步长 | 状态 | 已完成任务 | AAA (%) | FAA (%) | GPU |
|---|---|---|---:|---:|---:|---:|
| S1-01_CE_natural_s2 | natural | validated_reuse | 10 | 60.337 | 40.140 | — |
| S1-02_Splus_natural_s2 | natural | validated_reuse | 10 | 63.234 | 44.578 | — |
| S1-03_Sminus_natural_s2 | natural | validated_reuse | 10 | 51.090 | 25.368 | — |
| S1-04_O_P_natural_s2 | natural | completed | 10 | 65.735 | 50.250 | 1 |
| S1-05_O_D_natural_s2 | natural | completed | 10 | 37.359 | 23.187 | 2 |
| S1-06_O_C_natural_s2 | natural | completed | 10 | 65.774 | 49.192 | 6 |
| S1-07_A_P_natural_s2 | natural | completed | 10 | 65.842 | 49.325 | 1 |
| S1-08_A_D_natural_s2 | natural | completed | 10 | 52.065 | 35.434 | 4 |
| S1-09_A_C_natural_s2 | natural | completed | 10 | 64.438 | 47.185 | 5 |
| S1-10_O_R_natural_s2 | natural | completed | 10 | 38.561 | 24.355 | 3 |
| S1-11_A_R_natural_s2 | natural | completed | 10 | 2.006 | 0.882 | 2 |
| S1-12_A_P_matched_s2 | matched | completed | 10 | 63.325 | 44.015 | 6 |
| S1-13_A_D_matched_s2 | matched | completed | 10 | 58.756 | 43.944 | 1 |
| S1-14_A_C_matched_s2 | matched | completed | 10 | 63.632 | 45.062 | 4 |
| S1-15_A_R_matched_s2 | matched | completed | 10 | 2.318 | 1.111 | 3 |
| S1-16_CE_matched_s2 | matched | validated_reuse | 10 | 59.955 | 39.565 | — |
