# 第二内层探针符号：实验追踪

状态快照：2026-10-06欧洲/马德里时间。代码已实现，CPU11/11与独立科学复核通过；已注册29jobs（14短测+1配对+14正式）。真实主机controller PID920989已运行，快照29pending、0短测/正式启动；新GPU短测及正式训练待空闲卡；实时进展以runtime/queue_state.json为准。原计划/配置已归档，旧队列保持原样。

| 作业 | 方向 | r | 状态 | 说明 |
|---|---|---:|---|---|
| B0_g0_probep1_residualp1_r1_k10_s2 | g0+(g3+−g2) | 1.0 | 同协议基线已注册，排队待短测 | seed2/rank10/ρ=.05 |
| B1_g1_probep1_residualp1_r1_k10_s2 | g1+(g3+−g2) | 1.0 | 复用已验收（CPU等价＋原validator/receipt） | seed2/rank10/ρ=.05 |
| N1_g0_probep1_residualm1_r1_k10_s2 | g0−(g3+−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N2_g1_probep1_residualm1_r1_k10_s2 | g1−(g3+−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N3_g0_probem1_residualp1_r1_k10_s2 | g0+(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N4_g1_probem1_residualp1_r1_k10_s2 | g1+(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N5_g0_probem1_residualm1_r1_k10_s2 | g0−(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N6_g1_probem1_residualm1_r1_k10_s2 | g1−(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| B0_g0_probep1_residualp1_r025_k10_s2 | g0+(g3+−g2) | 0.25 | 同协议基线已注册，排队待短测 | seed2/rank10/ρ=.05 |
| B1_g1_probep1_residualp1_r025_k10_s2 | g1+(g3+−g2) | 0.25 | 复用已验收（CPU等价＋原validator/receipt） | seed2/rank10/ρ=.05 |
| N1_g0_probep1_residualm1_r025_k10_s2 | g0−(g3+−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N2_g1_probep1_residualm1_r025_k10_s2 | g1−(g3+−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N3_g0_probem1_residualp1_r025_k10_s2 | g0+(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N4_g1_probem1_residualp1_r025_k10_s2 | g1+(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N5_g0_probem1_residualm1_r025_k10_s2 | g0−(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N6_g1_probem1_residualm1_r025_k10_s2 | g1−(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |

启动核验时间：2026-10-06T19:36:29.266876+00:00。GPU2–6全部忙碌，后台每60秒检查，max_parallel=5。
