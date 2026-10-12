# 第二内层探针符号：实验追踪

状态快照：2026-10-06欧洲/马德里时间。代码已实现，CPU11/11与独立科学复核通过；已注册29jobs（14短测+1配对+14正式）。新GPU短测及正式训练待空闲卡；实时进展以runtime/queue_state.json为准。原计划/配置已归档，旧队列保持原样。

| 作业 | 方向 | r | 状态 | 说明 |
|---|---|---:|---|---|
| B0_g0_probep1_residualp1_r1_k10_s2 | g0+(g3+−g2) | 1.0 | 同协议基线已注册，排队待短测 | seed2/rank10/ρ=.05 |
| B1_g1_probep1_residualp1_r1_k10_s2 | g1+(g3+−g2) | 1.0 | CPU等价通过，原receipt复用待验收 | seed2/rank10/ρ=.05 |
| N1_g0_probep1_residualm1_r1_k10_s2 | g0−(g3+−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N2_g1_probep1_residualm1_r1_k10_s2 | g1−(g3+−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N3_g0_probem1_residualp1_r1_k10_s2 | g0+(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N4_g1_probem1_residualp1_r1_k10_s2 | g1+(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N5_g0_probem1_residualm1_r1_k10_s2 | g0−(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N6_g1_probem1_residualm1_r1_k10_s2 | g1−(g3−−g2) | 1.0 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| B0_g0_probep1_residualp1_r025_k10_s2 | g0+(g3+−g2) | 0.25 | 同协议基线已注册，排队待短测 | seed2/rank10/ρ=.05 |
| B1_g1_probep1_residualp1_r025_k10_s2 | g1+(g3+−g2) | 0.25 | CPU等价通过，原receipt复用待验收 | seed2/rank10/ρ=.05 |
| N1_g0_probep1_residualm1_r025_k10_s2 | g0−(g3+−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N2_g1_probep1_residualm1_r025_k10_s2 | g1−(g3+−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N3_g0_probem1_residualp1_r025_k10_s2 | g0+(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N4_g1_probem1_residualp1_r025_k10_s2 | g1+(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N5_g0_probem1_residualm1_r025_k10_s2 | g0−(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |
| N6_g1_probem1_residualm1_r025_k10_s2 | g1−(g3−−g2) | 0.25 | 代码/CPU通过，排队待短测 | seed2/rank10/ρ=.05 |

允许GPU2–6，排除0/1，最多5并行；双次空闲检查和共享index/UUID锁。全部短测与配对验收通过后正式训练自动接续。
