# 三层证据链实验追踪

状态核对UTC：2026-10-11T20:17:41.453495+00:00。三seed SGD50父模型已复用；7项准备/短测/配对/数值验收通过。正式分支运行3/6、已完成0/6；其余3条及3组离线诊断自动接续。

| Run | 用途 | seed | 状态 | GPU |
|---|---|---:|---|---:|
| prepare_start_s2 | prepare | 2 | completed | 4 |
| prepare_start_s0 | prepare | 0 | completed | 5 |
| prepare_start_s1 | prepare | 1 | completed | 6 |
| smoke_S_s2 | training | 2 | completed | 4 |
| smoke_R_s2 | training | 2 | completed | 5 |
| smoke_pairing_s2 | pairing | 2 | completed | 4 |
| smoke_diagnostics_s2 | diagnostic | 2 | completed | 4 |
| formal_S_s2 | training | 2 | running | 4 |
| formal_R_s2 | training | 2 | running | 5 |
| formal_S_s0 | training | 0 | running | 6 |
| formal_R_s0 | training | 0 | pending | — |
| formal_S_s1 | training | 1 | pending | — |
| formal_R_s1 | training | 1 | pending | — |
| offline_s2 | diagnostic | 2 | pending | — |
| offline_s0 | diagnostic | 0 | pending | — |
| offline_s1 | diagnostic | 1 | pending | — |

S=SAM(.05)1–25→SGD26–50；R=S_F g2(下降首探针η||g0||、外.25)1–25→SGD26–50。50轮cosine连续，无重置。

runtime/queue_state.json为实时权威；本表为上述时间快照。正式结果按完整验收填FORMAL_RESULTS.csv/RESULTS.md，配对完整400增强输入。

启动证据：LAUNCH_VERIFIED.json；完整计划：EXPERIMENT_PLAN.md；独立GPU验收：docs/GPU_GATE_REVIEW.json。
