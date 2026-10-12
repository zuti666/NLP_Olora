# 三层证据链实验追踪

核对时间：2026-10-12T11:14:35.577906+02:00（Europe/Madrid）。正式训练验收6/6；准备/短测7/7；正式离线诊断验收0/3，其中seed2数值门槛失败，seed0/1依赖受阻、未启动。

队列状态：halted_failure，控制器及全部记录进程已退出。GPU4/5/6核对时均2MiB、0%，无计算PID。未重新启动队列。queue_state.json的counts含一个过时running计数；按jobs实际重算为completed13/failed1/pending2/running0。

| Run | 用途 | seed | 状态 | 原执行GPU |
|---|---|---:|---|---:|
| prepare_start_s2 | prepare | 2 | completed | 4 |
| prepare_start_s0 | prepare | 0 | completed | 5 |
| prepare_start_s1 | prepare | 1 | completed | 6 |
| smoke_S_s2 | training | 2 | completed | 4 |
| smoke_R_s2 | training | 2 | completed | 5 |
| smoke_pairing_s2 | pairing | 2 | completed | 4 |
| smoke_diagnostics_s2 | diagnostic | 2 | completed | 4 |
| formal_S_s2 | training | 2 | completed | 4 |
| formal_R_s2 | training | 2 | completed | 5 |
| formal_S_s0 | training | 0 | completed | 6 |
| formal_R_s0 | training | 0 | completed | 4 |
| formal_S_s1 | training | 1 | completed | 6 |
| formal_R_s1 | training | 1 | completed | 5 |
| offline_s2 | diagnostic | 2 | failed | 6 |
| offline_s0 | diagnostic | 0 | pending（前置数值门槛失败） | — |
| offline_s1 | diagnostic | 1 | pending（前置数值门槛失败） | — |

S：SAM(ρ=.05) e1–25→SGD e26–50；R：S_F g2（下降首探针θ−ηg0、外r=.25）e1–25→SGD e26–50。共用每seed Task0 SGD50父模型；50轮cosine连续，无重置。

训练6/6通过独立复核：2,400次SGD提交、4,200次梯度读取，全部3seed配对输入/共享起点正确，51执行文件与190原生source哈希未变。

首个正式离线诊断seed2/S/e1/batch0：同点梯度通过，但Hg0与H(Hg0)的登记距离有限差分误差没有两个相邻点≤.08；无OOM/NaN，不等同于训练失败。仅尝试1/36状态批次；0/72尺度修正、0/24有效步长匹配，不能宣称H²g机制成立。

结果：RESULTS.md / FORMAL_RESULTS.csv。最新独立核验：audits/status_20261012/INDEPENDENT_TRAINING_AUDIT.md；数值失败：OFFLINE_FAILURE_AUDIT.md；实际进程：HOST_PROCESS_AUDIT.json。
本追踪表为时间快照；原始队列状态、退出凭证和原始失败记录均保留。
