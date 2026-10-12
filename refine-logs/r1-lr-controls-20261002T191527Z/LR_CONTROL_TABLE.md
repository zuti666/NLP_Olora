# ImageNet-A：十任务学习率与有效步长对照

更新 2026-10-02T19:27:35.017942+00:00；主对照 3/8，新增正式训练 0/5。

统一seed2、rank10、rho=.05、10×20类、50epochs/task；无EWC/Fisher/融合/replay，保留原生RNG、机制评估和workers8。
外半径r=1，只有G025为.25。LR基础首任务.04、后续.02，cosine floor为起点×.1；标称倍率同时作用于峰值和下限。

|组|最终方向|LR倍率|AAA (%)|FAA (%)|LA (%)|BWT (pp)|遗忘 (pp)|状态|来源|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| S | g1 | 1.0 | — | — | — | — | — | pending | new_full10_native |
| S1.5 | g1 | 1.5 | — | — | — | — | — | pending | new_full10_native |
| S2 | g1 | 2.0 | — | — | — | — | — | pending | new_full10_native |
| Sum | g1+g3 | 1.0 | 67.861 | 54.726 | 61.583 | -7.619 | 7.619 | completed | verified_historical_full10 |
| Mean2 | (g1+g3)/2 | 2.0 | — | — | — | — | — | pending | new_full10_native |
| Graw | g3 | 1.0 | 65.993 | 50.679 | 56.728 | -6.721 | 7.011 | completed | verified_historical_full10 |
| Gmatch | g3，匹配独立S实际有效步长 | 1.0 | — | — | — | — | — | pending | new_full10_native |
| G025 | g3，r=.25 | 1.0 | 66.230 | 50.799 | 57.317 | -7.242 | 7.366 | completed | verified_historical_full10 |

已有正常LR Mean（次级对照）：AAA 65.951，FAA 49.927；历史S仅作背景，新S完成前不填其分数。
Gmatch目标严格来自新S完成轨迹对应(task,step)的真实FP32 SGD有效BA＋完整seen FC位移，不来自G当前状态g1，不复制S模型参数。
每步匹配相对误差≤1e-3；未完成、不匹配、错误batch/RNG、无正根或非有限值停止后续放行。临时SGD scalar恢复后才推进原生scheduler。
Mean2与Sum在实数算术中具有相同名义更新；FP32 projector保留，短测不宣称完整轨迹逐位一致。
AAA/FAA为任务等权口径。已有控制的缺失逐步trace不追溯补造；单seed不作统计显著性结论。
- G025 vs Graw：AAA +0.236，FAA +0.120 pp。
