# ImageNet-A：原生 r=1 九方向对照

更新 2026-10-02T12:02:40.825376+00:00；完整结果 7/9。

统一 seed2、rank10、r=1、rho=.05、10×20类、每任务50epochs、无EWC/Fisher/权重融合。
保留原生全局RNG/DataLoader与历史机制评估设置；AAA/FAA为任务等权口径。

| 方向 | AAA (%) | FAA (%) | LA (%) | BWT (pp) | 遗忘 (pp) | 状态 | 来源 |
|---|---:|---:|---:|---:|---:|---|---|
| 原版 GAM，γ=0.1 | 68.029 | 55.341 | 61.719 | -7.087 | 7.087 | completed | verified_completed_r1 |
| g1+g3，γ=0 | 67.861 | 54.726 | 61.583 | -7.619 | 7.619 | completed | verified_completed_r1 |
| 仅 g1 | 62.166 | 45.132 | 53.436 | -9.227 | 9.482 | completed | verified_completed_r1 |
| 仅 g3 | 65.993 | 50.679 | 56.728 | -6.721 | 7.011 | completed | verified_completed_r1 |
| 仅 g0 | 59.453 | 39.338 | 51.360 | -13.358 | 13.604 | completed | verified_completed_r1 |
| 仅 g2 | 63.838 | 49.989 | 54.856 | -5.408 | 6.280 | completed | verified_completed_r1 |
| g0+(g3-g2) | 64.102 | 47.870 | 54.714 | -7.604 | 7.604 | completed | verified_completed_r1 |
| A：g0+(g2-g1) | — | — | — | — | — | pending | new_native_r1_AB |
| B：g1+(g2-g1)⊥g1 | — | — | — | — | — | pending | new_native_r1_AB |

A回加跨探测点残差；B在全部可训练LoRA A/B与完整seen FC坐标上统一去掉残差的平行分量。
A/B均关闭原GAM γ正交修正，使用名义学习率，不做方向或有效步长范数匹配；B的投影是本次有意干预。
每组保留四次GAM探针与一次clean CE，逐batch验收真实方向、原FP32projector、扰动还原后的实际SGD。
记录投影系数、正交比例、夹角、参数与有效BA+FC更新范数；只观察步长，不改变优化器更新。
未完成格不填准确率。原七行来自完成凭据/原始矩阵核验，r=.25结果不进入本表。
方向与幅度共同变化，单seed不能证明统计显著、等价或普遍必要；跨GPU轨迹不宣称逐位一致。
