# ImageNet-A: r=1 七方向对照

更新 2026-10-02T07:27:22.902121+00:00；完成 4/7。

统一 seed2、LoRA rank10、外半径 r=1、内层 rho=.05、10×20类、每任务50 epochs。
AAA/FAA 为任务等权口径；EWC/Fisher/权重融合关闭。保留历史原生数据加载及机制评估。

| 方向 | AAA (%) | FAA (%) | 状态 | 来源 |
|---|---:|---:|---|---|
| 原版 GAM，γ=0.1 | 68.029 | 55.341 | completed | historical_verified |
| g1+g3，γ=0 | 67.861 | 54.726 | completed | historical_verified |
| 仅 g1 | 62.166 | 45.132 | completed | historical_verified |
| 仅 g3 | 65.993 | 50.679 | completed | historical_verified |
| g0 | — | — | pending | new_native_extension |
| g2 | — | — | pending | new_native_extension |
| g0+(g3-g2) | — | — | pending | new_native_extension |

原版/双分支/单分支包含方向和更新幅度的共同影响；未做额外范数匹配。
g0 保留四次探测，因此是同探测流程下的 SGD 方向对照；浮点扰动还原与纯单次前向 SGD 可能存在微小差别。
新三组仅在训练 optimizer 实例上选择最终方向，不改变机制评估中的 GAM。
单 seed 结果属于当前配置的消融证据，不能直接声称统计显著或普遍必要。
尚未完成的格不填分数；短测和 r=.25 结果不进入本表。
