# 实验追踪：末端中心差分与联合后向替换

状态：plan_only；automatic_start=false。首轮seed2、rank10、rho=.05、r=1、ImageNet-A10×20。新执行器尚未实现，无新增排队任务。

| Run ID | 阶段 | 最终方向 | 优先级 | 状态 | AAA / FAA | 备注 |
|---|---|---|---|---|---:|---|
| FF | 参考 | g0+g3F+−g2F | MUST | REUSE_VALIDATED | 65.176 / 47.584 | 当前严格B0；待新入口回归门 |
| FB | 参考 | g0+g2F−g3F− | MUST | REUSE_VALIDATED | 65.385 / 48.632 | 当前严格N5；待新入口回归门 |
| CPLUS | M1→M2 | g0+(g3F+−g3F−)/2 | MUST | PLAN_ONLY | — | 每步5CE，尚未注册 |
| CMINUS | M1→M2 | g0−(g3F+−g3F−)/2 | MUST | PLAN_ONLY | — | 与CPLUS同构造、仅修正反号 |
| BF | M1→M2 | g0+g3B+−g2B | MUST | PLAN_ONLY | — | 后向定位、正向末端，每步4CE |
| BB | M1→M2 | g0+g2B−g3B− | MUST | PLAN_ONLY | — | 完全后向局部读数，每步4CE，无正侧诊断 |

多seed、有效步长匹配和半径扩展均为结果驱动的条件后续，不登记当前任务。现有第二探针队列不修改。
