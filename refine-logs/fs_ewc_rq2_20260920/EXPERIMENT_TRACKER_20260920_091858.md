# FS-EWC RQ2：执行清单

更新：2026-09-20。已完成 CPU 来源审查和旧数据重分析；未启动新训练、模型前向或 GPU Fisher 测量。详细设置见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

| ID | 顺序 | 工作与配置 | 优先级 | 当前状态 | 训练需求/交付 |
|---|---|---|---|---|---|
| S01 | 1 | seed0旧JSON：两方法×train/test，G/Q95/prefix半径；epsilon=.001/.005/.01/.02/.05 | MUST | DONE_CPU | 无；324行damage、40行半径CSV；保留test不一致结果 |
| S02 | 1 | anchor/t1–t9/源码/两任务pair/有效权重/数据变换/环境；补r0/端点forward/raw-F/吞吐 | MUST | PARTIAL | 20 checkpoint CPU结构、head、backbone、位移检查通过；模型forward与梯度检查未做 |
| S03 | 2a | 邻域raw trace：2 anchors，r=0/.5/1/2/5/10/20，K16公共方向，固定256 train | MUST | TODO_IMPLEMENT_EVAL | 无训练；194状态，49,664逐样本梯度；保存确定性CE和raw trace |
| S04 | 2b | 2 anchors×2 actual t9方向，r=0/.25/.5/1/1.5/2，D与raw trace | MUST | TODO_IMPLEMENT_EVAL | 无训练；最多24状态；r0可复用 |
| S05 | 2b | 全task旧任务损伤/accuracy/drift；seed0 t9路径9节点，I与S_path | MUST | PARTIAL_CPU | 全task CSV已出；路径梯度/求积未测；最多18状态 |
| S06 | 5 | r=1/5/10/20，等预算GAM-directed proxy，方向交叉评估 | NICE-TO-HAVE | CONDITIONAL | 无训练；可直接复用inner-search才做，不称worst-case |
| S07 | 3 | A表：普通/shaped Task0→相同SGD+EWC Task1；a11/a21/a22/D0/drift/raw trace | MUST | MISSING_SHAPED_TO_SGD | 最少补1段作历史对照；严格配对需验证历史RNG，否则两个现有anchor各续1个Task1；无默认四格 |
| S08 | 4 | seed1/2同配置，B/C共用完整序列，所有seed必做B1与t9路径 | MUST-FORMAL | TODO | 预计4条完整十任务链；B1+t9新增108,544逐样本梯度；另计A分支/B2 |
| S09 | 4 | 3 seeds、r=0:.25:20、K100公共方向，全train/test，G/Q95/半径 | MUST-FORMAL | TODO | 无额外训练；96,012次数据集前向；历史非配对曲线不混合统计 |
| S10 | 5 | seed0 t1/t5路径、full-model轻量邻域、256与全train估计稳定性 | OPTIONAL | CONDITIONAL | 优先复用；full展示目录尚需匹配；不默认完整路径 |
| S11 | 5 | 仅必要时加强保护隔离：无保护的关键pair | OPTIONAL | CONDITIONAL | 缺分支才补；不默认2×2×3 seeds |

## 首批三个工作包

1. **S01+S02剩余部分**：S01已完成；为S02实现离线endpoint恢复，固定新的train评估视图，完成test历史复现/新视图端点一致性、逐样本梯度检查和计时。
2. **S03+S04**：同一个离线入口，共用256样本、坐标、公共方向；raw Fisher不使用归一化保护矩阵。
3. **S05剩余部分**：旧全task CSV已有，补两个t9路径的9节点测量、5/9节点求积检查及S_path。

S07的CPU配置准备可与上述离线评估并行；其训练不属于本轮已执行工作。训练/评估顺序取决于测量正确性与资源，不取决于结果是否支持预期。

## 防止错误复用

- 主线是 ImageNet-A/rank10/lambda2000/no averaging；ImageNet-R ratio-FS/lambda1.2e6 pair不填主表。
- `upper_bounds`不是顺序两任务对照；full-model seed9 smoke不算正式结果；指定top1% Fisher-cache作业失败不算完成。
- 旧半径train为一次缓存随机增强视图，新deterministic train需独立标记；旧原始train数值不作为新transform复现硬门槛。
- 旧跨T的动态trace约1是归一化；旧方向索引不代表方法间配对；`delta_norm`字段须区分范数与平方范数。
- 新Fisher/路径代码尚未实现；配置和计划完整不等于测量完成。
