# 当前视觉 GAM+EWC：RQ2 结果组织与机制接续

用户已确认研究对象为当前视觉 GAM+EWC 实现。ImageNet-A / ViT-B16 / QV LoRA rank10 / seed0 的原恢复队列已全部完成。A 保留完成状态；本目录新增 GPU 方案尚未入队。

| 入口 | 用途 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 核验后的配置、A/B/C支持边界、两线最小补测、精确测量规模与结果判据 |
| [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) | 已完成整理与 proposed GPU 工作分开登记 |
| [CLAIMS_FROM_RESULTS.md](CLAIMS_FROM_RESULTS.md) | 独立同系列 result-to-claim 判断，partial，语义评估 provisional |
| [METHOD_ALIGNMENT.md](METHOD_ALIGNMENT.md) | 实际视觉 GAM dispatch、更新公式、raw Fisher/保护矩阵/参数坐标区分 |
| [figures/](figures/) | A表、B/C图、图注、LaTeX、数据与可复现脚本 |
| [followup_spec.json](followup_spec.json) | 结构化补测规格；planning-only，不能作为可运行队列配置 |
| [EVIDENCE_PACKET.json](EVIDENCE_PACKET.json) | 原始CSV数值与来源；[evidence_precheck.json](evidence_precheck.json)核对12个摘要SHA |

当前结论是部分支持：B的train256 raw trace较低，但test平均正损伤没有改善、Gmean容忍半径相同；C在真实十任务端点上显示更大位移、更小损伤，且t9端点直线段S_path较低。A的两行控制续训没有表现出更好的a21/a22，因此不能把A/B/C直接串成纯起点几何的因果链。

最小接续：GPU0/1补test raw Fisher与GAM-generated方向交叉诊断；GPU2/3补t1/t5端点路径，复用t9。第一批无训练任务。正式多seed确认和细半径网格按保留的主张决定，不机械重跑A或铺开Fisher estimator比较。

原始结果保存在 `../execution/results_seed0_recovery1/` 和 `../execution/report_seed0/`；旧历史随机增强半径图不与当前确定性诊断混算。`before_organization/`保存本次更新前的索引与执行清单。
