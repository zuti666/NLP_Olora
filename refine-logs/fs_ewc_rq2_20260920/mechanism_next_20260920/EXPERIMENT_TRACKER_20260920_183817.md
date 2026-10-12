# RQ2 机制检验接续清单

范围：2026-09-20 用户最新 A/B/C 三块方案。新增 GPU 工作当前均为 proposed，未入队；已完成 A 不自动重训。

| ID | 工作 | 状态 | 资源/依赖 | 交付/验收 |
|---|---|---|---|---|
| R0 | 旧恢复队列15作业与12测量摘要核验 | DONE | CPU | EVIDENCE_PACKET、SHA存在性检查；A/B/C seed0已完成范围确定 |
| R1 | 方法版本/坐标/历史源码审查 | DONE_CURRENT_VISUAL_GAM_SELECTED | CPU | METHOD_ALIGNMENT；用户指定当前视觉GAM+EWC；与9/17稿差异作为溯源说明 |
| R2 | A表、B/C主图与caption组织 | DONE_CPU | CPU | 原始CSV可复现、PDF/PNG、负结果可见 |
| R3 | 独立result-to-claim复核 | DONE_PARTIAL_PROVISIONAL | CPU | same-family/provisional判据与缺口 |
| B-test | test307 rawF，r0/1/5/10/20，K16 | PROPOSED_NOT_QUEUED | GPU0/1；协议自检 | 130状态、39910梯度样本求值，train/test分开 |
| B-directed | 2源方向×2anchors×4r | PROPOSED_NOT_QUEUED | GPU0/1；两方向资产 | 16状态、9008梯度样本求值+搜索；proxy/exact目标区分 |
| C-mid | t1/t5 的方向路径积分；复用t9 | PROPOSED_NOT_QUEUED | GPU2/3；target-aware代码 | 基础34状态/8704梯度样本求值；9→17→33自适应求积 |
| B-radius | 公共细网格loss-only tolerance复核 | CONDITIONAL | 独立离线工作 | 2368新增状态；若需要半径主张才执行，不以结果符号择点 |
| Replicate | 同方法版本seed1/2 | CONDITIONAL_FORMAL | 先检查匹配资产 | 缺checkpoint时40个task-training segments；不能复用不匹配seed0冒充finalFS-EWC |
| A-extra | common-F/head或保护开关额外干预 | OPTIONAL_NOT_DEFAULT | 仅更强因果主张触发 | 保留现有A，不机械扩展4格 |

实际测量完成须有新协议 manifest、summary complete、求积/端点自检与汇总；本清单中的 proposed 不表示已运行。

图表质量复核：pass（same-family/provisional），细节与最终小修见 figures/FIGURE_REVIEW.md 和 figures/REVIEW_RESOLUTION.md。
