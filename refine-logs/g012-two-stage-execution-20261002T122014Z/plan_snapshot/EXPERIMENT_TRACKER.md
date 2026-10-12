# 两阶段实验追踪

日期：2026-10-02。所有下列条目为 PLANNED_ONLY；未注册到任何调度器，未开始，未占GPU。单seed2，10×20，rank10，r=.25，rho=.05。N=自然，M=同状态正向g1+有效步幅匹配。

| Run ID | 阶段/批次 | 条件 | 最终方向 | 优先级 | 状态 | 主要目的 |
|---|---|---|---|---|---|---|
| S1-01 | M1 | CE-N | g0 | 核心 | PLANNED_ONLY | 普通方向基线 |
| S1-02 | M1 | S+-N | g1+ | 核心 | PLANNED_ONLY | 正向第一侧；匹配身份共用 |
| S1-03 | M1 | S−-N | g1− | 核心 | PLANNED_ONLY | 下降第一侧 |
| S1-04 | M1 | O_P-N | g2P | 核心 | PLANNED_ONLY | 原外层位置 |
| S1-05 | M1 | O_D-N | g2D | 核心 | PLANNED_ONLY | 直接下降外层位置 |
| S1-06 | M1 | O_C-N | g2C | 核心 | PLANNED_ONLY | 下降侧符号修正 |
| S1-07 | M1 | A_P-N | g0+g2P−g1+ | 核心 | PLANNED_ONLY | 原Sec；阶段2同协议共享 |
| S1-08 | M1 | A_D-N | g0+g2D−g1− | 核心 | PLANNED_ONLY | 用户直接下降方案 |
| S1-09 | M1 | A_C-N | g0+g2C−g1− | 核心 | PLANNED_ONLY | 下降侧修正 |
| S1-10 | M2 | O_R-N | g2R | 核心 | PLANNED_ONLY | 正向侧反转外层 |
| S1-11 | M2 | A_R-N | g0+g2R−g1+ | 核心 | PLANNED_ONLY | 补齐四格 |
| S1-12 | M2 | A_P-M | g0+g2P−g1+ | 核心 | PLANNED_ONLY | Sec尺度控制；同协议共享 |
| S1-13 | M2 | A_D-M | g0+g2D−g1− | 核心 | PLANNED_ONLY | 下降方案尺度控制 |
| S1-14 | M2 | A_C-M | g0+g2C−g1− | 核心 | PLANNED_ONLY | 修正方案尺度控制 |
| S1-15 | M2 | A_R-M | g0+g2R−g1+ | 核心 | PLANNED_ONLY | 反向外层尺度控制 |
| S1-16 | M2 | CE-M | g0 | 核心 | PLANNED_ONLY | 匹配基线 |
| S2-01 | M4 | Hstart-N | g0+H(phi1)d12 | 核心 | PLANNED_ONLY | 起点响应 |
| S2-02 | M4 | Hmid-N | g0+H(mid)d12 | 核心 | PLANNED_ONLY | 中点响应 |
| S2-03 | M4 | Hstart-M | g0+H(phi1)d12 | 核心 | PLANNED_ONLY | 起点尺度控制 |
| S2-04 | M4 | Hmid-M | g0+H(mid)d12 | 核心 | PLANNED_ONLY | 中点尺度控制 |
| S2-05 | M4 | Residual-only-N | g2−g1 | 核心 | PLANNED_ONLY | 删除最终g0基底 |
| S2-06 | M4 | Residual-only-M | g2−g1 | 核心 | PLANNED_ONLY | 删除基底的尺度控制 |
| S2-B01 | M3/M4 | Sec-backend-N | g0+g2−g1 | 后端变化时必须 | PLANNED_CONDITIONAL | 同后端Sec；不与旧native混作配对 |
| S2-B02 | M3/M4 | Sec-backend-M | g0+g2−g1 | 后端变化时必须 | PLANNED_CONDITIONAL | 同后端Sec匹配 |

核心完整重训22项，后端变化再加2项=24。新统一四次探测协议，历史五次执行结果仅作背景，不扣除新框架基线预算。新seed/r/rho扫描未列入。

## 数值验收事项

| Gate ID | 验收 | 状态 | 依赖 |
|---|---|---|---|
| G0 | 数据/预训练/初始参数/新FC/RNG/环境哈希绑定 | PLANNED_ONLY | 未来实施 |
| G1 | float64四格符号、恢复、指定U、一步SGD身份 | PLANNED_ONLY | 阶段1之前 |
| G2 | 正向g1匹配身份、BA+FC几何、退化/无根失败显式记录 | PLANNED_ONLY | matched之前 |
| G3 | arbitrary-vector HVP解析/同点梯度/中心差分/恢复 | PLANNED_ONLY | 阶段2之前 |
| G4 | native可行或建立math共同后端及Sec复训 | PLANNED_ONLY | G3 |

## 条件附加，未计入核心且不自动排队

| ID | 设置 | 完整训练数 | 触发原因 |
|---|---|---:|---|
| X1 | A_D_ref/A_C_ref固定减去g1+ | 自然2，必要时匹配2 | O族与A族模式不一致，需拆参照 |
| X2 | A_grad，外点改为phi+r*N(g0) | 自然+匹配2 | 需要直接证明c10定位信息的特异必要性 |
| X3 | 双节点HVP响应 | 自然+匹配2 | 单点替代不足、数值已可靠 |
| X4 | math CE/g2同协议基线 | 每方向/模式单独计数 | 要作跨旧native之外的CE/g2阶段内主张 |

追踪更新必须保存版本。未获得新的执行指令前，不将PLANNED_ONLY改为WAITING/RUNNING。
