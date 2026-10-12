# g1/g3 机制实验追踪

| Block | 项目 | 数量 | 状态 | 说明 |
|---|---|---:|---|---|
| reuse | 原pure g1/g3/sum/original | 4 | validated | 已核验，不重训 |
| A | mean_sum | 1 | preparing | 仅beta改为.5/.5 |
| A | g3 r=0/.1/.25/.5 | 4 | preparing | r1复用 |
| B | 三来源×task0/4/9 | 9 anchors | implementing | 同状态离线诊断 |
| C | 矩阵获取/保持分析 | 3原始臂 | complete | BASELINE_MATRIX_ANALYSIS.json |

真实状态以training/runtime/state.json及diagnostics队列state为准；准备状态不是已启动。
