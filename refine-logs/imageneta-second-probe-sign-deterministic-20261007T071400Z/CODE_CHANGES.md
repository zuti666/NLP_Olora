# 确定性执行修改

训练前两探针、外点、选定第三探针、最终U和SGD定义保持不变；启用严格torch deterministic算法、验证已有cuBLAS workspace，记录实际执行flag。执行协议v2，因此两个旧B1本轮重跑。

隔离诊断沿用native恢复版：训练selected g3引用复用、只新增反侧1CE/task、恢复完整状态和R1。source/effective-step/native-lifecycle均不改数学。原失败及数值门FAIL留证，无放宽旧门。新33项队列保留严格真实输入/g012/点位配对；没有Fisher/权重融合/投影/HVP/最终归一化。
