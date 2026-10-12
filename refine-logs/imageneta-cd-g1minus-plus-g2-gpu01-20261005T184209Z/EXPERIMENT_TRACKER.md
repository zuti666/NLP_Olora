# GPU0/1执行跟踪

核验时间2026-10-05T20:46:39.156372+02:00。新controller PID=1923021存活，仅GPU0/1，max2。旧队列已停止，无旧正式训练开始；2条旧短测已完成并保留。

新GPU0/1四条真实短测及配对检查全部通过，正式训练已实际开始：

- formal_C_g1minus_g2_r025_k10_s2：GPU0，5次实际SGD，Task0/Epoch1，CE=1.978500。
- formal_C_g1minus_g2_r1_k10_s2：GPU1，5次实际SGD，Task0/Epoch1，CE=2.090764。

其余正式作业自动接续。训练代码与原run相同；seed2/rank10/rho.05/r.25或1，U=g1−+g2，不除2，无g3/Fisher/EWC/融合/投影。每步4CE/1SGD日志检查通过，CUDA witness通过。尚无新的完整10任务最终成绩。实时状态runtime/queue_state.json，结果summary.csv/RESULTS.md。
