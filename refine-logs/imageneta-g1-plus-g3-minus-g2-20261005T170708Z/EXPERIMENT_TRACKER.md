# 实验跟踪

新增方法 U=g1+(g3−g2)，r=.25/1，seed2/rank10/rho.05。两个正式训练点已注册在 suite.json，先经过两个GPU短测与配对检查。CPU6/6、四个真实CLI dry-run、独立代码/schema审查、fresh-context文档调用通过。代码/配置已冻结。

核验时间 2026-10-05T19:22:07.299810+02:00；后台controller PID=1012149，状态=waiting_for_idle_gpu，计数={'pending': 5}。GPU2–6仅空闲时接续；GPU0/1排除。短测已开始0/2，正式训练已开始0/2。

新组尚无最终成绩。实时状态以 runtime/queue_state.json 为准；每次完成后自动刷新 summary.csv / RESULTS.md，部分任务结果只标进度。既有g0B运行未更改，原30份冻结文件核对一致。
