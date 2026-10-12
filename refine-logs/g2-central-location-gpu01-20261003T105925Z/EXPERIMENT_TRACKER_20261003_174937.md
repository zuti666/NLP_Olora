# ImageNet-A P/C/CD：实验进程和已有完整结果

核验时间：2026-10-03T17:49:36.907889+02:00（Europe/Paris）。控制器PID2806544存活，正式结果validated 2/3、running1、failed0；整体6/8 completed、1running、1pending。GPU0目前空闲，GPU1训练CD。离线共同状态诊断等待CD完整完成，自动接续。

统一seed2、rank10、ρ=.05、r=.25、最终仅g2自然SGD、FP32 math attention；10tasks×50epochs。每组正式2550更新、10200训练CE、80零更新隔离重放CE。P/C的独立矩阵复算和validate_job均通过，摘要、任务矩阵、原始trace和完成收据SHA一致。

| 组别／定位q | 完整状态 | AAA（%） | FAA（%） | LA（%） | BWT（pp） | 遗忘（pp） |
|---|---|---:|---:|---:|---:|---:|
| P：g+−g0 | 10/10 validated | 66.621 | 49.995 | 56.600 | -7.339 | 7.339 |
| C：g0−g− | 10/10 validated | 66.696 | 49.952 | 56.488 | -7.262 | 7.386 |
| CD：(g+−g−)/2 | 运行中 | — | — | — | — | — |

AAA口径为十阶段等权、每阶段已见任务等权平均；FAA为最终十任务等权。native JSON的所有55个下三角单元平均AAA不是这里的AAA。另附最终逐样本准确率：P=57.60%，C=57.80%，不与FAA任务均值混用。

CD已完成5/10任务，正在第6任务（零基task5），最近完成epoch50/50；已观测1900次真实更新。运行中final_summary是起始占位（completed_tasks0、暂无counters），collector某些0计数不可当作真实运行0；完整进度以原始trace/metrics/tasks为准。CD阶段AAA/FAA不进入最终对比。

C−P：AAA +0.074pp，FAA -0.043pp。当前单seed下正向与反向符号修正定位的成绩接近，没有一致的优劣；尚不能判断CD是否更好或声明统计等价、证明Hessian机制。本轮没有同协议CE基线，不据此声称优于SGD。

所有已观测epoch的CE有限；P/C实际Fisher/融合/投影/HVP/create_graph调用均0。native未观测上三角NaN是结构占位，不作为训练失败。

实时队列：[queue_state.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/runtime/queue_state.json)；核验快照：[STATUS_LATEST.json](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/monitor/STATUS_LATEST.json)；结果表：[summary.csv](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/summary.csv)。原启动时tracker已过期，本报告依据宿主机进程、实时队列和原始结果更新。训练与调度源码未修改。
