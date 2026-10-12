# ImageNet-A g0+g1+g2 进度

核验时间2026-10-06T14:53:08.436064+00:00。2/2真实GPU短测和配对验收通过，2/2正式训练已开始，GPU0/1。当前实际更新分别4/3，均Task0附近，完整十任务成绩尚无。

固定seed2/rank10/rho=.05，r=.25/1；原正向四探针，g3计算但不用于最终U=g0+g1+g2，每步4CE/1SGD，Fisher/融合/HVP/投影/范数匹配均0。CPU9/9、独立复核、4实际CLI、source190hash、warm环境与freshdoc检查通过，两个GPUkernelwitness通过。

这是启动快照；实时状态runtime/queue_state.json，自动summary.csv/RESULTS.md仅验收完整运行后填最终指标。
