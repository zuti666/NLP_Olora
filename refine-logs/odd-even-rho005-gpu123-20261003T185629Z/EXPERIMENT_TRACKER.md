# 七组奇偶响应实验运行记录

最新核验：2026-10-05T12:54:23.342714+02:00（UTC+2）。

**全部完成：7/7正式、7/7短测、1/1配对检查；0运行、0失败、0排队、0中断。** 最后一项于2026-10-04 02:26:26（UTC+2）正常结束。

| 实验 | GPU | 完成任务/epoch/更新 | AAA | FAA | 状态 |
|---|---:|---|---:|---:|---|
| OE01_SAM_s2 | 1 | 10/500/2550 | 63.102 | 44.809 | completed |
| OE02_Backward_s2 | 2 | 10/500/2550 | 62.615 | 43.714 | completed |
| OE03_Central_s2 | 3 | 10/500/2550 | 62.994 | 44.327 | completed |
| OE04_CE_s2 | 1 | 10/500/2550 | 60.232 | 39.260 | completed |
| OE05_Negative_s2 | 2 | 10/500/2550 | 50.801 | 25.235 | completed |
| OE06_Average_s2 | 3 | 10/500/2550 | 60.944 | 41.290 | completed |
| OE07_ReflectedAverage_s2 | 1 | 10/500/2550 | 59.262 | 37.932 | completed |

完整核验与阶段结果：[RESULTS_ANALYSIS.md](reports/20261005_125423/RESULTS_ANALYSIS.md)。

本次只读核验训练数据并更新报告，没有修改队列、配置或冻结训练代码。实时原始状态见runtime/queue_state.json。旧EXPERIMENT_TRACKER_20261003_211143.md为启动时快照，保留历史。
