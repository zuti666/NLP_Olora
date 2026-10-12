# 四方向补充seed实验进度

核验时间：2026-10-12T12:58:02.910473+02:00。仅物理GPU4/5/6，最多三组并行。8/8短测、同seed四方向配对全部通过；正式3运行、5等待、0完成，失败0。

| 实验 | GPU | 已观察更新 | 当前任务/epoch |
|---|---|---|---|
| OE01_SAM_s0 | 4 | 24 | 0/3 |
| OE02_Backward_s0 | 5 | 22 | 0/3 |
| OE03_Central_s0 | 6 | 25 | 0/3 |

其余：seed0基点、seed1四方向自动接续。父seed2四项经严格校验复用；新完整结果尚无。完成后THREE_SEED_SUMMARY.csv报告三seed均值±样本SD和PAIRED_DIFFERENCES.csv配对差。

实际进度：runtime/queue_state.json；日志：runtime/jobs/<id>/console.log；只暂停新启动可创建runtime/PAUSE。公式及配置见EXPERIMENT_PLAN.md。
