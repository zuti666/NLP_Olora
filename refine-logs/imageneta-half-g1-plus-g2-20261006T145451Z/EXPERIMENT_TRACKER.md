# ImageNet-A (g1+g2)/2 进度

核验时间2026-10-06T15:02:44.252136+00:00。后台控制器PID465967已运行，状态waiting_for_prior_campaign，5项待接续（2短测+1配对+2正式）。只用GPU0/1；当前两卡继续运行g0+g1+g2，本轮等待其2项正式全部完成并验收后开始GPU短测。尚无本轮GPU训练或完整结果。

CPU9/9、独立代码/验收检查、4实际CLI、source190hash、unchangedwarm环境与freshdoc通过；实际GPUwitness/模型短测仍待空闲卡。新U=(g1+g2)/2，名义LR保持.04/.02，rho.05、r.25/1、seed2rank10，四正向探针g3计算但不用最终U，无Fisher/实际融合/HVP/投影/额外范数匹配。

原g0sum代码/config28frozenfiles保持不变且控制器存活。本记录为启动快照；实时runtime/queue_state.json，summary.csv/RESULTS.md在正式完整预算验收后填最终指标。
