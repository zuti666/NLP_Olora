# 全200类五方法补齐追踪

2026-10-10，seed2。新增项未注册/未运行，本计划没有自动接续队列。

| ID | 训练范围 | 方法 | 状态 | 准确率% / 来源 |
|---|---|---|---|---|
| L-SGD | LoRA rank10 | g0 | 复用已完成 | 63.001975；upward-eight-fork/formal_single200_g0_s2 |
| L-SAM | LoRA rank10 | g1+ | 复用已完成 | 64.186965；upward-eight-fork/formal_single200_g1_s2 |
| L-MIRROR | LoRA rank10 | g0+(g0-g1-) | 计划，待隔离代码扩展 | — |
| L-G2 | LoRA rank10 | 固定rho=.05、r=1的g2 | 复用已完成 | 68.597762；upward-eight-fork/formal_single200_g2_s2 |
| L-GAM | LoRA rank10 | beta全1、gamma=.1完整GAM | 计划，待隔离代码扩展 | — |
| F-SGD | 全量FT | g0 | 计划，待FT范围实现 | — |
| F-SAM | 全量FT | g1+ | 计划，待FT范围实现 | — |
| F-MIRROR | 全量FT | g0+(g0-g1-) | 计划，待FT范围实现 | — |
| F-G2 | 全量FT | 固定rho=.05、r=1的g2 | 计划，待FT范围实现 | — |
| F-GAM | 全量FT | beta全1、gamma=.1完整GAM | 计划，待FT范围实现 | — |

正式训练之前需要实现及验收。现有six-direction单任务seed0/1另记录于原队列，不重复登记为这里的新增作业。
