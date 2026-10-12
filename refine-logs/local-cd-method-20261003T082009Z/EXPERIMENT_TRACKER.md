# Local-CD实验追踪

当前仅保留实验计划。按用户要求，Local-CD待执行清单已清空，禁止自动接续；原本未提交到任何运行队列。

M0已通过：CPU核心7组、框架3组、独立代码复核；M1六组短测与M2六组正式训练均改为plan_only；active suite包含0个作业。

| Run ID | 方法 | 步长 | seed | 状态 |
|---|---|---|---:|---|
| L01_CE_natural_s2 | ce | natural | 2 | plan_only |
| L02_Forward_natural_s2 | forward | natural | 2 | plan_only |
| L03_Backward_natural_s2 | backward | natural | 2 | plan_only |
| L04_CD_natural_s2 | central | natural | 2 | plan_only |
| L05_Backward_matched_s2 | backward | matched | 2 | plan_only |
| L06_CD_matched_s2 | central | matched | 2 | plan_only |

处理时间：2026-10-03T10:46:09.535830+02:00。移出草案待执行清单：6项短测＋6项正式训练；0项已运行。原始配置、计划与代码保留，其他实验队列未修改。
