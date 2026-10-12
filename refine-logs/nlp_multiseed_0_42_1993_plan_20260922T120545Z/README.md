# NLP 三 seed 补充计划与历史审计

本目录仅包含计划、审计与机器可读清单；未启动训练，未修改旧结果。

- `EXPERIMENT_PLAN.md`：完整计划，推荐统一四卡的96条新增Order；另列最小补缺72条备选。
- `EXPERIMENT_TRACKER.md`：36个模型×方法×seed运行组状态，每组3个Order。
- `SEED_ORDER_LEDGER.csv`：108个目标单元，同时区分历史存在、严格四卡可复用、缺失seed及八卡迁移。
- `RUN_QUEUE_DDP4.csv`：96条新增Order的顺序计划，固定GPU3/4/5/6。
- `TARGET_PROTOCOL_DRAFT.json`：12个模型×方法模板；参数冻结与OLoRA+GAM正则语义验证尚待M0。
- `BASELINE_HISTORY_AUDIT.*`、`EWC_HISTORY_AUDIT.*`：历史查找、证据与排除理由。
- `CURRENT_SOURCES_AUDIT.json`：144个所选stage的指标/训练文件与SHA256复核。
- `EXECUTION_DESIGN.*`：源码、环境、模型、数据、seed/GA语义与启动器依赖。
- `COMPUTE_ESTIMATE.json`、`HISTORICAL_RUNTIMES.json`：耗时估算和历史逐阶段依据。
- `GPU_STATUS_AT_PLANNING.json`：只读设备快照，GPU3当时占用；不是GPU预约。
- `VALIDATION.json`、`PLAN_REVIEW.md`：计划覆盖和队列一致性检查。
- `selected_*`：当前主表/图的数据与来源索引副本，原文件未改动。

目标为2模型×6方法×3seeds×3Orders=108条完整Order；12条四卡历史可按计划复用，72条缺seed，24条旧八卡迁移。每条Order包含4阶段。正式三seed统计必须在每个目标seed齐全且验证通过后生成。
