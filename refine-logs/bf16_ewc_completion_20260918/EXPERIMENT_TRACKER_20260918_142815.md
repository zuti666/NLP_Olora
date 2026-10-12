# BF16 EWC 补全任务表

日期：2026-09-18。关联计划：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本计划尚未启动训练。

| ID | 任务 | 方法/配置 | 优先级 | 状态 | 结果/说明 |
|---|---|---|---|---|---|
| P0 | 获取 BF16 硬件并核对旧协议 | 五张支持 BF16 的 GPU；检查旧 O1 的代码/模型/数据/依赖 | 必须 | TODO：缺兼容设备 | 当前十张 Quadro RTX6000 compute7.5 不满足 BF16 |
| P1 | 准备五卡入口与 dry-run | DDP5 / BS1 / GA8 / λ1 / OLoRA0 / seed1993 | 必须 | TODO | 现有单卡 runner 和旧 O2/O3 shell 均不能直接使用 |
| P2 | 两任务 smoke | 每任务2步，独立测试目录 | 必须 | TODO | 不计正式结果 |
| B16-EWC-1B-O2-S1993 | 完整 Order2 四阶段 | IncLoRA+EWC，BF16，5卡，总batch40 | 必须 | TODO | 准确率待运行 |
| B16-EWC-1B-O3-S1993 | 完整 Order3 四阶段 | 同上 | 必须 | TODO | 准确率待运行 |
| B16-EWC-1B-O1-S1993 | 新完整 Order1 四阶段 | 同一快照与协议 | 条件必须 | CONDITIONAL | 旧协议差异不能消除时运行；严格新统一基线需此项 |
| R-OLD-O1 | 历史 Order1 | IncLoRA+EWC，BF16，5卡，seed1993 | 复用候选 | DONE；复用需审查 | 59.3125% |
| R-FP32-O2 | 已完成 FP32 Order2 | IncLoRA+EWC，单卡GA40 λ0.2 | 仅留档 | DONE | 61.6743%，不能标成BF16 |
| R-FP32-O3 | 已完成 FP32 Order3 | 同上 | 仅留档 | DONE | 64.4539%，不能标成BF16 |
| P3 | 汇总与填论文表 | 原始JSON → CSV/来源清单/LaTeX | 必须 | TODO | 通过协议检查才填三Order平均 |

如只取得单张 BF16 GPU：新建单卡BF16分支，三个 Order全部运行，GA40、CLI λ0.2；不把该分支的 O2/O3拼接旧五卡 O1。
