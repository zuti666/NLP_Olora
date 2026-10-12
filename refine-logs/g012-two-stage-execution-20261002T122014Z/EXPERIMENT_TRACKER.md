# 当前实验追踪

快照：2026-10-03T08:23:42.055704+02:00（巴黎时间）。正式S1 16/16全部完成，整个计划16/24完整训练已验收；0运行、8项阶段2正式训练未启动。

队列状态：halted_failure。逐job真实计数32 completed（16短测+16完整）、2 failed诊断、17 pending（8阶段2短测+8阶段2完整+1响应诊断）。顶层counts含过时running字段，以逐job状态和进程身份核验为准。controller及本轮worker无匹配存活进程。GPU0/1仍排除。

Stage1_mechanism_diagnostics未通过首批同状态梯度逐位一致检查；HVP_full_gate仅2/12响应、1/6批次通过冻结有限差分门槛，full_stage2_allowed=false。两项失败不改变16条完整训练的完成凭据，但严格机制归因未验收；阶段2不得描述为已经完成。

本次仅汇总核验，未重启、修改或追加队列。

[ImageNet-A r=.25 全部历史与本轮结果](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r025-all-results-20261003T062250Z/REPORT.md)；[28条独立完整训练CSV](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r025-all-results-20261003T062250Z/all_10task_results.csv)；实时原始状态：[queue_state.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/queue_state.json)。
