# ImageNet-A 六行两任务表执行计划

2026-09-20。用户已明确授权执行；最新指定配置覆盖早先 ImageNet-R 和无轨迹平均要求。仅完成表格所需前两个任务，邻域与长序列任务暂不加入。

目的：比较完整 GAM/EWC 组合与两种阶段开关消融；固定第一任务 anchor 的分支复用保证与其对应 EWC 缓存一致。shaping 改变 anchor 和其 Fisher，不能把跨 anchor 的比较解释为完全隔离保护矩阵。

协议详见 `scripts/imageneta_two_task_table/README.md` 与 `config.json`。六行统一执行 epoch20/40 的 CE 标准化 softmax(beta3) 因子平均，再以 .3 混入任务末端，随后重算 Fisher；λ=0 的两行也测量 Fisher，但不将其加入训练目标。

执行顺序：CPU代码/数学测试→实际GPU六行3epoch检查→正式seed0的2anchors+6分支+诊断+表格→seed1→seed2→三seed均值±样本标准差。

正式任务数：24个50epoch训练段（6个第一任务模型+18个第二任务分支），18组端点评估。两卡并行。GPU小时未实测，不伪造预算；空闲等待与运行时间分别记录。当前只授权使用物理GPU4/6。

输出同时保留旧布局B1/B2 proxy列与逐样本 raw EF 增强表。新表必须明确ImageNet-A、SGD、连续Q/V LoRA、轨迹混合和实际γ/.2damping；不得复用旧ImageNet-R fresh K/V/Adam的caption或数值。

判断规则：按每seed计算BWT、位移与CE损伤，再聚合；训练seed与评估采样的波动不混淆；新任务accuracy必须同时报告；负损伤不截断。完整方法、第一阶段shaping-only、第二阶段shaping-only分别列行，不预设排序和正向结果。

状态与日志由 `campaign/pipeline_state.json`、`campaign/STATUS.md` 自动更新。GPU检查未通过前正式训练保持依赖阻塞；不把CPU测试或smoke标为表格完成。
