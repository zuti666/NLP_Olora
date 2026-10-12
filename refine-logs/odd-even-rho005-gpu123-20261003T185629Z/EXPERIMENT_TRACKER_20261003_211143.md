# 七组奇偶响应实验运行记录

记录时间：2026-10-03T21:11:43.504233+02:00。实时状态以runtime/queue_state.json为准。

**已启动正式训练：**七组GPU两任务短测全部完成，CPU七组配对通过；首批三组已在GPU1/2/3训练，后续四组等待首批全部完成。

| 实验 | 方向 | GPU | 验收时进度 | 状态 |
|---|---|---:|---|---|
| OE01_SAM_s2 | g+ | 1 | Task0 / Epoch3，24更新 | running |
| OE02_Backward_s2 | g0+(g0−g−) | 2 | Task0 / Epoch3，23更新 | running |
| OE03_Central_s2 | g0+(g+−g−)/2 | 3 | Task0 / Epoch3，24更新 | running |
| OE04_CE_s2 | g0 | 空闲1/2/3 | 等待首批三组全部验证完成 | pending |
| OE05_Negative_s2 | g− | 空闲1/2/3 | 等待首批三组全部验证完成 | pending |
| OE06_Average_s2 | (g++g−)/2 | 空闲1/2/3 | 等待首批三组全部验证完成 | pending |
| OE07_ReflectedAverage_s2 | g0+[g0−(g++g−)/2] | 空闲1/2/3 | 等待首批三组全部验证完成 | pending |

固定配置：ImageNet-A 10×20，rank10，seed2，rho=.05，50epochs/task，batch128，SGD LR首任务.04/后续.02，cosine，math/FP32。七组均从头训练且每步统一3CE、1SGD，无外层探针/EWC/Fisher/融合/投影。

七个短测每组2更新、6训练CE、12隔离诊断CE；所有禁用工作计数0。初始参数、g0/g+/g−及正负位置、R0/R1、数据增强、FC新行身份配对通过。正式worker各自新seed2CUDA witness通过，实际loss有限，未见OOM。

控制器PID3366664；GPU编号+UUID共享锁，无抢占。59份本轮文件冻结、190源文件核对通过，89项旧实验文件哈希未改变。

预计每组1.5–2GPU小时、总10.5–14GPU小时；三卡持续空闲且按两阶段依赖运行，墙钟约4.5–6小时加短测。当前尚无完整正式AAA/FAA，结果将由collect_results.py在任务通过完成核验后自动汇总。

证据：LAUNCH_VERIFIED.json、gates/DOC_CHECK.json、gates/INDEPENDENT_REVIEW.json。完整设置及运行顺序见EXPERIMENT_PLAN.md。
