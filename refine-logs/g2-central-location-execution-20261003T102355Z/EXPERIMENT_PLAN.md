# ImageNet-A：P/C/CD 外层定位实验执行计划

用户已授权加入并执行。实施原始 [计划](../g2-central-location-plan-20261003T092204Z/EXPERIMENT_PLAN.md) 的三组主实验；逐字复制的批准版本保存在 plan_snapshot/。

## 固定设置与三组变量

令 g0=∇L(θ0)，u0=g0/(||g0||+1e-12)，g±=∇L(θ0±0.05u0)。

| 实验 | 外层定位差分 q | 最终更新 |
|---|---|---|
| O_P | g+−g0 | θ0−η∇L(θ0+0.25N(q)) |
| O_C | g0−g− | 同上 |
| O_CD | (g+−g−)/2 | 同上 |

全部按 g0→g+→g−→g2 进行四次普通 CE 一阶梯度；每次回到原始参数，最终仅提交一次 g2 自然 SGD 更新。同批数据和前向随机状态；正、负探针都执行。没有 g3、EWC/Fisher、融合、投影、最终梯度归一化、HVP 或 replay 数据。

固定 ImageNet-A 本地 train5981/test1519，200 类顺序0…199，10×20 类；ViT-B/16 augreg2_in21k_ft_in1k、Q/V LoRA rank10、无 alpha/rank 缩放、训练 seenFC。seed2，每任务50epochs、batch128保留尾批；SGD momentum0/weight_decay0，首任务LR0.04、随后0.02，cosine50、最低比率0.1。ρ=gam_grad_rho=0.05，r=gam_grad_norm_rho=0.25。FP32 math attention，AMP/TF32关闭，严格确定性。

## 队列与验收

先运行三组两任务短测，再检查配对初始化和增强身份，全部通过后自动启动三组正式训练。正式完成后执行共同状态离线诊断：三轨迹的Task0/4/9各两个固定批次，108次普通CE梯度、零SGD。每条正式2550更新/10200训练CE/80任务入口隔离重放CE；每条短测2更新/8训练CE/16重放CE。

CPU代码验收、输入资产和六个真实入口dry-run通过后冻结。独立代理执行 ENVIRONMENT.md 精确命令。真实GPU短测是正式训练的必需条件；CPU验收不等于GPU验收。队列GPU2–6、并行上限3，空闲阈值<500MiB且<10%，共享双锁。不使用0/1，不抢占忙卡。任一任务失败停止后续启动，不自动重试。

## 指标与解释范围

主要报告十阶段任务等权AAA，同时报告FAA、LA、BWT、遗忘及完整准确率矩阵。所有比较是同协议新训练；历史P/C结果只作背景。共同状态诊断分离正负差分的共同项与不对称项，并检查定位方向及其 g2 变化，不预设CD优于单侧。单seed结论为探索性，不能据此声称普遍显著或证明Hessian机制。

上轮直接差分回加Local-CD继续仅保留计划，不加入本轮队列。实时进度以runtime/queue_state.json及EXPERIMENT_TRACKER.md为准；本文件冻结时尚未进行GPU训练。
