# ImageNet-A：P/C/CD 外层定位实验执行计划

用户已授权加入并执行原始 [计划](../g2-central-location-plan-20261003T092204Z/EXPERIMENT_PLAN.md) 的三组主实验，批准的科学版本保存在 `plan_snapshot/`。最新授权将GPU路由改为只使用物理GPU0/1、并行上限2，覆盖旧计划中的GPU2–6安排。新运行目录为 `/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z`，父运行是 `../g2-central-location-execution-20261003T102355Z`。

## 固定设置与三组变量

令 g0=∇L(θ0)，u0=g0/(||g0||+1e-12)，g±=∇L(θ0±0.05u0)。

| 实验 | 外层定位差分 q | 最终更新 |
|---|---|---|
| O_P | g+−g0 | θ0−η∇L(θ0+0.25N(q)) |
| O_C | g0−g− | 同上 |
| O_CD | (g+−g−)/2 | 同上 |

全部按g0→g+→g−→g2进行四次普通CE一阶梯度；每次回到原始参数，最终仅提交一次g2自然SGD更新。同批数据和前向随机状态；正、负探针都执行。没有g3、EWC/Fisher、融合、投影、最终梯度归一化、HVP或replay数据。

固定ImageNet-A本地train5981/test1519，200类顺序0…199，10×20类；ViT-B/16 augreg2_in21k_ft_in1k、Q/V LoRA rank10、无alpha/rank缩放、训练seenFC。seed2，每任务50epochs、batch128保留尾批；SGD momentum0/weight_decay0，首任务LR0.04、随后0.02，cosine50、最低比率0.1。ρ=gam_grad_rho=0.05，r=gam_grad_norm_rho=0.25。FP32 math attention，AMP/TF32关闭，严格确定性。

## 队列与验收

先在新目录重跑三组两任务短测，再检查配对初始化和增强身份，全部通过后自动启动三组正式训练。正式完成后执行共同状态离线诊断：三轨迹Task0/4/9各两个固定批次，108次普通CE梯度、零SGD。每条正式2550更新/10200训练CE/80任务入口隔离重放CE；每条短测2更新/8训练CE/16重放CE。8个job的ID、DAG、科学配置与原执行目录相同，只改GPU路由和独立文件路径。

全部trainer源码、测试和CPU gate逐字节复用旧执行目录，沿用16+9共25项CPU通过证据；来源与差异证明见 `GPU_ROUTING_CHANGE.json`。环境继续使用 `Pilot_new_local`，没有package变更。新六个真实入口CPU dry-run和独立GPU路由差异审核通过后冻结；独立代理随后执行 `ENVIRONMENT.md` 精确绝对命令产生新DOC_CHECK。真实GPU短测是正式训练必需门禁；CPU验收不等于GPU验收。

队列只允许物理GPU0/1、并行上限2，空闲阈值<500MiB且<10%，共享编号/UUID双锁。GPU0空闲、GPU1约12GB忙是root停旧控制器时的宿主机快照；分配时重新读取NVML，忙卡等待。任一任务失败停止后续启动，不自动重试。旧控制器已安全停止，旧结果及冻结文件保留；新目录不继承任何旧runtime、outputs或启动/结果收据，准备阶段尚未冻结或进行训练。

## 指标与解释范围

主要报告十阶段任务等权AAA，同时报告FAA、LA、BWT、遗忘及完整准确率矩阵。所有比较是同协议新训练；历史P/C结果只作背景。共同状态诊断分离正负差分的共同项与不对称项，并检查定位方向及其g2变化，不预设CD优于单侧。单seed结论为探索性，不能据此声称普遍显著或证明Hessian机制。

上轮直接差分回加Local-CD继续仅保留计划，不加入本轮队列。实时进度以 `runtime/queue_state.json` 及root维护的 `EXPERIMENT_TRACKER.md` 为准。
