# ImageNet-A 第二局部探针符号实验：确定性执行版

日期：2026-10-07。原首短测被诊断同侧重算断言拦截、0正式开始；恢复版14/14真实短测均通过，但共同g012/外点逐位配对失败。原始证据未改动。固定数值预算未通过，未扩大阈值。严格deterministic两卡预检已完成2SGD各自、共同g012/phi012逐位一致；本版启用同一执行设置并重跑B1两基线，以保持正式对照执行协议一致。

## 固定科学设置

ImageNet-A 5981 train/1519 test，200类固定0…199，10×20类；ViT-B/16 augreg2_in21k_ft_in1k，LoRA rank10，seed2。每任务50epochs、batch128、SGD momentum/weight_decay=0，Task0 LR=.04、后续=.02，cosine最低比例.1。内半径rho=.05；各切片外半径r=1或.25。没有EWC/Fisher/融合/SGV/replay/HVP/梯度投影/最终更新范数匹配。

令N(v)=v/(||v||+1e-12)，范数覆盖全部可训练LoRA A/B和seen FC。固定前两段：

- g0=g(theta0)；theta1=theta0+rho N(g0)，g1=g(theta1)。
- q=g1−g0；theta2=theta0+r N(q)，g2=g(theta2)。
- 仅干预第三位置：theta3±=theta2±rho N(g2)，g3±=g(theta3±)。
- 恢复theta0后提交一次自然SGD；各训练step固定4CE/1SGD，同一R0在各探针重放，最后只推进到g0后的R1。

## 八格、两半径、16正式训练

| ID | 选定第三侧 | 残差符号 | 最终更新U |
|---|---|---|---|
| B0 | + | + | g0+(g3+−g2) |
| B1 | + | + | g1+(g3+−g2) |
| N1 | + | − | g0+(g2−g3+) |
| N2 | + | − | g1+(g2−g3+) |
| N3 | − | + | g0+(g3−−g2) |
| N4 | − | + | g1+(g3−−g2) |
| N5 | − | − | g0+(g2−g3−) |
| N6 | − | − | g1+(g2−g3−) |

新方向12项+严格执行协议B0/B1四项=16正式；两条旧B1仅历史参考，正式表不复用。每组2550更新/10200训练CE/10隔离诊断CE，无额外clean训练梯度。仅保留单位化构造探针，不归一化最终U；差分系数±1，不除rho，不乘r/rho。g2位置/定义没有反向。

## 执行改进及证据

新增torch.use_deterministic_algorithms(True,warn_only=False)，不支持时失败，不回退。CUBLAS_WORKSPACE_CONFIG=:4096:8原恢复suite已有，本版沿用；TF32/AMP关闭。训练g012/第三探针公式、U及SGD数学不变；修改的是CUDA执行协议，可能改变浮点轨迹，因此B1补跑。支持与限制参照[PyTorch2.4确定性说明](https://docs.pytorch.org/docs/2.4/generated/torch.use_deterministic_algorithms.html)。

原恢复14snapshot/42同半径配对：初始化、数据、R0/R1 exact；q/N(q)/外扰动relative L2约4e-4，超原定1e-4预算。失败保存在evidence/strict_native_pairing_failure.json与数值审计；不称machineprecision、性能差异或跨GPU硬件归因。同GPU重复亦有差异。新的两卡预检在同硬件型号得到真实位等，是已观察现象，不保证任何平台都位等。

诊断复用训练侧g3，额外测一次反侧普通CE，并完全恢复模型、buffers、grad、缓存、模式、pending U和R1。每任务一次，只记录几何/幅度，不插入训练更新，不把训练引用复用称独立重算等价。

## 接续与验收

33队列项=16个2tasks×1epoch×1batch短测+1 CPU真实配对门+16正式。短测4CE/1SGD、无禁用模块，保存Task0首步g012/phi012实际CPU张量及SHA；输入、初始化、同r g012/phi012严格逐位配对全通过后才启动正式。

GPU0/1继续排除；允许实际可见GPU2–6，max_parallel5，每60秒查询memory/util/PID，并在index/UUID双锁后第二次确认空闲；不抢占其他作业。r=1优先N5/N6/B0/B1/N1/N2/N3/N4，再r=.25。任一失败停止自动接续。

CPU、CUDA kernel witness与真实短测分列验收。正式结果须10tasks/500epochs/2550updates/10200训练CE完整、原矩阵重算/receipt/hash完整、有限loss与禁用模块计数0后填表。任何partial或smoke分数都不能填最终AAA/FAA。主要AAA，同时FAA/LA/遗忘/BWT和逐任务矩阵，任务等权百分比；sample-weighted单列。

本轮能判断第二探针侧别和残差符号的作用；第一上升探针固定，不能声称消融第一部分。单seed自然步长仍同时包含方向与幅度，不预设哪组获胜。

## 路径

配置EXPERIMENT_SETTINGS.csv及configs；执行trainer/four_probe_train.py、trainer/direction_step.py；ENVIRONMENT.md记录真实主机环境和只读命令；runtime/queue_state.json为实际进度；summary.csv/results.json仅收纳验收完成结果。执行revision和原始失败证据见EXECUTION_REVISION.json/evidence。当前CPU/helper、GPU验收以各自gate和实时状态为准。
