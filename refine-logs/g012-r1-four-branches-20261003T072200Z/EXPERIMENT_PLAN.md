# ImageNet-A：r=1 的正反探针与差分符号实验

复制已完成 r=.25 Stage1 的科学设置，仅将参与外点构造的 `gam_grad_norm_rho` 改为 **1.0**；`gam_grad_rho=.05`。同一16格设计中，12个外点相关条件新跑完整十任务，4个最终更新与外半径无关的基准经原始记录审计和实际更新等价检查后复用。

令 N(v)=v/(||v||₂+1e-12)，所有可训练 LoRA 因子与已见分类头一起归约。每一步计算 g0=g(phi)、g1+=g(phi+rho N(g0))、g1−=g(phi−rho N(g0))。外点始终从原基点 phi 构造 phi2=phi+r N(q)，g2=g(phi2)。

| 分支 | 第一侧梯度 | q | O 自然步长 | A 自然步长 | A 匹配步长 |
|---|---|---|---|---|---|
| P | g1+ | g1+−g0 | 新训练 | 新训练 | 新训练 |
| D | g1− | g1−−g0 | 新训练 | 新训练 | 新训练 |
| C | g1− | g0−g1− | 新训练 | 新训练 | 新训练 |
| R | g1+ | g0−g1+ | 新训练 | 新训练 | 新训练 |

O=g2；A=g0+(g2−第一侧梯度)。原矩阵没有 O matched，本轮也不增加该条件。四个基准为 g0 natural、g1+ natural、g1− natural、g0 matched；复用的实际旧轨迹仍标记 r=.25，不能用于本轮 r=1 外点机制统计，不复制旧 completed 到新队列。

## 固定训练设置

ImageNet-A 原 train/test，5981/1519张、200类；固定类顺序0…199、10×20类、无shuffle/replay。seed2，ViT-B/16 augreg2_in21k_ft_in1k 预训练，Q/V LoRA rank10、BA scale1，无alpha/rank缩放。每任务50 epochs，共500 epochs/2550次真实SGD/299050样本次；batch128，不丢末批。SGD momentum=0/weight_decay=0，Task0 LR=.04、后续=.02；每任务cosine T_max50、最低LR为起点的.1倍，scheduler每个完整epoch后推进。

训练保持FP32、TF32关闭、原生注意力后端。使用与 r=.25 相同的 EpochLoader：data_seed=seed*100000+task*1000+epoch，epoch从1开始；8训练workers、4评估workers。固定增强/采样与模型RNG分离。每个探针先精确恢复基点参数、buffers、模式、梯度、LoRA controls/cache，再重放同一步R0；最终模型RNG只推进到g0产生的R1。每步恰好四次普通CE梯度 g0/g1+/g1−/g2，再执行一次SGD；不计算g3。

关闭EWC/Fisher、任务内与端点融合、delta-W附加惩罚、SGV、原GAM正交修正及最终更新额外范数归一化。Fisher计算由 ewc_skip_task_fisher=true 跳过；EWC lambda=0、lambda_flat=1。NME类均值读取关闭。模型、原始源码与加载路径沿用冻结source190文件。

## 匹配定义和验收

matched条件在候选自身当前模型状态、同批次下，用 raw g1+ 的名义LR真实FP32 SGD反事实位移作为目标。测量覆盖所有无缩放LoRA有效B@A及全部已见FC weight/bias的联合L2范数；用正标量调节候选方向的实际SGD长度。D/C也匹配正向g1+，不会改为负侧参照。暂存反事实参数后精确恢复，最终只提交一次真实 optimizer.step，学习率恢复后推进scheduler。

这是原 r=.25 方案的同状态匹配定义，与另一队列“读取独立完整S轨迹”的Gmatch不同，不混用结果。匹配真实误差阈值保持1e-3；缺失、无正根、非有限值或误差超限即失败，不静默回退到自然LR。

CPU门验证r1的四分支符号、外点从基点构造、A减去正确侧、参数/RNG恢复与FP32实际匹配。四baseline每个radius跑5步随机closure，r=.25/r=1下最终参数、U、reference/actual norm、scalar和最终RNG逐位一致，同时unused外点/g2确实改变，支持功能性复用；该CPU证据不是完整ViT跨GPU逐位轨迹证明。

先运行原16条件各两任务×1epoch×1batch短测（每条件2次真实更新，共32次）。所有短测完成并验收、相同初始参数/输入/标签/索引/LR配对以及实际外点参数距离r=1检查通过后，自动放行12条正式训练。新结果只有十任务全预算和全部记录验收后才填入主表。

## 运行和记录

沿用最近授权的物理 GPU2、3；优先空闲GPU3。GPU2上原独立S匹配实验继续完成，释放后本队列自动使用，不抢占其他作业。最大并行2，共享物理index和UUID双锁、实时显存<500MiB/利用率<5%、立即重查UUID。控制器与worker记录host PID/starttime和持久状态，失败停止新启动、保留证据、已运行任务自行完成。

本队列仅Stage1训练，不接续此前失败的同状态诊断/HVP或第二阶段响应任务。记录每步正负探针loss、g0/g1±/g2/q/c21/U范数、方向cosine、参数与有效BA+FC探针距离、实际更新长度、匹配误差、恢复检查，以及输入/标签/索引hash。报告任务等权AAA为主，同时FAA、LA、BWT、遗忘和逐任务矩阵；比较r=.25/r1下P/C保留与D/R反号效应，区分O和A、自然和matched。单seed结果不作统计显著性或Hessian机制已验证声明。

执行说明见 RUN.md，环境复用见 ENVIRONMENT.md；CPU proof、历史基准审计、配置及代码全部冻结到 RUN_PINS.json。动态队列在 runtime/queue_state.json；完整16格结果自动更新 RESULTS.md / summary.csv / results.json。
