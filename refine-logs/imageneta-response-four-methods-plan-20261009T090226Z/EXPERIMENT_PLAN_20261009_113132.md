# ImageNet-A四方法：单任务与持续学习实验计划

日期：2026-10-09，Europe/Paris。问题：当前下降响应重定位与两种历史使用方式，能否改善相同LoRA框架的学习与保持？首轮统一seed2；本次生成实际代码和配置并执行实现短测，正式训练未注册。

## 假设与最小证据

| 假设 | 要排除的替代解释 | 最小证据 | 实验 |
|---|---|---|---|
| 最终更新历史能够改善重定位 | 多一次梯度、不同模型/步长/数据 | current_g2与history_g2同为3次读取，所有设置相同，最终准确率/AAA及时间配对 | 四单任务、四十任务表中的同预算两行 |
| 历史信息的使用方式影响效果 | 把基点缓存与最终缓存混为同一算法 | 保存/加载和日志证明base/final真实来源，比较直接融合及重定位，公开已有负面结果 | 完整四方法与机制诊断 |

SGD为真实1次梯度基线，history_fusion为2次，两个g2方法为3次；相同epoch/样本曝光比较不等于相同计算预算。单seed仅为探索，不写统计显著结论。没有新的frontier组件或Hessian求逆贡献声明。

## 统一基础设置

ImageNet-A历史train5981/test1519，固定0…199类；ViT-B/16、augreg2_in21k_ft_in1k，Q/V LoRA rank10、scale1、可训练LoRA与分类头。batch128，普通SGD、momentum/WD0，50epochs/task，原生epoch cosine T_max50、最低比例.1。全程FP32/native attention、严格确定性、TF32/AMP关闭。

探测半径r=h=eta*globalnorm(g0)，history_weight=.5，radius_scale1；旧GAM固定r=1/rho=.05仅为未启用的参考字段。lambda_flat1是原容器字段，当前不执行原GAM梯度分解。EWC lambda0、skip_task_fisher=true，Fisher近似/归一化/阻尼、权重融合、replay、投影、QR/HVP、最终裁剪/范数匹配全部关闭。

正常每step 1/3/2/3梯度，全部另有1次无梯度postloss前向并恢复状态。记录这一额外费用。各方法独立初始化，布局内相同已实现输入增强、样本顺序、学习率与首梯度；两布局的CE任务定义不同。

## 第一阶段：全部200类作为一个任务

`response_layout=single200, init_cls=200, increment=200, nb_tasks=1, total_sessions=1`，真正使用全部200类与200行分类头。不是十任务序列的Task0截断。唯一任务50epochs、LR=.04，cosine最低.004。

| Run | 方法 | 梯度/step | 正式更新 | 主指标 | 状态 |
|---|---|---:|---:|---|---|
| formal_single200_sgd_s2 | sgd | 1 | 2350 | epoch50全200类测试准确率 | 配置草案 |
| formal_single200_current_g2_s2 | current_g2 | 3 | 2350 | 同上 | 配置草案 |
| formal_single200_history_fusion_s2 | history_fusion | 2 | 2350 | 同上 | 配置草案 |
| formal_single200_history_g2_s2 | history_g2 | 3 | 2350 | 同上 | 配置草案 |

每epoch47batch；每张图像50遍，299050训练样本暴露。画四方法50epoch准确率与训练CE曲线，报告固定epoch50，不从测试曲线挑最佳checkpoint。补充最后更新范数/动态半径分布、秒/epoch、总时间/梯度读取数。

单任务FAA/任务阶段AAA只是在唯一阶段重复最终准确率，不是50epoch的曲线均值；BWT和遗忘undefined，保存null，不填0。

## 第二阶段：10×20持续学习

`response_layout=cl10, init_cls=increment=20, nb_tasks=total_sessions=10`，全部200类。每task50epochs，Task0 LR=.04/最低.004，后续LR=.02/最低.002。任务/头扩张清空历史，epoch内及跨epoch保留。

| Run | 方法 | 梯度/step | 正式更新 | 主指标 | 状态 |
|---|---|---:|---:|---|---|
| formal_cl10_sgd_s2 | sgd | 1 | 2550 | 任务等权AAA、FAA、BWT | 配置草案 |
| formal_cl10_current_g2_s2 | current_g2 | 3 | 2550 | 同上 | 配置草案 |
| formal_cl10_history_fusion_s2 | history_fusion | 2 | 2550 | 同上 | 复用已完成旧HRF，保留来源 |
| formal_cl10_history_g2_s2 | history_g2 | 3 | 2550 | 同上 | 配置草案 |

每task batch数[10,8,8,5,4,3,3,3,4,3]；仍是每图50遍，299050样本暴露。500个局部task epochs与单任务50个全局epochs不是10倍样本训练。两布局的分类竞争、任务顺序、学习率重置/历史清空和评估不同，不用跨布局绝对准确率归因历史机制。

主表列方法、最终信号、FAA、AAA、BWT、LA、forgetting、梯度数与时间；绘制十阶段等权准确率和旧任务保持。FAA/AAA/BWT从原始三角任务矩阵独立重算，沿用native按任务组的准确率精度。不要混入历史固定半径正向GAM/SAM成绩。

## 已有结果复用

旧HRF源：../imageneta-history-response-execution-20261008T175403Z/outputs/formal_history_fused_k10_s2。

| 已完成旧配置 | AAA | FAA | BWT | LA | Forgetting |
|---|---:|---:|---:|---:|---:|
| history_fusion / 旧history_fused | 60.483 | 41.297 | -10.757 | 50.978 | 10.880 |
| 无历史Future Mirror（额外参考） | 62.507 | 44.138 | -11.312 | 54.319 | 11.312 |

旧HRF当前未显示AAA/FAA整体收益。复验190源/26pins、10checkpoint/2550trace/receipt和原始矩阵；新旧HRF24步含dropout/变化LR逐位一致，十阶段CE/梯度、数据label/类序/分组评估和基础协议相同。允许标为旧运行来源的匹配算法结果，不能称新训练已完成或宣称整个GPU轨迹已由CPU对照证明。结果索引results/PRIOR_RESULTS.csv和gates/PRIOR_RESULTS_REUSE_AUDIT.json。

## 运行顺序与决策

| 阶段 | 作业 | 验收/决策门 | 成本 |
|---|---|---|---|
| M0 数值与入口 | 34CPU测试、独立四方法解析/缓存篡改、16CLI配置 | 公式/来源/真实位移/布局/指标均正确 | CPU |
| M1 实现短测 | 4个single200各2updates；4个CL各2task×2updates；1配对 | 实际梯度预算、头200/20、R0恢复、base/final缓存、任务reset、Fisher/融合0 | 24SGD更新，共54训练梯度，另24诊断前向 |
| M2 单任务主实验 | 4个50epoch single200 | 收集固定epoch50和完整曲线，先判断学习/稳定性 | 21150梯度、9400诊断前向，另测试评估 |
| M3 CL主实验 | 新3个cl10，复用1个HRF | AAA/FAA/BWT/LA、同预算current_g2 vs history_g2 | 新17850梯度、7650诊断前向，另测试评估 |
| M4 后续确认 | 若探索结果有价值，才追加seed0/1 | 配对均值±样本SD；负面结果也保留 | 本轮未注册 |

新增正式训练共7项：4single200+3cl10。当前suite仅8短测+1门控，没有正式训练接续。先单任务后CL是分析顺序，阶段结果不会自动选择/修改w、LR、半径或缓存语义；NaN/OOM/恢复错误记录后停止，不用裁剪掩盖。

旧2梯度HRF一次完整CL实测1.558GPUh，可作粗略预算参考；新7项约10–14GPUh、两张空闲卡约5–7小时，需正式运行吞吐校正。单任务额外每epoch测试费用单列。GPU2–6空闲分配、最多2项并行、0/1排除，保留其他任务。

## 日志与解释

每step记录method/source、g0/g1/g2/final哈希与norm、历史向量/实际位移哈希和norm、h/ell、历史是否有效、c范数、relocation从base的哈希/实际距离、g0·d、norm(d)/norm(g0)、真实postloss变化、输入/类别身份、梯度/前向/SGD次数和禁止工作计数。

主要机制结论来自current_g2与history_g2：同为3梯度、最终均只提交g2，历史只改变选点。与SGD的差值仍涉及额外读取成本。history_fusion的缓存是base，history_g2是final，不能把两者混成同一历史响应消融；两者差异还包括直接融合/选点及2/3梯度成本。跨batch历史及虚拟点最终信号不等于纯基点Hessian割线。

优先正文：两布局分开的四方法表及同预算定位比较。附录：完整曲线、动态半径/缓存诊断、时间和负面结果。暂不自动扫描w/r/λ，不新增QR、投影、逆Hessian或多方向优化。
