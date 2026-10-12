# 新实现、计划与进度核查

核查结论：阶段1核心实现与计划一致，未发现分支符号、外层起点、状态恢复或提交更新的核心错误；已经通过真实GPU短测。阶段2的任意向量HVP公式实现符合规格，但真实ViT中心差分门槛尚未通过，不能将公式正确与数值验收成功混同。

本次仅检查运行中的冻结代码与结果。原始CopyFlatCV代码未改；本目录78个冻结执行文件以及原始190个文件哈希均保持不变。队列继续执行，GPU0/1排除。

## 代码在哪里、如何接入

以下文件位于本运行目录的trainer/中：

| 文件 | 位置 | 实际职责 |
|---|---:|---|
| four_probe_train.py | 35、76、110 | 冻结配置；替换学习器_build_optimizer与_step_batch_with_log；沿用原任务训练/分类器扩展/余弦调度/评估/保存流程 |
| direction_step.py | 137、152、160、166 | 四次普通CE探测、P/D/C/R构造、最终更新方向选择 |
| direction_step.py | 242、275、292、304 | 临时正向g1有效位移参考、匹配标量求解、唯一实际SGD step |
| response_hvp.py | 55、107、164、232 | 参数/梯度/buffer/cache/RNG事务；共同math后端；独立HVP图；Hstart/Hmid |
| effective_step.py | 122、175、201 | 全部LoRA BA及全部seen FC的有效位移、多项式和实际FP32更新复核 |
| hvp_gate.py | main入口 | 真实检查点的同点一阶梯度与中心差分HVP验收 |
| mechanism_diagnostics.py | 82、253 | 12个共同保存状态×2预注册批次的只读四格与响应比较 |

旧GAM源文件：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py。

旧GAM路径计算g0/g1/g2/g3，并使用beta组合、gamma正交修正、direction projector，再交给基础优化器。当前实验入口直接使用torch.optim.SGD；不会执行旧GAM.step或gradient_decompose。保留第一上升探针与外层定位思想，去掉g3探针，增加下降侧g1−及显式最终方向选择。旧学习器还额外读取clean CE，所以旧主线五次梯度与本轮四次探针不是同一执行协议。

旧方法以减扰动恢复参数；当前用原始快照copy_恢复，并恢复非注册LoRA缓存、buffer、梯度、模式与随机状态。当前LoRA前向即便dropout=0也会创建临时随机Linear，因此各探针都重放R0，整步只推进到g0前向后的R1；数据增强使用独立固定序列，评估也做完整状态恢复。

## 本轮真实计算

phi包含全部可训练Q/V LoRA A/B因子及全部seen FC，骨干冻结。令N(v)=v/(norm(v)+1e-12)：

g0=g(phi)，g1+=g(phi+.05N(g0))，g1−=g(phi−.05N(g0))。

| 分支 | 第一侧梯度 | q | 外层点 |
|---|---|---|---|
| P | g1+ | g1+−g0 | phi+.25N(q) |
| D | g1− | g1−−g0 | phi+.25N(q) |
| C | g1− | g0−g1− | phi+.25N(q) |
| R | g1+ | g0−g1+ | phi+.25N(q) |

外层点从原phi出发，并非从phi1出发。最后选择O_m=g2m或A_m=g0+(g2m−g1m)。CE/S+/S−分别只提交g0/g1+/g1−，但也执行相同四次探测。所有临时读数先恢复原参数，再提交一次SGD更新。

g2=g0+c10+c21，其中c10=g1−g0、c21=g2−g1；A=g0+c21删除最终加性c10，但仍用c10定位外点。D/C依然使用第一次差分，反向探针不等于删除第一部分。

自然更新phi'=phi−etaU，不做最终单位化。匹配版本仅调节正标量tau，使候选更新的有效长度与同状态正向g1+在名义eta下的实际长度一致：norm²=sum_l norm²(Delta(B_l A_l))+norm²(Delta FC)。它没有投影或正交分解，不能控制层间分配。求最小正根后以实际FP32参数值复核；误差超限报错，不静默回退学习率。16个短测最大匹配相对误差3.7614633e-8，门槛1e-3。

Fisher估计、权重融合更新、禁止模块尝试均为0。已经完成的每组有20次关闭状态的融合钩子读取，验证权重/参数引用未改；它们是空操作，不能误写为任何融合函数都从未调用。没有梯度正交修正；探针构造需要的N(v)保留。

这是专用冻结入口：rho=.05、r=.25、rank10在four_probe_train.py中覆盖配置。未来若扫描它们，单改JSON不够，需建立新运行并更新入口/验收/冻结清单。

## 两阶段预算与条件

统一ImageNet-A 10×20类、seed2、ViT-B/16 augreg2_in21k_ft_in1k、rank10、scale1。每任务50epochs，batch128；SGD momentum/weight decay=0，Task0 LR=.04、后续=.02，cosine最低LR为初始值的.1；FP32、TF32关闭，无replay/Fisher/EWC/权重融合。

| 批次 | 完整训练 | 数量 |
|---|---|---:|
| 阶段1首批 | CE/S+/S−；O_P/O_D/O_C；A_P/A_D/A_C，自然步长 | 9 |
| 阶段1补充 | O_R/A_R自然；四个A匹配；CE匹配 | 7 |
| 阶段2 | Hstart/Hmid/Residual-only，自然与匹配 | 6 |
| math桥接 | 同math后端Sec=A_P，自然与匹配 | 2 |

合计24完整训练+24短测+1HVP完整门槛+2只读诊断=51作业。阶段1全部16短测通过才释放首批9组；首批完成才释放后7组。阶段1全部完成后运行共同A_P检查点Task0/4/9×2批次的完整HVP门槛与机制诊断；门槛及第二阶段8短测通过，才释放第二阶段8完整训练。

阶段2固定P：d=phi2−phi1，从实际快照取detach向量。Sec提交g0+(g2−g1)，Hstart提交g0+H(phi1)d，Hmid提交g0+H((phi1+phi2)/2)d，Residual-only提交g2−g1。HVP通过对gradL(x)点乘stopgrad(d)再反传实现，不对端点生成反传，不重复乘r/rho。不同训练会有不同轨迹；严格同状态响应比较另用共同参考检查点进行。

原生注意力二阶反传不支持，故第二阶段使用共同math后端，新增两组Sec桥接。初始math同点一阶梯度误差0，但Hstart/Hmid最佳中心差分相对误差0.011692/0.017888，高于0.005，且没有相邻两个h同时通过。保留失败；完整门槛未运行，若仍失败则第二阶段不启动，不放宽阈值。

主要报告AAA，辅以FAA、LA、BWT/遗忘。O族仅自然步长，不能单独称纯方向效应；A_P与A_C同时改变采样侧和减法参照；单seed仅描述性比较。历史成绩不能充当新协议严格配对基线。

本文件下方进度由当次队列、日志、退出及验收receipt读取生成；之后实时进度以runtime/queue_state.json为准。

## 本次实时快照

检查时间：2026-10-02 17:08:29，巴黎时间。控制器身份与当前5个worker存活已核验。

完整训练：已验收3/24，运行5，失败0，等待16；第一阶段短测16/16通过，第二阶段短测8组尚未启动。GPU2–6均在运行本轮训练，GPU0/1排除。

| 完整条件 | 状态 | GPU | 进度 | AAA | FAA | 遗忘 |
|---|---|---:|---|---:|---:|---:|
| S1-01_CE_natural_s2 | completed | 2 | 10任务/2550更新 | 60.337 | 40.140 | 12.240 |
| S1-02_Splus_natural_s2 | completed | 3 | 10任务/2550更新 | 63.234 | 44.578 | 10.269 |
| S1-03_Sminus_natural_s2 | completed | 6 | 10任务/2550更新 | 51.090 | 25.368 | 22.018 |
| S1-04_O_P_natural_s2 | running | 4 | Task1 epoch9/50，566/2550更新 | — | — | — |
| S1-05_O_D_natural_s2 | running | 5 | Task0 epoch26/50，253/2550更新 | — | — | — |
| S1-06_O_C_natural_s2 | running | 6 | Task0 epoch16/50，158/2550更新 | — | — | — |
| S1-07_A_P_natural_s2 | running | 3 | Task0 epoch8/50，75/2550更新 | — | — | — |
| S1-08_A_D_natural_s2 | running | 2 | Task0 epoch7/50，70/2550更新 | — | — | — |

本seed自然步长下，正向g1相对g0的AAA/FAA增益为2.897/4.438个百分点；反向g1为-9.246/-14.772个百分点。

这只检验第一邻点梯度直接用于最终更新的效果，不能推出跨点c21响应依赖正向第一探针，需等O/A四格完成。当前不计算单seed显著性，也不以尚未完成的阶段指标参与排名。
