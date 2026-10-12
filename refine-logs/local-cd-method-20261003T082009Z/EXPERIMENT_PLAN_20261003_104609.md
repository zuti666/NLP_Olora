# Local-CD：局部中心差分回加实验计划

日期：2026-10-03T10:20:09.863446+02:00。当前状态：已实现框架，CPU验收另见CPU_VALIDATION.json；六组短测与六组正式训练均仅保留计划，已从待执行清单移除。本方案不修改其他运行队列。最后复核：2026-10-03T10:33:17.166064+02:00；核心7组和框架3组CPU检查均通过，独立代码复核通过。

## 问题与方法

已有P/C用正侧或符号校正的负侧梯度差分选择外点，然后读取g2。本轮提出不同的局部更新：不构造外点，不读取g2/g3，直接将两侧共同响应加回g0。

定义同一批次、同一前向随机函数和同一原始状态上的g0=∇L(θ)，u0=g0/(||g0||+ε)，g±=∇L(θ±h u0)。冻结u0作为探针向量，不对其生成过程反传。

Uforward=g0+(λH/h)(g+−g0)；Ubackward=g0+(λH/h)(g0−g−)；Ucd=g0+(λH/(2h))(g+−g−)。最后从原θ提交一次θ←θ−ηU。

首轮h=.05、λH=.05、ε=1e−12，因此Uforward=g+，Ubackward=2g0−g−，Ucd=g0+(g+−g−)/2。h、λH、η分别表示采样半径、修正强度、整体学习率。不能漏掉λH而把中心修正放大20倍；不对最终U或修正再归一化。

方法入口名local_cd_v1，local_method可选ce/forward/backward/central；Local-CD是项目新增方法名，尚不作全局算法新颖性声明。旧P/C、O_C和A_C准确率不能填入新方法表。

相关文献已有梯度范数正则的单侧有限差分实现：[GNP，ICML 2022](https://proceedings.mlr.press/v162/zhao22i.html)。[ICML 2023有限差分GR](https://proceedings.mlr.press/v202/karakida23a.html)研究平方范数、比较前后向与双重反传，并讨论中心差分需要额外梯度。此处用非平方范数的局部响应，在LoRA持续学习中验证对称化；不以中心差分公式本身主张算法首次提出。有限差分偏差也可能影响优化结果，近似更准确不是训练更好的充分条件。

## 待验证主张

| 主张 | 最小证据 | 反解释与边界 |
|---|---|---|
| C1 局部中心修正能产生有用的持续学习更新 | L04对L01的完整AAA/FAA/LA/BWT/Forget，原始十任务矩阵 | 单seed固定配置描述性证据，不保证所有h/λ都有效 |
| C2 两侧对称化有额外训练价值 | L04对L02/L03；L06对L02/L05 | 匹配用于排查总体有效步幅，不能证明层间分配、函数变化或轨迹相同 |

本轮不把更高的有限差分近似阶、较低随机方差、更平坦解、更少遗忘等主张混为一个结论；不加入新的容量、数据或预训练模型差异。

## 理论边界

在足够光滑、固定u0的条件下，(g+−g−)/(2h)=Hu0+O(h²)；单侧为O(h)。中心O(h²)可在L局部C4、相关导数有界时保证。有限精度减法误差可能随1/h放大，因此不默认越小h越好。

忽略ε且g0非零，Hu0=∇||∇L||，更新近似L+λH||∇L||的梯度。保留ε时，精确极限对应L+λH Fε(||∇L||)，Fε(s)=s−ε log(1+s/ε)。这是可训练LoRA因子与分类头坐标中的梯度范数，不是BA运动范数，也不是邻域最大梯度范数的原GAM完整目标。

恒H二次模型上，任意λH均有Uforward=Ubackward=Ucd=g0+λH Hu0；λH=h时等于g+。所以对称化的额外价值只能在非线性有限邻域或数值行为中检验，不能通过二次函数“证明算法比SAM更好”。

令c=(g+−g−)/2，a=(g++g−−2g0)/2，κ=λH/h。则Uf=g0+κ(c+a)，Ub=g0+κ(c−a)，Ucd=g0+κc=(Uf+Ub)/2。c/a先称共同有符号响应/侧间不对称；解释为纯曲率成分需局部展开支持。

## 六组正式训练

| ID | 方法 | h=λH=.05时最终方向 | 步长 | 问题 |
|---|---|---|---|---|
| L01 | CE | g0 | natural | 同三探针预算基点方向对照 |
| L02 | Local-Forward | g+ | natural | 相同系数的SAM式方向参考 |
| L03 | Local-Backward | 2g0−g− | natural | 负侧符号校正直接回加 |
| L04 | Local-CD | g0+(g+−g−)/2 | natural | 对称局部响应直接回加 |
| L05 | Local-Backward | 2g0−g− | matched | 排查相对Forward的步幅影响 |
| L06 | Local-CD | g0+(g+−g−)/2 | matched | 检验对称化收益是否保留 |

所有条件都执行g0→g+→g−三个普通一阶梯度计算，包括CE和Forward。Forward匹配为身份条件，不另计训练；不使用旧三/四/五探针成绩替代本协议的基线。

## 固定训练与评估

ImageNet-A原train/test及类别顺序，200类固定10×20；seed2；沿用相同预训练ViT-B/16与冻结骨干，Q/V LoRA rank10、IdentityScale，以及全部已见类别FC可训练。每任务50epochs、batch128；SGD，momentum/weight_decay均0，Task0 LR=.04，后续=.02；各任务cosine，最低LR=.1×初始LR；FP32、TF32关闭。

六组统一math注意力后端、严格确定性配置、固定增强采样序列。普通math重放基础设施已有24/24复核通过，但其范围是旧Sec/Residual；新三探针仍需独立短测。HVP门禁继续未放行，不过本方法训练不调用create_graph、二阶反传或真实HVP算子。

EWC λ=0，任务Fisher跳过，近似Fisher关闭，in-task/endpoint权重融合关闭，SGV/正交修正/replay关闭；lambda_flat=1。模型学习器名称沿用冻结源以复用数据和分类头生命周期，实际梯度入口为新增local_cd_v1；运行计数必须显示Fisher、merge、projection、autogradHVP均0。

新方法没有外层半径r。有效配置和summary移除gam_grad_norm_rho/gam_grad_rho，记录active_probe_parameters={h,strength,eps}及outer_radius=null。原r=.25只是历史来源标签，不是本方法生效超参。

主指标任务等权AAA；同时FAA、LA、BWT、Forget、完整三角任务矩阵及样本口径单列。十任务检查FAA=LA+.9BWT。归约定义沿用原审核通过的native生命周期：对全部已见类argmax评估，以真实标签计准确率。

## 更新尺度控制

在候选自己的当前状态，反事实Forward更新使用相同h和λH：δref=||Ψ(θ−ηUf)−Ψ(θ)||，Ψ含全部未缩放LoRA B@A以及全部seen FC权重/偏置。为Ub或Ucd求正标量τ，使||Ψ(θ−τU)−Ψ(θ)||=δref。

复用现有实际因子FP64 Gram几何和最小正根匹配器，包含ΔBA=−τ(B UA+UB A)+τ² UB UA；用实际舍入后的参数复核，相对误差阈值1e−3。无正根或复核失败时恢复并停止，没有回退名义LR。λH不等于h的可配置实现也必须匹配Uf，不能错误匹配raw g+。

## 运行指标

每步记录g0/g+/g−范数，前/后/中心差分估计范数，修正/g0比例，U与g0夹角，c/a范数及||a||/||c||，前后差分余弦；实际正负探针距离及BA+FC变化，最终LoRA/FC/联合有效位移、τ/η与匹配误差。记录三次梯度、唯一SGD更新、R0重放及R1单次推进、恢复检查、分类头继承与样本/增强序列身份。

每个任务首个训练批次先做两次完整三探针的零更新重放检查，要求g0/g+/g−/U逐位相同；失败保存报告并停止。其6次梯度计入isolated_diagnostic_ce_gradient_passes，不能算入训练预算或抹除失败。

## 执行顺序与预算

| 阶段 | 内容 | 放行标准 | 状态 |
|---|---|---|---|
| M0 | CPU解析、独立Hessian、零梯度、状态/RNG、唯一更新、非单调匹配/失败恢复 | 全部CPU组通过，代码独立审查完成 | 已通过：核心7组＋框架3组；独立审查通过，见CODE_REVIEW.md |
| M1 | 六组各Task0/1一epoch一batch短测 | 各2个更新、6训练CE、12诊断CE；头扩展/重放/匹配/恢复通过；增强序列和首任务初始化一致 | 仅保留计划，不排队 |
| M2 | 六组各十任务50epoch完整训练 | 全部M1通过后，先CE与Forward，再Backward/CD自然及匹配；保持每任务入口重放门禁 | 仅保留计划，不排队 |
| M3 | 原始矩阵汇总与共同/不对称响应分析 | 精确完成预算、无模块混入、匹配失败不得算完成；所有六组使用同一口径 | 等待M2 |

若任务数据保持原协议，每条2550实际更新、7650训练CE梯度；六条15300更新、45900训练CE，另360任务入口诊断CE；六组短测合计108CE（36训练+72诊断）。GPU小时待真实短测测量后外推，不用梯度次数比例冒充墙钟加速。候选允许GPU2–6，排除GPU0/1，每卡一条，使用当时空闲资源；本次没有排队或启动。

## 结果判定与条件后续

CD优于CE但接近Forward：支持局部修正可用，未显示对称化额外价值。CD优于两个单侧且匹配仍保留：支持该协议下对称化训练价值。自然优势匹配后缩小：更新尺度参与收益。数值更准确但CL不更好：近似精度不是训练效果的充分条件。CD不如单侧：保留单侧邻域信息可能有价值，也可能固定强度或数值误差不合适，不能否定所有配置。

完成第一轮后，若需要分离探针半径与修正强度，才考虑CD的h∈{.025,.05,.1}且λH=.05，以及λH∈{.025,.05,.1}且h=.05；共享中心，总共只新增4个单seed点。它们是条件后续，不列入当前六组，也未加入队列。更广泛seed/dataset或效率对照另行设计。

## 代码与当前证据

[核心local_step.py](/data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/local_step.py)；[训练入口train.py](/data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py)；[CPU验证validate_cpu.py](/data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/validate_cpu.py)；[配置生成make_plan.py](/data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/make_plan.py)；[suite.json](/data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/suite.json)；[commands.md](/data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/commands.md)。

用户附件中的sandbox压缩包和8项CPU结果没有作为本机验收证据；本机重新实现并独立验证，核心7组与框架3组共10组通过；实际结果保存CPU_VALIDATION.json和FRAMEWORK_VALIDATION.json，独立复核见CODE_REVIEW.md。真实ImageNet-A性能暂无新结果。新方法的GPU短测与正式训练均待执行。

## 用户调度更新

2026-10-03T10:46:09.535830+02:00：按用户要求，本方案仅保留计划，不进入运行队列。active suite.json已清空jobs（0项），automatic_start=false、status=plan_only。配置和代码保留；下文实验顺序仅描述未来方案，不授权自动提交。其他实验队列未修改。
