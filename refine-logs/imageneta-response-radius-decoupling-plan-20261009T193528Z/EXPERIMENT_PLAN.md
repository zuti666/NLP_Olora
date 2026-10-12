# 探测半径与学习率解耦：最小实验计划

**问题**：cosine同步减小提交步长和两次探测距离，是否限制当前响应重新定位的持续学习收益？
**主线**：原 Current-G2、History Fusion、History-G2 定义保持不变。先只给已有最佳 Current-G2 添加独立半径新变体。
**状态**：计划与三份配置草案已保存；未实现、未注册队列、未运行。

## Claim Map
| Claim | 需要的证据 | 实验块 |
|---|---|---|
| C1：探针几何从真实更新学习率解耦值得保留 | 同seed/任务/SGD序列/三次梯度预算下，固定内或外半径的新变体改善AAA，FAA/BWT和成本同时报告 | 2×2核心对照 |
| C2：内、外探针的作用不同 | FixedInner与FixedOuter的差异及同checkpoint/同batch半径稳定性诊断 | 核心对照+离线诊断 |

不能预先声称解耦更好，或只凭单seed准确率证明曲率/Hessian机制。历史缓存的总体必要性不是本轮识别目标。

## 实验块

### B0：原结果复用与定义冻结（必须）
保留SGD及三原方法的7新训练+1旧HRF结果，冻结源码/配置/receipt来源；核心基线是已验证CL10 Current-G2：AAA62.870334、FAA45.252、BWT−9.652222。旧h=eta||g0||、第一探针theta−eta*g0、外探针theta+h*N(a)，最终仅g2。旧HF/HG2缓存源及h/ell不改。

### B1：Current-G2的2×2（必须，仅3次新增完整训练）
设a=g0−gradL(theta1)，h=eta||g0||。所有外层theta2都从BASE theta构造，d=gradL(theta2)，只执行一次原SGD更新−eta*d。表中固定值均为训练参数空间中的绝对L2距离。

| 设置 | 内预测点theta1 | 外探测点theta2 | 状态/问题 |
|---|---|---|---|
| Original | theta−eta*g0 | theta+h*N(a) | 已有seed2结果复用 |
| CurrentG2-FixedInner | theta−0.05*N(g0) | theta+h*N(a) | 新增：有限差分距离固定的效果 |
| CurrentG2-FixedOuter | theta−eta*g0 | theta+0.05*N(a) | 新增：外响应探针不随cosine收缩的效果 |
| CurrentG2-FixedBoth | theta−0.05*N(g0) | theta+0.05*N(a) | 新增：两处几何同时解耦及交互 |

执行建议次序：FixedOuter、FixedInner、FixedBoth。原版结果复用，无需重复训练。三组均为3次训练梯度+1次被动更新后loss前向/step，保持同R0探针重放、只提交g0对应R1、相同epoch调度与事务恢复。不归一化最终更新，不加投影、历史缓存、梯度裁剪或额外系数。

统一：ImageNet-A 10×20类，fixed order0…199；train5981/test1519；ViT-B/16 augreg2_in21k_ft_in1k、Q/VLoRArank10/scale1；seed2/batch128；每任务50epochs；SGD momentum0/WD0；Task0 LR .04、后任务 .02，epochcosine最低比率.1；FP32/native attention/strict deterministic，TF32和AMP关闭；无EWC/Fisher/权重融合/replay。每组2550更新、299050样本曝光。所有非目标配置、训练数据增强和初始化配对保持原设置。

.05处于已有CL10 h中位数0.1033→0.00350之间，是预注册中心而非旧GAM参数继承或最优值。

**主要指标**：任务等权AAA；次要FAA/BWT/LA/forgetting、最后样本准确率、实际计算量和时间。按同一口径比较；单seed标准差记N/A。

**解读**：FixedOuter改善而FixedInner无明显改变，支持外探针幅度路径值得保留；FixedInner改善，则首先检查有限半径非线性/差分数值；FixedBoth优势提示组合效应。无优势则当前数据不支持固定.05半径优于原耦合定义，不否定所有独立半径设置。

### B2：同状态探测稳定性（可与B1并行的离线诊断）
在原版已有Task0/Task4/Task9最终checkpoint，固定一个预先选择的训练batch/增强/RNG，对内距离h、.025、.05、.1比较a/distance的方向cosine、范数比和相邻半径差异；不执行SGD，不用test集选择rho/R。这检查不同尺度的有限差分响应，不能等同于证明H准确。

训练日志逐步保存eta、||g0||、h、实际内外FP32距离、||a||/||g0||、||g2−g0||/||g0||、g0与g2夹角、实际提交步长、更新后同batch loss，以及每个fallback/FP32不可分辨事件。若后续声称“epoch50差分已进入噪声区”，需要额外晚期step状态的同点尺度诊断；当前仅有最终task checkpoint，不能把跨epoch轨迹变化当作这个证据。

### B3：小范围扫描（可选、第一轮结果后再决定）
只扫描有希望的几何方式：
- 若FixedBoth值得保留，令rho=R=delta，delta=.02/.05/.1；.05复用，只加2次seed2训练。
- 若仅FixedOuter值得保留，内仍theta−eta*g0，仅扫描R=.02/.05/.1。
- 若仅FixedInner值得保留，外仍h，仅扫描rho=.02/.05/.1。
不预先展开两个半径的三维学习率网格，不同时修改原学习率序列。额外单任务50epoch对照属于后续扩展，本轮核心先聚焦十任务AAA。

### B4：历史变体（本轮不启动）
HF/HG2保持原定义。先讨论清楚固定探针几何时是保留h/ell历史缩放，还是采用q_actual/ell并独立限制/控制融合强度；二者会改变不同因素。不能直接把h改rho后声称最终更新必然随eta衰减。后期历史分量主导是已有相关性诊断；不能因此给原方法加新系数后仍算原定义。

## Run Order and Milestones
| 阶段 | 内容 | 新正式数量 | 接续条件 |
|---|---|---:|---|
| M0 | 原方法定义与已有baseline复用 |0| 现有receipt/hash已通过 |
| M1 | 独立新代码分支、原legacy逐步等价、公式/恢复/真实距离/CLI测试 |0| 必须显式支持新radius mode |
| M2 | 三新变体配对短测 |0| fullclass/head/direction/3grad/1SGD/RNG/purity通过 |
| M3 | FixedOuter/FixedInner/FixedBoth完整CL10 |3| 逐任务矩阵+2550更新验收 |
| M4 | 展示AAA/FAA/BWT和半径诊断，决定是否扫描 |0| 明确探索性single-seed结论 |
| M5 | 仅一种模式两个额外半径点 |可选2| 不属于当前已登记任务 |

## Compute and Data Budget
原Current-G2 CL10约2.07GPUh/次。核心3次估计6.3GPUh；最多两卡约4.2小时，不含排队。若再加2点约4.2GPUh。仅使用用户指定GPU5/6/7中的空闲卡；当前主机无GPU7，未来执行使用5/6，不能改用0–4。这里仅记计划，不启动队列。

## Risks and Mitigations
固定半径可在小梯度区产生过大相对响应或离开局部区；先做半径/实际步长/loss诊断，并保留原退化处理，不暗中改变学习率或最终方向尺度。只固定内半径仍有外层h耦合。历史q/ell有额外学习率依赖，暂由纯current对照隔离。一个seed只用于探索，不能给统计显著性。

## 配置状态
configs三份JSON是proposal-only。其response_radius_mode=independent_inner_outer_proposal_v1故意不被原trainer接受，防止未实现时未知radius字段被忽略后错误跑成原算法。原3方法原文件不修改。EXPERIMENT_SETTINGS.csv/json记录完整设置，未注册到任何运行队列。

## Final Checklist
- [x] 原方法定义与结果冻结、现有日志耦合核查。
- [x] 核心3新增点及配置草案、主要指标/预算明确。
- [x] 内/外半径与历史缩放分开解释。
- [ ] 新变体实现、短测、正式训练（当前均未进行）。
