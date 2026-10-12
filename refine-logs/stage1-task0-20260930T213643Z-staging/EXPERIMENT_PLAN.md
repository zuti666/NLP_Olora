# Stage 1：独立单任务 g1/g3 实验

日期：2026-09-30。用户授权本轮只做单seed、单任务；2/3任务在结果出来后再展开。本文先注册设置与解释边界，不将未完成训练计为结果。

## 固定设置

- 数据为当前 ImageNet-A 固定class_shuffle=false的第一个20类，沿用现有train/test划分与预处理。DataManager仍保留10×20的类划分，仅max_train_tasks=1终止；不能改成init_cls=200，也不能改increment=0。
- 全部10臂从同一pretrained ViT-B/16、同一seed2初始化独立训练，不从10-task checkpoints接着训练。冻结主干，Q/V LoRA rank10＋分类头可训练；初始化参数和数据顺序记录hash。
- 50epoch，batch128，初始LR=.04，cosine最小LR=.004（明确固定cosine_min_lr_ratio=.1），SGD momentum=0、weight_decay=0，无early stopping，不按test最优epoch选checkpoint。Task0最终指标取epoch50。
- 内层rho=gam_grad_rho=.05；外层r=gam_grad_norm_rho按矩阵变化；lambda_flat=1，gamma_GAM=0，adaptive=false，beta2=1。
- ewc_lambda=0、ewc_skip_task_fisher=true；Fisher估计/归一化/damping、task内/末权重融合、SGV、回放、附加ΔW惩罚全关闭。
- 主因果组CE/S/G/sum/mean均保留clean缓存＋完整GAM四probe。CE只把最终更新替换为g0，保留相同探测开销；它的计算时间不代表高效标准SGD。TrueSAM作为独立两次梯度的实现对照，不能与主组混作同开销对比。
- 训练数据采样/增强使用独立、固定的随机生成流；追加评估/快照/诊断保留并恢复RNG和model modes。这样新增指标与TrueSAM不同forward次数不改变各组训练数据顺序。

## 独立训练矩阵（10次，每次50epoch，seed2）

| ID | 最终方向 | beta1 / beta3 | 外半径r | 角色 |
|---|---|---|---:|---|
| CE | g0 | 最终方向显式覆盖 | 1 | 普通CE更新，同GAM探测流程 |
| S | g1 | 1 / 0 | 1 | SAM-like主参考 |
| G | g3 | 0 / 1 | 1 | 更深probe |
| Mix-Sum | g1+g3 | 1 / 1 | 1 | 未平均组合 |
| Mix-Mean | (g1+g3)/2 | .5 / .5 | 1 | 同方向、半幅值 |
| G-r0 | g3 | 0 / 1 | 0 | g3(r0)≈g1数值sanity |
| G-r01 | g3 | 0 / 1 | .1 | 外半径扫描 |
| G-r025 | g3 | 0 / 1 | .25 | 外半径扫描 |
| G-r05 | g3 | 0 / 1 | .5 | 外半径扫描 |
| TrueSAM | 标准两梯度SAM | 不适用 | 不适用 | rho=.05实现对照 |

G(r=1)在两个block共用，故不是11次。所有结果为新的独立单任务训练，旧10-task Task0仅作额外来源核对，不填充此表。

## 新增运行指标

每epoch：训练clean CE、训练sample accuracy、固定heldout CE和sample accuracy、每类准确率/宏平均（可用时）、实际使用LR、训练时长、eval时长、吞吐、GPU峰值显存、实际梯度/forward次数。Train在线CE和epoch-end eval CE取样/模式不同，不直接把差当严格generalization gap。

每epoch固定第一batch：记录g0/g1/g2/g3范数、cos(g1,g3)、cos与clean方向、a=<g3,g1>/||g1||²、R_perp=||g3-a*g1||/||g3||、训练最终方向范数、实际参数变化、LoRA BA与FC有效变化。记录sample indices/input hash，保证比较可追溯。TrueSAM训练只产生g0/g1；为核验实现，第一batch额外运行隔离且完整恢复的GAM诊断，单独标记source与梯度次数，其g2/g3不冒充SAM训练中间量。TrueSAM含额外诊断的wall time不能作为纯两梯度SAM速度。

保存epoch 0/10/20/30/40/50检查点，包括累计LoRA A/B、FC及任务元数据，另存optimizer/scheduler/RNG侧文件。保存不是after_task，不允许因此递增task或合并权重。新task不训练，所以单任务报告不使用遗忘、BWT或AAA/FAA主指标。

主指标a11：epoch50的Task0 heldout sample accuracy；同时显示最终CE、完整50epoch曲线和实际计算成本。所有结果只有seed2，无跨seed误差条或显著性结论。

## 同状态局部诊断

先在S/G/Mix-Sum/Mix-Mean四个最终checkpoint和一份共享的S epoch0 checkpoint执行，共5个anchor，每anchor4个固定当前task批次，每批128。其余epoch快照保留，可按初步结果扩展轨迹诊断。复用已通过真实Task4短测的paired_gam_diagnostic实现；每候选后完整恢复模型状态和RNG。

- 分解g3=a*g1+r_perp；同时构造g1、g3、sum、half_sum、a*g1、(1+a)g1、g1+λr_perp及(1+a)g1+λr_perp，λ=0/.5/1/2；记录alpha方向插值0/.25/.5/.75/1。
- 所有反事实匹配同anchor下−.02*g1实际引起的联合有效步幅：concat(各LoRA B'A'−BA, FC weight/bias差)，记录LoRA与FC分量。这里.02是固定虚拟诊断步长，不是Task0训练LR=.04，也不是epoch50已衰减的LR。实际相对匹配误差≤.1%，失败明确标记。
- 对GAM r=0/.1/.25/.5/1记录probe参数/BA/FC距离与输出KL；SAM沿g0方向匹配GAM g3 probe的参数距离与KL，比较独立eval批次loss改善。
- KL对齐使用一个固定probe-metric batch；loss可在同一eval批次上度量，属于局部探索性诊断，不把它包装为独立长期泛化验证。
- task0无旧task loss或forgetting；所有这些匹配实验是局部一步，不能当作长期训练AAA或自动证明因果机制。

## 怎样解释并选择G*

1. 首先检查CE/S/G/Sum/Mean的a11和损失曲线，以及S与G-r0、S与TrueSAM的一步数值一致性和独立训练差异。实现不一致先排查，而非解释几何。
2. Sum和Mean的初始同状态方向相同、范数差2；完整训练后轨迹会分离。两者差异检验幅值贡献，但Mean优于G本身仍不足以完全排除步幅影响，需要matched-effective-step。
3. 候选G*仅为已真实训练的g3(r)或Mean(r=1)；未训练Mean(r<1)不能直接选定；若候选r*<1，再补该候选的最终checkpoint匹配诊断，不把r=1的局部证据转移到r*。Sum保留为尺度对照。
4. 优先要求a11/CE不明显退化，并且同状态匹配支持有效方向，再冻结r、rho、beta及训练协议；不能仅按单个test最高点宣布机制最优。若两类证据矛盾，报告未判定，不自动扩展2/3-task队列。
5. 本轮沿用的heldout集若用于选G*，其成绩属于开发/探索结果，不是选后的独立确认；进一步确认需预先固定方案与独立验证证据。用户本轮只要seed2，不自动追加seed0/1。

## 后续阶段（仅计划，未注册自动运行）

2-task从pretrained重新独立训练，固定S=g1和选定G*，比较SS、GS、SG、GG。记录a11、a21、a22，旧任务变化R1=a21−a11、新任务获取a22、平均A2，以及完整Task1学习轨迹。Δform=R_GS−R_SS，Δcont=R_SG−R_SS，interaction=R_GG−R_GS−R_SG+R_SS。

差分R去掉初始准确率的加性偏移，但并不自动控制solution异质性、天花板效应或相同更新规则下不同实际轨迹；formation解释要保留这一限定。

3-task也从pretrained独立重训，先SSS/GSS/SGS/SSG/GGG，固定超参、不重新扫r/rho/beta。报告a矩阵、Task0跨两次后续学习的保持、Task1获取/后续保持、最后历史集准确率、新Task2获取。是否追加交互组由结果决定。

## 执行顺序与验收

独立数值/CPU检查→真实单批短测（含CE、S、G、sum、mean、TrueSAM）→10项正式队列→完成后只启动上述四个最终checkpoint及共享epoch0离线诊断→汇总Task0结果并判断G*。不自动启动2/3-task。

沿用Pilot_new_local和190文件冻结source，新增wrapper与指标hook单独留档，不覆盖旧source或旧实验。只在GPU0–6显存<500MiB且util<5%的设备启动，并与现有队列共用文件锁；不停止用户其它任务。当前旧训练仍占GPU0–4，空闲GPU动态分配。

验收包括仅Task0/50epoch、20类数据范围、原始sample数量、初始参数与训练数据顺序hash、实际probe次数、所有损失/参数有限、Fisher/融合调用被禁止、六个epoch快照及真实完成凭据。缺失/异常不计成功，不自动盲目重试。
