# g1 / g3 机制实验：执行设置

日期：2026-09-30。依据用户附件，实施首阶段三个实验块。所有新增训练仅 seed 2；沿用已验证冻结源码与环境。附件原文随本计划保存。此文件描述设置与判据，尚未完成的实验不计为结果。

## 问题与结论边界

C1：g3 相对 g1 的收益，来自更大更新尺度、更远探测，还是新增方向？
C2：g1+g3 的收益主要改善新任务获取、旧任务保持，还是二者的权衡？
首轮不扩展到 Full/Fixed-A/Dynamic-LoRA、Fisher/Hessian spectrum、rank 或更多 seeds。

当前已有纯 GAM 三臂：g1 AAA/FAA=62.166/45.132；g3=65.993/50.679；sum=67.861/54.726。它们不能单独证明方向互补，因为 sum 未除2、且比较来自不同训练轨迹。
原始矩阵显示获取均值 LA：g1=53.436，g3=56.728，sum=61.583；最终旧9任务均值44.911/51.074/54.076；遗忘9.482/7.011/7.619。sum 的遗忘高于 g3，不能预设其优势是抗遗忘。

## 固定训练协议

- ImageNet-A，class_shuffle=false，10×20类；seed=2；ViT-B/16 冻结预训练主干；Q/V LoRA rank10 与分类头可训练。
- 每任务50epoch，batch128；SGD momentum=0、weight_decay=0；首任务LR=.04，随后=.02；cosine；无回放。
- lambda_flat=1，gam_grad_gamma=0，gam_adaptive=false；内层 gam_grad_rho=.05。
- ewc_lambda=0，ewc_skip_task_fisher=true；任务内/任务末权重融合false；Fisher估计、归一化、damping、额外ΔW惩罚关闭。
- 保留4次 GAM probe＋1次clean梯度缓存；不因系数为0提前跳过前向，避免变更RNG消耗。
- g0=∇L(θ)，g1=∇L(θ+ρN(g0))，g2=∇L(θ+rN(g1−g0))，g3=∇L(θ+rN(g1−g0)+ρN(g2))。

## A：更新尺度与连续半径（5次新正式训练）

| ID | 最终方向 | beta1,beta2,beta3 | gam_grad_norm_rho | 目的 |
|---|---|---|---:|---|
| mean_sum | (g1+g3)/2 | .5,1,.5 | 1 | 与已有sum比较，检查未平均的幅值贡献 |
| g3_r0 | g3 | 0,1,1 | 0 | 核验r=0退化为g1的端点，仍单独跑正式训练 |
| g3_r01 | g3 | 0,1,1 | .1 | 连续外层干预 |
| g3_r025 | g3 | 0,1,1 | .25 | 连续外层干预 |
| g3_r05 | g3 | 0,1,1 | .5 | 连续外层干预 |

r=1 g3、g1、sum和原gamma=.1基线均复用之前完成且hash固定的结果。原gamma=.1仅为参考，不混入gamma0的受控切片。此前strength=.5与half-scale实验仍含EWC/不同r/gamma，不复用为本次尺度对照。

半sum保持同一组合公式的方向、改变幅值，但半sum胜g3仍不能独立证明方向收益：两者范数未必相同。r序列同时改变探测方向与距离，需结合B中的距离匹配。r=0与g1的数学等价在实际GAM CPU float64预检中验证；实际训练是否接近另外报告，不能预填相同成绩。

## B：同状态机制诊断（已有checkpoint，无重训）

来源：g1、g3、sum三条轨迹，各取task0、4、9终点，共9个anchor。每个anchor主分析使用当前task独立固定批次；首个smoke1batch，正式4batch，batch128。所有候选在每个anchor内配对，不跨轨迹做向量加减。task0没有旧任务，旧任务指标记NA。

恢复同一冻结主干、cumulative LoRA A/B、对应fc_state；每次诊断前保存参数、buffer和CPU/CUDA RNG，分支结束copy恢复。不得将各task累计LoRA相加；loader的_known_classes为下一任务语义，当前任务CE必须显式用[start:end]分类头切片和labels-start。checkpoint未存训练中期RNG，因此只称task-end局部诊断。

1. 保存g0/g1/g2/g3原始范数、cos(g1,g0)、cos(g3,g0)、cos(g1,g3)，以及a=<g3,g1>/||g1||²、r_perp=g3−a*g1、R_perp=||r_perp||/||g3||。另分LoRA与FC报告能量占比。a不截断；零梯度、a<−1或接近−1明确标记。
2. 对g1、g3、sum、scale=(1+a)g1、g1+λ*r_perp（λ=0,.5,1,2）做相同步幅一步反事实。再做alpha=0,.25,.5,.75,1的单位方向插值N((1−alpha)N(g1)+alpha*N(g3))。这些是离线方向探测，不是新增长期训练曲线。
3. 匹配实际有效权重更新范数：目标取同anchor/batch下−.02*g1造成的有效变化；LoRA部分精确计算B'A'−BA，再与FC权重/偏置变化拼接求Frobenius/L2范数。单独记录LoRA、FC及参数空间范数。用标量求根并实测误差≤1%；无解/零目标/非有限记录unmatched，不能回退后宣称匹配。注意这不是仅匹配LoRA因子坐标步长，也不保证功能距离相同。
4. 评估独立当前任务batch与旧任务宏平均loss变化、准确率/margin（可用时）。旧数据仅用于离线评估，不进入更新方向或训练队列；不得把训练batch诊断称为严格未见训练集的泛化评估。
5. 在同anchor扫描r=0,.1,.25,.5,1：记录probe参数距离、BA有效距离、FC变化、固定batch all-seen logits上的KL、g1/g3真实有效更新夹角。比较g2和g3，判断末尾ρ小步是否提供额外信息。
6. larger-radius SAM诊断：先令SAM probe沿N(g0)走到与g3 probe相同的参数距离；进一步在固定诊断batch上搜索KL匹配的SAM半径并标记匹配误差。只搜索距离不搜索准确率，非单调/无括区不强行匹配。不把局部匹配结果伪称长期训练AAA。

重要退化：当1+a>0，scale与g1为同一条射线。经过相同有效步幅匹配后，两者应重合，可作sanity；不能重复计为独立方法。正交分解是离线诊断操作，原5个训练不会启用GAM的正交修正。

Hessian/Taylor解释只适用于局部近似，r=1不能默认高阶余项可忽略；R_perp大只说明新增方向多，不证明有益。r与D_fun/FAA相关也不是中介因果证明。

## C：准确率矩阵与呈现

主指标AAA_task=阶段均值再平均；FAA_task=最终任务宏平均。另报sample AAA/FAA、LA=对角均值、最终旧9task均值、BWT=旧任务(最终−对角)均值、forgetting=旧任务(历史最高−最终)均值。BWT与forgetting不混用。

图1：实际已完成r点的AAA/FAA曲线，下方对应离线D_dir、D_W、D_fun（按anchor配对汇总）。图2：matched-effective-step当前任务loss改善 vs 旧任务loss增量，颜色代表方向；图3：alpha/λ的离线方向曲线和R_perp分布。离线loss曲线不可标为AAA曲线。

只有seed2：按task/source分层报告配对差、median/IQR，不把batch/anchor当独立seed或宣称跨seed显著性。

## 执行与预算

先完成CPU数学检查和配置/source/environment验证；正式训练复用已通过真实任务验证的冻结训练代码。5个训练并行上空闲GPU，单GPU每次历史耗时2.42–2.47h，总计约12.1–12.4GPUh，实际排队/IO可能增加。诊断先真实GPU smoke检查恢复、切片和匹配，再启动9anchor队列；运行时间以首anchor实测更新。

GPU池0–6；训练max_parallel=5，诊断max_parallel=2；双方共用/tmp/olora_gamfs_gpu_locks并在启动前重查显存<500MiB、util<5%。只在空闲设备启动；失败保留日志，禁止重复覆盖已有输出。

验收：每训练10任务、500epoch、10条skip-Fisher，EWC/融合贡献均0，原始矩阵有效下三角有限，completion receipt和hash一致。诊断验收：r0恒等、分解残差、参数/buffer恢复、有限性、实际匹配误差与匹配失败数量。

后续条件项：若离线方向干预支持新增方向收益，再安排匹配尺度的长期g1+λ*r_perp或alpha训练；若距离匹配的SAM也表现相近，优先解释为探测距离效应。任何长期新增方向训练都应独立注册；本批队列仅含上面的5个正式训练和9个离线anchor。
