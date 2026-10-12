# 本次设置、结果来源与历史一致性核查

核查日期：2026-09-20。恢复队列于 15:36:08 UTC / 17:36:08 马德里时间完成，15/15 作业通过；两条训练各完成50轮，当前分析只包含 seed0，配对训练重复数 n=1，跨 seed 标准差 N/A。重新核验15份作业完成凭据和12份诊断结果哈希，通过；预期的2个续训分支均有效，无缺失分支。此前17:30的10/15是历史快照。

本次结果没有实现“仅在 Task0 使用 GAM，之后统一用 SGD，就提高两任务平均准确率”的预期。它与此前“后续任务也持续使用 GAM”的优势不构成同设置下的矛盾。代码执行符合当前续训计划，尚未发现造成该差距的配置串线、恢复失败或未完成训练问题。下面将直接证据与仍需验证的原因区分。

**1. 本次实际运行了什么。**

| 项目 | 实际设置 |
|---|---|
| 数据与任务 | ImageNet-A，类别顺序固定；Task0为前20类，Task1为随后20类 |
| 本次训练范围 | 恢复历史 Task0，仅训练 Task1；配置保留10任务定义，但 `max_train_tasks=2` |
| 两个分支 | 历史 SGD Task0→SGD+EWC；历史 GAM Task0→SGD+EWC |
| 模型 | ViT-B/16；Q/V LoRA rank=10，旧主干冻结，旧分类头20行保持不变 |
| Task1 数据 | 997张训练图；测试旧任务307张、新任务254张 |
| 训练预算 | 50 epochs，batch=128，每轮8 batches，共400次参数更新 |
| 优化器 | 两支均 SGD，momentum=0，weight decay=0；LR=.02，cosine最低值=.002 |
| EWC | λ=2000，γ=.9，η=0；各自恢复对应 Task0 的 Fisher，trace归一化 |
| GAM shaping | 两支 Task1 都关闭，`optimizer_type=sgd`、`lambda_flat=0` |
| 其他 | 无旧数据回放，无 in-task Fisher 更新，无权重平均、梯度投影或额外 ΔW penalty |
| 随机与精度 | seed0；配对 Task1 RNG 协议；full FP32，matmul/cuDNN TF32关闭 |

决定实际分支的代码是 [train_continuation.py:92](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py:92)：`method = 'sgd' if ns.mode == 'continuation' else ns.method`，随后写入 `optimizer_type` 和 `lambda_flat`。因此 `--method gam` 和模型名中的 `gam_intask` 在此次续训中不表示 Task1 仍使用 GAM。完整冻结代码和实际运行配置见 [AUDIT_EXECUTION.md](AUDIT_EXECUTION.md)。

Task1实际目标为：

\[
L=\mathrm{CE}(z_{20:40},y-20)+\frac{2000}{2}\sum_j\bar F^{(0)}_j\bigl(\Delta W_j-\Delta W^{(0)}_j\bigr)^2.
\]

这里 ΔW 是24个 Q/V有效 LoRA 权重矩阵，F是该分支恢复的保护矩阵。CE只比较当前任务20类，没有直接加入旧20类作为交叉熵负类。对应 [任务损失与EWC实现](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:263)。EWC返回值含1/2，日志文字标签省略该因子；本报告按实际代码解释。

训练增强为 RandomResizedCrop224（scale=.05–1）+RandomHorizontalFlip(.5)+ToTensor；测试和诊断为 Resize256+CenterCrop224+ToTensor，均不做 Normalize。最终汇总里“训练与测试都使用这一输入处理”应理解为诊断对 train/test 数据划分均用确定性处理，不能理解为训练优化也使用 CenterCrop。

两分支配置仅6项路径/命名/恢复来源不同；科学超参相同。新分类头哈希、51次训练loader遍历（50训练+1末端Fisher）的全部样本顺序哈希及每次首batch像素/标签哈希完全匹配。没有记录每个batch的像素哈希，配对结论限定在上述审计范围。199个冻结源文件哈希一致；两分支的350个恢复状态键精确核验，300个冻结状态和旧分类头在续训前后逐元素不变。

**2. 1.54个百分点具体来自哪里。**

下表两任务平均采用原始class-IL矩阵保留两位小数后的均值，与此前汇报一致。class-IL表示推理时在全部40个已见类别中选择，不提供任务编号。

| 指标 | 普通起点→SGD | GAM起点→SGD | GAM−普通 |
|---|---:|---:|---:|
| 旧任务，40类class-IL | 84.36% | 84.04% | −0.32 pp |
| 新任务，40类class-IL | 74.80% | 72.05% | −2.75 pp |
| 两任务等权平均 | 79.580% | 78.045% | −1.535 pp |
| 旧任务，仅旧20类 | 86.64495% | 86.31922% | −0.32573 pp |
| 新任务，仅新20类 | 88.19% | 87.40% | −0.79 pp |
| 全部561张图按样本平均 | 80.04% | 78.61% | −1.43 pp |

平均差距约89.6%来自新任务项。新任务局部head正确数为224/254与222/254；扩展到40类后为190/254与183/254。因加入旧类别而损失的正确数分别为34与39。因此7张图的总差距可以按计数分解为“局部分类少2张，加上跨任务类别竞争额外少5张”。这是正确数的分解，未保存逐样本预测，不能据此认定两分支错的是同一批图。

旧任务的局部head正确数为266/307与265/307，40类正确数为259/307与258/307，两分支都因加入新类别额外少7张。不能笼统说此次1.54 pp下降是旧知识全部被破坏，或把所有差距归于旧任务的跨类竞争。

从代码看，训练只优化新20类内部CE，而报告主指标要求40类联合竞争；两分支的旧head、有效LoRA起点和Fisher本来就不同。因此局部分类表现接近，但联合预测差距扩大，是与训练/评价定义相容的现象。它定位了性能差距，并未证明究竟是logit尺度、表征变化、更新方向或Fisher结构造成差异；这些仍需要进一步对照。

**3. 低Fisher trace为什么没有保证高准确率。**

| 旧任务诊断量 | 普通起点→SGD | GAM起点→SGD |
|---|---:|---:|
| Task0起点 raw empirical-Fisher trace | 1465.49693 | 761.90077 |
| Task1后同口径 raw trace | 2008.38787 | 940.41253 |
| Task0 test local CE，起点→末端 | .442424→.423178 | .413621→.400233 |
| Task0 test local CE变化D0 | −.019246 | −.013388 |
| 有效权重L2位移 | .374950 | .349876 |
| 同环境旧任务seen准确率遗忘 | 1.302932 pp | 3.583062 pp |

GAM起点和末端的旧任务CE与经验Fisher迹均较低；两个分支的旧任务test local CE都改善。与此同时，GAM的40类平均准确率更低。这些事实可以同时成立：经验Fisher迹基于固定256张旧训练图、旧20类head的逐样本梯度平方；准确率依赖测试集argmax，主结果还涉及新任务和40类竞争。经验Fisher迹也不能直接等同于Hessian曲率。

训练保护矩阵与上述诊断raw trace不是同一个量。继承的F是batch_empirical估计并经过trace归一化，两支总trace都约1，但逐坐标内容不同；不能推断GAM raw trace减半，所以本次EWC强度减半。当前对照改变的是“Task0模型/旧head/对应Fisher”这一组继承状态，尚未隔离纯几何效应。

日志中GAM分支平均新任务CE更高（约.60433 vs .56100），平均Fisher损失反而更低（约9.17e−5 vs 2.16e−4）。这些数据不支持“GAM用了更大λ或更大的已观察EWC损失，所以学不动”的说法；SGD分支没有完整梯度分解，损失小也不等于EWC梯度没有影响。

固定旧head的CE损伤、0–1准确率遗忘、权重位移和trace应分别报告。尤其本次SGD起点当前FP32复评为85.667752%，而历史继承矩阵写85.34%；计算当前遗忘必须用前者。训练日志的 `delta_norm` 实际为L2平方（约.140587/.122413），不应与上表L2位移混用。

**4. 和此前结果是否一致。**

| 实验来源与实际分支 | Task1后旧任务class-IL | 新任务class-IL | 两任务平均 |
|---|---:|---:|---:|
| 历史 SGD→SGD | 82.08% | 75.20% | 78.640% |
| 历史 GAM→GAM | 81.76% | 81.50% | 81.630% |
| 本次 SGD→SGD | 84.36% | 74.80% | 79.580% |
| 本次 GAM→SGD | 84.04% | 72.05% | 78.045% |

历史GAM优势为+2.99 pp，本次为−1.535 pp；历史GAM Task1继续启用shaping，本次关闭，所以这不是同一个实验的复现。历史GAM旧任务class-IL本来就比SGD低约.32 pp，其两任务优势主要来自新任务高6.30 pp。

两份Task0 checkpoint和两份Fisher sidecar经重算SHA确认同源。但新旧Task1随机协议不同，旧日志没有本次配对RNG收据；新训练明确关闭TF32，旧运行缺少同等精度开关记录。旧SGD→SGD和新SGD→SGD本身也相差+.94 pp，故不能把历史GAM→GAM与本次GAM→SGD的全部差值都当作“去掉GAM”的因果效应。原始来源、参数逐项比较见 [HISTORICAL_COMPARISON.md](HISTORICAL_COMPARISON.md)。

本次完成的B/C也要求收窄结论：B中GAM在全部测量半径的旧训练子集raw trace更低，六个非零半径的train正损伤G与Q95更低；但test六个非零半径的平均损伤与G均略高，test Q95在5/6半径较低，容忍半径并未扩大。不能宣称“所有方向、所有数据划分均更平、更稳”。

C重新评估的是历史十任务完整训练链。GAM→GAM链Task9旧任务test local CE增量.03435，SGD链.26013；端点直线路径S_path分别.014793与.368058。两条路径求积及数值界检查都通过，保留了历史GAM链方向敏感度较低的证据。C及bridge使用旧Task9−Task0方向，并非新A的Task1更新方向，也不是逐步训练路径，不能直接用来解释这次A的新任务准确率差。完整结果与反例见 [MEASUREMENT_ANALYSIS.md](MEASUREMENT_ANALYSIS.md)。

**5. 当前可支持的结论与下一步。**

1. seed0下，仅继承GAM Task0状态而统一进行SGD+EWC续训，未改善两任务class-IL平均准确率；当前数据不能支持该收益预期。
2. 旧任务低经验Fisher迹、低CE，以及旧全程GAM链较低路径敏感度仍有证据；这些证据不能升级为“只塑造初始点就保证后续分类更好”。
3. 若继续验证，先在本次相同FP32/RNG协议下补GAM→GAM，避免直接拿历史数字做因果消融；随后做预定多seed成对重复。若要隔离起点几何与保护矩阵，再设计共享/交叉Fisher对照；若要定位联合分类差距，保存逐样本40类logits并分析跨任务混淆。上述为建议，本次分析未启动额外训练。

本次分析新增文件：[有效性台账](VALIDITY_LEDGER.json)、[执行审计](AUDIT_EXECUTION.md)、[历史比较](HISTORICAL_COMPARISON.md)、[A/B/C测量分析](MEASUREMENT_ANALYSIS.md)。实验配置、原始checkpoint和原始结果保持原样。
