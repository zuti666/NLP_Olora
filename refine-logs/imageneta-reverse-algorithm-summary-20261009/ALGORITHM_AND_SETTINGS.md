# ImageNet-A反向探针实验：设置、核心操作与SGD/SAM/GAM的区别

本说明依据已冻结代码核对。最重要的区别是三个独立选择：探针位置在哪一侧、测得的差分取哪个符号、最终把差分接到哪个基点梯度。所有实验最终仍在原位置执行一次下降更新；“下降侧探针”不是让最终优化器执行梯度上升。

统一记号：g(theta)=当前批次CE损失的普通梯度，theta包含可训练LoRA A/B和已见分类头参数；N(v)=v/(||v||2+1e-12)。归一化覆盖这些参数的联合向量。所有公式是代数表示，FP32实际训练保留代码原有括号顺序。

共同科学设置为ImageNet-A train5981/test1519、200类固定顺序0…199、10×20任务、seed2、ViT-B/16 augreg2_in21k_ft_in1k预训练、冻结骨干、Q/V LoRA rank10、已见FC可训练、无alpha/rank缩放。每任务50epochs、batch128保留尾批；SGD momentum=0、weight_decay=0，首任务LR=.04、以后.02，逐任务cosine最低为起点的.1倍。

两个半径必须区分：rho=gam_grad_rho=.05用于局部正/负探针；r=gam_grad_norm_rho=.25或1用于外点定位。LoRA rank固定10，和外半径r不是同一个量。SAM局部组没有外点，配置中的外半径0只是未使用字段，不是将GAM外点消融后就自动等于本组SAM。

SAM局部和最新GAM末端分量组关闭EWC/Fisher及其计算、权重融合、最终梯度投影/范数匹配、SGV、HVP、replay。rho/r构造扰动所需的N仍保留。较早P/D/C/R的A组额外研究过有效步长匹配，应和自然步长分开。s/e分解是差分的奇偶响应解释，不是原GAM正交投影。

所有局部读取使用同一批增强后输入与标签；新方向执行器每次从theta0快照/R0重建探针，最后只提交g0读数后的R1，再从theta0执行theta0-eta*U。原GAM采用加减扰动恢复，原生历史wrapper还可能有额外clean读取和全局RNG推进；同数学方向不代表相同计算成本或逐位轨迹。

SAM局部反向与中心实验先读取：

\[
g_0=g(\theta_0),\quad
g_1^+=g(\theta_0+\rho N(g_0)),\quad
g_1^-=g(\theta_0-\rho N(g_0)).
\]

\[
s_1=(g_1^+-g_1^-)/2,\qquad e_1=(g_1^++g_1^--2g_0)/2.
\]

| 设置 | 最终U的代码表达式 | 代数解释 |
|---|---|---|
| SGD方向零修正 | g0 | g0 |
| 正向SAM方向 | g1+ | g0+s1+e1 |
| 直接下降侧梯度 | g1- | g0-s1+e1 |
| 下降侧镜像 | g0+(g0-g1-) | g0+s1-e1 |
| 局部中心 | g0+(g1+-g1-)/2 | g0+s1 |
| 基点加偶响应 | (g1++g1-)/2 | g0+e1 |
| 基点减偶响应 | g0+[g0-(g1++g1-)/2] | g0-e1 |

共同七组每步都计算g0/g1+/g1-三次梯度，最后只做一次SGD，以统一探测预算。普通SGD算法只需一次梯度；标准实用SAM只需g0/g1+两次；下降侧镜像的最小实现只需g0/g1-两次。当前七组的g0行是SGD更新方向对照，不是一次梯度的运行成本基线。

SAM的原目标为邻域min-max；实用算法近似定位内层上升侧点，再读取扰动点梯度，忽略扰动关于参数的链式导数。[原SAM Eq.1–3](https://arxiv.org/html/2010.01412v3)。直接g1-不再是上升侧读数；下降镜像又进一步改变了最终方向，因此它是SAM式局部响应变体，不能宣布与SAM优化同一个精确目标。

在固定当前u=N(g0)、足够局部光滑条件下，正向、下降镜像和中心保留相同领先项g0+rho*H0*u，但偶响应系数分别为+1/-1/0；直接g1-反转领先响应。中心仍含高阶奇响应，不能称为只剩一个精确Hessian项。实际代码没有计算Hessian或HVP。

较早第一段外层定位实验把局部梯度变化转成外点方向q：

\[
\theta_2=\theta_0+rN(q),\qquad g_2=g(\theta_2).
\]

| 分支 | 第一探针侧 | q | 核心干预 |
|---|---|---|---|
| P/后续F | 正侧 | g1+-g0 | 原顺序正向响应 |
| D | 负侧 | g1--g0 | 只换探针侧，不校正差分取向 |
| C/后续B | 负侧 | g0-g1- | 负侧读取并反转差分 |
| R | 正侧 | g0-g1+ | 保留正侧读取，反转外层方向 |
| CD | 两侧 | (g1+-g1-)/2 | 中心外层估计 |

外点从theta0构造，不能写成theta1+rN(q)。C/B仍沿正的qB=g0-g1-移动；它不是简单把整个GAM外层移动乘-1。

这批还独立比较不同最终接法：O=g2；A=g0+(g2-g1对应侧)；后来sum版本为g1对应侧+g2，不除以2。C/D的对应侧是g1-，P/R的对应侧是g1+。这些是外点梯度或跨位置残差，和末端g3-g2不同。P/D/C/R的O/A覆盖r=.25/1；CD外层中心组固定r=.25且最终更新g2，不能把它的成绩当作SAM局部中心g0+s1。

P/D/C/R旧比较每步读取g0/g1+/g1-/g2四次，无g3。A的matched条件用候选自身当前状态的正向g1+名义LR真实BA+FC位移作为目标，以正标量调整实际步长；不是更换差分公式，也不是SAM/GAM本身固有的处理。

第二段反向实验固定第一段P，然后在外点读数：

\[
g_3^{\pm}=g(\theta_2\pm\rho N(g_2)),\qquad
U=b+\tau(g_3^{\sigma}-g_2),
\quad b\in\{g_0,g_1^+\},\ \sigma,\tau\in\{+1,-1\}.
\]

这里theta3是theta2±rho*N(g2)，不是theta2±rho*g2。theta2和g2在这些第二段符号比较中保持构造规则不变；负侧g3-不意味着g2也变成了某个新的g2-。

| 第二探针符号sigma | 差分系数tau | 加入b的残差 | 对应组 |
|---|---|---|---|
| +1 | +1 | g3+-g2 | B0/B1 |
| +1 | -1 | g2-g3+ | N1/N2 |
| -1 | +1 | g3--g2 | N3/N4 |
| -1 | -1 | g2-g3- | N5/N6 |

每行分别用b=g0和b=g1+，得到8个方向；各跑r=.25/1，共16条完整训练。表中的g1+减去残差是“正侧梯度作为基点再减残差”，不是负侧读数g1-。第二探针侧与残差符号是两个独立变量，负侧读取后使用g2-g3-才是修正取向的后向响应。

联合替换再允许第一段使用B/C构造。令a=F/B，theta2a=theta0+r*N(qa)，g2a=g(theta2a)，g3a±=g(theta2a±rho*N(g2a))：

| 条件 | 第一段q | 末端残差 | 最终U |
|---|---|---|---|
| FF | g1+-g0 | g3F+-g2F | g0+残差 |
| FB | g1+-g0 | g2F-g3F- | g0+残差 |
| BF | g0-g1- | g3B+-g2B | g0+残差 |
| BB | g0-g1- | g2B-g3B- | g0+残差 |
| Center+ | g1+-g0 | (g3F+-g3F-)/2 | g0+残差 |
| Center- | g1+-g0 | -(g3F+-g3F-)/2 | g0+残差 |

BF/BB/Center±新增组固定r=1；FF/FB复用同严格协议r1的已验收记录。四个F/B条件每步4次梯度；Center两侧都读，每步5次。中心只除以2，未除以rho，也没有额外乘r/rho。若换成/(2rho)，rho=.05会使当前修正扩大20倍，改变了强度。

BB的完整顺序是g0→负侧g1-→qB=g0-g1-→theta2B=theta0+rN(qB)→g2B→负侧g3B-→U=g0+(g2B-g3B-)→恢复theta0并执行一次SGD。两处局部上升侧读取被替代，但梯度范数引导的外层位置保留。

原版GAM在本项目beta全1、非adaptive时采用：

\[
g_0\to g_1^+\to q=g_1^+-g_0\to
g_2=g(\theta_0+rN(q))\to
g_3^+=g(\theta_2+\rho N(g_2)).
\]

\[
h_-=g_0+g_2,\quad h_+=g_1^++g_3^+,\quad
U_{GAM}=h_+-\gamma_{GAM}\left(h_--\operatorname{proj}_{h_+}h_-\right).
\]

投影公式为忽略数值保护epsilon的代数写法；代码使用全局内积/范数。历史配置gamma_GAM=.1，gamma=0且SGV关闭时最终方向就是g1++g3+，不除以2。

| 算法/变体 | 使用的信息 | 最终U | 必要训练梯度读数 |
|---|---|---|---|
| 普通SGD | 当前点 | g0 | 1 |
| 标准实用SAM | 当前点+上升侧局部点 | g1+ | 2 |
| 下降侧镜像局部变体 | 当前点+下降侧局部点 | g0+(g0-g1-) | 2 |
| 局部中心变体 | 当前点+局部两侧 | g0+(g1+-g1-)/2 | 3 |
| 原版GAM方向构造 | 两段局部探针及外点 | h+−gamma*hminus_perp | 4 |
| 当前GAM分量FF/FB/BF/BB | 同类外点及末端差分 | g0+定向残差 | 4 |
| 外点中心分量 | 同类外点及末端两侧 | g0±(g3+-g3-)/2 | 5 |

表是方向构造的最小/必要读数，不包含wrapper的额外clean梯度及诊断读数。SAM局部七组主动统一为3次；旧原生g1-only wrapper还执行额外clean+完整四探针，不能叫作两读取SAM成本基线。

因此当前分量组既不是SGD原方向，也不是完整GAM。它们仍用SGD作最终提交器，只替换p.grad中的U；FF/BB没有完整的g1+g3聚合，也不含原GAM投影修正。单独保留g1时，选中的方向是SAM式方向；额外探针、恢复/RNG及外部模块仍须核对才可判断实现与轨迹是否一致。

实现证据：

- [局部七方向选择](/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z/trainer/odd_even_step.py:120)：镜像保持g0+(g0-gminus)的括号顺序，中心为g0+.5*(gplus-gminus)。
- [原版GAM探针与最终组合](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py:239)，[gradient_decompose](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py:122)。
- [P/D/C/R定位](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/direction_step.py:153)。
- [联合反向和中心实现](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/trainer/direction_step.py:220)，[实际SGD提交](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/trainer/direction_step.py:366)。
- [27条分协议结果与缺口](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-sam-gam-sign-baseline-summary-20261008T174804Z/RESULT_ANALYSIS.md)：SAM已有同math协议g0，末端strict-native仍缺U0=g0；不能用旧native非strict或math结果补齐。局部g0-s1未在七组里执行，直接g1-还带+e1。完整原版GAM两阶段后向且保留gamma投影的替代版本也没有由BB证明。

已观察的代表结果：SAM正向/下降镜像/局部中心AAA/FAA分别63.102/44.809、62.615/43.714、62.994/44.327，同局部g0为60.232/39.260；直接g1-为50.801/25.235。末端r1 FF/BB为65.176/47.584与65.468/48.935；Center±为65.197/48.170与2.064/.651。这些来自不同明确执行协议，应在各自协议内解释，单seed自然步长不能证明算法完全等价或单独Hessian因果机制。
