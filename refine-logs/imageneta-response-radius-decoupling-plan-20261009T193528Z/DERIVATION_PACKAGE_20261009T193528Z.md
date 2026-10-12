# 学习率与响应探测半径的推导

## Target
核对已完成三种非 SGD 响应方法的半径—学习率耦合，判断独立半径的新对照如何定义。原 Current-G2、History Fusion、History-G2 的公式、配置和结果保留原定义。本文件不修改算法实现，不注册或启动训练。

## Status
COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

用户对直接 fusion 修正的局部判断成立，但“镜像”应明确为 2g0−g1；当前最佳 Current-G2 的最终梯度还经过响应方向的第二次探测，其局部表达涉及 H²g0。晚期数值噪声是待检验假设，不能从学习率衰减或准确率下降直接确认。

## Invariant Object
固定当前任务、已实现 minibatch/增强、R0 随机状态和 LoRA+当前分类头坐标的梯度场 g(theta)=grad L(theta)。比较对象分为探针位置、提交向量 d 和最终真实更新 −eta*d，三者分别记录。

## Assumptions
- 局部推导固定 theta、minibatch 和随机状态。训练轨迹上的不同 eta 会访问不同 theta 和不同 H；H 在一个固定点的定义不含 eta，不等于不同学习率训练拥有同一 H。
- 近似采用二阶连续可微损失，H 局部 Lipschitz，探针足够小；g0 与 Hg0 非零，归一化分母远离退化区。
- 写 N(v)=v/||v|| 表示非退化理想归一化。实际实现须保留 eps/零方向/FP32 不可分辨回退和真实位移审计，不把 rho 当作无条件精确实际距离。
- 近似是当前 minibatch 训练坐标中的一阶梯度 Taylor 分析，不是全模型/Fisher 几何结论或持续学习的收益证明。

## Notation
eta：本次真实 SGD 学习率；g0：基点梯度；H：基点 Hessian；g1：下降预测点的梯度；a=g0−g1：当前响应。
h=eta||g0||：原法的理论下降探针距离及 G2 外探针距离。
rho：新变体中独立内半径；R：独立外半径，二者不是 LoRA rank，也不是原 GAM 参考半径字段。
p：上一实际更新缓存的完整向量（HF 为 base g0，History-G2 为 final d）；ell：上一更新的实际 FP32 参数位移范数；w=.5。

## Derivation Strategy
代码中的精确位置/向量定义 → 当前 batch 的局部响应近似 → 最终更新量标度 → 三处半径/历史缩放耦合 → 最小 2×2 新变体对照。

## Derivation Map
1. 固定 eta 时下降探针为 theta−eta*g0，原探测距离 h 随 eta 和当前梯度共同变化。
2. 直接 fusion 提交 g0+a，近似修正为 eta*H*g0；真实参数修正再乘 −eta。
3. Current-G2 把 a 单位化后在基点外移 h，H 再作用一次；内探针的标量 eta 在理想单位化方向中消去。
4. 独立 rho 主要影响有限差分的非线性和数值质量；独立 R 直接控制外探针的额外影响幅度。
5. 有历史时 ell 和跨 batch 差分不能替换成同 batch 的 Hessian 响应，且 h/ell 可能抵消学习率因子。

## Main Derivation

### 1. 原法的精确耦合（identity / code evidence）
\[
h=\eta\|g_0\|,\quad \theta_1=\theta-\eta g_0,\quad g_1=\nabla L(\theta_1),\quad a=g_0-g_1.
\]
有有效历史时：
\[
c=(1-w)a+w\frac{h}{\ell}(p-g_0).
\]
无历史时 c=a。History Fusion 提交 d=g0+c；Current-G2 与 History-G2 在 BASE 而不是预测点计算
\[
\theta_2=\theta+hN(c),\quad g_2=\nabla L(\theta_2),\quad d=g_2.
\]
History Fusion 没有这个外层 g2 探针。所有方法真实 SGD 更新仍为 theta_new=theta−eta*d。
代码来源是已冻结 response_step.py:162/194/210/236/249。第83–84行严格拒绝独立半径模式，原 GAM gam_grad_rho/gam_grad_norm_rho 仅为参考元数据。

### 2. 无历史直接 fusion / 镜像（approximation）
\[
g_1\approx g_0-\eta Hg_0,\quad a\approx\eta Hg_0,
\]
\[
d_{\rm fusion}=2g_0-g_1\approx g_0+\eta Hg_0,
\quad \Delta\theta\approx-\eta g_0-\eta^2Hg_0.
\]
用户的 −eta²Hg0 指的是参数修正，不是梯度响应 a。独立归一化下降探针 theta1=theta−rho*N(g0) 下：
\[
a_\rho\approx\rho\frac{Hg_0}{\|g_0\|},\qquad
\Delta\theta_{\rm mirror}\approx-\eta g_0-\eta\rho\frac{Hg_0}{\|g_0\|}.
\]
该近似描述无历史直接融合方向 2g0−g1，不可直接套在 g2-only 最终方向上。

### 3. Current-G2（approximation）
\[
N(a)\approx\frac{Hg_0}{\|Hg_0\|},\quad
 g_2\approx g_0+h\frac{H^2g_0}{\|Hg_0\|},
\]
\[
\Delta\theta_{\rm CurrentG2}\approx-\eta g_0
-\eta^2\|g_0\|\frac{H^2g_0}{\|Hg_0\|}.
\]
独立内 rho、外 R 下：
\[
a_\rho\approx\rho\frac{Hg_0}{\|g_0\|},\quad
 g_2^{\rho,R}\approx g_0+R\frac{H^2g_0}{\|Hg_0\|},
\]
\[
\Delta\theta^{\rho,R}\approx-\eta g_0
-\eta R\frac{H^2g_0}{\|Hg_0\|}.
\]
忽略稳定器和非退化条件下，外响应误差为 O(R*rho+R²)，最终更新误差还乘 eta。内 rho 在一阶单位方向里消去；固定内半径仍不能消除外 h=eta||g0|| 导致的 eta² 标度。固定外 R 才把该额外参数修正的领先标度从 eta² 改为 eta。

### 4. 历史项不能只按 eta² 描述（identity / interpretation）
原历史系数为
\[
\beta_{\rm history}=w\frac{\eta_t\|g_{0,t}\|}{\ell_{t-1}},\quad
\ell_{t-1}=\|\theta_t-\theta_{t-1}\|_{\rm actual\ FP32}.
\]
理想 SGD 下 ell 约为 eta_prev||d_prev||；平滑 cosine 里 eta_t/eta_prev 接近1，因此当前响应变小不保证历史分量一起变小。真实代码始终用实际位移，不用理论 eta_prev*d_prev 重建缓存。

若以后定义 history-fixed-radius 新变体，把分子直接换成固定 rho 或实际独立探针距离 q，会得到 w*q/ell。ell 随真实 SGD 缩小时该系数可能增长；直接 fusion 的历史更新 −eta*w*q/ell*(p−g0) 在跨 batch 差分不随 eta 消失时甚至不保证按 eta 衰减。G2-only 的归一化能去掉响应幅度，但不能去掉被历史噪声主导的方向。这也是先在无历史 Current-G2 上拆分半径因素的理由。

后续若研究历史，需要预注册“几何半径”和“融合幅度”各自控制规则。保留原 h/ell 而只替换探针几何，可以检验半径因素，但不应宣称全部学习率依赖已消除；替换为 q/ell 则同时改变历史缩放。两者不能混为同一个单因素消融。本轮计划不改这两个历史方法。

## Remarks and Interpretation
14,700次已完成非SGD更新日志验证 h=eta||g0||，未发现 fallback、第一探针FP32不可分辨或零当前响应；G2外探针也没有不可分辨事件。实际距离约为理论h的1±1.5e−6。这不能证明所有差分都数值准确，只排除了已记录的这些明显退化。

Current-G2 CL10按任务内epoch汇总中位数：epoch1/25/50 h=0.103295/0.021774/0.003499；||a||=0.663856/0.159928/0.026098。单任务||a||为0.209128/0.372649/0.086768，说明训练轨迹上响应不必单调随学习率降低。

CL10最后10epochs的加权历史/当前分量范数比中位数：HF71.50，History-G2 32.32；这支持后期响应配比变化的诊断，不证明它造成性能下降。相邻历史batch经常不重叠，历史项包含跨batch变化。

## Boundaries and Non-Claims
- 原三方法和已有结果保持原定义；新固定半径是独立命名的额外对照。
- 固定半径保留探测尺度，但不保证性能提升、数值可靠、正定H或稳定下降。
- g0很小时，固定rho产生的响应相对g0可过大；Hg0接近0会使N(a)敏感。不可用局部公式保证训练稳定。
- 不在同一个位置/同一个batch比较的曲线变化，不能作为 H 不变的证据。
- 有历史的HRF/HG2不能直接套用无历史fusion/Current-G2的领先修正式。
- g1为下降预测梯度，不是之前正向SAM/GAM的同名探针；训练探针未执行HVP或Hessian求逆。

## Open Risks
独立rho/R=.05只是日志量级内的预注册探索中心，不是已确认最优值。零/近零方向、固定半径非局部效应、相对更新幅度、跨batch历史噪声和单seed方差均需报告。最小 2×2 比较见 EXPERIMENT_PLAN.md；当前没有启动新训练。
