# ImageNet-A第二内层探针符号消融计划

日期：2026-10-06，欧洲/马德里时间。当前状态：用户已授权执行；新四探针执行器已实现，CPU验收11/11、队列检查11/11、独立科学及逐字文档验收通过，29项队列清单已注册；真实主机controller PID920989已核验存活，状态waiting_for_idle_gpu、29pending、0短测/正式启动。现有训练队列与结果保持原样。原始计划和配置保存在plan_archive/。

## 研究问题和计数

固定原P分支的g0→g1→g2，只干预第二局部探针的上升/下降，以及其梯度差分的正负号。问题是：有效性依赖第二探针“上升采样”本身，还是依赖最终有限差分修正的方向？补充检查同一残差加到g0或g1是否改变作用。

完整为最终基点g0/g1×第二探针+/−×残差符号+/−的8格。用户最初提到的两个正向反差分，加后来的四个下降侧条件，共6种新方向；原来两个正向残差作基线，并非遗漏最初两组。

沿用已有r=1与r=.25两个固定切片，不扫描新半径。每切片6种新方向+1个g0正向残差严格基线补跑，复用1个g1正向残差完整结果（须同状态/执行语义一致性门通过）。总计12条新方向正式训练＋2条基线补跑＝14条新正式训练，2条旧结果复用。r=1优先，r=.25随后。若只先完成r=1，需7条新正式训练，不自动扩展seed/rank/半径。

## 核对完整GAM及本轮关注的分量

原论文目标包含预测损失与邻域最大梯度范数正则；g3−g2是正则梯度的有限差分代理，不是一个标量优化目标。附录D式41给出∇R^(1)_r≈(r/ρ)(g3−g2)，算法2给出四梯度及最终分解。[GAM原论文附录D](https://arxiv.org/html/2303.03108v3#A4)。

当前项目原生实现的固定β1=β2=β3=1时，h+=g1+g3、h−=g0+g2，U=h+−γ(h−−proj_h+(h−))。代码 [gam.py:122](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/gam.py:122)，完整step [gam.py:239](/data/140-1/users/liying/CopyFlatCV/optimer_PerturabtionType/gam.py:239)。本轮不运行这段分解，以直接方向分支替代最终U，γ=0、投影调用0、无EWC/Fisher/融合。

本轮差分系数固定±1，不除以ρ、不乘r/ρ、不单位化最终U；保持与已有g0/g1+d32的方向定义相同。不声称这六个方向完全等于原始GAM目标的梯度。

## 固定前两段

令g(θ)=∇L(θ)，N(v)=v/(||v||+1e−12)，norm覆盖所有可训练LoRA A/B和seen FC。

$$g_0=g(\theta_0),\quad\theta_1=\theta_0+\rho N(g_0),\quad g_1=g(\theta_1),$$
$$q=g_1-g_0,\quad\theta_2=\theta_0+rN(q),\quad g_2=g(\theta_2).$$

此处第一探针仍是正向，q的符号不变，外点从原始θ0构造。只改变后面：

$$\theta_3^\pm=\theta_2\pm\rho N(g_2),\quad g_3^\pm=g(\theta_3^\pm),$$
$$c_+=g_3^+-g_2,\qquad c_-=g_3^--g_2.$$

g2没有更换定义，不写g2−。从θ2继续扰动，不能从θ0/θ1生成第三点。不是先真实做一个下降更新；测完需精确恢复θ0，再提交一次θ←θ0−ηU。

## 八格及六种新增方向

| ID | 第二探针 | 最终U | 角色 |
|---|---|---|---|
| B0 | +ρN(g2) | g0+(g3+−g2) | 旧结果仅历史参考；补同协议基线 |
| B1 | +ρN(g2) | g1+(g3+−g2) | 已有完整两半径结果，可核验后复用 |
| N1 | +ρN(g2) | g0+(g2−g3+) | 新：仅反转正向残差 |
| N2 | +ρN(g2) | g1+(g2−g3+) | 新：仅反转正向残差 |
| N3 | −ρN(g2) | g0+(g3−−g2) | 新：下降采样，保留原残差顺序 |
| N4 | −ρN(g2) | g1+(g3−−g2) | 新：下降采样，保留原残差顺序 |
| N5 | −ρN(g2) | g0+(g2−g3−) | 新：下降采样，校正残差符号 |
| N6 | −ρN(g2) | g1+(g2−g3−) | 新：下降采样，校正残差符号 |

## 固定训练设置

ImageNet-A本地train5981/test1519、200类固定顺序0…199、10×20类；ViT-B/16 augreg2_in21k_ft_in1k，LoRA rank10；seed2；ρ=.05；r在对应切片固定为1或.25；50epochs/task，共500epochs/2550更新。batch128，SGD momentum/weight decay0，Task0 LR=.04、后续=.02，cosine最低LR比例.1。每组独立相同初始化；确定性epoch数据采样/增强；FP32、native attention、TF32off，AMPoff。

lambda_flat=1、ewc_lambda=0、ewc_skip_task_fisher=true；Fisher近似、归一化、阻尼、权重融合、SGV、replay、HVP、投影、最终范数匹配均关闭。β/γ保留字段不决定直接方向分支，实际方向由update_base/second_probe_sign/second_difference_sign统一构造并验收。

每训练步统一4次CE梯度：g0,g1,g2,g3选定侧。各点重放同一R0、恢复参数/梯度/buffers/模式/缓存；最后RNG仅推进至g0后的R1。不为负探针组额外改变数据增强或额外clean CE。固定快照诊断可以额外测两侧，但放在隔离诊断计数中并严格恢复全部状态，不插入训练梯度。

## 已有结果及复用边界

以下AAA/FAA均任务等权%；历史g0与新g1不作为同执行器配对排名。

| 设置 | r=.25 AAA/FAA | r=1 AAA/FAA | 来源与用途 |
|---|---:|---:|---|
| g1+(g3+−g2)，自然 | 65.297/48.660 | 65.549/48.089 | 新4CE、精确恢复/R0，B1候选复用 |
| g0+(g3+−g2)，自然 | 65.230/48.663 | 64.102/47.870 | 旧5CE、减扰动恢复/不同RNG，仅历史参考 |
| g0+(g3+−g2)，匹配 | 65.142/48.565 | 未找到对应完整结果 | 不并入自然步长对照 |

有限范围内扫描O-LoRA/CopyFlatCV直属campaign/configs的972份配置，没有发现N1–N6完整结果。审计来源保存在evidence/history_audit.json。

## 主要对比和可证伪解释

局部H2=∇²L(θ2)时，c+≈ρH2N(g2)、c−≈−ρH2N(g2)。这一近似不保证有限ρ下实际方向或幅值接近，不能假定结果应相同。

| 对比 | 回答的问题 | 若结果接近/分离的解释 |
|---|---|---|
| B0↔N5，B1↔N6 | 正向差分vs下降侧符号校正差分 | 接近：第二上升探针不是唯一构造；分离：有限差分侧别/幅度/轨迹仍可能重要 |
| N1↔N3，N2↔N4 | 反向修正的两种估计 | 接近：相反修正方向有一致效果；分离：两侧非线性或幅度不能忽略 |
| B0↔N1，B1↔N2 | 固定正向探针，只翻最终残差符号 | 检验最终修正符号的作用 |
| N3↔N5，N4↔N6 | 固定下降探针，只翻最终残差符号 | 检验下降测量后是否需要符号校正 |
| 同探针/残差符号下g0↔g1 | 更新基点作用 | 单seed自然步长同时含方向与幅度，不能仅称几何机制 |

不能用本轮说明“第一上升探针不必要”：g0→g1→g2原样保留。不能把一阶接近写成性能等价，也不预设哪组赢。固定快照记录余弦/范数与真实更新幅度；只有完成训练才报告AAA/FAA。

## 指标与结果呈现

主指标AAA；同时FAA、LA、BWT、forgetting、逐任务矩阵和sample-weighted终点（分列，不混口径）。自然步长单seed不填SD=0，写N/A。正文表按第二探针侧×残差符号四行，g0/g1两列；每个r独立一张。

每步记录loss0/1/2/3、delta_loss3、g0/g1/g2/g3/c/U范数、cos(c,g0/g1/g2)、残差/基点范数比、实际LoRA BA+FC联合位移、参数空间第三探针长度与符号、forward/gradient/SGD/Fisher/merge/projection/HVP计数。每任务隔离固定快照至少测一批：cos(c+,−c−)、||c+||/||c−||、两侧损失差、两个符号校正候选有效步长。隔离诊断不更新模型、不消费训练RNG。

## 执行顺序与预算

M0：新执行器实现（已完成）；CPU解析/数学/完整状态恢复检查（11/11通过）；在同状态下验证8个U公式，以及所有组g0/g1/g2逐位一致。正式入口必须拒绝未读取新符号字段的配置。

M1：每个新方向及B0基线先2tasks×1epoch×1batch短测，每步4CE/1SGD；检查无Fisher、实际融合、投影、HVP；g3距离为.05且方向符号符合。B1做同状态旧/新代码等价门；失败则停止复用，修复或明确补跑，不静默忽略。

M2：r=1优先，先N5/N6（下降符号校正）、B0严格基线，再N1/N2、N3/N4；均50epochs/task。随后相同7项r=.25。每卡仅一个作业；原队列不抢占、不修改；沿用共享index/UUID锁和双次空闲检查。执行默认排除GPU0/1，允许其余实际可见空闲GPU；计划阶段不分配设备。

M3：14个新正式结果完整验收（10tasks/500epochs/2550updates/10200训练CE、原矩阵重算、receipt哈希、所有观察有限）后合并2个B1复用结果。任何失败仅标失败，不用partial终点填表。若需要确认方向而不是总幅度，下一轮才增加同定义有效步长匹配或多seed，不自动扩展。

成本暂按历史约2.2 GPU小时/完整4CE训练估算，14项约30.8 GPU小时；这是估算，不是当前可用GPU上的承诺。允许GPU2–6，max_parallel=5；按每张卡实际空闲调度。GPU内核witness、真实模型短测及完整多任务生命周期尚未执行，本计划不称环境/训练ready。

## 配置与实现位置

16个正式科学条件的配置已注册，另有14份短测配置；新变量为second_probe_sign=±1、second_difference_sign=±1、update_base=g0/g1。前三个梯度步骤必须在执行器内固定正向P几何，禁止通过negative rho误改第一探针。原g1+d32代码可复制到新隔离执行目录再扩展：[direction_step.py:132](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/trainer/direction_step.py:132)。最终构造：

```python
q = g1 - g0
point2 = base + r * normalize(q)
g2 = grad_at(point2)
point3 = point2 + second_probe_sign * rho * normalize(g2)
g3 = grad_at(point3)
c = g3 - g2
anchor = g0 if update_base == "g0" else g1
U = anchor + second_difference_sign * c
# restore original base and submit exactly one natural SGD update
```

新执行器已实现上述配置语义，旧runner没有修改。11项CPU验收覆盖八格、共同g012、两半径旧B1逐位等价、完整状态/RNG恢复、4CE/1SGD及隔离诊断。新的suite.json注册14短测+1CPU配对+14正式，所有短测与配对检查通过后正式自动启动；真实GPU短测仍须等空闲卡，不能将CPU通过称为正式训练通过。

## 2026-10-06 执行扩展与队列

代码位置：trainer/direction_step.py（两个独立符号及g0/g1基点）、trainer/four_probe_train.py（严格字段解析与真实生命周期）、tests/test_second_probe_cpu.py（CPU11项验收）、run_queue.py（空闲GPU后台调度）、validate_job.py（逐步/逐任务验收）。ENVIRONMENT.md给出可逐字复核与启动命令。CPU_GATE.json已通过，两项B1历史receipt与原validator将在复用/汇总时再次核验。

本次真实主机仅可见GPU0–6。排除0/1，GPU2–6共享index/UUID锁、双次idle检查、无compute PID、最多5并行、60秒周期检查；不会停止/抢占其他任务。首次查询全部忙碌，GPU模型短测及正式训练待空闲。新suite共29jobs：14短测(2tasks/1epoch/1batch)、1个不占GPU的配对gate、14完整正式训练。先r=1的N5/N6，再同协议B0与其余方向；r=.25随后。

诊断隔离于训练：每任务首batch在同一pre-update快照额外测两侧g3，各1普通CE；每项正式训练10200训练CE+20诊断CE、2550真实SGD，诊断不改变参数/梯度/cache/训练RNG。首批g012/point012指纹仅在Task0更新前按同radius配对；不同最终方向训练后产生的Task1梯度不要求相同。主要汇总AAA/FAA/LA/BWT/遗忘，样本加权指标另列，结果区分new/reused。

启动验收凭证为LAUNCH_VERIFIED.json，快照时间2026-10-06T19:36:29.266876+00:00；所有60执行文件与190原始source文件固定。两项B1旧结果已通过原validator和receipt复用验收。任务后续会自动接续，实时状态以runtime/queue_state.json为准，本快照不代替未来状态。
