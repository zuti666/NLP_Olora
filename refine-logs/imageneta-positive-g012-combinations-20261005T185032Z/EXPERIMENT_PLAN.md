# ImageNet-A 正向 GAM：g0/g1/g2 组合与平行分量消融

本轮新增三种最终更新，每种在外半径 r=0.25 和 r=1.0 独立完成一次十任务训练，共6项正式实验，统一seed2。已有ImageNet-R实现可核对公式，数据集/seed/学习率/RNG协议不同，不能替代本轮ImageNet-A结果。

## 共同探针

本文g1均为原始GAM的正向首探针g1⁺，不是上一组C/D的g1⁻。令N(v)=v/(||v||+1e−12)，所有范数覆盖全部可训练LoRA和分类头参数：

\[
 g_0=\nabla L(\theta),\quad
 g_1=\nabla L(\theta+\rho N(g_0)),\quad
 g_2=\nabla L(\theta+rN(g_1-g_0)),\quad
 g_3=\nabla L(\theta+rN(g_1-g_0)+\rho N(g_2)).
\]

rho=gam_grad_rho=0.05；r=gam_grad_norm_rho取0.25或1.0。外层位置从原theta构造，不从theta1累计。依据“四次梯度”要求保留g3第四探测以控制预算，三组均不将g3放入最终更新。每个探针前恢复全部模型参数/缓冲区并重播同一batch的R0，最终恢复theta并保留g0之后的R1。g3仅作预算与探针loss诊断，修改其梯度不能改变U；CPU测试验证这一点。

## 三种最终方向

定义d1=g1−g0，d2=g2−g0。P_perp_g0(d)=d−a*g0，a=<g0,d>/(||g0||²+1e−12)。内积在全部可训练参数上作一次全局归约，不逐层投影。

| 组 | 配置update_direction | 最终U | 要检验的差别 |
|---|---|---|---|
| 1 | g1_plus_g2 | g1+g2 = 2g0+d1+d2 | 两个探测点梯度直接相加 |
| 2 | g0_plus_perp_g1_g2_residuals | g0+P_perp_g0(d1)+P_perp_g0(d2) | 固定基点分量，加入两个残差的垂直部分 |
| 3 | g1_plus_g2_minus_g0 | g1+g2−g0 = g0+d1+d2 | 去掉直接相加中重复的一份g0 |

恢复原参数后仅执行一次theta←theta−eta*U。不除以2、不作最终单位化或范数匹配。组2分别计算两个全局投影，其余组无梯度投影；原GAMgamma正交修正均不执行。

在相同theta、batch和探针状态下，U1−U3=g0；U3−U2=(a1+a2)g0（数值稳定项保留）。因此1与3检查额外一份基点梯度，3与2检查两个残差平行于g0的部分是否必要。三条训练轨迹会分开，最终分数不能单独证明Hessian机制。自然步长下方向和总幅度都可能变化，记录幅度和余弦用于解释，不改训练步长。

组2不同于历史B=g1+P_perp_g1(g2−g1)以及g0+B：历史投影参考g1，本组参考g0。组3不同于历史P/A=g0+g2−g1，符号与基点不同。上一组C/D使用下降首探针，本轮为正向首探针。

## 固定设置

| 项目 | 值 |
|---|---|
| 数据 | ImageNet-A，train5981/test1519，固定类序0…199，10×20类 |
| 模型 | SeqLoRA，ViT-B/16 augreg2_in21k_ft_in1k，rank10 |
| Seed | 2，各组从同一预训练状态独立初始化 |
| 训练 | 每任务50epochs，batch128，总500epochs/2550SGDupdates |
| 优化 | SGD，momentum=0，weight_decay=0 |
| LR | Task0=.04，后续=.02，逐任务cosine，最低为该任务初始LR的.1 |
| 探针 | rho=.05，r=.25或1；正向P分支，4CE/1SGD每步 |
| 参数空间 | 全部可训练LoRA+分类头统一归约；构造探针时N单位化 |
| 数值 | FP32，TF32关闭，原生attention，ordinary autograd，create_graph=false |
| 关闭模块 | EWC=0、ewc_skip_task_fisher=true；Fisher近似/归一化/阻尼、权重融合、replay、SGV、HVP、最终归一化/匹配均关闭 |

lambda_flat=1只作配置兼容；本轮直接构造U分支，beta/gamma不参与最终组合。配置中compute_g3_probe=true、g3_used_in_final=false；实际每步计数为依据，不能仅凭配置名判断生效。

## 队列与验收

- 6个短测：每种方向×r=.25/1，2任务×1epoch×1batch；每项2SGD、8CE、2次g3探针。
- 1个配对门：核对六组真实增强后的batch身份、标签/任务切片、Task0初始参数；不声称跨GPU位级梯度相同。
- 6个正式训练：每种方向×r=.25/1；所有短测与配对门通过后接续。
- 只使用GPU0、1，最多两卡并行，共享index/UUID租约，空闲阈值memory<500MiB且util<5%。
- 前置条件：上一组imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z的4项正式实验全部完成且完成收据/结果哈希通过，才允许本轮任何GPU作业入场。现有C/D及其他队列继续运行。
- CPU解析、公式、全局投影、四探针位置、g3独立性、完整状态/RNG恢复、一次SGD验证与独立代码复核先完成；逐GPU领取后执行seededCUDA witness，再执行实际模型/数据短测。
- 短测、预算或结果验收失败则停止后续接续并保留失败证据；不修改冻结代码重试。

## 指标和现有参照

主要呈现任务等权AAA与FAA；同时保存LA、BWT、forgetting、最终逐样本准确率和三角任务矩阵。每步存g0/g1/g2/g3、d1/d2、U的范数、U相对各梯度的余弦、探针loss/有效权重位移、实际SGD更新幅度、真实forward/CE/SGD/g3/投影/Fisher/融合/HVP计数。组2另存a1/a2、垂直残差范数（平行范数由abs(a)*||g0||计算）、正交误差与U沿g0分量的保持误差。

已验证的任务等权历史参照（%，不重复训练）：

| 方向 | r | AAA | FAA |
|---|---:|---:|---:|
| g0 | 与r无关 | 60.337 | 40.140 |
| g1⁺ | 与r无关 | 63.234 | 44.578 |
| g2(P) | .25 | 66.663 | 49.937 |
| g0+g2(P)−g1⁺ | .25 | 65.900 | 49.141 |
| g2(P) | 1 | 65.735 | 50.250 |
| g0+g2(P)−g1⁺ | 1 | 65.842 | 49.325 |

参照同seed/rank/rho/LR/数据/状态恢复协议，历史无用第四探针为g1⁻，本轮为g3；二者都恢复状态且最终只推进R1。本轮测试证明无用探针不影响U，仍不声称GPU位级等价。旧natural RNG/五CE口径结果只作背景，不混入受控表。完整来源和摘要/矩阵哈希见historical_inventory.json。

完成十任务并通过队列验收后才填最终AAA/FAA；中途任务成绩仅作进度。单seed结果不称统计显著。

代码 trainer/direction_step.py，训练生命周期 trainer/four_probe_train.py；所有命令/config在suite.json，冻结RUN_PINS.json。实时状态runtime/queue_state.json；自动结果summary.csv、results.json、RESULTS.md。
