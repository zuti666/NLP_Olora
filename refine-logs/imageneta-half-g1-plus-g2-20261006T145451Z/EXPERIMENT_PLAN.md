# ImageNet-A：(g1+g2)/2 更新幅度对照

新增一个最终方向，在r=.25、r=1各独立训练一次十任务序列，共2正式；seed2、rank10、内层rho=.05与上一轮一致。名义学习率与其余参数均保持原值，只将最终g1+g2梯度乘0.5。

## 梯度与更新

原始GAM正向探针，N(v)=v/(||v||+1e−12)，覆盖所有可训练LoRA/分类头：

\[
g_0=\nabla L(\theta),\quad
 g_1=\nabla L(\theta+\rho N(g_0)),\quad
 g_2=\nabla L(\theta+rN(g_1-g_0)),\quad
 g_3=\nabla L(\theta+rN(g_1-g_0)+\rho N(g_2)).
\]

\[
U=\frac{g_1+g_2}{2},\qquad\theta\leftarrow\theta-\eta U.
\]

g1为正向首探针，不是C/D的g1−。沿用原四探针，g3只计算以控制预算，不进入U。每探针前完整恢复参数/缓冲区/缓存并重播R0，最终恢复原theta并只保留g0后R1，再执行一次普通SGD。没有投影、正交修正或梯度分解，没有Fisher/EWC、权重融合、replay、HVP、额外最终单位化/范数匹配。名义LR不再减半；半和已经写入p.grad，日志direction_factor=.5而scale=nominal_lr。

## 固定设置

| 项目 | 值 |
|---|---|
| 数据 | ImageNet-A train5981/test1519，固定类序0…199，10×20类 |
| 模型 | SeqLoRA，ViT-B/16 augreg2_in21k_ft_in1k，rank10 |
| Seed | 2，独立从相同预训练初始化 |
| 训练 | 每任务50epochs，batch128，共500epochs/2550SGD |
| 优化 | SGD momentum=0，weight_decay=0 |
| LR | Task0=.04，后续=.02，cosine最低各任务初始LR的.1 |
| 探针 | rho=gam_grad_rho=.05，r=gam_grad_norm_rho=.25或1 |
| 模式 | P/half_g1_plus_g2/natural；最终梯度因子.5 |
| 数值 | FP32，TF32off，native attention，create_graph=false |

lambda_flat=1与beta/gamma只作旧配置兼容，实际U由明确新方向分支决定。每步真实4CE/forward、1g3、1SGD，投影/Fisher/融合/HVP/oppositeg1均0。

## 比较与结论边界

主要配对复用已完成g1+g2：r=.25任务等权AAA/FAA66.865/50.606；r=1为68.605/54.722。辅对照g1+g2−g0分别66.832/50.552、64.775/46.873；P/g2分别66.663/49.937、65.735/50.250。六完整历史控制摘要/矩阵/trace/config/receipt哈希再次核验，见historical_inventory.json。未找到此半和方向已有同协议ImageNet-A结果；旧(g1+g3)/2与(g0+B)/2不等价，不能复用。

在同一参数、batch和探针状态下，本轮U与原求和方向相同、参数梯度范数减半；数学上等价于原求和使用eta/2。两个半径均保留原eta，不再重复减半。FP32/FP64测试验证共同梯度状态下2Uhalf=Usum与同一次SGD的半学习率身份。LoRA有效BA+FC位移可能含非线性乘积项，不宣称其范数严格减半，也不是动态有效步长匹配。

若半和接近原和成绩，支持原收益并非依赖双倍求和幅度；若明显下降，支持更新幅度/有效学习率是影响因素。若优于原和，则该半径可能更适合较小更新。独立轨迹的探针梯度会随训练分开，不能把终值差解释为固定方向的纯因果贡献；单seed不称统计显著或Hessian机制证明。

## 队列与验收

只用GPU0/1，最多2并行。当前0/1正在跑imageneta-g0-plus-g1-plus-g2-20261006T144354Z两项正式；本轮任何GPU工作必须等待这2项全部完成且收据/配置/结果哈希验收通过，再自动执行2个短测（2任务×1epoch×1batch）＋配对门，随后2正式。共5作业。旧运行不修改、不停止；GPU2/3不在本轮池。

共享物理index+UUID租约与空闲检查memory<500MiB/util<5%，领取卡后独立seededCUDA witness，然后真实模型/数据短测。CPU公式/位置/四探针预算/R0-R1完整恢复/无g3贡献/一次SGD先验收；源码190项、冻结artifact、独立代码/验收逻辑、四实际CLI和fresh documented invocation通过后启动后台等待。短测或正式验收失败则停止新增接续并保留失败证据。

主指标任务等权AAA，辅指标FAA/LA/BWT/forgetting；另存逐样本准确率，完整三角任务矩阵与每步loss/范数/余弦/有效BA+FC位移、恢复检查和计数。完整10任务、500epochs、2550updates、exit0与hash/矩阵重算通过后才接受最终成绩，中途分数仅作进度。

冻结代码/config/计划RUN_PINS.json。实时runtime/queue_state.json，完成自动summary.csv/RESULTS.md；新代码trainer/direction_step.py，不修改原批次。
