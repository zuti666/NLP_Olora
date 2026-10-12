# ImageNet-A r=1：Sum12 / Mean12 / g1+d32 三seed补齐计划

日期：2026-10-06。用户授权三条件seed0/1/2，在空闲GPU直接运行。仅补缺失单元，不修改已完成代码、正在训练的实验或旧等待队列。

## 研究问题与判定

主问题：g1+g2的十任务收益是否跨seed复现，减半最终更新后是否仍保留收益？辅问题：g1+(g3−g2)是否具有稳定的长期学习/保持效果？本轮用同seed的三条件配对比较；只改变最终更新公式，不改变探针位置、数据、初始化、LR和预算。不以性能差值直接声称Hessian机制。

AAA为主指标，FAA为辅；同时报告LA、BWT、遗忘、10阶段逐任务矩阵和样本加权指标。只在预定seed0/1/2齐全且全部验收后报告正式均值±样本SD、配对差值±样本SD、每seed差值和胜出次数。n<3保持缺失状态，不缩减预定seed集合、不补零。

## 固定设置

| 项目 | 设置 |
|---|---|
| 数据 | ImageNet-A train5981/test1519，类序0…199，10×20类，无replay |
| 模型 | ViT-B/16 augreg2_in21k_ft_in1k，SeqLoRA rank10，LoRA scale1，无alpha/rank缩放 |
| Seed | 0、1、2；独立从同预训练模型初始化，各条件同seed配对 |
| GAM | rho=gam_grad_rho=.05，r=gam_grad_norm_rho=1.0，eps=1e−12 |
| 训练 | 每任务50epochs；batch128，总500epochs、2550更新、299050训练样本 |
| 优化 | SGD，momentum=0、weight_decay=0；Task0LR=.04、随后=.02，cosine最低初始LR的.1 |
| 数值 | FP32，TF32off，native attention，无AMP，create_graph=false |
| 禁用 | EWC、Fisher计算、阻尼/归一化、实际权重融合、GAM正交修正、投影/分解、SGV、HVP、最终范数匹配 |
| 原值兼容 | lambda_flat=1；beta/gamma不参与自定义最终U构造，gamma=0 |
| 探针预算 | 四次正向CE梯度/forward、一次自然SGD；Sum/Mean也计算g3维持预算 |

## 探针与三种更新

N(v)=v/(||v||+eps)，范数覆盖所有可训练LoRA因子和分类头参数。

\[
g_0=\nabla L(\phi),\quad
 g_1=\nabla L(\phi+.05N(g_0)),\quad
 g_2=\nabla L(\phi+1.0N(g_1-g_0)),\quad
 g_3=\nabla L(\phi+1.0N(g_1-g_0)+.05N(g_2)).
\]

外层位置从原phi构造，不从phi1累加。每次探针重放同R0并精确恢复参数/梯度/buffers/模式/缓存；最终恢复基点，随机状态只推进到g0后的R1，再安装U执行一次SGD。

| 条件 | 最终U | g3参与更新 | 名义LR倍率 |
|---|---|---|---:|
| Sum12 | g1+g2 | 否，但计算 | 1 |
| Mean12 | .5*(g1+g2) | 否，但计算 | 1 |
| g1+d32 | g1+(g3−g2) | 是 | 1 |

Mean12仅在最终梯度中乘.5一次，LR不再减半。与Sum12在同状态下参数梯度方向相同，数学上等价于Sum12用半LR；LoRA B@A有效更新含乘积项，不宣称BA+FC位移严格减半，也不等同动态有效步长匹配。

## 已有配置、结果与去重

| 条件 | seed0 | seed1 | seed2 |
|---|---|---|---|
| Sum12 | 新增正式 | 新增正式 | 已完成复用AAA68.604840/FAA54.722 |
| Mean12 | 新增正式 | 新增正式 | 旧队列已排队，等待sum012后启动；不重复 |
| g1+d32 | 新增正式 | 新增正式 | 已完成复用AAA65.548813/FAA48.089 |

本新队列仅6正式＋6真实模型短测＋1配对门。已有两个seed2完整收据、原矩阵、2550更新和源哈希已核对；Mean12 seed2来自imageneta-half-g1-plus-g2-20261006T145451Z的formal_halfsum12_r1_k10_s2。旧queue不取消、不转移、不改GPU池，collector严格等待它完成验收后导入第9单元。外部seed2未完成时不会启动重复训练，统计保持partial。

原生(g1+g3)/2及g0+(g3−g2)不是本轮目标，不复用为对应条件。原生5CE/global RNG与本轮4CE/exact restore分组，不混合多seed均值。新训练器在FP32/FP64正常范围的非线性fixture中与历史Sum12、旧排队Mean12、g1+d32的四梯度/U/SGD逐位一致；旧Mean采用.5g1+.5g2，新Mean采用.5*(g1+g2)，测试不宣称极端浮点范围的普遍等价。

## 执行顺序与资源

使用已经验收的Pilot_new_local环境；不安装/升级软件。当前宿主机只有GPU0–6：GPU2/3空闲，0/1跑sum012，4/5/6跑旧seed01队列。新池2–6、最多2并行，优先2/3；不占用0/1、不抢占任何现有进程。共享/tmp/olora_gamfs_gpu_locks的index+UUID双租约，领取前后检查memory<500MiB、util<5%；每worker绑定闲置UUID到逻辑cuda:0。

1. CPU公式、完整恢复、历史实现兼容、实际seed0/1/2与无重复减半验收；独立代码/队列审核、190源文件哈希和冻结pins、fresh documented dry-run。
2. 六短测：三条件×seed0/1，每项2任务×1epoch×1batch＝2更新、8CE。领取GPU后先运行seeded FP32 CUDA forward/backward witness，再真实模型与数据短测。
3. 配对门：仅在同seed内比较三方向的Task0初始参数、真实batch/labels/indices、LR与切分；不要求不同seed相同，不把轨迹参数称持续相同。
4. 六正式按Sum12 seed0/1→Mean12 seed0/1→g1+d32 seed0/1接续。每项10任务/500epochs/2550更新；任何短测或验收失败停止新增接续并保留证据。
5. 每次完成自动更新summary.csv/THREE_SEED_SUMMARY.csv/RESULTS.md；本队列六条结束后若旧Mean12 seed2仍未验收，则等待外部结果，不领取GPU；完整9单元后报告三seed与配对比较。

历史同条件两条正式约2.02/2.04GPU小时；6新增约12–15GPU小时，两GPU约6–8小时，加短测和共享负载。旧Mean12 seed2仍按原队列接续，不把等待时间当作训练失败。

## 指标与验收

每步记录四loss、梯度/差分/U范数、余弦、LoRA BA+FC实际更新长度、direction_factor、nominal_lr和实际scale、R0/R1/状态恢复、真实梯度/forward/SGD/Fisher/merge/HVP/投影计数；diagnostic不参与训练。最终要求exit0、有效seed/rank/rho/r/direction、完整切分和更新坐标、所有观察值有限、零禁用工作、一次自然SGD、trace/matrix/config/trainer哈希、任务等权指标独立重算。

Sum−Mean是更新幅度/有效LR对照；Sum与d32同时改变最终组合和幅度，不直接视作纯方向因果；已有原生Gmatch仅匹配独立轨迹，不能代替本轮精确同状态匹配。Mean与Sum的差异也不能单独证明或否定曲率机制。

## 文件

实时runtime/queue_state.json；配置configs/；执行trainer/four_probe_train.py和direction_step.py；已有来源evidence/inventory.json；环境evidence/environment_gpu.json；独立验收audits/；冻结RUN_PINS.json；结果summary.csv、THREE_SEED_SUMMARY.csv、results.json、RESULTS.md。
