# ImageNet-A 三探针七方向正式实验

日期：2026-10-03（Europe/Paris）。依据用户最新授权，使用空闲物理GPU1/2/3，并行上限3。本轮七组全部从头训练，统一三次梯度读取；已有两探针Backward结果只作独立参考，不算成本相同的本轮结果。仅包含自然步长七组，不加入匹配步幅、额外负s角点或系数扫描。

## 统一构造

全体可训练LoRA A/B与seen FC构成θ。每步在相同数据增强张量、同一随机前向函数R0上读取：

g0=∇L(θ0)，u=g0/(||g0||+1e−12)；  
g+=∇L(θ0+0.05u)，g−=∇L(θ0−0.05u)；  
s=(g+−g−)/2，e=[(g+−g0)+(g−−g0)]/2。

每次探针先恢复同一基点与状态；读取顺序g0→g+→g−。最后恢复θ0，只推进g0生成的R1，并执行一次θnew=θ0−ηU。探针不是已接受的优化步。不计算外点g2/g3，不使用HVP或create_graph。

| ID | 阶段 | update_direction | 最终U与固定FP32算式 |
|---|---|---|---|
| OE01_SAM_s2 | 首批 | g1_plus | 直接复制g+；代数上g0+s+e |
| OE02_Backward_s2 | 首批 | backward_reflected | g0+(g0−g−)；保持原括号顺序，代数上g0+s−e |
| OE03_Central_s2 | 首批 | central | g0+0.5*(g+−g−)，即g0+s |
| OE04_CE_s2 | 后续 | g0 | 直接复制g0 |
| OE05_Negative_s2 | 后续 | g1_minus | 直接复制g−；代数上g0−s+e |
| OE06_Average_s2 | 后续 | even_average | 0.5*(g++g−)，即g0+e |
| OE07_ReflectedAverage_s2 | 后续 | even_reflected | g0+[g0−0.5*(g++g−)]，即g0−e |

实数代数等价不保证不同浮点重排逐位相同；身份方向与Backward使用上述明确代码路径。诊断中的s/e计算不改变更新算式。e与s不要求正交。

## 固定配置

| 项目 | 设置 |
|---|---|
| 数据 | ImageNet-A，本地train5981/test1519；类别0…199；10×20类 |
| 预训练模型 | ViT-B/16 augreg2_in21k_ft_in1k，冻结骨干 |
| 可训练参数 | Q/V LoRA rank10，无alpha/rank缩放；seen FC |
| seed | 2，固定任务/批次/增强序列和分类器扩展初始化 |
| 训练 | 50epochs/task；batch128，保留尾批；完整10任务 |
| SGD | momentum0、weight_decay0 |
| LR | task0 .04，后续.02；cosine T_max50、最低比例.1 |
| 数值 | FP32、math attention、AMP/TF32off、严格确定性 |
| 内层半径 | gam_grad_rho=.05 |
| 外层半径 | 不存在；gam_grad_norm_rho=0仅兼容元数据，outer_radius=null |
| 更新 | natural；无最终方向归一化、无额外差分强度系数 |
| 禁用 | EWC/Fisher/阻尼/权重融合/SGV/投影/HVP/create_graph/replay数据 |

统一method_name=odd_even_local_v1、probe_branch=local_symmetric。ewc_lambda=0，ewc_skip_task_fisher=true，in_task_weight_merge=false，task_endpoint_weight_merge=false，approximate_fisher_in_task=false。代码与日志同时核验禁用项实际调用次数为0。

## 运行顺序

1. CPU验收：七种公式、同随机状态读取、状态恢复、唯一SGD、异常回滚、真实训练入口、完成验证器与配置范围。
2. 七组两任务短测：每组2tasks×1epoch×1batch；只使用GPU1/2/3空闲卡，每卡一个worker。每个worker先做新seed2CUDA witness。
3. CPU七组配对核验：Task0初始参数、冻结骨干、FC新行初始化、任务划分与实际输入/增强顺序相同；不要求不同方法训练后的参数或梯度相同。
4. 所有短测与配对通过后，首批OE01/OE02/OE03在三张空闲卡并行。
5. 首批三组全部完整验证通过后，再启动OE04/OE05/OE06/OE07；后三张卡空闲时自动接续剩余一组。

共15项队列任务：7smokes+1CPU pairing+7formal。任何失败停止新任务启动，保留已有日志，不自动改配置、重试或抢占。不同队列共用/tmp/olora_gamfs_gpu_locks的GPU编号与UUID锁；取锁前后都重新查询，要求显存<500MiB且util<10%。只允许[1,2,3]，不使用GPU0/4/5/6。现有计划及实验目录保持只读。

## 计数与预算

- 每次正式：2550次SGD、7650训练CE；每任务首批两次三探针零更新重放共6CE，十任务60诊断CE。
- 七次正式总计17850更新、53550训练CE、420诊断CE，共53970CE。
- 每次短测：2更新、6训练CE、12隔离CE；七次短测共14更新、126CE。
- 本轮合计17864更新、54096CE；CPU配对不调用模型/CUDA，不产生训练更新。
- 三探针正式估计每组1.5–2GPU小时，七组约10.5–14GPU小时。由于首批三组屏障及后续四组，三卡持续空闲时预计约4.5–6小时加短测；实际以epoch耗时更新。资源等待不计入该估计。

## 记录指标与结果解释

主指标为十阶段任务等权AAA，辅以FAA、LA、BWT、Forget及完整任务矩阵；样本加权指标独立命名。seed2是探索性单seed，不报告跨seed显著性或等价。

每步记录loss0/loss+/loss−、g0/g+/g−/s/e/U范数、s/g0、e/g0、e/s、cos(s,e)、cos(g0,U)、实际±探针参数距离、实际BA+FC位移、学习率、CE/SGD/禁用调用计数、输入身份与状态恢复结果。零范数余弦用null，非有限训练数值立即失败。所有零更新重放诊断计数与训练分开。

主要比较：
- 首批三个方向：固定共同项g0+s，比较e的+、−、0。
- 后续g0与g0±e：分离偶响应单独作用。
- g0与Central：比较正共同响应增量。
- g−与g+：同一状态表示下e系数均+1、s系数改变符号；不同独立轨迹的e数值不相同。

自然步长结果同时含方向与幅度效应。本轮不自动增加匹配步幅作业；后续按单独授权推进。

## 已有结果与实现隔离

此前math两探针Backward完成AAA62.615/FAA43.714，本轮重新训练，检验统一三探针协议下该方向表现。旧native四探针SAM/g0/g−与外层P/C/CD仅作为历史背景，不作为本轮七组结果替代。

190个原始源文件、环境spec、输入资产哈希沿用已验证冻结源；新代码只写入本目录trainer/odd_even_step.py与odd_even_train.py及其CPU测试。旧反向回加、外层P/C/CD、Local-CD计划均不修改；gates/ISOLATION_BEFORE.json记录旧文件哈希。软件环境canonical SHA4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8不变，不安装包。

启动前保存新代码/配置/文档/证据的RUN_PINS；实际14条训练入口dry-run、独立审核、文档命令核验完成后启动。运行状态以runtime/queue_state.json为准，启动与短测/正式训练验收另存LAUNCH与LAUNCH_VERIFIED，不修改冻结计划。

研究依据与已核验结果：
../odd-even-rho005-plan-20261003T183222Z/EXPERIMENT_PLAN.md  
../odd-even-rho005-plan-20261003T183222Z/EXISTING_RESULTS.csv

