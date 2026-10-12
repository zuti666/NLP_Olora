# 用户引用的正式三 seed B1/B2：配置与结果复核

2026-09-21，CPU 只读复核原始结果；没有新增训练或模型评估。该组是 ImageNet-A、ViT-B/16、rank10 Q/V LoRA 的新训练三 seed 实验。`full_rq2` 表示完整实验覆盖，不表示全模型微调。

## 配置与历史 seed0 的关系

| 项目 | 正式三 seed |
|---|---|
| 数据 | ImageNet-A，200 类，固定顺序，10×20 类，无 replay memory |
| 模型 | ViT-B/16，224 输入，12 层 Q/V rank10 LoRA；冻结预训练 backbone |
| 初始化 | 从历史 checkpoint 只提取 300 个冻结 backbone 张量；重新初始化 LoRA 和分类头。每 seed 的 SGD/GAM 配对初始化；训练 seeds 0/1/2 |
| Task0 | SGD 或 GAM，50 epochs，lr0.04，batch128 |
| 后续 Task1–9 | SGD→SGD，GAM→GAM，50 epochs/task，lr0.02，EWC lambda2000 |
| 其他训练参数 | cosine，momentum0，weight decay0；GAM rho0.05、norm_rho0.2、gamma0.1、lambda_flat1 |
| 随机与精度 | workers4、每任务 reset RNG，FP32，matmul/cuDNN TF32 均关闭 |
| 诊断样本 | 同样的 Task0 train1259/test307；训练子集256。数据清单和解码张量与历史 recovery1 一致 |
| 诊断随机性 | 所有训练 seed 共用 direction_seed=subset_seed=20260920；三 seed 表示训练重复，不是三份随机方向库 |
| raw Fisher | true-label 逐样本 CE 梯度平方均值，有效 Q/V 坐标；无 trace normalization、floor、regularizer 或 lambda 乘数 |

训练 EWC 保护矩阵仍是 batch-mean-gradient-square 的有效权重对角估计，分别归一化任务 Fisher 与累计 Fisher；它与上述诊断 raw Fisher 不同。Task0 没有历史 Fisher，因此其 EWC 项为零。新序列继承各自新 Task0 的权重和新估计的 Fisher，并未继承历史训练的保护矩阵。

旧 recovery1 是历史 seed0 checkpoint；正式组重新训练了 6 个 Task0 和 6 条序列。正式 seed0 的 SGD/GAM checkpoint 哈希分别以 `12c1c40a`、`c3986b84` 开头，历史的为 `b352fcee`、`268cfd85`，有效权重哈希也不同。主要学习率、epoch、rank 相同，但训练 workers 从历史8改为4，并显式规定配对 RNG 和 FP32。因此不能逐点拼接两批数值。

B1 的正式 dense 是81半径×100方向，历史当前图是7半径×16方向（另外已排队补11–19）；B2正式路径用全1259训练样本、t1/5/9三个终点，历史逐阶段扩展用256子集、t1–9。坐标定义相同，但模型与样本量不同。

来源：[seed0 配置](../configs/seed0.json)、[训练程序](../train.py)、[冻结 backbone 核验](../frozen_base_witness.json)、[执行协议](../EXECUTION.md)、各 `formal_training/*/run_manifest.json` 和随机数收据。

## 完成性及本次复核

- 正式队列88/88作业complete；B1/B2诊断54/54，无缺失或重复组合，summary均有限且SHA256与覆盖记录一致。
- B1从24份原始JSONL重新聚合48,960个状态：dense每组8001，train256 sparse97，test307 sparse49，fulltrain1259 sparse13；6个seed×method组合全部通过。重新核验100个公共方向、原点恢复、模型/数据/协议身份、损伤和raw trace聚合。
- B2独立核验6条trajectory与18条path。训练Fisher覆盖t0–9；测试Fisher仅t0/1/5/9。18条路径均有9个节点，用1259个旧任务训练样本；积分、恢复和数值界检查全部通过。由原始节点重建积分的最大误差5.6e-17。
- 当前88作业范围含B3；本文件只解读用户引用的B1/B2。未混入旧pilot或full-model ImageNet-R实验。

机器可读复核：[REQUESTED_B1_SOURCE_CHECK.json](REQUESTED_B1_SOURCE_CHECK.json)，复算脚本：[check_requested_b1_sources.py](check_requested_b1_sources.py)。该检查不重新执行模型梯度，也不替代完整独立实验完整性审计。

## B1 关键结果

以下均值±样本标准差以训练seed为单位，配对n=3。

| r=20 指标 | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|
| train平均正损伤G | 0.046783±0.001856 | 0.033971±0.000603 |
| train Q95 | 0.056841±0.001778 | 0.041607±0.001058 |
| test平均正损伤G | 0.079931±0.002713 | 0.089728±0.003374 |
| test Q95 | 0.124231±0.003401 | 0.128929±0.004505 |
| raw trace，train256、16方向 | 2650.23±94.91 | 1403.40±19.92 |
| raw trace，全train1259、4方向 | 2357.68±153.67 | 1261.13±68.03 |
| raw trace，全test307、16方向 | 9432.00±620.08 | 8309.85±417.17 |

G定义为方向上 `[L0(theta+ru)-L0(theta)]_+` 的均值，而不是绝对loss。训练G在240/240个半径×seed比较中更低；测试G在225/240个比较中更高。这些相关探针不是240次独立训练重复。

epsilon=.01时，训练容忍半径（seed0/1/2）从10.75/10.5/10.5变为12.25/12/12.25；测试从6.75/7/7变为6/6.25/5.75。因此正式密集数据确实分辨出训练容忍半径优势，同时测试容忍半径变小。各Fisher范围的所有已测半径均3/3seed下降，但这没有伴随测试损伤普遍下降。

## B2 关键结果

此处t是零基任务编号，t9表示已学完10个任务。

| t9 指标 | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|
| 有效权重位移 | 1.253913±0.038792 | 1.441559±0.010447 |
| 旧任务train有符号CE损伤 | 0.122414±0.018683 | 0.032320±0.004972 |
| 旧任务test有符号CE损伤 | 0.229435±0.035624 | 0.033362±0.013129 |
| 归一化路径方向敏感度S_path | 0.239649±0.037153 | 0.015193±0.000273 |
| Task0测试准确率，固定旧20类头 | 81.1075±1.1744% | 86.6450±0.3257% |

配对n=3的t9 test损伤差为−0.196073±0.025706，S_path差为−0.224456±0.037209；三个seed方向一致。三个路径终点t1/5/9共9个配对，FS-EWCLoRA均位移更大、S_path与test损伤更低。但t1训练损伤3/3更高，t5训练损伤2/3更高，不能宣称所有阶段所有损伤都下降。

路径为各方法Task0到对应端点的有效权重直线插值，不是优化器每一步的真实曲线；后续GAM持续使用GAM，因此B2不能单独识别“仅Task0起点改变”的因果作用。

## 结论边界

实验覆盖已完成，报告的“部分支持”来自实测反例，不是漏跑。可支持训练邻域正损伤更低、训练容忍半径更大，以及持续GAM链在末端具有更大位移、更小旧任务损伤和更低路径方向敏感度。不能把它扩写为训练/测试通用保护区域扩大，或仅凭raw Fisher下降就保证任意方向、任意阶段损伤更小。

完整原报告：[RQ2_MECHANISM_FINAL.md](RQ2_MECHANISM_FINAL.md)。
