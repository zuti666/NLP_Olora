# ImageNet-A：P／C／D双梯度组合的代码与结果比较（2026-10-06）

本次只读核验训练与队列，保存分析文档，没有启动实验。P/C/D各两个半径，正式结果6/6完整验收；无失败、无缺失。逐原始矩阵重算AAA、FAA、LA、BWT、遗忘，与收据及矩阵/summary SHA256一致。每条均10任务、500 epochs、2550更新。

## 共同设置

ImageNet-A、固定200类顺序、10×20类、seed2、ViT-B/16、LoRA rank10；内半径ρ=.05，外半径r=.25/1；batch128，SGD，首任务LR=.04、随后=.02，cosine，momentum/weight decay为0。全部可训练LoRA A/B与已见类别FC参与探针及更新。无EWC/Fisher、实际权重融合、投影、HVP、replay或最终范数匹配。

记g(φ)=∇L(φ)，g0=g(φ)，N(v)=v/(||v||+ε)，g1±=g(φ±ρN(g0))。外层均从原基点φ构造，g2X=g(φ+rN(qX))。完成探针后还原φ，单次更新φ←φ−ηUX。

| 分支 | 外层方向q | 最终方向U |
|---|---|---|
| P | g1+−g0 | g1++g2P |
| C | g0−g1− | g1−+g2C |
| D | g1−−g0=−qC | g1−+g2D |

C/D同一代码只由branch决定q的符号；同状态下外层点关于φ对称。P→C同时改变测量第一梯度的侧别、外层有限差分构造和最终组合中使用的第一梯度，不能作为只改一个因素的实验。

## 实际代码与探针预算

- [C/D direction_step.py:132](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/trainer/direction_step.py:132)：g0、g1+、g1−；第139行q分支，第140行原基点外层位移，第144行U=g1−+g2。第189行起自然SGD，提交base−LR·U。
- [P direction_step.py:183](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/trainer/direction_step.py:183)：g0、正向g1、g2、g3；[第70行](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/trainer/direction_step.py:70)提交g1+g2。g3不作为compose_direction参数，没有路径进入U。
- [state_transaction.py:50](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/trainer/state_transaction.py:50)：参数、梯度、buffers、模块模式、缓存、attention/BN控制和RNG恢复。两批运行逻辑相同；每探针重放R0，最终只推进到g0后的R1。

C/D每步[g0,g1+,g1−,g2]，P每步[g0,g1+,g2,g3]，均4次梯度；C/D的g1+、P的g3是未用于U的预算/诊断探针。六项都实际10200次CE梯度、2550次SGD，无Fisher/融合/投影/HVP。P的通用effective_config字段compute_g3_probe=false被自定义执行器覆盖；不能只读该字段判断g3是否执行。初始权重、数据/标签/索引和LR序列相同，但更新后参数轨迹分离；未使用探针的身份差别不声明逐位数值或耗时相同。

## 完整最终成绩

AAA与FAA均为任务等权%，LA为刚学完各任务时对角均值；遗忘单位为百分点。

| 分支 | r | AAA | FAA | LA | 遗忘 |
|---|---:|---:|---:|---:|---:|
| P | .25 | 66.865 | 50.606 | 57.856 | 8.189 |
| C | .25 | 66.462 | 50.479 | 57.641 | 7.958 |
| D | .25 | 58.766 | 41.123 | 52.156 | 12.259 |
| P | 1 | 68.605 | 54.722 | 59.447 | 5.799 |
| C | 1 | 67.628 | 51.461 | 58.499 | 7.977 |
| D | 1 | 64.869 | 48.709 | 54.108 | 5.999 |

| 对比，百分点 | r=.25 ΔAAA / ΔFAA | r=1 ΔAAA / ΔFAA |
|---|---:|---:|
| P−C | +.403 / +.127 | +.977 / +3.261 |
| C−D | +7.696 / +9.356 | +2.759 / +2.752 |
| P−D | +8.098 / +9.483 | +3.736 / +6.013 |

## 局部解释：C为何接近P，D为何方向相反

在共同φ、相同batch/随机状态、梯度可微且ρ足够小的局部近似下，令u=N(g0)、H=∇²L(φ)：

$$g_1^+\simeq g_0+\rho Hu,\qquad g_1^-\simeq g_0-\rho Hu.$$

所以qP≈ρHu、qC≈ρHu、qD≈−ρHu。忽略ε且g0非零，Hu也对应∇φ||g0||。P/C均近似向梯度范数上升方向选外点，D则反向。C使用下降侧测差分后重新校正差分符号，保留了P的外层方向一阶结构；D下降测差分但不校正符号。有限ρ下P/C不完全相同，此推导不保证大r处的梯度或训练结果，也不是已完成HVP验证。

同一初始权重和首batch、r=1日志直接核验：基点loss=3.564009，第一正/负探针约3.780648/3.352423；外层loss为P6.579128、C6.704959、D2.590209。这支持外点符号实际起效，但只代表共同起点这一批次，不能推断全轨迹或Hessian性质。

## 结果支持与不支持的结论

1. r=.25下降第一探针经C符号校正后几乎保留P性能（FAA只低.127），因此第一SAM式上升探针不是本设置保留较好持续学习表现的唯一可行构造。C仍使用第一探针，并非删除第一步。
2. C−D两个半径均为正，同下降第一探针下外层方向符号确实影响完整训练效果。它比P−C更能单独检验外层符号，但最终步幅仍随外点梯度改变。
3. r=1 P相对C的FAA优势3.261，伴随LA+.948、BWT+2.570及遗忘−2.178。第一探针/组合在大扰动下可能更影响保持；P→C同时改多个相关因素，不能单独确定是哪一个造成差异。
4. r=1 C比D的FAA高2.752，LA高4.391，但遗忘也高1.978，C−D收益不能统一归因于减少遗忘。
5. 三者均自然步长，U不除2、不匹配实际LoRA BA+FC位移。六项单seed，SD为N/A；未分离方向、更新幅度和完整轨迹适应，不能声称机制已被证明或P/C完全等价。

下一步若要区分第一梯度本身与外层位置，只需补交叉条件g1−+g2P、g1++g2C，并采用统一探针执行器；配合匹配有效步长及多seed再确认。固定同一模型状态直接测cos(qP,qC)、q与梯度范数方向的关系可检验局部解释。本轮未加入或运行这些新条件。

[累计114项完整汇总](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-gam-cumulative-results-20261006T150045Z/REPORT.md)。逐原始结果路径、计数与本次重算证据见同名JSON。

## 补充：加入O/A后的P/C分析

见[O/A/双梯度求和扩展比较](/data/140-1/users/liying/O-LoRA/refine-logs/IMAGENETA_PC_O_A_SUM_COMPARISON_20261006.md)，包含18条已有正式训练、步长匹配定义、P/C差分符号改写及真实方向余弦。
