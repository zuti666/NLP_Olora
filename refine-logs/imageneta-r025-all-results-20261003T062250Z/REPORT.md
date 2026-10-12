# ImageNet-A：GAM外半径 r=0.25 全结果核验

快照：2026-10-03T08:23:42.055704+02:00。共28条独立完整十任务训练（早期native 1、旧recovery 9、旧A/B 2、新四探针16）；另有4条独立单任务训练。没有把身份参考、O-LoRA镜像目录、smoke短测或诊断批次计作独立完整训练。本次只核验和整理，没有新增训练、修改训练代码或改变队列。

共同训练设置：ImageNet-A 200类，固定类别顺序、不shuffle，10×20类；ViT-B/16，LoRA rank10；seed2；每任务50epochs，共500epochs、2550实际更新；batch128，SGD，无momentum/weight decay，首任务LR=.04、后续LR=.02，cosine最低比例.1；内层ρ=gam_grad_rho=.05，外层r=gam_grad_norm_rho=.25。全部正式十任务结果关闭EWC/Fisher和权重融合；lambda_flat=1。旧B主动对c21残差投影，不应称其无正交操作。

AAA/FAA/LA以任务等权百分比报告；BWT/Forget以百分点报告。AAA=十阶段各阶段已见任务平均准确率的平均；FAA=最后一行十任务均值；LA=矩阵对角均值；BWT=前九任务最终准确率减其刚学习完准确率的平均；Forget=前九任务历史峰值减最终准确率的平均。FAA=LA+.9BWT。每个原始10行三角矩阵均独立重算，未与逐样本准确率或类别等权准确率混用。

## 早期原生GAM分支结果

最早的纯g3：AAA/FAA=66.230 / 50.799，LA=57.317、Forget=7.366、BWT=-7.242。β=(0,1,1)、γ=0；这是仅g3更新，不是原版完整GAM。其原生JSON的AAA=59.735090909使用55个三角单元均值，本报告统一按十阶段等权重算为66.229609。

在明确搜索的三个根目录、旧索引和有效配置范围内，未找到r=.25、γ=.1的原版完整GAM，γ=0的原生g1+g3和，或含EWC/Fisher/融合主线的完整训练。已知相关完整GAM结果使用r=1，不挪用到本表。该结论是有边界的本地检索，不是所有磁盘上的绝对不存在证明。早期g3采用global数据RNG，不能与后来的固定epoch增强recovery g3直接配对或合并。

## 旧单方向和差分消融：11条独立训练

探针：g0=g(φ)，g1=g(φ+ρN(g0))，φ2=φ+rN(g1−g0)，g2=g(φ2)，g3=g(φ2+ρN(g2))。五次CE梯度：原生四次探针，加学习器的clean CE一次。旧实现按减扰动恢复，探针内自然推进RNG；11条recovery/A/B运行的输入、标签、索引、epoch采样与LR序列核对一致。

自然步长使用原LR。匹配步长在候选自身当前模型状态上匹配参考g1的真实LoRA B@A与全部已见FC联合位移；旧参考保留原生clean+(g1−clean)的FP32算术路径。并非全训练轨迹共享相同参数，也不是只匹配梯度范数。

| 更新方向 | 自然AAA / FAA | 匹配AAA / FAA |
|---|---:|---:|
| g0 | 60.947 / 38.893 | 60.592 / 39.588 |
| g1 | 63.828 / 45.120 | 63.828 / 45.120 |
| g2 | 66.586 / 50.720 | 65.565 / 48.968 |
| g3 | 67.179 / 50.845 | 65.562 / 48.428 |
| g0+(g3-g2) | 65.230 / 48.663 | 65.142 / 48.565 |
| g0+(g2-g1) | 66.105 / 50.531 | 未运行 |
| g1 + perp_g1(g2−g1) | 65.897 / 49.418 | 未运行 |

g1自然与匹配两格是同一身份对照，仅一条完整训练。recovery的Ufull实际为g3，不是完整GAM。B中c21=g2−g1，a=<g1,c21>/||g1||²，采用全可训练LoRA因子与FC的全局FP64归约，提交g1+c21−a·g1；它没有使用原版GAM的γ修正，但保留实验有意加入的残差投影。

## 本轮四分支：16条独立训练全部完成

统一四次梯度为g0、g1+、g1−、g2。每个探针精确恢复原快照，并重放同一批次R0，最后只提交一次更新并推进到g0的R1；不计算g3、不做GAM正交分解。与旧五次探测协议有实质差异，所以本轮使用自己的g0/g1参考，跨协议不合并排名。

g1±=g(φ±ρN(g0))；φ2=φ+rN(q)，始终从原基点φ出发。O_m=g2_m，A_m=g0+(g2_m−g1_selected)。

| 分支 | 被减去的第一侧梯度 | 外层方向q |
|---|---|---|
| P | g1+ | g1+−g0 |
| D | g1− | g1−−g0 |
| C | g1− | g0−g1− |
| R | g1+ | g0−g1+ |

| 最终方向 | 自然AAA / FAA | 匹配AAA / FAA |
|---|---:|---:|
| g0 | 60.337 / 40.140 | 59.955 / 39.565 |
| g1+ | 63.234 / 44.578 | 同左：身份参考 |
| g1− | 51.090 / 25.368 | 未运行 |
| O_P | 66.663 / 49.937 | 未运行 |
| O_D | 46.887 / 24.999 | 未运行 |
| O_C | 66.647 / 49.793 | 未运行 |
| O_R | 48.943 / 26.857 | 未运行 |
| A_P | 65.900 / 49.141 | 65.509 / 48.042 |
| A_D | 62.695 / 45.153 | 63.085 / 45.166 |
| A_C | 66.758 / 49.864 | 65.516 / 48.103 |
| A_R | 2.169 / 0.485 | 2.031 / 0.554 |

本轮匹配参考为候选自身当前状态、同批次的raw g1+真实BA+FC位移。g1+匹配是身份参考，不另训练。实际匹配相对误差最大约1.02205e−6，满足冻结标准。A_R两条训练均完整、有限、验收通过，但发生性能崩溃；不得把它们写成NaN或执行失败，也不得因Forget较小而认为保持能力好。

## 全部完整训练的LA、BWT和遗忘

| 协议 | job | 方向 | 步长 | AAA | FAA | LA | BWT | Forget |
|---|---|---|---|---:|---:|---:|---:|---:|
| 旧recovery | screen_u0_matched_r025_s2 | g0 | matched | 60.592 | 39.588 | 51.819 | -13.590 | 13.813 |
| 旧recovery | screen_us_matched_r025_s2 | g1 | matched | 63.828 | 45.120 | 55.577 | -11.619 | 11.619 |
| 旧recovery | screen_ug_matched_r025_s2 | g0+(g3-g2) | matched | 65.142 | 48.565 | 55.348 | -7.537 | 7.868 |
| 旧recovery | screen_uo_matched_r025_s2 | g2 | matched | 65.565 | 48.968 | 55.285 | -7.019 | 7.142 |
| 旧recovery | screen_ufull_matched_r025_s2 | g3 | matched | 65.562 | 48.428 | 54.810 | -7.091 | 7.224 |
| 旧recovery | screen_u0_natural_r025_s2 | g0 | natural | 60.947 | 38.893 | 51.884 | -14.434 | 14.434 |
| 旧recovery | screen_ug_natural_r025_s2 | g0+(g3-g2) | natural | 65.230 | 48.663 | 55.326 | -7.403 | 7.624 |
| 旧recovery | screen_uo_natural_r025_s2 | g2 | natural | 66.586 | 50.720 | 56.824 | -6.782 | 7.036 |
| 旧recovery | screen_ufull_natural_r025_s2 | g3 | natural | 67.179 | 50.845 | 57.429 | -7.316 | 7.316 |
| 旧A/B | screen_c21_base_natural_r025_s2 | g0+(g2-g1) | natural | 66.105 | 50.531 | 56.923 | -7.102 | 7.102 |
| 旧A/B | screen_c21_perp_natural_r025_s2 | g1+((g2-g1)-a*g1) | natural | 65.897 | 49.418 | 55.640 | -6.913 | 6.957 |
| 早期native | s3_puregam_onlyg3_r0p25_k10_s2 | g3 | native_nominal_lr | 66.230 | 50.799 | 57.317 | -7.242 | 7.366 |
| 新四探针 | S1-01_CE_natural_s2 | g0 | natural | 60.337 | 40.140 | 51.005 | -12.072 | 12.240 |
| 新四探针 | S1-02_Splus_natural_s2 | g1_plus | natural | 63.234 | 44.578 | 53.703 | -10.139 | 10.269 |
| 新四探针 | S1-03_Sminus_natural_s2 | g1_minus | natural | 51.090 | 25.368 | 45.184 | -22.018 | 22.018 |
| 新四探针 | S1-04_O_P_natural_s2 | O_P | natural | 66.663 | 49.937 | 56.484 | -7.274 | 7.362 |
| 新四探针 | S1-05_O_D_natural_s2 | O_D | natural | 46.887 | 24.999 | 40.117 | -16.798 | 16.798 |
| 新四探针 | S1-06_O_C_natural_s2 | O_C | natural | 66.647 | 49.793 | 56.285 | -7.213 | 7.344 |
| 新四探针 | S1-07_A_P_natural_s2 | A_P | natural | 65.900 | 49.141 | 55.840 | -7.443 | 7.567 |
| 新四探针 | S1-08_A_D_natural_s2 | A_D | natural | 62.695 | 45.153 | 52.618 | -8.294 | 8.418 |
| 新四探针 | S1-09_A_C_natural_s2 | A_C | natural | 66.758 | 49.864 | 56.241 | -7.086 | 7.086 |
| 新四探针 | S1-10_O_R_natural_s2 | O_R | natural | 48.943 | 26.857 | 40.390 | -15.037 | 15.037 |
| 新四探针 | S1-11_A_R_natural_s2 | A_R | natural | 2.169 | 0.485 | 1.641 | -1.284 | 3.028 |
| 新四探针 | S1-12_A_P_matched_s2 | A_P | matched | 65.509 | 48.042 | 54.217 | -6.861 | 7.207 |
| 新四探针 | S1-13_A_D_matched_s2 | A_D | matched | 63.085 | 45.166 | 54.009 | -9.826 | 9.949 |
| 新四探针 | S1-14_A_C_matched_s2 | A_C | matched | 65.516 | 48.103 | 54.196 | -6.770 | 7.017 |
| 新四探针 | S1-15_A_R_matched_s2 | A_R | matched | 2.031 | 0.554 | 0.982 | -0.476 | 0.476 |
| 新四探针 | S1-16_CE_matched_s2 | g0 | matched | 59.955 | 39.565 | 51.690 | -13.472 | 13.517 |

## 单任务结果：单列，不作为十任务AAA

四条均为纯native g3、rank10、ρ=.05、r=.25、50epochs，仅Task0；测试307张、20类。独立task0 seed2与formation seed2是两个入口/轨迹，不是同一次运行。

| 实验 | seed | 样本准确率 | 类别等权准确率 | 测试CE |
|---|---:|---:|---:|---:|
| task0_g_r025_k10_s2 | 2 | 86.971 | 88.689 | 0.400347 |
| formation_g_r025_s0 | 0 | 86.971 | 89.075 | 0.423335 |
| formation_g_r025_s1 | 1 | 86.319 | 87.989 | 0.435827 |
| formation_g_r025_s2 | 2 | 86.645 | 88.462 | 0.398190 |

formation三seed样本准确率86.645±.326，类别等权准确率88.509±.544（样本标准差，n=3）；这不是十任务持续学习性能或方向比较的多seed置信证据。另18条future短续训仅使用r=.25形成的起点，续训有效配置为g1、外半径字段1、100steps，不是本报告的完整r=.25两任务训练，列在排除CSV中。

## 结果支持的结论与边界

- 旧协议下，g0+(g3−g2)比g0自然步长高4.282 AAA、9.770 FAA；匹配后仍比g1高1.314 AAA、3.445 FAA。旧g0+(g2−g1)比g0高5.158 AAA、11.638 FAA；g1+perp(g2−g1)比g1高2.069 AAA、4.298 FAA。因此两种差分都有描述性收益，但后两条目前只有自然步长。
- 旧g3比g2的自然AAA只高.592、FAA只高.125；匹配后g2/g3 AAA相差.004、FAA反而g2高.540。尚无稳定证据说第二次内层探针g3必然优于g2。
- 新协议P/C外点方向都有效：O_P/O_C AAA差.016、FAA差.144；反向第一探针配原样差分D明显变差，而反向第一探针配反号差分C恢复。这支持“第一探针侧与外层差分符号需要一起解释”，不能称C删除了第一部分。
- 新A_P/A_C匹配后分别比g1+高2.276/2.282 AAA、3.464/3.525 FAA，收益不完全来自总体有效更新步幅。自然A_C比A_P高.859 AAA，但匹配后只差.006 AAA/.061 FAA，方向与幅值的解释需分开。
- 新A_D匹配AAA比g1+低.149、FAA高.588，收益远弱于A_P/A_C；A_R两种步长均崩溃。差分加回基点不是对任意符号或位置都有效。
- 这些都是单seed2完整训练的描述性比较，不声称统计显著、跨seed等价、效率优势或已证明Hessian机制。P/C仍用第一次差分选外点，A_P/A_C也改变减法参考，不能由当前结果声称第二部分与第一部分独立。

## 队列和机制诊断

正式S1 16/16已完成验收；阶段2的8条正式训练尚未启动。实际逐job计数为32completed（16smoke+16formal）、2failed诊断、17pending（8S2smoke+8S2formal+1response诊断）、0running。队列halted_failure，保存的顶层counts含过时running字段，不能作为实时进程证据；独立核查controller/worker身份均不存活。

Stage1机制诊断在Task0 entry第一批次失败：P/D中g0、g1+、g1−未逐位一致。梯度在分支选择前已计算，内部状态恢复检查通过；但报告只记录真假，没有差异范数或重复同分支对照，现有证据无法区分CUDA非确定性与未捕获状态。完整训练完成记录成立，严格同状态机制归因尚未验收。

完整math HVP门禁失败：12/12同点一阶梯度比较relative L2为0且恢复通过，但仅2/12响应、1/6批次满足冻结有限差分与相邻两点条件。HVP为有限值，数值验证未通过；不是OOM，也不是曲率机制被否定。阶段2Hstart/Hmid/Residual-only以及math Sec桥接8条没有完整结果。本次未放宽阈值或重启队列。

## 文件与复查来源

- [all_10task_results.csv](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r025-all-results-20261003T062250Z/all_10task_results.csv)
- [single_task_results.csv](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r025-all-results-20261003T062250Z/single_task_results.csv)
- [excluded_anchor_only_continuations.csv](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r025-all-results-20261003T062250Z/excluded_anchor_only_continuations.csv)
- [原始新队列](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/queue_state.json)；[本轮suite](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/suite.json)。
- evidence三份独立核验JSON含原配置、完成凭据、原始矩阵、hash及检索范围；逐条CSV保留原始来源链接。

