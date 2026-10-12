# FS-EWC：新 RQ2 补充实验计划

日期：2026-09-20。计划版本时间：08:25:49 UTC。
状态：设计文档；未启动训练或评估，未修改现有训练/绘图脚本。

用户已确认：当前 GAM 实现与最终 FS-EWC 等价。本计划接受该对应关系，不再将方法身份列为待验证项。文中 FS-EWC 指对应完整方法，anchor shaping-on/off 指明确分阶段的诊断消融。

问题：solution shaping 是否降低旧任务在参数邻域中的敏感度，并使模型在真实后续更新下容忍更大位移而保持较小旧任务损伤？
主实验：ImageNet-A / ViT-B/16 / LoRA rank 10。
支持性复验：ImageNet-R / full fine-tuning，独立报告，不能据其范数与 LoRA 的数值大小做参数化因果比较。
设计依据：当前 fqa_r、Fig_T、table1-1 日志及用户提供的新 RQ2 思路；本地无 FINAL_PROPOSAL 等前置文档。

## 1. 主张与最小证据

| ID | 主张                                                     | 最小可信证据                                                                                                                   | 实验块 |
| -- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------ |
| C1 | FS-EWC 改变旧任务邻域敏感度，并扩大经验损伤容忍半径      | 同数据、同坐标、同随机方向/半径下的 raw empirical sensitivity 与正向损伤；预先指定多个容忍度；train/test 分开；至少3个训练seed | B1     |
| C2 | 该收益在真实未来更新中仍存在，不能仅由更小的参数位移解释 | 实际旧任务 loss damage、位移、raw path sensitivity；固定未来优化器的 anchor shaping 对照                                       | B2、B3 |

要排除的解释：收益仅来自更小位移；仅来自训练中的 trajectory averaging；把最终 accuracy 较高误当成相对 anchor 损伤较小；把 normalized trace≈1 误当成原始敏感度测量。

全文不主张所有方向都安全。随机方向均值、95%分位及 sampled max 都是经验统计，不是全空间 supremum。若仅训练集成立，主张限定在训练诊断集，不把测试反例删去。
本项目没有需要额外验证的 LLM/VLM 等组件，不设此类实验块。

## 2. 已核对的当前配置与复用边界

| 项目           | LoRA fqa_r / Fig_T                                                 | full 随机 gam_beta_sweep                          | full topF_cached                             |
| -------------- | ------------------------------------------------------------------ | ------------------------------------------------- | -------------------------------------------- |
| 数据           | ImageNet-A，固定类别顺序，10×20类                                 | ImageNet-R，Task0 20类                            | ImageNet-R，Task0 20类                       |
| 主干/参数      | ViT-B/16，rank10，有效 delta_w                                     | ViT-B/16，full_model / params                     | full-model anchor，诊断 no_fc / params       |
| 已有seed       | 0                                                                  | 0                                                 | 0                                            |
| Task0训练      | lr .04，50 epochs                                                  | lr .01，50 epochs                                 | 从 stage1 checkpoint 加载，当前评估 epochs=0 |
| 后续训练       | lr .02，50 epochs/task，cosine，batch128，momentum0，weight_decay0 | 当前只训练Task0                                   | 当前纯评估                                   |
| EWC            | lambda2000，gamma .9，单任务及累计F归一化                          | lambda2000，F归一化                               | lambda2000，F归一化                          |
| shaping        | beta1/2/3=1，rho .05，norm_rho .2，gam_gamma .1，lambda_flat1      | rho .05，norm_rho .2，gam_gamma .03               | 来源stage1采用rho .02，gam_gamma .03         |
| 原半径与方向数 | r=0:.25:20，K100                                                   | r=0:.5:50，K200                                   | 计划r=0:.25:20，K200，top10%                 |
| 实际完整性     | LoRA四JSON完整；Fig_T有t0..9                                       | 101半径完整，endpoint与multi-Fisher是不同pass     | EWC到8.75，FS-EWC到10；1%分支失败            |
| 诊断数据       | FQA batch64/max20；Fig_T实际train1259、test307                     | FQA max20；其他loss/grad可all，不能默认为同一pass | FQA max20，topF方向来自各自cache             |
| 当前动态F      | Fig_T重算Task0 F，但normalize=True、trace≈1                       | 不含完整Task0未来序列                             | 固定anchor multi-Fisher                      |

关键来源：

- LoRA FQA：../fqa_r/lora/Fig_stage2_r-2-lora-imageneta/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/gam_seed0.log，第6–101行。
- Fig_T：../Fig_T/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/gam_T10_seed0.log，第6–107行；SGD同目录。
- full随机：../fqa_r/full/stage2_r-full-imagentr/gam_beta_sweep/gam_beta111_eval.log，第16–100行。
- full topF：../fqa_r/full-fisherImportant/topF_batch_10pct_cached/gam.log，第66–105行。
- full阶段开关模板：../table1-1/unified_stage1_lam10000_selected/shell/factored_s1_seed0.log，第66–76行；该模板lambda=10000，不能直接混入LoRA lambda2000。

LoRA FQA与Fig_T对各方法配置了完全相同的Task0 checkpoint来源路径。Fig_T明确 weight_merge=False、无in-task path snapshots；可以作为无averaging的主证据链。新补实验沿用该设置。

当前工作区未保存这套LoRA t0..9 checkpoint及训练源码；日志记录了原服务器保存路径。执行前只需核验/取回这些既有资产，不能用另一批full-model checkpoint替代。原服务器是否仍保留文件尚未检查。完整权重若已丢失，才按同配置补训。

## 3. 全部实验共享的诊断协议

### 3.1 数据、loss与head

- 固定 Task0 类别、样本ID、确定性预处理和 eval 模式；关闭诊断时的数据增强、dropout，固定BN状态（如有）。
- 主 loss 为无EWC/无shaping正则项的旧任务 cross-entropy，固定Task0 local head。全已见类别accuracy另报，不能混为相同loss。
- LoRA密集loss评估使用全部已知Task0 train1259/test307，若恢复后样本数不同先定位数据版本差异。
- raw Fisher试运行用固定分层256个train样本；正式B2使用全部Task0 train1259。B1的正式邻域trace可先用预注册256样本估计，再在r=0/1/10/20的固定方向子集验证全1259样本是否改变结论。
- 任一理论界核验必须让 D 与 Fisher 使用完全相同的样本集合、权重、loss/head；若F用256子集，同步另算该子集的D，不能拿全数据D检验子集F的界。
- test损伤为必要结果；测试Fisher用于离线诊断，不参与训练、超参数选择或按方法挑方向/阈值。

### 3.2 参数空间与恢复检查

- LoRA沿现有 delta_w 的同名同shape有效权重块建共同坐标；记录模块列表、维度、LoRA缩放常数和拼接顺序。
- 用有效权重 \(W_{\rm eff}\) 的Frobenius范数定义半径，不能用A/B因子范数替代。
- 任意有效权重随机扰动可超出固定rank可实现集合，明确标为ambient weight-space诊断；B1真实方向与B2路径提供训练相关连接。
- 线性插值实现 \(W(s)=W(0)+s[W(t)-W(0)]\)，不直接插值A、B因子。后者乘积会产生不同路径。
- 确认插值s=0和s=1分别重现两个endpoint的旧任务loss。若head或未列入delta_w的其他状态也发生变化，扩展坐标/状态以覆盖它们，或将诊断改为明确固定这些状态的条件实验；不能用不一致的endpoint damage代入理论式。
- 每次perturbation后恢复anchor；检查state/weight hash、loss和预测恢复。无优化器更新。

### 3.3 训练F与诊断F分别保存

训练的 ewc_normalize_fisher=True、ewc_normalize_accumulated_fisher=True 保持不变。新增诊断计算未归一化、真实标签的逐样本 empirical Fisher：

\[
F^{emp}_0(\theta)=\frac1N\sum_i g_i(\theta)g_i(\theta)^\top,\quad
T_0(\theta)=\frac1N\sum_i\|g_i(\theta)\|^2.
\]

方向量直接计算：

\[
q_\Delta(\theta)=\frac1N\sum_i(g_i(\theta)^\top\Delta)^2.
\]

不要用 batch-mean gradient平方、true-label-sampled/model Fisher、NCM Fisher 或 normalized trace替代。方向量不能用 diagonal F quadratic冒充完整逐样本方向量；可直接求逐样本方向导数而无需存完整F。

本计划中的 diagnostic_fisher_*、path_* 等新字段是待实现的数据协议，不声称当前CLI已有对应参数。不要全局关闭EWC归一化来获得raw诊断F。

### 3.4 指标与统计

\[
D_k(r)=L_0(\theta+ru_k)-L_0(\theta),\quad
G(r)=K^{-1}\sum_k[D_k(r)]_+.
\]

同时报告G95、sampled max、signed mean、绝对loss、accuracy及相对自身anchor的accuracy下降。
预先固定epsilon集合 {.001,.005,.01,.02,.05}，以loss单位报告，不按方法单独改变。

\[
r_\epsilon^M=\max\{r_j:\max_{i\le j}M(r_i)\le\epsilon\},\quad M=G\ \text{或}\ G95.
\]

超过扫描上界记“≥r_max（截尾）”。每个训练seed先计算半径，再报告seed均值/标准差及每seed点；方向波动与训练seed波动分开。seed0/1/2为正式目标，同seed方法共享初始化、任务划分和可控数据顺序；新增seed仍固定类别顺序，类别顺序鲁棒性放附录。
方向bootstrap仅表述给定checkpoint的Monte Carlo不确定性。3个训练seed不用于强显著性宣传。

## 4. 实验块

### B1：统一邻域容忍度与raw sensitivity（必须，正文）

- 主张：C1。
- 系统：既有EWC/FS-EWC各自的Task0 anchor；seed0先离线试运行，正式seed0/1/2。
- 密集loss半径：0:.25:20；K100独立随机单位方向。同seed的两个anchor、所有半径和train/test复用同一方向库；r=0只算一次。
- 稀疏raw trace：r={0,.5,1,2,5,10,20}；非零半径固定取上述方向库中的16个方向；r0仅一个状态。先256个分层train样本，必要的数据量一致性检查按3.1执行。
- 因既有test优势不一致，在r={0,1,10,20}、同样16个方向补固定test集raw trace。该分析用于解释泛化差异，不参与方法调参。
- 记录 mean/raw-trace Q95/sampled max；均值代表平均邻域状态敏感度，sampled max不是理论supremum。
- 旧seed0方向向量未保存，不能伪装成新paired方向库。旧损伤结果保留为历史再分析，新paired sweep独立run_id。

必须附带一个低成本方向连接诊断：

- 从两方法各自t9 checkpoint构造真实 \(\Delta_{0\to9}\) 单位方向；验证坐标一致。
- 用两个anchor × 两个真实来源方向交叉测loss。
- 等半径集合 {0,.25,.5,1,1.5,2}，并加入两方法各自真实位移范数（每seed至多8个r）。
- 与同半径随机方向的分布比较。当前真实范数约1.20/1.40而random显著分离常在r10–20，必须避免把大r随机结果直接当真实更新的等价代理。
- 交叉方向诊断为局部响应比较，不是新的训练方法或另一个seed。

成功标准：同一split、预设多个epsilon下，FS-EWC的半径在各seed大体一致扩大，同时raw sensitivity相应降低；报告全部预设点。
失败解释：只有G变小但trace不降，则“更耐扰动”可成立，trace机制尚不支持；只有train改善则限定主张；sampled max反例不被mean优势抹去。
目标图：G/G95-r、raw sensitivity-r、r_epsilon-epsilon；真实方向对照作为小面板或附图。

### B2：实际未来更新与路径Fisher（必须，正文）

- 主张：C2；补齐C1到真实轨迹的连接。
- 系统：完整EWC、完整FS-EWC的t0..9 checkpoints，无averaging，保留现有训练设置。
- 每个endpoint：Task0 train/test local loss、accuracy、相对anchor damage、位移范数及平方；Task0原始逐样本trace。
- 所有endpoint的raw trace先train主诊断集；test保留loss全轨迹，在t0/1/5/9追加raw trace检查与泛化相关的差异。
- 选择t={1,5,9}，沿直线theta0+sDelta评估；节点s={0,1/8,...,1}，每节点计算D对应的loss、raw trace及完整方向量q_Delta。
- 复合Simpson得到I8；复用s={0,1/4,...,1}得I4。保存绝对和相对求积差。
- 预定求积稳定条件建议 abs(I8-I4) <= 1e-8 + .05*max(abs(I8),abs(I4))；这是初始数值门限，需在不比较方法优劣的精度pilot中确认适用dtype与量级，随后冻结。
- 若未达稳定、出现明显 D²>I8，或路径尖峰未解析，加密到16段，再视稳定性到32段；不得把离散求积误差判作理论失败，也不得按有利方向调整门限。
- 理论检验比较D²与I，或|D|与sqrt(I)，保留signed D。另报告 \(\|\Delta\|^2\int T_0\,ds\) 的trace上界和界的松紧。
- 若参数子空间不覆盖所有影响loss的改变，先修复3.2中的endpoint一致性，再解释数值界。
- t0共享状态可一次梯度流计算三个目标方向q；t1/5/9 endpoint的q与其trace一起算，避免重复。

成功标准：实际损伤更小且不必靠更小位移；raw/path sensitivity与之相符；积分数值稳定。上界成立本身不证明shaping因果，不能以“更紧”代替“损伤更小”。
失败解释：只有endpoint trace下降、path不下降则不宣称沿整条路径降低；只有损伤改善则保留效果、收窄机制解释。
目标图：实际D/accuracy-t、位移-t、raw trace-t；路径图或D²-I散点放正文/附录按版面选择。

### B3：固定未来优化器的两任务隔离实验（必须，用于shaping/protection主张）

每seed从同一预训练初始化训练两个Task0 anchor，然后分别分支。Task1统一SGD、lr .02、50 epochs、batch128、相同数据顺序/调度；无averaging、无新附加约束。

| 分支 | Task0 shaping               | Task1 optimizer | Task1 protection |
| ---- | --------------------------- | --------------- | ---------------- |
| A    | off（SGD anchor）           | SGD             | lambda0          |
| B    | on（FS-EWC shaping anchor） | SGD             | lambda0          |
| C    | off                         | SGD             | lambda2000       |
| D    | on                          | SGD             | lambda2000       |

- 两个Task0 anchors可复用B1/B2，不重复训练。相同anchor的protection0/2000分支从完全相同checkpoint及Fisher缓存出发。
- C对应固定后续优化器的EWC对照；D对应anchor shaping-only消融。D不是“后续任务也持续GAM”的完整FS-EWC。完整方法单列，从B2读取，避免混淆阶段开关。
- B-A观察无EWC保护时anchor shaping带来的收益；D-C观察相同lambda下的收益；C-A和D-B观察保护作用，并可报告交互项。
- 每个anchor仍用自己的Fisher：D-C包含anchor及其对应保护矩阵的共同变化；B-A不受保护矩阵影响。若仍需隔离保护矩阵结构，再做交叉Fisher，仅作为附录，不扩大核心四格。
- 主指标：a11、a21、a22、BWT；固定旧任务数据的base/final loss与signed/positive D0；新任务性能；effective-W位移；Task0 anchor/endpoint raw trace；有条件复用B2路径诊断。
- a22和新任务训练loss必须报告，避免“少学新任务所以旧任务不坏”的混淆。
- 成功标准：固定未来优化器时，shaping改善旧任务damage且保持可接受的新任务学习；位移和B1的等范数对照共同排除仅缩小移动的解释。
- 失败解释：仅a21更高而D0更差，报告最终性能而非损伤机制；效果只在持续GAM的完整方法出现，说明anchor shaping单独不足，修改因果叙述。
- 不预设必须满足D的位移大于C；即使真实位移较小，B1等半径诊断仍可判断solution响应是否改变。
- 目标表：四格 × a11/a21/a22、D0、位移、raw trace。主要放正文。
- 现有full factored_s1可参考控制逻辑，但其full架构和lambda10000不能直接用作本LoRA四格中的D行。

## 5. 支持性full-model复验与暂不扩展的工作

优先完成LoRA链条。随后从同一来源的一对ImageNet-R full-model anchors复验B1，不能继续把旧random和旧topF两个来源拼成空间对照。
可选已在本地保存的table1-1 stage1_sgd_probe / stage1_gam_probe task_00.pt作为一致来源；保持其已训练配置（lr .01、50epochs、shaping rho .02/gamma .03），不与旧gam_beta111 rho .05作同模型比较。
诊断范围须统一：预先记录完整参数集合，若路径诊断需包含旧head等实际变化，则覆盖该变化；若采用no_fc，只报告对应条件扰动的结论。
full半径可沿用0:.5:50、K200，先pilot后定正式扩展；不比较其半径数值与LoRA的半径来解释LoRA优势。
跨数据集/参数化结果为支持性复验，若论文主张明确同时覆盖两者，则相应B1复验升级为必须项。

核心计划暂不做：9种Fisher估计器全套重画、top1%/bottomk大规模扫描、Hessian全套诊断、merge/damping新增消融、full-vs-LoRA漂移因果比较。旧残差图保留附录诊断，不占新主线中心。

## 6. 执行次序、成本与停止条件

| 阶段 | 首要任务                                                                     | 是否训练               | 停止/继续条件                                  |
| ---- | ---------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------- |
| M0   | 现有JSON重算G/Q95/r_epsilon；核验checkpoint和sample IDs；小规模诊断数值pilot | 否                     | endpoint恢复、F定义和坐标通过，才扩展          |
| M1   | seed0两anchor的B1稀疏raw trace与真实方向对照；B2先t9路径                     | 否                     | 记录train/test差异与积分稳定性；结果负面也保留 |
| M2   | seed0正式paired sweep、全部endpoint、t1/5/9路径                              | 否（checkpoint存在时） | 所有字段完整，不按结果删实验                   |
| M3   | B3四格seed0；补seed1/2并复用anchor到B1/B2/B3                                 | 是，缺失部分           | 以预先主张判定支持范围，失败时收窄而非换指标   |
| M4   | full-model复验、更多方向/样本数、类别顺序等                                  | 条件执行               | 取决于论文主张范围和预算                       |

默认正式训练seed=0/1/2；先利用seed0验证测量实现。所有新任务状态为TODO，历史已有结果不能标为新诊断已完成。

计数是工作量，不是实测GPU小时：

- B1 dense loss：每anchor每split为1+80×100=8001个状态评估。2方法×3seed×2split合计96,012个dataset forward passes。每pass按相应诊断集分batch；不要误称为96,012个单batch。
- B1 train sparse trace：每anchor1+6×16=97个状态，6anchors共582个状态；固定256样本时为582×256=148,992个逐样本梯度计算。测试trace：每anchor1+3×16=49，共294个状态；若用全部307个test样本则90,258个样本梯度计算。
- B1 real-direction loss：2方法×2来源方向×2split×3seed×至多8r，至多192个dataset forward passes，可去重r0。
- B2 train endpoint+path：每seed2方法×[10 endpoints+3目标×7 interior]=62个全诊断集梯度pass；3seed为186。以1259样本计为234,174个逐样本梯度计算。若每条路径从8段加密到16段，每seed最多再增加2×3×8=48个pass。
- B2测试raw trace若只取t0/1/5/9：2方法×3seed×4=24个状态，307样本时7368个逐样本梯度计算；test路径积分作为附加项另计。
- B3冷启动：3seed×[2 Task0 anchor+4 Task1分支]=18个50epoch任务训练段。anchor复用后只增加12个Task1段；符合配置的既有C分支还可复用。
- 完整10-task序列若seed1/2缺失：2方法×2seed×10=40个50epoch任务段；其中4个Task0 anchor与B3共享，不重复计算。
- 全部LoRA B1/B2/B3若完全冷启动：6 anchors+54后续序列任务+12四格Task1=72个任务段（未做同分支去重）；已有seed0两条完整序列可复用时上界减为52段，再按可复用分支去重。
- 总GPU小时/墙钟时间暂不报数值：硬件、逐样本梯度实现和吞吐未测。M0计时得到t_forward_dataset、t_rawF_dataset、t_task50，然后按上述数量求和；单卡墙钟为任务和，多卡按实际可并行作业调度，另加I/O。
- 数据准备：checkpoint、原训练配置/代码、Task0固定诊断集与旧head。无需人工标注。最大开销预计为密集前向sweep与逐样本梯度，具体排序由pilot决定。
- 若预算紧，先用现有loss曲线完成历史再分析，优先新raw Fisher及真实方向/path；正式paired dense sweep与多seed按预算排期，不能把旧独立方向数据称为新paired数据。

## 7. 输出数据协议与数值检查

每个run保存独立JSON/CSV，保留旧结果。建议新结果目录 rq2_supplement/<run_id>/；这里仅规划目录，不已创建实验结果。

身份/复现字段：
method、train_seed、checkpoint_path/hash、task_index、diagnostic_task=0、sample_ids/hash、split、head_scope、loss_definition、parameter_scope/module_names/hash、coordinate=effective_weight、dtype、preprocess、direction_seed/algorithm/library_hash、radius、direction_id、F_estimator、F_normalized=False、sample_count。

neighborhood逐方向字段：
base_loss、perturbed_loss、loss_delta、positive_loss_delta、base_acc、perturbed_acc、raw_fisher_trace（若该点测量）、direction_source、real_update_norm、finite_flag。
可生成mean/q95/max，但保存逐方向原值，不只存summary。方向可用可复现的分块seed/算法和norm/hash记录，不强制保存巨大向量库。

trajectory/path字段：
task、base_loss、endpoint_loss、D、D_positive、D_squared、delta_norm、delta_norm_sq、s、node_loss、raw_trace、q_full_empirical、integration_rule、I4、I8、I16（若有）、integration_error、bound_gap、normalized_protection_F_stats（单独命名）。

M0检查：

- r0的D为数值零，随机方向范数约1，扰动恢复后base loss一致。
- 两方法同seed方向库一致，样本和head一致。
- raw trace与逐样本梯度平方均值一致；方向量与逐样本方向导数平方均值一致。
- 在小模型/少量参数上显式构造完整F核对q，防止误用diag approximation或batch-mean gradient。
- s0/s1重现实际endpoint；不满足时不得报告path bound。
- 求积收敛检查；D²-I不稳定时加密，并核对loss/head/坐标，而非调整方法数据。
- 新的raw诊断与训练保护缓存分别命名和保存，不能覆盖normalized EWC Fisher。

## 8. 风险与结果解释规则

- 已有test不一致：完整报告；若新结果仍不一致，C1限定train，单独分析泛化，不能选择更有利epsilon。
- 真实更新与随机方向的尺度差异：B1交叉真实方向与B2直接路径是必要连接。
- normalized Fisher：保留训练策略，只新增raw测量；动态normalized quadratic可作附图，不当trace机制证据。
- batch empirical与逐样本 empirical不同：以明确定义的逐样本真实标签梯度为理论主诊断，旧估计器留作比较。
- taskwise优化器混淆：B3所有未来分支统一SGD；完整FS-EWC结果单列。
- shaping也改变Fisher保护矩阵：B-A先观察无保护差异，D-C明确含其共同影响，交叉F仅按需要补充。
- 更高最终accuracy与更小damage不同：始终记录起点、终点、D、BWT和新任务学习。
- 理论上界可能很松：满足界不证明机制、界紧不等于方法好；核心证据是raw sensitivity、等半径损伤和实际损伤之间的对应。
- 局部经验半径不是完整安全球：不把有限随机方向q95写成worst-case保证。
- full与LoRA的数据集、空间、超参不同：不合并范数值；跨参数化因果比较不属于核心计划。

## 9. 首批三个作业与交付

R001：CPU历史再分析，统一G、G95、连续前缀r_epsilon，输出train/test与旧配置对应表。
R002：seed0 EWC+FS-EWC共同的诊断sanity bundle：r0/1/10各少量方向、256样本raw F、有效权重插值endpoint检查、计时。
R003：seed0 t9真实方向交叉loss与9节点path，raw F不归一化，数值稳定后扩到t1/5及3seeds。

正文安排：B1三指标图；B2真实损伤/位移/raw sensitivity图；B3四格机制表。旧fixed-F residual与多估计器对比放附录。

## 10. 检查清单

- [X] 接受用户确认的GAM=FS-EWC对应关系。
- [X] 主张限定为两个，必须实验限定为B1/B2/B3。
- [X] 训练配置、诊断新增项和支持性full复验分开。
- [X] trajectory averaging关闭；不引入额外组件。
- [X] train/test、归一化/raw、逐样本/batch、范数/平方范数分开。
- [X] 正/负结果均有解释与主张收缩规则。
- [X] 所有运行数量为计划，尚未启动实验或报告新GPU结果。
- [ ] 在原服务器核验并读取既有LoRA逐任务checkpoint与代码。
- [ ] 完成M0计时后填写GPU小时和执行排期。