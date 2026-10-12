# ImageNet-A 纯 GAM 分量与 FD/HVP：已完成证据和补充实验

核验时间：2026-10-01 20:34 UTC。读取用户附件 `ea8304d7-5f94-4795-804f-90959a17aa34/Pasted text.txt`，并对照 ImageNet-R 七组实验、ImageNet-A 原始任务矩阵、曲线、配置和完成记录。此次未新增训练、未改变已部署队列。

## 已完成的十任务证据

ImageNet-A 的七组自然步幅方向消融，在 **r=1、rho=.05、seed2** 下已有 **4/7**。共同设置是原始固定10×20类、ViT-B/16、LoRA rank10、batch128、每任务50 epochs；Task0 LR .04→.004，后续 .02→.002，cosine；SGD momentum=weight_decay=0，无EWC、Fisher、权重融合和重放。

原矩阵10×10的55个已观测值均有限。9条历史纯GAM完整轨迹的 config/raw-metrics/completion receipt 哈希重新核验通过；每条都有完整10任务训练。上三角结构性空值不算失败。

| r=1方向 | 已完成 | AAA，任务等权 | FAA，任务等权 | AAA，样本加权 | FAA，样本加权 |
|---|---|---:|---:|---:|---:|
| 原版 GAM，gamma=.1 | 是 | 68.028517 | 55.341 | 71.253 | 62.08 |
| g1+g3，gamma=0 | 是 | 67.860863 | 54.726 | 71.103 | 61.49 |
| g1 | 是 | 62.165531 | 45.132 | 65.842 | 51.88 |
| g3 | 是 | 65.993172 | 50.679 | 69.998 | 58.85 |
| g0 | 缺同协议完整十任务结果 | — | — | — | — |
| g2 | 缺完整十任务结果 | — | — | — | — |
| g0+(g3-g2) | 缺完整十任务结果 | — | — | — | — |

AAA任务等权定义为先平均每阶段已见任务准确率，再平均10阶段；不同于原生JSON的55格三角平均。样本加权AAA来自各阶段全已见样本准确率曲线。ImageNet-R当前主报告是样本加权FAA/阶段等权AAA，因此跨数据集展示必须标清口径，不能混用列。

另外5条已完成结果：

| 方向/半径 | 任务等权AAA / FAA | 样本加权AAA / FAA |
|---|---|---|
| (g1+g3)/2，r=1 | 65.950948 / 49.927 | 69.861 / 58.06 |
| g3，r=0 | 62.299385 / 45.253 | 65.936 / 51.94 |
| g3，r=.1 | 64.651266 / 49.384 | 68.504 / 57.01 |
| g3，r=.25 | 66.229609 / 50.799 | 70.226 / 58.92 |
| g3，r=.5 | 66.132657 / 50.692 | 70.011 / 58.85 |

完整纯GAM十任务证据目前都只有seed2；尚无同协议多seed确认或全训练有效步幅匹配结果。旧 clean-SGD 采用不同预算/seed/learner，不填进当前g0；旧lambda=0实验仍可能计算Fisher或融合，也不当作纯g0。

这些结果说明：自然步幅g3比g1高3.827641 pp AAA、5.547 pp FAA；原版gamma=.1比sum只高.167654 pp AAA、.615 pp FAA。这是单seed观察，不能证明正交项必要。sum与mean分别为AAA67.860863/65.950948、FAA54.726/49.927，提示必须控制更新幅度；不能把sum的优势直接全部归因于方向组合。

## 已做的局部诊断不能替代什么

已有独立Task0训练及同checkpoint单步诊断；已有 `q=g3-a*g1` 正交残差和单位方向插值扫描。十任务诊断9个checkpoint/33个batch，825个等有效步幅候选已匹配；独立Task0诊断5个起点/20个batch，500个候选已匹配。

这些扫描不等于 `delta_G=g3-g2`；也没有生成独立完整训练的AAA/FAA。单步CE不能证明该差分带来的长期效果。Task0或未来100步试验不是重新训练的两任务/三任务持续学习。

## 当前队列，非完成结果

宿主 cudahpc40 的进程身份与starttime实际核验通过：

- formation：120项注册任务中30完成（12短测+18正式Task0），6运行、84待运行、0失败；30/30完成记录重新验收通过。正式Task0为18/24；未来78条S更新与6条短切换尚未开始。
- component-horizon：等待进程PID241392存活，25/25 pending；仍等待上述父120项完成并退出。无提前启动GPU。
- 新component队列包含10短测、9次正式十任务训练、6曲率诊断。固定 **r=.25、rho=.05、seed2**；自然{g0,UG,g2,g3}4条+匹配{g0,g1,UG,g2,g3}5条。匹配g1严格保持名义LR，可兼用自然g1对照。尚无这些新方向的训练结果。
- 已排6条曲率诊断只在新 **g1与g3** 轨迹的Task0/4/9执行；没有UG轨迹的诊断，也没有HVP作为更新的训练。

ImageNet-R状态文件及日志显示2运行（原版Task0 epoch32、sum epoch33）、5pending；截至20:33 UTC日志仍更新。尚无完整成绩。它属于cudahpc15，本次没有远程核验PID，不把本机/proc缺失当作退出。

## 对照新附件，最小补充顺序

### 1. 先完成已排的r=.25五方向及步幅控制

现有队列已经覆盖核心问题：`g0`、`g1`、`UG=g0+(g3-g2)`，以及`g2`、`g3`。继续等待即可，不重复登记这些r=.25训练。

用自然与匹配版本分别回答实用收益、排除总更新尺度后的收益。匹配目标是候选自身当前状态/同batch以名义LR沿原生g1更新的实际concat(LoRA B@A,已见FC weight/bias)长度；它不同于formation中独立S轨迹的参考步幅。每步实际验证相对误差<=1e-3。

如果UG>C，支持该形式的外层修正有训练价值；如果UG>S且尺度受控，支持该设置下修正来源改变有价值；比较UG与g3、g2与g3再判断外层位移梯度和局部差分的作用。性能排序本身不证明Hessian中介或协同机制。

### 2. r=1是新附件与R实验的明确设置，需要单独补齐

现队列的r=.25不等于附件的r=1。若要补齐A的七组 **r=1自然步幅** 表，名义缺口为 **g0、g2、UG三条十任务训练**。

复用历史四条的前提是新增三条沿用原native数据/增强RNG、训练预算和评测协议。若沿用新component wrapper的每task/epoch配对RNG，则历史g1/g3仅作背景：为附件三组主比较及与g3连接，建议同批 **g0/g1/UG/g2/g3共5条**，其中g1/g3是协议配对控制。r=1的匹配版本可待r=.25筛选信号出现后只确认S/UG，避免立即再跑全部方向。

不把旧r1与新r.25轨迹混成严格配对。R与A预训练都核到augreg2_in21k_ft_in1k；两者分类头均为扩展SimpleLinear并复制旧行。实际差异包括dataset、seed、后续LR、learner clean pass和新数据RNG协议。跨数据集支持可分别比较相对基线趋势，不能将绝对准确率合并为同配置结果。

### 3. 在UG自己的checkpoint补同状态FD/HVP审计

当UG完整轨迹存在后，在Task0/4/9各做一次，共 **3条短诊断**。原队列6条US/Full诊断可继续复用，但不能替代UG状态的审计。

固定第一次差分和外层点theta_r，令u2为同点g2的原生归一化方向并detach。记录：

- `E=||(g3-g2)/rho - H(theta_r)u2||/(||H(theta_r)u2||+eps)`；
- `A=cos((g3-g2)/rho,H(theta_r)u2)`；
- `K=||g3-g2||/(rho*||H(theta_r)u2||+eps)`；
- 只改变第二差分数值步长h=.05/.025/.0125的forward/central FD收敛；
- 修正/base梯度范数比、两者余弦、实际BA+FC更新长度，以及显存/时间。

现curvature代码已实现E/A/K对应字段与h扫描，因此可扩到UG，不需要另建Hessian谱工程。HVP近零必须标记；恢复参数/buffers/RNG；所有比较使用同batch/增强、同损失和同参数范围。

### 4. UG有值得解释的收益后，仅替换第二次差分，做完整训练

| 配对版本 | 外层点生成 | 最终梯度 |
|---|---|---|
| FD/FD | N(g1-g0) | g0+lambda_corr*(g3-g2) |
| FD/HVP | 相同第一次FD生成规则 | g0+lambda_corr*rho*H(theta_r)*stopgrad(N(g2)) |

先固定lambda_corr=1、rho=.05、r与验证出信号的UG队列相同；beta/gamma、seed、LR、训练预算、数据序列不变。这里lambda_corr是新增修正系数，**不是ewc_lambda，也不是lambda_flat**。

不能漏乘rho；直接提交HVP相当于局部近似适用时把修正放大约20倍。外层偏移本身不穿过其生成计算图求导。两条训练轨迹分离后固定的是位置生成规则，并非跨轨迹同一个数值theta_r。

若原生attention不能二阶求导，需要math SDPA，则 **FD控制也在同math后端重训**：不能把后端变化归因于HVP精度。先同点CE/梯度和有限差分验收，再正式跑两条自然步幅配对；观察性能差异后补两条匹配有效步幅版本，可形成 **最多4条新增完整训练**。仅有离线HVP对齐不能当成这两/四条已完成。

在共同后端、数据协议和步幅目标已相同时，FD基线可以复用。若HVP不受支持则明确记录，不填造数；OOM或NaN属于失败，需要查实现/成本。

### 5. 只在有信号后扩展

- lambda_corr=1的UG无收益，不能泛化为所有强度都无用。需要时做对称小检查：`g0+lambda_corr*(g1-g0)` 与 `g0+lambda_corr*(g3-g2)`，lambda_corr=.5/2，共4个点；lambda_corr=0复用g0，1复用S/UG。不要直接更改rho同时改变第一次位置与第二次幅值。
- 第二次HVP替换值得研究，才加HVP/FD与HVP/HVP，形成位置生成×外层修正的2×2。两个新版本先分别登记，不同时改掉两个环节后只报一条结果。
- 当前seed2维持开发筛选；论文关键UG对C/S比较可在配置冻结后补seed0/1，每个seed运行C/S/UG，共6条确认轨迹。它是建议，不改变用户现有单seed队列；若宣称UG接近g3，还需把g3纳入确认对照。
- 两任务SS/XS/SX/XX与三任务条件分支用于formation/continuation/persistence因果问题。它们与FD/HVP精度问题不同，均待十任务结果后选择候选，再独立重训。

## 原始证据与复查入口

- 四方向完整原始结果：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/results.json`。
- mean与半径结果：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/results.json`。
- 原矩阵/config/receipt逐条路径与哈希：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-20261001T185744Z/review/REUSE_AUDIT.md`及JSON。
- parent进度：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z/runtime/queue_state.json`和`results/formation_endpoints.csv`。
- 已部署分量队列：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-20261001T185744Z/suite.json`、`runtime/handoff_state.json`。
- 真正更新代码：同component目录`trainer/component_horizon.py`；现HVP代码：`diagnostics/curvature_audit.py`，只做checkpoint诊断，不执行优化器训练更新。
- 历史正交残差：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z/residual_direction_analysis_20261001/REPORT.md`。
- ImageNet-R七组计划：`/data/140-1/users/liying/Flatness_CV/refine-logs/imagenetr_pure_gam_directions_20261001/EXPERIMENT_PLAN.md`；状态与汇总在同目录`queue_state.json`和`summary.csv`。

本报告的后续数量为条件实验建议；未写入运行清单、未增加或暂停任何现有GPU任务。
