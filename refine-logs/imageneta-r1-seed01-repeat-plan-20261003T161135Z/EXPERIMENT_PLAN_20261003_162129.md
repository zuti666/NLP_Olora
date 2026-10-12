# ImageNet-A r=1：seed0/seed1 完整重复实验计划

**日期**：2026-10-03 18:21 CEST。**阶段**：完成盘点、配置草案与实施计划；尚未注册运行队列或启动训练。用户已明确选择仅重复全部纯GAM对照。指定GPU池4/5/6，每卡一项、最多三项并行。

**目的**：检验原版GAM、单梯度与组合的seed2趋势是否在seed0/1保持，并区分更新方向、总幅度与探针符号的影响。主计划30个科学设置×2个新seed＝60次完整训练；逐条件复用已有seed2，最终每行seeds0/1/2共3次。保持各组历史执行协议。

## 主张与证据

|待检验主张|关键对照|证据与论文位置|
|---|---|---|
|C1：组合/外点方向相对单一g1的收益可跨seed重复|原版GAM、Sum、Mean、g0/g1/g2/g3、两种基点残差与B正交残差|同协议AAA为主，FAA/LA/BWT/遗忘辅助；逐seed差与三seed均值±样本标准差；正文方向表|
|C2：观察到的增益并非仅由名义LR或更新总幅度决定；P/C符号构造的趋势可以重复|S1.5/S2、Mean2、独立S-Gmatch、P/D/C/R自然与匹配A|分开报告自然步长和有效步长匹配；正文机制对照，完整负结果在附录|

若正增益只出现在一个seed，按不稳定结果报告；若Gmatch/LR控制消除增益，更新幅度解释得到支持。不把小遗忘而极低LA解释为良好保留，不从准确率推断Hessian因果机制。未采用LLM/VLM或新增模型部件。

## 已完成结果与去重

盘点覆盖CopyFlatCV与O-LoRA对应目录，并检查formation、component-horizon、先前半径扫描及含EWC/融合目录。31条相关seed2完整运行记录对应30个科学设置：原生g1旧direct与新审计S是同配方两次运行。30行主索引在COMPLETED_SEED2_INDEX.csv，每行绑定源配置、入口、原始指标路径与SHA。

三seed主统计预先固定使用审计S的seed2：AAA62.313192/FAA44.969；它与旧g1 raw config仅prefix/outputs_root不同，source/环境相同，审计g1_control保留原native最终梯度和SGD、不增加CE/RNG调用。旧g1 seed2=62.165531/45.132留作历史重复，不当成额外seed、不平均成合成seed2、不择优。新S seed0/1各仅跑一次，同时服务核心方向表、LR表及对应seed Gmatch参考。

未找到可直接填入本轮纯GAM表的完整十任务seed0/1。formation的seed0/1即使JSON写nb_tasks=10，实际入口是formation_task0.py，只训练Task0，不能复用。含EWC/Fisher/融合的seed0/1已完成但方法不同，按用户范围不加入。r=.25、其他rho/rank、Full参数、1/2/3-task、离线方向扫描及已取消队列均不进入本轮。

## 共同固定设置

|项目|固定值|
|---|---|
|数据|ImageNet-A，历史5981 train/1519 test；固定类别0…199，10×20类，class_shuffle=false|
|模型|ViT-B/16，augreg2_in21k_ft_in1k；冻结骨干，Q/V LoRA rank10及分类头训练；无alpha/rank缩放|
|半径|外层gam_grad_norm_rho=1；内层gam_grad_rho=.05；非adaptive|
|新增seed|0和1；初始化/数据随机流使用各自seed，类别划分不变；每项从预训练模型独立训练|
|预算|每任务50epochs，batch128，末批保留；500epochs、2550实际更新、299050样本呈现/正式run|
|优化|SGD momentum=0、weight_decay=0；基础Task0 LR .04→.004，后续 .02→.002；cosine Tmax50；指定LR倍率同时作用峰值和下限|
|纯GAM开关|lambda_flat=1；ewc_lambda=0，ewc_skip_task_fisher=true；任务内Fisher、归一化/阻尼、权重融合、SGV、额外漂移惩罚、回放关闭|
|正交操作|N01保留原GAM gamma=.1；其他组gamma=0；N09明确实施残差对g1的正交投影，属于该组的被测变量|
|数值/环境|复用Pilot_new_local及190文件冻结source；FP32、现有backend与TF32设置按各协议保留，不更换预训练权重/数据管线|

## 两套执行协议必须分别统计

N族为历史native生命周期：原生global RNG/DataLoader、四次GAM探针g0→g1+→g2→g3与额外clean CE，共5CE；mechanism_eval/grad_stats保持true、train/eval workers8；保留FP32 clean+selected−clean合成与原生unperturb。N01/Sum/g3/Mean用原生beta系数；N05–N09与LR控制使用各自历史方向hook。原GAM的梯度正交修正、B的正交残差和FP32合成运算要区分。

P族为新四探针严格恢复协议：g0→g1+→g1−→g2，共4CE；每探针从原基点精确恢复，回放同R0，最终RNG仅前进到g0的R1。每task/epoch按seed*100000+task*1000+epoch隔离采样与增强随机流，evaluation隔离RNG，eval workers4，历史机制评估关闭；范数归约与恢复运算沿用本族helper。图像transform算子同源，实际随机流不同。

因此N05的g0、N03的g1、N06的g2、N08的A不能与P族同名方向合并或省略。P族不计算g3。独立S-Gmatch与P族同状态raw g1+匹配回答不同问题，不能换参考。

## B1：原生方向核心表（9设置，新增18次）

|ID|方向/设置|已有seed2 AAA|已有seed2 FAA|新增seed|
|---|---|---:|---:|---|
|N01|original_GAM：h_plus-gamma*h_minus_perp; h_plus=g1+g3, h_minus=g0+g2, gamma=.1|68.029|55.341|0、1|
|N02|Sum：g1+g3|67.861|54.726|0、1|
|N03|g1 / S（审计基准）：g1|62.313|44.969|0、1|
|N04|g3：g3|65.993|50.679|0、1|
|N05|g0：g0|59.453|39.338|0、1|
|N06|g2：g2|63.838|49.989|0、1|
|N07|ug：g0+(g3-g2)|64.102|47.870|0、1|
|N08|A：g0+(g2-g1)|64.595|49.807|0、1|
|N09|B：g1+((g2-g1)-a*g1), a=<g1,g2-g1>/<g1,g1>|64.826|47.590|0、1|

B1各组与历史同设置比较。N01 vs N02检验原GAM正交修正；N02 vs单分支、N10检验组合及幅度；N07/N08分别检验g3−g2和g2−g1基点回加；N09检验去掉g2−g1平行分量。N09自然步长仍可能改变总幅度。

## B2：平均、学习率与有效步长（额外5设置，新增10次）

S、Sum、Graw复用B1同seed记录；不重复注册。N03/S必须保存每步实际有效位移，N14/Gmatch严格依赖同seed S完整完成。

|ID|方向/设置|已有seed2 AAA|已有seed2 FAA|新增seed|
|---|---|---:|---:|---|
|N10|Mean：(g1+g3)/2, LR x1|65.951|49.927|0、1|
|N11|S1.5：g1, LR x1.5|62.636|46.419|0、1|
|N12|S2：g1, LR x2|62.899|46.589|0、1|
|N13|Mean2：(g1+g3)/2, LR x2|68.002|54.730|0、1|
|N14|Gmatch：g3; actual effective step matched to independent same-seed S|64.683|47.877|0、1|

Mean为正常LR，Mean2为两倍LR，均是(g1+g3)/2。N14匹配独立S训练轨迹对应(task,step)的实际无缩放LoRA BA+全部seen FC联合位移；逐步核对数据/标签、RNG、类切片、参数结构与名义LR，误差门限1e−3；只执行一次真实SGD，缺失参考/无根/非有限/越限停止该运行并保留失败证据。

## B3：正反探针与差分符号（16设置，新增32次）

定义phi1±=phi±rho*N(g0)，phi2=phi+r*N(q)，O=g2，A=g0+(g2−g1对应侧)。P用g1+、q=g1+−g0；D用g1−、q=g1−−g0；C用g1−、q=g0−g1−；R用g1+、q=g0−g1+。

自然O四组、自然A四组、匹配A四组共12组；加g0 natural、g1+ natural、g1− natural、g0 matched四个基准。基准最终更新虽与r无关，但随seed变化，seed0/1均须新跑；新配置统一r1。基准seed2复用r.25已验证的半径无关结果，仅复用训练性能，不把旧r.25外点trace当r1诊断。

|ID|方向/分支/步长|已有seed2 AAA|已有seed2 FAA|新增seed|
|---|---|---:|---:|---|
|P01|g0 / P / natural|60.337|40.140|0、1|
|P02|g1_plus / P / natural|63.234|44.578|0、1|
|P03|g1_minus / P / natural|51.090|25.368|0、1|
|P04|g2 / P / natural|65.735|50.250|0、1|
|P05|g2 / D / natural|37.359|23.187|0、1|
|P06|g2 / C / natural|65.774|49.192|0、1|
|P07|A / P / natural|65.842|49.325|0、1|
|P08|A / D / natural|52.065|35.434|0、1|
|P09|A / C / natural|64.438|47.185|0、1|
|P10|g2 / R / natural|38.561|24.355|0、1|
|P11|A / R / natural|2.006|0.882|0、1|
|P12|A / P / matched|63.325|44.015|0、1|
|P13|A / D / matched|58.756|43.944|0、1|
|P14|A / C / matched|63.632|45.062|0、1|
|P15|A / R / matched|2.318|1.111|0、1|
|P16|g0 / P / matched|59.955|39.565|0、1|

P族matched的目标为候选自身当前模型状态raw正向g1+在名义LR下的实际BA+seen FC总位移；D/C也用正向g1+作目标。保留R自然/匹配负结果重复，不根据seed2低成绩删行或改变超参。

## 实施与里程碑

|阶段|工作|新增运行数|放行条件|
|---|---|---:|---|
|M0 计划（本次已做）|30条件盘点、60配置草案、去重、代码入口与seed参数化清单|0|配置差异只含seed和输出标识；P族4基准另将无效外半径.25设置为1|
|M1 实施与短测|在新目录移植薄runner/validator，按seed分组检查，保持数值helper/旧记录|60短测|native28组前2任务×1epoch=504更新；P族32组前2任务×1batch=64更新；合计568更新，scheduler Tmax50不压缩；全部通过且无跨seed参考|
|M2 原生核心|B1九设置×seed0/1|18正式|GPU4/5/6空闲与lease检查；首批N01_s0、N01_s1、N03_s0；随后优先N03_s1|
|M3 原生步长|B2五设置×seed0/1|10正式|对应seed的S完成后才放行Gmatch；其他项依计划顺序接续|
|M4 四探针|B3十六设置×seed0/1|32正式|该族两个seed各自短测全通过；四基准不能省略|
|M5 汇总|合并30条seed2与60新完成记录|0|30×3完整性/配置/指标校验，缺失不补0，失败与低成绩分开|

只使用物理GPU4、5、6，max_parallel=3；每次启动前显存<500MiB且利用率<5%，共享GPU index/UUID锁并立即重查。忙卡等待；不占用其他GPU；旧已完成/取消队列不复苏。首批卡位建议GPU4 N01_s0、GPU5 N01_s1、GPU6 N03_s0，其后动态取本池空闲卡，seed不永久绑定某张卡。

短测单独记录，不能当正式成绩。正式训练开始前应复核seed实际生效、数据/新增头初始化按seed变化；同seed配对的短测应保持各协议约定的数据/模型状态身份。测试无需跨GPU参数轨迹逐位相同，但每次自己的方向/实际SGD必须满足声明数值路径。

## 必须完成的最小代码适配

现有源码足以复用，不能只改JSON然后启动旧冻结队列。旧common/contract/validator/collector存在seed2、GPU池及job集合硬编码。后续实施在新运行目录完成：

1. N族保留src.main及实际factory路径；旧original/Sum/g3/Mean系数直接进入原生GAM，不能把这些JSON塞到固定beta=(1,1,0)的选择器runner。g0/g2/ug/A/B与LR控制必须安装相应历史hook，裸调用src.main会漏掉方向替换。
2. S/Gmatch的step_reference.py把seed2断言改成显式候选/参考同seed断言，保留所有batch/RNG/hash/预算和1e−3门限。禁止把seed2 S或另一个seed S用于新Gmatch。
3. P族native_lifecycle已支持seed0..4，其内部seed2 surrogate是兼容旧base校验的有意机制，最终恢复实际seed，不能盲目全局替换。应修正four_probe_train summary/dryrun字面seed2、validator读job.seed、builder每seed完整16行及smoke配对按seed分组。
4. 新manifest/collector按(protocol,condition,seed)唯一键统计，GPU池固定4/5/6，所有配置/数值helper/source/environment绑定新hash。旧run pins、运行矩阵、收据和日志保留。

详见IMPLEMENTATION_MAP.md与audits中的结构化源码定位。本目录REPETITION_MANIFEST.draft.json明确executable=false；60份JSON是参数草案，尚未注册运行controller。

## 指标与三seed统计

主指标AAA_task=(1/10)Σ_t[(1/(t+1))Σ_{j≤t} A_tj]；FAA_task=最终10任务平均。报告三seed原始值、mean±sample SD(ddof=1)，对每个预先指定同协议对照报告逐seed百分点差、配对均值±样本SD、胜出seed数。样本加权AAA/FAA单列，不能混用native JSON的另一AAA定义。n=3仍需保留逐seed值，不作普遍必然性结论。

同时保存LA、BWT、遗忘、10任务原始矩阵、每epoch loss/acc/LR、每任务checkpoint、实际step/样本计数。匹配组保存target/actual/relative error与缩放系数；诊断已有方向范数/cosine、LoRA/FC位移分量。原生历史机制诊断按原配置保留，新增日志只能观测、不能增加随机forward/backward改变原流程。低分完整训练保留，NaN/中断不得当有效零分。

比较不跨协议拼接；相同seed的不同方法是配对样本，同一seed两次g1重复不是两个独立seed。所有当前设置及失败分支在seed0/1结果出来前固定，避免按新结果删负例。

## 计算预算

按源seed2实际收据耗时累加：N族28次约71.64 GPUh；P族32次约66.72 GPUh；正式合计约138.36 GPUh。三卡持续空闲时理想约46.1小时，考虑短测/初始化/共享IO，排期按50–60小时；设备忙时另加等待。正式共有153000更新、17943000样本呈现；短测568更新另计。每任务保存checkpoint，启动前以旧目录大小估算磁盘余量，未设定新的收费资源。

## 完成清单

- [x] 原版GAM、单梯度、组合、Mean/Mean2/LR/Gmatch全部纳入
- [x] P/D/C/R及其同协议4基准全部纳入
- [x] 去重S/Sum/Graw；固定canonical seed2 S
- [x] 用户确认只重复纯GAM；EWC/Fisher/融合条件排除
- [x] 60份配置草案、来源SHA、正式追踪表齐备
- [ ] 新运行目录的seed-aware runner/validator实施与门禁
- [ ] GPU4/5/6正式队列注册与启动
- [ ] seeds0/1/2三seed汇总
