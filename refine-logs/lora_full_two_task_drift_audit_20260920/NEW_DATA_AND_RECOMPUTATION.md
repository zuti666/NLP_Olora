# 较新数据、今日新端点与参数移动重算

2026-09-20。本轮只读取实验产物、CPU重算参数差和整理分析；没有启动GPU、修改训练代码或干预已有后台队列。下文区分此前已有实验与核查期间刚完成的两条训练。代码task0/task1分别为论文第一个/第二个任务。

## 结论

**有较新结果，并且已经完成了可恢复模型的权重级重算。不能笼统说LoRA移动算错，也不能把所有full结果都判为包含新分类头。**

| 资产 | 是否有结果 | 参数移动的结论 |
|---|---|---|
| 旧May ImageNet-R LoRA六行表 | 18份原日志完整，未保存匹配权重 | 已核验日志与公式；fresh分支的合并前有效移动等于新BA，未发现算术错误；无法重新加载原权重核验 |
| 今日ImageNet-A：SGD→SGD、GAM→SGD | 12:39 UTC两条训练完成，seed0；专门端点评估尚未运行 | 本轮从完整anchor/endpoint重算通过，r=0.374950/0.349876 |
| 此前ImageNet-A：SGD→SGD、GAM→GAM十任务 | 20个完整checkpoint | 既有CPU核验通过，t0→t9 r=1.197584/1.400473；后续优化器不同 |
| Aug ImageNet-R ratio-FS六行 | 六变体×三seed，端点sidecar存在 | 本轮重算EWC与仅Task1 shaping共6对，r均值0.427166/0.448335 |
| 旧full六行表 | 日志、final存在；旧anchor覆盖 | 总范数可复算日志，纯backbone只能条件分解，不能完整恢复原权重差向量 |
| 另一套ImageNet-R full完整序列 | 20个完整checkpoint | 本轮重算18个相邻步与累计距离；日志本来已排除头，t0→t1 r=2.041041/2.602965 |
| ImageNet-A full历史训练 | 确有结果，部分日志有非有限loss | 未找到这些run匹配权重，不能做权重级位移重算；不同于上述R full序列 |

## 1. 计算原则：从有效权重相减，不能机械地套BA或delta_norm

正确的共享backbone位移为 `||W_eff,end − W_eff,start||`。固定预训练基底会抵消；只有当基底也训练时才需加上它的变化。A/B因子距离不是同一指标。

- 旧fresh adapter：新任务B=0、新BA起点=0，旧分支固定，因此该任务合并前差值等于新BA。
- 新CopyFlatCV连续adapter：第二任务开始时BA已经训练过，**不为0**；应算 `BA1−BA0`，不能只算最终BA1。
- 新ImageNet-R机制 `mechanism.steps.1.delta_norm` 是**平方范数**；ImageNet-A专用 `task0_trajectory.steps.t.delta_norm` 已经是**范数**。
- Full新序列的机制字段是**相邻任务**backbone平方位移；t9字段是t8→t9，不是t0→t9。累计位移需从两个对应endpoint直接相减，不能把每步norm求和当净位移。

## 2. 今日刚完成的ImageNet-A关键pair：已经重算

读取时两个训练均于2026-09-20 12:39 UTC完成，COMPLETE receipt有效。共同后续训练为SGD+EWC，关闭后续shaping（lambda_flat=0），rank10、lambda2000、LR=.02、50epochs、batch128、momentum/weight_decay0，TF32关闭，seed0。两条恢复各自Task0模型及对应EWC保护状态，未来训练规则一致；相同lambda不意味着保护矩阵相同。

本轮从完整 `task_00.pt`、`task_01.pt` 计算24块Q/V的有效差，确认全部非LoRA backbone不动、旧20类head逐位相同。与训练机制记录的平方位移最大误差1.48e-9；完整两端SHA已保存。

| 今日seed0分支 | a11（当前FP32） | a21（当前FP32） | a22（trainer） | BWT（当前FP32） | r（CPU有效权重） | 旧任务test local CE增量 |
|---|---:|---:|---:|---:|---:|---:|
| SGD Task0→SGD+EWC |85.667752|84.364821|74.80|-1.302932|**0.374949571**|-0.019245973|
| GAM Task0→SGD+EWC |87.622150|84.039088|72.05|-3.583062|**0.349875797**|-0.013388091|

a11必须用本次FP32重评：trainer矩阵第一行继承历史SGD 85.34，与当前重评85.667752不同；混用会污染BWT。a21使用40类seen-head，local CE仅旧20类；两个目标不同，因此local loss改善与class-IL accuracy下降可以同时发生。a22仅保留trainer显示的两位精度。

**本次shaped起点的移动略小，但a21/BWT和新任务a22并未更好；local CE相对自身起点的改善也较小。** shaped endpoint绝对local CE为.400233，低于普通起点分支的.423178，但它自身起点已经更低（.413621 vs .442424）。因此不能仅凭最终loss较低称其损伤更少。另一个指标seen-head CE增量为.119736 vs .156537，方向又不同，必须固定评价目标后讨论。此结果是单seed，尚不能推断普遍统计效应。

训练hook已完成上述accuracy/loss测量；专门A endpoint raw Fisher等评估仍被pipeline阻断，不能说完整A机制表已完成。

路径：

- 本次CPU数值：[today_imageneta_pair_recomputed.csv](today_imageneta_pair_recomputed.csv)。
- 精确anchor、endpoint、metrics、配置及SHA：[today_imageneta_pair_provenance.json](today_imageneta_pair_provenance.json)。
- 两个今日run：`/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/training/A_{sgd,gam}_to_sgd_seed0/`。
- 每个run内：`checkpoints/task_01.pt`、`run_manifest.json`、`task0_local_metrics.jsonl`、`COMPLETE.json`。
- 重算脚本：[recompute_today_a_lora_cpu.py](recompute_today_a_lora_cpu.py)，只需CPU。

## 3. 新版ImageNet-R LoRA：数量级差异已改变

Aug-2026 ratio-FS、seeds3/4/5，SGD LR.02、lambda1.2e6、归一化并damping的保护矩阵、持续同一adapter。算法与旧Adam/fresh-KV六行表不同，也并非换数据集的单因素实验。

| 变体 | a11均值 | a21均值 | a22均值 | BWT均值 | r均值 |
|---|---:|---:|---:|---:|---:|
| Vanilla |89.36|84.59|88.89|-4.77|0.850642|
| ratio-FS only |89.47|84.80|89.54|-4.66|0.856600|
| EWC |89.36|85.90|87.82|-3.46|0.427166|
| ratio-FS＋EWC |89.47|85.74|88.19|-3.72|0.461659|
| 仅Task1 ratio-FS＋EWC |89.47|85.96|87.82|-3.51|0.448335|
| 仅Task2 ratio-FS＋EWC |89.36|85.59|88.38|-3.78|0.456989|

六行均来自18个原始JSON重新聚合；本轮权重级重算范围为EWC和仅Task1 shaping六对。其r分别为 `0.427165612±0.001369011`、`0.448335488±0.001936117`（均值±样本标准差），最大平方误差7.59e-9，各自anchor的A/B/FC与parent SHA匹配，旧head均未变。

本组EWC和仅Task1 shaping的平均BWT为-3.4633/-3.5100，**没有显示平均遗忘改善**。不能只沿用旧六行表中Task1 shaping的正向结论。

这里控制组平均 `||BA0||=.782015`、`||BA1||=.991294`，正确差分范数是.427166。如果误认为每个新任务BA重新归零，直接用最终BA，就会把.427错算成.991。

来源与计算：[NEW_LORA_RESULTS.md](NEW_LORA_RESULTS.md)、[六行汇总](new_imagenetr_ratio_summary.csv)、[逐seed权重重算](new_r_lora_drift_recomputed.csv)、[配对来源](new_r_ratio_pair_assets.json)。原实验根为 `/data/140-1/users/liying/CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/`。

## 4. Full到底哪里有问题，哪些已重算

旧full六行的“含新头的零扩展参数距离”可以作为一个数学定义，但若称为共享backbone训练移动、再与LoRA有效更新比较，**解释口径不成立**。需要拆成backbone、旧分类头变化、新分类头真实初始化→终点变化；新头相对0的范数必须另标。

一个直接例子来自本轮新R LoRA control seed3：有效backbone移动为 **.428629**，新头相对0的范数为 **4.655879**。如果也按旧full口径把新头计入，总距离就变为 **4.675567**。仅仅统计范围不同，就能把同一个LoRA运行从.429写成4.676。这个数是口径演示，不能替换有效backbone指标。见 [classifier scope comparison](new_r_lora_classifier_scope_comparison.csv)。

旧full六行能否重算：

1. 原日志总范数已核验，开平方正确。
2. 现存final有新头，可在“final对应旧日志且旧头不动”的条件下扣除新头，得到此前1.51–2.66的backbone标量估计。
3. 本轮进一步检查六个final，只有当前模型，没有旧参数快照；原日志也无按层分解。因此**不能绕过被覆盖的anchor，完整验证旧backbone差向量、共同子空间和路径**。不要把新实验的checkpoint拿来修补旧表。详见 [FULL_HISTORY_RECOVERY.md](FULL_HISTORY_RECOVERY.md)。

另一个较新的ImageNet-R full序列，Task0–Task9的20个完整checkpoint全部存在，历史记录已经明确不含FC。本轮20个模型、18步直接CPU重算与日志最大平方误差4.79e-8，旧head均精确不动：

| 路径 | 普通SGD线backbone r | GAM线backbone r | 普通线旧Task0 CIL acc | GAM线旧Task0 CIL acc |
|---|---:|---:|---:|---:|
| t0→t1 |2.041041|2.602965|80.97|85.53|
| t0→t9 |5.571344|7.605112|60.22|57.39|

起点accuracy分别92.92/93.71。两任务时GAM移动更大、旧accuracy下降更小；到t9最老Task0反而下降更多，必须保留这一反例。该系列后续仍为GAM/SGD各自继续，且采用NCM batch Fisher保护，不是今日ImageNet-A的同未来SGD对照，也不是旧full六行表。

原始根：`/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c_fullmodel/`。报告及全部路径：[NEW_FULL_RESULTS.md](NEW_FULL_RESULTS.md)、[重算CSV](new_full_checkpoint_drift.csv)、[模型身份清单](new_full_checkpoint_manifest.json)。CSV中的K/V投影仅对应旧fresh-KV表的坐标；新版LoRA用Q/V，不能直接把该K/V当新版共同坐标。主表使用全共享backbone有效位移。

ImageNet-A full则确有6–7月历史accuracy结果，所查79份记录中69份十任务矩阵完整，但其中10份日志含NaN/Inf，不等于69次有效成功。未找到这些run的配对权重，故此轮不能从权重重算它们。完整范围见 [inventory](new_full_results_inventory.csv)。

## 5. 今天新增诊断的完成范围

截至pipeline记录2026-09-20 12:39:07 UTC，两个sanity和两条训练完成；总体failed。邻域SGD因 `Runner.measure` 的 `direction` 参数重复传入报错，另一方法邻域取消，后续专门A endpoint、bridge、trajectory/path与汇总被阻断。此轮未修改该队列或修复其代码。

已完成r=0初始点诊断：同256个确定性旧训练样本、旧20类头、24个有效Q/V下，未归一化逐样本raw Fisher trace为SGD **1465.496927**、GAM **761.900774**。这证明该协议下起点统计不同，但没有完成邻域与路径曲线。结合今日新pair的accuracy/BWT，不能从起点trace较低直接推出后续旧任务保留必然更好。

精确测量：`/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/results_seed0/{sgd,gam}/anchor_measurement.json`。该组sanity不是训练终点Fisher；Task1 endpoint的raw Fisher尚未测量。

## 6. 建议采用的表格口径

正文使用共享backbone有效权重差的L2范数，明确起止任务、坐标及是否排除分类头；另列平方范数和头的变化供审计。三seed逐seed求norm再平均；seed0不包装为统计证据。新旧结果按算法/数据/精度分别展示，不把现存反例删除或把缺失测量补成结论。

本轮已经完成上述关键端点的CPU复算。原六行缺失历史资产及未完成的GPU测量属于不同问题，不能靠对现有数字再开方或改列名解决。
