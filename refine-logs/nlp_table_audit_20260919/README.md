# NLP 表格、配置与训练证据核查

核查日期：2026-09-19。查找范围包括当前 O-LoRA 仓库及用户所指 CopyFlatCV/GAM-EWC/NLP_O-LoRA 导出包；导出包仅保存日志摘录，没有额外原始训练 console。范围是用户表中五个实际启用的方法；注释掉的 SeqFT/EWC/LwF/L2P 不在本次结果核查范围。以下 EWC 指 IncLoRA+EWC，GAM-EWC 指 `ewclora_gam`，不混入 SAM+OLoRA+EWC 或 FisherShape 变体。

**Llama 表中 30/30 个 Order 值、10/10 个 Avg 均匹配。30 个 Order 都能关联成功训练 console；合计 120 个阶段有对应 train_results 和 trainer_state。数值来源完整，不代表全表配置一致。**

独立复核进一步重算了全部Llama结果的912,000条原始预测，并逐条核对测试集标签/文本，全部匹配。审查报告中“FAIL”针对严格受控的五方法比较和完整T5-large覆盖这两个条件；表内Llama数值转录及原始预测重算为PASS，未发现伪造标签或按模型输出归一化得分。

新 Llama-1B BF16 EWC 已完成：12/12 正式阶段，FAA=67.194079/61.917763/66.536184，Avg=65.216009（显示65.22）。完成时间2026-09-18 20:20:14 UTC。检查时无本项目训练进程，7张 RTX6000 Ada 均0%利用率、各2MiB显存；见 [运行快照](runtime_status.json)。

本表的新1B EWC来自 `experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z`。之前的五卡BF16/batch40/seed1993队列 `experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z` 另有三order结果59.31/56.84/67.23、Avg61.13，保持单独记录；更早的FP32补跑也未混入本表。CopyFlatCV原计划中的“尚未启动”是制定计划时的状态，本次运行状态以新队列state.json为准。

|模型|方法|GPU数|每卡batch|GA|有效batch|seed|data_seed|精度|
|---|---|---:|---:|---:|---:|---:|---|---|
|Llama-1B|SeqLoRA / IncLoRA / OLoRA|8|1|8|64|42|None，回退seed42|BF16|
|Llama-1B|新 IncLoRA+EWC|4|1|16|64|42|None，回退seed42|BF16|
|Llama-1B|旧 IncLoRA-GAM-EWC|5|1|8|40|1993|1993|BF16|
|Llama-3B|SeqLoRA / IncLoRA / OLoRA|8|1|8|64|42|None，回退seed42|BF16|
|Llama-3B|IncLoRA+EWC / IncLoRA-GAM-EWC|8|1|8|64|1993|1993|BF16|

共同的 Llama 设置：LoRA r8、alpha32、dropout0.1、q_proj/v_proj；epoch1；首任务 LR1e-3，后续1e-4，constant且warmup0；source/target长度512/50。四任务均7600测试样本，最终每order评估30400样本。FAA为最终四任务exact-match百分数的均值；Avg为三个order的均值，不是多seed均值。

比较范围：

- 两个 Llama 模型各自的前三种基线，实际batch、seed、精度、LR、epoch相互对齐，可做相同记录协议下的单seed比较。
- 新1B EWC对齐前三基线的有效batch64、BS1、seed42和训练更新数，但物理DDP4/GA16与旧DDP8/GA8不同；随机数分配及归约路径不相同，不能声称逐位复现。新lambda0.5补偿旧代码先CE除以GA再加EWC的累计比例，对齐旧GA8×lambda1的名义强度。
- 当前1B EWC与GAM-EWC的batch和seed都不同。GAM-EWC每任务更新数350/125/250/100，batch64队列为218/78/156/62（DBpedia/Amazon/Yahoo/AGNews）。不能将70.20−65.22=4.98个百分点单独归因为GAM。
- 3B EWC与GAM-EWC两者的记录协议一致，可比较这组seed1993结果（70.19→72.96）；它们与前三基线的seed/data_seed不同，不构成严格配对的控制变量比较。
- 历史run没有保存逐run执行源码及基座权重hash。历史候选源码与五月日志匹配，但不能据当前checkout倒推旧EWC是否启用了OLoRA正交项。详见 [3B说明](llama3_notes.md)。所有结果仅单seed，不能据此宣称多seed稳定提升。

现有T5结果（FAA%，只对完整三order计算Avg）：

|方法|small O1|small O2|small O3|small Avg|large O1|large O2|large O3|large Avg|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|SeqLoRA|34.32|44.26|50.22|42.94|56.40|66.24|69.88|64.17|
|IncLoRA|48.01|57.84|55.46|53.77|—|77.39|70.37|—|
|OLoRA|62.45|61.68|57.36|60.49|77.59|76.50|77.55|77.21|
|IncLoRA+EWC|44.00|57.77|55.07|52.28|71.57|72.57|71.46|71.87|
|IncLoRA-GAM-EWC|43.91|53.70|56.47|51.36|64.71|67.14|71.60|67.82|

**T5-small：15/15有实际结果；T5-large：14/15有实际结果。** 全部真实结果均有四阶段train_results、trainer_state和adapter；各最终JSONL已由独立信息核查代理重算，均每任务7600样本并与对应JSON指标匹配。T5-small的全部标签和原文另外与当前CL_Benchmark测试集逐条匹配。

T5训练日志与配置限制：

- T5-small：五方法三个order均未在当前仓库找到对应成功训练console。能由状态和adapter确认epoch1、LR1e-3、r8/alpha32/dropout0.1/q,v；更新数与batch64一致，但不能唯一推回GPU数、BS、GA，更不能从目录名证明训练seed、BF16和长度。后验评估日志do_train=False，只能证明评估设置。
- T5-large：SeqLoRA O1和EWC/GAM-EWC全部order共7个有完整训练console；其余7个真实baseline结果有训练状态和模型，但对应训练console未找到。EWC/GAM-EWC为8×BS8×GA1=64，seed/data_seed1993，BF16，epoch1，LR1e-3，长度512/50，EWC lambda/gamma=1/1。GAM rho=0.02、norm_rho=0.2、grad_gamma=0.03。SeqLoRA O1为4×BS16×GA1=64，其余上述通用设置同样有日志证实。
- T5-large IncLoRA O1缺失。现存T5large命名目录实际base=t5-small，并且与对应T5small目录四阶段全部adapter tensors和最终预测完全相同；不得算作large或独立重复实验。完整JSON中的runtime字段有差异，详见 [重复目录证据](t5small_evidence.json)。
- T5-large EWC O1还存在4×BS16×GA1=64的73.10结果。本表使用与O2/O3/GAM队列一致的8×BS8×GA1=64之71.57，不混用物理batch拆分不同的run，也不择优替换O1。
- T5三种旧基线的最终指标常来自最终adapter的四份evalFlat后验评估；EWC/GAM来自第四阶段原位评估。两者均覆盖四任务，不能用后验评估中的epoch0断言未训练；是否完全相同的历史推理实现仍受源码版本缺失限制。

现存T5结果不支持普遍增益：small的GAM-EWC平均51.36低于EWC52.28；large为67.82低于71.87。small训练配置缺证据，不能把差异单独归因为算法；large EWC/GAM记录配置对齐，但也只是单seed结果。

建议修订：保留Llama数值，统一方法命名为IncLoRA+EWC / IncLoRA-GAM-EWC，并在caption或Appendix逐组注明batch、seed及新1B硬件拆分。若目标是严格五方法对比，优先补齐1B GAM-EWC的batch64/seed42协议及3B EWC/GAM与基线相同seed的结果；T5-large补真正的IncLoRA O1。T5缺失训练console的配置项应标未知或在新跑时完整留存。

证据索引：

- [Llama逐单元格CSV](llama_table_evidence.csv)：30行，原始指标、训练console路径、实际配置、SHA256。
- [Llama均值核对CSV](llama_average_evidence.csv)：10行。
- [1B逐阶段证据](llama1_evidence.json)；[1B说明](llama1_notes.md)。
- [3B逐阶段证据](llama3_evidence.json)；[3B说明](llama3_notes.md)。
- [T5逐单元格CSV](t5_table_evidence.csv)；[T5覆盖汇总](t5_coverage_summary.json)。
- [T5-small证据](t5small_evidence.json)；[T5-small说明](t5small_notes.md)。
- [T5-large证据](t5large_evidence.json)；[T5-large说明](t5large_notes.md)。
- [独立完整性复核](EXPERIMENT_AUDIT.md)：同系列fresh reviewer，语义判断按技能标注provisional；数值存在/一致性为确定性检查。
