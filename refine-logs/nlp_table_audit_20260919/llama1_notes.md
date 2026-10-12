# Llama-3.2-1B 原始证据核对

Checked: 2026-09-19T12:31:45.884434+00:00

表中15个order值和5个Avg均与原始最终任务指标一致；每个order都有四阶段train_results、trainer_state，以及可以用train_runtime和最终predict_exact_match精确关联的成功训练console日志。每个order最终评估四任务、合计30400条样本。

|Method|O1|O2|O3|Avg|GPU×BS×GA|seed / data_seed|
|---|---:|---:|---:|---:|---|---|
|SeqLoRA|66.90|55.71|68.06|63.56|8×1×8=64|42 / None|
|IncLoRA|64.52|56.19|65.44|62.05|8×1×8=64|42 / None|
|OLoRA|58.61|53.12|58.42|56.71|8×1×8=64|42 / None|
|IncLoRA+EWC|67.19|61.92|66.54|65.22|4×1×16=64|42 / None|
|IncLoRA-GAM-EWC|70.84|70.06|69.70|70.20|5×1×8=40|1993 / 1993|

所有行：BF16 AMP，LoRA r=8、alpha=32、dropout=0.1、q/v投影，epoch=1，首任务LR=1e-3后续=1e-4，constant scheduler，max_source_length=512、max_target_length=50。SeqLoRA不增rank；其余增rank属于方法差异。

可比较范围：前三个历史基线互相对齐。新EWC对齐有效batch64、seed42、BS1、训练数据/步数，但物理DDP4/GA16不同于旧DDP8/GA8，因此不是相同随机流/逐位复现。新EWC lambda=0.5补偿旧实现中CE/GA后再加EWC带来的累计比例，不能仅比较目录lambda数字。

GAM-EWC旧结果为5卡有效batch40、seed/data_seed1993，其更新数350/125/250/100不同于batch64的218/78/156/62（按dbpedia/amazon/yahoo/agnews列）。因此当前表中EWC与GAM-EWC不是控制变量消融；GAM-EWC与前三基线也存在batch和seed混杂。

源代码版本限制：旧run未保存逐run执行源码、模型权重和环境hash，不能把当前checkout中EWCLORA的OLoRA-base规则直接套用于历史结果。旧日志打印olora_lamda_1=0.5本身不证明ewclora_gam激活了OLoRA损失。新run有冻结source、input哈希与逐rank运行时审计。

逐单元格完整路径、日志行号、指标SHA256见 llama1_evidence.json。
