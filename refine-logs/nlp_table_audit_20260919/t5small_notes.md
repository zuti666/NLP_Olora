# T5-small 原始证据核查（2026-09-19）

五类方法均有三个顺序的完整原始结果。15 个非重复 run 各有 4 阶段 train_results.json、trainer_state.json 和 adapter。全部最终 JSONL 已重新计算 exact match，并与 CL_Benchmark 的 test.json 按样本 ID 核对标签和文本。

| 方法 | Order-1 | Order-2 | Order-3 | Avg. |
|---|---:|---:|---:|---:|
| seqlora | 34.32 | 44.26 | 50.22 | 42.94 |
| inclora | 48.01 | 57.84 | 55.46 | 53.77 |
| olora | 62.45 | 61.68 | 57.36 | 60.49 |
| ewc_only | 44.00 | 57.77 | 55.07 | 52.28 |
| ewc_gam | 43.91 | 53.70 | 56.47 | 51.36 |

注意：现存 T5-small 成功训练的控制台日志没有找到，不能把 posthoc eval log 当作训练 log。SeqLoRA / IncLoRA / OLoRA 的最终数值来自 final adapter 的 4 份 evalFlat 文件；EWC / GAM-EWC 来自第四阶段原位评估的 predict_results.json。

可以确认：四阶段训练 epoch=1、LR=0.001，更新步数按任务为 DBpedia 219 / Amazon 79 / Yahoo 157 / AGNews 63；adapter 验证实际基座为 t5-small，r=8、alpha=32、dropout=0.1、target_modules=q,v。步数与有效 batch 64 一致，但无法唯一确定物理卡数、每卡 batch 和 GA。目录 seed-1993、Ba16-1/Ba8-1、以及归档 BF16 只是声明，实际 training seed/data_seed、训练精度和长度配置未被运行日志证实。

因此这些数值可列为现存 T5-small 结果；尚不足以声明五方法的全部实际训练配置一致。后验评估日志（例如 SeqLoRA O1）显示 do_train=False、BF16、512/50、eval batch=64、单卡，不能用于证明训练设置。

重复目录：Order-1 的 IncLoRA_T5large_..._redo1_seed-1993 其 adapter base 是 t5-small；它和 T5small 同名设置的全部最终 raw prediction 哈希一致；全部阶段 adapter tensor 也逐元素一致。train_results / trainer_state / metric 文件中的运行时间字段不同，所以这些完整文件的哈希不同。不能将其视为独立种子结果，且不能列入 T5-large。

详细逐 run 证据、哈希、原始路径与精确计数见 [t5small_evidence.json](t5small_evidence.json)。
