# 两轮实验分组索引

| 结果组 | 实际卡数 | 每卡 batch | GA | 有效 batch | seed | CLI EWC λ | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧 BF16 batch40/seed1993 | 5 | 1 | 8 | 40 | 1993 | 1.0 | 已完成，单独保留 |
| 新 BF16 batch64/seed42 | 4 | 1 | 16 | 64 | 42 | 0.5 | 已完成，12/12正式阶段通过 |

旧结果目录：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z`。新结果目录：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z`。

新运行原 PID：`71236`，已退出并释放 GPU。正式三个 Order 从原始基座分别开始；两轮 checkpoint、原始预测和均值互不混用。新 CSV 和 LaTeX 名称均包含 `bf16_b64_s42_ddp4_bs1_ga16_lam0p5`。

旧 SeqLoRA/IncLoRA/OLoRA 对照是8卡×BS1×GA8，batch64、seed42。当前只有7张物理卡，因此新运行用4卡×BS1×GA16对齐有效batch和seed；并非原样8卡复现。λ0.5补偿现有实现中的GA缩放，记录参考GA8/λ1。
