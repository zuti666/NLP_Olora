# 原始结果路径

以下为匹配当前表格的33个模型/方法/Order运行。每个运行目录下有 `1-任务名` 至 `4-任务名` 四个阶段；绘图读取其中的 `predict_results.json`，训练证据为 `train_results.json` 及日志。

1B新OLoRA三个Order未定位，未列入已确认源。具体每个点的来源和SHA256见 `source_manifest.json` 与 `taskwise_accuracy.csv`。

| Model | Method | Order | 原始运行目录 |
|---|---|---|---|
| Llama-3.2-1B | EWCLoRA | 1 | [order_1](/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_1) |
| Llama-3.2-1B | EWCLoRA | 2 | [order_2](/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_2) |
| Llama-3.2-1B | EWCLoRA | 3 | [order_3](/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_3) |
| Llama-3.2-1B | FS-EWCLoRA | 1 | [order_1](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_1) |
| Llama-3.2-1B | FS-EWCLoRA | 2 | [order_2](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_2) |
| Llama-3.2-1B | FS-EWCLoRA | 3 | [order_3](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_3) |
| Llama-3.2-1B | IncLoRA | 1 | [IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | IncLoRA | 2 | [IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | IncLoRA | 3 | [IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_3/IncLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | SeqLoRA | 1 | [SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | SeqLoRA | 2 | [SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | SeqLoRA | 3 | [SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_3/SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-1B | TreeLoRA | 1 | [order_1](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama1b/full/order_1) |
| Llama-3.2-1B | TreeLoRA | 2 | [order_2](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama1b/full/order_2) |
| Llama-3.2-1B | TreeLoRA | 3 | [order_3](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama1b/full/order_3) |
| Llama-3.2-3B | EWCLoRA | 1 | [EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | EWCLoRA | 2 | [EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | EWCLoRA | 3 | [EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | FS-EWCLoRA | 1 | [EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | FS-EWCLoRA | 2 | [EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | FS-EWCLoRA | 3 | [EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993) |
| Llama-3.2-3B | IncLoRA | 1 | [IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | IncLoRA | 2 | [IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | IncLoRA | 3 | [IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | OLoRA | 1 | [OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | OLoRA | 2 | [OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | OLoRA | 3 | [OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | SeqLoRA | 1 | [SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | SeqLoRA | 2 | [SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | SeqLoRA | 3 | [SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4) |
| Llama-3.2-3B | TreeLoRA | 1 | [order_1](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama3b/full/order_1) |
| Llama-3.2-3B | TreeLoRA | 2 | [order_2](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama3b/full/order_2) |
| Llama-3.2-3B | TreeLoRA | 3 | [order_3](/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/llama3b/full/order_3) |
