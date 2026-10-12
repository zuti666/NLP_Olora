# 3B SeqLoRA / IncLoRA 历史 seed 复核

检查时间：2026-09-26T06:17:01.189496+00:00

历史 seed42 结果存在。当前队列中的 seed42 是四卡迁移复跑，并非历史缺失。

| 方法 | seed | Order1 | Order2 | Order3 | Avg | 历史配置 |
|---|---:|---:|---:|---:|---:|---|
| SeqLoRA | 42 | 67.86 | 67.26 | 67.40 | 67.51 | DDP8 × batch1 × GA8 = 64 |
| IncLoRA | 42 | 65.23 | 66.74 | 69.55 | 67.17 | DDP8 × batch1 × GA8 = 64 |

本轮新训练为 DDP4 × batch1 × GA16 = 64。相同有效 batch 不等于逐步数值行为或源码完全一致。

按最小补缺口径，以上两方法需要新增 seed0 和 seed1993 的三个Order；按严格四卡口径，需保留 seed42 迁移。

截至本次检查，新 seed0：SeqLoRA三个Order已验证；IncLoRA Order1/2已验证，Order3 stage2执行中。两方法 seed1993 仍待运行。

目前仅建立本复核索引，未改动队列、训练配置、主表或历史结果。

## 原始指标与日志
- IncLoRA / Order1 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log)
- IncLoRA / Order2 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-yahoo/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log)
- IncLoRA / Order3 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-dbpedia/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log)
- SeqLoRA / Order1 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log)
- SeqLoRA / Order2 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-yahoo/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log)
- SeqLoRA / Order3 / seed42：
  - [预测指标](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-dbpedia/predict_results.json)
  - [训练日志](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log)
