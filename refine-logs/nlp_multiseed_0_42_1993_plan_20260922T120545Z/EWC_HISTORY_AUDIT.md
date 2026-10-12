# EWC / FS 历史多 seed 审计（2026-09-22）

当前主表的 EWC 与 FS 各模型均只存在一个有效 seed。未发现额外 seed0 主表方法完整结果。已检查每 Order 的四阶段预测、train_results、trainer_state 及日志中的 TrainingArguments；seed 不由文件名推断。

|模型|方法|seed|Orders|4卡协议可复用|历史8卡名义batch64|需补seed|
|---|---|---|---|---|---|---|
|1B|EWCLoRA|42|1,2,3|是，GA16/λ0.5|—|0,1993|
|1B|FS-EWCLoRA|1993|1,2,3|是，GA16/λ1|—|0,42|
|3B|EWCLoRA|1993|1,2,3|否|有，GA8/λ1|0,42；严格4卡另补1993|
|3B|FS-EWCLoRA|1993|1,2,3|否|有，GA8/λ1|0,42；严格4卡另补1993|

在这两个方法合计36个 model×method×seed×order 目标单元中：12个有当前表的 batch64 结果，6个为严格4卡结果。名义batch64复用方案需补24个order；统一4卡方案需补30个order。8卡和4卡的相同有效batch不是精确训练协议等价，尤其GAM按rank本地扰动/累积窗口改变。

旧 EWC λ0.5 为 GA16 对旧 GA8/λ1 的缩放补偿；FS 的 GAM closure 均值/方向投影机制不能套用该λ补偿。3B改4卡时需由训练语义审计明确纯EWC λ，不能只复制旧λ1。

## 所有候选运行与判定

|模型|实际方法|seed日志值|Order|四阶段完整|FAA|判定|目录|
|---|---|---|---|---|---|---|
|Llama-3.2-1B|EWCLoRA|42|1|是|67.1941|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_1`|
|Llama-3.2-1B|EWCLoRA|42|2|是|61.9178|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_2`|
|Llama-3.2-1B|EWCLoRA|42|3|是|66.5362|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_3`|
|Llama-3.2-1B|FS-EWCLoRA|1993|1|是|69.8454|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_1`|
|Llama-3.2-1B|FS-EWCLoRA|1993|2|是|68.125|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_2`|
|Llama-3.2-1B|FS-EWCLoRA|1993|3|是|70.6184|selected_4gpu_reusable|`/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/full/order_3`|
|Llama-3.2-3B|EWCLoRA|1993|1|是|68.3585|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|EWCLoRA|1993|2|是|69.1776|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|EWCLoRA|1993|3|是|73.0395|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|FS-EWCLoRA|1993|1|是|72.7138|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|FS-EWCLoRA|1993|2|是|73.5132|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|FS-EWCLoRA|1993|3|是|72.6612|selected_8gpu_nominal_batch64_only|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-1B|EWCLoRA|1993|1|是|59.3125|excluded_batch40|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/EWCLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-1B|GAMFisherShape-SAM-OLoRA-EWC|1993|1|是|67.25|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-1B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-1B|FS-EWCLoRA|1993|1|是|70.8355|excluded_batch40|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/EWCLoRA-GAM_Llama-3.2-1B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-1B|GAMFisherShape-SAM-OLoRA-EWC|1993|2|是|61.3388|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-1B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-1B|EWCLoRA|1993|2|否|—|failed_no_metrics|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/EWCLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA16_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-1B|FS-EWCLoRA|1993|2|是|70.0592|excluded_batch40|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/EWCLoRA-GAM_Llama-3.2-1B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-1B|GAMFisherShape-SAM-OLoRA-EWC|1993|3|是|67.9375|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_3/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-1B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-1B|FS-EWCLoRA|1993|3|是|69.7007|excluded_batch40|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_3/EWCLoRA-GAM_Llama-3.2-1B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993`|
|Llama-3.2-3B|GAMFisherShape-SAM-OLoRA-EWC|1993|1|是|73.2105|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-3B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-3B|GAMFisherShape-SAM-OLoRA-EWC|1993|2|是|72.7599|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-3B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-3B|GAMFisherShape-SAM-OLoRA-EWC|1993|3|是|68.8158|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/GAMFisherShape-SAM-OLoRA-EWC_Llama-3.2-3B_Lora8_Ba1-GA64-GB64_E1_LR1e-3_to_1e-4_rho0.05_OL10.5_OL20.0_lambda1.0_gamma1.0_rawFisher_seed-1993`|
|Llama-3.2-1B|EWCLoRA|1993|1|是|59.3125|excluded_batch40|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z/full/order_1`|
|Llama-3.2-1B|EWCLoRA|1993|2|是|56.8421|excluded_batch40|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z/full/order_2`|
|Llama-3.2-1B|EWCLoRA|1993|3|是|67.2303|excluded_batch40|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_ddp5_20260918T145252Z/full/order_3`|
|Llama-3.2-1B|EWCLoRA|1993|2|是|61.6744|excluded_precision_batch|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_supplement_20260916T213647Z/full/order_2`|
|Llama-3.2-1B|EWCLoRA|1993|3|是|64.454|excluded_precision_batch|`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_supplement_20260916T213647Z/full/order_3`|
|Llama-3.2-1B|EWCLoRA-SAM-via-FisherShape|42|1|是|68.9013|excluded_different_method|`/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/full_gpu6_20260811_seed42/order_1/EWCLoRA-SAM-via-FisherShape_Llama-3.2-1B_Lora8_Ba1-GA64_E1_LR1e-3_to_1e-4_rho0.05_lambda1.0_gamma1.0_normfalse_damp0.0_seed-42`|

## 排除与去重

- 1B 旧 EWC seed1993 和旧 FS seed1993 的 batch40 结果不计入 batch64 多 seed。
- FP32/单卡GA40/λ0.2的EWC补充Order2/3不计入。CopyFlat目录里的8份prediction文件与该运行SHA256逐个一致，是导出副本，不是新增seed。
- EWCLoRA-SAM-via-FisherShape seed42（Order1，GA64）是不同优化器/方法，不能补作当前FS seed42。
- GAMFisherShape-SAM-OLoRA-EWC seed1993 的两模型三orders为不同方法，不能补作当前FS。
- 启动失败、smoke、reload-eval、v1/v2/v3包装器重试均不构成独立seed。
- 正式现有主表12个候选的四阶段结果完整性与日志参数证据见JSON runs；历史源码没有完整运行哈希的部分仍保留该限制。

## 证据

- 当前参数：`/data/140-1/users/liying/O-LoRA/figures/llama_taskwise_orders_20260920/combined_v6/protocol_configs.csv`
- 当前来源：`/data/140-1/users/liying/O-LoRA/figures/llama_taskwise_orders_20260920/combined_v6/run_sources.csv`
- 逐阶段指标路径、SHA256、训练步数、训练耗时匹配、seed/data_seed、BF16、GA、优化器和所有排除目录详见 `/tmp/nlp_multiseed_ewc_history.json`。
