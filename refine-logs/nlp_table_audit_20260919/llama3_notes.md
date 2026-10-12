# Llama-3.2-3B 原始训练日志与表格核验（2026-09-19）

范围：用户表格中 5 方法 × 3 顺序 = 15 个结果单元。逐一读取原目录中的 train_results.json、trainer_state.json、predict_results.json 和成功 console 日志；未用历史汇总表替代原始数字。

**结论：15/15 个表格数字及 5/5 个均值均匹配；15/15 有完整对应训练 console 日志，60/60 个阶段的 train_runtime 与 train_loss 精确匹配 console，trainer_state 的终步/运行时间匹配，四阶段预测数字也逐一匹配 console。不是只有后期评估日志。日志均无 traceback。**

**配置：所有 3B 行都是 8 卡 × 每卡 1 × GA8 = 有效 batch64，BF16；基础三方法 seed42/data_seed=None，而 EWC 与 GAM-EWC seed1993/data_seed1993。因此 batch、主要训练日程对齐，但不是严格相同 seed 的控制对照。**

| 方法 | O1 原始 FAA | O2 原始 FAA | O3 原始 FAA | 原始均值 | 对应成功训练阶段 |
|---|---:|---:|---:|---:|---:|
| SeqLoRA | 67.8553 | 67.2632 | 67.4046 | 67.507700 | 12/12 |
| IncLoRA | 65.2303 | 66.7434 | 69.5493 | 67.174333 | 12/12 |
| OLoRA | 63.3651 | 62.9737 | 63.9243 | 63.421033 | 12/12 |
| IncLoRA+EWC | 68.3586 | 69.1776 | 73.0395 | 70.191900 | 12/12 |
| IncLoRA+EWC+GAM | 72.7138 | 73.5132 | 72.6612 | 72.962733 | 12/12 |

共同配置：LoRA r=8，alpha=32，dropout=0.1，target_modules=q_proj/v_proj；AdamW-HF β=(0.9,0.999)、eps=1e-8、weight_decay=0、max_grad_norm=1；每任务名义 1 epoch，首任务学习率 1e-3、后续任务 1e-4，各阶段 constant scheduler、warmup=0；source length512、target/generation length50。最终评估均为 30,400 样本。

四任务 train_samples/global_step 一致：DBpedia 14000/218，Amazon 5000/78，Yahoo 10000/156，AGNews 4000/62。末尾未凑满的 GA 窗口导致实际 epoch 报 1.00 或 0.99，不能把目录 E1 解读为绝对每样本均更新一次。

EWC 配置 λ=1、γ=1、fisher_max_batches=None、project_eps=1e-12。两种 EWC 方法均保留全部 12 个阶段 Fisher 文件，各顺序日志有 3 次加载、4 次保存的链条。GAM-EWC optimizer_type=gam-adamw_hf、grad_rho=.02、grad_norm_rho=.2、grad_gamma=.03、beta1/2/3=1、adaptive=False、reduce=mean、perturb_eps=1e-12。

历史普通基线运行时间为 2026-04-05/06；EWC/GAM-EWC 为 2026-05-18/19。日志指向同一原始 3B 模型路径，并保存训练设置，但没有发现逐 run 固定的源码/环境 hash。

## 历史方法语义

Git 版本 `898adf6bb13eb6efc00e31764eb2bcabd014498d`（2026-05-16）的 trainer 行 214/388/947/2301 与五月成功日志中的相同行号和文案吻合，支持将其作为历史实现解释，尚不能取代逐次运行源码 hash。`git show 898adf6bb13eb6efc00e31764eb2bcabd014498d:src/uie_trainer_lora.py` 可复核。

该版本 EWCLoRA 为 IncLoRA 增量分支加 full-Delta-W Fisher 惩罚（trainer:250–266, 2296–2305）。GAM-EWC 在 CE 闭包之外缓存 EWC 梯度，再做 per-parameter GAM/EWC 方向投影（trainer:405–439,1826–1854）。没有把 OLoRA 正交惩罚加到这些分支上。所有 EWC/GAM 日志的 orthogonal_loss 行数为 0，普通 OLoRA 为每份 4112 行。打印默认 olora_lamda_1=.5 不意味着启用了该分支。勿将当前九月源码中的 legacy OLoRA-base 行为倒推到五月结果。

若与九月 batch64/seed42 的 1B EWC 补跑并列表格，至少应注明 3B EWC/GAM 仍为 seed1993，而 1B EWC 已换为 seed42；不能将全表描述成统一 seed42。相同 global batch 也不能独立证明不同 GA 的 EWC/GAM 正则缩放一致。

## 逐单元证据（链接到真实文件和 LF 行号）

详细逐阶段 CLI、字段行号、runtime/loss、预测、Fisher 链条都在 [llama3_evidence.json](/data/140-1/users/liying/O-LoRA/refine-logs/nlp_table_audit_20260919/llama3_evidence.json)。

### SeqLoRA / Order1：67.8553 → 67.86

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:5384)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:21960)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:5850) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:11160) (78 updates)；[3-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:16642) (156 updates)；[4-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order1_llama3B/SeqLoRA_LLaMA2-3b.log:21919) (62 updates)。

### SeqLoRA / Order2：67.2632 → 67.26

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-yahoo/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:5392)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:21971)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:5858) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:11164) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:16443) (62 updates)；[4-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order2_llama3B/SeqLoRA_LLaMA2-3b.log:21930) (156 updates)。

### SeqLoRA / Order3：67.4046 → 67.40

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/SeqLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-dbpedia/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:5400)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:21971)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:5736) (156 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:11042) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:16321) (62 updates)；[4-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_SeqLoRA/scripts_order3_llama3B/SeqLoRA_LLaMA2-3b.log:21930) (218 updates)。

### IncLoRA / Order1：65.2303 → 65.23

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:6296)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:25539)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:6762) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:12956) (78 updates)；[3-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:19328) (156 updates)；[4-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/IncLoRA_LLaMA2-3b.log:25498) (62 updates)。

### IncLoRA / Order2：66.7434 → 66.74

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-yahoo/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:6296)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:25539)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:6762) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:12956) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:19123) (62 updates)；[4-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/IncLoRA_LLaMA2-3b.log:25498) (156 updates)。

### IncLoRA / Order3：69.5493 → 69.55

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/IncLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-dbpedia/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:6304)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:25539)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:6640) (156 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:12834) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:19001) (62 updates)；[4-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/IncLoRA_LLaMA2-3b.log:25498) (218 updates)。

### OLoRA / Order1：63.3651 → 63.37

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:6296)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:29651)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:8506) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:15324) (78 updates)；[3-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:22944) (156 updates)；[4-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order1_llama3B/OLoRA_LLaMA2-3b.log:29610) (62 updates)。

### OLoRA / Order2：62.9737 → 62.97

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-yahoo/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:6296)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:29651)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:8506) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:15324) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:21987) (62 updates)；[4-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order2_llama3B/OLoRA_LLaMA2-3b.log:29610) (156 updates)。

### OLoRA / Order3：63.9243 → 63.92

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/OLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-dbpedia/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:13)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:6304)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:29651)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:7888) (156 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:14706) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:21369) (62 updates)；[4-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_IncLoRA_OLoRA/scripts_order3_llama3B/OLoRA_LLaMA2-3b.log:29610) (218 updates)。

### IncLoRA+EWC / Order1：68.3586 → 68.36

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-agnews/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:6415)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:29923)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:8625) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:15493) (78 updates)；[3-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:23163) (156 updates)；[4-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:29879) (62 updates)。

### IncLoRA+EWC / Order2：69.1776 → 69.18

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-yahoo/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:6415)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:29923)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:8625) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:15493) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:22206) (62 updates)；[4-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA_LLaMA2-3b.log:29879) (156 updates)。

### IncLoRA+EWC / Order3：73.0395 → 73.04

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA_Llama-3.2-3B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-dbpedia/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:6423)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:29923)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:8007) (156 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:14875) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:21588) (62 updates)；[4-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA_LLaMA2-3b.log:29879) (218 updates)。

### IncLoRA+EWC+GAM / Order1：72.7138 → 72.71

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_1/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-agnews/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6399)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25747)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6865) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:13093) (78 updates)；[3-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:19499) (156 updates)；[4-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25703) (62 updates)。

### IncLoRA+EWC+GAM / Order2：73.5132 → 73.51

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_2/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-yahoo/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6399)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25747)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6865) (218 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:13093) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:19294) (62 updates)；[4-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order2_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25703) (156 updates)。

### IncLoRA+EWC+GAM / Order3：72.6612 → 72.66

- 原始最终结果：[predict_results.json](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_3B/order_3/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-dbpedia/predict_results.json:3)。
- 成功日志：[启动设置](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:91)、[有效 batch64](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6407)、[最终 FAA](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25747)。
- 四阶段训练 runtime/loss 精确匹配日志：[1-yahoo](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:6743) (156 updates)；[2-amazon](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:12971) (78 updates)；[3-agnews](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:19172) (62 updates)；[4-dbpedia](/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order3_llama3B/EWCLoRA-GAM_LLaMA2-3b.log:25703) (218 updates)。

