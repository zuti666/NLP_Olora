# BF16 batch64 / seed42 运行结果

状态：completed。正式阶段验证12/12；运行0、失败0、待执行0。两阶段smoke均通过。结束时间：2026-09-18 22:20:14 CEST；总耗时108.62分钟。Supervisor PID71236已退出，训练GPU已释放。

独立复核原始预测共91,200条，每Order30,400、每任务7,600。重算FAA与输出一致，48个rank审计一致，无所需数值非有限问题。

## 当前结果：最终四任务 exact match（%）

| Order | DBPedia | Amazon | Yahoo | AGNews | FAA |
|---|---:|---:|---:|---:|---:|
| 1 | 79.3026 | 37.1184 | 63.7763 | 88.5789 | 67.1941 |
| 2 | 73.5921 | 38.9868 | 66.9211 | 68.1711 | 61.9178 |
| 3 | 77.5132 | 50.5921 | 58.7632 | 79.2763 | 66.5362 |

三个Order平均：**65.2160%**。仅对当前协议求平均。

## 历史对照（batch64 / seed42）

| 方法 | Order1 | Order2 | Order3 | Avg. |
|---|---:|---:|---:|---:|
| SeqLoRA | 66.90 | 55.71 | 68.06 | 63.56 |
| IncLoRA | 64.52 | 56.19 | 65.44 | 62.05 |
| OLoRA | 58.61 | 53.12 | 58.42 | 56.71 |
| IncLoRA+EWC | 67.19 | 61.92 | 66.54 | 65.22 |

历史三类对照使用8GPU/BS1/GA8，新EWC使用4GPU/BS1/GA16、CLI λ0.5；batch与seed对齐，但设备数、随机流及源码环境来源不同。历史FAA来自四位小数的分任务指标，新FAA来自原始整数正确数。以上不构成严格同执行配置、多seed消融。

## 旧 batch40 / seed1993 单独保留

旧EWC结果Order1/2/3为59.3125、56.8421、67.2303，平均61.1283%。其原始文件与完成状态保持原样，不并入新均值。

新原始结果：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/final_results_bf16_b64_s42_ddp4_bs1_ga16_lam0p5.json`。
状态：`/data/140-1/users/liying/O-LoRA/experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/state.json`。
分组索引：`/data/140-1/users/liying/O-LoRA/refine-logs/bf16_ewc_b64_s42_20260918/RESULTS_INDEX.md`。
