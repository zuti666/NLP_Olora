# 多 seed 实验跟踪表

计划日期：2026-09-22；本次仅完成历史审计与计划，训练未启动。每行代表三个独立 Order，各四个任务阶段。实际队列以 `RUN_QUEUE_DDP4.csv` 为准。

| 运行组 | 模型 | 方法 | seed | Order 1/2/3 | 历史证据/待办 |
|---|---|---|---:|---|---|
| 1b_seqlora_s0 | Llama-3.2-1B | SeqLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_seqlora_s42 | Llama-3.2-1B | SeqLoRA | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |
| 1b_seqlora_s1993 | Llama-3.2-1B | SeqLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_inclora_s0 | Llama-3.2-1B | IncLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_inclora_s42 | Llama-3.2-1B | IncLoRA | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |
| 1b_inclora_s1993 | Llama-3.2-1B | IncLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_olora_gam_s0 | Llama-3.2-1B | OLoRA+GAM | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果；先验证lambda0.25的累积语义 |
| 1b_olora_gam_s42 | Llama-3.2-1B | OLoRA+GAM | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计；先验证lambda0.25的累积语义 |
| 1b_olora_gam_s1993 | Llama-3.2-1B | OLoRA+GAM | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果；先验证lambda0.25的累积语义 |
| 1b_treelora_s0 | Llama-3.2-1B | TreeLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_treelora_s42 | Llama-3.2-1B | TreeLoRA | 42 | 3/3 已完成，可复用 | 输入与方法兼容性最终核对后索引复用 |
| 1b_treelora_s1993 | Llama-3.2-1B | TreeLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_ewclora_s0 | Llama-3.2-1B | EWCLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_ewclora_s42 | Llama-3.2-1B | EWCLoRA | 42 | 3/3 已完成，可复用 | 输入与方法兼容性最终核对后索引复用 |
| 1b_ewclora_s1993 | Llama-3.2-1B | EWCLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_fs_ewclora_s0 | Llama-3.2-1B | FS-EWCLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_fs_ewclora_s42 | Llama-3.2-1B | FS-EWCLoRA | 42 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 1b_fs_ewclora_s1993 | Llama-3.2-1B | FS-EWCLoRA | 1993 | 3/3 已完成，可复用 | 输入与方法兼容性最终核对后索引复用 |
| 3b_seqlora_s0 | Llama-3.2-3B | SeqLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_seqlora_s42 | Llama-3.2-3B | SeqLoRA | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |
| 3b_seqlora_s1993 | Llama-3.2-3B | SeqLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_inclora_s0 | Llama-3.2-3B | IncLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_inclora_s42 | Llama-3.2-3B | IncLoRA | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |
| 3b_inclora_s1993 | Llama-3.2-3B | IncLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_olora_gam_s0 | Llama-3.2-3B | OLoRA+GAM | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果；先验证lambda0.25的累积语义 |
| 3b_olora_gam_s42 | Llama-3.2-3B | OLoRA+GAM | 42 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计；先验证lambda0.25的累积语义 |
| 3b_olora_gam_s1993 | Llama-3.2-3B | OLoRA+GAM | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果；先验证lambda0.25的累积语义 |
| 3b_treelora_s0 | Llama-3.2-3B | TreeLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_treelora_s42 | Llama-3.2-3B | TreeLoRA | 42 | 3/3 已完成，可复用 | 输入与方法兼容性最终核对后索引复用 |
| 3b_treelora_s1993 | Llama-3.2-3B | TreeLoRA | 1993 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_ewclora_s0 | Llama-3.2-3B | EWCLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_ewclora_s42 | Llama-3.2-3B | EWCLoRA | 42 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_ewclora_s1993 | Llama-3.2-3B | EWCLoRA | 1993 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |
| 3b_fs_ewclora_s0 | Llama-3.2-3B | FS-EWCLoRA | 0 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_fs_ewclora_s42 | Llama-3.2-3B | FS-EWCLoRA | 42 | 0/3，待运行 | 未找到目标seed的同协议完整历史结果 |
| 3b_fs_ewclora_s1993 | Llama-3.2-3B | FS-EWCLoRA | 1993 | 0/3 四卡结果，待迁移 | 旧八卡3/3完整；保留为参考，不混入四卡seed统计 |

## 当前里程碑

- [x] 历史查找、去重、144个阶段文件与日志证据核验。
- [x] 108个目标单元、96条四卡补跑清单已建立。
- [x] GPU3–6状态已只读检查；GPU3占用，4/5/6空闲。
- [ ] 新启动器与运行时审计支持任意目标seed；输入、方法和正则语义冻结。
- [ ] 四张指定GPU同时空闲，预检和小样本两阶段验证通过。
- [ ] 96条新增完整Order训练完成。
- [ ] 108条Order验证通过，三seed统计和新表/图另存。

本表中的“已有”是完成状态，不表示新启动任务。运行失败或缺失不能记为0准确率；不按结果好坏筛选seed。
