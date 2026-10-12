# v3：加入历史 Llama-1B OLoRA

按用户本轮明确选择，补入历史 OLoRA 的三个 Order。来源为用户提供的 [历史对照报告](/data/140-1/users/liying/O-LoRA/refine-logs/bf16_ewc_b64_s42_20260918/STATUS_REPORT.md)、[新旧分组索引](/data/140-1/users/liying/O-LoRA/refine-logs/bf16_ewc_b64_s42_20260918/RESULTS_INDEX.md)，并逐阶段读取实际 `predict_results.json`。两份参考文档另存于本目录 `reference_*.md`。

| 历史1B OLoRA | Order-1 | Order-2 | Order-3 | Avg. |
|---|---:|---:|---:|---:|
| FAA (%) | 58.61 | 53.12 | 58.42 | 56.71 |

这些是用户选定的历史结果；此前缺少来源的另一行 56.96/55.74/67.32 不用于这版图。图中的 EWC 继续使用原来的 batch64/seed42 新运行，其他方法和模型结果沿用v2。

| 模型 | 预览 | 矢量图 |
|---|---|---|
| Llama-3.2-1B | [PNG](llama_1b_taskwise_orders_methodcolors_v3.png) | [PDF](llama_1b_taskwise_orders_methodcolors_v3.pdf) / [SVG](llama_1b_taskwise_orders_methodcolors_v3.svg) |
| Llama-3.2-3B | [PNG](llama_3b_taskwise_orders_methodcolors_v3.png) | [PDF](llama_3b_taskwise_orders_methodcolors_v3.pdf) / [SVG](llama_3b_taskwise_orders_methodcolors_v3.svg) |

## 来源与核对

三个历史 OLoRA 运行目录如下，每个目录下都有1至4阶段的 `predict_results.json` 和 `train_results.json`：

- [Order-1](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_1/OLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4)
- [Order-2](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_2/OLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4)
- [Order-3](/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_1B/order_3/OLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4)

实际协议：8卡×每卡batch1×GA8=64，seed42、data_seed=None，BF16，OLoRA λ₁=0.5、λ₂=0，每任务新增rank8。三个最终指标文件的SHA256与原始审计索引一致。

新增12个训练阶段、30个真实评估点；全包现在覆盖144阶段、360点、36个最终FAA。其余330个点与v2完全相同。历史分数来自四位小数的数据集指标，未从FAA反推曲线、未重新遍历原始预测JSONL。Order-3四个已舍入子项的均值58.421025与日志总均值58.4211存在微小舍入差，均显示58.42。

## 样式和数据文件

沿用v2设计：方法颜色、四种数据集线型、无点标记；OLoRA为绿色，FS-EWCLoRA为红色加粗线。图例在三子图下方分两行。仅stage4有一次观测的数据集仍以真实位置的短线显示，线段宽度是显示符号，不是插值或误差区间。纵轴20–100完整覆盖新增最低值22.8684。

- `historical_olora_added_points.csv`：本次新增的30个点和原始文件来源。
- `taskwise_accuracy.csv`：全部360个点；`stage_seen_task_aa.csv`：各阶段已见任务AA。
- `source_manifest.json`、`run_sources.csv`：阶段与运行路径索引。
- `final_summary.csv`、`RESULTS.md`：两个模型各方法的完整结果表。
- `protocol_configs.csv`：已更新1B历史OLoRA的真实协议。
- `data_verification.json`、`verification.json`：数据、PDF与旧文件保护核对。
- `latex_includes_v3.tex`：两组新图及对应图注，已说明历史OLoRA来源和协议差异。

所有新结果独立保存在`methodcolors_v3/`，v1与v2文件均保持原样。

## 复现

在原服务器、本目录下依次运行：

```bash
python build_v3_data.py
python gen_llama1b_v3.py
python gen_llama3b_v3.py
```

绘图脚本复用父目录`paper_plot_style.py`。已有CSV时只需后两条命令；没有必要重新训练。原始构建索引与v2数据需在原目录可读。
