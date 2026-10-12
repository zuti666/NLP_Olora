# Llama 各任务顺序的准确率曲线

主图按模型分两组，每组横排三个 Order。颜色区分数据集，线型/标记区分方法；每条曲线从该数据集第一次学习后开始。另提供仅表格第一行 SeqLoRA 的两组图。

| 内容 | 文件 |
|---|---|
| Llama-1B 主图 | [PNG](llama_1b_taskwise_orders.png) / [PDF](llama_1b_taskwise_orders.pdf) / [SVG](llama_1b_taskwise_orders.svg) |
| Llama-3B 主图 | [PNG](llama_3b_taskwise_orders.png) / [PDF](llama_3b_taskwise_orders.pdf) / [SVG](llama_3b_taskwise_orders.svg) |
| 仅 SeqLoRA 1B | [PNG](llama_1b_seqlora_taskwise_orders.png) / [PDF](llama_1b_seqlora_taskwise_orders.pdf) |
| 仅 SeqLoRA 3B | [PNG](llama_3b_seqlora_taskwise_orders.png) / [PDF](llama_3b_seqlora_taskwise_orders.pdf) |
| 每个 Order、每个阶段、每个数据集的数值 | [RESULTS.md](RESULTS.md) / [CSV](taskwise_accuracy.csv) |
| 最终结果与顺序变化幅度 | [final_summary.csv](final_summary.csv) |
| 数据集首次学到后至最终的变化 | [dataset_changes.csv](dataset_changes.csv) |
| 原始结果路径 | [SOURCE_PATHS.md](SOURCE_PATHS.md) / [CSV](run_sources.csv) |
| 论文插图及表格 | [latex_includes.tex](latex_includes.tex) / [verified_final_table.tex](verified_final_table.tex) |

已定位132个阶段，绘制330个真实数据点；33个可追溯的最终FAA均与用户表格两位小数一致。1B新OLoRA的3个数值未找到匹配来源，主图明确标缺失，没有用旧OLoRA或插值替代。

1B：已确认的五种方法均在Order-2取得各自最低FAA。SeqLoRA的Order-1/2/3分别为66.90/55.71/68.06，顺序极差12.35个百分点；FS-EWCLoRA为69.85/68.13/70.62，极差2.49个百分点。SeqLoRA在Order-2中，Amazon从首次学习后的52.82降至25.82，AGNews从86.66降至62.79。

3B：SeqLoRA三个Order的FAA为67.86/67.26/67.40，极差0.59个百分点；FS-EWCLoRA为72.71/73.51/72.66，极差0.85个百分点。FS-EWCLoRA在Order-1/2取得最高FAA，Order-3最高为TreeLoRA73.46。总体FAA相近不表示每个数据集都稳定，例如3B SeqLoRA在Order-2的Amazon从55.43降至38.21。

以上为当前单seed结果的描述，seed、DDP卡数/GA、优化器及适配器预算不完全一致，不能把差值解释为仅由某一个组件造成。FS-EWCLoRA曲线对应实际ewclora_gam分支。详见protocol_configs.csv与图注。

绘图数据直接读取原始阶段指标，新运行最终点使用已经保存且核验的整数正确数；本轮没有重新计算原始预测JSONL。四位小数的历史记录按其实际精度保留。SC/TC任务类型聚合字段不作为数据集指标。3B SeqLoRA Order-1前三阶段日志包含未学任务，已按真实任务顺序屏蔽；未使用包含未学任务的总均值替代已见任务表现。

复现：先在原服务器执行`python build_data.py`提取原始指标；已有CSV时可直接运行`python gen_llama1b.py`和`python gen_llama3b.py`。第一行单方法版本使用`gen_llama1b_seqlora.py`与`gen_llama3b_seqlora.py`。最后运行`python package_results.py`汇总并打包。依赖Python和Matplotlib；字体优先使用仓库已有Times New Roman，否则使用DejaVu Serif。
