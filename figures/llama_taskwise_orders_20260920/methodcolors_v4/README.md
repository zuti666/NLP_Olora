# v4：使用 OLoRA+GAM 的真实曲线

按用户选择，将两种模型的绿色曲线更新为 **OLoRA+GAM**。图例、数据文件、结果表和图注均采用该方法名；纯OLoRA保留在v3中。

| OLoRA+GAM | Order-1 | Order-2 | Order-3 | Avg. |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 65.81 | 58.28 | 66.92 | 63.67 |
| Llama-3.2-3B | 71.62 | 70.94 | 72.05 | 71.54 |

未舍入的四个数据集指标均值：1B为65.8059、58.2796、66.917775，三个Order平均63.6677583333；3B为71.615125、70.944075、72.049325，三个Order平均71.536175。历史指标保留四位小数，最后统一舍入显示。

| 模型 | 预览 | 矢量图 |
|---|---|---|
| Llama-3.2-1B | [PNG](llama_1b_taskwise_orders_methodcolors_v4.png) | [PDF](llama_1b_taskwise_orders_methodcolors_v4.pdf) / [SVG](llama_1b_taskwise_orders_methodcolors_v4.svg) |
| Llama-3.2-3B | [PNG](llama_3b_taskwise_orders_methodcolors_v4.png) | [PDF](llama_3b_taskwise_orders_methodcolors_v4.pdf) / [SVG](llama_3b_taskwise_orders_methodcolors_v4.svg) |

## 原始来源与配置

每个模型的三个Order分别取自以下路径模式：

```text
/data/140-1/users/liying/O-LoRA/logs_and_outputs_llama_{1B,3B}/order_{1,2,3}/
OLoRA-GAM-002_Llama-3.2-{1B,3B}_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/
{stage}-{dataset}/predict_results.json
```

对应训练日志：

```text
/data/140-1/users/liying/O-LoRA/Logs_Scripts_Short_Paper/
Llama-3.2-{1B,3B}_AllOrders_IncLoRA_OLoRA/scripts_order{1,2,3}_llama{1B,3B}/
OLoRA-GAM_LLaMA2-{1b,3b}.log
```

`run_sources.csv`提供36个当前结果的实际完整路径；`source_manifest.json`记录逐阶段文件路径和SHA256，以及6个GAM训练日志路径、哈希和配置。24个阶段的训练耗时均匹配各自日志，训练结果、阶段状态和预测文件均存在。

OLoRA+GAM真实配置：`lora_strategy=olora`，`optimizer_type=gam-adamw_hf`；8卡×batch1×GA8=64，seed42、data_seed=None，BF16；OLoRA λ₁=0.5、λ₂=0；GAM rho=0.02、norm-rho=0.2、gamma=0.03、三个beta均为1，adaptive=False。每任务1 epoch，首任务LR1e-3，后续LR1e-4，rank8、alpha32、dropout0.1、q_proj/v_proj。

GAM底层Torch AdamW及其eps/weight decay默认值的说明来自对应源码推断，历史日志无实际optimizer对象或执行源码哈希，不能将其当作历史运行时审计。其他方法的seed、DDP/GA与适配器预算差异继续在图注和`protocol_configs.csv`披露。

## 数据和样式

从v3保留300个其他方法的数据点，替换60个纯OLoRA点为60个OLoRA+GAM点。全包仍为144个阶段、360个评估点、36个最终FAA。曲线直接使用实际dataset exact-match字段，排除SC/TC聚合，未学习任务留空；本轮未重读原始预测JSONL。

设计沿用v3：方法颜色、四种数据集线型、无点marker、下方两行图例。OLoRA+GAM为绿色，FS-EWCLoRA为醒目红色加粗。仅一次观测的数据仍在真实stage4位置用18pt短线显示，宽度是显示符号而非插值或误差区间。纵轴20–100完整覆盖所有结果。

- `olora_gam_selected_points.csv`：本次选定的60个GAM评估点及来源。
- `taskwise_accuracy.csv`：全部360个曲线点；`stage_seen_task_aa.csv`：阶段平均。
- `final_summary.csv`、`RESULTS.md`：两模型全部方法最终结果。
- `data_verification.json`、`verification.json`：数据一致性、旧文件保护及PDF核对。
- `latex_includes_v4.tex`：新图和图注。

新结果独立保存在`methodcolors_v4/`，原v1、v2、v3全部保留。

## 复现

在原服务器本目录运行：

```bash
python build_v4_data.py
python gen_llama1b_v4.py
python gen_llama3b_v4.py
```

绘图脚本复用父目录`paper_plot_style.py`。已有本目录CSV时仅需后两条命令。数据重建需要原实验目录及v3索引可读。
