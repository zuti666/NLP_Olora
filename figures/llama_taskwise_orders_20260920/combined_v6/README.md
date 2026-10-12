# Llama 分任务曲线合图 v6

单张 2×3 合图，上排 Llama-3.2-1B，下排 Llama-3.2-3B；三列对应 Order 1、2、3。

## 本版设计

- 一套顶部方法/数据集图例，沿用 v5 的颜色、四种线型和小标记；FS-EWCLoRA 为红色。
- (a)–(f) 的“模型–Order”标签增至 10.8 pt 并加粗。删除 Training stage 和重复的阶段数字，横轴保留按学习顺序排列的数据集名称。
- 图幅 7.2×4.35 英寸，适合双栏论文通栏放置。相较两张 v5 的总高度 7.70 英寸，原始导出总高度减少约 43.5%。
- 同一 Order 的上下两模型共用 y 范围；不同列按数据收紧范围。所有真实观测值保留，边界至少留 2 个百分点。

| Order | y 范围 (%) | 1B 实际 min–max (%) | 3B 实际 min–max (%) |
|---|---:|---:|---:|
| 1 | 24–93 | 26.4342–89.6842 | 30.3553–90.5395 |
| 2 | 23–92 | 25.8158–87.0000 | 36.2237–89.8026 |
| 3 | 39–94 | 41.5921–89.6579 | 42.2500–91.0658 |

绿色曲线的数据来自 **OLoRA+GAM**，显示名称按用户要求仍为 **`olra`**；实际方法名在 CSV 和图注中保留。每个 Order 最后引入的数据集只显示 stage 4 的真实单点，不画短线或作坐标偏移。

## 文件与复现

`llama_1b_3b_taskwise_orders_combined_v6.pdf` 为单页矢量论文图；同名 PNG 为预览、SVG 为可编辑版本。三个文件是同一张合图的不同格式，不再单独输出 1B/3B 图。

从本目录执行：

```bash
python gen_combined_v6.py
```

依赖 Python、Matplotlib 及上一级 `paper_plot_style.py`（归档包内包含）。服务器字体采用已有的 Times New Roman，其他环境若缺失则回退到 DejaVu Serif。代码只写本目录。

数据由 `taskwise_accuracy.csv` 读取，与 v5 逐字节一致（360 个观测）。`source_manifest.json`、`run_sources.csv` 提供真实结果路径与日志索引，`protocol_configs.csv` 保存配置差异，`RESULTS.md` 保留最终 FAA。不同方法的 seed、DDP/GA、优化器默认值及适配器预算差异已在图注说明。

`latex_include_v6.tex` 可用于论文；`*_plotted_data.json` 保存全部绘图坐标，`*_style_check.json` 和 `VALIDATION.json` 保存检查结果。已有版本未覆盖，其哈希保存在 `previous_files_sha256.json`。
