# Llama 分任务曲线 v5

两组图分别对应 Llama-3.2-1B、Llama-3.2-3B；每组横向排列 Order 1–3。

本版复用 v4 数据与绘图代码，另存于 `methodcolors_v5`：

- 方法用颜色区分，FS-EWCLoRA 为醒目红色；图例位于全部坐标轴上方。
- 数据集用四种线型和四种 3 pt 小标记区分：DBPedia 实线+圆点、Amazon 虚线+方块、Yahoo 点线+三角、AGNews 点划线+菱形。
- 每个 Order 最后引入的数据集只有 stage 4 的一次观测，因此只绘制该位置的小标记。
- 绿色曲线采用真实 **OLoRA+GAM** 结果，按用户要求将方法图例显示为 **`olra`**。CSV、来源索引和统计结果继续保留实际方法名称；LaTeX 图注解释这一显示别名。
- 保留每个实际观测值及原始坐标，不对重叠点加偏移。

## 文件

- `llama_1b_taskwise_orders_methodcolors_v5.{pdf,png,svg}`
- `llama_3b_taskwise_orders_methodcolors_v5.{pdf,png,svg}`
- `latex_includes_v5.tex`：两组图的 LaTeX 引用与图注。
- `taskwise_accuracy.csv`：绘制的 360 个真实观测值，与 v4 文件逐字节一致。
- `source_manifest.json`、`run_sources.csv`：结果文件和训练日志来源。
- `protocol_configs.csv`：方法配置；seed、DDP/GA、优化器默认值、适配器预算存在差异，图注保留相关说明。
- `RESULTS.md`、`final_faa_by_order.csv`：各 Order 的最终 FAA。
- `*_plotted_data.json`、`*_style_check.json`、`VALIDATION.json`：绘制坐标与校验结果。
- `FIGURE_REVIEW.md`：增量视觉检查。

## 重新生成

从本目录运行：

```bash
python gen_llama1b_v5.py
python gen_llama3b_v5.py
```

脚本依赖 Python、Matplotlib 及上一级的 `paper_plot_style.py`，仅写入本目录。归档包包含该公共样式文件。字体优先使用服务器上已有的 Times New Roman；其他环境若无该字体，公共样式会使用备用字体。

旧版文件的 SHA-256 保存在 `previous_files_sha256.json`，本版生成过程未覆盖旧版。
