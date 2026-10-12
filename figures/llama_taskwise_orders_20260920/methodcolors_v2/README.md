# 方法颜色 / 数据集线型：v2

复用上一版的真实数据、三个 Order 布局和共享字体配置，改动仅涉及视觉编码和图例位置。原目录内的旧图片、数据、脚本和压缩包均保留；新结果位于 `methodcolors_v2/`，文件名带 `_methodcolors_v2`。

| 模型 | 预览 | 矢量图 |
|---|---|---|
| Llama-3.2-1B | [PNG](llama_1b_taskwise_orders_methodcolors_v2.png) | [PDF](llama_1b_taskwise_orders_methodcolors_v2.pdf) / [SVG](llama_1b_taskwise_orders_methodcolors_v2.svg) |
| Llama-3.2-3B | [PNG](llama_3b_taskwise_orders_methodcolors_v2.png) | [PDF](llama_3b_taskwise_orders_methodcolors_v2.pdf) / [SVG](llama_3b_taskwise_orders_methodcolors_v2.svg) |

- 方法用颜色区分：SeqLoRA灰蓝、IncLoRA蓝、OLoRA绿、TreeLoRA紫、EWCLoRA金黄；FS-EWCLoRA使用醒目的红色 `#E31A1C`，线宽略加粗，图例文字同为红色加粗。
- 数据集仅用四种线型区分：DBPedia实线、Amazon虚线、Yahoo点线、AGNews点划线。所有曲线和图例都不使用点标记。
- 图例移到整组三个子图下方：方法颜色一行、数据集线型一行；不遮挡曲线。
- 最后才学到的数据集只有一次观测。其真实stage4位置用18pt宽的对应线型短线显示；短线宽度仅为显示符号，不是跨训练阶段的区间、插值或误差范围。
- 数据仍为132个已核实训练阶段、330个观测点。1B新OLoRA来源缺失，继续明确标注且不绘制；FS-EWCLoRA仍对应实际GAM-EWC分支。协议差异见随附`protocol_configs.csv`。

复现：在本目录运行 `python gen_llama1b_v2.py` 和 `python gen_llama3b_v2.py`。两者复用父目录`paper_plot_style.py`，读取本目录与上一版完全一致的CSV快照。原文件SHA256清单保存在`original_files_sha256.json`，复核结果见`verification.json`。

`latex_includes_v2.tex`提供两种模型的新图注，说明方法颜色、数据集线型、单次观测短线以及实验协议。生成脚本改编自上一版`plot_taskwise.py`，无需重新提取指标或运行训练。
