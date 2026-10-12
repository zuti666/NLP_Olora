# Fine-grained continual learning: taskwise average accuracy

五个数据集沿用当前表格顺序：**Aircraft、Cars、CUB-200、Flowers、5Datasets**。按用户前一条表格修改要求，Pets 不再单独出图；Oxford-IIIT Pet 仍是 5Datasets 的第五个任务。

## 指标与数据

每个点是完成第 `t` 个任务训练后的测试平均准确率：

`AA_t = (a[t,1] + ... + a[t,t]) / t`。

这里包含当前任务，每个已见任务等权。它是原始测试准确率下三角矩阵的第 `t` 行均值，并不是训练集准确率、截至当前阶段的累计 AAA、按测试样本数加权的准确率或 NME 指标。曲线最后一个点对应 FAA，所有阶段点的平均值对应表格的 stage-averaged AAA。

- 共 35 组已完成运行、315 个阶段点；前四个数据集各 10 个任务，5Datasets 为 5 个任务。
- 每条曲线均为一次运行，`n=1`。五张独立图不绘制误差带；按用户新增要求，横排合并图沿用参考图单次运行的 **固定 ±0.2 个百分点示意带**，并在图下注明它 **不是 SD、SE 或置信区间**。该带没有估计实验波动，不能作为统计显著性的依据。
- 曲线读取原始矩阵精度，未把表格中已四舍五入的 FAA/AAA 当作逐阶段数据。
- `independent_metric_audit.json` 独立复算 35 个原始来源，校验来源哈希、矩阵完整性以及与表格的关系；全部通过。
- 35 个 FAA 端点和 AAA 阶段均值均与当前表格在显示精度范围内一致。历史表中少数边界值有不同舍入惯例，例如 Cars/EWCLoRA 的原始 FAA 为 34.555，表格显示为 34.56。

5Datasets 学习顺序：**CUB-200 → FGVC-Aircraft → Stanford Cars → Oxford Flowers-102 → Oxford-IIIT Pet**。图中横轴 1–5 对应这一顺序；这里的任务均值对各数据集等权，不按类别数加权。

## 运行选择与可比性

所有曲线沿用当前结果表对应的运行。EWCLoRA 使用用户此前指定的“每个数据集已归档、有效运行中的最高 FAA”，其完整任务矩阵来自同一次运行，没有逐阶段择优或拼接不同运行。GAM-EWCLoRA/5Datasets 使用已完成运行的正式 receipt（FAA 83.37，AAA 81.1556666667）。

这些图比较有记录的历史配置，不能解释为严格控制其他变量后的方法效应。共同设置为 ViT-B/16、LoRA rank 16 和 batch size 128，但训练配置和方法实现仍有差异，例如：

| 项目 | 配置 |
|---|---|
| Aircraft 学习率 | SDLoRA 为 0.02，其他方法为 0.05 |
| 前四个数据集训练轮数 | 每任务 40 epochs |
| 5Datasets 训练轮数 | InfLoRA/OLoRA/SDLoRA 为每任务 40 epochs；Vanilla/CL-LoRA/EWCLoRA/GAM-EWCLoRA 为 30 |
| EWCLoRA seed | Cars/Flowers 为 1，5Datasets 为 2，Aircraft/CUB-200 为 0 |
| 其他方法 seed | 0 |
| EWCLoRA 系数 | Aircraft 200；Cars 80000；CUB-200 200000；Flowers 10；5Datasets 50000 |
| EWCLoRA 实现 | 不同历史运行使用的封装/Fisher 实现也有差异，见数据和来源清单 |

EWCLoRA 是按最终测试 FAA 回顾选择的最佳已归档运行，并非对所有方法实施统一验证集选择。历史预训练权重标识相同也不代表已验证权重文件逐字节相同。有关完整实现与配置，参见工作区 `refine-logs/lora_config_audit_20260919/` 和 `refine-logs/ewclora_best_audit_20260919/`；图不能消除这些既有可比性限制。

## 图片与样式

`output/` 下每个数据集均提供 300 dpi PNG、矢量 PDF 和可编辑 SVG：

| 数据集 | 文件名前缀 |
|---|---|
| Aircraft | `aircraft_taskwise_aa` |
| Cars | `cars_taskwise_aa` |
| CUB-200 | `cub200_taskwise_aa` |
| Flowers | `flowers_taskwise_aa` |
| 5Datasets | `5datasets_taskwise_aa` |

`all_datasets_taskwise_aa.pdf` 现为 **一页、一横排五个子图**，从左至右依次为 Aircraft、Cars、CUB-200、Flowers、5Datasets，子图顶部标注数据集名称。配套 `all_datasets_taskwise_aa.png`（300 dpi）和 `.svg`。原五页版保留为 `all_datasets_taskwise_aa_multipage.pdf`。

合并图的 35 条中心曲线与原图逐点相同，没有更换运行、平滑或修改均值。`horizontal_settings.json` 记录示意带定义，`output/all_datasets_taskwise_aa_verification.json` 记录布局、标题、带数量和数据一致性检查。`band_data_availability.md` 说明当前没有覆盖全部曲线的同配置多 seed 数据；局部 EWC Cars/Flowers 的重复运行不能直接作为当前最佳 seed 单线的标准差带。

参考 `lora_taskwise_aa.pdf` 的最终版本：Times New Roman、14 pt 轴标签、10 pt 刻度、1.5 pt 实线及原有颜色/标记。为新增的 OLoRA、CL-LoRA 分配浅蓝色菱形、紫色倒三角；GAM-EWCLoRA 使用品红色星形，但保留实际方法名称。七项图例放在坐标轴下方，避免遮挡曲线；横排合并图按用户要求使用 13 pt 粗体数据集标题，独立图的数据集名称仍置于 caption。纵轴按各数据集实际范围设置，并未强制各图共享范围。

图例顺序按行读取：Vanilla LoRA、InfLoRA、OLoRA、SDLoRA、CL-LoRA、EWCLoRA、GAM-EWCLoRA。独立图图例分两行，横排合并图采用一行共享图例。所有图没有插值、平滑或跨 seed 聚合。

## 复现

环境：Python 3，NumPy，Matplotlib；PDF 导出检查还使用 PyMuPDF。字体及其原始分发包、许可与来源说明位于 `fonts/`。

```bash
python plot_taskwise_aa.py
```

上述命令生成独立图、保留五页版，并按 `horizontal_settings.json` 重建横排合并图。只重建横排图可运行：

```bash
python plot_horizontal_aa.py
```

从任何目录运行该脚本均可；输入按脚本位置定位。只需要本包的冻结数据，不需要训练日志目录，也不启动实验。单独重画某张图可以运行 `gen_aircraft.py`、`gen_cars.py`、`gen_cub200.py`、`gen_flowers.py` 或 `gen_5datasets.py`。例如：

```bash
python gen_aircraft.py
```

`build_data.py` 是首次归档时的来源提取脚本，需要原工作区及日志路径；一般只需运行绘图脚本，不必重新提取数据。

主要可复核文件：

- `data/stage_accuracy.csv`：315 个逐阶段绘图值，长表格式。
- `data/final_summary.csv`：35 组 FAA、AAA 与表格显示值。
- `data/curves.json` 和每个运行的单独 JSON：冻结下三角矩阵、完整曲线、seed 和部分配置。
- `source_manifest.json`：原始文件路径和 SHA-256。
- `output/render_verification.json`：绘图对象与冻结曲线逐值一致检查、图例位置、输出哈希。
- `output/all_datasets_taskwise_aa_verification.json`：新增横排合并图、标题及示意带检查。
- `independent_metric_audit.json`：独立来源与指标核验。
- `reference_style.md`：参考图的样式核验。
- `latex_includes.tex`：五张图的 LaTeX 引用与完整 caption。

LaTeX 示例按双栏论文 `figure*` 中约 `0.78\textwidth` 的宽度提供，以保留图例可读性。若继续缩小到很窄的单栏，应先检查最终版面的文字大小。
