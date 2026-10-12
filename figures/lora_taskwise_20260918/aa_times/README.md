# 当前阶段任务平均准确率 AA：Times New Roman 版

本版按当前阶段对所有**已学习任务**等权平均，绘制 AA；同时提供同口径累计 AAA 作为对照。保留五种方法，全部使用实线，以颜色和点标记区分，图例放在每个子图内部，字体为真实 Times New Roman。

- **[AA 主图 PDF](output/lora_taskwise_aa.pdf)** / [PNG](output/lora_taskwise_aa.png)：本次请求的指标。
- [论文排版用 AA 2×2 版](output/lora_taskwise_aa_paper_2x2.pdf)：约 7 英寸宽时仍保留可读字号；原 1×4 版用于宽屏/原尺寸展示。
- [同口径 AAA 对照 PDF](output/lora_taskwise_aaa.pdf) / [PNG](output/lora_taskwise_aaa.png)。
- [绘图函数](plot_lora_aa.py)、[来源清单](sources.json)、[逐任务原始准确率](output/raw_task_accuracy.csv)。
- 四个数据集的独立图为 `output/<dataset>_aa.{pdf,png,svg}`，数据表为 `output/<dataset>_stage_statistics.csv`。

## AA 和 AAA 的区别

令 `a[t,j]` 为模型完成第 t 个任务训练后，在第 j 个任务测试集上的分类准确率。这里使用全局已见类别空间下的分类结果，不使用 task-oracle 或 prototype 结果。

```text
AA_t  = (a[t,1] + ... + a[t,t]) / t
AAA_t = (AA_1 + ... + AA_t) / t
```

**AA 横向平均当前模型在各任务上的表现；AAA 再沿学习时间平均历次 AA。** 例如各阶段 AA 为 90%、80%、70%，第 3 阶段的 AA 为 70%，而 AAA 为 80%。因此，观察模型当前保留了多少已学能力，应看 AA 曲线；AAA 描述整个学习过程的平均表现，变化通常更平缓。

“所有任务”在这里指截至当前已经学习的 1…t 个任务，不包含未来尚未学习的任务。那些任务在原始日志中未评估或记录为 NaN，不能纳入当前均值。

## 与前一版的第二处差别：按任务等权

前一版 `merged_main` 的 `accuracy` 来自日志 `total`，是各测试样本等权的总体准确率，`aaa` 是这些总体准确率的累计平均。本版 AA 则从每个阶段的任务分组准确率重算，每个任务权重相同。

任务测试样本数相同时，两者相同或仅有日志舍入差异；样本数不同时可能明显不同。这不是重跑实验或修改结果，而是统计口径改变。例如 ImageNet-A 的 EWCLoRA seed0：原 `total=55.04%`，该阶段十个任务的等权平均为 `AA=45.910%`。

本目录的 AAA 对照也从新 AA 累计计算，保证两张图使用同一口径；它不再复用旧版按样本加权的 AAA 数值。差异完整保存在 [metric_comparison.csv](output/metric_comparison.csv)，旧版文件继续保留。

## 原始数据和色带

解析每份日志最后一轮完整的 `=> Accuracy: {...}` 或 `=> CNN: {...}`，根据 `init_cls/increment` 核对每个任务的类别范围，计算当前行前 t 项的平均。未来任务的 NaN 不参与计算。每一行的 `total` 还会与同一次运行最后输出的 top1 曲线逐阶段核对，防止混入追加日志的上一轮结果。

所有 51 份日志均有完整的任务分组；22 份 GAM 日志的整个有效下三角还与同目录 `cl_metrics.json` 中 `cnn.matrices.final` 逐项一致。表格和原始日志中的其他同名 AAA/ACA 标量不直接用于本图。

- Vanilla LoRA、InfLoRA、SD-LoRA：各为 seed0，图例带 †；沿用用户要求的固定 ±0.2 个百分点示意色带，**不代表 SD、方差或置信区间**。
- EWCLoRA：四个数据集分别为 n=5/3/4/5；GAM-EWC+Merge 为 n=4/6/6/6，色带为实际均值 ± 样本标准差。
- 跨 seed 标准差先在每次试验内算出 AA/AAA，再逐阶段汇总。单 seed 的真实标准差、方差仍留空，示意带宽单独保存在 `band_metadata.csv`。
- 保留历史方法配置和旧表来源差异；GAM-EWC+Merge 沿用此前审计确认的实现名称。CL-LoRA/BiLoRA 按要求暂不绘制。

## 字体、样式和重绘

PDF 嵌入 Times New Roman，PNG 使用同一字体渲染，SVG 的字体声明也为 Times New Roman。局部字体在 `fonts/` 中注册，脚本禁止静默回退到替代字体；字体来源、SHA256、原始完整字体包及许可一并保存。原始安装程序未执行，也未修改系统字体。

每条线的线型统一为 `-`，线宽相同；仅颜色和 marker 不同。图例自动选择各子图内部空白区域。

保留参考图的一行四列组织，同时另存约 7.2 英寸宽的 2×2 论文版，避免将宽屏四联图缩至正文宽度后图例过小。`latex_includes.tex` 默认引用论文版；独立数据集图仍保留。

```bash
cd /data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/Figures/lora_taskwise_20260918/aa_times
python plot_lora_aa.py --metric aa    # 当前阶段 AA
python plot_lora_aa.py --metric both  # AA 和同口径 AAA
```

主要接口为 `read_task_accuracies(run)`（读取当前阶段各任务准确率）、`summarize(runs, metric)`（跨 seed 统计）和 `plot_dataset(ax, groups, dataset, metric='aa')`（绘图）。同目录数据与函数可一起复用。

数值、12 份 PDF 的嵌入字体、矢量输出、线型及图例位置已核验，详见 [verification_report.json](verification_report.json) 和 [独立图面复核](FIGURE_REVIEW.md)。

## 紧凑重排版（2026-09-24）

[PDF](output/lora_taskwise_aa_compact_tighty.pdf) / [PNG](output/lora_taskwise_aa_compact_tighty.png) / [SVG](output/lora_taskwise_aa_compact_tighty.svg)。
子图顺序为 ImageNet-R、ImageNet-A、DomainNet、CIFAR-100。画布由 13.4×3.65 英寸改为 13.4×2.95 英寸，裁切后总高度缩短约 19%。数据、色带、字体、颜色和标记沿用最终主图；图例继续自动选择子图内空白位置。

纵轴按所有曲线及色带的完整范围加 3% 边距、向外取整到 0.5 设置。四个范围依次为 `[54, 95.5]`、`[43.5, 89]`、`[66.5, 89]`、`[83, 100]`，所有结果完整显示。脚注中的 seed 数已随子图顺序调整。

```bash
cd /data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times
python restyle_aa_main.py --dataset-order imagenetr imageneta domainnet cifar100 --stem lora_taskwise_aa_compact_tighty --compact --tight-y
```

数值与布局核验见 [verification JSON](output/lora_taskwise_aa_compact_tighty_verification.json)，排版示例见 [LaTeX](output/lora_taskwise_aa_compact_tighty.tex)。

### 移除 InfLoRA 的版本

[PDF](output/lora_taskwise_aa_compact_tighty_no_inflora.pdf) / [PNG](output/lora_taskwise_aa_compact_tighty_no_inflora.png) / [SVG](output/lora_taskwise_aa_compact_tighty_no_inflora.svg)。移除 InfLoRA 曲线、色带、图例项及脚注名称；保留参考版的尺寸、纵轴范围、子图顺序和其余曲线样式。已逐条核对剩余 16 条曲线及色带的 SVG 几何与样式和参考版一致。

```bash
python restyle_aa_main.py --dataset-order imagenetr imageneta domainnet cifar100 --stem lora_taskwise_aa_compact_tighty_no_inflora --compact --tight-y --exclude-inflora
```
