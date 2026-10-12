# 最新 ImageNet-R/A 三 seed 曲线（2026-09-25）

以 `lora_taskwise_aa_compact_tighty_no_inflora.pdf` 为样式参考，更新 ImageNet-R、ImageNet-A 的 FS-EWC 数据来源。保留 ImageNet-R → ImageNet-A → DomainNet → CIFAR-100 顺序，不绘制 InfLoRA。

## 两种口径，分别保存

- **与用户最新 FAA/AAA 汇总一致的样本加权版**：[PDF](output/lora_taskwise_sample_aa_updated.pdf) / [PNG](output/lora_taskwise_sample_aa_updated.png) / [SVG](output/lora_taskwise_sample_aa_updated.svg)。每个方法都使用同阶段 CNN global top1。为统一口径，基线曲线也从其原有日志读取 global top1；没有只把新 FS 曲线替换成另一种指标。
- **沿用旧图任务等权口径的版本**：[PDF](output/lora_taskwise_task_aa_updated.pdf) / [PNG](output/lora_taskwise_task_aa_updated.png) / [SVG](output/lora_taskwise_task_aa_updated.svg)。只更新两组 FS-EWC 曲线及其色带；其余 14 条曲线的 SVG 点坐标与参考图完全相同。

两版画布均为 13.4×2.95 英寸，裁切后约 13.271×2.888 英寸；Times New Roman、14 pt 轴标签、13 pt 粗体数据集名、10 pt 刻度、1.5 pt 实线、4 pt 标记，与参考版一致。纵轴从参考范围出发，仅在数据或色带超界时向外扩展；本批两版无需扩展。

## 最新 FS-EWC 对应数值

均为 seeds 0/1/2，± 为样本标准差（ddof=1）。FAA 是最后阶段曲线值，AAA 是每个 seed 全部十阶段曲线的平均，再跨 seed 汇总。

| 数据集 | 口径 | FAA (%) | AAA (%) |
|---|---|---:|---:|
| ImageNet-R | 样本加权 | 79.810 ± 0.207 | 83.508 ± 0.108 |
| ImageNet-A | 样本加权 | 62.080 ± 1.616 | 71.656 ± 0.247 |
| ImageNet-R | 任务等权 | 79.749 ± 0.219 | 83.570 ± 0.125 |
| ImageNet-A | 任务等权 | 54.981 ± 1.907 | 68.344 ± 0.124 |

- 样本加权：直接读取日志最后一条完整 `CNN top1 curve`，并与每阶段 `CNN: {'total': ...}` 交叉核对。
- 任务等权：`AA_t = mean(a[t, 1:t])`。各任务测试样本数不等时，与 global top1 不同。
- 两种 AAA 都是十阶段曲线的平均。**不使用** `cl_metrics.json` 中平均整个下三角矩阵的同名 `cnn.final.AAA` 字段。
- 相邻任务点只用直线连接；没有数值插值、平滑、端点调整。

## 数据替换和方法身份

[sources.json](sources.json) 固定了所有原始路径和哈希。保留原图 35 份日志，替换 ImageNet-R/A 历史 FS 两组共 12 份日志，换入最新 6 份，总共绘制 41 份运行。历史被替换记录单独保存在 `replaced_historical_runs`，不进入本版均值或色带。其他方法和 DomainNet/CIFAR-100 继续读取原来源。

- ImageNet-R：[最新汇总来源](/data/140-1/users/liying/CopyFlatCV/refine-logs/imagenetr-r1-seeds012-20260924/results.json)。三次 GAM-EWC+A，`gam_grad_norm_rho=1.0`。
- ImageNet-A：[最新汇总来源](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-seeds01-20260924/MULTISEED_RESULTS.json)。新增 seeds 0/1 和已有同配置 seed2，`gam_grad_norm_rho=1.0`。
- 图例 `FS-EWC` 继续作为这一历史实现系列的显示别名；最新六次运行实际方法类型均为 `as1_normfisher_gam_intask_damp_weightmergethenfisher`，配置别名为 `ewclora_normfisher_gam_intask_damp`。没有混入其他 FS 实现。
- FS seed 数按子图顺序改为 `3/3/6/4`；EWCLoRA 仍为 `4/5/3/5`。
- Vanilla LoRA、SD-LoRA 各为单 seed，保留 ±0.2 pp 示意带，不是估计的 SD 或 CI。其余为均值±样本标准差。不同方法继续沿用各自历史训练配置，因此图展示这些运行的描述性结果。

## 新绘图函数与复现

[plot_updated_lora_aa.py](plot_updated_lora_aa.py) 提供：

- `load_updated_runs()`：重新读取 hash 校验后的原始日志，将最新六份日志矩阵与各自原始 JSON 逐项核对。
- `plot_updated_dataset()`：按指定的统一口径绘制一个数据集，保持方法颜色和标记。
- `plot_updated_figure()`：生成四联图及 PDF、PNG、SVG、LaTeX 和显示参数。

```bash
cd /data/140-1/users/liying/O-LoRA/figures/lora_taskwise_updated_20260925
python plot_updated_lora_aa.py --metric both
# 单独生成：--metric sample 或 --metric task
python verify_outputs.py
```

`build_sources.py` 用于显式重建来源清单；通常重绘无需运行。生成脚本复用旧目录的 `plot_lora_aa.py` 日志解析、统计和 Times New Roman 字体，因此移动目录时也需保留该依赖。

## 核验与配套文件

- [逐阶段均值/标准差/色带](output/stage_statistics.csv)
- [全部 FAA/AAA 汇总](output/summary_metrics.csv) / [每 seed 数值和原日志路径](output/per_seed_metrics.csv)
- [原始日志与六份矩阵交叉核验](output/source_verification.json)
- [实际 SVG/PDF 核验](output/render_verification.json)：每版 140 个真实绘图点与数据表一致；全部色带位于坐标范围内；PDF 为矢量，文字无越界，尺寸与参考相同。
- [ImageNet-R 独立原始来源核验](imagenetr_source_audit.json) / [ImageNet-A 独立原始来源核验](imageneta_source_audit.json)
- [样本加权版 LaTeX](output/lora_taskwise_sample_aa_updated.tex) / [任务等权版 LaTeX](output/lora_taskwise_task_aa_updated.tex)

旧图、旧原始数据清单和原绘图函数均保持原样。本次未启动训练或修改原始实验结果。
