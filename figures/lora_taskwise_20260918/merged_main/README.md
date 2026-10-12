# LoRA 逐任务汇总图（2026-09-18 更新）

本版将已有的 **Vanilla LoRA、InfLoRA、SD-LoRA 单 seed 日志**加入主图，与 EWCLoRA、GAM-EWC+Merge 合并绘制；暂不绘制缺少原始曲线的 CL-LoRA、BiLoRA。四个数据集分别出图，并提供与示意图相同组织方式的四联图。所有中心曲线直接来自日志，未调整终点以匹配旧表。

## 结果文件

- [累计 AAA 四联图（PDF）](output/lora_taskwise_aaa.pdf) / [PNG](output/lora_taskwise_aaa.png)
- [阶段准确率四联图（PDF）](output/lora_taskwise_accuracy.pdf) / [PNG](output/lora_taskwise_accuracy.png)：纵轴含义对应用户提供的示意图。
- [绘图脚本](plot_lora_taskwise.py)、[原始日志来源及校验值](sources.json)。
- `output/` 同时保存各数据集独立图、逐 seed 原始点、逐阶段统计表和色带元数据。

本版单独保存到 `paper/Figures/lora_taskwise_20260918/merged_main/`，与之前仅包含多 seed 方法的版本区分。

## 色带必须这样解释

**三种单 seed 方法的细色带是固定 ±0.20 个百分点的示意带，不是标准差、方差、标准误或置信区间。** 图例中的 † 标识这些方法，图注也注明此含义。该带宽只控制绘图显示，不改变原始准确率和累计 AAA；不能据此声称这些方法在重复试验中稳定。

EWCLoRA 和 GAM-EWC+Merge 的色带仍为 **跨 seed 均值 ± 样本标准差（ddof=1）**。各组实际完整运行数如下：

| 方法 | CIFAR-100 | DomainNet | ImageNet-R | ImageNet-A |
|---|---:|---:|---:|---:|
| Vanilla LoRA † | 1 | 1 | 1 | 1 |
| InfLoRA † | 1 | 1 | 1 | 1 |
| SD-LoRA † | 1 | 1 | 1 | 1 |
| EWCLoRA | 5 | 3 | 4 | 5 |
| GAM-EWC+Merge | 4 | 6 | 6 | 6 |

对于 n=1，统计表中的实际标准差和方差保持空值，**不填 0，也不填示意带宽**。单独保存的色带元数据通过 `band_kind`、`half_width_pp` 区分固定示意带和实际样本标准差；百分点平方单位的方差仍只由多 seed 原始数据计算。

具体记录见 [band_metadata.csv](output/band_metadata.csv) 和 [band_settings.json](output/band_settings.json)；图注另存 [latex_includes.tex](output/latex_includes.tex)。

## 指标和来源说明

横轴为已完成训练的任务数：DomainNet 共 5 阶段，其余数据集共 10 阶段。阶段准确率 A[r,t] 为第 r 次运行完成第 t 个任务后的已见类别分类准确率；累计 AAA 为 `AAA[r,t] = (A[r,1] + ... + A[r,t]) / t`。先对每个 seed 独立计算累计 AAA，再汇总跨 seed 均值与样本标准差。

三个单 seed 基线使用已经找到的 seed0 日志，**未确认与用户旧表中的多次试验统计对应，终值可能不同**。本版按用户要求纳入主图，保留这一来源限制。日志的重复副本不视为独立试验。

旧表中的 FS-EWCLoRA 行此前被追溯到历史 GAM+weighted merge/Fisher 实现，因此沿用 **GAM-EWC+Merge** 名称。其统计使用同配置全部完整有限运行，可能与旧表所用 seed 子集不同；不能将它作为当前另一种 FS 更新公式的证据。来源、seed、排除运行及配置差异保留在 `sources.json` 和输出数据记录中。不同方法沿用历史训练配置，图展示历史学习轨迹。

## 重绘

依赖 Python、NumPy、Matplotlib；不需要 GPU。脚本根据 `sources.json` 读取对应原始日志。

```bash
cd /data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/Figures/lora_taskwise_20260918/merged_main
python plot_lora_taskwise.py
```

可通过 `--single-seed-band 0.1` 将示意带改为 ±0.1 个百分点；图注和色带元数据随参数同步更新。`--single-seed-band 0` 可隐藏示意带。

保留原始日志位置时可直接重绘；迁移到其它机器需同步日志并更新清单中的路径。引用或转存图片时，应同时保留图中的 † 和色带说明。

## 核验记录

已读取 51 份原日志，覆盖 20 个方法×数据集组合。新旧版本的原始点、阶段统计、终值汇总和旧表对照 CSV 完全一致；本轮只改变三条单 seed 曲线的展示方式。350 条色带记录中，210 条为固定示意带、140 条为真实标准差带。10 份 PDF 均为矢量输出，已检查图中文字、图例与色带说明。详见 [verification_report.json](verification_report.json)。
