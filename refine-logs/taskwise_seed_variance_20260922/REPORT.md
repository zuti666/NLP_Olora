# 当前细粒度表格与曲线的多 seed / 方差核查

核查日期：2026-09-22。范围为当前 Aircraft、Cars、CUB-200、Flowers、5Datasets 五数据集及七种方法，使用既定35个结果，不重新挑选最高分、不修改表或图、不启动训练。

## 当前展示结果不是多 seed 平均

35个方法×数据集格子均由一个运行产生。Vanilla LoRA、InfLoRA、OLoRA、SDLoRA、CL-LoRA、GAM-EWCLoRA全部展示seed0。EWCLoRA按表格顺序展示seeds **0、1、0、1、2**。它们各自仍是n=1；不同数据集用不同seed不构成同一格的多seed重复。

因此，当前已展示的35个单次结果，跨seed样本方差及标准差均为 **NA（无法估计）**，不能写为0。当前横排图中固定±0.2个百分点的带是明确标注的示意范围，既不是实验方差，也不是SD或置信区间。

## 另外可确认的同配置重复组

在当前EWCLoRA所用配置的邻近seed目录，Cars和Flowers分别各有seed1、seed2。四份原始结果均通过此前来源SHA-256复核，完整下三角矩阵有限；组内已记录训练参数除了seed和路径类元数据没有差异。当前表只选用了两组各自的seed1，没有使用下面的均值。

| 数据集 | EWCLoRA λ | Seed | FAA (%) | AAA_stage (%) | 是否当前表选中 |
|---|---:|---:|---:|---:|---|
| Cars | 80000 | 1 | 34.555000 | 43.942118 | 是 |
| Cars | 80000 | 2 | 32.144000 | 42.481264 | 否 |
| Flowers | 10 | 1 | 53.434000 | 65.449362 | 是 |
| Flowers | 10 | 2 | 51.258000 | 65.023630 | 否 |

采用样本方差 `s² = sum_i (x_i − mean(x))² / (n−1)`，即ddof=1。准确率及SD用百分点表示，方差单位为百分点平方（pp²）。

| 数据集/方法 | n | FAA 均值 ± SD | FAA 样本方差 (pp²) | AAA 均值 ± SD | AAA 样本方差 (pp²) |
|---|---:|---:|---:|---:|---:|
| Cars / EWCLoRA | 2 | 33.349500 ± 1.704834 | 2.9064605 | 43.211691 ± 1.032979 | 1.06704658 |
| Flowers / EWCLoRA | 2 | 52.346000 ± 1.538664 | 2.3674880 | 65.236496 ± 0.301038 | 0.09062376 |

每个seed先独立计算逐任务下三角行均值AA_t；FAA取最后一个AA_t，AAA取所有阶段AA_t的均值；然后在相同指标上跨seed计算方差。没有把任务当作重复实验，也没有先混合两seed的矩阵再估计AAA方差。

这两组均属于已记录配置一致的历史重复，但历史代码、数据和权重文件并非全部被不可变绑定；n=2也不足以给出稳定精确的方差估计。如改为多seed展示，均值必须同步变为上表数值，不能给原best-seed中心直接附加这组SD并称作“均值±SD”。

## 其他 GAM 运行批次的两组统计（不是当前表对应的方差）

在相邻 `gam-ewc-gapfill-launch-20260917/attempt2` 完成清单中，CUB-200 和 Flowers 各有两次 GAM-EWCLoRA 运行，seeds均为1、2，λ均为2000。它们显式启用 Fisher RNG、clean-probe RNG 和 loader RNG 隔离，并绑定新的冻结代码。当前表的历史seed0没有这些开关/版本的完整运行绑定；Flowers历史seed0连GAM扰动参数也未明确记录。因此，以下统计只针对该批次的seed1、2，不与原表seed0合并成n=3。

| 数据集 | Seed | FAA (%) | AAA_stage (%) |
|---|---:|---:|---:|
| CUB-200 | 1 | 73.568000 | 76.342083 |
| CUB-200 | 2 | 72.665000 | 75.445499 |
| Flowers | 1 | 84.109000 | 90.577130 |
| Flowers | 2 | 81.592000 | 88.494829 |

| 数据集/方法（另一运行批次） | n | FAA 均值 ± SD | FAA 样本方差 (pp²) | AAA 均值 ± SD | AAA 样本方差 (pp²) |
|---|---:|---:|---:|---:|---:|
| CUB-200 / GAM-EWCLoRA | 2 | 73.116500 ± 0.638517 | 0.4077045 | 75.893791 ± 0.633980 | 0.40193084 |
| Flowers / GAM-EWCLoRA | 2 | 82.850500 ± 1.779788 | 3.1676445 | 89.535980 ± 1.472409 | 2.16798830 |

`gam_alternative_cohorts.json` 保存源文件、参数、receipt及哈希。`compute_alternative_gam.py` 独立重读原矩阵，核对4个运行的metric/log/config/receipt哈希并复算统计，输出 `alternative_gam_raw_metrics.csv`、`alternative_gam_summary.csv`、`alternative_gam_stage_variance.csv`。组内配置只差seed、loader_seed、prefix。

## 检查范围与复现

- `baseline_availability.md`：四基线×五数据集，原method/optimizer/dataset根及同prefix候选；20格均仅seed0。
- `gam_cl_availability.md`：GAM/CL选中来源及邻近campaign范围，含非同配置候选的排除理由。
- `variance_audit.json`：EWC五种当前配置的目录覆盖、原始来源哈希、4个重复运行的指标、2组统计量。
- `current_selection.csv` / `.json`：当前35格的单seed身份及方差不可估计标记。
- `replicate_raw_metrics.csv`：4个原始运行的FAA/AAA。
- `replicate_summary.csv`：2组方差、SD、均值。
- `replicate_stage_variance.csv`：2组×10任务的逐阶段均值、样本方差和SD，可供后续真实误差带使用。
- `compute_variance.py`：从原始矩阵重算，不依赖已四舍五入的表格。

建议新增实验时固定每个数据集的训练配置，以相同的预设seed列表完成重复，并同时报告均值与样本SD；当前任务只核查和计算，未发起额外实验。
