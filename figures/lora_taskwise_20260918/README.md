# LoRA 逐任务曲线与累计 AAA（2026-09-18）

按提供的示意图组织：CIFAR-100、DomainNet、ImageNet-R、ImageNet-A 四列，同时输出每个数据集的独立图和逐阶段数据表。仅处理 LoRA 方法，不启动训练。

**当前可追溯的多 seed 主图覆盖 EWCLoRA 和 GAM-EWC+Merge；原表中全部七种 LoRA 方法的曲线尚不能完整恢复。** 其它方法的来源状态见下表。

## 直接查看

- [累计 AAA 四联图（PDF）](output/lora_taskwise_aaa.pdf) / [PNG](output/lora_taskwise_aaa.png)
- [阶段准确率四联图（PDF）](output/lora_taskwise_accuracy.pdf) / [PNG](output/lora_taskwise_accuracy.png)：这一版对应示意图的纵轴含义。
- 单独图：`output/{cifar100,domainnet,imagenetr,imageneta}_{aaa,accuracy}.{pdf,png,svg}`。
- 每个数据集的表格：`output/<dataset>_table.md`、`_table.tex`、`_stage_statistics.csv`。
- [全部逐 seed 原始点](output/raw_per_seed.csv)、[均值/标准差/方差](output/stage_statistics.csv)、[表格终值核对](output/table_comparison.csv)、[原始来源与 SHA256](sources.json)。
- [包含单 seed 参考的 AAA 预览](output_with_references/lora_taskwise_aaa.png)：额外三个方法带 †，n=1，无色带，未绑定旧表，不能当作原表的复现图。

![累计 AAA](output/lora_taskwise_aaa.png)

## 指标定义

设第 r 次试验学完第 t 个任务后，在全部已见测试样本上的分类准确率为 A[r,t]。

```text
accuracy[r,t] = A[r,t]
AAA[r,t]      = (A[r,1] + ... + A[r,t]) / t
mean[t]       = mean_r(metric[r,t])
std[t]        = sqrt(sum_r((metric[r,t] - mean[t])**2) / (n - 1))
```

横轴是**已完成训练的任务数**，不是任务内部的 epoch。每条曲线有 10 个点，DomainNet 有 5 个点。

先对每个 seed 计算累计 AAA，再计算跨 seed 的均值和标准差，因此保留了阶段之间的相关性。AAA 图最后一点对应各试验 AAA 的均值；accuracy 图最后一点对应最终阶段准确率。标准差也分别从对应的每 seed 指标计算。

色带为 **均值 ± 样本标准差（ddof=1）**，单位为百分点；实际方差另存 CSV，单位为百分点平方。色带不是置信区间。单 seed 的方差不能估计，表中留空、图中不画色带。

读取原日志最后一条 `CNN top1 curve:` 或精确的 `(curve) top1 Acc:`。不使用 NME、prototype 或 `with task` 的曲线。旧 `cl_metrics.json` 的同名 AAA 字段可能按下三角矩阵所有元素加权，其定义与本图不同，因此不直接读取那个标量。

## 来源覆盖和表格差异

| 方法 | CIFAR-100 | DomainNet | ImageNet-R | ImageNet-A |
|---|---|---|---|---|
| EWCLoRA | seeds 0–4，n=5 | seeds 0–2，n=3 | seeds 0–3，n=4 | seeds 0–4，n=5 |
| GAM-EWC+Merge | seeds 0/1/2/4，n=4 | seeds 0–5，n=6 | seeds 0–5，n=6 | seeds 0–5，n=6 |
| Vanilla LoRA | seed0 参考 | seed0 参考 | seed0 参考 | seed0 参考 |
| InfLoRA | seed0 参考 | seed0 参考 | seed0 参考 | seed0 参考 |
| SD-LoRA | seed0 参考 | seed0 参考 | seed0 参考 | seed0 参考 |
| CL-LoRA | 未找到原始曲线 | 未找到原始曲线 | 未找到原始曲线 | 未找到原始曲线 |
| BiLoRA | 未找到原始曲线 | 未找到原始曲线 | 未找到原始曲线 | 未找到原始曲线 |

主图只纳入前两行。前三种普通基线的现存 seed0 日志与用户表格的终值不完全对应；原日志的多份相同 SHA256 副本只算一次，无法充当多次试验。参考版明确以 † 标注这些单次运行。检索记录保存在 `baseline_reference_sources.json`、`missing_cl_bi_sources.json`。

旧表 FS-EWCLoRA 行被已有 GAM-EWC 审计追溯到历史 GAM+weighted merge/Fisher 实现，因此这里使用 `GAM-EWC+Merge` 名称，不能把该日志当作当前另一个 FS 更新公式的证据。此组包含同配置全部完整有限运行，没有按照表格终值筛选 seed。

| 数据集 | EWCLoRA：Final / AAA，mean ± SD | GAM-EWC+Merge：Final / AAA，mean ± SD |
|---|---|---|
| CIFAR-100 | 87.896 ± 0.111 / 92.265 ± 0.188 | 87.873 ± 0.205 / 91.965 ± 0.212 |
| DomainNet | 73.387 ± 0.135 / 79.494 ± 0.132 | 73.860 ± 0.052 / 79.487 ± 0.033 |
| ImageNet-R | 72.575 ± 0.579 / 78.380 ± 0.690 | 78.033 ± 0.302 / 82.736 ± 0.129 |
| ImageNet-A | 56.418 ± 0.880 / 67.093 ± 0.622 | 59.150 ± 1.171 / 70.466 ± 0.490 |

EWCLoRA 的 CIFAR/R/A 与原表对应，DomainNet 按日志精度应四舍五入为 73.39 / 79.49，与原表中心值各差 0.01。GAM 全运行集合与旧 FS 行的统计集合不一致：例如 ImageNet-A 原表 60.06 对应旧 seeds2/3/4 子集，这里全六 seed 为 59.15；CIFAR 原表 88.03 未与完整集合匹配。所有差值都在 `table_comparison.csv` 中保留，不移动曲线终点去拟合旧表。

排除/未启动状态也保留在 `sources.json` 和 `output/computation.json`：CIFAR GAM seeds3/5 训练 NaN；DomainNet EWC seed3 与 ImageNet-R EWC seed4 因数据文件缺失而未完成；DomainNet EWC seed4 未启动。图中均值描述完整有限运行，不能省略这些分母解释。

不同方法保留历史训练配置，例如 ImageNet-A EWCLoRA 10 epochs、GAM 50 epochs；图用于展示历史学习轨迹，不能解释为同预算的组件因果消融。每个 seed 的已记录配置包含在来源清单和 `output/parsed_raw_runs.json` 中。

## 重绘和绘图函数

依赖：Python、NumPy、Matplotlib。主脚本不依赖 pandas、PyTorch 或 GPU。

```bash
cd /data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/Figures/lora_taskwise_20260918
python plot_lora_taskwise.py --metric both

# 只画一个数据集的累计 AAA
python plot_lora_taskwise.py --dataset cifar100 --metric aaa

# 可选：加入单 seed 参考（†）
python plot_lora_taskwise.py --manifest sources_with_references.json \
  --output output_with_references --metric both
```

`plot_lora_taskwise.py` 的函数接口：

- `read_task_curve(log_path, n_tasks, expected_sha256=None)`：读取原日志、检查长度/数值/训练失败。
- `cumulative_aaa(curves)`：输入 `[seed, stage]` 或单条 `[stage]`，逐 seed 累计平均。
- `summarize_curves(curves, metric="aaa", ddof=1)`：返回每阶段 mean、SD、variance。
- `load_raw_runs(manifest_path)`：直接读取清单中每份原日志并校验 SHA256，避免误读修改后的文件。
- `plot_dataset(ax, groups, dataset, metric="aaa", ddof=1)`：在一个 Matplotlib 轴上绘制该数据集及色带。

后续找到其它方法的原始多 seed 日志后，按 `sources.json` 的结构加入 `dataset/method/seed/log/include/sha256` 条目即可重绘。应加入同一配置的完整运行集合，并记录不完整/失败原因；不得把单 seed 参考与不同配置的试验合并为误差带。支持的七种方法已经配置好统一颜色、标记与图例。

`build_sources.py` 是本次明确来源集合的生成记录，生成默认主清单与参考清单；默认绘图只读取已经固定的清单。`output/latex_includes.tex` 可从本目录作为 LaTeX 工程根目录使用，在其它工程中使用时调整图片路径。

## 已完成的验证

- 从 39 份不同原日志重读主图的 8 个方法×数据集组，并校验来源 SHA256。
- 验证两种日志解析只读取分类准确率，排除 task-oracle/prototype/NME 曲线。
- 用阶段间反相关的两次试验检查 AAA 色带：累计平均后方差可为零，不能用阶段 SD 的平均代替。
- 单 seed 的标准差留空；完整 seed 数在每个独立图图例和汇总图脚注中列出。
- 输出 PDF（矢量）、PNG、SVG 以及各数据集逐任务 CSV/Markdown/LaTeX 表格。
- 用独立的 `statistics.mean/stdev` 复核主图 345 个逐 seed 阶段点（参考版 450 个点）的全部汇总值，结果通过；见 `verification_report.json`。
- 检查 20 份 PDF 均包含字体和矢量路径、无嵌入栅格图；检查全部 PNG 和渲染后的 PDF，无标签裁切或图例覆盖。
- GPT-5.6-Sol xhigh 独立只读复核通过，见 `FIGURE_REVIEW.md`。
