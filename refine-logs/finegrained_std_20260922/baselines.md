# 四个历史基线的多 seed 核查（2026-09-22）

扫描 CopyFlatCV 与 Flatness_CV，包括 ignored/hidden 文件及符号链接，读取 6798 个 `.log` 文件开头的配置块；共发现 3149 个 `cl_metrics.json` 路径。按 model_name、dataset 从日志内容选择候选，再与当前表的非 seed 配置逐字段比较。

结论：20 个方法×数据集组合均未找到可与当前选定记录合并的其他 seed；每格仅 seed 0，n=1，跨 seed 的样本标准差和方差不可计算。重复备份与同 seed 的参数搜索不能充当独立重复。

| Method | Dataset | Seeds | n | FAA | AAA_stage | Sample SD |
|---|---|---|---:|---:|---:|---|
| Vanilla LoRA | aircraft | 0 | 1 | 40.66000 | 39.51207 | unavailable |
| Vanilla LoRA | cars196 | 0 | 1 | 28.52000 | 39.41488 | unavailable |
| Vanilla LoRA | cub200 | 0 | 1 | 57.79300 | 70.00153 | unavailable |
| Vanilla LoRA | flowers | 0 | 1 | 53.05200 | 66.90461 | unavailable |
| OLoRA | aircraft | 0 | 1 | 36.09800 | 37.80625 | unavailable |
| OLoRA | cars196 | 0 | 1 | 32.85500 | 42.29808 | unavailable |
| OLoRA | cub200 | 0 | 1 | 62.37600 | 70.16712 | unavailable |
| OLoRA | flowers | 0 | 1 | 48.89800 | 64.25757 | unavailable |
| SDLoRA | aircraft | 0 | 1 | 25.95900 | 26.63324 | unavailable |
| SDLoRA | cars196 | 0 | 1 | 34.89100 | 43.27635 | unavailable |
| SDLoRA | cub200 | 0 | 1 | 62.00600 | 69.52474 | unavailable |
| SDLoRA | flowers | 0 | 1 | 49.04700 | 64.58805 | unavailable |
| InfLoRA | aircraft | 0 | 1 | 40.95800 | 41.22798 | unavailable |
| InfLoRA | cars196 | 0 | 1 | 34.34500 | 42.20133 | unavailable |
| InfLoRA | cub200 | 0 | 1 | 60.60800 | 70.60121 | unavailable |
| InfLoRA | flowers | 0 | 1 | 60.06000 | 72.97994 | unavailable |
| Vanilla LoRA | het5 | 0 | 1 | 66.80400 | 68.80630 | unavailable |
| OLoRA | het5 | 0 | 1 | 77.58000 | 74.52350 | unavailable |
| SDLoRA | het5 | 0 | 1 | 74.42000 | 74.09900 | unavailable |
| InfLoRA | het5 | 0 | 1 | 77.74800 | 77.01627 | unavailable |

所有来源路径、SHA256、候选配置差异均见 `baselines.json`。纯 GAM 的 seqlora 多 seed 与 Vanilla SGD 的优化器不同，已排除。历史日志未固化的代码与权重身份不宣称完全一致。
