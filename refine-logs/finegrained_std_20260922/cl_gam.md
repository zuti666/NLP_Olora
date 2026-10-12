# CL-LoRA / GAM-EWCLoRA 多种子审计

当前表的 CL-LoRA 和 GAM-EWCLoRA 各五个数据集均只有一个已核实同协议 seed，因此 SD 和样本方差均不可估计，不能填 0。

| 方法 | 数据集 | n | FAA | AAA_stage | SD |
|---|---|---:|---:|---:|---|
| CL-LoRA | aircraft | 1 | 29.284000 | 26.528942 | -- |
| CL-LoRA | cars196 | 1 | 34.508000 | 42.169252 | -- |
| CL-LoRA | cub200 | 1 | 78.291000 | 80.878104 | -- |
| CL-LoRA | flowers | 1 | 97.822000 | 98.334387 | -- |
| CL-LoRA | het5 | 1 | 66.984000 | 61.986800 | -- |
| GAM-EWCLoRA | aircraft | 1 | 46.660000 | 43.650512 | -- |
| GAM-EWCLoRA | cars196 | 1 | 51.741000 | 55.436196 | -- |
| GAM-EWCLoRA | cub200 | 1 | 74.224000 | 76.474856 | -- |
| GAM-EWCLoRA | flowers | 1 | 86.747000 | 91.755568 | -- |
| GAM-EWCLoRA | het5 | 1 | 83.370000 | 81.155667 | -- |

另找到 RNG 控制协议的 GAM-EWCLoRA CUB/Flowers，三次运行只改变 seed、loader_seed、prefix、outputs_root，源代码快照相同，均完整结束。可以单独报告如下统计，不能给原表历史 seed0 的 74.22/76.47 或 86.75/91.76 添加这些 SD：

| 新协议数据集 | seeds | FAA mean (sample SD) | AAA_stage mean (sample SD) |
|---|---|---:|---:|
| cub200 | 1,2,3 | 73.719333 (1.137575) | 76.342905 (0.897817) |
| flowers | 1,2,3 | 82.912667 (1.263098) | 89.248787 (1.153878) |

协议差异：新 campaign 显式启用 `ewc_preserve_fisher_rng=true`、`ewc_isolate_clean_rng=true`、`deterministic_loader_rng=true`。历史 seed0 未固化这些设置；Flowers 还未记录 GAM rho/norm_rho/gamma，因此无证据证明只变 seed。

CL-LoRA 搜索只找到正式 seed0 和 smoke，烟雾测试不能计为复现实验。未找到 Sep19 以后的新细粒度 CL/GAM-EWC 正式多 seed 结果。GAM-Het5 的 83.37/81.1556666667 已从 5×5 CNN 矩阵重新计算；不同方法、超参 sweep 与副本全部排除。

所有原始 metrics SHA256、配置及 receipt SHA256、独立种子统计保存在 `cl_gam.json`。
