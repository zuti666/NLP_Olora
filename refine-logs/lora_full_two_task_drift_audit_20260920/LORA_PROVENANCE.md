# ImageNet-R 两阶段 LoRA 表：原始来源与重新计算

2026-09-20。只读原始资产，CPU 解析日志与重新计算；未启动 GPU、训练或已停止的队列。

## 结论

六行 LoRA 表找到了精确来源，**不在当前 CopyFlatCV 的 fixedfs/ratiofs 新实验目录，而在 `/data/115-2/users/liying/low-rank-cl-new` 的 2026-05 历史实验**。重新解析 18 份原始 shell log，并逐字段核对 `table_source.csv` 后，表中的 42 个数值（每行 a11/F1/a21/a22/BWT/T2/r_eff）全部符合三 seed 均值和显示精度。

`r_eff` 的 1.560、3.131、0.0106、0.0204、0.392、0.129 **不是同一列把平方范数和范数混填**。原日志 `delta_norm` 是新 Task-2 adapter 的 BA 平方 Frobenius 能量；表先对每个 seed 开平方，再取均值。这个操作由原数据完整复算确认。

但表的语义仍需修正：Shaping + EWCLoRA 的真实 method 是 `ewclora_gam_projected`，另两个 temporal rows 是 `ewclora_gam_factored`；不能仅用是否有同一 `F` 项来解释六行。`T2` 来自新 Task-2 数据上的 endpoint Fisher buffer mass，不是累计 Fisher，也不是旧任务 Fisher 敏感度。checkpoint 保存标志为 False，未发现这批实验的权重文件，因此此次是日志级独立复核，不能声称完成了历史 checkpoint 的逐 tensor 复算。

## 精确原始路径

公共实验标识：`verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7`。

- 生成及采集脚本：`/data/115-2/users/liying/low-rank-cl-new/scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh`
- 配置根目录：`/data/115-2/users/liying/low-rank-cl-new/configs_sweeps/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/seed{0,1,2}/{setting}.json`
- 原始 shell log：`/data/115-2/users/liying/low-rank-cl-new/logs/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/shell/{setting}_s012_rho0p05_normrho0p5_lam1e7_seed{0,1,2}.log`
- trainer log：`/data/115-2/users/liying/low-rank-cl-new/logs/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/{setting}/{method}/imagenet-r/t10/seed{0,1,2}.log`
- 逐 seed 来源表：`/data/115-2/users/liying/low-rank-cl-new/result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv`
- accuracy 聚合：同目录 `summary_by_setting.csv`。
- 机制指标：同目录 `mechanism_metrics.csv`、`mechanism_metrics_long.csv`、`fisher_mass_summary_by_phase.csv`。
- 原始计划清单：同目录 `run_manifest.txt`。
- 论文旧版本：`/data/140-1/users/liying/CopyFlatCV/newmethod_design_wiki/Making_EWC_Reliable__Stabilizing_Laplace_Posterior_Surrogates_for_Replay_Free_Continual_Learning (1)/sections/A_appendix.tex`，第 15–17 行明确写 `sqrt(delta_norm)`；第 200–235 行注释保留平方能量和累计 Fisher 的旧表。

| 用户表行 | setting | 原 method | GAM Task 1/Task 2 | lambda |
|---|---|---|---|---:|
| Vanilla LoRA | baseline | ewclora | 关/关 | 0 |
| Shaping-only LoRA | gam_only | ewclora_gam_factored | 开/开 | 0，use_ewc=False |
| EWCLoRA | control | ewclora | 关/关 | 10000000 |
| Shaping + EWCLoRA | full | ewclora_gam_projected | 开/开，含 projected 机制 | 10000000 |
| EWCLoRA + Task-1 shaping | setting1 | ewclora_gam_factored | 开/关 | 10000000 |
| EWCLoRA + Task-2 shaping | setting2 | ewclora_gam_factored | 关/开 | 10000000 |

这些路径和每份文件的 SHA-256、原日志行号、原值及解析 JSON pointer 全部写入 `lora_sources.json`；原日志机制字典保存为 `lora_raw_log_records.json`。没有把 CopyFlatCV 中数值不同的 fixedfs/ratiofs 结果套入用户表。

## 配置核验

18 份配置统一：ImageNet-R，class order 不 shuffle，init/increment 20，2 个训练任务，预训练标识 `vit_base_patch16_224_in21k`，rank 10，batch 128，50 epochs/task；**Adam，LoRA 与 classifier LR 都为 0.0005，weight decay 0.005，cosine**。有 EWC 的四行 lambda=1e7，gamma=1.0。GAM 参数 rho=.05、norm_rho=.5、beta1/2/3=1、gamma=.03。seed 为 0/1/2。

这不是此前 ImageNet-A SGD/.04/.02/lambda2000 的 RQ2 序列，也不是 ImageNet-R ratioFS/lambda1200000 新序列。不能跨这些不同实验直接比较 radius 量级。

`flat_eval_param_scope=delta_w`；原日志 `endpoint_param_count=14155776`。实际模型对应 12 层 K/V 共 24 个 768×768 更新矩阵（具体实现核查见本目录独立代码审查报告）。这不是 A/B 因子本身的拼接 norm，也不是包含 classifier 的完整网络 norm。

## 原数据重新计算

| setting | mean(delta_norm) | mean(sqrt(delta_norm)) | sqrt(mean(delta_norm)) | seed0 / seed1 / seed2 的 r_eff |
|---|---:|---:|---:|---|
| baseline | 2.4352150053 | 1.5602912279 | 1.5605175441 | 1.5944633730 / 1.5567562660 / 1.5296540447 |
| gam_only | 9.8028364876 | 3.1309361853 | 3.1309481771 | 3.1258957584 / 3.1431301835 / 3.1237826141 |
| control | 0.00011224150 | 0.0105613045 | 0.0105944087 | 0.0093801133 / 0.0110878296 / 0.0112159706 |
| full | 0.00041509181 | 0.0203636275 | 0.0203738019 | 0.0200120067 / 0.0212667579 / 0.0198121179 |
| setting1 | 0.1540267510 | 0.3920524474 | 0.3924624199 | 0.3971247821 / 0.3679954094 / 0.4110371507 |
| setting2 | 0.0167949847 | 0.1287698647 | 0.1295954655 | 0.1109753447 / 0.1467487662 / 0.1285854833 |

最后一行可直接区分两种聚合：按原始逐 seed 正确计算得到 0.129；如果先均值后开根号会显示 0.130。平方能量都很小的 control/full 则确实开平方得到约 .01/.02，不是丢了平方根。

| setting | a11 | F1 | a21 | a22 | BWT | T2 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 90.726667 | 19.885229 | 84.853333 | 89.586667 | -5.873333 | 22.468632 |
| gam_only | 93.343333 | 10.945784 | 84.796667 | 93.813333 | -8.546667 | 10.470718 |
| control | 90.726667 | 19.885229 | 87.946667 | 83.450000 | -2.780000 | 24.599286 |
| full | 93.343333 | 10.945784 | 89.100000 | 88.330000 | -4.243333 | 17.931238 |
| setting1 | 93.343333 | 10.945784 | 91.823333 | 82.566667 | -1.520000 | 18.733594 |
| setting2 | 90.726667 | 19.885229 | 83.070000 | 89.213333 | -7.656667 | 41.315535 |

## Fisher 列到底是什么

采集脚本第 637–651 行明确区分：

- 表 F1 = `after_T1_fisher_total_mass` = Task-1 机制字典 `endpoint_fisher_total_mass`。
- 表 T2 = `after_T2_fisher_total_mass` = Task-2 机制字典 `endpoint_fisher_total_mass`。
- 另有 `accumulated_fisher_total_mass`，在 gamma=1 时大致等于 F1+T2；这是旧注释表末列，不能继续给新 T2 列贴累计标签。
- 另有 `endpoint_emp_fisher_trace`，数值例如 baseline seed0 在 Task1 为 121.2731，在 Task2 为 130.3508；它不是当前表中的 23.7279/22.8181。两个估计器字段不能互换。

原日志 Task2 的 `endpoint_task_start=20, endpoint_task_end=40`。当前对应代码 `methods/ewclora.py:75` 用当前任务 train_loader 计算 Fisher；默认 `fisher_mode=batch_empirical`，当前代码第 583–626 行计算 `(1/N) sum_b |B_b| (grad mean CE_b)^2`。这是**批次均值梯度平方代理**，不是逐样本 empirical Fisher。这里所谓 normalize 仅按样本数平均，未将矩阵除以 trace；`utils/mechanism_eval.py:1398` 对 buffer 做 `clamp_min(0)` 后求和，无正数 floor。累计是 `gamma*old + current`，gamma=1。由于历史未保存源码 commit，上述估计器实现属于当前匹配旧方法源代码的解释；实际日志字段、数值、参数空间和任务范围是直接记录。

## 限制及表述建议

1. r 列可保留，建议标题明确 `||Delta W_KV,new^(2)||_F`，caption 写“先逐 seed 计算 norm，再在 seeds 0/1/2 上求均值”；不要写成 A/B 因子位移，也不要把新 adapter 能量泛化到所有持续更新 LoRA 实现。
2. 必须标明 full 行包含 projected 更新；仅 C/F/P 三个符号掩盖了优化机制差异，不能将六行视为完全匹配的同一 shaping 目标开关消融。
3. T2 应明确为“当前任务的 batch-gradient Fisher proxy trace”，不是旧任务敏感度，也不是 accumulated trace。
4. r 只覆盖 backbone 的 K/V 更新。Task2 增加新 classifier；old-class accuracy 在 seen-class 候选集合中评估。很小的 backbone drift 与几 pp 的 a11→a21 下降并不矛盾：新增 classifier 竞争也会改变正确预测。不能只凭这两列判断 drift 统计错了。
5. 历史 cohort 的所有原日志 `save_ckp=False`，按实验 tag 搜索未找到任何 .pt/.pth；因此不能以别的实验 checkpoint 代替权重级审计。当前结论是“精确日志溯源和重新聚合通过”，不是“旧权重重新加载通过”。
6. 不应直接把此 LoRA norm 与 full-finetune 表的完整参数 norm 比大小；参数坐标范围、优化器/学习率/保护强度、seed 聚合和 classifier 处理都需要对齐。Full-model 结果另由独立审查报告解释。

复算命令：

```bash
python /data/140-1/users/liying/O-LoRA/refine-logs/lora_full_two_task_drift_audit_20260920/audit_lora_sources.py
```

输出：`lora_sources.json`、`lora_raw_log_records.json`、`lora_recomputed_per_seed.csv`、`lora_recomputed_summary.csv`。18 runs × 7 表指标全部从原 log 独立解析并核对，通过后才产出结果。
