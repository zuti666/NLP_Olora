# ImageNet-A：七行两任务诊断表

状态：15/21个seed×分支结果通过核验，6个未有完成结果。GAM→SGD无EWC已纳入主表，位于FS-EWCLoRA之后。 Shaping only与Task-2 shaping各三个seed待完成；已有父起点可复用。

第一/第二任务对应代码task0/task1。数值为seed0/1/2均值±样本标准差；a21/a22为40类已见类别下的任务分组准确率，BWT逐seed用a21−a11计算。第一任务无历史保护损失；第二任务含P的行λ=2000，其余λ=0。

| Variant | Task 1 | Task 2 | a11 | B1 batch proxy | a21 | a22 | BWT | d_QV_cont | 两任务平均 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla LoRA | C | C | 87.08±0.38 | 31.441±4.235 | 82.41±1.13 | 76.38±0.40 | -4.67±1.35 | 0.3775±0.0163 | 79.39±0.74 |
| Shaping only | C+F | C+F | 87.30±0.56 | 21.933±2.982 | 待补 | 待补 | 待补 | 待补 | 待补 |
| EWCLoRA | C | C+P | 87.08±0.38 | 31.441±4.235 | 82.41±0.86 | 76.38±0.40 | -4.67±1.05 | 0.3749±0.0167 | 79.39±0.60 |
| FS-EWCLoRA | C+F | C+F+P | 87.30±0.56 | 21.933±2.982 | 83.39±0.33 | 80.84±0.82 | -3.91±0.65 | 0.5009±0.0159 | 82.11±0.40 |
| Task-1 shaping without EWC | C+F | C | 87.30±0.56 | 21.933±2.982 | 86.54±0.82 | 72.96±1.38 | -0.76±1.36 | 0.3583±0.0147 | 79.75±1.10 |
| Task-1 shaping | C+F | C+P | 87.30±0.56 | 21.933±2.982 | 86.64±0.98 | 72.96±1.38 | -0.65±1.49 | 0.3570±0.0148 | 79.80±1.18 |
| Task-2 shaping | C | C+F+P | 87.08±0.38 | 31.441±4.235 | 待补 | 待补 | 待补 | 待补 | 待补 |

缺失行中的a11/B1是其已完成、待复用父起点的测量，不代表第二任务已执行。

**指标对应**

- B1取正式anchor训练日志的in_task_fisher_final_raw_trace：原生batch-gradient平方proxy，未归一化、无路径平均。SGD/GAM的原始逐seed值与路径见individual.csv。
- 之前报告的1451.954/634.510是固定旧train256上的逐样本raw Fisher；它与本表B1不同，不能混填。两种指标均已记录。
- 实际LoRA作用在Q/V，且第二任务延续同一LoRA分支，故将旧表fresh-adapter r_KV改为d_QV_cont（持续分支的端点间位移）；该量排除分类头，不是A/B因子的参数距离，也不可与旧fresh-KV范数直接横比。
- F表示GAM实现的shaping更新，不声称代码直接优化一个显式Fisher-trace惩罚。
- FS-EWCLoRA复用正式sequence_gam的t01矩阵及B2 t1有效位移；父anchor与其它同seed分支一致，无需重训这3条。
- FS分支尚无B3同口径的endpoint-train256逐样本raw Fisher，但用户本表只需第一任务B1，因此不阻塞这行。
- 当前表不包含full-model对应实验；caption不声称尚未完成的full-model附录结果存在。

**已完成分支的配对差值**

| 对照 | 指标 | 均值±SD | seed0/1/2差值 |
|---|---|---:|---|
| fs_ewclora-ewclora | a21 | +0.977±0.867 | +0.650/+0.320/+1.960 |
| fs_ewclora-ewclora | a22 | +4.463±0.453 | +4.720/+4.730/+3.940 |
| fs_ewclora-ewclora | average_accuracy | +2.720±0.215 | +2.685/+2.525/+2.950 |
| fs_ewclora-task1_shaping | a21 | -3.257±1.175 | -2.930/-4.560/-2.280 |
| fs_ewclora-task1_shaping | a22 | +7.877±0.681 | +8.270/+7.090/+8.270 |
| fs_ewclora-task1_shaping | average_accuracy | +2.310±0.919 | +2.670/+1.265/+2.995 |
| task1_shaping-ewclora | a21 | +4.233±0.650 | +3.580/+4.880/+4.240 |
| task1_shaping-ewclora | a22 | -3.413±0.992 | -3.550/-2.360/-4.330 |
| task1_shaping-ewclora | average_accuracy | +0.410±0.737 | +0.015/+1.260/-0.045 |
| task1_shaping_no_ewc-vanilla | a21 | +4.127±0.500 | +3.580/+4.240/+4.560 |
| task1_shaping_no_ewc-vanilla | a22 | -3.413±0.992 | -3.550/-2.360/-4.330 |
| task1_shaping_no_ewc-vanilla | average_accuracy | +0.357±0.508 | +0.015/+0.940/+0.115 |
| task1_shaping-task1_shaping_no_ewc | a21 | +0.107±0.185 | +0.000/+0.320/+0.000 |
| task1_shaping-task1_shaping_no_ewc | a22 | +0.000±0.000 | +0.000/+0.000/+0.000 |
| task1_shaping-task1_shaping_no_ewc | average_accuracy | +0.053±0.092 | +0.000/+0.160/+0.000 |

上述差值仅用于三seed均完成的匹配分支。缺失分支不参与差值，也不以估计值填充。GAM起点→未来SGD与未来持续GAM是不同对照，不能混用。

**审查与复现**

- [配置实现与最少补训（历史审计）](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/six_row_imageneta_20260921/CONFIG_IMPLEMENTATION_AUDIT.md)
- [连续GAM结果复用](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/six_row_imageneta_20260921/CONTINUOUS_GAM_REUSE.md)
- [指标定义审查](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/six_row_imageneta_20260921/METRIC_MAPPING_AUDIT.md)
- [原始缺失实验清单（历史审计）](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/six_row_imageneta_20260921/MISSING_EXPERIMENTS.md)
- [LaTeX表](table_imageneta.tex) / [逐seed CSV](individual.csv) / [完整数据与来源SHA](table_data.json) / [完整性台账](VALIDITY_LEDGER.json)
- CPU复现：原始15个已完成单元格用build_table.py；补充全表用seven_row_extension_20260921/report.py。脚本不启动模型或训练，只读取结果并在CPU计算权重位移。
