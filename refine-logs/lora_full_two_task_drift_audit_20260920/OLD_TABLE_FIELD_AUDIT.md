# 旧 ImageNet-R LoRA 六行表：字段复核与修改建议

2026-09-20。再次直接读取已定位的18份原始日志和原始 `table_source.csv`，独立解析 accuracy 与 mechanism 字典；未扫描其他仓库、未启动GPU、未加载替代实验的权重。**原表42个数值全部复核通过，主要应修改指标说明及实验可比性的表述，而非替换数字。**

## 原始数据是否确认

源仓库：`/data/115-2/users/liying/low-rank-cl-new`。

逐seed原始汇总：[table_source.csv](/data/115-2/users/liying/low-rank-cl-new/result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv)。原始日志位于同实验标签的 `logs/.../shell/{setting}_s012_rho0p05_normrho0p5_lam1e7_seed{0,1,2}.log`，精确路径与此次读取的SHA保存在 [old_table_field_recheck.json](old_table_field_recheck.json)。更完整的配置、原始行号、字段指针见 [lora_sources.json](lora_sources.json)。

下列值按原表精度显示，均在此次重新读取日志后得到：

| setting／表行 | a11 | F1原字段 | a21 | a22 | BWT | T2原字段 | r_eff |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline／Vanilla |90.73|19.885|84.85|89.59|−5.87|22.469|1.560|
| gam_only／Shaping-only |93.34|10.946|84.80|93.81|−8.55|10.471|3.131|
| control／EWCLoRA |90.73|19.885|87.95|83.45|−2.78|24.599|0.0106|
| full／Shaping + EWCLoRA |93.34|10.946|89.10|88.33|−4.24|17.931|0.0204|
| setting1／Task-1 shaping |93.34|10.946|91.82|82.57|−1.52|18.734|0.392|
| setting2／Task-2 shaping |90.73|19.885|83.07|89.21|−7.66|41.316|0.129|

这些是 seeds 0/1/2 的均值。原始 cohort 记录 `save_ckp=False`，没有找到与其匹配的历史权重；已确认的是日志来源、公式与重新聚合，不能表述成“历史checkpoint逐tensor复算通过”。

## 简洁字段审计表

代码 `task=0/1` 对应论文 `Task 1/2`。

| 原表字段 | 原始CSV字段／原始日志字段 | 判断 | 建议列名与解释 |
|---|---|---|---|
| a11 | `R1_A_after_T1`；第一次 `=> Accuracy['00-19']` | 正确 | 保留 `a11 (%)`；第一任务结束时旧20类 accuracy |
| a21 | `R2_A_after_T2`；第二次 `=> Accuracy['00-19']` | 正确 | 保留 `a21 (%)`；第二任务结束，在已见40类候选中评价旧20类样本 |
| a22 | `R2_B_after_T2`；第二次 `=> Accuracy['20-39']` | 正确 | 保留 `a22 (%)`；第二任务结束，在已见40类候选中评价新20类样本 |
| BWT | 逐seed `a21−a11` 后求均值 | 正确 | `BWT (pp)`；由未对均值再次舍入的源数计算，不用表中两位小数反算 |
| `tr(F1)` | `after_T1_fisher_total_mass`；`mechanism[0].endpoint_fisher_total_mass` | 数值正确，原名称不足以明确估计器 | 建议 `T_batch,1`；第一任务训练数据、当前任务head、有效K/V坐标的 batch-gradient Fisher proxy trace |
| `T2` | `after_T2_fisher_total_mass`；`mechanism[1].endpoint_fisher_total_mass` | 数值正确；不能解释为旧任务敏感度或累计Fisher | 建议 `T_batch,2`；第二任务训练数据、当前任务head、同一K/V坐标的同类proxy trace |
| `r_eff` | `theta2_minus_theta1_norm_sq`；`mechanism[1].delta_norm`，先逐seed开平方再平均 | 正确；原日志是平方范数，表中是范数 | 建议 `r_KV = ||ΔW_KV,new^(2)||_F`；排除分类头，指fresh新adapter在第二任务合并前的有效权重更新 |
| Training objective／C,F,P | 每行配置中的 method、use_ewc、GAM启用阶段 | 阶段开关基本对应，但无法表达所有实现差异 | 建议 `Training settings`，并增列或脚注明确 factored/projected 实现；不要描述为唯一变化是同一F项开关的完整匹配消融 |

这里的Fisher数值是训练保护所用的批次均值梯度平方代理：按样本数加权平均各batch的 `(grad mean CE)^2` 再对有效K/V坐标求和。其概念不是逐样本 empirical Fisher trace，也不是另一个日志字段 `endpoint_emp_fisher_trace`。当前匹配代码未做trace/max归一化、未加正floor；历史运行未绑定源码commit，因此估计器细节属于匹配实现核查，字段名称、数值及任务范围则直接由日志确认。详见 [LORA_METRIC_CODE.md](LORA_METRIC_CODE.md)。

## BWT为什么与显示出来的accuracy相减差0.01

Vanilla的原始三seed平均为 `a11=90.7266666667`、`a21=84.8533333333`，平均差为 `−5.8733333333`，显示为 **−5.87**。如果先把两个均值分别打印为90.73、84.85再相减，会得到−5.88；这是二次舍入引起的差异，不能据此把原表改成−5.88。

Shaping-only同理：`84.7966666667−93.3433333333=−8.5466666667`，显示为 **−8.55**；表面上的 `84.80−93.34=−8.54` 不是应使用的计算。其余四行在两位小数下恰好没有这个差别。这里“未舍入”特指未对三seed均值作显示舍入；原始日志的每seed accuracy 本身已保存到两位小数，不能声称恢复了更高精度的原始预测计数。

表下注释可写：`Values are means over seeds 0, 1, and 2. BWT is computed per seed before averaging and display rounding; effective-update norms are computed per seed before averaging.`

## 保留这些数值时必须同步补充的含义

1. **更新范围。** 本实验在K/V上添加LoRA：新任务A随机初始化、B为0，所以新BA起点为0；旧分支固定，合并前的有效移动确为新BA范数。初始A本身不为0，不应统计因子A/B相减来替代有效权重移动。这条结论限定于此旧fresh实现，不适用于新CopyFlatCV持续adapter的 `BA_end−BA_start`。
2. **变体实现。** `full` 是 `ewclora_gam_projected`；`gam_only/setting1/setting2` 是 `ewclora_gam_factored`；`baseline/control` 是 `ewclora`。至少将full行标成“Shaping + EWCLoRA (projected)”并解释另三行factored，否则C/F/P排列容易让读者误认为这是一组严格的单因素消融。
3. **不能把列间关系扩大成机制结论。** T2是新任务Fisher代理，不能用它支持“旧任务路径敏感度下降”。相同EWC规则与lambda也不保证各起点的保护矩阵相同。a21由40类竞争决定，r仅覆盖K/V，不包含新增classifier；很小的r与数个百分点BWT下降可以同时出现。
4. **不能把full-finetune与本列直接作同范围比较。** 此列是K/V有效更新，不是全参数范数；full表应另行对齐backbone/head和起止点范围。旧六行的参数移动数字无需因两表量级差异而直接改动。

最小可行修订：**保留42个数字，重命名两个Fisher列、明确K/V有效更新定义及三seed聚合，注明projected/factored差别，并收窄旧任务敏感度与完全匹配消融的解释。** 若正文需要严格一致的算法比较，应另用已完整匹配的新实验表，不要以新实验权重替换这批历史数字。
