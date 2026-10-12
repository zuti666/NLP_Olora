# Full fine-tuning 六行表：来源、位移与 Fisher 审查

2026-09-20；只读取旧日志、JSON、checkpoint，并做 CPU 数值计算。没有启动 GPU、训练或计划队列。

## 结论

1. 表中六行 accuracy、BWT、Task2 Fisher 和 drift 都能逐字段对应历史 JSON；drift 是 `sqrt(delta_norm)`，开方本身正确。
2. **Full-model drift 统计含分类头，本次旧 LoRA 表的 effective ΔW 只统计 K/V。两表 drift 不能直接以数值大小比较。** Full 日志的坐标数是 85,829,416 = backbone 85,798,656 + 40×769 的 FC；在现存final对应历史日志的条件下，新增 Task2 分类头行相对补零旧锚点的能量占表中平方位移约 73%–90%（路径/时间强匹配，未有历史SHA）。因此 4.6–5.1 的 full drift 很大部分不是旧任务共享 backbone 的实际变化。
3. Full 表的 Task2 trace 列是 `accumulated_fisher_total_mass`，即历史累计 precision/buffer；不是 Task2 raw Fisher，更不是在 Task2 endpoint 对旧 Task1 重新估计的 Fisher。代码 Task0/Task1 对应论文 Task1/Task2。
4. 当前同路径 Task1 anchor **已被后续运行覆盖**。现存 anchor accuracy 是 92.77/93.08，旧表是 92.61/93.24；旧 Task2 日志运行于 6月3日/5日，现存 anchor 写于6月9日。因此当前 checkpoint pair 的重算只能用于定位问题，不能冒充表的原始位移修正。
5. Full 表还有目标与版本问题：FT、FS 两行来自旧运行族；EWC 和三条 ablation 来自后续统一族。FS-EWC 与 Task2 shaping 实际使用逐张量 projected GAM-EWC，不能只用 `C+F+P` 隐藏这一更新规则。

## 原始字段与全部路径

机器清单 [full_sources.json](full_sources.json) 给出每行 JSON、原始 shell log、最终 checkpoint 完整路径与 SHA256、日志配置；[full_table_raw_fields.csv](full_table_raw_fields.csv) 可直接检查字段。

统一根路径：`/data/140-1/users/liying/CopyFlatCV/`。

| 用户表行 | 源方法/运行族 | a11/a21/a22 | 原始 delta_norm（平方范数） | sqrt(delta_norm) | Task2 endpoint buffer trace | Task2 accumulated buffer trace |
|---|---|---|---:|---:|---:|---:|
| FineTune | baseline / old |92.61 / 78.46 / 92.47|24.4091972443|4.9405664902|867.5386963|1650.3566895|
| FisherShape | gam_only / old |93.24 / 81.92 / 93.44|26.3250126006|5.1307906409|858.5704956|1631.3020020|
| EWC | control / unified |92.61 / 85.53 / 90.66|22.5312992719|4.7467145766|873.1298828|1655.9481201|
| FS-EWC | factored_full / unified |93.24 / 90.25 / 90.79|21.8330962667|4.6725898886|859.0006714|1631.7321777|
| EWC+Task1 shaping | factored_s1 / unified |93.24 / 87.89 / 87.31|22.7878817377|4.7736654405|866.2412109|1638.9725342|
| EWC+Task2 shaping | factored_s2 / unified |92.61 / 82.08 / 93.86|21.1143645776|4.5950369506|859.1786499|1641.9965820|

Accuracy 字段：`cnn.matrices.final`；位移：`mechanism.steps['1'].delta_norm`；Task2 trace：`mechanism.steps['1'].accumulated_fisher_total_mass`。BWT 是 `a21-a11`。不要误用 `train_delta_norm_mean`（epoch训练期间平均）或 `endpoint_flat_emp_fisher_trace`（另一套评估器）。

例如 EWC 的两个 Fisher 字段分别为：Task2 endpoint buffer=873.12988，累计 buffer=1655.94812；额外 flat-eval Fisher 统计=8053.04090。它们不是同一数值。

主要文件族：

- FT: `outputs_logs/config_newData/base/logs_inc/ewcon_probe/baseline_lr0.01_e50/imagenetr/0/fullmodel_baseline_e50_lam2000_s0/exp_run/20/fullmodel_baseline_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json`
- FS: 同层 `gam_only_lr0.01_e50/.../fullmodel_gam_only_e50_lam2000_s0/...`。
- EWC: `ewcon_probe/stage2_control_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_lr0.01_e50/.../fullmodel_stage2_control_e50_lam10000_froms1lam2000_rho0p02_norm0p2_gamma0p03_s0/...`。
- 另外三行：`ewcon_probe_gam_factored/unified_factored_{full,s1,s2}_lr0.01_e50/.../fullmodel_unified_factored_{full,s1,s2}_e50_lam10000_s0/...`。
- 每个 `exp_run/checkpoints/final_model.pt` 均存在，字典含 `tasks=1` 和 `model_state_dict`。
- 上述短写的完整无歧义路径都在 `full_sources.json`。

论文当前手填表：[6_experiments.tex](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/6_experiments.tex:2482)。既有来源审查：[SECTION6_EXPERIMENT_AUDIT_20260917.md](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/SECTION6_EXPERIMENT_AUDIT_20260917.md:107)。历史汇总 CSV `section6_two_phase_20260917.csv` 仅含 later 版本；FT/FS 需查 `section6_two_phase_old_20260917.csv`。没有找到专门生成当前手填六行表的单一脚本；本次新增 `extract_full_sources.py` 让此表逐字段可复算。

## 为什么 full drift 被分类头主导

[`delta_fisher_drift_metrics`](/data/140-1/users/liying/CopyFlatCV/utils/mechanism_eval.py:2203) 把所有 delta tensor 的平方求和，历史名叫 `delta_norm`。[`_compute_param_drift_metrics`](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1134) 对当前参数减历史 checkpoint。[`_match_tensor`](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1194) 在头从20类扩成40类时，以零补齐历史张量，因此新增20行的最终绝对权重也计入 drift；这并非训练前新头随机初始化到训练后的位移。

历史日志证明包含头：152块、85,829,416元素；当前源码新增了默认 `ewc_include_fc=false`，因此不能用今天默认值覆盖历史实际坐标。头的新增参数并不属于旧任务共享表征。更清楚的论文指标是 backbone-only drift；本次旧 LoRA 对应有效 K/V 更新；如要跨 full/LoRA 比绝对范数，应另外报告同一 K/V 子空间（并承认 full 模型其余层仍变化）。

对现存最终模型 CPU 统计新增20行的平方范数，与历史总能量相减可得到以下**条件重建**。条件为旧头不动、历史缓冲按零填充、现存最终模型对应日志。新增头范数本身是在现存最终文件上精确计算的；其与历史总能量相除、相减的解释依赖最后一个身份条件。Task2 CE 只使用新类 logits，且 momentum=weight decay=0，EWC 对旧头从零位移开始；现存各同族终点的旧头全部逐比特一致，支持旧头未改变。但原始 Task1 anchor 已覆盖，因此下列结果不是完全闭合的原始 checkpoint pair 复算。

| 行 | 表中 drift | 新FC能量占总能量 | 条件重建 backbone drift |
|---|---:|---:|---:|
| FineTune |4.940566|81.72%|2.112272|
| FisherShape |5.130791|73.11%|2.660724|
| EWC |4.746715|88.81%|1.587518|
| FS-EWC |4.672590|89.51%|1.513072|
| Task1 shaping |4.773665|89.74%|1.529367|
| Task2 shaping |4.595037|79.76%|2.067432|

这个分解会改变“位移相近”的解读：EWC→Task2 shaping 的表中总范数略降，但条件重建 backbone 位移反而从1.588增加到2.067；新头能量下降掩盖了旧表征移动增加。Full表的旧准确率恶化与这一方向一致，但单seed关联不能据此证明因果。

## checkpoint 身份与重算边界

两个历史 resume 目录仍存在，但内容已覆盖：

- SGD `stage1_sgd_probe_lr0.01_e50/.../task_00.pt`，mtime UTC 2026-06-09 18:07:29，Fisher trace58.080825，当前 `cnn` a11=92.77。
- GAM `stage1_gam_probe_lr0.01_e50/.../task_00.pt`，mtime UTC 2026-06-09 18:07:29，Fisher trace6.335946，当前 a11=93.08。

表中使用的终点/恢复日志在6月3日和5日。当前两个 anchor 与全部相关 Task2 终点的旧FC分别存在平方差0.012521535和0.161216856；相同旧anchor族的各Task2终点旧FC彼此逐比特完全相同。这进一步排除“用当前同路径anchor直接复现历史值”的可靠性。

现存六条最终 checkpoint 的修改时间，与原始日志最后 `Saved final model` 时间（Europe/Madrid UTC+02）都吻合至0.25秒以内；路径完全相同、`tasks=1`、40类头也一致。FT示例：文件2026-06-03 11:30:43.835，日志11:30:44.072；FS示例11:29:42.144与11:29:42.252。故最终文件来源有很强的路径/时间/结构一致性，当前 SHA256、mtime 与日志记录已写入 `full_sources.json`；但没有历史执行时的SHA256，仍不是密码学身份闭环。73%–90%的比例须保留“现存final对应旧日志”的条件。

本次仍作诊断性 CPU 复算，保存在 [full_checkpoint_recompute.json](full_checkpoint_recompute.json)：当前两个已覆盖anchor→旧Task2终点，分别计算全backbone与Q/V投影距离。**Q/V是本次排查时计算的另一个子空间，本次旧LoRA表实际用K/V；加上anchor不匹配，这些Q/V数值有两重不一致，不能视为共同坐标比较，也不作为表格修正值。** 主报告故不列该数值表。未来必须以原始正确anchor计算full的K/V变化，再与LoRA K/V有效变化对齐；full全backbone变化另列。

原始anchor的K/V精确值目前未恢复。另检查相邻 `stage1_factored_*_e50` 和 `_3` anchors，也不能按原始身份匹配；没有偷换。

## Fisher 口径

日志 `fisher_mode=batch_empirical`、batch128、max100；Task2共2889样本，实际23批。其定义是对 minibatch 平均CE的梯度平方、按batch样本数加权平均：`sum_b n_b*(grad mean CE_b)^2 / sum_b n_b`。不是 `mean_i grad CE_i squared` 的逐样本 empirical Fisher。采用当前任务的20类 logits；与跨任务旧任务损伤不是同一个数据口径。

累计公式为 `Lambda2=.9*Lambda1+F2_processed`。由六行 endpoint/cumulative 反推 prior trace 得到 SGD 869.7977–869.7980、GAM858.5904–858.5906，恰对应表869.80/858.59。**这验证其累计解释，但不构成原始Task1 raw Fisher恢复。** 旧FT JSON附带的step0甚至为869.0517，不能与表869.80当作同一字段；源anchor文件也已覆盖。各row中此处采用完整连续的step1字段，不拼接今天覆盖后的step0。

这些858/869数值几乎是大维度 Fisher floor 的总量。保存的同代 sibling `stage1_factored_gam_a_e50` checkpoint 中，85,781,995/85,814,036 个元素**精确等于1e-5**，trace858.601020；SGD sibling有85,344,871个元素等于1e-5，trace868.883038。85,814,036×1e-5=858.14036。因此这代约858的统计有明显floor主导，不能作为未处理raw Fisher证据。

仍需区分“直接观测”与“对目标行的推断”：floor在可恢复sibling checkpoint直接证实；主表旧anchor被覆盖，无法逐元素核验主表所有行的同一floor。主表endpoint均值≈1e-5、累计≈1.9e-5与该机制高度一致，日志normalization=True却trace远非1也再次说明不能写成trace-normalized raw Fisher。当前源文件已移除旧floor，未找到绑定于6月3日执行版本的源码快照。

## 配置与推断边界

seed0，ImageNet-R原始类00–19→20–39，ViT-B/16 augreg2_in21k_ft_in1k，50epochs/task，LR=.01/.01，batch128，SGD+cosine，momentum/weight decay=0；EWC Task2 lambda10000、gamma.9；GAM rho=.02、normrho=.2、gamma=.03、beta1/2/3=1。无轨迹平均。预训练模型在历史日志明确打印 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`（不是从当前默认值推断）；旧LoRA表配置为 `vit_base_patch16_224_in21k`，另有Adam/SGD、学习率、weight decay、EWC lambda/gamma、seed数等差别。因此相同ViT-B/16名字不能使两表成为严格只改变参数化的配对实验。

`factored_full` 和 `factored_s2` 在 Task2 对每个参数张量执行 `gGAM - dot(gGAM,gEWC)/(||gEWC||²+eps)*gEWC + gEWC`；没有负内积才投影的判断，既移除冲突也移除同向分量。见 [EWConProbe_GAMFactored.py](/data/140-1/users/liying/CopyFlatCV/models_CL/EWConProbe_GAMFactored.py:277)。`factored_s1` Task2为标准SGD+EWC。因此这里不只是更换shaping在哪一阶段启用。

另有统一族 baseline=92.61/79.56/93.17、gam_only=93.24/82.08/93.72，和用户表旧族FT/FS不同；完整数据也保存于本次清单。统一族 joint GAM-EWC=93.24/89.78/91.91（总drift4.7140、累计trace1631.6011），它与 projected 行不同，不要混名。

建议论文暂保留可溯源accuracy，drift改名明确 `all-parameter drift including zero-padded classifier` 或恢复旧anchor后统一重算backbone-only；trace列明确 `processed accumulated precision buffer trace`。若主结论需要LoRA/full统一机制比较，需相同旧任务数据、相同未归一化逐样本估计、相同有效K/V坐标及保留完整端点hash。
