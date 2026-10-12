# ImageNet-R 两任务 LoRA / full finetune：位移、Fisher 与表格来源审查

日期：2026-09-20。此次仅检查历史文件、解析原始日志和进行 CPU 计算；没有启动 GPU、训练或恢复已停止的队列。论文 Task1/Task2 对应代码 task0/task1。

## 结论

**LoRA 表的六个 r 数值未发现开方、聚合或起点扣除错误；对于这套旧实现的两任务合并前 endpoint，统计新分支 BA 正是有效 K/V 权重移动。真正需要修改的是指标解释及跨表比较。** Full 表的 drift 包含新增分类头相对补零历史头的绝对范数；两表 Fisher 列一个是当前任务 buffer、一个是累计 buffer，训练配置也不匹配。因此不能从 0.0106 对 4.747 推断 LoRA 只移动了 full 的约 1/450，更不能据此直接比较旧任务敏感度。

还发现一个独立的旧代码问题：task-end 分别相加 A/B，不是无损合并 BA。从第二次合并开始会产生交叉项。表中 accuracy/drift 均在第二次合并前计算，所以这不自动推翻当前两任务表；但会影响其后保存状态及长序列解释。

证据范围：LoRA 的18份原日志、三 seed 聚合和代码定义已核对，原实验关闭 checkpoint 保存，不能完成历史权重级复算。尚未取得每次历史运行的源码 SHA，代码对应依据当前源码及运行前版本的关键函数核对。Full 六份指标 JSON 与日志已对应，现存 final 文件有很强的时间/路径匹配；原 Task1 anchor 路径已被后续实验覆盖，所以不能把现存 anchor→final 的距离当成原表修正值。

## 1. 初始 BA 是不是零，以及应该计算什么

原 LoRA 实现位于 `/data/115-2/users/liying/low-rank-cl-new`，不是此前 ImageNet-A 的 CopyFlatCV 连续更新同一 adapter 实现。它注入 **K/V**，每个任务创建/重置新分支，旧分支和预训练 backbone 冻结。没有额外 alpha/r 缩放。

初始化是 **新 A 随机、新 B=0，所以 BA=0**；不是 A、B 都为零。BA 起点为零也不表示不能训练：B 的数据梯度一般非零。每层有效权重为

\[
W^{\mathrm{eff}}=W^{\mathrm{pre}}+\bar B\bar A+B_{\mathrm{new}}A_{\mathrm{new}}.
\]

物理位移总应从两个有效权重 endpoint 相减：

\[
r_{KV}=\left(\sum_{\ell,c\in\{K,V\}}
\|W^{\mathrm{eff}}_{\ell c,2}-W^{\mathrm{eff}}_{\ell c,1}\|_F^2\right)^{1/2}.
\]

在本表两任务实现中，第一次合并是精确的，第二任务期间旧分支与基底固定、新分支起点为0，因此差值恰好为第二任务的 `B_new @ A_new`。**需要考虑原始权重的定义，但固定的预训练权重会在相减时消去；无需把 ||W_pre+BA|| 当移动。**

如果另一套实现持续训练同一 adapter，则应计算 `s2*B2@A2 − s1*B1@A1`，不能只取最后 BA。若基底也训练，还必须加上基底的变化；若有多个分支，应按真实 forward 求总有效权重。

直接拼接 A、B 计算参数距离不适合作为与 full 权重比较的指标：`A→cA, B→B/c` 保持 BA 不变，却会改变因子范数。因子空间距离可以另作优化器诊断，但应与有效权重距离分开命名。

代码、执行时点、版本核对见 [LORA_METRIC_CODE.md](LORA_METRIC_CODE.md)。关键源码：[初始化](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:257)、[forward](/data/115-2/users/liying/low-rank-cl-new/models/vit_ewclora.py:289)、[位移枚举](/data/115-2/users/liying/low-rank-cl-new/methods/ewclora.py:345)。

## 2. LoRA 原始数值复算

原 `delta_norm` 实际是平方 Frobenius 范数。表中使用 `mean_seed(sqrt(delta_norm_seed))`，不是 `sqrt(mean_seed(delta_norm_seed))`。18份日志独立解析后，六行42个已展示数值均与原始记录及显示精度一致。

| 行 | 原日志平方范数的均值 | 逐 seed 开方后均值 ± 样本标准差 | 原表 r |
|---|---:|---:|---:|
| Vanilla LoRA | 2.4352150053 | 1.560291 ± 0.032549 | 1.560 |
| Shaping-only LoRA | 9.8028364876 | 3.130936 ± 0.010613 | 3.131 |
| EWCLoRA | 0.0001122415 | 0.010561 ± 0.001025 | 0.0106 |
| Shaping + EWCLoRA | 0.0004150918 | 0.020364 ± 0.000788 | 0.0204 |
| EWCLoRA + Task1 shaping | 0.1540267510 | 0.392052 ± 0.021965 | 0.392 |
| EWCLoRA + Task2 shaping | 0.0167949847 | 0.128770 ± 0.017887 | 0.129 |

不是混用平方/开方造成的数量级差异，三个 seed 也均存在这些差异。此处只是日志层面重算，不是加载历史 LoRA 权重重算。

原始逐 seed 表：[table_source.csv](/data/115-2/users/liying/low-rank-cl-new/result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv)。本次重新解析结果：[逐 seed](lora_per_seed_recomputed.csv)、[聚合](lora_summary_recomputed.csv)、[原日志路径/字段/哈希](lora_sources.json)。

## 3. 为什么相同 lambda 下 0.0106 和 0.392 会相差37倍

EWC 约束的是加权移动，不是统一的欧氏半径预算。以本表旧对角 Fisher buffer 为 F1，日志记录

\[
q=\sum_j F_{1,j}\Delta_j^2,\qquad d_F=q/\|\Delta\|^2,
\qquad L_{EWC}=\lambda q/2.
\]

以下为逐 seed 指标的均值；`mean(q/r²)` 与 `mean(q)/mean(r²)` 不应互换。

| 行 | r 均值 | q 均值 | q/r² 均值 | endpoint EWC penalty 均值 |
|---|---:|---:|---:|---:|
| EWCLoRA | 0.010561 | 9.9963e-11 | 9.0503e-7 | 0.000499815 |
| Shaping + EWCLoRA | 0.020364 | 4.6455e-11 | 1.1231e-7 | 0.000232275 |
| EWCLoRA + Task1 shaping | 0.392052 | 9.4409e-9 | 6.1197e-8 | 0.047204578 |
| EWCLoRA + Task2 shaping | 0.128770 | 3.8672e-10 | 2.3950e-8 | 0.001933581 |

四行均 lambda=1e7，18份日志的 `lambda*q/2` 与 penalty 字段一致。Task1 shaping/control 的 r 比为37.12；单位平方位移的旧 Fisher 权重均值约低14.8倍，**同时实际 endpoint penalty 约高94.44倍**。故大位移既伴随不同方向权重，也伴随不同实际加权移动，不能声称“同等保护预算下仅因更平坦而允许37倍位移”。全局 trace 19.885→10.946 也不能单独解释37倍变化。

这仍不是旧任务完整 Fisher 的真实路径敏感度：F1 是旧对角 batch-gradient buffer，q/r²只是该固定 buffer 下的方向加权量，不包含沿途变化及非对角项。可以解释欧氏范数为何不与 EWC 能量一一对应，不能代替新计划中的路径测量。

优化规则也有差别：`full` 实际使用 `ewclora_gam_projected`，`setting2` 也进入 projected GAM+EWC；`setting1` 的后续训练是 Adam+EWC。只写同一个 C+F+P 目标，隐去了投影更新机制。LoRA rank10 本身也不规定更新范数必须很小。

此外，r 只涵盖 K/V，新增分类头仍在学习；a11→a21 的候选类别从20变40。即使共享表征变化小，新类 logits 的竞争也可能降低旧类 accuracy。这个机制在代码上存在，但尚未单独量化其贡献，不能据此归因全部下降。

## 4. Full 表真正的位移口径问题

六行 full drift 都正确等于原 `mechanism.steps['1'].delta_norm` 的平方根。问题出在被加总的参数：历史日志包含152个张量、85,829,416个坐标，即 backbone 85,798,656 加40类头。旧头只有20类，历史比较将其补零到40类，因此新增20行贡献的是 **最终新头相对0的范数**，包含新头出生时随机初始化的能量，并非其训练前后移动。

这可作为“零扩展网络”的距离定义，但不能无说明地视为旧共享参数的训练位移。LoRA 表则仅24块 K/V有效矩阵、14,155,776个坐标，不含头。两表的绝对范数不同范围，不能直接比较倍数。

现存 final 文件与历史日志路径、结构、保存时间一致（六行时间差均<0.25秒）。在这些 final 对应原记录、旧头不动的条件下，CPU 计算显示新头能量约占总平方位移73%–90%。条件分解如下；由于原 anchor 已覆盖，**不得直接作为已验证的纠正值替换论文**。

| 行 | 已验证日志 r | 新头能量占比（有条件） | 扣除新头后的 backbone r（有条件） |
|---|---:|---:|---:|
| FineTune | 4.940566 | 81.72% | 2.112272 |
| FisherShape | 5.130791 | 73.11% | 2.660724 |
| EWC | 4.746715 | 88.81% | 1.587518 |
| FS-EWC | 4.672590 | 89.51% | 1.513072 |
| EWC + Task1 shaping | 4.773665 | 89.74% | 1.529367 |
| EWC + Task2 shaping | 4.595037 | 79.76% | 2.067432 |

例如最后一行的总范数比 EWC 小，但条件分解的 backbone 范数反而更大；因此从旧总范数直接推断“移动更少却遗忘更多”并不稳健。

Full 原始 Task1 anchor 路径现存模型是6月9日写入，a11=92.77/93.08；表中6月3–5日实验用的是92.61/93.24。不能用今天同路径文件直接相减。具体源码、CPU结果、身份核对和完整路径见 [FULL_PROVENANCE.md](FULL_PROVENANCE.md)、[full_sources.json](full_sources.json)。

## 5. 两表 Fisher 也不能横向比较

| 口径 | LoRA 表 | Full 表 |
|---|---|---|
| Task2 列原字段 | endpoint_fisher_total_mass | accumulated_fisher_total_mass |
| 任务数据 | 当前新任务训练数据 | 当前任务估计再与历史保护 buffer 累计 |
| 累计规则 | 表中未用累计列；gamma=1 | Lambda2=0.9*Lambda1+F2_processed |
| 参数空间 | 有效 K/V | 历史 full 参数及分类头 |
| 严格逐样本 raw Fisher | 否，batch均梯度平方代理 | 否，batch代理及历史处理后的累计 buffer |

因此 LoRA 的24.599不是 full 的1655.95那一种统计。Full EWC行当前任务 buffer trace 实际为873.1299，1655.9481则是累计结果。Full 的869.80/858.59可由连续Task2 endpoint/cumulative字段反推旧buffer至显示精度，但部分旧JSON的step0字段不同，不能说已恢复原始Task1 raw估计。

Full 同代可恢复 sibling checkpoint 中大量元素精确等于1e-5，85,814,036个参数的 floor 基线即858.14036。主表约858/869的量级及日志均值与此一致，提示该统计受 floor 主导。**floor的逐元素直接证据来自 sibling；主表旧anchor被覆盖，不能宣称已逐元素验证所有目标行。** 这类trace不能当未处理的曲率证据。

两表还并非只改变full/LoRA参数化的对照：LoRA是3seeds、Adam、LR=.0005、wd=.005、lambda1e7、gamma1、in21k；full是seed0、SGD、LR=.01、wd0、Task2lambda1e4、gamma.9、augreg2_in21k_ft_in1k。预训练权重、GAM半径、参数范围也不同。Full FT/FS两行来自较早运行族，其他四行来自后续统一族。

## 6. 联合两表后可以支持什么

比较每种参数化内部的 matched 后续保护分支，Task1 shaping 相对 EWC：

| 对照：仅 Task1 shaping vs EWC | 旧任务 a21 变化 | BWT 变化 | 新任务 a22 变化 |
|---|---:|---:|---:|
| LoRA（三seed均值） | +3.8767 pp | +1.2600 pp | -0.8833 pp |
| Full（seed0） | +2.36 pp | +1.73 pp | -3.35 pp |

这支持一个有限、清晰的观察：**在各自配置内，先 shaping、后续采用常规 EWC 的分支保留旧任务更好，同时有新任务性能代价。** 它不排除保护矩阵随起点变化的影响，也不等于统一参数化下的因果证明。

不能概括为“所有 shaping 都减少遗忘”：LoRA两阶段都shaping时，相比EWCLoRA，a21从87.95升至89.10，但自身起点a11也更高，BWT从-2.78变为-4.24；full对应BWT则由-7.08改善到-2.99，趋势不一致。LoRA shaping-only 的最终旧accuracy与vanilla几乎一样，BWT反而更差。仅Task2 shaping在两表都加重新旧任务取舍。

当前证据可支撑“初始化/旧解的选择重要”及“欧氏位移不能独自解释遗忘”；尚不足以证明“raw旧任务敏感度更低导致更大可容忍位移”。后一主张仍需要同坐标、同旧任务数据的raw Fisher和loss增量测量，不能用这里的T2列替代。

## 7. 独立发现：旧 LoRA 合并并不保持函数

原代码执行 Abar+=Anew、Bbar+=Bnew，因此

\[
(\bar B+B)(\bar A+A)=\bar B\bar A+BA+\bar BA+B\bar A.
\]

第一次旧因子为0，合并无交叉项；第二次已有旧因子，通常产生额外项。trainer先测accuracy，after_task先记录drift再合并，因此两任务表可按合并前endpoint理解。**若重新开展长序列、评估合并后的模型，必须单独修正/控制这一生命周期问题，不能直接沿用本表的 r 解释。** 当前任务仅审查，未修改原方法或启动重训。

## 8. 论文与后续测量的具体修正

1. LoRA保留已核验数值，明确“合并前endpoint、新分支K/V有效权重、三seed先求norm后取均值”，增加seed标准差。
2. Full旧r明确标作“含补零扩展分类头的参数距离”；在恢复正确anchor前，不宣称是已核验的纯backbone位移。若正文需要机制比较，暂不以该列比较绝对移动量。
3. 两张Fisher列分别写出实际buffer定义和数据来源；当前任务/累计buffer均不得写成旧任务raw路径敏感度。
4. 明示projected GAM-EWC更新与Adam/SGD、lambda、预训练、seed数差异。Full旧运行族与统一族的FT/FS不能默认为同一重复。
5. 后续统一保存训练起点、合并前endpoint及合并后状态的hash。主位移采用全共享backbone有效权重变化；LoRA未更新的backbone坐标自然为0。另列共同K/V子空间供方向诊断，但承认full其余层仍变化。分类头拆分成旧头变化、新头实际初始化→终点变化。
6. 原始LoRA权重未保存、full原anchor被覆盖，阻止本轮完成统一checkpoint重算。先寻找可证明身份的备份；确无备份才需要按统一协议重跑，不能用其他实验权重填补。用户已停止队列，本轮不执行这一建议。

## 9. 路径、产物与复算

- LoRA原实验根：`/data/115-2/users/liying/low-rank-cl-new`；cohort：`verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7`。
- 原配置：上述根下 `configs_sweeps/<cohort>/seed{0,1,2}/{setting}.json`。
- 原日志：上述根下 `logs/<cohort>/shell/{setting}_s012_rho0p05_normrho0p5_lam1e7_seed{0,1,2}.log`。
- 原采集脚本：[run_verify_imagenet_r_1x1_table_metrics_s012.sh](/data/115-2/users/liying/low-rank-cl-new/scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh:637)。
- Full原实验根：`/data/140-1/users/liying/CopyFlatCV`；完整逐行metrics/log/config/ckpt路径见 [full_sources.json](full_sources.json)。
- 当前论文手填full表：[6_experiments.tex](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/6_experiments.tex:2482)。
- 本次汇总：[lora_summary_recomputed.csv](lora_summary_recomputed.csv)、[full_table_raw_fields.csv](full_table_raw_fields.csv)。
- 联合对照：[joint_table_audit.csv](joint_table_audit.csv)、[旧任务保留对比](joint_retention_contrasts.csv)、[算术检查](joint_consistency_checks.json)。
- 专项报告：[LoRA来源](LORA_PROVENANCE.md)、[LoRA代码](LORA_METRIC_CODE.md)、[Full来源与计算](FULL_PROVENANCE.md)。
- 可重复脚本：`audit_lora_sources.py`、独立解析器 `recompute_lora_from_logs.py`、`extract_full_sources.py`、`recompute_full_cpu.py`。CPU脚本涉及现存checkpoint的结果须保留报告中的身份限制。

本轮没有替换原论文表格、修改源实验、调用GPU或重启队列。
