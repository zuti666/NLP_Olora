# 细粒度 CL-LoRA 对比表与配置核验

检查日期：2026-09-18，Europe/Madrid。**可以作为注明配置差异的单次运行结果对照；当前证据不足以称为严格统一协议、公平预算或统计显著的方法比较。**

本次保留用户原表的有效来源，补入六项已完成 CL-LoRA；另提供增加 Vanilla LoRA + GAM 的扩展表。没有重新挑选各 sweep 的最高分，没有混合另一工程的多 seed 结果，也没有启动新训练。

## 可直接使用的文件

- [原七方法 LaTeX](table_main.tex)：可替换用户的 `table*`；已恢复可见表注。
- [扩展八方法 LaTeX](table_expanded.tex)：增加五组完整的 Vanilla LoRA + GAM。
- [PDF 预览](comparison_preview.pdf)：结果表及配置对照，共两页。
- [扩展结果表](table_expanded.md)、[逐项结果 CSV](results.csv)、[逐项配置 CSV](configurations.csv)。
- [全部来源与文件哈希](source_manifest.json)、[复算核验记录](verification.json)、[描述性差值](deltas_vs_vanilla.csv)。

原七方法表共 42 个方法×数据集单元，其中 41 个已完成并通过原始矩阵、任务数、有限性与来源 SHA 核验；扩展表共 46 个合格完整结果。每个单元均有 FAA 和 AAA 两个数。GAM-EWC/5Datasets 尚在运行；纯 GAM 的旧 Het5 只有 3/5 阶段，均不得填作最终结果。结果 CSV 还保留最终 BWT 和平均遗忘，供后续分析使用。

## 指标口径

设 a[t,i] 为学完第 t 个任务后、在第 i 个任务上的准确率：

- FAA = 最终矩阵行的任务等权平均。
- AAA_stage = 每阶段已见任务准确率的等权平均，再对阶段取等权平均。
- 原 JSON 的 `cnn.final.AAA` 是整个下三角单元直接平均，会给后期阶段更多权重。表中全部使用重新计算的 AAA_stage，不能混用。
- 这些是任务宏平均百分数；不是各任务测试样本数加权的 pooled accuracy。5Datasets 是五数据集任务流，不是前五列的平均。

## 已确认的共同基础设置

日志支持同一模型标识 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`、rank16、batch128、SGD 基优化器、momentum=0、weight_decay=0、cosine、无回放、不打乱类别。GAM 使用 SGD 基优化器但改变更新方向；这不表示其更新与普通 SGD 相同。

| 数据集 | 任务规模 | 任务数 | 常用 LR | 每任务 epochs |
|---|---|---:|---:|---:|
| CUB-200 | 20+9×20 | 10 | .01 | 40 |
| Cars196 | 16+9×20 | 10 | .05 | 40 |
| Aircraft | 10×10 | 10 | .05；SDLoRA=.02 | 40 |
| Flowers102 | 12+9×10 | 10 | .0025 | 40 |
| Pets | 5+8×4 | 9 | .0025 | 40 |
| 5Datasets | CUB200→Aircraft100→Cars196→Flowers102→Pets37 | 5 | .02 | 30 或 40，见下表 |

所有类别均纳入任务流。CUB 是 10 个任务、每任务 20 类。新 CUB 实际 train/test 为 9430/2358；不能默认为另一公开标准 split。5Datasets 共635类，其统一预处理不等同于五个单独任务的全部预处理设置。

## 影响比较的明确差异与证据缺口

| 项目 | 实际情况 | 对比较的影响 |
|---|---|---|
| seed | EWC/Pets 与 EWC/5Datasets 是 seed2；其余入表结果为 seed0 | 不是完整配对的同 seed 比较；所有主表单元只有单次运行 |
| 5Datasets 训练轮数 | Vanilla、CL-LoRA、EWC 为30；InfLoRA、OLoRA、SDLoRA为40；运行中的GAM-EWC为30 | 同列训练预算不同，不能直接宣称等预算优劣 |
| Aircraft LR | SDLoRA=.02，其余=.05 | 已声明的方法特定超参；不能写全方法相同学习率，独立调参规则也未完整归档 |
| EWC λ | CUB/Cars/Aircraft/Flowers/Pets/Het5依次200000/2000/200/2000/200000/50000；GAM-EWC全部2000 | 直接相减不能隔离 GAM 或 EWC 单组件的作用；纯GAM与GAM-EWC也不能当作仅λ不同的配对 |
| EWC 实现 | Cars/Flowers 是历史 α=0 两次 probe wrapper；其余是 native 单次 backward | 整行并非同一执行路径，额外 probe 与 RNG 消耗差异不可隐去 |
| 适配器与分类头 | CL-LoRA 保留 shared/specific 分支、KD、训练集原型分类器；InfLoRA冻结A，SDLoRA有额外缩放参数 | 属于方法差异，可在方法比较中披露；相同rank不保证相同参数、内存、训练或推理成本 |
| 历史 GAM 参数 | GAM-EWC的Flowers/Pets未记录rho/norm_rho/gamma；新增纯GAM五项的GAM专属参数也未在日志字段中核实；部分Fisher与RNG字段未记录 | 不能用当前默认值补成历史实参，也不能声称完整复现 |
| 权重 | 34个相关历史日志均有相同模型ID，但无固定revision/SHA；新CL-LoRA已锁定并核验权重SHA | 模型ID一致已确认；历史权重字节一致仍未知 |
| 数据与代码 | 新运行有冻结源码、路径/标签manifest；历史缺少完整不可变记录 | 不能证明新旧逐样本、逐代码完全同一 |

CL-LoRA 六项为本框架的 `cllora_aligned` 移植版，应在论文中写明。其方法参数为 Q/V adapters、shared blocks 0–5、specific blocks 6–11、KD系数5、temperature2、正交项1e-4；训练集原型用于全已见类别分类，不能将其与仅线性头的普通LoRA当作单组件消融。

数据专项检查没有发现可直接证明历史 Flowers 标签错配的证据：旧实现按目录排序索引，新实现按官方 image ID 命名，但在当前8189张规范文件副本上逐项重放得到 **0个路径差异**。当前三个数据变换构造函数的新旧 AST 也一致。历史目录与运行时源码缺失，因此既不能断言历史数据不一致，也不能证明完全一致。Flowers高分的原因目前不能分配给某一个因素。

## 可以从当前表陈述什么

可以报告：在各自已记录的配置下，新CL-LoRA得到CUB78.29/80.88、Cars34.51/42.17、Aircraft29.28/26.53、Flowers97.82/98.33、Pets90.17/91.92、Het5 66.98/61.99（FAA/AAA）。可以列出来源明确的描述性差值；不可将这些差值称为已控制混杂的因果增益。

1. CL-LoRA在CUB、Flowers、Pets的归档数值较高，但Aircraft FAA为29.28，低于Vanilla40.66与GAM-EWC46.66，不能概括为全面领先。
2. 加入纯GAM后，Cars FAA为52.20，高于GAM-EWC51.74；Pets FAA为80.92，高于78.82；Aircraft AAA为43.82，高于43.65。GAM-EWC也不是所有指标最优。
3. Het5的CL-LoRA30轮与InfLoRA40轮并非等训练预算；不能把66.98与77.75的差距全部归因于方法本身。
4. 另一项目现有三seed结果使用rank10/batch64、不同checkpoint、部分Adam/20轮以及Pets4任务，已在基线专项报告列为独立补充，未混入当前主表均值。

建议正文用“under documented/archived configurations”，保留单seed、轮数、λ、wrapper与移植版的表注，暂不加粗最优结果，不作显著性结论。

## 如果要形成严格比较

优先冻结当前共同权重文件、数据/标签清单、任务流和评估口径，再为需要支持的主要结论补同一 seed 列表的对照。Het5先统一30或40轮；声明各方法分类头及adapter存储/计算差异，并实际测量预算。若研究EWC组件贡献，应在同一GAM实现、相同Fisher/RNG处理与预先固定参数下比较λ=0和λ=2000；目前这张混合历史来源表不能替代该实验。多seed扩展应使用相同配置，不把异协议已有seed直接合并。

## 专项证据

- [四个历史基线24项审核](baseline_audit.md)：每项原始JSON/log/YAML及哈希、指标复算、参数预算解释。
- [EWC与GAM-EWC审核](ewc_audit.md)：11个完成结果、wrapper/native与Fisher参数差异、Het5运行快照。
- [数据与checkpoint审核](data_audit.md)：旧/新变换、Flowers路径重放、CUB分割与预训练证据。
- [构建脚本](build_comparison.py)：仅读实验源，生成独立表格、结果与核验文件。

这是检查时刻的快照；运行进度以原队列 `state.json` 和完成receipt为准。
