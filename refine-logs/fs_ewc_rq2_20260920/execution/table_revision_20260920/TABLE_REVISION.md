# 表格修订与完整计划的关系

日期：2026-09-20。输入为用户六行ImageNet-R表，以及08:25:49 UTC版[完整RQ2计划](FULL_RQ2_PLAN_082549.md)。本轮完成只读结果核验、重新聚合与表格导出，没有启动补充训练。

**结论：要得到附件B3要求的新四格机制表，应补齐四格；当前两行只能作为seed0部分结果。B1/B2的全部实验不是生成四格表的前置条件，但它们与3个训练seed是附件机制主张的正式证据要求。** 当前15/15是已经排入队列的精简版完成数，不能代表附件整个计划完成。

附件的四格与旧ImageNet-R的六行不是同一设计。四格统一后续SGD，交叉改变Task0 shaping与后续EWC；旧六行还改变后续GAM开关和projected更新。完整执行附件自然产出四格主表，以及单列的完整方法序列结果，不会自动得到一个严格同设置的旧六行复刻表。

| 附件B3分支 | Task0 shaping | Task1优化器 | Task1 EWC λ | seed0 | seeds1/2 |
|---|---|---|---:|---|---|
| A | off | SGD | 0 | 未执行 | 未执行 |
| B | on | SGD | 0 | 未执行 | 未执行 |
| C | off | SGD | 2000 | 完成 | 未执行 |
| D | on | SGD | 2000 | 完成 | 未执行 |

因此B3为seed0 **2/4**、正式seed0/1/2合计 **2/12**。A/B能观察没有保护矩阵参与时的起点作用；D−C则包含起点和其对应Fisher的共同变化。只做C/D无法计算完整shaping×protection交互项。详见 [B3状态CSV](B3_FOUR_CELL_STATUS.csv)；空白是未运行，不能当0。

最小补足是复用两个seed0 anchor，各增加一条λ=0、50epoch的Task1续训，再对两个端点评估相同指标。当前 `train_continuation.py:105` 将λ写死为2000，且命名未区分λ，后续扩展需要支持独立配置/输出目录并维持既有配对RNG；重启原队列不会自动执行缺失格。先核对新增分支与既有C/D的数据、head初始化、顺序和增广收据，再合成四格表。

正式执行应继续补seed1/2四格及B1/B2，复用anchor和通过身份/随机协议核验的完整SGD序列Task1端点。多seed、test诊断和负结果均应按预先范围保留，不以seed0结果是否有利决定删减。full-model支持性复验在附件中是条件项；只有论文主张覆盖该参数化时才升级为必须。

**当前已有Fisher和参数移动，可直接更新两行表。** 下表使用论文Task 1/2编号，对应代码Task0/1；BWT使用同环境复评精度，负值表示遗忘。Fisher两列均针对旧Task1，而不是把新任务Fisher混进来。

| 分支 | a11 (%) | a21 (%) | a22 (%) | BWT (pp) | 旧任务raw trace，起点 | 旧任务raw trace，续训后 | 有效Q/V位移 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C：普通起点→SGD+EWC | 85.67 | 84.36 | 74.80 | −1.30 | 1465.497 | 2008.388 | .374950 |
| D：GAM起点→SGD+EWC | 87.62 | 84.04 | 72.05 | −3.58 | 761.901 | 940.413 | .349876 |

raw trace是同一256张旧训练图、固定旧20类head、确定性变换、有效Q/V坐标的逐样本梯度平方均值，没有trace归一化/floor。参数移动是24个有效矩阵的联合Frobenius范数：`sqrt(sum ||B2 A2 − B1 A1||_F²)`，不含分类头；它不是LoRA因子位移，也不是日志中未开平方的 `delta_norm`。当前GAM的Fisher和移动较小，但BWT及新任务准确率没有提高；不能将低trace直接写成保持能力收益。

已输出 [当前完整两行LaTeX](imageneta_current_complete.tex) 和 [原始精度CSV](imageneta_current_complete.csv)。这些是实际完成结果，不是预期四格的替代品。统计样本n=1，跨seed标准差不可估计。

**用户原六行表的数字也已找到准确来源，应更正早期未溯源判断。** 它属于 `/data/115-2/users/liying/low-rank-cl-new` 的ImageNet-R、fresh K/V LoRA、Adam lr=.0005、EWC λ=1e7、seeds0/1/2；18份运行日志对应42个表内均值都可复算。旧checkpoint未保存，审计是日志与匹配实现级别，不声称历史权重逐tensor复验。

| 原变体 | a11 | 批梯度Fisher₁ | a21 | a22 | BWT | 批梯度Fisher₂ | r_KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla LoRA | 90.73 | 19.885 | 84.85 | 89.59 | −5.87 | 22.469 | 1.560 |
| Shaping-only | 93.34 | 10.946 | 84.80 | 93.81 | −8.55 | 10.471 | 3.131 |
| EWCLoRA | 90.73 | 19.885 | 87.95 | 83.45 | −2.78 | 24.599 | .0106 |
| Shaping+EWCLoRA（projected） | 93.34 | 10.946 | 89.10 | 88.33 | −4.24 | 17.931 | .0204 |
| EWC+Task-1 shaping | 93.34 | 10.946 | 91.82 | 82.57 | −1.52 | 18.734 | .392 |
| EWC+Task-2 shaping（projected） | 90.73 | 19.885 | 83.07 | 89.21 | −7.66 | 41.316 | .129 |

这组历史数据对其设置下Task-1 shaping的保持收益有支持：相对EWCLoRA，a21平均+3.8767pp，BWT平均+1.2600pp，平均准确率+1.4967pp，BWT和平均准确率的差值在三个seed均为正。a22均值−.8833pp，三个seed差值为+.84、+.42、−3.91pp，不能说新任务完全无代价。按seed标签配对的BWT差为+.63、+.64、+2.51pp，均值±样本SD为1.26±1.08pp；不等于已证明历史随机过程严格配对。

更强的“低Fisher普遍带来低遗忘”主张并不成立：shaping-only与完整shaping+EWC的Fisher较低，但各自BWT比对应普通分支更负。Task-1 shaping起点准确率先高2.6167pp，其最终旧任务优势3.8767pp不能全部当作新增保持收益，扣除起点后才得到BWT的1.26pp。六行还涉及factored/projected实现差异，不宜称为唯一变化只是同一目标项开关的严格消融。

原表Fisher是 `endpoint_fisher_total_mass`：各任务自身数据、自身head的batch-mean-gradient-square proxy trace。第二列Fisher₂因此是新任务代理，不是旧任务在续训后的敏感度，也不是累计Fisher；不能与上方新表的raw逐样本旧任务trace按数值拼接。r_KV则确实是逐seed先对 `delta_norm` 开平方再平均，为第二任务合并前fresh新adapter的有效K/V更新范数。

已输出修订列名/更新规则的 [旧六行均值版LaTeX](imagenetr_legacy_verified.tex) 与 [均值±样本SD版LaTeX](imagenetr_legacy_mean_sd.tex)。原始数值保留，更新Fisher定义、K/V范围、projected规则和三seed聚合说明。逐项来源见 [USER_TABLE_PROVENANCE.md](USER_TABLE_PROVENANCE.md)，Fisher/移动字段详解见 [FISHER_DRIFT_AUDIT.md](FISHER_DRIFT_AUDIT.md)。当前环境没有pdflatex/tectonic，已核对表格列数和数据导出，未声称完成PDF编译。

完整附件与已完成作业的逐项差异见 [FULL_PLAN_GAP_AUDIT.md](FULL_PLAN_GAP_AUDIT.md)。现有诊断和训练结果应按范围复用，补充实验的目的在于完成既定对照与证据，不是确保产生正结果。
