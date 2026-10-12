# 当前 EWCLoRA 行是否是各数据集最好结果

核查日期：2026-09-19。结论：不是。当前表是已归档可用配置的汇总，没有统一执行“每个数据集选取最高 FAA”的规则。前五个数据集都找到更高的有效单次 FAA；Het5 当前值是本次可采信记录中的最高 FAA，但不能由此断言其配置已充分调优。

本次仅核查现有记录，未修改主表、训练配置或启动训练。FAA 为 CNN 最终阶段的任务宏平均；AAA 为各阶段任务宏平均再取平均。每一对 FAA/AAA 必须来自同一次运行。

跨 CopyFlatCV 和 Flatness_CV 检索3009份指标文件，按日志实际配置筛出77个正 lambda EWC 候选。对下述 FAA/AAA 极值涉及的11个不同运行再次独立复算完整 CNN 下三角，并核对最后日志矩阵、非有限训练状态及指标/日志 SHA；全部通过。检索最大值仅限本次可读记录，不能代表未运行配置的全局最优。

## 当前结果与已见较高结果

下表“已见最高 FAA 运行”允许不同 EWC 实现、lambda 和 seed，仅用于回答现表是否取最高分，不是新的受控方法比较。保留三位小数以避免两位显示掩盖差异。

| 数据集 | 当前表 FAA / AAA | 已见最高 FAA 运行的 FAA / AAA | 该运行配置 | 当前表是否最高 FAA |
|---|---:|---:|---|---|
| CUB-200 | 60.21 / 69.97 | 60.406 / 69.400 | 历史 alpha=0 wrapper，lambda=200000，seed=0 | 否；更高 FAA 的 AAA 反而较低 |
| Cars | 29.54 / 39.74 | 34.555 / 43.942 | alpha=0 wrapper，probe/Fisher RNG 均隔离，lambda=80000，seed=1 | 否 |
| Aircraft | 40.21 / 38.94 | 41.832 / 39.695 | 历史 alpha=0 wrapper，lambda=200，seed=0 | 否 |
| Flowers | 48.45 / 62.42 | 53.434 / 65.449 | alpha=0 wrapper，probe/Fisher RNG 均隔离，lambda=10，seed=1 | 否 |
| Pets | 64.17 / 71.09 | 66.426 / 73.713 | alpha=0 wrapper，Fisher RNG 隔离，lambda=200000，seed=0 | 否 |
| 5Datasets | 73.87 / 73.07 | 73.868 / 73.068 | native one-pass，lambda=50000，seed=2；即当前记录 | 本次可采信范围内是；未证明充分调参后的最优 |

这些运行的骨干标识、rank=16、batch=128、基本任务划分及单数据集40 epochs/Het5 30 epochs大体对应现表。alpha=0 wrapper 仍进行两次 CE probe，而 native EWC 是一次 CE backward。RNG 隔离、seed 及 lambda 也有差异；历史数据/权重/源码未全部逐字节绑定，因此不能把这些替换描述为统一协议下的性能提升。

## 当前行实际选择规则

- CUB、Aircraft、Pets、Het5：采用已有 native EWC 完整有效记录。在这四个数据集各仅找到一个正式完整 native 记录，没有形成 native 多 lambda 搜索；Pets/Het5 为 seed 2，其余为 seed 0。
- Cars、Flowers：采用历史 seed 0、固定 lambda=2000 的 alpha=0 wrapper 记录。生成表格的脚本明确按 family、seed、status、lambda 过滤，没有按最高 FAA 排序。
- native 配置生成器明确把各数据集 lambda 称为固定参考值（anchors），而非最优性声明。

源码证据：[表格生成规则](../cllora_comparison_20260918/build_comparison.py:48)；[native 配置生成器](/data/140-1/users/liying/CopyFlatCV/scripts/generate_as1_fivearm_native_controls.py:58)。

## 仅在原历史 seed 0 lambda sweep 内比较

共复算35个正 lambda 点。Cars lambda=160000 的训练非有限；Het5 lambda=100000、200000 有已记录的并发输出覆盖风险；Pets lambda=200、2000 的 metrics SHA 完全相同，惩罚效应需进一步核实。风险记录保留在原始汇总并标注，不当作独立可靠最优证据。

Cars 在原来的同一历史实现和 seed 下，lambda=80000 的 **30.697 / 40.681111** 同时优于当前 lambda=2000 的 **29.543 / 39.735614**。因此即便不切换 seed/RNG 实现，现表也不是最高分。

Flowers 同一历史搜索中，最高 FAA 是 lambda=10 的 **48.652 / 62.408981**；最高 AAA 是 lambda=200 的 **48.576 / 62.422186**。当前 lambda=2000 为 **48.454 / 62.420013**。AAA 的后两者显示两位均为62.42，但原始值不同。

其他历史搜索：CUB 最高 FAA 对应 lambda=200000；Aircraft 最高 FAA 对应 lambda=200、最高 AAA 对应 lambda=2000；Het5 排除有覆盖风险的两条后，历史 wrapper 最高 FAA 为 lambda=50000 的72.262，低于当前 native 73.868。这些 wrapper 搜索不能替代 native 搜索结论。

## FAA 最优与 AAA 最优不能拼接

以下是本次核查中的明确实例：

- CUB 的最高 FAA 运行为60.406 / 69.399929；另一个双RNG隔离、seed1的 wrapper 运行 AAA 达70.6471，但 FAA 为60.160。
- Aircraft 的最高 FAA 运行为41.832 / 39.695065；历史 lambda=2000 的41.321 / 39.757967有更高 AAA。
- Flowers 的最高 FAA 运行为53.434 / 65.449362；Fisher RNG 隔离、lambda=20000、seed0 的52.199 / 65.924706有更高 AAA。
- Pets 的最高 FAA 运行为66.425556 / 73.712628；Fisher RNG 隔离、lambda=20000、seed0 的66.375556 / 73.739711有更高 AAA。
- Het5 的当前最高可信 FAA 运行为73.868 / 73.067933；双RNG隔离、lambda=50000、seed1 的73.072 / 73.074567有略高 AAA。两条 AAA 显示两位都为73.07。

## 对论文表的含义

当前表可描述为“archived configurations”，不能称“每个方法在各数据集调参后的最佳结果”。若需报告已有搜索的最高值，应另列探索性搜索结果，明确选取 FAA 还是 AAA，并带出同次运行的另一指标。不能把按测试分数挑选的单次最大值称为独立验证集选参结果，也不能将最佳 seed 当作多 seed 均值。

正式方法比较应先固定 EWC 实现及统一选参规则，再报告同一 seed 集合的结果。当前证据能回答“不是都选了最高分”，不能证明最佳超参数已经找到。

## 可复查文件

- [当前六格原始来源与配置](current_table_sources.json)
- [35点历史搜索复算与风险标记](historical_sweep.json)
- [Cars/Flowers 按实现族的 FAA/AAA 最大值](cars_flowers_by_family.json)
- [六数据集极值与11份原始矩阵的独立复核](summary.json)
- [跨目录全部候选与排除原因](extra_candidates.json)
- [极值运行逐条日志、配置与来源链接](extra_runs.md)
- [Cars/Flowers 既有专项审计](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/EWC_CARS_FLOWERS_WIKI_AUDIT_20260918.md)
- [历史搜索与 Het5 并发覆盖证据](/data/140-1/users/liying/CopyFlatCV/newmethod_design_wiki/FISHERSHAPE_5DATASET_HET5_AUDIT_AND_PLAN.md:169)
