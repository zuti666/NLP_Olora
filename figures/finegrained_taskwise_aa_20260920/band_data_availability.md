# 误差带数据可用性核查（2026-09-20）

结论：现有冻结的 35 条曲线不足以直接为整张图生成真实的跨 seed SD。每个方法×数据集格子当前均只选择了一次运行，不能从一条曲线的任务波动推导 seed 方差。此前审计中存在两个局部同配置重复组，不能据此为其他 33 条曲线补造 SD，也不应把其他协议或 EWC λ 的结果拼在一起。

本次只读已有来源/审计元数据；没有训练、广泛搜索、新选择运行或修改绘图和原始数据。以下“未确认”限定于这些已有元数据，不声称全磁盘不存在其他结果。

## 当前冻结选择

[独立指标审计](independent_metric_audit.json)中 35/35 条记录均为 `single_run: true`，每格只有一个 seed。除 EWCLoRA Cars/Flowers=seed1、Het5=seed2 外，其余均为 seed0。当前曲线中心是这些选定运行，不是多 seed 平均。

CL-LoRA 六项正式运行以及新增 GAM-EWC/Het5 的 [collected.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/collected.json)只列 seed0。当前所保留的五项 CL 曲线没有可由该完成清单直接组成的多 seed 组。

## 可确认的局部重复例外

[既有 seed_groups.json](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/ewc_cars_flowers_wiki_20260918/seed_groups.json)中有以下两组，并且组内包含当前选中的 seed1 来源；组内 `shared_logged_parameter_differences={}`。对应 [per_run.json](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/ewc_cars_flowers_wiki_20260918/per_run.json)均标为 verified，40 epochs，保存了完整 10 阶段下三角。

| 数据集/方法 | 共同配置与 seeds | 当前 FAA（seed1） | 已存两 seed FAA 均值 ± 样本 SD |
|---|---|---:|---:|
| Cars / EWCLoRA | α=0 wrapper，probe/Fisher RNG 均隔离，λ=80000，seeds 1/2 | 34.555 | 33.3495 ± 1.704834 |
| Flowers / EWCLoRA | α=0 wrapper，probe/Fisher RNG 均隔离，λ=10，seeds 1/2 | 53.434 | 52.3460 ± 1.538664 |

这两组允许在明确局限的情况下另做 n=2 的逐阶段 mean±sample-SD；需要把该两条中心曲线改为两 seed 平均，端点将不再等于当前表中的 best-seed FAA。本次未复算或添加误差带。不能沿用 best-seed 中心再把这组 SD 标成常规 mean±SD。

同一清单中还有 Flowers seeds0/1/2 的组，但它们属于只隔离 Fisher RNG、不同 λ 或 damping=.2 的配置，不等于当前双 RNG 隔离的选中配置，不能合并成当前曲线的三个或更多 seeds。其他 EWC 的历史 λ 搜索与 native/wrapper、RNG 实现也不能充当重复 seed；[EWC 审计说明](/data/140-1/users/liying/O-LoRA/refine-logs/ewclora_best_audit_20260919/REPORT.md:22)已记录这些差异，Het5 native 只有一条已审计完整运行。

## 不能复用的另一工程多 seed

[baseline_audit.md:103](/data/140-1/users/liying/O-LoRA/refine-logs/cllora_comparison_20260918/baseline_audit.md:103)记录 low-rank-cl-new 的 InfLoRA/SDLoRA seed0/1/2，但 rank10、batch64、Adam、20 epochs、不同预训练标识等与当前 rank16、batch128、SGD、40 epochs 不同。Flowers 另有 classifier-head offset 失败标志。[协议对齐记录:44](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/PROTOCOL_ALIGNMENT_AND_SEED0_PLAN_20260918.md:44)明确这些三 seed 结果不是当前旧 seed0 的重复，不能合并计算 mean±SD。

## 图注建议

- 若保持当前 35 条中心曲线和表格端点：目前不能为整图声称真实多 seed SD/CI；单条运行也不能把 SD 写成 0。
- 若用户确认固定 ±0.2 pp 仅用于参考样式，应在图内或紧邻图注写明：**“阴影为固定 ±0.2 个百分点的示意范围，非跨 seed 标准差或置信区间。”** 不将该范围称为实验测得误差。
- 若只接受真实多 seed 误差：保留单 seed 曲线无误差带；上述两个 n=2 组可另行展示并改变中心为均值，其余需要同配置重复实验。是否扩展实验不在本次绘图授权范围内。
