# 已有 ImageNet-R empirical Fisher 结果

更新：2026-09-20T16:28:54.764304+00:00。只解析既有 JSON，未运行模型。

已有逐样本 raw EF；仍缺目标两阶段表在统一有效 Q/V 坐标、旧任务 local20 损失下的两端点 EF。

| 数据组 | 起点训练 | 三 seed raw trace（均值±样本标准差） | 坐标/损失 |
|---|---|---:|---|
| imagenetr_endpoint_region_pilot_s345_v1 | sgd | 188.978218 ± 17.361223 | A/B因子+分类头；旧样本上的40类CE |
| imagenetr_endpoint_region_pilot_s345_v1 | ratio | 155.687034 ± 14.015789 | A/B因子+分类头；旧样本上的40类CE |
| imagenetr_endpoint_region_heldout_s345_v2 | sgd | 616.062105 ± 44.358006 | A/B因子+分类头；旧样本上的40类CE |
| imagenetr_endpoint_region_heldout_s345_v2 | ratio | 494.634147 ± 22.624542 | A/B因子+分类头；旧样本上的40类CE |

- 每个测量使用160张样本（每旧类8张），seeds3/4/5。这里的 ratio 是 Ratio-FS，不是 GAM。
- pilot_s345_v1 使用新版六行表的同一批 Task0 anchors，但诊断样本未从训练中排除；六个 SGD/Ratio-FS JSON 各自 complete，整套汇总因 SAM 分支缺失而 incomplete。它可作训练样本描述性诊断，不能冒充 held-out 证据。
- heldout_s345_v2 是另一次80/20 fit/diagnostic划分后重新训练的独立 cohort；summary complete，eligible 为 true，但其记录的两组核心 claim_support 均为 false。不能把该组 Fisher 拼入旧六行表。
- 所有这些 trace 都是 Task0 起点先扩成40类头后测得，既不是第二任务结束后的旧任务 trace，也不是有效 Q/V 权重坐标下的 trace。
- 因子坐标会受 A/B 等价缩放影响，且包含头；与有效权重坐标的 raw trace 不可横向比较绝对值。新评估需要明确坐标、样本、头范围和端点。
- 已有训练保护 sidecar 的归一化 trace≈1 是另一指标，不能替代 raw EF。

## 原始来源

- [imagenetr_endpoint_region_pilot_s345_v1 summary](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/summary.json)
- [sgd seed3](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/sgd_seed3.json)：175.599114978
- [sgd seed4](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/sgd_seed4.json)：182.738662635
- [sgd seed5](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/sgd_seed5.json)：208.596875218
- [ratio seed3](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/ratio_seed3.json)：144.810885541
- [ratio seed4](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/ratio_seed4.json)：150.746072885
- [ratio seed5](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_pilot_s345_v1/diagnostics/ratio_seed5.json)：171.504145028
- [imagenetr_endpoint_region_heldout_s345_v2 summary](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/summary.json)
- [sgd seed3](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/sgd_seed3.json)：581.100397938
- [sgd seed4](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/sgd_seed4.json)：601.125494118
- [sgd seed5](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/sgd_seed5.json)：665.960422919
- [ratio seed3](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/ratio_seed3.json)：476.846625232
- [ratio seed4](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/ratio_seed4.json)：486.957664087
- [ratio seed5](/data/140-1/users/liying/CopyFlatCV/result_summary/imagenetr_endpoint_region/imagenetr_endpoint_region_heldout_s345_v2/diagnostics/ratio_seed5.json)：520.098150265
