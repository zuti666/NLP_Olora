# 实验追踪

状态：只完成计划；无新代码、无新训练、无排队任务。

| Variant | Inner distance | Outer distance | Seed | Priority | Status |
|---|---|---|---|---|---|
| Original Current-G2 | eta*norm(g0) | eta*norm(g0) |2|baseline|validated_existing_reuse|
| CurrentG2-FixedInner |.05|eta*norm(g0)|2|core|planned_not_implemented_not_registered|
| CurrentG2-FixedOuter |eta*norm(g0)|.05|2|core|planned_not_implemented_not_registered|
| CurrentG2-FixedBoth |.05|.05|2|core|planned_not_implemented_not_registered|

其余SGD、原HRF、原History-G2沿用已完成成绩，公式和缓存规则保持原定义。只有得到首轮结果后，再考虑一种模式下.02/.1两个额外半径点。
