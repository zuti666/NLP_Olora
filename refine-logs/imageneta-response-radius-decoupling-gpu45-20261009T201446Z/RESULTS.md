# Current-G2 radius decoupling results

Queue status: completed. Updated UTC: 2026-10-10T00:37:40.829302+00:00.

All performance values are task-equal percentages. Only accepted full 10-task runs enter FAA/AAA/BWT. Original is source-attributed reuse; smoke accuracies are excluded.

| Method | Inner radius | Outer radius | Status | FAA | AAA | BWT | Updates |
|---|---|---|---|---:|---:|---:|---:|
| CurrentG2-Original | eta*norm(g0) | eta*norm(g0) | completed_reused | 45.252 | 62.870 | -9.652 | 2550 / 2550 |
| CurrentG2-FixedOuter | eta*norm(g0) | 0.05 | completed | 43.869 | 63.017 | -11.280 | 2550 / 2550 |
| CurrentG2-FixedInner | 0.05 | eta*norm(g0) | completed | 45.267 | 62.865 | -9.220 | 2550 / 2550 |
| CurrentG2-FixedBoth | 0.05 | 0.05 | completed | 44.214 | 63.102 | -10.813 | 2550 / 2550 |

New variants submit only g2 from BASE; three gradient reads plus one passive post-update loss forward per normal step. Fixed radii are global L2 distances over trainable LoRA and classifier parameters.
