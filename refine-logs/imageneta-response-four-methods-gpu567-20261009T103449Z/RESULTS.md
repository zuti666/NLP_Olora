# ImageNet-A response-method validation

Updated (UTC): 2026-10-09T19:19:44.505270+00:00. Queue: completed.

Seven new formal runs are registered. The cl10 history-fusion row reuses the verified existing result; it is not a new run. All rows use seed 2 and rank 10. The predictor/relocation radius is dynamic, eta × global gradient norm; the old fixed GAM radii are inactive. EWC/Fisher, weight fusion and projection are disabled.

## All 200 classes in one task

Final test accuracy is measured at epoch 50. BWT is undefined; per-epoch accuracy and partially completed training are not final results.

| Method | Status | Final test accuracy (%) | Updates | GPU | Run |
|---|---|---:|---:|---:|---|
| SGD | completed | 63.000 | 2350/2350 | 5 | formal_single200_sgd_s2 |
| Current-response relocated gradient | completed | 63.660 | 2350/2350 | 6 | formal_single200_current_g2_s2 |
| History-response direct fusion | completed | 63.530 | 2350/2350 | 5 | formal_single200_history_fusion_s2 |
| History-response relocated gradient (final-gradient cache) | completed | 61.490 | 2350/2350 | 6 | formal_single200_history_g2_s2 |

## Ten tasks of 20 classes

FAA and AAA use equal task weights. BWT uses final accuracy minus the accuracy immediately after learning each earlier task. Partial runs have empty metric columns.

| Method | Status | FAA (%) | AAA (%) | BWT (pp) | Origin | Updates | Run |
|---|---|---:|---:|---:|---|---:|---|
| SGD | completed | 39.779 | 60.426 | -13.044 | new_formal | 2550/2550 | formal_cl10_sgd_s2 |
| Current-response relocated gradient | completed | 45.252 | 62.870 | -9.652 | new_formal | 2550/2550 | formal_cl10_current_g2_s2 |
| History-response relocated gradient (final-gradient cache) | completed | 34.593 | 56.051 | -13.994 | new_formal | 2550/2550 | formal_cl10_history_g2_s2 |
| History-response direct fusion | completed | 41.297 | 60.483 | -10.757 | reused_existing | 2550/2550 | formal_history_fused_k10_s2 |

The two task layouts have different task/head and learning-rate schedules; do not interpret their difference as a controlled algorithm effect. Results are single-seed measurements. No performance benefit is claimed for pending runs.

Provenance: [reused result pin](gates/REUSED_RESULT_PIN.json); [reuse audit](gates/REUSED_RESULT_AUDIT.md); [queue state](runtime/queue_state.json).
