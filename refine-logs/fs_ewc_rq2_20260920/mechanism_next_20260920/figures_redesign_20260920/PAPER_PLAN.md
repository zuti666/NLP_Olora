# B/C publication redesign

User-authorized CPU redraw of existing ImageNet-A seed0 data. Method display names are EWCLoRA (historical SGD+EWC) and FS-EWCLoRA (historical GAM+EWC); raw data keys remain sgd/gam. Move dataset, stage-definition and estimation details to captions. Do not change measurements, omit contradictory values from summaries, enforce monotonic trends, or infer an enlarged tolerance radius.

| Figure | Panels | Layout | Source |
|---|---|---|---|
| B | mean positive damage; Q95 positive damage; raw neighborhood Fisher; prefix-safe tolerance radius (user confirmed) | 2x2, coherent typography | execution/report_seed0 B CSVs and results_seed0_recovery1 raw states |
| C | (a) actual cumulative displacement, (b) signed loss change, (c) fixed-old-head accuracy, (d) final chord directional sensitivity | 1 row x 4 columns; legends in panels a/b/c; d category ticks EWCLoRA / FS-EWCLoRA | C_current_FP32_task_trajectory.csv, C_path_summary.csv |

C horizontal coordinate T is total learned tasks 1…10; explanatory T=1/dataset footer removed from image per user, retained in caption. Sbar is final Task0→Task9 straight-chord train256 quantity only. No title or prose finding in plot. All PDF output vector, PNG300dpi, source CSV+scripts+captions retained. Root renders/inspects before independent skill-required figure review.
