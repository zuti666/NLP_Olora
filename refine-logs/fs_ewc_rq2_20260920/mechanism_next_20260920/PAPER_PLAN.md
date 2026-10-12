# RQ2 figure organization

A remains the completed controlled continuation table, with a22, base-relative damage, distinct loss/head definitions and inherited-Fisher caveat.

| ID | Type | Description | Data source | Priority |
|---|---|---|---|---|
| B | 2x2 data plot | Gmean, G95, raw neighborhood empirical Fisher mean/Q95/sampled maximum, prefix-safe radius vs epsilon; train and test damage both visible | execution/results_seed0_recovery1/*/neighborhood/states.jsonl and report_seed0/B*.csv | main |
| C | 2x2 data plot | Task0 accuracy, local CE damage, directional path sensitivity, effective-weight displacement; only t9 path currently measured, do not invent a path curve | report_seed0/C_current_FP32_task_trajectory.csv and C_path_summary.csv | main |
| Bridge | existing appendix | 2 anchors x 2 realized directions; preserve source chain identity | report_seed0/B_actual_direction_bridge.csv | appendix |

One training seed. Random directions are not independent training replicates. No titles inside plots, vector PDF plus PNG, reproducible scripts, preserve old outputs.
