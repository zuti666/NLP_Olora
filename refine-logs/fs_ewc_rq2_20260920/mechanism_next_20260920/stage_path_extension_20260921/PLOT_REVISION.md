# Compact Times New Roman revision

The original PDF and PNG have been regenerated from the verified measurement sources, with unchanged measurement CSV hashes. Size: 7.6 × 1.85 inches (26% less height). Actual Times New Roman is embedded in the PDF.

The four y labels are Weight drift, Loss of Task 0, Accuracy of Task 0 and Average Fisher. Every x label is Number of Tasks. Subplot letters are removed. Axis limits use a 4% margin around the measured range (the Fisher lower limit remains zero), independent of legend size. Legend corners are lower right, upper left, lower left and upper left; the accuracy panel is slightly wider to fit its internal legend.

The second panel still plots signed cross-entropy change relative to the method's Task0 anchor. Average Fisher still denotes normalized directional empirical-Fisher sensitivity along each endpoint chord; the caption defines the quantity. Only presentation changed.

Regenerate on CPU:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/stage_path_extension_20260921/report_stage_paths.py
```

Previous scripts and figures are archived under plot_revisions/. Completed GPU campaign manifests remain the historical execution records; this presentation revision is tracked separately in PLOT_REVISION.json.
