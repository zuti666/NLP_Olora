# Independent Stage1 Task0 g3 radius sweep

Files: `task0_g3_radius.png` (300 dpi) and `task0_g3_radius.pdf` (vector).

Left: held-out Task0 **sample accuracy** $a_{11}$ in percent. Right: sample-averaged held-out cross-entropy in nats. These are independent single-task results, not ten-task AAA/FAA. All five radius arms trained for 50 epochs and are evaluated at the **fixed final epoch 50**; no peak-based checkpoint selection is used. The outer radius is $r\in\{0,0.1,0.25,0.5,1\}$, with $\beta_1=0$, $\beta_3=1$, $\gamma=0$, and inner radius $\rho=0.05$. The horizontal axis preserves numeric radius spacing; connecting segments guide the eye and are not fitted curves.

At this audit snapshot, Stage1 has **9 of 10 planned arms completed; standard SAM is still running**. Every displayed arm is complete. All results use only seed 2 and the same 307 held-out samples; no error bars or statistical significance are claimed. A radius with the highest observed accuracy in this sweep is not an established robust optimum. CE and accuracy are different outcomes and need not rank the radii identically.

The four new radius endpoints come from `task0/radius_final_results.csv`, verified by `task0/radius_completed_audit.json`; the $r=1$ endpoint is the original completed G arm in `task0/final_results.csv` from the **same independent Stage1 campaign**. No old ten-task Task0 values are included. Inputs are frozen separately under `inputs/task0_g3_radius/`, preserving the earlier five-arm curve snapshot and its input files unchanged. The independent Stage1 data RNG and training protocol must not be pooled with the older ten-task experiment.

Reproduce using `PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B plot_task0_radius.py`. Input hashes and validation are recorded in `task0_g3_radius_manifest.json`; the merged plotting values are also available in `task0_g3_radius_values.csv`. The graph displays only sample accuracy and CE; class-macro accuracy remains a separate CSV column.
