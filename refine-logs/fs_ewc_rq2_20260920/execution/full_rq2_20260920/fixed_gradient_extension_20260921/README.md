# Fixed Task0 positive-gradient experiment

At each formal seed/method's own Task0 checkpoint, compute the gradient **g of full Task0 training mean CE** in the 24 effective Q/V matrices, normalize once as **u = +g / ||g||**, and evaluate `theta(r) = theta0 + r*u`. The old 20-class head and all other model state stay fixed. This is a fixed positive-gradient direction, with no iterative training or direction updates.

Each of seeds 0/1/2 × SGD/GAM contributes **one direction and 93 radii**: 0–20 in steps of 0.25; 25, 30, 35, 40, 45, 50; and 0.001, 0.005, 0.01, 0.025, 0.05, 0.1. Directions are specific to each method/seed; they are not paired random directions.

The full 1,259-image Task0 training set determines the gradient. Every radius records full train1259/test307 CE and accuracy, plus unnormalized empirical Fisher trace on the unchanged 256-image training subset. Fisher has no additional EWC coefficient, floor, or normalization.

## CPU publisher

```bash
python publish_fixed_gradient.py --check-only
python publish_fixed_gradient.py --execute
```

Use `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python` for `--execute`.

Default / `--check-only` uses only the standard library and is read-only. It reports the six expected jobs and validates hashes for completed receipts. `--execute` exits with code 2 while any job is missing, or code 1 if validation fails; it writes no completion receipt in either case. There is no GPU launch in this publisher.

Before publication, all six jobs must pass raw-state coverage, finite metrics, formal checkpoint/head/data/coordinate identity, origin recovery, and fixed-direction checks. The publisher independently loads saved tensors on **CPU** with `weights_only=True`; rehashes the gradient/direction; verifies `u=+g/||g||`, unit global Frobenius norm, and positive `g·u`; and verifies that gradient sample records and labels are exactly the formal 1,259 Task0 training records.

A separate finite-difference witness at h = 0.0001, 0.0005, 0.001 verifies initial ascent and agreement with the analytic directional derivative. **No positivity or monotonicity condition is imposed on the 93-point plotted curve.** A local gradient does not guarantee monotonically increasing loss at larger radii.

## Figures and data

Both figures use Times New Roman, one row of three panels, blue EWCLoRA / orange FS-EWCLoRA, and internal legends:

1. Signed training CE change `D_train`, including any negative values without clipping.
2. Task0 training accuracy (%).
3. Raw empirical Fisher trace.

- Full range: `figures/full/Figure_B1_fixed_task0_gradient_full.{png,pdf}`.
- Companion near origin: `figures/near_origin/Figure_B1_fixed_task0_gradient_near_origin.{png,pdf}`, radius 0–2.
- `figures/all_per_seed.csv` exports signed `D_train`, `D_test`, positive parts `G_positive_train`, `G_positive_test`, train/test CE and accuracy, and raw Fisher.
- `figures/all_mean_sample_sd.csv` reports three-seed arithmetic means and **sample SD**, after validating the complete seed cohort. One direction is not treated as multiple replicates.
- Each figure directory contains plotted data, mean/SD CSV, caption, LaTeX include, and validation JSON with source hashes.

The final completion receipt is **`report/fixed_gradient_complete.json`**, duplicated as **`publisher_receipt.json`**. Queue success requires `status="complete"`, `expected_jobs=validated_jobs=6`, `gradient_task=0`, `direction_sign=1`, and `automatic_numeric_and_layout_checks="passed"`. Input and output hashes are recorded. Rendered-image visual review remains a separate step.

This evaluates a fixed gradient in **effective weight coordinates**. It is not the actual LoRA optimizer's factor-space update direction, not a Task1 update, and not a future-task forgetting experiment. Existing measurements and figures are preserved.
