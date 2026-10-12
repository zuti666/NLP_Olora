# Fixed Task0 positive-gradient diagnostic

The user's final clarification is to increase old-task loss locally as the
radius grows. This protocol therefore uses **positive gradient ascent**:

\[
g=\nabla_W\frac1{1259}\sum_i\mathrm{CE}_i(W_0),\qquad
u=+g/\|g\|_F,\qquad W(r)=W_0+r u.
\]

The gradient is computed once at each method's own Task0 anchor, using all 1259
Task0 training examples and its fixed 20-class head. Gradient computation uses
batch size 8 (157 full batches and a final batch of 3), sums each batch's CE
gradients in FP64, and divides by 1259 once. Global normalization uses FP64 on
CPU; the actual saved and applied direction is FP32. There is no EWC penalty,
GAM shaping term, optimizer update, momentum, or LoRA-factor gradient.

The coordinate space is the same 24 effective Q/V matrices as the random-ray
diagnostic. The effective gradient is full-matrix and may have rank greater than
the training LoRA rank; label it an effective-weight gradient diagnostic.
Both anchor and direction vary by method, so this is not an anchor-only causal
comparison. It is not actual future-task learning.

All 93 points independently restore the original anchor before applying the
same saved direction. Radii are 0–20 in steps of 0.25, then 25, 30, 35, 40, 45,
50, plus 0.001, 0.005, 0.01, 0.025, 0.05, and 0.1. Every point evaluates all
1259 training and 307 test examples and the same 256-example training-subset
raw empirical Fisher. The data/model/checkpoint/transform/head definitions
match the formal ImageNet-A experiment, not historical ImageNet-R runs.

The gradient artifact records both the full mean gradient and the normalized
direction, plus their tensor/file hashes, sample IDs/image hashes, norm, and
actual dot product. The gate requires unit norm, `g dot u = +||g||`, and an
actual increase of full-training CE for a small positive step. Central finite
differences at h = 0.0001, 0.0005, 0.001 must include a derivative within 5% of
the analytic value. Every finite-difference probe and radius restores the anchor.
No monotonic-increase gate is imposed at large radii.

The primary loss curve is **signed** change relative to the same method's own
anchor. Positive-part damage is separately retained. One fixed direction is
measured per seed/method; this is not an average over 100 random directions.

Example CLI, using the queue-selected visible GPU:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u diagnostic_fixed_gradient.py --config configs/seed0.json --method sgd --job fixed_gradient --device cuda:0
```

Use `--validate-only` for CPU source/config/reference checks. `check_worker.py`
also independently compares the batching rule with full-batch autograd on a
small CPU classifier, checks positive-direction finite differences, and rejects
an actually reversed vector. These fixtures are not experiment measurements.

Actual measurements are under `results/seedS/M/fixed_gradient/`:

- `gradient_direction.pt`: `gradient_mean_fp64`, `unit_positive_gradient_fp32`,
  and `coordinate_order`.
- `gradient_manifest.json`: sign/task/objective/sample/norm/tensor/file metadata.
- `gradient_sample_manifest.json`: all 1259 training image records and hashes.
- `finite_difference_witness.json`: full-training losses at the six probes.
- `manifest.json`, `states.jsonl`, `summary.json`: 93 measured radius states.
- `extension_receipt.json`: written only after all numeric/identity checks pass;
  top-level `gradient_task=0`, `direction_sign=1`, `status=complete`.

The sibling `sparse_train/` directory contains only the unchanged FullRunner's
initial loader identity and restoration witnesses. It does not contain this
experiment's measurements and is not a second completed job.

Existing diagnostic sources and previous experiment outputs remain unchanged.
