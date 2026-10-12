# Independent final data and original figure review

Status: **passed_with_interpretation_and_linear_scale_limitations**.

Reviewer is the existing protocol audit agent, with prior context and publisher authorship; this is not a fresh-context review. Numerical checks were independently recomputed from raw files using a separate reader. No frozen code, source data, or original figures were modified.

Verified 24 raw jobs, all 132 new states and 132 matched old rows, 264 published per-seed rows, 352 mean/sample-SD rows, and 352 paired-statistic rows. Current hashes match all 264 published sources and 17 outputs. Old raw scans contain 1116 available states; the matched 11 radii per job enter these figures.

Task0 uses its own positive full-1259-image CE gradient; Task1 its own negative full-997-image CE gradient with an untrained new20 head. Evaluation uses each anchor’s fixed old20 head. Each scan uses one saved unit direction, held fixed across all 11 radii. Small-step finite differences have the intended sign. Raw Fisher is the unnormalized mean per-example gradient-squared trace on 256 old training images, without EWC lambda.

## Findings

- New `(0.05,1)` has higher mean signed damage and absolute CE than both old controls at all ten nonzero radii, for both directions. This intervention does not improve the tested fixed-direction robustness.
- New `(1,1)` has origin mean CE 2.998642 and accuracy 6.910%. CE is close to ln(20)=2.995732. Its low damage increment starts from a poorly fitted anchor, and is not evidence of preserving useful task knowledge.
- Counts over the correlated radius grid are descriptive; they are not independent trials or statistical significance tests.

- Task0 Fisher at r=3.5 has raw seed0/1/2 values 3.36362443, 76107577.1, 277.310756; mean 25369285.9, sample SD 43940649.1. This is a stored measurement, not an aggregation or normalization error.

## Task1 direction: three-seed means

| Group | r | Signed damage | Absolute CE | Accuracy (%) | Raw Fisher |
|---|---:|---:|---:|---:|---:|
| ewc | 0 | 0 | 0.0916534 | 97.9878 | 1451.95 |
| ewc | 0.5 | 0.276405 | 0.368058 | 89.8067 | 3729.62 |
| ewc | 1 | 0.99269 | 1.08434 | 70.0291 | 7135.06 |
| ewc | 2 | 2.47169 | 2.56335 | 25.2846 | 7866.01 |
| ewc | 5 | 3.57028 | 3.66193 | 6.19539 | 8268.54 |
| old_gam | 0 | 0 | 0.0562178 | 99.1528 | 634.51 |
| old_gam | 0.5 | 0.357613 | 0.413831 | 90.7863 | 3496.83 |
| old_gam | 1 | 1.46447 | 1.52069 | 60.4448 | 8242.15 |
| old_gam | 2 | 2.69759 | 2.75381 | 16.4946 | 11778.6 |
| old_gam | 5 | 3.90393 | 3.96015 | 6.08949 | 12448.2 |
| probe005_norm1 | 0 | 0 | 0.107097 | 98.1732 | 1007.94 |
| probe005_norm1 | 0.5 | 0.624605 | 0.731703 | 89.0389 | 3637.21 |
| probe005_norm1 | 1 | 2.15127 | 2.25837 | 36.5634 | 7221.27 |
| probe005_norm1 | 2 | 3.10019 | 3.20729 | 15.7268 | 8892.75 |
| probe005_norm1 | 5 | 4.57984 | 4.68693 | 5.45406 | 25953 |
| probe1_norm1 | 0 | 0 | 2.99864 | 6.91025 | 0.933796 |
| probe1_norm1 | 0.5 | 0.000785288 | 2.99943 | 6.77787 | 0.871114 |
| probe1_norm1 | 1 | 0.00306655 | 3.00171 | 6.77787 | 1.04373 |
| probe1_norm1 | 2 | 0.0112321 | 3.00987 | 6.80434 | 1.136 |
| probe1_norm1 | 5 | 0.249343 | 3.24798 | 6.24835 | 8983.84 |

## Actual original PNG/PDF inspection

Both real PNGs were opened and visually inspected: one row of three panels, four consistent curve colors/markers/labels, internal legends, all 11 measured points from r=0 to 5, and no data or label clipping. PDF fonts are embedded Times New Roman. Axis bounds contain all means and full mean±sample-SD bands.

The original Task0 Fisher panel has a substantial readability limitation: the real r=3.5 single-seed spike sets the linear scale near 10^7 and compresses the other curves against zero. Those values are not zero. Separate log-Fisher views retaining all individual seeds were subsequently reviewed and passed, as documented below.

Symmetric sample-SD ribbons may extend below zero although all Fisher observations are nonnegative. Lower ribbons are not negative Fisher measurements.

Anchor fit and absolute CE must accompany interpretation; lower Fisher or smaller damage alone do not establish experimental success.

## Completed supplementary log-Fisher review

Both new real log-Fisher PNGs were opened and visually reviewed. PASS: the first two panels retain identical mean±sample-SD statistics and limits; the third uses a logarithmic display of unchanged positive raw values, thick arithmetic means, and three thin individual-seed traces per group. All 132 individual values for each task, including every large peak, lie within the displayed limits. No Fisher SD band, flooring, clipping, or geometric averaging is used. Colors, labels and internal legends match; Times New Roman is embedded in the PDFs. The compressed original linear-Fisher view remains preserved.

The caption explicitly distinguishes the Fisher individual-seed traces from the first two panels’ SD bands. The readable figures resolve the original visual compression without altering measured values.

## Chinese interpretation and alignment check

ANALYSIS_ZH.md key means, r=1 damage examples, all-ten-nonzero-radius comparisons, underfitting interpretation, and single-seed Fisher peak match the independent raw recomputation. Its ImageNet-A scope and distinction from actual future learning are explicit.

All 24 saved actual gradient tensor files were additionally reloaded on CPU. Their fixed directions exactly reconstruct as the requested ±gradient/norm in FP32. Independent FP64 dot products confirm the within-anchor cosine and local Task0 slope along the Task1 direction. Old GAM mean slope=0.128276399, new (0.05,1) mean slope=0.241475575, ratio=1.88246301. Thus the rounded 1.88× statement is supported as a local diagnostic, without extrapolating it to finite-radius loss.

Readable receipt SHA256: `c978b0863e7fcbed69a1ee6478b53754296dd8c9e8940c408bb72154a955c12c`. Chinese analysis SHA256: `7aed4fe6e0ca170ad82027cc849148a081bb75c2bf85eb800400898514bc48b3`.
