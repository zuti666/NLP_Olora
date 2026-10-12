# Controlled GAM training-radius comparison

All six new Task0 training runs and all twelve fixed-direction diagnostics have completed. Old controls are selected from completed, hash-verified measurements at the same eleven radii. This report uses formal ImageNet-A; it does not relabel these results as historical ImageNet-R.

Legend pairs denote (GAM probe radius, gradient-difference ascent radius), in training parameter coordinates. The horizontal diagnostic radius is a different quantity: global effective Q/V weight drift.

## Anchor fit before perturbation

Mean ± sample SD over three training seeds. Each anchor appears once here; both diagnostic directions must reproduce this same origin.

| Model / training radii | Mean training CE | Training accuracy (%) | Raw Fisher trace |
|---|---:|---:|---:|
| EWCLoRA | 0.0916534 ± 0.00386 | 97.9878 ± 0.321 | 1451.95 ± 44.3 |
| FS-EWCLoRA (0.05, 0.2) | 0.0562178 ± 0.000821 | 99.1528 ± 0.0459 | 634.51 ± 36.1 |
| FS-EWCLoRA (0.05, 1) | 0.107097 ± 0.0031 | 98.1732 ± 0.159 | 1007.94 ± 40.3 |
| FS-EWCLoRA (1, 1) | 2.99864 ± 0.0263 | 6.91025 ± 1.31 | 0.933796 ± 0.835 |

**Interpretation rule:** signed damage subtracts each model's own origin CE. A smaller increment can coexist with a worse-fitted anchor or larger absolute perturbed loss. Read the origin fit, signed damage, absolute CE and accuracy together; the report does not treat smaller damage alone as a robustness improvement.

## Task0 positive-gradient direction

| r | Model / training radii | Signed old-task damage | Absolute old-task CE | Accuracy (%) | Raw Fisher |
|---:|---|---:|---:|---:|---:|
| 0.5 | EWCLoRA | 0.560041 | 0.651694 | 81.8904 | 7165.81 |
| 0.5 | FS-EWCLoRA (0.05, 0.2) | 0.62265 | 0.678868 | 81.7051 | 6773.01 |
| 0.5 | FS-EWCLoRA (0.05, 1) | 1.80379 | 1.91089 | 61.901 | 6278.98 |
| 0.5 | FS-EWCLoRA (1, 1) | 0.0228311 | 3.02147 | 6.51311 | 0.69272 |
| 1 | EWCLoRA | 1.43414 | 1.5258 | 56.394 | 14414.7 |
| 1 | FS-EWCLoRA (0.05, 0.2) | 1.86758 | 1.9238 | 44.6651 | 11255.1 |
| 1 | FS-EWCLoRA (0.05, 1) | 2.88503 | 2.99213 | 13.1321 | 12664.9 |
| 1 | FS-EWCLoRA (1, 1) | 0.0585232 | 3.05717 | 6.4072 | 1970.62 |
| 2 | EWCLoRA | 2.99551 | 3.08716 | 21.1544 | 20640.5 |
| 2 | FS-EWCLoRA (0.05, 0.2) | 3.22949 | 3.2857 | 10.3521 | 9887.69 |
| 2 | FS-EWCLoRA (0.05, 1) | 3.82866 | 3.93576 | 6.32777 | 5015.91 |
| 2 | FS-EWCLoRA (1, 1) | 0.524514 | 3.52316 | 5.58644 | 21614.9 |
| 5 | EWCLoRA | 4.14563 | 4.23728 | 6.08949 | 12363.4 |
| 5 | FS-EWCLoRA (0.05, 0.2) | 4.58674 | 4.64295 | 4.92454 | 21962.5 |
| 5 | FS-EWCLoRA (0.05, 1) | 5.43904 | 5.54614 | 5.13635 | 12152.4 |
| 5 | FS-EWCLoRA (1, 1) | 1.08762 | 4.08626 | 5.58644 | 29.3667 |

Paired differences are computed within the same training seed, then summarized across seeds. Different anchors have their own gradient directions; this comparison changes both the trained anchor and the direction it induces.

| New arm vs control | Lower / higher mean signed damage over 10 nonzero radii | Lower damage but higher absolute CE |
|---|---:|---:|
| FS-EWCLoRA (0.05, 1) vs EWCLoRA | 0 / 10 | 0 |
| FS-EWCLoRA (0.05, 1) vs FS-EWCLoRA (0.05, 0.2) | 0 / 10 | 0 |
| FS-EWCLoRA (1, 1) vs EWCLoRA | 10 / 0 | 6 |
| FS-EWCLoRA (1, 1) vs FS-EWCLoRA (0.05, 0.2) | 10 / 0 | 6 |

These counts describe the measured radius grid; radii are correlated and are not independent replicates or a probability of superiority.

## Task1 negative-gradient direction

| r | Model / training radii | Signed old-task damage | Absolute old-task CE | Accuracy (%) | Raw Fisher |
|---:|---|---:|---:|---:|---:|
| 0.5 | EWCLoRA | 0.276405 | 0.368058 | 89.8067 | 3729.62 |
| 0.5 | FS-EWCLoRA (0.05, 0.2) | 0.357613 | 0.413831 | 90.7863 | 3496.83 |
| 0.5 | FS-EWCLoRA (0.05, 1) | 0.624605 | 0.731703 | 89.0389 | 3637.21 |
| 0.5 | FS-EWCLoRA (1, 1) | 0.000785288 | 2.99943 | 6.77787 | 0.871114 |
| 1 | EWCLoRA | 0.99269 | 1.08434 | 70.0291 | 7135.06 |
| 1 | FS-EWCLoRA (0.05, 0.2) | 1.46447 | 1.52069 | 60.4448 | 8242.15 |
| 1 | FS-EWCLoRA (0.05, 1) | 2.15127 | 2.25837 | 36.5634 | 7221.27 |
| 1 | FS-EWCLoRA (1, 1) | 0.00306655 | 3.00171 | 6.77787 | 1.04373 |
| 2 | EWCLoRA | 2.47169 | 2.56335 | 25.2846 | 7866.01 |
| 2 | FS-EWCLoRA (0.05, 0.2) | 2.69759 | 2.75381 | 16.4946 | 11778.6 |
| 2 | FS-EWCLoRA (0.05, 1) | 3.10019 | 3.20729 | 15.7268 | 8892.75 |
| 2 | FS-EWCLoRA (1, 1) | 0.0112321 | 3.00987 | 6.80434 | 1.136 |
| 5 | EWCLoRA | 3.57028 | 3.66193 | 6.19539 | 8268.54 |
| 5 | FS-EWCLoRA (0.05, 0.2) | 3.90393 | 3.96015 | 6.08949 | 12448.2 |
| 5 | FS-EWCLoRA (0.05, 1) | 4.57984 | 4.68693 | 5.45406 | 25953 |
| 5 | FS-EWCLoRA (1, 1) | 0.249343 | 3.24798 | 6.24835 | 8983.84 |

Paired differences are computed within the same training seed, then summarized across seeds. Different anchors have their own gradient directions; this comparison changes both the trained anchor and the direction it induces.

| New arm vs control | Lower / higher mean signed damage over 10 nonzero radii | Lower damage but higher absolute CE |
|---|---:|---:|
| FS-EWCLoRA (0.05, 1) vs EWCLoRA | 0 / 10 | 0 |
| FS-EWCLoRA (0.05, 1) vs FS-EWCLoRA (0.05, 0.2) | 0 / 10 | 0 |
| FS-EWCLoRA (1, 1) vs EWCLoRA | 10 / 0 | 5 |
| FS-EWCLoRA (1, 1) vs FS-EWCLoRA (0.05, 0.2) | 10 / 0 | 4 |

These counts describe the measured radius grid; radii are correlated and are not independent replicates or a probability of superiority.

## Measurement scope

Loss is mean cross-entropy per image, not a dataset sum. Each diagnostic direction is computed once at the trained Task0 anchor: +∇L0 from 1259 old training images, or −∇L1 from 997 next-task training images and a paired untrained new20 classifier. All plotted evaluations use the anchor's own fixed old20 classifier. Raw Fisher is the mean squared norm of per-example true-label CE gradients on the same 256 old training images, with no EWC lambda or extra normalization. Signed negative damage is retained. The curves are fixed effective-weight rays, not real sequential optimizer trajectories.

Figures: `figures/task0_positive/Figure_task0_positive_training_radius_intervention.pdf` and `figures/task1_negative/Figure_task1_negative_training_radius_intervention.pdf`. The corresponding directories contain PNG, caption and LaTeX include files.
