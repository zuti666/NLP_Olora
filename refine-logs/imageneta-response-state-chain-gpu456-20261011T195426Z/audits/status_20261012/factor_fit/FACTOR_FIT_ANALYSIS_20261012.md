# Factor, fitting and common-SGD-tail analysis (2026-10-12)

Six of six formal branches are accepted. This analysis uses three paired seeds; layers, epochs, sampled updates and fit bins are repeated observations.

S: SAM for Task1 epochs 1–25, then SGD for 26–50. R: native S_F g2 for 1–25, then the same SGD schedule for 26–50. Both start from one shared Task0-SGD50 state per seed. This is a two-task protocol, distinct from prior ten-task results.

## Paired outcomes

| Epoch | Old R−S (pp) | New R−S (pp) | FAA R−S (pp) | Clean train CE R−S | New conditional test CE R−S |
|---|---:|---:|---:|---:|---:|
| 25 | 0.326 ± 1.724 | 2.100 ± 0.227 | 1.213 ± 0.832 | 0.02040 ± 0.00414 | -0.01184 ± 0.01029 |
| 50 | 0.434 ± 0.752 | 1.837 ± 1.639 | 1.136 ± 0.463 | 0.01858 ± 0.00245 | -0.01401 ± 0.00653 |

FAA improves in all three seeds at both endpoints. At e50, old accuracy improves in seed 0 and is exactly equal in seeds 1 and 2. New accuracy is equal in seed 0 and improves in seeds 1 and 2. The gain therefore cannot be described as consistent forgetting reduction.

During common SGD, FAA increases by 0.790 ± 0.944 pp for S and 0.713 ± 1.171 pp for R. The paired advantage changes from 1.213 to 1.136 pp: persistence, not amplification.

## Fitting match (descriptive)

| Seed | Shared first-entry bins | Mean old R−S (pp) | Mean new R−S (pp) | Mean conditional test CE R−S | CE-bin range |
|---|---:|---:|---:|---:|---|
| 0 | 11 | 1.954 | 2.434 | -0.03496 | [0.10, 1.15) |
| 1 | 9 | -0.507 | 2.756 | -0.02998 | [0.10, 0.60) |
| 2 | 11 | -0.207 | 2.720 | -0.02619 | [0.10, 1.15) |

Conditional new-task test CE is lower for R in all 31 shared bins. New joint40 accuracy never decreases in these bins; old accuracy is mixed across seeds. This supports a descriptive improvement in new-task testing at approximately matched fitting, while old-task retention remains unresolved.

Each bin has width .05 and selects the first entering epoch. Fitting is approximate; epoch and cosine LR can differ between branches. The 31 bins are not 31 independent trials, and seed coverage differs. No significance claim follows from a pooled bin win rate.

## All-layer factor and effective-state evidence

| Epoch | Branch | Median Qhat (3-seed mean±SD) | Median Qhat change | Median Q drift (Frobenius norm) | LoRA displacement from start | FC displacement from start |
|---|---|---:|---:|---:|---:|---:|
| 25 | S | 0.315203 ± 0.000098 | -0.001045 ± 0.000103 | 0.000671 ± 0.000094 | 0.35075 ± 0.01480 | 2.18926 ± 0.00843 |
| 25 | R | 0.314986 ± 0.000138 | -0.001300 ± 0.000164 | 0.002430 ± 0.000199 | 0.48367 ± 0.01688 | 2.31310 ± 0.00305 |
| 50 | S | 0.315020 ± 0.000105 | -0.001227 ± 0.000112 | 0.000686 ± 0.000095 | 0.37779 ± 0.01786 | 2.33961 ± 0.01012 |
| 50 | R | 0.314826 ± 0.000177 | -0.001449 ± 0.000207 | 0.002441 ± 0.000200 | 0.47549 ± 0.01701 | 2.45904 ± 0.00518 |

All 24 Q/V modules are included, with within-seed medians and IQR preserved in JSON. Across seeds R has larger effective LoRA displacement and larger Q drift from the common start. Qhat decreases more in R, but its absolute change is small and is not evidence of a unique beneficial balancing mechanism.

Direct CPU checkpoint comparison measures ||BA_R−BA_S|| and FC differences, so the two states differ in effective weights rather than merely in factor coordinates. The FC contributes more than 95% of the squared joint displacement from the shared start in both arms; its role is not experimentally isolated.

| Epoch | BA R−S difference norm (mean±SD) | FC R−S difference norm (mean±SD) | Joint norm (mean±SD) |
|---|---:|---:|---:|
| 10 | 0.20827 ± 0.00999 | 0.21912 ± 0.01520 | 0.30236 ± 0.01684 |
| 25 | 0.29331 ± 0.00963 | 0.29873 ± 0.01338 | 0.41868 ± 0.01550 |
| 50 | 0.28239 ± 0.00902 | 0.30684 ± 0.01181 | 0.41701 ± 0.01474 |

The actual ΔQ identity is accurate: maximum relative reconstruction error 3.75e-07; maximum state-scale error 9.98e-16, over 14,400 layer-update records. Reconstructing an algebraic identity validates measurement, not a causal mechanism.

During the common SGD tail, median first-order ΔQ is approximately 5e−9, near the residual from committing FP32 factor tensors; finite-step second-order drift remains. Earlier SAM/S_F probes produce larger first-order drift. Under a loss depending on BA, equal-LR base-point SGD cancels the ideal first-order imbalance change algebraically; a gradient evaluated at displaced factors need not cancel when committed at the base. The observations are consistent with that structure, but do not show that the imbalance drives generalization.

## Claim boundaries

Supported: a response-trained Task1 state gives higher two-task FAA, mainly through new-task testing, and the difference remains after 25 common SGD epochs. Effective LoRA and FC state differences are directly observed.

Not established: consistent old-task retention, unique mediation by Qhat or A/B allocation, scale independence, head independence, exact Hessian correspondence, or transfer to ten tasks/full fine-tuning. Offline numerical and matched-step diagnostics must be reported separately.

No new training, GPU use, threshold changes or frozen-file edits were performed. The complete JSON includes the source hashes, all seed records, all-layer summaries and claim boundaries.
