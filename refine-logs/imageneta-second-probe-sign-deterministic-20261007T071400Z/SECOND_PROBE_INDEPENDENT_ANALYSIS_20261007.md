# Independent scientific interpretation: second probe side × residual sign

Source: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z`; results snapshot `2026-10-07T13:02:33.768401+00:00`. This is read-only analysis of the 10 accepted formal endpoints. Root separately checks receipts/protocol integrity; I verified finite metrics and recomputed AAA, FAA, LA, BWT and forgetting from all 10 raw task matrices to tolerance 1e-10. No queue, source, or training changes.

## Scope and completeness

- 10/16 accepted formal conditions: r=1 is complete (8/8); r=0.25 contains N5/N6 only (2/8). B0/B1/N1/N2 at r=0.25 are running in this snapshot; N3/N4 pending. Their progress scores are not final evidence.
- One seed (2), rank 10, rho=0.05, natural SGD; all comparisons have paired seed n=1. Sample SD is N/A. Tasks do not supply independent seed replicates.
- P construction is fixed: g0 → g1 → g2; only third side, residual sign, and anchor differ. Let c±=g3±−g2. Every arm still uses four training CE gradients and one SGD update.

## Accepted raw endpoints

| r | Arm | Anchor + signed residual | AAA | FAA | LA | Forgetting | BWT |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | B0 | g0+c+ | 65.176 | 47.584 | 54.308 | 7.471 | -7.471 |
| 1 | B1 | g1+c+ | 65.335 | 47.562 | 54.761 | 7.999 | -7.999 |
| 1 | N1 | g0−c+ | 1.911 | 0.619 | 0.717 | 0.798 | -0.109 |
| 1 | N2 | g1−c+ | 1.720 | 0.554 | 0.717 | 0.181 | -0.181 |
| 1 | N3 | g0+c− | 1.934 | 0.456 | 1.440 | 2.370 | -1.093 |
| 1 | N4 | g1+c− | 1.862 | 0.673 | 0.651 | 0.362 | 0.024 |
| 1 | N5 | g0−c− | 65.385 | 48.632 | 54.035 | 6.526 | -6.003 |
| 1 | N6 | g1−c− | 65.906 | 48.917 | 55.435 | 7.366 | -7.242 |
| 0.25 | N5 | g0−c− | 64.845 | 48.015 | 55.112 | 7.886 | -7.886 |
| 0.25 | N6 | g1−c− | 65.378 | 48.720 | 55.220 | 7.222 | -7.222 |

AAA is primary; FAA is secondary. Accuracies and accuracy contrasts are percentage points. Forgetting is mean peak-to-final loss on previous tasks; BWT is mean diagonal-to-final change on previous tasks, so forgetting need not equal −BWT. The metrics are task macro averages; sample-weighted accuracy is a separate estimand.

## Main result at r=1

The dominant result is the interaction between probe side and residual sign. B0/B1 (+ side, + residual) and N5/N6 (− side, − residual) retain successful learning: AAA 65.176–65.906, FAA 47.562–48.917, LA 54.035–55.435. N1–N4, which reverse only the side or only the residual sign relative to a successful arm, have AAA 1.720–1.934, FAA 0.456–0.673 and LA 0.651–1.440. In this slice, matching the side and residual signs selects the successful four conditions; neither side nor residual sign alone describes the result.

| r=1 contrast (candidate − reference) | Interpretation | ΔAAA | ΔFAA | ΔLA | Δforgetting | ΔBWT |
|---|---|---:|---:|---:|---:|---:|
| N5−B0 | coordinated probe and residual reversal | +0.209 | +1.048 | -0.273 | -0.946 | +1.468 |
| N6−B1 | coordinated probe and residual reversal | +0.571 | +1.355 | +0.674 | -0.633 | +0.757 |
| N1−B0 | residual reversal at positive probe | -63.265 | -46.965 | -53.591 | -6.673 | +7.362 |
| N2−B1 | residual reversal at positive probe | -63.615 | -47.008 | -54.044 | -7.818 | +7.818 |
| N5−N3 | residual reversal at negative probe | +63.451 | +48.176 | +52.595 | +4.156 | -4.910 |
| N6−N4 | residual reversal at negative probe | +64.043 | +48.244 | +54.784 | +7.003 | -7.267 |
| N3−B0 | probe reversal at positive residual | -63.242 | -47.128 | -52.868 | -5.101 | +6.378 |
| N4−B1 | probe reversal at positive residual | -63.473 | -46.889 | -54.110 | -7.637 | +8.023 |
| N5−N1 | probe reversal at negative residual | +63.474 | +48.013 | +53.318 | +5.728 | -5.894 |
| N6−N2 | probe reversal at negative residual | +64.186 | +48.363 | +54.718 | +7.184 | -7.061 |
| B1−B0 | anchor g1 minus g0 | +0.159 | -0.022 | +0.453 | +0.528 | -0.528 |
| N2−N1 | anchor g1 minus g0 | -0.192 | -0.065 | +0.000 | -0.617 | -0.072 |
| N4−N3 | anchor g1 minus g0 | -0.072 | +0.217 | -0.789 | -2.008 | +1.118 |
| N6−N5 | anchor g1 minus g0 | +0.521 | +0.285 | +1.400 | +0.840 | -1.239 |

Residual-minus versus residual-plus at the + side changes AAA by −63.265 pp (g0) / −63.615 pp (g1); the same sign reversal at the − side changes AAA by +63.451 / +64.043 pp. The difference-in-differences is +126.716 / +127.659 pp. Balanced marginal side effects are only +0.116 / +0.356 pp, and marginal residual-sign effects only +0.093 / +0.214 pp; those averages conceal the crossover and should not be interpreted as side/sign irrelevance. These are descriptive factorial contrasts from the completed eight cells, with no standard error or significance claim.

N5−B0 and N6−B1 are coordinated side-and-residual reversals, not side-only effects. They improve observed AAA by +0.209 / +0.571 pp and FAA by +1.048 / +1.355 pp. N6 has the highest observed AAA and FAA among this completed r=1 slice. The small differences among successful arms establish neither equivalence nor a reproducible ranking.

## What can be said about necessity

The second *local ascent* (theta3+=theta2+rho N(g2)) is not required for good performance in this r=1 seed2 configuration: replacing it by the local descent-side probe and reversing the residual sign produces N5/N6 with successful learning and slightly higher observed AAA than B0/B1. Keeping a plus residual on the descent side (N3/N4) collapses learning, so the result depends on how the measured residual enters U.

This does not test whether the third/local probe itself is necessary: every arm computes it and no c=0 arm exists. It also cannot remove or assess the first ascent, q=g1−g0, theta2, or the entire P construction, which remain fixed. “Minus residual always works” and “descent probe always works” are contradicted by N1/N2 and N3/N4 respectively.

## Learning and retention

N1/N2 acquire Task0 at 7.17% and all later task diagonals are 0. N4 acquires Task0 at 6.51% and all later diagonals are 0; N3 is 7.17% at Task0 and 7.23% at Task6, with the other diagonals 0. Their low forgetting/BWT magnitude arises on an already poorly learned matrix. It is not evidence of useful retention; the dominant failure is acquisition. This occurred under finite completed traces, not a missing-final-score artifact.

For successful arms, the exact matrix identity FAA=LA+0.9×BWT gives an accounting decomposition (not a causal mechanism):

- N5−B0: ΔLA −0.273 pp, plus 0.9ΔBWT +1.321 pp, yields ΔFAA +1.048 pp. Thus its final advantage is compatible with better retention outweighing slightly lower average acquisition. Forgetting improves by 0.946 pp.
- N6−B1: ΔLA +0.674 pp, plus 0.9ΔBWT +0.681 pp, yields ΔFAA +1.355 pp. Both acquisition and retention terms improve; forgetting improves by 0.633 pp.
- B1−B0: ΔAAA +0.159 but ΔFAA −0.022; ΔLA +0.453 is offset by 0.9ΔBWT −0.475. N6−N5: ΔAAA +0.521, ΔFAA +0.285; ΔLA +1.400 is offset by 0.9ΔBWT −1.115. The g1 anchor therefore has no unconditional retention advantage.

The corrected descent pairs have mixed task effects: N5−B0 final task accuracies have 6 positive, 3 tied, 1 negative differences; N6−B1 have 6 positive, 2 tied, 2 negative. Their diagonal changes are also mixed. These task counts are descriptive, not a seed win count or a significance test.

## Restricted r=0.25 comparison

Only N5/N6 are accepted. The two valid same-arm cross-radius pairs are:

| r=0.25 − r=1 | ΔAAA | ΔFAA | ΔLA | Δforgetting | ΔBWT |
|---|---:|---:|---:|---:|---:|
| N5 | -0.540 | -0.617 | +1.077 | +1.360 | -1.882 |
| N6 | -0.528 | -0.197 | -0.215 | -0.143 | +0.020 |

Both completed descent-corrected arms have slightly lower AAA at r=0.25 (N5 −0.540 pp; N6 −0.528 pp). N5 gains acquisition (+1.077 LA) but loses retention (−1.882 BWT, +1.360 forgetting); N6 is mixed with small changes. At r=0.25 the available anchor contrast N6−N5 is +0.533 AAA / +0.705 FAA / +0.108 LA / −0.663 forgetting. This cannot establish a radius trend or extend the r=1 side×sign crossover: six r=0.25 cells are unfinished, so no eight-arm r=0.25 ranking or cross-radius universal claim is estimable.

## Geometry and amplitude diagnostics (hypothesis only)

Task0 begins at one shared state and common batch, so the eight repeated same-r diagnostics do not provide eight independent observations. At r=1, c+ and −c− have cosine 0.826191 and norm ratio ||c+||/||c−||=0.975176 (norms 0.286293 and 0.293581); the relative difference ||c+−(−c−)||/||c+|| is about 0.598. At r=0.25 the cosine is 0.930790, norm ratio 0.981985 and relative difference about 0.376. Their loss changes around theta2 have opposite signs: r=1 +0.312776 / −0.306003; r=0.25 +0.243347 / −0.237990. They are related, imperfect finite differences; they are not exactly opposite gradients.

Under a local smooth-gradient Taylor heuristic, c±≈±rho H(theta2)N(g2), so c+ and −c− share a leading correction direction, while higher-order terms can differ. That is consistent with the empirical successful sign product and the Task0 cosines. No Hessian/HVP is measured, and these observations do not identify a causal Hessian mechanism or bound higher-order error over training.

Natural step amplitudes differ. At Task0 the measured joint unscaled-BA-plus-FC step norms are B0 0.171912 vs N5 0.172021, and B1 0.175720 vs N6 0.175831. Across all 2550 updates the mean norms are B0 0.030366, B1 0.031508, N5 0.029491, N6 0.031309. The mismatched-sign arms have much larger trajectory means (N1 0.254426, N2 0.376274, N3 0.376515, N4 0.971176) and much larger peaks; their mean training CE is 3.523–4.529 versus 0.762–0.779 for the successful r=1 arms. These are associated trajectory diagnostics. They cannot separate initial direction choices from subsequent amplitude/optimization feedback. Same nominal LR and similar first-step norms are not amplitude controls.

The selected-side diagnostic reuses the training g3 tensor; the opposite side adds one isolated CE gradient and restores state. Its exactness flag is clone/reference equality, not an independent recomputation equivalence proof.

## Claim revision and next evidence

**Supported descriptive claim:** In the completed ImageNet-A P-branch r=1 seed2 factorial slice, pairing the + probe with the + residual coefficient or the − probe with the − residual coefficient separates successful acquisition from failure. Corrected local descent (N5/N6) achieves AAA 65.385/65.906 versus ascent B0/B1 65.176/65.335, so a second local ascent is not necessary for successful learning under this setting.

**Unsupported extrapolations:** exact/equivalent ascent and descent updates; always better descent; always better g1 anchoring; improved retention inferred from collapsed low forgetting; necessity of the third probe or first ascent; isolated direction/Hessian causality; a complete r=0.25 ranking; statistical/generalization claims from one seed.

Next evidence, if requested later: finish the six queued r=0.25 conditions to estimate their factorial contrasts; paired additional seeds to test small successful-arm differences; a budget-controlled c=0/no-third-probe comparison to test the probe itself; amplitude-controlled direction comparisons to test geometry apart from step magnitude. No extra experiments were launched by this analysis.

Structured contrasts, matrix decomposition, and diagnostic statistics are saved in `/tmp/second_probe_contrasts_20261007.json`.
