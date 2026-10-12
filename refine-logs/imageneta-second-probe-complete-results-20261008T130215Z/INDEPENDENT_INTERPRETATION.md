# Independent scientific interpretation: ImageNet-A centered terminal response and joint backward replacement, 2026-10-08

New four formal conditions are complete and accepted in collector (CPLUS, CMINUS, BF, BB); FF and FB are explicit prior-run references. Independent recomputation from six raw triangular task matrices agrees with reported metrics within 1.42e-14. Each is seed2, r=1, rho=.05, rank10, ten tasks, 50 epochs/task, natural SGD. Paired statistical n=1; sample SD N/A. Smoke conditions are excluded.

## Raw endpoints, task-macro percent

| Condition | AAA | FAA | LA | BWT | Forgetting |
|---|---:|---:|---:|---:|---:|
| FF | 65.176120 | 47.584 | 54.308 | -7.471111 | 7.471111 |
| FB | 65.385122 | 48.632 | 54.035 | -6.003333 | 6.525556 |
| Center+ | 65.197483 | 48.170 | 54.145 | -6.638889 | 6.772222 |
| Center− | 2.064185 | 0.651 | 0.651 | 0 | 1.168889 |
| BF | 64.608366 | 47.339 | 53.924 | -7.316667 | 7.440000 |
| BB | 65.467711 | 48.935 | 54.844 | -6.565556 | 6.700000 |

## Center response conclusion

Define s2=(g3plus-g3minus)/2 and e2=(g3plus+g3minus-2*g2)/2. Forward correction is s2+e2, backward-corrected correction is s2-e2. Center+/− use g0±s2, both five training-gradient calls per step and zero extra diagnostic gradient calls. This directly controls e2 removal and common-response sign within the two new center arms.

Center+−FF: +0.021363 AAA, +0.586 FAA. Center+−FB: -0.187638 AAA, -0.462 FAA. Normal training survives exact deletion of e2 in this fixture; Center+ is between the two one-sided baselines. Center+−Center− is +63.133299 AAA/+47.519 FAA. This supports the common directed response being sufficient to retain baseline-like learning in this setting and its sign materially affecting stability. It does not establish that e2 is globally useless, that s2 is the sole source of all method gains, or that a one-sided readout is unnecessary in every protocol.

The deleted e2 is not numerically negligible. In the shared initial Task0 batch, ||s2||=.27705455, ||e2||=.08554301 (ratio .3087587), cosine(s2,e2)=-.0445781, cosine(s2+e2,s2-e2)=.8261915. On the Center+ training trajectory, across 2550 steps median ||e2||/||s2||=.4392544 (p95=1.762417), median cosine between corrected sides=.6928928. Same-state initial inputs, g012 and phi012 are exact across the two center arms. Later different-training-trajectory norm statistics cannot serve as a same-state causal comparison or independent replicates.

## Center− collapse chronology

Task0 online train accuracy peaks at 76.806990% in epoch5. It falls below50% in epoch7 (46.624305%) and below10% in epoch9 (6.115965%). Task0 endpoint test accuracy is6.51%; Center+ is85.34%. Center− has LA=.651%, so the low endpoint occurs through failure to acquire/maintain the first task, not merely late forgetting. BWT=0 and forgetting1.169 are misleading if read as good retention: almost nothing was learned by the task endpoints.

Every emitted required step/epoch loss is finite. Its Task0 maximum base training CE=115.5620 vs3.5640 for all normal arms. Its median residual/base norm during epochs6–8 is .953587 vs .156519 for Center+, while median U norm is13.3034 vs1.88262. Largest Task0 U norm517.078 vs4.34140; largest actual BA-plus-FC effective update norm160.536 vs.171959. CMINUS has only3/500 Task0 cos(U,g0)<0, and median cosine=.990354. Therefore failure should not be simplified to every update taking a first-order CE-ascent direction. Strong update amplification/LoRA effective-step growth accompanies the sign intervention and provides an unresolved stepsize/stability explanation. No measured HVP/Hessian cause is identified.

## Joint replacement: 2x2 descriptive factorial

First factor: forward qF=g1plus-g0 versus corrected backward qB=g0-g1minus. Second factor: forward cF=g3plus-g2 versus corrected backward cB=g2-g3minus. All four final bases areg0. Estimand interaction=(BB−BF)−(FB−FF); four cells are present and valid, but n=1.

| Contrast, pp | AAA | FAA |
|---|---:|---:|
| BF−FF, replace first with second F | -.567754 | -.245 |
| BB−FB, replace first with second B | +.082589 | +.303 |
| FB−FF, replace second with first F | +.209002 | +1.048 |
| BB−BF, replace second with first B | +.859345 | +1.596 |
| BB−FF, jointly replace both | +.291591 | +1.351 |
| interaction difference of differences | +.650343 | +.548 |

The second corrected backward estimator improves the observed FAA in both first-location conditions. First-location replacement has small sign-changing observed effects, rather than a uniform gain. The factorial interaction is descriptive and not statistically established. BB is highest of these four observed cells, but not proven superior/equivalent.

BB counters verify 2550SGD updates,10200ordinary training CE gradient passes,2550negativeg1 and2550negativeg3; training_local_positive_gradient_passes=0,training_g1_positive_gradient_passes=0,training_g3_positive_gradient_passes=0,diagnostic_gradient_calls=0. Joint replacement truly executes without either local ascent-side gradient readout. It still uses the outer norm-guided pointtheta2, four gradient calls, and g0 correction. It is a component variant; it is not complete originalGAM including itsg1/g3combination/projection, and it does not reduce per-step calls relative toFF.

## Acquisition / retention accounting

For ten tasks, FAA=LA+.9BWT exactly under these definitions; this is an accounting identity, not causal mediation.

- Center+−FF FAA+.586 = acquisition-.163 + retention+.749. Observed endpoint benefit is attributed arithmetically to better old-task endpoint retention, despite a small LA decrease.
- FB−FF FAA+1.048 = acquisition-.273 + retention+1.321.
- BB−FF FAA+1.351 = acquisition+.536 + retention+.815. Both contribute.
- BB−FB FAA+.303 = acquisition+.809 + retention-.506. BB's slight gain overFB is improved acquisition despite worse BWT and slightly greater forgetting (+.174444). Do not call it uniformly better retention.

## Defensible scope and next tests

The current data strengthen two specific claims: deletion of the even response preserves normal learning while the odd response's sign still separates success/collapse; and both local gradient readouts can jointly move to the descending sides with corrected orientation while retaining performance in this seed/protocol. They do not prove fullGAM equivalence, performance acceleration, loss-side irrelevance, a Hessian-only cause, direction-only advantage, or cross-seed/radius generality.

If later authorized, first validate paired additional seeds for FF/FB/BB/Center+, and use predeclared effective-step controls on Center+/Center− to distinguish correction sign from stepsize amplification. Any matched-direction test must maintain the same gradient calls, data/RNG and frozen geometry. No additional training was started by this analysis.
