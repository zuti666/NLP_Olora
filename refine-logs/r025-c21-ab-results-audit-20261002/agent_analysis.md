# Independent final-result audit: r=.25 C21 A/B

The two new full runs completed on 2026-10-02 at 10:47 UTC. Both have ten task evaluations, 500 epochs and 2,550 training steps. Independent matrix recomputation agrees with recorded results. All eleven full runs (nine old plus two new) have identical non-target effective configuration, initialization and new-head initialization, all 2,550 augmented batch identities, and all 500 epoch sampler/LR sequences. Old `us/matched` is the exact nominal-SGD identity reference for g1, so it is also the natural-step baseline.

All values below are percentages or percentage points, with task-equally-weighted AAA/FAA.

| Natural update | AAA | FAA | LA | BWT | Forgetting |
|---|---:|---:|---:|---:|---:|
| g0 | 60.947 | 38.893 | 51.884 | -14.434 | 14.434 |
| g1 | 63.828 | 45.120 | 55.577 | -11.619 | 11.619 |
| g2 | 66.586 | 50.720 | 56.824 | -6.782 | 7.036 |
| g3 | 67.179 | 50.845 | 57.429 | -7.316 | 7.316 |
| g0+(g3-g2) | 65.230 | 48.663 | 55.326 | -7.403 | 7.624 |
| A: g0+(g2-g1) | 66.105 | 50.531 | 56.923 | -7.102 | 7.102 |
| B: g1+(g2-g1)_perp_g1 | 65.897 | 49.418 | 55.640 | -6.913 | 6.957 |

Supported interpretation:

1. A improves over g0 by 5.158 AAA / 11.638 FAA, over g1 by 2.277 / 5.411, and over prior g0+(g3-g2) by 0.875 / 1.868. Transferring c21 to g0 therefore retains a substantial continual-learning benefit in this protocol. A is still 0.481 AAA / 0.189 FAA below g2. Algebraically A=g2-(g1-g0); this is a useful intervention, not an exact reproduction of g2 training. Against g0, A improves both LA (+5.039) and BWT (+7.332).
2. B improves over g1 by 2.069 AAA / 4.298 FAA, retaining 75.0% of the AAA gain and 76.75% of the FAA gain that g2 has over g1. Its LA changes only +0.063, while BWT improves +4.706 and forgetting decreases 4.662. Thus the observed benefit is mostly old-task retention, not better immediate acquisition. A residual parallel to g1 is not the sole explanation of g2's improvement over g1.
3. B remains below g2 by 0.690 AAA / 1.302 FAA. The corresponding LA deficit is 1.184, while BWT is only 0.131 worse. Removing the parallel residual retains much, but not all, of the benefit; most of this particular g2-versus-B FAA gap is associated with acquisition. Identity FAA=LA+0.9*BWT holds with ten tasks here.
4. g3 remains the largest natural AAA/FAA of these seven runs. A/B are mechanism interventions, not new performance winners. B has the lowest displayed forgetting but does not have the highest FAA or least-negative BWT; these metrics must not be conflated.

Limits: one seed, one r=.25 setting, no statistical significance claim, no Hessian causal proof. B's projection is global over trainable LoRA-factor and seen-FC gradients, not over effective BA coordinates. The natural updates are not norm-matched: mean actual effective BA+FC step divided by the same-state counterfactual g1 step is A 1.2984, B 1.1192, g2 1.3973, g3 1.4539. Consequently B rules out an explanation exclusively through adding a parallel residual; it does not by itself rule out effective-step magnitude changes. These ratios use each candidate's own evolving state and are not cross-trajectory causal decompositions.

Projection correctness: all 2,550 B steps passed their global formula and orthogonality checks; maximum relative orthogonality error 3.7041e-8. Both new runs used nominal learning rates throughout, four closures plus the clean CE pass, and the source remains free of EWC/Fisher/weight merging by the frozen configuration and runtime validators.

Reproducible independent checks and full per-stage data: `independent_audit.py`, `agent_results.json`. No source, queue or experiment output was changed by this audit.
