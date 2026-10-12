# Minimal G2 study: independent design review

Status: **approved as two-task, single-seed mechanism screening**, subject to the implementation and pairing gates. This is not acceptance of any future numerical result.

The six branches constitute a useful minimum: B0 (SGD), the four cells R25/S_F/F_S/S_S that independently toggle the two distances, and R100 retaining the previously studied larger outer radius. All branches must train Task1 from one identical SGD Task0 parent. Code task IDs 0/1 mean the first/second tasks in prose.

| Branch | First descending distance a | Outer response distance r | Submitted gradient |
|---|---:|---:|---|
| B0 | none | none | g0 |
| R25 | 0.05 | 0.25 | g2 |
| S_F | eta*norm(g0) | 0.25 | g2 |
| F_S | 0.05 | eta*norm(g0) | g2 |
| S_S | eta*norm(g0) | eta*norm(g0) | g2 |
| R100 | 0.05 | 1.0 | g2 |

The fixed/coupled four-cell design estimates the effect of these registered **policies**. It does not isolate learning-rate decay from gradient-norm decay, because h=eta*norm(g0) contains both and can differ in initial scale from the fixed values. Initial-scale-matched eta-only/norm-only policies belong to a later round.

## Identified contrasts

1. R25 minus S_F and F_S minus S_S assess fixed versus coupled inner distance at each outer policy.
2. R25 minus F_S and S_F minus S_S assess fixed versus coupled outer distance at each inner policy.
3. Their difference assesses an interaction, descriptively for seed2 only.
4. R100 minus R25 assesses the larger outer distance while retaining a=0.05. It establishes neither a monotone curve nor an optimum.
5. Each G2 branch minus B0 measures the resulting natural-step training effect. Update magnitude and gradient-read budget also change; the result is not a direction-only effect.

q=(g0-g1)/a is a diagnostic response estimate for positive a. Its normalized direction equals N(g0-g1); no r/a or extra gamma belongs in the final g2 update. On a quadratic loss, inner distance cancels from the normalized response to first order/exact arithmetic, whereas outer distance changes g2-g0. Actual FP32 displacements and indistinguishability must be reported.

## Endpoints and leakage restrictions

The primary endpoint is current-Task1 **development cross-entropy at epoch50**, not best-epoch performance. The proposed fixed 10% stratified development split is taken exclusively from original training data, with an explicit split seed, exact manifest, class counts, disjoint IDs, and fallback rules for sparse classes. Every new parent/branch uses the same reduced training partition. Existing formal full-training results cannot fill this table's baseline.

All six arms are prespecified, so untouched test outcomes may be reported together after completion. They must not feed automated radius, epoch, window, or arm selection. Old-task development/test accuracy is a separate retention diagnostic, not the current-task selection criterion. Report Task0 accuracy at the common fork, final Task0 accuracy, their signed difference, Task1 learning, numerical failures, and actual compute cost separately. A two-task BWT or averaged accuracy must be explicitly labelled; it is not ten-task FAA/AAA.

The same parent requires identical trainable parameters, inherited old classifier rows, newly initialized classifier rows, frozen backbone, RNG/dropout state, realized Task1 minibatches, epoch schedule, and loss slice. Equal seeds alone are insufficient. Fresh optimizer state is correct here because ordinary SGD has no momentum or weight decay.

## Offline common-state checks

The fixed SGD Task1 reference supplies e=0,1,10,25,49 states and four prespecified current-training batches per state. All virtual probes must restore parameters, buffers, gradients, mode and RNG, use the same realized batch, and perform no optimizer update. Freezing v when scanning r separates direction from distance. Compare finite-difference q(a) with q(a/2), and [g(theta+r*v)-g0]/r with its half-distance analogue.

At the same r, SAM direction u, response direction v, opposite -v, and an independently seeded random direction provide local direction controls. Natural-step and actual-step-matched counterfactual updates at the two preset eta values (0.02/0.002) help distinguish response geometry from amplitude and learning rate. They are local counterfactuals, not trained-method accuracy comparisons. Any matching routine must state whether it targets ideal eta*norm(U) or measured FP32 parameter displacement, and record achieved discrepancy.

A smaller Taylor discrepancy need not imply better generalization. Neither the offline response agreement nor six branch endpoints proves an H-squared mechanism, nonlocal Hessian causality, covariance benefit, inherited three-task plasticity, a beneficial late-training window, or multi-seed robustness.

## Budget and operational conditions

The parent is trained once and shared by six Task1 continuations; the experiments are not six independent two-task retrainings or six ten-task runs. B0 normally costs one gradient read per update; pure G2 normally costs three, with explicit numerical fallback accounting. Offline probes are counted separately. No replay, history, EWC/Fisher, fusion, projection, HVP, clipping or final normalization is enabled.

Only physical GPUs 4/5/6 may be used, admitted after index+UUID locking and an under-lock idle check. Running older radius jobs retain priority through their locks; no unrelated process may be terminated. CPU design approval must be followed by the bounded parent/fork/pairing gate before formal work.

## Deferred rather than silently omitted

The other six first-round arms in the longer proposal (trained SAM, R05/R10/R50, A025/A100), matched-initial-scale radius policies, training windows, three-task inheritance and multi-seed/full-sequence confirmation remain follow-up work. This minimum does not make claims those experiments are needed to establish.
