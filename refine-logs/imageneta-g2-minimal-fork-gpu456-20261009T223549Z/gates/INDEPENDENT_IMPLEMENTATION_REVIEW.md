# Independent implementation review

**Status: passed for the reviewed trainer, data and offline-diagnostic source.** This is CPU/source acceptance, conditional on the native GPU parent, six fork and pairing gates before formal admission. Queue, GPU host state and final training results are outside this review's acceptance scope.

## Reviewed implementation

The new `probe_study_train.py` registers exactly one shared SGD Task0 parent and six Task1 branches. The conditions map to SGD, fixed a/r, fixed-inner/coupled-outer, coupled-inner/fixed-outer, coupled/coupled and retained outer r=1. Each branch performs one native Task1 continuation, not a retrained parent or an extra ten-task sequence. The raw condition/mode/radius/role fields are checked against the registry before routing into the audited response core.

The core computes g1 at a descending point, forms g0-g1, and reads g2 at the base plus the outer response-direction displacement. Only g2 is submitted with coefficient one. Ordinary SGD is still checked against one exact FP32 parameter addition; there is no history cache, EWC/Fisher, fusion, HVP, clipping, final norm rescaling or training projection. The original `response_step.py` remains byte-identical to its frozen original (SHA256 `fe5b2056df14ba10eb8a00722b126986fc83b19cd66905b5bfe7dcfd30eb2e70`). Fixed radius validity is based on the registered distance, direction validity and actual distinguishability, rather than an unrelated eta-dependent reference-distance guard.

Complete parent restoration binds the accepted summary, successful worker receipt, checkpoint directory, file hashes, protocol, seed2, rank10, development split, full model identity and frozen origin. The parent test prefix is independently hash-bound. Complete snapshot serialization explicitly preserves gradients, requires-grad inventory, module modes, relevant caches and RNG, because ordinary parameter serialization alone does not retain gradients. Task1 classifier expansion is delegated to the native lifecycle after exact parent restoration; bounded GPU pairing must prove equality of initial trainable/frozen parameters, old/new head rows, first gradients, realized batches and nominal schedules across branches.

The source training arrays remain unchanged. A reproducible class-local split chooses held-out original-training IDs with seed20261010, floor(0.1*n) bounded to [1,n-1]. Membership is disjoint and exhaustive and checked against image-path/label fingerprints. Native training loaders exclude all dev IDs; dev uses deterministic test transforms on held-out **training** images. The fixed current-task dev CE endpoint at epoch50 is observational and restores state/RNG. It is not best-epoch selection, and test outcomes do not select radii or epochs.

## Offline diagnostic

Only SGD B0 Task1 snapshots are accepted, with smoke/formal budget parity. The registered epochs are 0,1,10,25,49; four fixed disjoint current-training minibatches per state plus one additional disjoint training batch for virtual-step loss are used. Dev and test images are excluded from diagnostic loss/gradient evaluation.

All points use entry-state transactions and replay R0. Outer radius scans retain one measured response direction; its hash is recorded. The h inner point preserves the native FP32 `theta.add_(g0, alpha=-eta)` operation order. Fixed-distance offline points explicitly construct unit directions before the add; this is algebraically equivalent to the training core's raw-vector/scalar-alpha placement but can differ slightly in FP32. Actual displacements are reported, and the diagnostic is not represented as bitwise reuse of formal fixed-radius probes.

The inner half-distance checks only claim registered .05/.025 and .1/.05 pairs. Missing h/2 and .025/2 pairs are explicitly unavailable. Outer half-distance controls are offline points, not additional training arms. Same-radius SAM, opposite response and independent random directions are compared at r=.25. Natural and matched virtual updates cross eta=.02/.002 at the same state. Matching targets measured FP32 parameter displacement, records achieved relative discrepancy and never changes formal training. Scalar parallel/perpendicular statistics are diagnostic only; no projected gradient is submitted.

The smoke is one epoch0 state × one batch × **8 gradient reads**, with zero real updates. The formal audit is five states × four batches × **20 reads = 400 gradient reads**, also with zero real updates. Additional loss-only and parameter-distance reads are separately counted.

## Independent evidence

- An independent Float64 SPD quadratic closed-form oracle covers all five G2 arms. Maximum FP32 final-gradient error is below 8.7e-7, all normal steps perform three reads, model state is restored, no history exists, and real commits equal natural SGD.
- The same oracle checks inner-radius direction invariance on a quadratic and verifies fixed/fixed probe hashes are identical at eta=.02/.002.
- The reviewer reran the eight new fork/split/snapshot tests and seven local diagnostic tests: **15/15 passed**. These include dropout/RNG replay, buffer/gradient/mode rollback, failure rollback, complete snapshot identity, actual displacement matching and exact BA plus full-FC effective-distance checks.
- The implementation author's separate native ViT/rank10 serialization check passed; runtime GPU pairing remains mandatory and is not claimed as completed here.

## Interpretation limits

This is a two-task, single-seed study of registered radius policies. Natural G2/SGD comparisons include update-magnitude and compute-budget differences. Fixed versus h does not isolate learning-rate decay from gradient-norm decay or match initial scale. Offline virtual loss improvements do not constitute trained accuracy gains or establish Hessian causality. Parent Task0 test accuracy uses 20 classes and the final Task0 test accuracy uses 40; conventional two-task BWT also includes common head expansion. Arm contrasts remain paired, but this is not an isolated training-only damage measure.

No acceptance is given here for full-sequence robustness, an optimal radius, learned gates, a late-training window, or three-task inheritance. Those require the deferred experiments described in the plan.
