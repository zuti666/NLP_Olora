# Four-method formal performance validation

User authorization (2026-10-09): only GPU 5, 6, 7. Host GPU7 is absent; use idle 5/6 only, max two parallel experiments. No queue on GPU0–4 is changed or preempted.

## Common settings

ImageNet-A fixed train/test (5981/1519 images), fixed class order0…199, ViT-B/16 augreg2_in21k_ft_in1k, Q/V LoRA rank10 with scale1, seed2, batch128. Ordinary torch.optim.SGD, momentum0, weight decay0. Fifty epochs per task; epoch cosine, floor 0.1×initialLR. Single200 LR .04/.004; CL10 task0 .04/.004, tasks1–9 .02/.002. Native attention, FP32, strict deterministic, TF32/AMP off. No Fisher/EWC, weight merge, replay, projection, QR, HVP, clipping or final direction norm matching.

## Methods

For current minibatch: g0=grad L(theta); h=eta||g0||; g1=grad L(theta−eta g0); a=g0−g1. g1 here is a descending SGD predictor, not the earlier positive SAM probe. N(v)=v/(||v||+eps). Probes replay the same realized batch and initial RNG; only the base forward advances the committed RNG.

| Method | Final submitted direction | History cache | Normal gradient reads |
|---|---|---|---|
| SGD | g0 | none |1|
| Current-only relocated gradient | grad L(theta+h N(a)) | none |3|
| History Response Fusion | g0+c | previous raw base g0 |2|
| History-guided relocated gradient | grad L(theta+h N(c)) | previous full submitted final gradient |3|

With valid history, c=.5a+.5h/ell(p−g0), ell is the norm of the ACTUAL previous FP32 parameter displacement. Without history c=a. History resets at task/head/layout changes and remains across epochs. Zero/unresolvable displacement is explicitly handled by the accepted fallback logic. Relocation is anchored at theta, never at theta−eta g0. One actual SGD applies the final direction; one passive post-update loss forward is logged for each method and restores committed RNG.

## Frozen execution and comparison

Phase1: independently train all200 classes as one task for four methods, 2350 updates each. Primary outcome final epoch50 test accuracy, plus loss/accuracy trajectories, gradient/forward/time/memory cost, history validity, response/direction/actual update norms and cosines. No BWT/forgetting exists for a one-task protocol.

Phase2: after acceptance of all four Phase1 jobs, train three new 10×20 experiments (SGD/current_g2/history_g2), 2550 updates each. Existing source-audited CL10 history_fusion is reused (AAA60.483056746, FAA41.297, BWT−10.756666667). All methods use the same task partition, native loss/head/epoch scheduling and evaluation. The old HRF formula, data and evaluator have independent CPU/native equivalence witnesses, not a claim of a new full GPU trajectory.

Both protocols see 299050 training sample exposures per method; task heads/loss/reset/LR differ. Compare methods WITHIN each protocol, and report CL10 task-equal FAA/AAA/BWT/LA/forgetting. Single seed is exploratory. Cross-minibatch history includes batch noise; history_g2 final-cache vector may be from a relocated point, so no pure Hessian or guaranteed descent claim follows.

Seven new full trainings, zero additional hyperparameter sweeps, zero new smoke-budget training. Four draft configs have been reused with new isolated output directories; no old source/results/queues are modified. Every lease repeats a real CUDA kernel witness, and every job completion verifies frozen source/pins, all actual SGD traces and serialized tensor caches before automatic continuation.
