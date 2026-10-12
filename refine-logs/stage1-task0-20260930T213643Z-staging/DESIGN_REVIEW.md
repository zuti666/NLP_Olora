# Stage 1 Task-0 design review

**Decision: APPROVE WITH EXECUTION GATES.** The proposed single-seed Task-0 stage is sufficient to answer the attachment's first-round questions. It should run the ten predeclared arms, save the requested epoch anchors, and defer all 2-task/3-task choices until the Task-0 result is complete. This review does not authorize extra seeds, another hyperparameter grid, or GPU launch.

## 1. Frozen scientific scope

Use seed 2 only, ImageNet-A classes `[0,20)`, and exactly 50 Task-0 epochs. The native `DataManager` was checked with the frozen configuration: Task 0 has 1,259 train images and 307 test images, so batch size 128 gives 10 training batches per epoch and a final batch of 107. Set `max_train_tasks=1`; changing only `nb_tasks` or `total_sessions` is not an adequate stop control because `DataManager` still constructs all ten task boundaries.

The ten arms are:

1. call-matched CE (`g0` final update);
2. S: `g1` final update through the complete GAM chain;
3. G-r1: `g3(r=1)`;
4. Mix-Sum: `g1+g3(r=1)`;
5. Mix-Mean: `(g1+g3(r=1))/2`;
6. G-r0;
7. G-r0.1;
8. G-r0.25;
9. G-r0.5;
10. native two-pass TrueSAM with `rho=0.05`.

G-r1 is shared by the composition and radius blocks; do not create an eleventh duplicate. Keep rank 10, Q/V LoRA plus FC, `rho=0.05`, `gamma=0`, batch 128, cosine LR 0.04, momentum 0, weight decay 0, no replay/Fisher/EWC/merge/orthogonal correction, and no reuse of a prior trained Task-0 state.

## 2. Exact optimizer paths are a hard gate

### 2.1 Main GAM family

S, G, sum, mean, and every `r` arm must use the same GAM learner and execute the clean-gradient cache plus all four GAM closures. The only intended changes are `beta1`, `beta3`, and `gam_grad_norm_rho`:

| Arm | `beta1` | `beta3` | `r` |
|---|---:|---:|---:|
| S | 1 | 0 | 1 |
| G-r* | 0 | 1 | * |
| Sum | 1 | 1 | 1 |
| Mean | 0.5 | 0.5 | 1 |

Keep `beta2` fixed because it does not enter the final update when `gamma=0`, although `g2` remains part of the probe chain. With the current `ewclora_youyue_fitArchitechture_gam` path, `lambda_flat=1` is required so the projector leaves the selected GAM gradient as the final update.

### 2.2 CE is call-matched in the implemented path

The final implementation correctly retains `lambda_flat=1` for every GAM-family arm. In CE, the wrapped real `gradient_decompose` first executes the normal four-closure GAM chain and then rewrites the decomposed gradient to stored `g0`. The native direction projector consequently computes `clean + 1*(g0-clean)=g0`; it does not change the selected direction. This preserves the clean-cache plus four-closure call count and CPU RNG consumption while applying `g0`. A plain one-pass SGD run is a different execution control and is not needed in this ten-arm stage.

This implementation is acceptable only with the existing runtime audits: stored final gradient equals `g0`, the post-unperturb base-SGD update equals the prediction from that actual gradient, and clean-cache versus `g0` discrepancy is logged. S, G, sum, and mean are likewise checked against their stated formulas rather than inferred from config.

### 2.3 TrueSAM uses an audited explicit branch

The final trainer keeps the common learner/model construction but explicitly replaces `_step_batch_with_log` for `stage1_direction=true_sam`. That branch calls the repository's real SAM `first_step` and `second_step` directly and therefore never reaches the GAM-only `set_direction_projector` path. This is equivalent to the previously required compatible base-learner route for the relevant update and is acceptable because the runtime check verifies two training gradients and the actual base-SGD step.

TrueSAM is an implementation control, not a call-matched causal arm: it uses two gradient passes while S uses the full GAM call chain. A difference between their final accuracies can include execution/RNG effects. The sharper equivalence check is a same-state, same-batch comparison between the TrueSAM second gradient and `g1`.

## 3. Starting-state and data-path equality

Every arm must independently reconstruct the pretrained ViT and initialize LoRA/FC from seed 2. Save one shared epoch-0 anchor after the 20-way head and LoRA parameters exist but before the first optimizer step. Require identical hashes across all ten arms for:

- frozen backbone tensors;
- all trainable LoRA A/B and FC tensors;
- Task-0 class order and train/test sample IDs;
- the first training batch indices and transformed-input hash.

If these differ, halt rather than interpret later accuracy differences. A single deduplicated epoch-0 checkpoint is preferable to ten nominal copies after hash equality is proven.

Use a dedicated DataLoader generator or otherwise prove identical sampler order/input hashes across arms. The LoRA forward creates temporary `Linear` modules and consumes CPU RNG, so different forward counts can otherwise alter later iterator/augmentation randomness. The core GAM family should have identical call paths; CE and TrueSAM still need their batch-order hashes recorded.

## 4. Epoch checkpoints and online diagnostics

Save anchors at completed epochs `{0,10,20,30,40,50}`. Epoch 0 means pre-update; epoch 10 means after ten full epochs and the corresponding scheduler step. Each anchor should contain:

- a full resumable model/optimizer/scheduler/RNG record;
- LoRA A/B plus FC sidecars in a layout loadable by `paired_gam_diagnostic.py`;
- completed epoch, LR, dataset/sample-order digest, source/config hash, and tensor hashes.

The native trainer only saves a task checkpoint after `after_task()`, and its `epoch_dynamics` collection is gated by `_cur_task > 0`; configuration flags alone will not produce Task-0 checkpoints at 0/10/20/30/40. The Task-0 runner therefore needs an explicit, audited epoch hook. Snapshotting must not invoke `after_task`, merge weights, compute Fisher, or consume an extra training batch.

For the first batch of every epoch, record pre-update `g0..g3`, their norms/cosines, `a`, `||r_perp||/||g3||`, the exact probe radii, the final applied-gradient norm, LR, batch indices, and input hash. Prefer reading the actual GAM state from the real step. If extra probes are run, restore parameters, buffers, modes, gradients, optimizer state, and Python/NumPy/CPU/CUDA RNG exactly; require a before/after hash check so diagnostics cannot change training.

At `r=0`, report `relative_L2(g3-g1)`, cosine, and maximum absolute error. G-r0 versus S must be treated as an implementation/RNG audit before it is interpreted as a radius endpoint.

## 5. Offline matched diagnostics are sufficiently bounded

The proposed offline package consists of exactly five anchors: S/G-r1/sum/mean at epoch 50, plus S at epoch 0, using four fixed batches per anchor. It is sufficient for the first result. It covers the common starting geometry, endpoint same-state decomposition, matched effective LoRA-BA-plus-FC update, radius-matched SAM, and optional controlled KL matching without creating more formal training arms. The cross-arm epoch-0 hashes and epoch-1 pre-update online records remain the hard evidence that the single S epoch-0 anchor is shared by all arms.

Do not automatically run every saved epoch through the full offline grid. The online first-batch statistics already give an emergence trajectory. Keep an epoch-30 offline analysis for exactly S, the eventually selected G candidate, and the relevant mix only if endpoint diagnostics are ambiguous. Because all checkpoints are saved, this decision can be made after the main table without retraining.

If the selected radius is `r*<1`, add exactly that selected G-r* epoch-50 checkpoint to the matched offline diagnostic. Otherwise the proposed four anchors do not provide matched-step evidence for the actual `G*`. This is a conditional single-anchor diagnostic, not a new sweep.

## 6. Metrics and interpretation

The Task-0 primary outcome is 20-class held-out top-1 accuracy at epoch 50. Also report train CE/accuracy, test CE/accuracy at the saved epochs, best-versus-final accuracy, gradient/probe metrics, and wall time. Do not report AAA, FAA, BWT, or forgetting for a single task.

Mean versus sum is a clean same-direction intervention, but only under the fixed optimizer/LR schedule. A performance difference shows sensitivity to multiplying this update by two; it does not identify an optimizer-independent universal scale effect.

The `r` sweep changes the outer radius, the sampled gradient, and the final `g2`-dependent tail jointly. Its accuracy curve is a dose response of the chained algorithm. It does not by itself prove a Hessian or curvature mechanism, especially at `r=0.5` or `1`, where a local Taylor approximation is not reliable. The matched-radius offline control is needed before saying direction matters beyond distance.

## 7. Selecting `G*` without overstating the evidence

Freeze the candidate set and deterministic tie rule before reading results. The actually trained eligible candidates are `g3(r)` for the five radii and Mix-Mean at `r=1`; do not imply that `(g1+g3(r*))/2` was tested for every radius.

If the official Task-0 test split is used to select `G*`, Task 0 becomes a development stage. Its winning accuracy is selection-biased and cannot also serve as an unbiased confirmation result. Label it explicitly as development evidence, freeze `G*` once, and use the later independently retrained 2-task/3-task stages as conditional confirmation. Do not choose a method, radius, checkpoint epoch, or reporting metric after inspecting the same test results without recording that choice.

With seed 2 only, the ten arms are deterministic paired trajectories under one seed. Epochs and minibatches are correlated diagnostics, not independent replicates. Report exact differences and trajectories without significance tests, confidence claims, or population-level statements. This stage can select a mechanism candidate and reject implementation hypotheses; it cannot establish seed robustness.

## 8. Boundary for the later formation claim

For the later 2-task design,

`Delta_form = (a21-a11)_GS - (a21-a11)_SS`

is a valid total contrast for choosing G rather than S during Task 0 while applying the S rule during Task 1. Subtracting each run's own `a11` adjusts the accuracy baseline, but it does not make the Task-1 continuation start states identical. GS and SS enter Task 1 with different representations, classifier margins, and gradient fields; ceiling effects and nonlinear accuracy remain.

Therefore describe a positive result as: **the Task-0 formation rule changes subsequent retention under a common continuation protocol**. Do not claim that this formula alone isolates an intrinsic robustness geometry or fully controls starting-state heterogeneity. Same fixed Task-1 batches, sampler/RNG receipts, acquisition `a22`, and same-state local interference diagnostics are supporting evidence for that stronger mechanism interpretation.

## 9. Launch-readiness checks

Before the formal GPU launch, the staged implementation should prove:

1. exactly ten jobs, seed `[2]`, `max_train_tasks=1`, no existing output directories;
2. Task-0 train/test counts `1259/307`, 10 optimizer steps per epoch, 500 total steps;
3. all prohibited Fisher/EWC/merge/replay/correction flags are off;
4. CE/S/G/sum/mean/r formulas are checked against the applied update;
5. TrueSAM's explicit branch bypasses the GAM projector and executes exactly two training-gradient passes;
6. checkpoints `{0,10,20,30,40,50}` exist and epoch-0 hashes agree;
7. online diagnostics are state/RNG neutral;
8. final status requires 50 completed epochs, finite losses/metrics, final Task-0 matrix shape `1x1`, and checkpoint cleanup/restoration checks;
9. GPU allocation occurs only through the idle-GPU queue and does not overwrite prior outputs.

The final staging directory now contains the trainer, all ten configs, queue/finalizer, source pins, native-compatible checkpoint writer, and offline diagnostic code. Its synthetic CPU audit passed all six unique direction implementations (`ce/s/g/sum/mean/true_sam`), common initial-state/data hashes, and 12 float32 low/zero-LR update cases using the frozen GAM/SAM optimizers. The reviewer independently reran the same audit from an isolated `/tmp` copy with CUDA hidden; it completed successfully in 1.9 seconds and reproduced the recorded hashes and update checks. This supports code-path correctness but does not replace the required short real-data GPU smoke before the ten formal runs.

## 10. Coverage verdict

The first round is sufficiently covered by:

- ten independent Task-0 runs;
- epoch-wise first-batch geometry;
- checkpoints at 0/10/20/30/40/50;
- epoch-50 matched diagnostics on S/G-r1/sum/mean;
- one shared S epoch-0 diagnostic;
- one conditional selected-r* endpoint diagnostic when `r*<1`.

No additional fixed-normalization runs, multi-seed confirmation, full epoch-by-arm offline grid, Fisher/Hessian spectrum, or 2-task/3-task factorial should be added before these results are complete.
