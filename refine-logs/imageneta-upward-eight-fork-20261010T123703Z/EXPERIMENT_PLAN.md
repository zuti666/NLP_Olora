# ImageNet-A upward probes: shared-parent forks and single-task controls

Created 2026-10-10. Proposal archived in USER_PROPOSAL.txt. Registered manifest: suite.json.

This experiment separates **outer-point selection**, **the update submitted at the original parameter point**, and **ordinary supervised benefits versus retention in continual learning**. The old ten-task results and descending-predictor experiments are references, not pooled observations for this new protocol.

## Fixed probe definition

All gradients are evaluated on the same current TRAIN batch and replay the same complete entry state/RNG R0. With N(v)=v/(||v||+1e-12):

\[
g_0=\nabla L(\theta),\quad g_1=\nabla L(\theta+.05N(g_0)),\quad s=g_1-g_0,
\]
\[
g_2=\nabla L(\theta+rN(s)),\quad g_3=\nabla L(\theta+rN(s)+.05N(g_2)),\quad d=g_3-g_2.
\]

The outer point starts from theta, not from the first SAM point. Every real update is theta <- theta - eta U, with SGD momentum=0 and weight decay=0. There is no projection, final norm matching, gradient clipping, EWC/Fisher, history fusion, model averaging, replay, HVP or second-order autograd. Virtual probes restore parameters, gradients, buffers, caches, modes and RNG. Only the base read's R1 advances in actual training.

| ID | Final U | Native gradient reads per update | Question |
|---|---|---:|---|
| g0 | g0 | 1 | SGD reference |
| g1 | g1 | 2 | SAM-style ascending-loss probe reference |
| g2_minus_s | g2-(g1-g0) | 3 | Remove the first-probe residual from the outer gradient |
| g2 | g2 | 3 | Submit the outer-point gradient |
| g2_plus_s | g2+(g1-g0) | 3 | Add the first-probe residual to the outer gradient |
| g0_plus_d | g0+(g3-g2) | 4 | Add the terminal residual to SGD |
| g1_plus_d | g1+(g3-g2) | 4 | Add the terminal residual to SAM |
| g3 | g3 | 4 | Submit the terminal-point gradient |

The equalities g2-s=g0+(g2-g1) and g2+s=g1+(g2-g0) mean these are eight unique directions, not additional distinct methods. g2 +/- s comparisons also control the number of gradient reads. Natural update amplitude differs across rules, so native training results alone do not isolate direction from amplitude.

## Stage 1: two-task common-parent forks

- Seeds **2,0,1**, independent parent per seed. Reuse the already accepted **seed2 SGD Task0 complete checkpoint**; newly train SGD Task0 for seed0/1. Each seed's eight Task1 rules use exactly the same parent, FC20 -> FC40 initialization, source, data split, current-task CE and input/LR stream.
- Dataset ImageNet-A, fixed native class order 0..199, original 10x20 partition. Train only Task0 classes0..19 and Task1 classes20..39 here. Reuse dev split seed20261010: Task0 1141 train/118 dev (original1259), Task1 906 train/91 dev (original997). No official test images moved into training/dev.
- ViT-B/16 augreg2_in21k_ft_in1k; global trainable Q/V LoRA **rank10** plus all seen FC rows; FP32/native attention, deterministic algorithms, TF32/AMP off.
- **rho=.05, r=1**, batch128,50epochs per trained task, drop_last=false. Task0 LR .04 -> .004; Task1 .02 -> .002 via cosine minimum ratio .1. 450 updates per parent;400 per Task1 fork.
- All eight rules x three seeds = **24 Task1 forks**. Additional seed2 upward g2 at **r=.25** = **one bridge**. Bridge reference: existing descending R25/S_F/R100 receipts, same parent/head/input/CE protocol verified before comparing. S_F has a dynamic descending probe and is identified separately.
- Re-run seed2 SGD Task1 even though its endpoint exists: the old run lacks 50 epoch test curves. The new run must reproduce all400 realized input identities/LRs and the accepted old final parameter hash. It is one reference rerun, not a new independent replicate.

### Measurements and interpretation

Epoch0 after FC expansion; each completed epoch records current train CE, current TRAIN-dev CE/accuracy, old/current official-test CE and accuracy over **all40 seen classes**, and old20-only accuracy as an auxiliary check. Old tests are passive reporting only. Record the old Task0 accuracy **before expansion** and old accuracy immediately **after expansion**, so expansion damage and subsequent training damage stay separate.

Endpoint FAA=(old40+current40)/2. BWT=old40_final-old20_parent. AAA=(parent_stage_accuracy+FAA)/2. **AAA is not an epoch average.** RetentionAUC=mean old40 accuracy across epochs1..50; it is separately labelled. Report paired seed differences, means +/- sample SD, final dev CE, native update costs and peak training memory. No best-test or best-epoch selection.

Save complete model/parameter-gradients/buffers/cache/modes/RNG/optimizer/scheduler snapshots at epochs **0,1,2,5,10,20,30,40,50**. Compare old accuracy against current dev CE only on overlapping dev-CE ranges, without extrapolation; retained accuracy at a selected CE is exploratory interpolation, not a new trained method. Report learning-versus-retention tradeoffs, not just final FAA.

Predeclared primary contrasts: g2 versus g2-s; g2+s versus g2; g2 versus g3; g0+d versus g0; g1+d versus g1. Report both dev learning and retention. No equivalence/noninferiority conclusion without a predeclared margin and appropriate uncertainty. No claim that these outcomes prove a Hessian mechanism.

## Stage 1b: offline same-state diagnostic (three jobs)

For each seed use newly completed SGD, SAM and g2 trajectory checkpoints at epochs **1,10,50**. For every one of these nine states, use two fixed current TRAIN probe batches with disjoint current TRAIN and old TRAIN holdout batches. All are disjoint from dev/test. The old TRAIN holdout is evaluated with no_grad, never used in real updates, gate selection or hyperparameter selection.

Construct all eight candidates from the same four reads. Evaluate their original step and a **nominal parameter-length match** to eta||g0||. Record achieved FP32 parameter displacement and error; LoRA effective BA+FC displacement is separately measured and is not matched. Evaluate current/old holdout CE changes, actual post-update current-batch gradients, cosine alignment and R_grad. Restore complete state after every candidate; zero committed training updates. Per seed:9 snapshots,18 probe batches,288 virtual candidates.

Native cost microbenchmarks actually skip unused probes and use the optimizer commit:3 warmup,5 timed,2 independent consistency trials per rule/snapshot. Exclude evaluation/serialization from timing. Microbenchmark virtual updates never survive. Report true1/2/3/4-read cost separately from passive diagnostics and old padded-four-probe experiments. GPU hardware/UUID must accompany measured timings; do not infer a cross-hardware speed comparison.

## Stage 2: full200 single-task generalization control

Six rules **g0,g1,g2-s,g2,g1+d,g3**, seeds2/0/1 = **18 jobs**. Independent initialization per seed, same initialization across methods. Use all original **5981 TRAIN /1519 TEST** images with a single FC200 head, 50epochs (not500),47 batches/epoch,2350 updates. Other model/probe/SGD settings remain fixed; one-task LR .04 -> .004. This panel deliberately uses the full original training split; the two-task panel uses its locked train/dev split, so they are not identical data regimes.

Record epoch-wise training CE, test CE, sample accuracy and equal-weight accuracy across the original ten20-class groups. Fixed endpoint50; no best-test selection. BWT/forgetting are **undefined/null**. Report this panel separately from two-task FAA/AAA. If a method helps both, ordinary supervised benefits remain a viable explanation; a larger retention benefit at comparable dev learning provides more specific evidence about continual learning.

## Deferred Stage 3: Task2 confirmation

Not registered as automatic training yet. After Stage1/2 evidence is collected, select a small preregistered confirmation set: SGD, SAM, g2, g2-s, plus at most one residual rule supported in all three paired seeds. For each seed continue a common accepted SGD Task1 parent with the same FC40 -> FC60 initialization, train only Task2 for50epochs at .02 -> .002, with the same locked train/dev membership and pure settings. Preserve both prior-task metrics, expansion baselines and full curves. Alternatively, method-specific Task1 -> Task2 continuations answer persistence along different trajectories and must be a separate panel. Do not silently mix these designs. Any choice based on Stage1 outcomes is explicitly exploratory and needs fresh-seed confirmation before a strong selection claim.

## Registration, storage and execution

New formal jobs:2 parents+25 forks+18 single200=**45 training jobs**;3 formal offline jobs. 53,200 real SGD updates and150,750 TRAIN gradient reads, plus one passive post-update CE per update. Gate jobs:16 bounded training smokes (1epoch/2batches),1 pairing gate,1 offline smoke. Total manifest **66 jobs**. All45 formal training runs depend on every smoke, common-state pairing and offline restoration gate passing.

Physical GPU pool **2,3,4,5,6**, at most5 concurrent jobs. GPU0/1 excluded. Admit only GPUs with no compute processes, <500MiB memory use and <5% utilization, with shared physical-index and UUID locks and a second idle check. No preemption. Failure stops subsequent admissions and drains already-running jobs; no silent batch-size/protocol retries. runtime/PAUSE halts admissions while preserving active jobs.

Code/config/docs/receipts are under this campaign directory. Bulk outputs use the exclusive writable directory `/data/115-2/users/liying/cache/tmp/imageneta-upward-eight-fork-20261010T123703Z/outputs`; local outputs is a symlink. >150GB complete snapshots are preserved; no pruning/compression of state. At registration the storage volume had ~474GB free versus ~45GB on the project volume. No old results/queues/source files changed. Every worker proves an actual CUDA forward/backward under the reused pinned environment; source190files and execution pins are checked before admission and acceptance.

Progress: runtime/queue_state.json. Accepted results: FORMAL_RESULTS.csv, AGGREGATED_RESULTS.json and RESULTS.md. Results are written only after successful worker exits and independent numeric/state/budget acceptance; pending/failed/smoke numbers never masquerade as formal results.
