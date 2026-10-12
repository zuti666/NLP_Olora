# Task0 formation and timing under a common Task1 SGD policy

Created 2026-10-10 UTC. Independent two-task, 20-epoch protocol; no historical 50-epoch score is pooled. User-authorized scope: six conditions × the same seeds **0, 1, 2**, 18 full trainings. Native GPU acceptance runs are excluded from the result table.

## Questions and interpretation

Primary question: does the Task0 training policy affect old-task accuracy after a common 20-epoch Task1 SGD policy? Compare conditions 2–4 with condition 1, jointly considering Task0 learning, the change caused by head expansion, new-task learning and final task-mean FAA.

Supporting question: can ten epochs of S_F G2 preserve the effect of twenty, and does their position matter? Compare 5 vs 6 and both vs 4. Early and late phases have equal G2 update/read counts but different learning rates and gradients under cosine. This is a comparison of stage policies, not a proof of an intrinsic early/late mechanism. The experiment does not isolate trajectory from endpoint, or prove a Hessian mechanism.

## Six fixed conditions

| ID | Task0 epochs 1–10 | Task0 epochs 11–20 | Task1 epochs 1–20 | Seeds |
|---|---|---|---|---|
| 1: `sgd` | SGD | SGD | SGD | 0, 1, 2 |
| 2: `sam` | SAM | SAM | SGD | 0, 1, 2 |
| 3: `gam_residual` | Residual GAM | Residual GAM | SGD | 0, 1, 2 |
| 4: `sf_g2` | S_F G2 | S_F G2 | SGD | 0, 1, 2 |
| 5: `sf_g2_then_sgd` | S_F G2 | SGD | SGD | 0, 1, 2 |
| 6: `sgd_then_sf_g2` | SGD | S_F G2 | SGD | 0, 1, 2 |

One optimizer and one cosine scheduler are retained throughout each task. At epoch 11 only the final direction policy changes; the model, learning-rate phase, optimizer and random streams are not restarted. A new task receives its ordinary native task optimizer and its own 20-epoch schedule.

## Algorithm definitions

All directions cover the concatenated trainable Q/V LoRA factors and the full classifier. Final parameter updates are ordinary SGD with momentum 0 and weight decay 0. There is no clipping, final norm matching or extra projection.

Let `g(x)=∇L(x)` on the identical augmented minibatch. `N(v)` denotes each method's existing global L2 normalization: SAM/residual GAM retain their norm-plus-epsilon denominator; S_F divides by the norm above its existing epsilon threshold. The kernels are reused exactly, without silently unifying those numerical conventions.

**SGD:** `U=g0=g(theta)`; one gradient read.

**SAM:** `g1=g(theta+0.05 N(g0))`, `U=g1`; two reads.

**Residual GAM:** keep the previous SGD+GAM residual definition, with positive first probe and fixed radii rho=.05, r=1:

```
g0 = g(theta)
g1 = g(theta + .05 N(g0))
g2 = g(theta + 1.0 N(g1-g0))
g3 = g(theta + 1.0 N(g1-g0) + .05 N(g2))
U  = g0 + (g3-g2)
```

This is not the accelerated summed GAM direction or the SAM+GAM residual. Four gradient reads, final SGD commit once.

**S_F G2:** reuse the accepted descending response kernel, with `a_t=eta_t ||g0||` and fixed outer radius .25:

```
g0      = g(theta)
g1minus = g(theta - eta_t*g0)
v       = N(g0-g1minus)
g2      = g(theta + .25*v)
U       = g2
```

Three gradient reads. The outer probe is measured from the original base theta. The inner probe remains coupled to the current cosine learning rate. No history is read. The old accepted fallback policy for degenerate responses is preserved and its actual count reported. Any numerical fallback quarantines the run from the accepted formal table and requires inspection; it is not silently pooled as the requested method. No new component or radius point is introduced.

## Frozen common configuration

- ImageNet-A, original 200-class order `0..199`, original 10×20 partition; train only Task0 classes 0–19 and Task1 classes 20–39. This does not repartition all 200 classes into two tasks.
- Preserve the recent train/dev split (split seed 20261010), recorded in `dev_split.json`: Task0 train1141/dev118, Task1 train906/dev91; official test307/254, never used for training. The split comes from the original1259/997 Task0/1 training samples. No change or resampling per condition or seed.
- ViT-B/16 `augreg2_in21k_ft_in1k`; frozen backbone, Q/V LoRA rank10, full classifier trainable, native sequential LoRA carry-over. No alpha/rank scaling is added. The dedicated `formation_task0_schedule` and `formation_task1_method` are authoritative; legacy model-factory optimizer/radius tags do not select the actual update kernel, which is recorded for every step.
- 20 epochs/task; batch128, drop_last=false, 9/8 training batches per Task0/1 epoch. LR .04/.02; epoch cosine with T_max=20 and eta_min=.1×initial LR. Final training epoch uses the pre-step cosine LR, and the scheduler reaches eta_min after its 20th step.
- Same initialization, loader order and augmentation seed rule within each seed. Virtual probes restore parameters, gradients, buffers, modes, native caches and replay the same R0 random state; only the base pass advances training RNG. Passive evaluation restores RNG. Record the new classifier rows after expansion to check paired initialization.
- FP32, native attention, deterministic execution; no AMP/TF32. Warm reuse of the pinned Pilot_new_local environment; no installation. `environment_spec.json` is a byte-identical inherited environment reference; this document, `protocol.json` and `suite.json` define the active 20-epoch workload and GPU456 restriction.
- EWC=0, Fisher calculation skipped, no in-task/endpoint weight fusion, replay, orthogonal correction, historical-gradient cache, HVP or auxiliary optimization.

## Metrics and reporting

Task0 endpoint: official old-task accuracy with its 20-class head and learning loss/development metrics. Task1 epoch0 is evaluated **after FC20→40 expansion and before any Task1 SGD update**. At epoch0 and every Task1 epoch1..20, old and new task accuracies and CE use the full seen 40-class classifier. Old20-only accuracy is auxiliary, not substituted for joint40 accuracy.

- `old_drop = old_start40 - old_final40` (positive means a decline during Task1 training).
- `expansion_damage = task0_end20 - old_start40`, reported separately.
- Final old/current joint40 accuracies; `FAA=(old_final40+current_final40)/2`.
- Optional traditional `BWT=old_final40-task0_end20` clearly includes head expansion; training-only BWT is `-old_drop`.
- Optional two-stage `AAA=(task0_end20+FAA)/2`; it is not an epoch-average statistic.
- Retention AUC is the arithmetic mean of old joint40 accuracies at epochs1..20, excluding epoch0.
- Phase summaries report LR, inner probe length, update norm and actual committed displacement. More favorable retention with weaker Task0 or Task1 learning is not treated as a successful result in isolation.

Report per-seed values, mean ± sample standard deviation, and paired seed differences for 2/3/4−1, 5−6, 5−4 and 6−4. All three requested seeds must pass acceptance before a row is labelled three-seed complete. Incomplete cohorts are labelled explicitly; no optimistic replacement of failed seeds.

## Queue and acceptance

25 registered jobs: six native GPU short runs (seed2; both tasks20epochs, max1batch/epoch), one pairing gate, then18 formal trainings. Short runs have40 real SGD commits each and include epoch11. They are validation, not scientific scores. Formal training has340 commits/job (180 Task0 +160 Task1), total6120 commits and10440 gradient reads. The short-run total is240 commits/400 reads.

CPU checks cover method definitions, state restoration, epoch10/11 switching and continuous cosine. Before execution, an independent reader follows `RUN.md` literally. Every leased worker runs a seeded CUDA forward/backward witness. The GPU pairing gate verifies equal starts and inputs, exact first-ten-epoch trajectories for 1 vs6 and4 vs5, paired newly initialized FC rows, and Task1 SGD-only execution. Formal jobs depend only on successful accepted gate receipts.

GPU admission is **physical4,5,6 only**, at most3 concurrent workers. A device must have no compute PID, <=500MiB used and low utilization under both shared index and UUID locks. Existing jobs are not stopped. If all three are occupied, the durable controller waits and launches automatically after idle admission. Logs/receipts are in `runtime/jobs/`, progress in `runtime/queue_state.json`, accepted result tables in `FORMAL_RESULTS.csv`, `AGGREGATED_RESULTS.json` and `RESULTS.md`, with per-epoch and paired-difference CSVs alongside them.

## Reproducibility

Source190 is pinned by `source_manifest.json`; new trainer, queue, configs, protocol and RUN instructions are sealed in `RUN_PINS.json`. Formal run commands are the registered argument arrays in `suite.json`. Native task checkpoints and the new epoch metrics are saved under exclusive `/data/115-2/users/liying/cache/tmp/<campaign>/outputs/`; the campaign-level outputs link points there. No previously frozen campaign is edited or reused as a 20-epoch result.
