# Independent design review: Task0 formation and timing, 20 epochs

Accepted as a new, prespecified two-task three-seed policy comparison, conditional on implementation, pairing and runtime acceptance. This review does not certify code or unexecuted GPU tests. Seed0/1/2 yield18 independent two-task trainings. No old50-epoch score or inherited50-epoch parent may fill a row.

| Condition | Task0 epochs1–10 | Task0 epochs11–20 | Task1 epochs1–20 | SGD commits / seed | Nominal gradient reads / seed |
|---|---|---|---|---:|---:|
| V1 | SGD | SGD | SGD | 340 | 340 |
| V2 | SAM | SAM | SGD | 340 | 520 |
| V3 | Residual GAM | Residual GAM | SGD | 340 | 880 |
| V4 | S_F G2 | S_F G2 | SGD | 340 | 700 |
| V5 | S_F G2 | SGD | SGD | 340 | 520 |
| V6 | SGD | S_F G2 | SGD | 340 | 520 |

The existing residual GAM definition is `g0+(g3−g2)`, with positive SAMfirstprobe rho=.05 and base-origin response radius1. It is not the accelerated sum or projection formula. S_F G2 uses the descending SGDprediction `theta−eta*g0`, the direction `N(g0−g1−)`, and a base-origin fixed outerradius.25. Preserve the existing kernels' respective normalization rules and fallback counters. No final update normalization, history, Fisher/EWC, weights fusion, replay or projection is added.

Task0 uses LR.04 and Task1 LR.02, each under an independent20-epoch cosine schedule with minimumratio.1. An epoch11 method change must retain the model, optimizer and scheduler objects, steps and counters. The native Task1 trainingloss remains current-task20-class CE; evaluation of both old and new tasks uses joint40-class predictions. Rank10, batch128, source initialization and within-seed realized augmented minibatches remain fixed.

The existing training-only development split is reused byte for byte: Task0 has1141 training /118 development images; Task1 has906/91. Batch128 gives9 and8 batches per epoch respectively, hence180+160=340 commits per run,6120 overall. Native probe budgets total10440 gradient reads in the no-fallback case, separately from passive evaluations. V5/V6 each use90 G2 updates; they have equal G2 epoch/read counts but not necessarily equal cumulative eta, innerdistance or update displacement.

The Task0-acquisition endpoint is evaluated among20 classes. At Task1, first expand the classifier to40 classes, preserve inherited oldrows, evaluate old/current accuracy before any training, then record all20 epochs. The primary oldaccuracy drop is `old_start40−old_end40`. Report the separate expansiondamage `Task0_old20−old_start40`, finalold/newjoint40 accuracies and taskmacroFAA. Low Task0 acquisition or weak Task1learning cannot be described as a retention benefit merely because accuracydrop is small. Traditional two-task BWT uses Task0old20 and therefore includes expansion; it must not be silently substituted for the post-expansion trainingdelta.

Compare V2/V3/V4 with V1 for formationpolicy outcomes, V5 with V6 for phasepolicy outcomes, and V5/V6 with V4 for partial-duration efficacy. Mean±sampleSD is across seeds0/1/2; differences should be computed withinseed. If AAA is included, identify it as the two-stage macro statistic. Epochwise retentionAUC is a distinct metric, with epoch0 excluded from the mean of completed epochs1…20. Test curves remain passive fixed-endpoint reports and cannot feed arm/epoch selection.

Before formal release, six seed2 bounded smokes must execute20epochs×1batch for each task, thereby testing the actual epoch11 boundary rather than only epoch1. V1 and V6 must be bitwise identical through Task0epoch10; V4 and V5 must be likewise identical. Require identical within-seed Task0 initialstate, newclassifier rowinitialization, realized inputtensors/order and LR. Learned Task0 endpoints are intentionally different, so the Task1 modelparameters and oldrows must not be forced common acrossmethods. R0 replay for allvirtualprobes, exactly one g0/R1 RNGadvance, isolated epochdataRNG, and passive evaluation state/RNG restoration prevent extra probes from changing augmentation or later FC initialization.

Only physical GPUs4/5/6 may be admitted after shared index+UUID locking and an under-lock idle check. Existing processes must not be preempted. Accepted CPU design/formula tests and fresh-command reproduction precede the GPU gates; actual leased-device witness, accepted six-smoke pairing receipts and frozen source/config hashes must precede formal work. Waiting for a free GPU is valid pending state, not proof that a CUDA gate passed.

These experiments test old-knowledge formation and scheduled method exposure. They do not isolate trajectory from endpoint, direction from update magnitude, or timing from LR/gradientnorm evolution. They do not establish a Hessian mechanism or transfer directly to ten-task or50-epoch protocols. The reviewed existing source paths and hashes, detailed metrics and mandatory gates are recorded in DESIGN_REVIEW.json.

## Registered plan and suite check

The newly authored plan, protocol, devsplit and suite were also reviewed. All25jobs, the18-member condition×seed grid,24config hash pins,20epoch budgets, GPU456 restriction and formal6120/10440 plus smoke240/400 update/read budgets match. The new trainer must clarify that inherited scalar g3/beta/response-method fields are inactive schedule placeholders, while actual residualGAM rows compute and useg3. Code and CUDA acceptance remain separate.

## Implementation follow-up resolved

The finalized plan now preserves the exact SAM/GAM norm-plus-epsilon and S_F norm-above-epsilon conventions. ResidualGAM input flags compute and useg3; the active per-epoch method schedule is authoritative. The accepted kernel still records numerical fallback, while validators quarantine any fallback run from the scientific table. Independent implementation review passed41 checks after rerunning13 CPU tests and verifying190source plus57sealed execution-input hashes. The independent JSON pins current plan/protocol/suite/devsplit versions. NativeCUDA and formal-result acceptance remain pending runtime gates.
