# Existing evidence for S3 mechanism continuation

This report was extracted without GPU work or changes to the prior campaigns. The adjacent JSON preserves source hashes, all 80 original mechanism dictionaries, per-run summaries, and contextual result tables; the CSV contains 8 runs × 10 tasks with selected raw and derived fields. All eight metric SHA256 values match the prior result index, and all runs have ten tasks and ten A/B/FC checkpoint groups.

## Frozen empirical controls are completed and reusable within that protocol

The 20260918 empirical campaign has 15/15 new and 12/12 reused Fisher runs marked completed and verified. Its five-arm comparison is:

| Arm, seeds1/2/3 | FAA mean±sample SD | AAA mean±sample SD |
|---|---:|---:|
| Native EWC | 39.524±1.398 | 60.764±.813 |
| Probe-matched EWC | 39.605±1.103 | 60.822±.882 |
| Half-scale GAM | 50.444±.401 | 66.424±.367 |
| GAM endpoint | 51.454±.767 | 67.393±.402 |
| GAM+A | 51.405±.941 | 67.383±.435 |

Paired FAA: raw GAM−probe +11.8487±.7681; half−probe +10.8387±.7805; probe−native +.0810±.3185; A added to GAM −.0487±.1737. A increments for seeds1/2/3 are +.044/−.249/+.059. AAA increment from A is −.0097±.0375. These results support a GAM direction effect beyond extra probes and a simple doubled direction scale. They do not support a positive incremental effect of the tested A construction.

Sources:
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_execution_20260918/RESULTS.md`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_execution_20260918/direction_table.csv`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_execution_20260918/paired_deltas.csv`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_execution_20260918/verification.json`

Scope: sample_empirical,500 examples,batch32,isolated RNG,endpoint-centered interventions; different from current S3 batch_empirical/native-RNG/A-on sensitivity. Do not pool them.

## What the existing S3 mechanisms already establish

The raw source for each task is `mechanism.steps` inside its `*_cl_metrics.json`; exact paths and hashes are in the CSV/JSON. The eight-run inventory is `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920/results.csv`.

For the S3 center, all ten task boundaries apply weight fusion. Factor-space merge shift L2 is .006006–.023065, equal to .0634%–.2528% of endpoint factor norm. This confirms execution but says nothing by itself about performance benefit. The merge is applied before the current Fisher estimate.

For two snapshots {20,40}, logged CE weights are approximately .0024726/.9975274 and entropy-effective count 1.01746 at every task. This is explained by population-standardized two-point CE: distinct values become approximately ±1, so beta3 gives softmax(±3) almost regardless of the CE gap. It is nearly selecting the lower-loss snapshot and then blending it with the endpoint, rather than a broad average. Effective count is exp(entropy), not inverse concentration. Every one of the80 logged snapshot pairs has lower CE at epoch40. The CSV includes rounded log CE for epochs20/40 and the epoch-indexed weights inferred from this ordering; original metrics only store min/max weights.

The center's task1–9 EWC/clean gradient ratio is .000701–.002520 (mean .001441, or .144%). At lambda20000 the ratio is .005353–.015778 (mean .010010, or1.001%); at lambda0 it is exactly0. EWC gradients execute but are very small. Center EWC/GAM loss ratio is .00004895–.00036238 (mean .00018665); this is a loss ratio, not a gradient ratio. Means here average logged task means equally and exclude task0 where no historical constraint exists.

Center GAM/clean cosine is .9267–.9661; correction/clean gradient norm ratio1.612–2.951. Across ten task means GAM norm averages4.497 and clean norm1.661. The gradient projector combines clean+s*(GAM−clean)+EWC.

Fisher is the batch-gradient dense-DeltaW proxy, not a per-sample estimator. Task raw traces are10.387–28.287; task and accumulated normalization give trace≈1. The accumulated mass1→6.51322 is accounting metadata, not the final Fisher trace. Damping uses alpha/d with alpha=.2 and d=14,155,776, yielding floor1.41285e−8; mean Fisher entry7.06425e−8. Repeated damping is applied after each normalized accumulation.

`in_task_path_normalize_snapshots` is a no-op for this branch: the implementation disables Fisher trajectory snapshots and its LoRA-factor path mean does not normalize snapshots. It is not an additional active mechanism requiring a standalone training ablation.

Relevant frozen source:
- `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920/source/models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py`:322–375 standardized CE weights;435–483 damping.
- `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920/source/models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py`:325–382 path weights and factor averaging;509–580 postmerge Fisher metadata.
- `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py`:1020–1110 gradient construction and measurements.

## Checkpoint diagnostics and limits

All eight S3 formal runs provide ten post-task groups containing A/B and classifier weights. These can support loading equivalence checks, task0 retention trajectories, and fixed-direction perturbation evaluations without retraining. Task-local epoch20/40 snapshots and premerge endpoint are not stored, so pure fusion operators or pre/post-fusion prediction counterfactuals cannot be reconstructed from those groups alone.

An existing current-protocol B4 pilot already restored six empirical base/GAM+A models:66 accuracy cells match the archived logs;120 exact post-direction logit restorations;720 conditions using20 shared Gaussian directions and radii0,.25,.5,1,2,3. The report is `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-imageneta-plan-20260918/execution/evaluation/B4_REPORT.md`; raw tables, restoration receipts, direction bank, and scripts are in that directory.

The pilot does not show that GAM+A is flatter along these dense-DeltaW random directions: at radius3 its baseline-subtracted CE damage exceeds base by .001170±.000785 across seeds. Effects are small; the pilot has20 rather than prespecified100 directions and no GAM-only arm. It compares method-specific task0 anchors and is associative rather than a same-anchor causal intervention. Do not turn this into a blanket flatter-geometry claim.

## Historical strata, contextual only for S3 continuation

Historical seeds0/1/2 GAM×A four-cell FAA: EWC40.328±.366;GAM51.667±1.124;EWC+A40.292±1.022;GAM+A52.071±1.379. A-on-GAM paired increment +.404±1.925. Its source is `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_ablation_reuse_20260918/factorial_summary.csv` and `factorial_paired_differences.csv`.

Six seed1 uniform windows use omega1,beta0 and span FAA52.330–52.518. M0 is endpoint[50]; other windows range from[10,20,30,40,50] through[48,49,50]. They do not directly compare the weighted{20,40},omega.3 center. Source: `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/imageneta_ablation_reuse_20260918/windows.csv`.

Historical seed2 Norm→Norm+Damp gives FAA51.016→52.150, but lacks the frozen matching needed to replace a new controlled ablation. The older omega scan jointly changed Fisher and weights; do not label it a pure weight-fusion coefficient experiment. Detailed provenance: `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/IMAGENETA_ABLATION_REUSE_AUDIT_20260918.md` and `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/LOGS_TRY_HYPERPARAM_TASK_RADIUS_AUDIT_20260918.md`.
