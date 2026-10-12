# Reviewer mechanism plan: S3 native protocol

Status: **review-ready, not yet measured**. This plan fixes the main 18-run campaign before looking at any new run. Predictions below are hypotheses, not verified mechanisms.

## Decision target and protocol lock

The campaign asks four bounded questions:

1. Does the current in-task LoRA weight-fusion module improve continual-learning accuracy under its native execution and RNG protocol?
2. Is the implemented factor fusion a faithful realization of a convex update-space mixture, and are its checkpoint weights doing more than choosing epoch 40?
3. At the selected operating point, do Fisher normalization, damping, and history materially affect training, and is the post-fusion Fisher anchor internally consistent?
4. Does GAM help because of its direction, its total scale, its perpendicular correction, or merely its extra stochastic forwards?

All new jobs inherit `base_config.json` (SHA256 `2192fad2d451bf7254a7652550769d14c6ec3efec54083dcd0311481a2acc3e6`) except for the overrides listed below. The locked protocol is ImageNet-A, 200 classes, 10 tasks of 20 classes, ViT-B/16, LoRA rank 10, 50 epochs, batch 128, SGD base optimizer, learning rates .04/.02, native `batch_empirical` Fisher, native training transform and shuffled loader, and native RNG consumption. Every new config also sets:

```json
{
  "s3_boundary_capture": true,
  "s3_fisher_location": "retained"
}
```

The boundary capture is observational: no extra forward and no extra Fisher pass, and Python/NumPy/Torch CPU/CUDA RNG state must be unchanged. The endpoint-anchor arm alone changes `s3_fisher_location` to `endpoint`. The source installer and sidecar implementation are additions around the frozen 190-file source; none of those 190 files may change.

The eight completed native seed-2 sensitivity runs are reused rather than rerun. In particular, the existing center supplies seed 2 for the three-seed fusion contrast. The completed `sample_empirical`, batch-32, RNG-isolated 20260918 campaign is a separate protocol and will be discussed only as external consistency evidence; its observations are never pooled with this campaign.

## Main 18-run campaign

All unlisted parameters stay at the center value. Priority 1 runs isolate a claimed component or a known execution confound. Priority 2 runs locate the responsible design choice at one seed.

| # | Job ID | Seed | Exact override from center | What it tests | Prediction if the mechanism matters | Priority |
|---:|---|---:|---|---|---|---:|
| 1 | `s3_mech_center_s1` | 1 | none | Native center replication and one member of the paired fusion contrast | Defines the seed-1 reference | 1 |
| 2 | `s3_mech_center_s3` | 3 | none | Native center replication and one member of the paired fusion contrast | Defines the seed-3 reference | 1 |
| 3 | `s3_mech_a_nomerge_matched_s1` | 1 | `in_task_weight_merge_start_task=10` | Fusion effect while retaining snapshots and the same quality-CE forward budget | Lower FAA/AAA than center if fusion helps | 1 |
| 4 | `s3_mech_a_nomerge_matched_s2` | 2 | `in_task_weight_merge_start_task=10` | Same as above; paired to the reused seed-2 center | Lower FAA/AAA than center if fusion helps | 1 |
| 5 | `s3_mech_a_nomerge_matched_s3` | 3 | `in_task_weight_merge_start_task=10` | Same as above | Lower FAA/AAA than center if fusion helps | 1 |
| 6 | `s3_mech_a_uniform_s2` | 2 | `in_task_path_weighted=true`, `in_task_path_beta=0` | Whether CE weighting adds value beyond an equal .5/.5 snapshot mixture with the same evaluation budget | Differs from center if loss-based checkpoint choice matters | 2 |
| 7 | `s3_mech_a_latest_s2` | 2 | `in_task_path_start_ratio=.6` | Whether the nearly one-hot two-point softmax is functionally just selecting epoch 40; epochs 20/40 are still captured/evaluated | Matches center if the current weighting rule contributes no graded information | 2 |
| 8 | `s3_mech_fisher_alpha0_s2` | 2 | `ewc_fisher_damping_alpha=0` | Necessity of isotropic damping | Degrades stability if the floor is useful | 2 |
| 9 | `s3_mech_fisher_alpha1_s2` | 2 | `ewc_fisher_damping_alpha=1` | Anisotropic Fisher versus trace-matched isotropic L2 | Degrades relative to center if Fisher geometry matters | 2 |
| 10 | `s3_mech_fisher_gamma0_s2` | 2 | `ewc_gamma=0` | Current-task Fisher only; all older history is removed before adding the new estimate | More forgetting if accumulated history matters | 2 |
| 11 | `s3_mech_fisher_gamma1_s2` | 2 | `ewc_gamma=1` | Undiscounted history versus center decay `.9` | Changes retention/plasticity if decay matters | 2 |
| 12 | `s3_mech_fisher_tasknorm_off_s2` | 2 | `ewc_normalize_fisher=false`; accumulated normalization remains `true` | Equal task-trace weighting versus raw task-trace weighting while keeping the final accumulated trace controlled | Changes results if high-trace tasks deserve more weight | 2 |
| 13 | `s3_mech_fisher_endpoint_anchor_s2` | 2 | `s3_fisher_location="endpoint"` | Anchor consistency: estimate the current Fisher at the pre-fusion endpoint while retaining the fused parameter as `theta_star`; exactly one Fisher pass | Differs from center if the small fusion move changes the locally relevant metric | 1 |
| 14 | `s3_mech_gam_projection_off_s2` | 2 | `gam_grad_gamma=0` | Contribution of the perpendicular GAM correction while retaining the probe-derived plus direction | Drops if the orthogonal correction is useful | 2 |
| 15 | `s3_mech_gam_rho025_s2` | 2 | `gam_grad_rho=.025` | Sensitivity to the first perturbation radius around center `.05` | Reveals under/over-perturbation; no direction is preselected | 2 |
| 16 | `s3_mech_gam_half_direction_s2` | 2 | `gam_grad_beta_1=.5`, `gam_grad_beta_2=1`, `gam_grad_beta_3=.5`, `gam_grad_gamma=.05` | Halves the complete raw GAM direction while leaving EWC unchanged | Intermediate performance if part of the gain is direction magnitude | 1 |
| 17 | `s3_mech_sgd_ewc_s2` | 2 | `optimizer_type="sgd"` | True single-forward SGD+EWC, removing all GAM probe execution while retaining center EWC and fusion settings | Large drop if GAM direction, rather than probe-side RNG alone, drives the gain | 1 |
| 18 | `s3_mech_a_off_s2` | 2 | `in_task_weight_merge=false` | Whole fusion module off, including snapshot/quality evaluations | Difference from matched no-merge measures execution/RNG side effects; this arm alone is not a causal fusion estimate | 1 |

The primary fusion estimate is the within-seed difference `center - matched_no_merge` for seeds 1, 2, and 3. Center seed 2 is the completed `s3_center_s2`; centers 1 and 3 are new. The direct `a_off - matched_no_merge` seed-2 difference diagnoses the native execution/RNG confound. It must not be relabeled as the fusion contribution.

## Offline factor-fusion analysis after training

For every available A-enabled boundary sidecar, compute the diagnostics below from the saved endpoint, epoch-20, epoch-40, and retained LoRA factors. These are CPU matrix calculations and add no training forward:

\[
\Delta_{\mathrm{fact}} = [(1-\omega)B_e+\omega\sum_k\pi_k B_k]
                           [(1-\omega)A_e+\omega\sum_k\pi_k A_k],
\]

\[
\Delta_{\mathrm{mix}}=(1-\omega)B_eA_e+\omega\sum_k\pi_kB_kA_k.
\]

Report per task and in aggregate:

- `cross_term_relative_error = ||Delta_fact-Delta_mix||_F / max(||Delta_mix||_F, eps)`;
- optional, uncollected extension: best rank-10 SVD residual of `Delta_mix`;
- optional, uncollected extension: excess error relative to the best-rank residual;
- effective-update merge shift `||Delta_fact-Delta_endpoint||_F / max(||Delta_endpoint||_F, eps)`;
- factor-space merge shift, selected checkpoint epochs, exact CE weights, entropy effective count, and CE gap.

The current operation blends A and B separately and therefore contains cross terms; it is not mathematically the convex update-space mixture above. A rank-10 SVD projection of `Delta_mix` becomes an optional implementation arm only when the prospective trigger below fires. A fixed-batch logit comparison is an offline follow-up requiring checkpoint restoration; it is not part of the RNG-neutral online observer.

## Correctness gates before reading accuracy

A failed gate invalidates the affected job; it is repaired and rerun under the same ID policy before any performance conclusion.

- Every run has the exact base-config hash lineage, exact source manifest, authorized host `cudahpc40`, one of currently assigned physical GPUs 0/1, 10 completed tasks, a 10x10 accuracy matrix with a finite observed lower triangle, and no nonfinite loss.
- Boundary capture leaves RNG digests identical across the observer and records zero extra forwards and zero extra Fisher estimates. All A-enabled start-task-0 arms apply 10 merges. Matched no-merge arms apply zero merges but retain two snapshots and two quality evaluations per task. Whole-A-off applies zero merges and omits those snapshots/evaluations.
- Center, matched no-merge, and uniform capture epochs 20 and 40. Uniform weights are `.5/.5` within numerical tolerance and have effective count 2. Latest-only selects epoch 40 with weight 1 while retaining both quality evaluations.
- Center and matched no-merge agree in optimizer steps, snapshot metric-forward counts, Fisher-pass counts, and task-update counts. Their only intended numerical difference is whether the retained factors are overwritten by fusion.
- `alpha=0` adds no isotropic floor. `alpha=1` produces a constant diagonal coefficient of `1/d` (constant-diagonal formula verified from code and tensor statistics; no Fisher tensor archive is saved in this phase) and trace 1 within `1e-6`.
- `gamma=0` has zero old-history contribution before adding each new task; `gamma=1` applies no decay. In task-normalization-off, the task contribution keeps its raw trace while the final accumulated Fisher remains trace 1 within `1e-5`.
- The endpoint-anchor arm records `current_fisher_source=pre_weight_merge_endpoint`, retains the fused factors as the posterior center, executes exactly one Fisher pass, and restores model parameters and module train/eval modes after estimation. The default arms record the retained/post-merge source.
- In a same-batch CPU/GPU preflight, the half-direction arm's non-EWC direction equals `.5 * center_raw_GAM_direction` with relative error at most `1e-6` (or absolute error at most `1e-8` for a near-zero tensor), while its EWC vector is unchanged to relative `1e-7`. The SGD arm logs one training forward per update and no GAM perturbation/probe events.

No extra endpoint-versus-retained Fisher is computed during training. A paired Fisher cosine would require another estimator pass and careful loader/RNG reconstruction, so it is explicitly uncollected in the online observer. The causal endpoint-Fisher run is the primary anchor-location evidence.

## Outcomes and prospective interpretation

Primary outcome: final task-macro average accuracy (FAA). Secondary outcomes: task-macro average anytime accuracy (AAA), BWT, forgetting, the full lower-triangular task matrix, the native sample-weighted final/anytime curve, old-task final accuracy, and new-task learning accuracy. Report all metrics; do not choose one after seeing results.

For the three paired fusion seeds, report each paired difference, the mean, sample SD, and the two-sided 95% paired t interval (`df=2`). Use `.5` FAA percentage point and `.3` AAA percentage point as prospective practical-equivalence margins:

- support for a positive fusion contribution requires all three paired FAA differences to have the same positive sign and the 95% interval to exclude zero;
- practical equivalence requires the entire paired interval to lie within `[-.5,.5]` FAA and `[-.3,.3]` AAA;
- otherwise the result is inconclusive. Failure to show benefit is not proof that all possible fusion rules are ineffective.

The following thresholds control optional compute; they are not significance tests:

- If uniform or latest differs from center by at least `.5` FAA or `.3` AAA, replicate **both** construction alternatives at seeds 1 and 3 (+4 runs). Replicating the full prespecified set avoids selecting only the better seed-2 arm.
- If any valid Fisher/history/geometry arm differs from center by at least `.75` FAA or `.4` AAA, or changes mean EWC/clean gradient ratio by at least 2x, replicate both endpoints of that axis at seeds 1 and 3. Do not promote the best arm to the baseline.
- Replicate the endpoint-anchor arm at seeds 1 and 3 if its seed-2 change reaches `.5` FAA, `.3` AAA, or a 10% change in the mean next-task EWC/clean gradient ratio.
- Implement a rank-10 SVD update-mixture arm only if median cross-term relative error exceeds 1% **and** exceeds the median best-rank-10 residual by more than .5 percentage point. Before training it, restore a fixed checkpoint/batch and require maximum absolute logit discrepancy above `1e-4`. If fusion itself has no supported performance effect, retain this as an implementation diagnostic rather than starting a model-selection sweep.
- Add seeds 1 and 3 for the already completed `r=.5` arm only if the paper intends to discuss that apparent seed-2 improvement. It remains a post-hoc robustness check and never becomes the baseline selected on test accuracy.
- Add seeds 1 and 3 for `lambda=0` only if the paper intends to claim EWC is necessary at this operating point. The existing flat lambda response and .144% center EWC/clean ratio currently do not support that claim.

Expected qualitative outcomes, fixed before measurement: matched no-merge will probably be close to center because the observed factor shift is only .0634%-.2528% and the separate protocol found a paired A increment of `−.0487 ± .1737` FAA. Latest-only will probably track center because the current two-snapshot standardized-softmax weights are about `.00247/.99753`; uniform can move farther. The Fisher arms will probably have small accuracy effects because the center EWC/clean gradient ratio averages .144%, though task tails can still matter. The endpoint-Fisher arm will probably be close to center because the fusion displacement is small. Half GAM should land below the full direction but above SGD+EWC, and true SGD+EWC should be much weaker if the separate-protocol direction evidence transfers. These predictions must not be rewritten as findings until the runs finish.

## Accumulated-normalization boundary

Turning `ewc_normalize_accumulated_fisher` off is **not a no-op** and is not algebraically equivalent to one fixed lambda change. With task normalization on and damping off, it multiplies the Fisher by a history-dependent scalar (the existing accumulated mass grows from about 1 to 6.51), so equivalence would require a different lambda at each task. With center damping `.2`, it also changes the relative isotropic share because `alpha/d` preserves trace only when its input trace is 1.

It is therefore outside the main 18 as a scale-schedule intervention rather than a clean geometry ablation. If explicit coverage is required, run one prespecified optional job, `s3_mech_fisher_accumnorm_off_alpha0_s2`, with:

```json
{
  "seed": [2],
  "ewc_normalize_accumulated_fisher": false,
  "ewc_fisher_damping_alpha": 0,
  "s3_boundary_capture": true,
  "s3_fisher_location": "retained"
}
```

Compare it only with the core `fisher_alpha0_s2` arm and describe it as raw history-dependent EWC scale. A geometry claim would additionally require a per-task lambda schedule matched to the observed trace or EWC-gradient norm, which needs new code and a training-only calibration rule. The plan JSON records this one-run option without adding it to the 18-run total.

## Run order and compute

Use all three authorized GPUs, at most one job per GPU. Waves are for fault isolation and early diagnostics; passing or failing a performance threshold does not cancel the remaining main campaign.

| Wave | GPU 4 | GPU 5 | GPU 6 |
|---:|---|---|---|
| 1 | matched no-merge s2 | half GAM direction s2 | SGD+EWC s2 |
| 2 | center s1 | matched no-merge s1 | whole A off s2 |
| 3 | center s3 | matched no-merge s3 | endpoint-Fisher anchor s2 |
| 4 | uniform fusion s2 | latest-only fusion s2 | Fisher alpha0 s2 |
| 5 | Fisher alpha1 s2 | Fisher gamma0 s2 | Fisher gamma1 s2 |
| 6 | task normalization off s2 | GAM projection off s2 | GAM rho .025 s2 |

At 2.5 GPU-hours per complete run, the main campaign costs about 45 GPU-hours, or 22.5 wall-clock hours under ideal two-GPU occupancy. Reserve 25% operational headroom: 56.25 GPU-hours / 28.125 ideal wall-clock hours. The optional accumulated-normalization job adds 2.5 GPU-hours; every two-seed confirmation adds 5 GPU-hours. This is a bounded design estimate, not a user-imposed budget ceiling.

## Coverage and deliberate omissions

Existing native runs already cover `lambda={0,200,2000,20000}`, `lambda_flat={0,.5,1}`, and the second GAM radius `gam_grad_norm_rho={.1,.2,.5}` at seed 2. They answer scale sensitivity but do not justify selecting `.5` as a new baseline. The new runs cover fusion contribution and implementation, execution/RNG matching, Fisher damping/history/task normalization/anchor location, GAM projection/first radius/total direction scale, and true single-forward SGD+EWC.

Do not run the following unless a prospective trigger above fires:

- `in_task_path_normalize_snapshots` toggles: it is a verified no-op in this branch.
- Dense beta, omega, lambda, gamma, alpha, or radius grids: they spend test evaluations on model selection without isolating a new mechanism.
- A second Fisher pass inside training for endpoint-versus-retained cosine: it alters estimator order/RNG and is unnecessary for the causal arm.
- An omega-1 arm solely to amplify a currently unsupported component. Consider it only if fusion has a supported effect but the observed shift is too small to interpret.
- Pooling historical, native, and isolated-RNG protocols. Report their agreement or disagreement in separate strata.

Scientific limits remain: only the fusion contrast has three prospective paired seeds; most design arms are seed-2 mechanism probes. The reused seed-2 center predates the observer wrapper, although the default wrapper is required to pass a numerical/RNG no-op preflight. Repeated access to ImageNet-A test metrics and the legacy seed-2 sweep makes hyperparameter-ranking claims exploratory. A null Fisher ablation at `lambda=2000` means the mechanism is inactive at this operating point, not that Fisher regularization is universally useless.


## Execution scope correction

The online observer saves only factors, snapshots and actual merge weights, with no extra model forward or Fisher pass. Paired endpoint/retained Fisher cosine and rank-projection SVD diagnostics have not been collected and are optional offline extensions; they are not prerequisites for the 18-run core. The implemented offline algebra measures the exact factor-vs-linear-DeltaW cross terms without SVD. Uniform/latest confirmation is triggered by |delta FAA| >= 0.5 or |delta AAA| >= 0.3 percentage points; same-direction signs alone do not trigger added runs. Optional runs are not registered in this campaign.


GPU reassignment: the user explicitly authorized physical GPUs0/1 after GPUs4/5/6 became busy. The active queue uses only0/1, maximum2parallel. The18fullruns require approximately45GPUhours, or22.5hours ideal elapsed after shortsmokes; occupancy/SGD runtimes affect the estimate. No other campaign is stopped.


Observer verification update: attempt1 cross-run equality failed before the first observer boundary (epoch1 snapshot CE already differed). Preserve that reproducibility failure; its source is unresolved, and do not attribute it to the observer. NATIVE_BOUNDARY_WITNESS.json instead tests six actual frozen native boundary paths at identical saved states: all50 factor/classifier parameters, gradients, buffers, modes and native metrics exactlymatch; RNGunchanged,0extra forward/Fisher. Formal eligibility requires this same-state witness and every real boundary invariant. Separate GPU-run matrix/tensor comparisons remain fully reported diagnostics, not a claim of bitwise reproducibility. This does not change training kernels/seeding/backend settings or any performance acceptance test.
