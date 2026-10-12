# Result-to-claim judgment: current visual GAM+EWC mechanism evidence

- **claim_supported:** `partial`
- **confidence:** `high` for the seed0, method-specific judgment; no cross-seed or cross-dataset inference
- **review_independence:** `same-family`
- **acceptance_status:** `provisional`
- **integrity_status:** `unavailable` (no scope-matched `EXPERIMENT_AUDIT.json`; this does not block the judgment)
- **evidence pre-check:** all 12 result summaries and 104 extracted numeric fields passed the deterministic existence/hash check; this verifies provenance, not semantic support
- **scope:** ImageNet-A, ViT-B/16, Q/V LoRA rank 10, current visual GAM+EWC implementation, training seed0

## Intended claim and verdict

The anticipated chain was that Task0 shaping produces a less sensitive neighborhood and a larger loss-damage tolerance, which in turn lets future learning move through less damaging directions and improves retention without sacrificing new-task learning.

That chain is **not supported as a causal or split-robust claim**. The evidence supports two narrower observations: the GAM Task0 anchor has lower train256 raw empirical-Fisher trace and lower sampled train-neighborhood positive damage in the chosen effective-weight coordinates; and the historical full GAM-to-GAM sequence reaches Task9 with less old-task damage and a much lower Task0-to-Task9 straight-segment directional sensitivity despite larger endpoint displacement. The paired two-task intervention and the test-neighborhood results prevent upgrading these observations into a general retention mechanism.

The target is the current visual GAM+EWC implementation. The method/source audit shows that this historical implementation differs from the September 17 paper's full FS-EWC formula and retained-anchor/Fisher details. That difference is a reporting boundary rather than a blocker: these results should be labeled current visual or historical GAM+EWC, not silently presented as a numerical evaluation of that paper version.

## Claim ledger

| ID | Claim | Verdict | Evidence and boundary |
|---|---|---|---|
| A1 | A is a controlled comparison of two inherited Task0 states under the same future SGD+EWC Task1 rule. | **yes** | Both branches use the same future optimizer and scientific hyperparameters. The RNG audit matches the new head, loader order, and first-batch image/label hashes for all 51 loader iterations. The inherited model, old head, and EWC Fisher are branch-specific, so the intervention is the **whole inherited state**, not geometry alone. |
| A2 | A shows that a GAM-shaped Task0 state improves later retention/plasticity under fixed future SGD+EWC. | **no** | The GAM-origin branch has lower train256 trace and slightly smaller effective-weight drift, but a21 is 84.04 vs 84.36, a22 is 72.05 vs 74.80, and the two-task mean is 78.045 vs 79.580. Train local CE change favors GAM (-0.00216 vs +0.00367), whereas test local CE improves more for SGD (-0.01925 vs -0.01339). A therefore supplies a counterexample to “lower train sensitivity guarantees better class-IL retention or plasticity.” |
| B1 | At seed0, the GAM anchor is less sensitive on the fixed train256 sample under the sampled random directions. | **yes** | At all six nonzero radii, train G and Q95 are lower for GAM. The train256 raw trace is 0.468–0.520 times the SGD value over the seven measured radii. This is a descriptive result for 16 paired directions per radius in the chosen coordinate system. |
| B2 | The lower neighborhood damage is split-robust and yields generally larger empirical tolerance radii. | **no** | On test, GAM D_mean and G are higher at all six nonzero radii. Test G and Q95 prefix-safe radii are identical between methods for every reported epsilon. Train G radii are also identical; train Q95 is larger only at epsilon 0.001 and 0.05, with the latter right-censored. |
| B3 | B establishes continuous-ball or worst-case robustness. | **no** | K=16 random directions and a seven-point radius grid provide sampled behavior only. The actual-direction bridge is informative but is neither a directed raw-Fisher search nor a certified supremum. Random directions are not training replicates. |
| C1 | In the historical ten-task chains, lower old-task test damage cannot be explained by moving less in effective-weight norm. | **yes** | At every saved Task1–Task9 endpoint, GAM-to-GAM has larger effective-weight displacement and lower Task0 test-local CE change than SGD-to-SGD. At Task9 the drift is 1.40047 vs 1.19758, while test CE damage is 0.03435 vs 0.26013. This establishes the descriptive failure of displacement magnitude as the sole explanation in this seed0 comparison. |
| C2 | Lower straight-segment directional sensitivity is associated with the lower Task9 damage. | **partial** | At Task9, S_path is 0.01479 for GAM and 0.36806 for SGD, D_subset is 0.03810 vs 0.16382, and both quadrature/bound checks pass. Only the Task0-to-Task9 endpoint chord was integrated; it is not the optimizer trajectory, and the two full chains also use different future optimizers. Association is supported; causal mediation is not. |
| M1 | The complete chain “Task0 shaping -> lower sensitivity -> larger tolerance -> better future class-IL performance” is established. | **no** | B reverses on test G and shows no general tolerance-radius gain; A has worse a21/a22 despite lower train trace; C concerns GAM-to-GAM rather than A's GAM-origin-to-SGD continuation. The blocks cannot be concatenated into a single causal chain. |

## What the results support

1. **A genuinely proves a package-level intervention result.** For this seed and continuation, replacing the inherited SGD Task0 package with the inherited GAM Task0 package while holding the future SGD+EWC procedure and audited randomness fixed does not improve the reported class-IL outcome. It also shows that lower train256 trace, lower train/subset CE damage, and smaller movement can coexist with worse a21/a22.
2. **B supports a train-sample, sampled-direction statement.** The current GAM anchor has lower raw per-example-gradient energy and lower positive CE damage on the fixed deterministic train split under the measured directions and radii.
3. **C supports direction dependence in the realized full sequence.** The GAM-to-GAM chain has larger endpoint displacement yet less Task0 test loss damage throughout Task1–Task9, and its Task9 endpoint chord has much lower per-unit-displacement directional empirical-Fisher sensitivity.

The `raw_trace_mean_train256` value repeated on B's train and test CSV rows is the same **train256-only** diagnostic; it is not a test-split Fisher measurement. In C, `I`, `S_path`, `D_subset`, and the numerical bound use the same train256 subset. Full-test loss and accuracy are reported alongside them but are not inputs to that bound.

## What the results do not support

- A pure causal effect of Task0 geometry: A changes the inherited weights, old head, and protection Fisher together.
- A claim that lower train raw Fisher implies lower test sensitivity, larger tolerance, better a21/a22, or better new-task learning.
- A general flatness or Hessian-curvature claim. The measured raw empirical Fisher is a true-label gradient second moment, not the full loss Hessian.
- A worst-case neighborhood claim, because no certified or sensitivity-directed supremum search was run.
- A claim that the optimizer followed the path used for C. The measured path is the straight segment between actual endpoints.
- A method-general, dataset-general, or population-level claim: there is one training seed, and the paired random directions are not independent training runs.
- A numerical claim about the September 17 full-paper FS-EWC implementation. The selected target here is the current visual GAM+EWC implementation.

## Coordinate and path caveats

B and C operate in 24 dense effective Q/V weight matrices with a joint Euclidean/Frobenius norm. Using effective weights removes the trivial reciprocal scaling ambiguity of LoRA factors A and B, but trace, radius, drift, and S_path still depend on the chosen layer coordinates and relative scaling; they are not invariant under arbitrary reparameterization. Random dense perturbations can leave the rank-10 LoRA manifold.

The C path is the ambient effective-weight chord `W(s)=W0+s(Wt-W0)`. Its endpoints are realized checkpoints; its intermediate points need not be states visited by training or even rank-10 factorizations. “Endpoint-segment sensitivity” is accurate; “optimization-trajectory sensitivity” is not.

## Suggested claim revision

> In one ImageNet-A seed of the current visual GAM+EWC implementation, the GAM-trained Task0 anchor exhibits lower raw empirical-Fisher trace and lower sampled positive loss damage on a fixed deterministic Task0 training sample in effective Q/V-weight coordinates. This advantage does not extend to mean positive test-neighborhood damage or to the reported test tolerance radii. Under a paired common SGD+EWC Task1 continuation, inheriting the GAM Task0 state does not improve class-IL accuracy. Separately, in the historical full GAM-to-GAM ten-task chain, old-task loss damage remains lower despite larger endpoint displacement, and the Task0-to-Task9 endpoint segment has lower directional empirical-Fisher sensitivity. These observations show direction-dependent, coordinate-specific sensitivity structure, but do not establish a causal chain from Task0 shaping to later accuracy.

## Exactly scoped minimal follow-ups

No additional A training is requested. Keep the completed two-row A table and its unfavorable a21/a22 result.

1. **B-test raw Fisher, no training:** reuse the existing 16 paired directions and evaluate both anchors on all 307 Task0 test images at `r={0,1,5,10,20}` with the fixed Task0 head and the existing FP32 effective-weight protocol. This is `2 x (1+4x16)=130` unique states and 39,910 per-example gradient evaluations. Report trace mean/Q95/sampled maximum beside test G/Q95; do not use test data to select directions.
2. **Finite GAM-directed proxy, no training:** use the current visual GAM factor/head inner-search once at each source anchor on an independent, fixed training calibration subset. Restore the anchor, form the exact induced `BA_after-BA_before` displacement (including the factor-product cross term), jointly normalize it in effective Q/V-weight space, and cross-apply the two source directions to both anchors at `r={1,5,10,20}`. This is 16 evaluation states on train256 and test307. Record the discarded head displacement and all direction hashes. Call this a GAM-directed aggregate-gradient proxy, not raw-Fisher maximization or a worst-case certificate. A separate direct W-space raw-trace ascent is optional only if the manuscript retains explicit trace-directed language.
3. **C early/mid chords, no training:** add Task1 and Task5 for both historical chains with `s={0,1/8,...,1}`, the same train256, 9-node Simpson integration, the 5-vs-9 convergence check, and the same-sample `D_subset^2 <= I` check. Together with the existing Task9 paths, this yields predeclared early/mid/late endpoint-segment evidence. It requires four new paths; shared-origin implementations need 34 unique state evaluations, otherwise 36 passes.

Do not run a dense tolerance grid unless the manuscript retains a larger-tolerance claim; the present result supports deleting that broad claim. Additional training seeds are required for an across-seed claim, but they are a later formal-confirmation stage and are not substitutes for these diagnostic gaps. The user-designated completed A is not automatically rerun or expanded to four cells.

## Evidence read directly

- [A_two_task.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/A_two_task.csv:1)
- [A_PAIRED_RNG_CHECK.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/A_PAIRED_RNG_CHECK.json:1)
- [B_neighborhood_deterministic.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_neighborhood_deterministic.csv:1)
- [B_tolerance_deterministic.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_tolerance_deterministic.csv:1)
- [B_actual_direction_bridge.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_actual_direction_bridge.csv:1)
- [C_current_FP32_task_trajectory.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/C_current_FP32_task_trajectory.csv:1)
- [C_path_summary.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/C_path_summary.csv:1)
- [RESULTS.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/RESULTS.md:1)
- [evidence_precheck.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/evidence_precheck.json:1)
- [METHOD_ALIGNMENT.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/METHOD_ALIGNMENT.md:1)

The separate six-row/new-trajectory campaign contains smoke/anchor artifacts but no formal result table at this gate, so none of its values are mixed into this judgment.
