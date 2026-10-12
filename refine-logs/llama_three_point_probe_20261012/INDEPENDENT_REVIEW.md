# Independent three-point Llama probe review

Outcome: PASS for documentation, CPU implementation, native integration, and admission/result provenance at the source hashes in DOC_CHECK.json. No unresolved substantive blocker remains. This is not a claim that a CUDA witness, native full-model smoke, formal update, or accuracy result has occurred.

The reviewer started with the compute skill, environment contract, warm local ledger, and the two new documents. The first execution was the documented command verbatim:

```bash
python3 scripts_llama/run_three_point_probe_llama.py --config configs/three_point_probe_llama/inclora_threepoint_central_llama1b_order1_tokv2_rho005.json --seeds 0 --run-root /tmp/three_point_probe_docs_20261012 --dry-run
```

It exited 0, emitted valid JSON, and left `/tmp/three_point_probe_docs_20261012` absent. Full stdout is in DOC_DRY_RUN_STDOUT.txt. A guarded stdlib reexecution prohibited imports of torch/numpy/Transformers/datasets/PEFT and prohibited all subprocesses; it reproduced the exact plan without triggering either guard. All 24 executable wrappers passed bash syntax and write-free literal dry runs. The canonical environment SHA remains 0715917e1ad74c80ae4defaf61a143c33d13857323b67d09ead666a792fa61b2. No packages were installed.

## Scientific implementation

All four modes read BASE, BASE+rho*u, and BASE-rho*u, where u is the same globally averaged, preclip g0 divided by its global L2 norm plus 1e-12. Both sides originate independently at BASE and rho is 0.05. The submitted raw directions are gplus, 2g0-gminus, g0+0.5*(gplus-gminus), and g0. BASE is restored before one final clip and one ordinary Transformers AdamW commit. The central direction averages the two one-sided raw directions before AdamW; it is not divided by rho and does not average two optimizer updates. The local Taylor motivation is mathematically consistent and has the appropriate smooth-loss, finite-radius, clipping, BF16, and AdamW limits.

Native Trainer routing separates this family from the historical optimizer families. Each clean effective window uses native DDP global averaging; both replays use the same cached microbatches and their individual Python/NumPy/torch/dropout states, local no_sync backward, and one explicit global mean. The outer RNG stream returns to its state after the clean pass. Probe-window routing bypasses ordinary pre-step clipping; the scheduler advances once after the one real commit. Only trainable LoRA masters are promoted to FP32 and frozen backbone precision is retained. Pure CE routing, fixed IncLoRA strategy, and explicit guards exclude G2/G3, Fisher/EWC, projection, weight fusion, and history paths.

The reviewer-owned nonlinear oracle uses L(x,y)=x^4/4+x^2*y/2+exp(y), with analytic gradient [x^3+x*y, x^2/2+exp(y)]. All 16 cases passed: four modes, clip enabled/disabled, and externally prepared clean gradient enabled/disabled. Each case checked analytic signed points and selected gradient, exact BASE and supported buffer restoration, zero or one clipping call as configured, exactly one HF AdamW commit/moment step, displacement against an independent ordinary AdamW reference, and three reads/two replays. Files: independent_cpu_oracle.py and INDEPENDENT_CPU_ORACLE.json.

## Fixed scope and costs

The reviewed catalog contains 1B/3B x Orders1/2/3 x four modes. Initial queue preparation admits exactly eight Order1/seed0 chains. Other Orders and seeds42/1993 remain catalog-only. Reviewed fixed settings include rank8/task and accumulated ranks8/16/24/32, alpha32/dropout0.1/QV adapters, DDP4/global64, BF16 forward/FP32 LoRA, LR0.001 then0.0001, constant scheduler, and corrected decoder_token_ids_v2. Task commits218/78/156/62 sum514 per chain. Eight chains imply32 formal stages,16 two-stage smoke stages,4112 formal commits,12336 CE phases,8224 replay phases. These are planned budgets, not observed GPU work.

## Admission, recovery, and provenance

The new queue waits for all physical GPUs0-3 to be idle and for shared UUID/index leases, rechecks idle state after locking, and carries the held lock descriptors into its own children. It neither edits nor stops the preceding campaign. Fresh seeded CUDA/BF16 witnesses precede native model admission. Every chain needs its own two-stage accepted smoke and first-window BASE/clean-gradient/batch fingerprints matched to the same model's fresh g0 smoke. PAUSE releases leases before waiting and repeats admission/witnesses on resume. STOP_AFTER_STAGE prevents subsequent admission and lets an active stage finish. Failed/stopped queues reject automatic restart; accepted receipts are reverified on safe resume, and interrupted existing outputs cannot be overwritten.

Preparation freezes source and model/data/environment inputs, binds the source population and hashes in a manifest, and pins prior accepted adapters before continuation. Stage acceptance validates trace identities, 3/2/1 budgets, backend/hyperparameters, decoder pipeline, FP32 adapter geometry and historical slices, and complete prediction identity/gold coverage. Result collection requires pinned accepted artifacts, complete four-stage chains, and fresh same-model/Order/seed comparisons. Smoke and partial chains retain blank FAA/AAA/BWT and paired deltas. The lower-triangular evaluation matrix and learning/retention measures are retained. This review inspected preparation logic; it did not prepare or certify an actual frozen campaign manifest.

## Findings resolved before final review

R1: Paired reflected-control deltas were initially absent. The final collector emits FAA/AAA/BWT/learning-accuracy differences against clean, ascending, and reflected controls; a full-population synthetic fixture checks central versus all three controls.

R2: Native train/predict timing JSONs, data pipeline audit, and Trainer state were initially unpinned. Acceptance now records four additional hashes; collection verifies them before timing/scoring and checks the decoder schema/version and expected native step. Dedicated tamper checks cover all four files. No scientific algorithm change was needed for either fix.

The final focused receipts match current source hashes:18 core checks,29 distinct new/legacy native Trainer checks including CPU two-rank Gloo, and14 launcher/queue/collector checks, total61. These owner receipts were independently inspected together with their logs and source pins; the reviewer separately executed16 analytic cases and the guarded/catalog stdlib checks. The synthetic collector fixture and tiny native CPU Llama runs are validation artifacts, not real trained-model results.

The reviewer did not run GPU diagnostics, allocate/query CUDA, prepare the actual campaign, launch a queue, signal processes, modify scientific code, alter the earlier frozen campaign, or send external messages. The new control and bulk roots were both absent at the final review. CUDA/full-model smoke and subsequent formal admission remain actual execution gates. Single-seed Order1 remains exploratory; the BF16 parameter-cast diagnostic is expressly observational. Exact final scientific, shared-helper, launcher, collector, test, config, wrapper, document, and validation hashes are in DOC_CHECK.json.
