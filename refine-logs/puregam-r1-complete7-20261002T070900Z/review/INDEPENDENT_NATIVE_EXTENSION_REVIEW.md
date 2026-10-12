# Independent review: historical native direction extension

Verdict: **PASS for CPU and static implementation review; GPU smoke release remains a separate required gate.** Reviewed 2026-10-02 by the independent `imageneta_fisher_fusion_logic` agent. No GPU process was started, no deployed campaign or frozen source was changed. The only artifact written by this review is this report.

## Frozen implementation and independent execution

- `legacy_direction.py`: `8776dc581d93e18f967d6fc20d17ce9bce508a06b49393621792832bec5873fc`
- `test_legacy_direction_cpu.py`: `1f92e1b539de7527dffdc9b4380a2d80343bac05a658d021437a3d15a2b86a3b`
- Frozen native `optimer_PerturabtionType/gam.py`: `0b37fca57a31952d911eba4ab99534b7952a6271c1f2c37dbf7a45b72fa5e40e`
- Independently read and SHA-256 checked all 190 entries in `source_manifest.json`: zero mismatches.

Executed independently with `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B test_legacy_direction_cpu.py`. All five selectors, four steps each, passed. Each produced the same parameters, gradients, losses, mechanism statistics, RNG states and subsequent inputs as its reference. The g1/native controls use an entirely unpatched native reference; g0/g2/UG use a minimal explicit direction-selection oracle. Eight negative cases were rejected, including corrupted native caches/projector/SGD, a missing SGD call, changed gamma/radius/lambda, and partial-install failure. Tests use the actual frozen GAM and AST-exact learner batch/projector/clean-gradient methods, but do not claim a full ViT/GPU execution.

## Training path

- `legacy_direction.py:75–104` enforces the pinned native GAM class, ordinary zero-momentum/decay SGD, base beta/gamma `(1,1,0,0)`, rho `.05`, r `1`, nonadaptive/no-SGV, lambda_flat `1`, complete trainable parameter scope and current-task class slice.
- `:112–145` wraps the actual closure and records four fresh gradient observations, then checks all four native caches exactly. The original decomposition executes before selecting `g0`, `g2`, or **`g0 + (g3 - g2)`**. Parentheses are preserved. The control selectors leave the native gradient tensor value untouched.
- `:149–168` calls the original learner projector and checks its actual FP32 expression `clean + 1 * (selected - clean)`. It does not replace this expression with direct assignment. Fisher and delta-penalty caches must be empty.
- `:170–190` observes the base optimizer only after native GAM has performed its own unperturb. The gradient reaching SGD must equal the projected gradient; the actual update must equal a cloned `add_(grad, alpha=-lr)` prediction exactly. The learning rate is read, not changed.
- `:197–199` requires four closures, five model forwards (one native clean-gradient pass plus four probes), one decomposition and one SGD update per training batch. Cached keys alone are insufficient evidence; fresh closure observations are checked.
- Frozen GAM `gam.py:241–289` remains the actual perturb/probe/unperturb/SGD sequence. The wrapper does not reimplement its perturbation geometry or restoration arithmetic.

## RNG, patch scope and restoration

`rng_sha()` reads Python, NumPy, CPU Torch and initialized CUDA RNG states without drawing/reseeding/restoring. Audit operations add tensor copies, reductions, hashes and synchronization, but no model forwards or backward calls. The successful wrapper leaves the original loader, main seeding, scheduler, evaluation and after-task calls in place. The independent CPU test deliberately consumes all three CPU RNG families in every forward and verifies later inputs as well as RNG states.

Only the first factory-created training learner is bound (`:222–247`). Optimizer and learner methods are patched on instances during the training batch (`:191–195`), never on the GAM class. Other optimizer instances retain their native methods. Batch cleanup restores methods and removes the forward hook in `finally`, including an installation failure (`:203–205`, `:236–246`); session cleanup restores the factory and learner method (`:249–258`). An aborted native step is not advertised as a resumed, fully restored training state: the job fails rather than continuing from a partially perturbed model.

`native_control` leaves the decomposition method entirely unwrapped. Its reported decomposition evidence is the pinned native call site plus fresh four-gradient caches and the following projected gradient; the trace explicitly distinguishes this from the directly counted wrapper invocation for the other selectors.

## Mechanism labels require a precise distinction

Do **not** label every field named `gam` as an independent native-g1 diagnostic. The active `mechanism_eval=true` / `mechanism_grad_stats=true` training statistics are accumulated by the original projector: `models_LoRAbasedCL/as1_normfisher_gam_intask.py:1048–1052` reads the current `p.grad`, and `:1077–1150` computes/accumulates its norms and cosines. In these new runs this gradient has already been selected as g0/g2/UG. Those historical field names remain `gam`, but their training-statistic values observe the selected direction. The clean/gam loss cache still follows the native closure-loss convention and is not itself proof of which direction was applied to SGD.

Separately, a newly constructed native GAM optimizer still receives the unchanged g1-base beta/gamma configuration (`models_CL/baseLearner.py:631–635,674–684`). The independent curvature-diagnostic routine `evaluation_weight_sharpness/curv_localization.py:1230–1244` also reads beta/gamma directly from configuration and does not know the training selector. Its resulting g1-base direction must not be relabeled g0/g2/UG. This suite has `flat_eval=false`, so that flatness-evaluation route is disabled. Class methods and other diagnostic optimizer instances are untouched by this extension.

The seven-row performance table uses CNN task-macro AAA/FAA from raw matrices, not these diagnostic direction labels. Actual training-direction claims should cite `direction_audit.jsonl`, including selected, projected and actual-SGD evidence.

## Suite and release checks

Read all eight config files and recomputed each suite config SHA: all matched. Compared their scientific settings with the verified historical shared configuration: all matched after excluding output/prefix/direction fields and the explicitly limited smoke task/epoch fields. Five smoke jobs each run two tasks for one epoch: 18 steps and 2,256 training examples. Three formal jobs each run all ten tasks for 50 epochs: 2,550 steps and 299,050 training examples. Every formal job depends on all five smoke jobs, and the queue additionally invokes the control comparison gate before releasing formal jobs.

The GPU control comparison requires native-control versus hooked-g1 input/target/RNG sequence equality, identical CNN matrices/curves and checkpoint tensor agreement under the declared `1e-7 + 1e-6 * max_abs(reference)` threshold; it records whether equality is actually bitwise. These GPU claims are pending smoke execution and are not inferred from this CPU report.

Issues identified during review and confirmed corrected in the final read:

1. Smoke epoch exceptions now apply to both candidate and historical shared dictionaries, with explicit `(max_train_tasks, init_epoch, epochs) == (2,1,1)` validation.
2. The CPU evidence contains `passed: true`, matching the run-pin gate.
3. Native LoRA checkpoint lists contain modules; `_compare` now requires identical module types and recursively compares their `state_dict` tensors.
4. Queue `fail()` now creates the job directory before writing a failure receipt, including a formal job blocked by smoke validation.
5. Partial installation is inside the batch `try/finally` and has a passing negative regression.

The new three performance jobs do not depend on the failed curvature jobs. This separation is scientifically valid because these runs make no HVP-completion claim and use the historical native FD/GAM path. It must not relabel the old diagnostic campaign or old 42-job queue as successfully completed.

## Interpretation boundaries

- g0 is an original-point gradient direction within the same four-probe process. Native floating-point perturb/unperturb round trips remain, so it is not an exactly equivalent implementation of ordinary one-forward CE-SGD.
- UG uses a finite-rho response `g3-g2`; this experiment does not establish an exact Hessian-vector-product identity.
- With gamma zero, native decomposition arithmetic still executes, while its orthogonal correction contributes zero to the selected update. Do not claim that decomposition arithmetic was removed.
- The table mixes verified historical runs with new runs under the same intended protocol, not a newly randomized multi-seed cohort. Natural direction changes may also change update magnitude; there is no extra norm matching here.
- All performance evidence is seed 2. A difference is an ablation observation for this setting, not statistical significance or a universal necessity result.

Contemporaneous runner/config pins for this review (these are review snapshots, not a substitute for the deployment RUN_PINS): `common.py=67ef03fe8b3b4a33e9ccf48e998dbc9beda0538b618abfdd49465f30ab34c355`; `launch_queue.py=cd7da50e1846a626187adaad03c76fcbc1c9a1bdf663280caafed7606a723ce8`; `run_job.py=e01ee4f7bf5e7288e6316e8d711651c961b55d9040297beecb9795503f6d3cfb`; `validate_smokes.py=7e8f7a4bdf2ee364391a228ee4c191290f0e3553149759b81e7aab6d50d3cd2c`; `suite.json=0274cc3ede0372194c405f63a0f2aa90d872dcaa8770ea25afc591741ba34f6e`; `REUSE_VERIFIED.json=8fd06fa8f0e73e92829bcbe756f41f259aa6461b02051943207bb761450c191f`.
