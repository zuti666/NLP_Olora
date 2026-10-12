Independent native r1 A/B readiness review

Verdict: PASS for implementation, independent CPU execution, configuration compatibility, result-table preparation, and static launch guards. Formal A/B training remains gated on three fresh GPU smoke completions and their validation. This reviewer did not launch GPU work or edit the implementation.

The reviewed selectors are A = g0 + (g2-g1) and B = g1 + [(g2-g1)-a*g1], with one global a = sum <g1,g2-g1> / sum ||g1||² over every trainable LoRA A/B factor and the entire seen-class FC. The difference and selected output retain the native training dtype; products and reductions use FP64. Exactly zero g1 uses a=0; a nonzero denominator receives no epsilon. There is no direction normalization or effective-step matching. Projection is in factor-gradient coordinates; it is not a BA tangent projection.

Static source review confirms that the original GAM performs all four probes and native decomposition; selection runs immediately after decomposition, followed by the original FP32 clean+1*(selected-clean) projector, native unperturb, and one real zero-momentum/zero-decay SGD call. Cache tensors are rechecked after selection. Temporary methods bind the training learner and optimizer instances, are restored on success/failure, and leave the GAM class and other learner/optimizer instances unchanged. residual_selection.py imports only math and torch, and contains no training, loader, scheduler, controller, or RNG operations. The runner calls the pinned src.main directly.

Effective-step observation captures real factor/FC values inside the base-SGD wrapper, after native unperturb. The reference is a cloned FP32 projected-g1 SGD update from that exact same restored point and nominal learning rate. The observer verifies that parameters, gradients, buffers, and RNG are unchanged before real SGD. It compares actual post-SGD values with the snapshot using FP64 low-rank Gram algebra for unscaled BA plus all FC parameters. No second optimizer step or extra forward/backward is executed.

Independent CPU rerun command (CUDA hidden):

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python test_legacy_direction_cpu.py --output /tmp/r1_ab_independent_integration_cpu.json

Executed from this campaign directory; exit code 0, PASS. Evidence covers 7 selectors × 20/40-class heads × 4 native steps: exact parameters, gradients, losses, mechanism statistics, Python/NumPy/Torch RNG, and subsequent input tensors versus the independent minimal selector oracle; unpatched native controls remain exact. Four-probe same-start caches match across selectors. Per batch, 4 gradient closures, 5 CE forwards, and 1 actual SGD are checked. The suite also covers 18 numeric checks and 16 negative cases, including global versus per-tensor projection, parallel/perpendicular components, zero/tiny nonzero denominators, nonfinite inputs, FP32 projector cancellation, cache/RNG mutation, hook cleanup, altered LR, and corrupted actual SGD. Six additional real native A/B updates have both actual and reference effective norms checked against an independent dense FP64 BA+FC oracle.

Separate reviewer-designed fixtures passed: a deliberately mixed two-block global projection has a=-0.2 and exact A/B outputs, with unchanged input caches and RNG; zero g1, zero delta, and tiny nonzero-denominator branches behave as specified. A two-LoRA-pair plus FC weight/bias fixture compares actual FP32 update tensors against explicit dense FP64 BA differences to 1e-12 relative tolerance (joint norm 0.2486518081133181).

The reviewer separately generated 8 actual native CPU A/B audit rows and exercised validate_new_step_record. All 8 valid rows were accepted. For each selector, matching flags, changed step scales, a mutated counterfactual-state flag, wrong geometry, wrong effective ratio, wrong projection coefficient, per-module projection scope, and failed orthogonality were rejected: 16 negative guard exercises.

All five configs have independently verified suite-bound hashes and match the completed formal g0 r1 config exactly apart from outputs_root/prefix; smoke configs additionally change only init_epoch/epochs to 1 and max_train_tasks to 2. Full runs preserve seed2, rank10, r=1, rho=.05, 10×20 classes, 50 epochs/task, batch128, nominal .04/.02 cosine SGD, no replay/EWC/Fisher/fusion, and mechanism_eval=mechanism_grad_stats=true. Formal budgets are each 2550 updates and 299050 sample exposures; each smoke is 18 updates and 2256 exposures.

The static queue has exactly three smoke jobs and two new formal jobs. Each formal job depends on all three smoke jobs and validate_smokes; the release check requires per-step own-spec gradient/projector/SGD identities, no state/scaling intervention, complete budgets/checkpoints/Fisher skips, same initialization, and paired input/target/RNG sequences. It does not assert cross-GPU trajectory equality. The queue uses a unique controller lock, physical GPUs 4/5/6, live idle checks, a shared lease inherited by the child, UUID recheck before spawning, host/PID-namespace and PID/start-time checks, and stops pending launches on failures. Run-pin validation explicitly binds the CPU report to current helper/test and native-source hashes. No RUN_PINS file was required for this static review; the parent freezes it after this report.

The initial table correctly contains seven completed historical r1 rows and two pending A/B rows with blank accuracy cells; no r=.25 rows are included. The reviewer independently verified reused config/receipt/metrics/suite hashes and completion-to-metrics bindings, then recomputed AAA, FAA, LA, BWT, and forgetting from each of the seven raw native CNN matrices. The normalized LA/BWT table aliases agree. Gains are emitted only after the corresponding new row completes validation.

Limits: this is a CPU and static readiness review, not actual full-ViT/GPU execution or a training-result claim. The CPU loader/RNG oracle is a deliberate global-RNG input proxy, while the real native DataLoader is preserved by source inspection and will be checked by fresh GPU pairing. The initial four historical runs lack per-step trace/checkpoint-hash evidence; their reuse is receipt/raw-matrix/config verification and does not establish retroactive per-step identities. Single-seed natural-step A/B comparisons do not isolate amplitude effects or establish statistical significance. No current blocking finding remains.

Reviewed file hashes (before parent RUN_PINS freeze):

```json
{
  "legacy_direction.py": "d137cad76f1fd2475bed0da43e5f8a671ff3f7fc708ceea202378a98d8f1bd3f",
  "residual_selection.py": "4284c85df9700e1ef992cedc1bf1e2d2f784f687d680a605ace580c3ea83eab6",
  "effective_step.py": "97aa2b7bb87e8771f59b336b9b035e590bf13c8c7f514592ad2055430ed7abf3",
  "test_legacy_direction_cpu.py": "aebcd7f3d939ff66afcd903098082623d1535f5273cff94315f769b3c6f56746",
  "common.py": "0716fc09a36db78cb593ec2d5a3bee9c09062c06976dd6d4c75e34f8a1da04bc",
  "run_job.py": "14d6a575ab92458ff517453779334c4bc6e8da9fc030aec51dc862d49a1bf4f4",
  "launch_queue.py": "87dd5e843c6f62f8fa05c74facf08bfa688d986099be0fec846e2703ad4d5ea0",
  "collect_results.py": "e5df4afb1246f68fbc95db95155dfb8e340bde3698f2a8bc3c57ab760aac797a",
  "validate_smokes.py": "2f28ef27cf1b16f1ae12349ff9454418389bef55c54fafd0c8083bd57a9f3f09",
  "summarize_nine.py": "aec35237e5be4a18fe0f6a735f8d735f6cf9ba881a206adfe558cfdbba7b8327",
  "suite.json": "118383012458f583cfcb7138380c7631ffadbf7529738d7589bd9b3abacc143d",
  "REUSE_VERIFIED.json": "9239e3be404e8305b264cf02f9107bcc37d63e93dc72878ad5399f50b29de986",
  "review/LEGACY_DIRECTION_CPU.json": "c0dd40dd3529d359e6b0d1754ac0948022cd92f5164ee11d8e217371eb1b949f"
}
```
