# Fixed lambda 20000: report implementation and CPU validation

This reporter implements only the four requested EWC arms at seeds 0, 1, 2. Lambda 20000 is explicitly fixed by the user. Neither selection nor completion is described as an optimum or a verified protection threshold. The SELECTION.json receipt and experiment_spec.json are hashed and checked.

Frozen implementation SHA256:

- report.py: `311adc316eecaf1e2fca4c3e57769fa99296e45a4732b600257bc6fb2ad0cb05`
- test_report.py: `b16bf6fdc0e3239ec98f5e8a5e177a5b7216643b9b5e73f13c7c91cb0f63410b`
- Read-only reused CPU helper, ../ewc_lambda_sweep_seed0_20260921/report.py: `1808e5b1c9767e8efb61ae3ff7ff76f511760ded087dfd904a710baa62377e67`. Only its generic RNG pairing and CPU effective-weight displacement helpers are invoked; its seed0 campaign validator is never invoked.

The exact arms are ewclora (SGD→SGD, C→C+P), fs_ewclora (GAM→GAM, C+F→C+F+P), task1_shaping (GAM→SGD, C+F→C+P), and task2_shaping (SGD→GAM, C→C+F+P). All twelve cells must have their own formal native COMPLETE.json to complete this report. Smoke, CPU validation, failed, divergent, and missing cells cannot satisfy the completion gate.

## Evidence gates

The collector checks the exact 12-cell specification, six same-seed parent paths and frozen SHA receipts, native 50-epoch Task0 completions, original Task0 checkpoint/Fisher provenance, and native Task0 metrics. B1_batch_proxy comes directly from the parent in_task_fisher_final_raw_trace (fit batch-gradient-square proxy, 1259 samples), not the historical train256 per-example estimator.

Prepared Fisher receipts must match the same method and seed, frozen parent checkpoint, full Task0 fit-Fisher input replay, positive raw trace, 24 Q/V blocks and 14155776 elements, and both normalization flags false. A pending Fisher receipt remains pending; a malformed existing receipt is invalid.

Each formal result requires a reproducible manifest identity, the frozen current trainer, fixed lambda/config/optimizer/seed/parent paths, exact Task0 restoration, native raw-Fisher replacement, complete paired Task1 loader/head RNG receipts, 400 finite updates, the actual final 50/50 epoch log, native CIL lower-triangle agreement, certified Task1 checkpoints, exact endpoint reload, and native endpoint local metrics. The reporter independently loads checkpoint tensors on CPU to check finite values, frozen backbone/old head, effective BA displacement, raw-Fisher-weighted displacement squared, and endpoint penalty scaling. It does not run model inference.

The telemetry requires a single native EWC injection per update, CE+P scalar loss agreement, finite component norms, and a positive second-update EWC gradient. GAM native clean_grad_cache and fisher_grad_cache retain their native source labels; the actual GAM update minus EWC is explicitly a shaping direction, not clean CE. Native GAM shaping and Fisher mechanism gradients must both be positive. Ratios use the sampled steps 0, 1, 8, 16, …, not all steps.

## Outputs and completion contract

Default outputs are report/{individual.csv,summary.csv,RESULTS.md,table.tex,results.json}. A report/COMPLETE.json is emitted only when all 12 results are valid and complete. Required fields are status=complete, expected_trials=12, completed_trials=12, validated_seeds=[0,1,2], ewc_lambda=20000 and lambda_star=20000. The receipt binds the reporter, specification, selection, and every output SHA.

Only a complete three-seed arm receives a mean and sample SD (ddof=1). Incomplete arms get no aggregate even if one or two seeds completed. Divergence telemetry is explicitly limited to the finite prefix and cannot supply endpoint metrics or substitute for a missing seed. Recollection removes any stale report COMPLETE before validation. Without --allow-partial, a partial collection saves diagnostics and then exits unsuccessfully.

Existing lambda0 controls are explicitly marked not_revalidated; no new lambda0 training is launched and no causal EWC protection gain is asserted. Historical normalized lambda2000 and raw lambda20000 are not treated as a matched scale.

## Validation performed

From the project root, CPU-only regression command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -m unittest discover -s refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922 -p test_report.py -v
```

Result: 12 tests passed, exit 0. Tests exercise complete and partial output receipts, sample SD, missing/duplicate seed rejection, divergent/invalid exclusion, nonfinite CIL and telemetry rejection, loss scaling and single-EWC checks, GAM source labels, prepared-Fisher seed/method/hash rejection, conflicting native terminal receipts, stale COMPLETE removal, source changes, unsafe output paths, and unexpected extra formal cells. Synthetic test results exist only inside temporary directories and are not empirical results.

Real partial validation command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/report.py --allow-partial --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/report_partial_validation
```

Result: exit 0; status partial, expected_trials 12, completed_trials 0, pending_trials 12, invalid_trials 0, diverged_trials 0. Six distinct real parent checkpoint hashes validated; parent_errors is empty. Every aggregate mean/SD is null and no COMPLETE.json exists. Durable outputs are report_partial_validation/results.json, individual.csv, summary.csv, RESULTS.md, and table.tex. SGD seed0's copied real raw-Fisher payload was verified (`cfdcd801f773d3d33cb48885c84ff31f032a60bf4a4cccbdead5008d67aacd95`); the other five prepared receipts were pending at validation time.

An additional read-only native interface check imported train.py, called its parser/build_config/experiment_identity for the real formal SGD seed0 arguments, and passed the resulting identity/config through report.validate_manifest with the real parent and prepared-Fisher receipt. It passed without model construction, GPU access, forward evaluation, or training. The CPU regressions and partial validation do not constitute an empirical full-completion test; formal GPU runs remain pending.

Final native-file compatibility check read the actual cpu_validation/cpu_ewclora_seed0/cpu_manifest.json and cpu_validation/cpu_task2_shaping_seed0/cpu_manifest.json, together with their CPU_VALIDATED.json receipts. Both passed reconstruction of the exact native identity SHA, using prepared_fisher.execution_ready (the native manifest does not store that field at top level). Verified identities are `a753c06dadcaa676b4743f29934b137eec7ce6d8fb810fabdde13a58635546ab` and `e5e7b936bb6c2bc11bfb3674a23513d1ea0e4669bea52a5c0476e060e6149a1b`. The reporter also accepts the native recovered_after_final_task=true receipt variant that omits its manifest path, while still verifying the already located manifest, identity, parent and complete native evidence. The 12 CPU regressions passed again after these compatibility corrections.
