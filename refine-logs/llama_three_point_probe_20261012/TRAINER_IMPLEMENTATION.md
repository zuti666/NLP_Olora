# Native three-point trainer integration

This is a separate optimizer family `three_point_probe` (`threepoint-adamw_hf`) in `src/uie_trainer_lora.py`. Legacy G2 and two-point execution remain separate. `src/run_uie_lora.py` accepts the existing radius/epsilon/proxy flags, with new mode values validated only for this family.

Each realized effective accumulation window reads the native clean gradient then replays the same micro-batches twice, at BASE plus and minus the same `rho * g0 / (norm(g0)+eps)`. Each micro-batch replay restores its clean-pass RNG (Python, NumPy, CPU, CUDA). Both side backpropagations run under DDP `no_sync`, followed by one explicit global mean reduction per side. Short final windows use their actual count for both the clean gradient and each replay. Only the selected final direction is clipped; BASE is restored before one HF AdamW commit. Scheduler and zero-gradient steps retain the native one-commit flow.

All four controls execute exactly three CE reads and two replay closures, including g0 and degenerate radius-zero cases. Modes: `g1_plus` -> `gplus`; `backward_reflected` -> `2g0_minus_gminus`; `central` -> `g0_plus_half_gplus_minus_gminus`; `g0` -> `g0`. No G2/G3/Fisher/HVP/history/projection/OLoRA regularization runs.

Training requires `inclora`, FP32 trainable LoRA with BF16 forward permitted, corrected `decoder_token_ids_v2`, HF AdamW epsilon 1e-6 and zero weight decay. The entrypoint also includes this family in adapter FP32 promotion and rank-first shared-dataset preparation.

New execution evidence is independent: `three_point_probe_step_trace.jsonl` with `ThreePointProbeTrainerStep_v1`, and `three_point_probe_training_audit.json` with `ThreePointProbeTrainerAudit_v1`. The final audit requires exactly three phases, two replay closures, one commit, both signs [1,-1], no fallback, correct selected direction and exact BASE restoration on every step. First-window fingerprints identify the realized parameters, raw clean global gradient, and effective batch.

Tests are in `tests/test_three_point_probe_trainer.py`; validation results will be recorded alongside this file.

Validation passed: 10 new distinct trainer checks and 19 existing trainer checks (29 total). The combined CPU-only host run passed all 28 original discovered cases, including two-rank Gloo for each of the three optimizer families; the additional native legacy-equivalence check passed separately. No GPU was used.
