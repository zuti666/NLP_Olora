# Current B radius extension

This run adds integer radii 11–19 to the existing seed0 ImageNet-A measurement. Each method uses the same 16 random directions, 256-example Fisher subset, Task0 anchor, original 20-class head, effective Q/V coordinates, decoder, checkpoints, source files and full FP32 settings as `execution/results_seed0_recovery1`. No training occurs.

`configs/radius_seed0.json` is the historical `diagnostic_config_recovery1.json` with exactly two changes: `output_dir` points to this extension's `results`, and `radii` is `[0,11,12,13,14,15,16,17,18,19]`. Historical raw files and receipts remain in their original directory.

## Commands

Use the pinned interpreter. Plan validation runs entirely on CPU and is the default:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
  refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/radius_extension_20260921/queue_radius.py --dry-run
```

Execute the approved queue after its final plan is frozen:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u \
  refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/radius_extension_20260921/queue_radius.py --execute
```

Use the same command with `--resume` for a stopped run. The existing scheduler rejects changed code/plan fingerprints, modified completed receipts, or live prior children. The default `campaign/plan.json` uses physical GPU 0 for SGD and GPU 3 for GAM, pins their historical UUIDs, and uses the shared `.cache/gpu_locks` UUID locks. It waits for idle GPUs. The queue executes the original `execution/full_rq2_20260920/campaign_runner.py` without changes.

## Gates and outputs

- Each method first runs the unchanged `Runner.sanity()` and writes its builtin report inside its new sanity directory. A newly measured extension anchor is checked numerically against historical `r0`.
- Each method then actually evaluates historical `r10_u00`, including train/test forwards and the Fisher calculation, into a separate `repeat_witness/states.jsonl`. This row is not included in production neighborhood results.
- Only after both gates pass does the method publish its passing sanity receipt. Production calls the unchanged `Runner.neighborhood()` and has exactly 145 rows: one extension anchor and 9 × 16 perturbations. The neighborhood summary includes `radius_extension_guard_passed: true` and `new_perturbation_states: 144`.
- `identity_guard.json` compares every historical manifest field except `config` and `protocol_sha256`; the config changes are separately constrained. The complete data manifest must match, including subset indices/IDs, decoded tensor hashes, decoder versions and source image hashes. Random direction metadata is checked against historical records before each perturbation.
- `anchor_comparison.json` and `repeat_witness/summary.json` retain individual observed numerical errors and tolerances. Loss absolute tolerance is `3e-6`; accuracy tolerance is `1e-10` percentage points; raw Fisher trace tolerance is `1e-6 + 1e-4 × abs(reference)`; directional Fisher tolerance is `1e-8 + 1e-4 × abs(reference)`. Point hashes, direction metadata, sample counts and coordinate/data/checkpoint identities require exact equality. Timing fields are excluded.
- A mismatch raises an error before production starts, with failure details preserved in `failure.json` and the available comparison receipts. Original source, raw data and receipt files are never rewritten.

The extension-only neighborhood summaries do not describe tolerance radii across the full legacy grid. The fifth queue stage, `publish_B`, automatically runs CPU postprocessing after both neighborhood jobs complete. It checks and merges 194 original states with 288 new perturbation states, regenerates the existing three-panel B figure at 16 radii, and saves a copy as `Figure_B_neighborhood_extended.pdf/.png`. `POSTPROCESS.json` records completion; rendered-figure review is performed separately. All losses use the original historical anchor as their damage reference. Subsequent calls to either existing B plotting entrypoint reuse the completed extension through `active_radius_extension.json`.

## Preparation checks

See `cpu_checks.json` for compile results, historical source/common/diagnostic hashes, data manifest self-consistency, all 16 regenerated CPU direction hashes against both methods, and a witness comparison check that rejects an injected trace mismatch. `campaign/dry_run.json` records queue validation. Runtime GPU identities, data/checkpoint hashes, numerical anchor matching and r10 repeat matching are checked during execution; CPU checks do not substitute for those gates.
