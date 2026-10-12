# CL/GAM preparation — no jobs launched

This owned directory contains a byte-for-byte copy of 172 files from the verified Sep18 formal snapshot, 24 formal candidate configs, 10 smoke configs, 6 existing seed0 reuse candidates, import/hash evidence, and copied original runner references. Nothing is registered in a live root queue.

Read `jobs_manifest.json` for per-job config paths, hashes, dependencies, source root and entrypoint. `selection_pending` is intentional: root first resolves the joost inventory and which historical GAM sources can be recovered.

## Ready exact-source groups

- CL-LoRA Aircraft/Cars/CUB/Flowers/Het5: reuse formal seed0 conditionally; prepared seed1/2 use exact same full source closure, training hyperparameters, loader and scheduler execution as formal seed0.
- GAM-EWC Het5: reuse formal seed0 conditionally; seed1/2 preserve exact source and actual behavior, including the ignored clean-isolation flag.
- No training resumption is needed: each missing seed trains independently from the pretrained backbone. Do not load another seed's trained model.

Only intentional config differences for those six groups are seed, prefix/output location, output tag and explanatory note. CL `min_lr=0` is inherited directly from the unchanged CosineAnnealingLR invocation. Do not add a ratio flag expecting CL to read it.

## First four historical GAM groups are candidates

Their seed0/1/2 configs preserve archived LR/epochs/rank/batch and recovered GAM settings. Workers=8 and Fisher-RNG preservation=false explicitly represent source defaults that were not fully recorded historically. Current candidate source is Sep18 formal; it is **not asserted to be the historical source**. Historical seed0 numerical parity and dependency identity need review before seed1/2. Parent may replace the source with a better recovered original source identified by joost inventory/audit.

Flowers rho=.2, normrho=.2, gamma=.1 are now explicit choices, supported by frozen source defaults and a separate controlled config. Historical Flowers log omitted them; do not claim recovered historical certainty.

## Effective settings that a launcher must preserve

- Four single datasets: 40 epochs/task; Het5: 30. LR Aircraft/Cars .05, CUB .01, Flowers .0025, Het5 .02.
- Rank16, batch128, SGD(momentum0,wd0), no replay and fixed class order.
- CL: shared blocks0–5, specific6–11, Q/V, KD5, temperature2, orth1e-4; workers2; direct loader pin=true,persistent=false; cosine minimum0.
- GAM: lambda2000, SGD base, cosine minimum.1×LR. Aircraft rho/normrho/gamma=.01/.1/.02; Cars .05/.2/.03; CUB/Flowers/Het5 .2/.2/.1.
- GAM-Het5 workers2; ordinary global-loader behavior; Fisher preservation=true. `ewc_isolate_clean_rng=true` remains in the config but this frozen source does not read it. Do not silently implement that flag for new seeds.
- This snapshot's shared LoRA training forward retains four temporary Linear initializations and their RNG consumption. Do not swap current Copy working `backbone/lora.py` into it.
- DataManager does not implement deterministic_loader_rng. Aircraft dataset loader_options make GAM Aircraft pin_memory/persistent_workers true; a top-level persistent_workers=false would not override them.

## Entry point and smoke

Original `run_job.py` and `launch_queue.py` are copied with `.reference` suffix for inspection only. Original runner accepts `--job` and resolves `suite.json` relative to itself; therefore it cannot be invoked against these configs without a new parent suite/runner. Never append jobs to or restart the original completed campaign.

The new root runner can copy its finite-gradient / finite-parameter guard unchanged and call:

`src.main.main(["--config", config_path, "--override", "device=[0]"])`

with this frozen code first on sys.path and cwd set to this code. `import_check.json` proves CPU-only imports of main/trainer/factory/CL/GAM resolved to this directory. No model was instantiated and no training was executed.

Each smoke sets max_train_tasks=2 and init_epoch=epochs=1 while preserving nb_tasks, class splits, rank and batch. Frozen trainer lines929–933 honor this cap, then loop at1111. This lets the runner validate a 2×2 CNN matrix and exercise task1's KD/EWC behavior. Full configs do not have max_train_tasks, so expected task count is10 or5. Smoke results are excluded from aggregation.

Root runner should write independent receipts with config/source/data/cache hashes, exact command/env, GPU UUID, start/end/exit, peak allocated bytes and final metric SHA. Enforce complete finite lower-triangle CNN matrix, no nonfinite training, and never overwrite prior receipts. Source manifest is `source_manifest.json`; ordered data manifests and original preflight are included. They bind paths+labels, not image bytes.

## Environment

Use the Python and environment dictionary already included in jobs_manifest. Configure user-authorized GPU2/3/4/5 exclusively in the root scheduler using freshly queried UUIDs; this preparation directory assigns no GPU. Each child needs PYTHONHASHSEED=str(seed) before startup, CUDA_VISIBLE_DEVICES=<one permitted UUID>, logical device=[0]. Set offline HF/Torch caches and OMP/MKL/OPENBLAS threads2, as in the original suite. `cache_manifest.json` verifies the explicit CL safetensors SHA; GAM's resolved cache identity should be recorded at launch.
