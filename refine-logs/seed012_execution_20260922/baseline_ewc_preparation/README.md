# Four baselines + EWC: execution preparation

Status: prepared and CPU-checked; no GPU training launched by this preparation. Old-result reuse and whether to activate fresh seed0 runs remain for the root/Joost decision. The candidate inventory is not a launched queue.

## Frozen sources

- `source_baselines_flatness_aug11/`: exact package files from Flatness_CV Git object `9c4439b397ec749d906f980cbda46632e06b08f3`. Includes src, utils, backbone, all LoRA learners, optimizers and evaluation dependency packages. No implementation edits.
- `source_ewc_sep18/`: exact files copied from `CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/`; source files are checked against that run's source_frozen.json where listed. No implementation edits.
- `source_manifest.json`: individual file SHA256, aggregate hashes and origins. These are recoverable candidate versions; their recovery alone does not prove binding to every historical March/Aug run.

Both source trees preserve the old per-forward temporary Linear construction and CPU RNG consumption. They deliberately do not use the modified live CopyFlatCV backbone. Original repositories and the root queue were not edited.

## Candidate configuration inventory

- 20 baseline groups × seeds0/1/2 = 60 formal candidates.
- 5 EWC groups × seeds0/1/2 = 15 formal candidates.
- 25 independent smoke candidates, one per method/dataset group.
- `candidate_jobs.json` gives each exact `cwd`, argv, config SHA, expected task count and source ID. All activations are pending; the declared authorized physical GPU pool is [2,3,4,5], while configs use logical `device=["0"]` after CUDA_VISIBLE_DEVICES assignment.
- Per-run output roots are isolated under this preparation directory. Formal and smoke outputs have different paths. Existing archival paths are never used for output.

The formal candidates retain table-specific LR/epochs: SDLoRA Aircraft LR=.02; others Aircraft=.05. Het5 Vanilla=30 epochs/task, InfLoRA/OLoRA/SDLoRA=40; EWC Het5=30. EWC λ is Aircraft200, Cars80000, CUB200000, Flowers10, Het550000. Aircraft/CUB resolve frozen legacy direction defaults explicitly to ratio with alpha=0 and both RNG-preserve flags false; Cars/Flowers preserve alpha0 with both flags true; Het5 preserves native_sgd with both flags true. This explicit default resolution is a property of the selected candidate code, not a new claim that missing historical log fields were recorded.

`min_lr=.1*lrate`, train_num_workers=8, InfLoRA lamb=.7/lame=.99 and OLoRA coefficients1/0 are explicitly pinned to the candidate code defaults. `metadata.json` records every resolved-default/location change from the original YAML. Within each group, the 3 formal configs differ only in seed, prefix and outputs_root.

## Entrypoint and environment

Use the job's exact `cwd` and command:

```text
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -m src.main --config <absolute-json-path>
```

Load `environment.json` and add `CUDA_VISIBLE_DEVICES=<one of 2,3,4,5>`. This preserves the previous offline HF/Torch cache locations and deterministic CUBLAS workspace setting. Do not run CPU preflight's empty CUDA_VISIBLE_DEVICES in a GPU job. Before activation, root preflight should bind the effective cached pretrained weight bytes and environment receipt to the actual launched job.

The explicit dataset root is **`/data/115-2/users/liying/datasets`**, matching the previously used five-dataset directory. `/data/115-2/datasets` is a different shared collection and lacks these required subdirectories. Both JSON data_root and FLATNESS_CV_DATA_ROOT are set; this overrides Flatness DataManager's `/data/140-0/datasets` hard default without changing source.

## Smoke limits

Smoke uses the existing trainer's `max_train_tasks=2`, `init_epoch=1`, `epochs=1`. Its JSON retains the full configured `nb_tasks` and `total_sessions`, but the baseline trainer overwrites effective `args['nb_tasks']` with2 after applying the cap; the JSON value must not be mistaken for the runtime value. Smoke therefore checks functionality and finite outputs only, and its numerical trajectory is not compared against formal runs. Formal configs have no task cap and retain the complete10/5-task stream. Task boundaries, rank16, batch128, method parameters and scheduler floor are otherwise retained. No data subsampling, Fisher-batch cap reduction or NME auxiliary-pass removal is introduced. EWC's original Fisher cap100 remains; the smoke exercises the second-task EWC penalty. Smoke numbers must not be pooled with formal metrics.

## Completed CPU checks

Both `python -m src.main --help` calls succeeded. Selected learners import from their frozen source directories; CUDA was never initialized and no model was instantiated. `cpu_preflight.py` checked full data metadata and saved realpath/label manifests:

| Dataset | Task widths | Train | Test |
|---|---|---:|---:|
| Aircraft | 10×10 | 6667 | 3333 |
| Cars | 16+9×20 | 8144 | 8041 |
| CUB | 10×20 | 9430 | 2358 |
| Flowers | 12+9×10 | 2040 | 6149 |
| Het5 | 200,100,196,102,37 | 29961 | 23550 |

All listed files exist; train/test realpath overlap is0. Baseline and EWC candidate trees produce identical path/label hashes and task boundaries for all five datasets. Het5 still includes Pets37. This establishes equality for the currently resolved data, not the entire historical data provenance. Source and all100 config hashes were checked after CPU preflight. See `cpu_preflight_baseline.json`, `cpu_preflight_ewc.json`, `data_manifests/`, and `validation.json`.

Additional CPU checks completed in separate processes: each source tree produced one real train item and one real test item per dataset (20 items total), all float32 tensors with shape3×224×224, finite values and labels in range. See `sample_items_baseline.json` and `sample_items_ewc.json`; this is a sample-content check, not exhaustive image decoding.

`check_weight_resolution.py` actually loaded `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)` on CPU with the specified offline HF_HOME and traced timm's HF file-resolution call. It resolved `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`, snapshot revision `063c6c38a5d8510b2e57df480445e94b231dad2c`, `model.safetensors` SHA256 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`, exactly the CL-LoRA weight file. A second explicit `hf_hub_download(..., revision=<resolved>, local_files_only=True)` resolved the same bytes. Exact cache/real paths, versions, pretrained_cfg and trace are in `weight_resolution.json`. CUDA remained uninitialized; this confirms current resolution, not historical March/Aug weight identity.
