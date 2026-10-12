# G0/G1/G2 two-stage environment

This run reuses `/home-local/liying/.conda/envs/Pilot_new_local/bin/python` on `cudahpc40.cvc.uab.es`. No packages were installed or upgraded. `environment_spec.json` was written before CPU imports and model validation. Its canonical SHA256 is `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8` (JSON sorted keys, compact separators). It is a separate local specification from the older `Pilot_new@e94a148f` ledger entry.

Physical GPUs **2, 3, 4, 5, 6** are allowed. GPUs **0 and 1 are forbidden**. Each worker must be bound by the root supervisor to one idle physical UUID under `/tmp/olora_gamfs_gpu_locks`, then use logical CUDA device 0. This reviewer has not claimed or executed a GPU.

## Verified software and assets

| Component | Actual version |
|---|---|
| Python | 3.11.15 |
| Torch / compiled CUDA | 2.4.1+cu121 / 12.1 |
| Torchvision / timm | 0.19.1+cu121 / 1.0.9 |
| NumPy / SciPy / scikit-learn | 2.2.6 / 1.17.1 / 1.8.0 |
| Pillow / safetensors | 12.1.1 / 0.4.5 |
| Hugging Face Hub / einops | 1.7.2 / 0.8.0 |
| tqdm / PyYAML / packaging | 4.66.5 / 6.0.2 / 24.1 |
| Matplotlib | 3.10.6 |

All required listed imports completed in a CPU-only process with bytecode writes disabled. Tensorboard and WandB are not installed and are not required by the historical scientific entrypoint.

The frozen `utils.inc_net.IncrementalNet` successfully executed the real offline pretrained loader: `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)`. It selected `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`, revision `063c6c38a5d8510b2e57df480445e94b231dad2c`, from `/home-local/liying/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k`. The `model.safetensors` snapshot link points to blob SHA256 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`, whose actual size is 346,284,714 bytes and whose bytes were SHA256-verified.

Use the environment variables in the JSON verbatim. The historical suite explicitly overrides `HF_HOME=/home-local/liying/.cache/huggingface` and `TORCH_HOME=/home-local/liying/.cache/torch`; the interactive shell defaults to `/data/115-2` caches. Set offline flags, thread counts 8, `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, bytecode suppression and unbuffered output. Per-worker temporary directories must be writable under the new run directory or `/tmp`. Training remains FP32 with AMP off and both TF32 flags false.

The source root is `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`. The old `source_manifest.json` SHA256 is `e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2`; all 190 recorded Python/JSON files matched actual bytes, and there are no additional unmanifested Python/JSON files. Plan snapshot hashes are in the JSON. No external files were modified.

## Actual data and step budget

The real `DataManager("imageneta", False, 2, 20, 20, cfg)` resolved `/data/140-1/datasets/imagenet-a/{train,test}`. Both ImageFolder splits have the same 200 lexicographically sorted WordNet folders; class indices stay 0..199 and there is no shuffle or replay. Train has 5981 images, test 1519. The JSON records path/label and path/label/byte-size hashes; image payload content hashes were not computed in this review.

| Task | Classes | Train | Test | Batches/epoch | Updates at 50 epochs |
|---|---:|---:|---:|---:|---:|
| 0 | 0–19 | 1259 | 307 | 10 | 500 |
| 1 | 20–39 | 997 | 254 | 8 | 400 |
| 2 | 40–59 | 993 | 258 | 8 | 400 |
| 3 | 60–79 | 596 | 165 | 5 | 250 |
| 4 | 80–99 | 426 | 102 | 4 | 200 |
| 5 | 100–119 | 260 | 71 | 3 | 150 |
| 6 | 120–139 | 357 | 83 | 3 | 150 |
| 7 | 140–159 | 317 | 85 | 3 | 150 |
| 8 | 160–179 | 398 | 90 | 4 | 200 |
| 9 | 180–199 | 378 | 104 | 3 | 150 |
| Total | 200 | 5981 | 1519 | 51 | 2550 |

Batch size is 128 and `drop_last=False`; `50 * sum(ceil(n_task/128)) = 2550`, with 500 completed task epochs. The train transform is RandomResizedCrop224 with scale (.05,1), ratio (3/4,4/3), bilinear antialiasing; RandomHorizontalFlip(.5); ToTensor. Test uses Resize256 bicubic antialiasing; CenterCrop224; ToTensor. Neither split normalizes image channels, despite the pretrained timm metadata listing mean/std.

Reuse the historical `EpochLoader` lifecycle: create an independent loader each epoch with `data_seed = 2*100000 + task*1000 + epoch` (epoch is 1-based), a torch Generator, the worker init function seeding NumPy and Python from `torch.initial_seed()`, 8 training workers, and `persistent_workers=False`. Separate data RNG from model RNG around iterator creation and each fetch. Record full epoch sampler digest and batch input/target/index hashes.

## Minimal scientific trainer hook

Preserve the native `SeqLoRA.incremental_train -> update_fc -> _train -> _build_optimizer -> cosine -> _run_regularized_train` lifecycle. The smallest replacement is `learner._step_batch_with_log`: perform the new probe protocol directly and return base-point detached logits, base-point CE, and a log dictionary with zero EWC/delta-W terms. The native epoch loop can then preserve scheduling and task persistence. Calling the historical batch function would execute an extra clean gradient plus native GAM's g0/g1/g2/g3 closures, violating the new budget. Changing only the old final gradient selector is insufficient.

A GAM optimizer shell can be retained to minimize persistence changes, provided its native `step` is never invoked: submit exactly one `base_optimizer.step()` with the chosen direction, using the shared `param_groups`; the cosine scheduler remains attached to the shell. Alternatively return plain SGD and update the new runner's persistence logic consistently. Verify momentum and weight decay are exactly zero; task0 LR .04, other tasks .02; cosine `T_max=50`, `eta_min=.1*initial_lr`, stepped once after each full epoch.

The actual model has 24 Q/V LoRA pairs (12 blocks × 2 channels), 48 A/B weight tensors, rank10 and 368,640 factor parameters. All pretrained backbone parameters, including `base_vit`, are frozen. `IdentityScale` gives actual BA scale1 with `learn_alpha=False`. At task0 the FC has shape20×768 plus bias20, making 50 trainable tensors and 384,020 trainable parameters. Native FC expansion to40×768 preserves the old20 rows exactly and yields 399,400 trainable parameters. LoRA factor objects persist across tasks without resetting.

`IncrementalNet.update_fc` creates and initializes the entire expanded `SimpleLinear` head using Kaiming uniform with `nonlinearity="linear"` and zero bias, then copies all old rows. All seen FC rows remain trainable and belong in the optimizer, Hessian coordinates and effective BA+FC norm. Current task CE must be exactly `cross_entropy(logits[:,20*t:20*(t+1)], labels-20*t)`. Older FC rows consequently have zero CE gradients but still belong in the declared geometry. Keep parameter/name order identical across gradient vectors and record newly initialized FC row hashes per task.

At each batch use a single original model/RNG snapshot and restore it before and after each ordinary read in the order **g0 -> g1+ -> g1− -> g2m**. The two first probes are phi±.05*N(g0); each outer point is **phi+.25*N(q)** from the original phi. CE/S+/S− still perform P's outer probe as a side diagnostic. Use the plan's exact `N(v)=v/(norm(v)+1e-12)`; the old `audit_core.unit` instead divides by norm without this epsilon and uses a hard1e-20 cutoff, so it is not a faithful replacement.

P/D/C/R select the plan's signed first difference; O submits g2, A submits g0+g2−the corresponding g1. Matched updates reuse g1+ at the same state and batch as the effective BA+full-FC reference; there is no additional reference forward. HVP branches keep these four ordinary reads and separately construct an arbitrary detached-vector HVP at the prescribed point. Rebuild each HVP graph at that point before restoring parameters; do not reuse a graph after in-place parameter restoration. HVP and Sec must share the same backend, stochastic realization and current-task CE slice.

The historical `audit_core.Snapshot` handles parameters/gradients, registered buffers, module modes and RNG, but does not cover unregistered module caches. Extend it to include the actual QKV cache fields (`delta_w_q_new_grad`, `delta_w_v_new_grad`, `external_delta_w_q`, `external_delta_w_v`, `_register_delta_hook`) and any new probe caches. The source LoRA forward allocates temporary random `nn.Linear` objects each call, even when those temporaries do not affect the output, so RNG restoration is required even with dropout off. Additional HVP or evaluation reads must not change future model or FC initialization streams. Isolate task FC initialization with a predeclared common stream or demonstrate its row hashes remain paired across all conditions.

Keep the existing forbidden-work guards and task-end tensor/parameter checks. No Fisher estimation, projection, weight merging or penalty calls may occur. Explicitly disable `compute_nme_class_means`: native `incremental_train` otherwise defaults to reading all seen training classes for NME at task end. Native periodic accuracy and all isolated diagnostics must restore parameters, buffers, cache, modes and RNG.

## Validation status and documented invocations

Tier1 imports, file integrity and actual offline CPU model/data construction passed. GPU kernel dispatch and actual training are **pending**; no GPU witness, HVP gate, smoke or full experiment success is implied by these CPU results.

The canonical witness command is stored in `environment_spec.json` under `smoke.gpu_tests[0]`, with expected sentinel regex. The root supervisor must run it after binding an allowed idle GPU UUID, using the JSON environment variables and FP32/TF32-off settings. It returns finite seeded32×32 matmul forward/backward evidence and the device name. This reviewer did not run it.

A fresh verifier must receive the final launcher documentation, this local environment record, and the exact root-provided invocation. The final documented dry-run invocation has not yet been supplied to this reviewer, so the Tier3 agent-follows-doc pass remains pending. Per the compute contract, the author of this document cannot self-certify that pass.
