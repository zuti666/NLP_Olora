# Raw Fisher preparation

`prepare_fisher.py` reuses the complete formal `anchor_sgd_seed0/checkpoints/task_00.pt` and **actually computes one native raw fit-Fisher pass**. It does not train Task0, convert the old normalized tensor, invoke `after_task()` again, or edit the parent checkpoint/source.

The GPU command is a preparation job in the campaign, after its all-GPU admission gate:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python prepare_fisher.py --gpu 0 --gate-receipt campaign/ADMISSION.json
```

The scheduler supplies the actual GPU. The script checks admission status, the seven-card policy/lock receipt and the assigned GPU UUID; it does not acquire cards itself. Do not run this command outside the authorized queue.

CPU validation can run before admission:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python prepare_fisher.py --validate-only
```

It hides CUDA, checks frozen-source and parent hashes, replays all 51 sampler passes with a dummy CPU DataLoader, and consumes the real 51st pass with the original four-worker train augmentation. It does not instantiate a model or write the prepared tensor/receipt. Four workers need local IPC permission; sandbox restrictions on Unix sockets can prevent this check. Temporary IPC files use local `/tmp`.

## Native estimator and replay

The task Fisher and accumulated Fisher normalization flags are both false. The estimator remains the original `batch_empirical` proxy: squared gradients of batch-mean true-label CE in effective Q/V coordinates, weighted by each batch's size and divided by the total sample count. This required sample averaging is retained. The original data split, train augmentations, batch 128, four workers and `persistent_workers=False` remain unchanged.

The original loader has its own generator seed `30260939`. `rng_state_00.pt` does not contain this generator and was saved after Fisher. The script instead reproduces the installed PyTorch DataLoader's generator evolution: 50 complete dummy iterations, then uses the saved generator state for the actual native Fisher pass. All 51 original sampler hashes must match.

Expected Task0 Fisher pass:

| Field | Expected value |
|---|---|
| Iteration | 50 (zero based; 51st pass) |
| Samples / batches | 1259 / 10 |
| Generator state before pass SHA256 | `e96fd158acab6aab57bf4ef2da130f385217e901e5261ed358fe9d6143baaed6` |
| All indices SHA256 | `a083e73464e380518570242d22fbced2c71555fdb715e64a0a41bb8ed05daeda` |
| First-batch pixels and labels SHA256 | `40e2699706d5d26d71ec6ab7182152ec1ea0da74518627476b5dbce3572c518f` |
| Expected raw trace | `34.06988796591759` (relative tolerance 1e-4, absolute tolerance 1e-5) |

All indices and first-batch pixels/labels are checked against original receipts. The original run did not save an all-batch pixel digest; preparation additionally records this digest for future checks without claiming it was independently compared to an original digest.

The complete network is restored through the original formal runner's `install_runtime`/`patched_load`, imported from `FORMAL/train.py`, not the new sweep runner. This avoids a circular dependency on prepared Fisher. The old normalized Fisher is cleared before any Fisher computation. Since resume sets `known_classes=20`, preparation directly calls the native `_compute_delta_fisher(..., class_offset=0)` with the original 20-class head; calling `after_task()` would use the wrong class offset.

The model's complete state (all parameters and buffers) is hashed before and after the pass and must be identical. The parent files are rehashed too. Expected Fisher output: 24 finite nonnegative tensors named `backbone.lora_vit.blocks.{0..11}.attn.qkv.{q,v}`, each 768×768, with 14,155,776 elements total. Empty-history accumulation with both flags false must leave every value unchanged.

## Shared artifacts

Only a successful native GPU pass publishes:

- `prepared_fisher/fisher_state_raw.pt`: `fisher_past_delta`, `raw_trace`, `stats`; compatibility fields `fisher_new_task_raw_trace`, `last_task_new_fisher_stats`, `fisher_past_trace_mass=0.0`.
- `prepared_fisher/PREPARED.json`: `status=complete`, parent checkpoint path/hash, original source manifest hash, shared payload path/hash, both normalization flags false, seed/task zero, estimator/source, trace/mean/count/shapes implied by verified tensor keys, full weight hashes, matched input receipt, producer path/hash, and admission receipt path/hash.

The payload uses tensors and plain Python values, so consumers can use `torch.load(..., weights_only=True)`. The receipt is atomically published after the tensor and all checks succeed. A rerun verifies an existing completed preparation before reusing it. Missing or invalid preparation must block every training trial.

## Validation status

CPU provenance, all 51 dummy sampler passes, and the real four-worker augmented Task0 pass were verified: indices and first-batch pixels/labels matched exactly, with CUDA uninitialized and no model forward/backward. The newly measured all-batch input digest is `6a86607137d85a2f3c9f05f22cb8ba27b27ef05a9e4d7976c4353c97f0b0eb1e`.

An additional CPU-only full restoration check instantiated the native learner with both normalization flags false and matched every restored network-state tensor to the original parent. Its full tensor digest was `0ca898a69206550ddab2e0cd4ae3373ca8bec87dce4724de3051304642fc37e2`; this check also performed no model forward/backward and initialized no CUDA context. The actual scheduler receipt schema was accepted in a fixture test, while blocked GPU, incomplete lock set, incorrect polling policy, and waiting status were rejected. A mismatched first-batch pixel digest was rejected before any batch could reach the native model.

Native GPU recomputation has not been run during implementation. Its result will be accepted only through the queued job and the checks above.
