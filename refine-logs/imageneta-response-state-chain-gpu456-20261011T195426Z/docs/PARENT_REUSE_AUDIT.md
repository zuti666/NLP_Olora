# Parent reuse and evidence-chain design audit

All three accepted 50-epoch SGD Task0 parents are reusable. No additional parent training is needed. The formal budget remains six 50-epoch Task1 branches.

| Seed | Accepted parent output | Complete parent file SHA256 |
|---|---|---|
| 0 | `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-upward-eight-fork-20261010T123703Z/outputs/formal_parent_sgd_s0` | `bcb8a7eed684ad0a1ff72bbc35a98380d5e7fa8f41afdf85aaf39f0dd127fd69` |
| 1 | `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-upward-eight-fork-20261010T123703Z/outputs/formal_parent_sgd_s1` | `5a337d7b0fbacad77432cb80df7f713b60b54f49b7c08272acfc0bac37637375` |
| 2 | `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g2-minimal-fork-gpu456-20261009T223549Z/outputs/formal_parent_sgd_s2` | `4a10149a64f62647fbd673df3bab7d34ef83115316a1c5c0963044655ee35cde` |

## Verified compatibility

Each parent trained Task0 only, classes 0–19, for 50 epochs: 450 native SGD updates and 450 gradient reads. Rank 10, batch 128, LR .04→.004 over one 50-epoch cosine schedule, momentum and weight decay 0. Task0 train/dev/test sizes are 1141/118/307. All use the same 190-file frozen source, train/dev split, pretrained revision, native FP32 and software. The learner source is `models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py`; there is no `models` subtree.

Seed 0 and 1 retain upward-family radius metadata, but all 450 trace rows submit `g0`, read one gradient and never read g1/g2/g3. Actual Fisher, fusion, HVP, projection, clipping and history counts are zero. These are ordinary SGD parents despite the metadata labels.

Each `checkpoints/task_000` manifest records task 0, epoch 50, head shape 20×768 and backbone task ID 1. All native LoRA A/B, FC, training-state and complete-parent file hashes were verified. `shared_parent.pt`, saved after Task0 completion, is the authoritative complete network and RNG. `training_state.pt` includes optimizer/base optimizer, scheduler, RNG, algorithm state and last batch identity. Its RNG hash matches the complete-parent manifest. All three schedulers record T_max=50, last_epoch=50 and final LR=.004. None contains trained Task1 state.

The separate 20-epoch formation experiments are excluded because their parent states were formed under a different schedule. Existing Task1 endpoints cannot replace these Task0 parents.

## Common start and formal budget

Restore the complete parent and RNG, perform native Task1 head expansion once per seed, save the whole expanded 40-class Task1 start state, and clone it for S and R. Compare parameters, buffers, modes, LoRA controls and all new FC rows before the first update. Pair augmented minibatches by logical (seed, task, epoch, batch), independent of probe count. Passive diagnostics must restore training state and RNG.

- S: native SAM for epochs 1–25, then SGD for 26–50. Positive first probe, normalized radius .05.
- R: native S_F direct g2 for epochs 1–25, then SGD for 26–50. Descending probe θ−ηg0; response g0−g1minus; outer point θ+.25N(response).
- Keep the full 50-epoch Task1 cosine schedule. At epoch 26, change only the update kernel; do not reset optimizer, factors, head, scheduler or RNG.
- Trainable coordinates are 48 Q/V LoRA tensors plus FC weight and bias, without alpha/rank scaling.
- Each branch commits 400 updates. S uses 600 training gradient reads, R uses 800. Six branches total 2400 updates and 4200 training gradient reads. Smoke gates and passive diagnostics are reported separately.

## Design boundaries

1. Use the exact loss Hessian H in the native LoRA+FC coordinates for the quadratic-response test. The local correction is rH²g/||Hg||. A generic Fisher or PSD proxy C gives the frozen-proxy derivative HᵀCg, rather than C²g, unless C approximates H. H can be indefinite; gᵀHg/2 is not necessarily a nonnegative energy.
2. Compute q=Hg and p=Hq at the same base point and loss, detaching q before the second HVP. Check repeated base gradients and centered finite differences over a registered stable range. Any attention-backend or precision change is offline, recorded and checked against native first-order gradients.
3. Keep formal methods native. SAM and S_F differ in probe sign, probe distance and gradient selection, so this is a complete-method comparison. Halving the two probe distances is an offline diagnostic and does not change the committed training LR. Register which saved LR is used after each epoch scheduler step.
4. Compare correction vectors, their norms and relative error. Report valid and degenerate counts for all 36 state×batch combinations at each scale. These are not independent seeds. Fixed probe batches are Task1 training samples and exclude holdout/test data.
5. Record exact ΔQ and effective ΔW=BΔA+ΔBA+ΔBΔA from base factors and actual committed changes. Record FC changes separately. Ordinary SGD first-order ΔQ cancellation is an algebra check, not a claim of constant finite-step Q. Effective-weight norms can use validated exact Gram contractions.
6. Offline effective-step matching includes all LoRA effective weights plus FC and uses one global scalar for R. Register solver bounds and failures; account for the quadratic BA term. This partly controls immediate scale, not the full trajectory. Do not normalize formal updates.
7. Define old-start accuracy after head expansion and before Task1 updates. Main old/new accuracies use joint 40-class competition; FAA is their equal mean. Store old CE growth, clean new-task training CE and joint Class-IL test CE separately. Old-20-only diagnostics cannot replace the retention metric.
8. Main paired comparisons are epochs 25 and 50 and each branch’s 25→50 tail change. Fit matching uses preregistered clean-training-CE first crossings within the observed common range. No extrapolation or test-based checkpoint selection. Durable state effects do not prove Q is the unique mediator.

## Audit receipt

The JSON receipt contains 66 passing checks, all 190 source-file checks, exact paths and hashes for accepted summaries, completion receipts, configs, traces, native weights, initial identities, source, environment and split evidence. No old campaign file or GPU process was changed.
