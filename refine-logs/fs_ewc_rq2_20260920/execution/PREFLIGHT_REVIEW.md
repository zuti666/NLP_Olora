# RQ2 independent preflight review

Date: 2026-09-20. Scope: read-only review of the plan, asset audits, and current CopyFlatCV source. This document is **not** a GPU measurement pass. No training or GPU evaluation was launched by this reviewer.

## Source constraints confirmed

- ImageNet-A evaluation uses PIL RGB, `Resize(256, interpolation=3)` (bicubic), `CenterCrop(224)`, `ToTensor()`, and **no mean/std normalization**. Both the new deterministic training diagnostic and test diagnostic must use this transform. Source: `utils/data.py::build_transform/iImageNetA`, `utils/data_manager.py::pil_loader`. Training continuation still uses the original random crop/flip transform.
- Task0 is the first 20 sorted ImageFolder classes with `class_shuffle=false`. Assert matching `class_to_idx` across train/test, 1259 Task0 training images and 307 test images. Persist the exact stratified 256 training IDs and image/tensor hashes; both anchors and all diagnostic blocks share them.
- The checkpoint's `network_state` contains the frozen pretrained backbone. Construct the matching architecture with `pretrained=False`, then strictly restore the complete state (including the exact-size classifier). Do not rely on downloading today's timm alias weights. Source `utils/inc_net.py::get_backbone` otherwise hardcodes `pretrained=True`.
- The main chain has one continuing rank10 adapter, `IdentityScale`, no historical adapter sum. Assert these facts at load. `Delta_l=B9_l A9_l-B0_l A0_l` over 12 Q and 12 V matrices; total dimension 14,155,776. Do not sum adapters across checkpoints or interpolate factors.
- Use `model.eval()` to set inference mode, but do **not** pass `eval=True` to the LoRA forward: that argument calls `reset(eval=True)` and switches to the other historical-adapter evaluation implementation (`backbone/lora.py::LoRA_ViT_timm.forward`). Default forward plus an explicit frozen Task0 classifier is the appropriate diagnostic path.
- The existing external-delta path injects dense weights before the identity scaling. Leaf external tensors can be used directly for effective-weight gradients while freezing A/B, the backbone, and the classifier. Ensure hook flags are off if hooks expect gradients on a frozen BA graph.

## Fisher and numerical requirements

For each **individual** example use Task0 local-head cross-entropy, without EWC or shaping penalties. Compute `trace=mean_i sum_l ||g_i,l||^2` and `q=mean_i (sum_l <g_i,l,Delta_l>)^2`. The inner products across all blocks must be summed before squaring. Batch-mean gradients or diagonal-Fisher quadratic forms are different quantities. Keep signed loss increments.

For vectorized/microbatched computation verify against two or more separate single-example backwards. A microbatch dimension must survive until each sample's squared norm and full directional derivative are formed; reduction before squaring is invalid. If using `autograd.grad(..., is_grads_batched=True)` or vmap, check attention backend compatibility and memory before the full run. A serial fallback is valid; merely changing the microbatch size must not change the estimator.

Use FP32 forward/gradients initially and FP64 scalar accumulation, with no Fisher normalization, floor, damping, or AMP. Persist software versions, attention backend, TF32 setting, source hashes, coordinate order, seed and direction hashes. The direction bank must come from a dedicated generator and remain common across anchors, radii and splits.

## Still-open execution gates

1. Strict restore succeeds for both anchors and both t9 checkpoints. Historical test local loss and accuracy reproduce; new deterministic train is not expected to equal historical cached random augmentation results.
2. At identical fixed inputs, direct t9 Task0 logits/loss agree with Task0 plus the full `Delta`; s=0 agrees with direct Task0. Check the whole subset/endpoints, not only a single convenient sample. Check non-LoRA parameters and old classifier remain unchanged.
3. Repeated evaluation agrees; r=0 gives zero damage; direction norm and perturbation norm match the requested values; clearing the external tensors restores the anchor.
4. Single-example versus batched-gradient estimates agree. Finite-difference directional derivatives agree at a documented step size; use the actual unit direction, which is less vulnerable to small random derivatives in 14 million dimensions. Check q and trace are finite/nonnegative without clipping them into agreement.
5. Measure 256-sample throughput and peak memory on one anchor before reporting a GPU-hour budget.
6. For each t9 path, compare nested 5/9-point Simpson integrals. A relative discrepancy above 5% triggers 17 nodes. Verify `D_subset^2 <= integral(q)` using **the same 256 examples, local head, and deterministic inputs**; retain an unconverged status if quadrature still fails. This is not a guarantee involving full-train/test loss.

## Continuation experiment requirements

Use a fresh paired RNG after restoring either Task0 checkpoint, before new-head creation and DataLoader creation; preserve its state/seed. `SimpleLinear` uses Kaiming uniform initialization with `nonlinearity='linear'` and zero bias, so replacing it by PyTorch's default `nn.Linear` changes the training experiment. Separate anchor identity (`sgd` or `gam`) from future optimizer (`sgd` for both), clear any `optimizer_type_by_task` override, set `lambda_flat=0`, and preserve the historical lambda2000/batch128/lr.02/50-epoch/cosine/no averaging configuration. Restore each anchor's own Fisher; do not claim the protection matrices are identical. Since historical continuation RNG is not verified, two fresh Task1 continuations from the two existing anchors provide the clean paired table without retraining Task0.

Further code review of the new diagnostic and continuation scripts is pending their implementation.

## Independent executable environment check

The exact GPU4 command in `ENVIRONMENT.md` was executed after confirming GPUs 4 and 6 each had 1 MiB allocated and 0% utilization. The sandbox could not contact NVML; the authorized host GPU route (`require_escalated`) succeeded. Exit status 0, an 8×8 forward/backward finite result, torch 2.4.1+cu121, torchvision 0.19.1+cu121, timm 1.0.9, Quadro RTX 6000. Evidence: `DOC_CHECK.json` and `env_witness_gpu4.json`. This validates the executable environment only; the scientific measurement gates above remain open.

## Initial rq2_common.py code review

The current `EffectiveQKV` / `PerExampleCollector` implementation is mathematically appropriate for this eval-mode, sample-independent ViT. Backward from the sum of individual CE losses yields unscaled per-example output adjoints. `bmm(adj.transpose(1,2), x)` reconstructs the exact per-example effective Q/V matrix gradient. The collector adds each module's directional dot product before squaring, so the directional statistic retains cross-module terms. `RestoredNetwork` uses a standard linear head whose parameters are fully overwritten by strict checkpoint restoration; its initialization therefore does not alter diagnostic outputs.

Required dynamic corroboration: compare the collector against independent one-example `autograd.grad(loss, effective_params)` reference values, not merely the same collector at batch size one. Also compare logits from the original restored LoRA forward and the replacement EffectiveQKV forward. No blocking mathematical issue identified in this initial common-code review; this is not a dynamic gate pass.

## Static implementation review completed

Reviewed `rq2_common.py`, `diagnostic.py`, `train_continuation.py`, `pipeline.py`, and `diagnostic_config.json`; exact reviewed hashes and nonblocking suggestions are in `CODE_REVIEW.json`. No blocking static finding. The model/data/gradient GPU gates must still pass. Path `status=complete` records measurement completion; numerical acceptance separately requires inspection of `quadrature_converged` and `bound_passed_with_2pct_numeric_tolerance`. The paired A branches still require post-run equality checks on new-head, sample-order and augmentation receipts.

## Final frozen-code review and queue decision

`CODE_REVIEW_FINAL.json` records the final code hashes and decision **NO_BLOCKER_TO_LAUNCH_WAITING_QUEUE**. `DOC_CHECK_campaign.json` independently executes the documented CPU dry-run: 15 jobs, 8 stages, only authorized GPU4/6 plus the CPU report stage, matching fingerprints, seed0-only scope. The initial legacy precision issue is explained by the controlled TF32-convolution reference; the new protocol keeps all production values FP32, restores the reference weights/hooks, remeasures all task endpoints, and uses current a11. Paired RNG receipt verification and both path numerical flags gate final completion. Occupied GPUs must leave the jobs waiting; this review does not claim the final scientific sanity or experiments have already passed.
