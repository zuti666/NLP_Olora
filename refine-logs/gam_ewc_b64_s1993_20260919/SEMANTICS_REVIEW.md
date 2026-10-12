# Llama-1B GAM-EWC batch64: implementation facts

Reviewed 2026-09-19. This is bounded source/log factfinding, not an independent experiment-integrity verdict. Source line references below describe the source hashes captured in `SEMANTICS_REVIEW.json`, before any new audit instrumentation.

## Configuration to preserve

Use **4 ranks × per-rank batch 1 × GA16 = effective batch64**, retaining **seed=1993, data_seed=1993, BF16, ewclora_lambda=1.0, ewclora_gamma=1.0**. Do not reuse the pure-EWC batch64 setting `lambda=0.5`.

The two successful historical logs contain 12 commands with identical method settings:

- `Logs_EWCLoRA_GAM/order1_llama1B_ewclora_gam_seed1993_20260516_003719_tmux.log`: first command line103, total batch40 line3025, GA8 line3026.
- `Logs_EWCLoRA_GAM/order2_order3_llama1B_ewclora_gam_seed1993_20260516_115952_tmux.log`: Order2 commands lines104,3829,6887,9896; Order3 lines13324,16844,19902,22911.

Preserve `lora_strategy=ewclora_gam`, `optimizer_type=gam-adamw_hf`, LoRA rank8/alpha32/dropout0.1, all three task orders, one epoch, first-task LR1e-3 then1e-4, constant schedule, warmup0, source length512, output length50, eval batch4, max grad norm1. GAM settings: rho0.02, norm-rho0.2, adaptiveFalse, beta1=beta2=beta3=1, gamma0.03, mean rank reduction, perturb/project eps1e-12.

Explicitly set `olora_lamda_1=0`, `olora_lamda_2=0`, Fisher normalization flagsFalse, Fisher damping0 and full-data Fisher estimation. The historical default OLoRA coefficient0.5 was printed in the logs but was unused by the historical GAM-EWC branch. Current code applies an OLoRA base in that branch, so leaving the default would change the method.

## Why lambda remains 1.0

Historical candidate commit `898adf6bb13eb6efc00e31764eb2bcabd014498d` and pre-instrumentation current source have identical ASTs for `_compute_micro_batch_loss`, `_run_gam_replay_pass`, `_compute_ewclora_grad_cache`, and `_build_ewclora_gam_projector`. `src/optimer_gam.py` is byte-identical to that commit.

Current trainer lines3664-3667 divide each CE microbatch loss by GA. Lines3743-3758 replay all cached microbatches. Lines3097-3102 compute a single clean-point EWC gradient cache immediately before each GAM optimizer step. Lines1446-1462 apply the tensorwise direction

`g_final = g_GAM - <g_GAM,g_EWC>/(||g_EWC||² + eps) * g_EWC + g_EWC`.

The cache comes from `0.5 * lambda * sum(F * DeltaW_new²)` (lines921-944). EWC is neither repeated GA times nor divided by GA. Consequently GA8→GA16 keeps the same nominal EWC strength with lambda1.0. The pure-EWC trainer instead adds its penalty to every already-scaled microbatch loss; its lambda0.5 compensation does not apply here. Halving lambda in GAM would change the added EWC direction and, through the nonzero epsilon, can also alter the projection.

Current trainer lines3624-3633 add OLoRA regularization before deferring GAM-EWC projection. Historical lines2296-2298 simply defer EWC and leave CE unchanged. Setting both OLoRA coefficients to zero recovers the relevant historical loss semantics without editing the algorithm.

## Actual optimizer and distributed operation

The name `gam-adamw_hf` is misleading: historical lines2185-2190 and current lines3445-3450 instantiate **`torch.optim.AdamW`**, not `transformers.AdamW`. It receives only model parameters and learning rate, so actual defaults are betas(0.9,0.999), eps1e-8, **weight_decay0.01**, amsgradFalse. The printed TrainingArguments `weight_decay=0.0` does not control this constructor. Preserve the constructor and audit the wrapped optimizer itself.

The trainer binds `GAM.model` to the actual DDP wrapper at lines1983-1984. `GAM.step` runs all four closure probes inside one DDP `no_sync` context, with no immediate backward in the initial data-collection pass (trainer lines2334-2335 and2497-2498). It computes four local directions, applies EWC projection, restores the local perturbation, then averages the final gradient across ranks, clips it and calls the base optimizer.

Thus four ranks with16 local microbatches give64 examples per optimizer update. GAM's nonlinear probes operate on16 local examples before four-rank averaging. This is not mathematically identical to eight ranks probing8 examples each, and changing synchronization inside each probe would change the historical algorithm. Keep the final-direction-only synchronization.

The retained legacy loop only updates complete accumulation windows. New full-stage update counts are dbpedia218, amazon78, yahoo156, agnews62, versus old350/125/250/100. Full post-task Fisher still processes14000/5000/10000/4000 examples, including the incomplete optimizer tail.

## Runtime audit requirements

Extend the opt-in protocol gate; existing gates only accept plain EWC/AdamW with either seed42 batch64 or seed1993 batch40. A separate GAM protocol should retain early CUDA rank binding and serialized shared dataset cache initialization in `run_uie_lora.py` lines667-680 and782-793.

Check actual DDP4/device binding, BF16 forward autocast, seed1993/data_seed1993, BS1/GA16, exactly64 fresh trainable A/B tensors,32 target modules, exact GAM settings, TorchAdamW wrapper/defaults, sampler disjointness/completeness, nonzero parameter updates, complete Fisher, frozen historical adapters, output writer/barriers, and fresh stage2 Fisher loading.

Additional GAM witnesses should count four closures and16 replay microbatches per closure for every full optimizer update, require each probe's DDP backward synchronization disabled, and count the final gradient sync/base optimizer step. Record positive perturbation norms, projector calls and direction changes; stage1 has no historical Fisher, while stage2 should eventually exhibit nonzero EWC cache and projection effect. Record actual synchronized-gradient agreement after `_sync_grad`, before clipping/base update.

**Do not require bit-identical parameter hashes across ranks for historical GAM.** Local floating-point `p.add_(e); p.sub_(e)` can leave different rounding residuals on each rank even if the final gradient is perfectly synchronized. `GAM_FLOAT32_UNPERTURB_WITNESS.json` gives a dependency-free IEEE754-binary32 CPU witness: equal initial values lead to four distinct restored hashes, maximum rank difference7.450580596923828e-9. It uses rounding after each add/subtract and does not train a model.

Retain hash observations, require identical initial state, record actual maximum absolute/relative differences after training, and check parameter agreement with disclosed `atol=1e-6, rtol=1e-5`. This is an observational tolerance, not proof of exact equality. Keep a meaningful synchronized-gradient check and require finite values. Do not introduce a parameter broadcast merely to force identical hashes; that would alter historical GAM behavior.

## Provenance limits

Historical log source-line references for total batch947, GA948, Fisher start351/update369/save388 all match the candidate historical source exactly. This strongly supports its relevance, but no executed source SHA or historical model-weight hash was retained, so exact historical source/checkpoint identity remains unproven.

The new cohort intentionally retains seed1993. It therefore remains distinct from seed42 Llama-1B baselines and the newly completed pure-EWC seed42 run. Save independent commands, frozen source/config/input hashes, smoke results, raw final predictions and protocol-bearing result filenames.
