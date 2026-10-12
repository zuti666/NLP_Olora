# Pure Current-G2 Llama Order1 implementation audit

Scope is **pure Current-G2 only**, with HF AdamW and Torch AdamW for Llama-3.2-1B/3B. The newly proposed EWC/OLoRA presets and G2 regularization hooks were removed at the user's request. No new full-model LLM training, GPU queue or FAA result was created.

The complete algorithm, implementation distinctions, configurations, flowchart, pseudocode and reproduction commands are in [G2_LLAMA_ORDER1.md](/data/140-1/users/liying/O-LoRA/scripts_llama/G2_LLAMA_ORDER1.md).

## Acceptance evidence

| Check | Result | Evidence |
|---|---|---|
| Current-G2 core | 20/20 CPU tests passed | CORE_TESTS.log |
| Trainer | 10/10 CPU tests passed | TRAINER_TESTS.log; TRAINER_CPU_RECEIPT.json |
| Launcher | 12/12 CPU tests passed | LAUNCHER_TESTS.log |
| Existing FS/GAMFS/SAM-EWC | 28/28 regression tests passed | LEGACY_REGRESSION_TESTS.log |
| Real CLI dataclass parser | 72 expanded argument sets passed | ACTUAL_CLI_PARSE.json |
| Historical input identity | 29 files match frozen inputs byte-for-byte | DATA_PROTOCOL_IDENTITY.json |
| Thin shell entry points | Four syntax checks and dry runs passed | Four scripts under scripts_order1_llama1B/3B |
| Independent document execution | Exact documented dry-run, no output directory or GPU | DOC_CHECK.json |
| Source/configuration/document pin | 16 files | SOURCE_SHA256.json |

The core test includes twelve consecutive updates matching the original CV Current-G2 implementation bit-for-bit under SGD and changing learning rates/dropout. The two AdamW backends each passed native tiny random Llama BF16 CPU training with actual FP32 rank8 LoRA, coupled radii, LR0.001 and final clipping1. Each completed two optimizer commits and three CE phases per commit, with zero fallback. A separate two-rank CPU Gloo test checks global mean gradients and identical final parameters. These results are implementation witnesses, not full Llama-1B/3B accuracy measurements.

The final CPU source receipt follows the minimal G2 dataset-cache fix: G2 joins the existing rank-first `main_process_first`/reuse-cache path, preventing concurrent datasets1.17 first-build renames. No packages or environment spec changed.

## Frozen baseline audit

BACKEND_AND_BASELINE_AUDIT.json recomputes all twelve historical model×method Order1 means/sample standard deviations from the original unrounded three-seed CSV. It also pins the frozen trainer, optimizers, stage specs and final report.

Historical FS-EWCLoRA uses four-probe GAM and **per-tensor projected EWC**; it is not a simple additive sum and not the recent GAMFisherShape-SAM class. Historical `gam-adamw_hf` uses Torch AdamW (epsilon1e-8, weight decay0.01), whereas plain `adamw_hf` uses Transformers AdamW (epsilon1e-6, weight decay0). Pure G2 provides these backend settings as separate presets.

New G2 also uses per-microbatch RNG replay and global phase-wise mean gradients. Legacy GAM constructs rank-local probes under `no_sync` and averages the final direction, and legacy SAM/GAM do not replay each microbatch's clean RNG. Therefore historical performance cannot by itself isolate the final gradient formula as the sole changing component.

## Planned execution and limits

Order1 is dbpedia→amazon→yahoo→agnews. Both models use rank8/alpha32/dropout0.1, BF16 forward/FP32 trainable adapters, effective batch64 (DDP4×batch1×GA16), LR0.001 then0.0001, constant scheduler and nominal one epoch/task. Formal stages are capped to historical full-window budgets218/78/156/62 (514 total).

Two backends × two models × seed0 are four Orders/sixteen formal stages. Three seeds0/42/1993 are twelve Orders/forty-eight formal stages, plus bounded smoke stages. This is a plan, not an active queue. Actual execution requires four explicitly selected idle GPUs and a successful two-task smoke chain. No full-model GPU acceptance, runtime or new G2 FAA is asserted.
