# Native NLP two-point probe integration

The native `UIETrainer` now supports `optimizer_type=probe-adamw_hf` (alias `probe_adamw_hf`). All three modes use Transformers AdamW with `adam_epsilon=1e-6` and `weight_decay=0`. The mode names describe submitted gradients; the `sgd` mode is a clean-gradient AdamW control rather than a literal SGD parameter update.

| Mode | Probe | Submitted gradient | CE phases / replay closures / real commits |
|---|---|---|---|
| `reverse` | `theta - rho*g0/(norm(g0)+eps)` | `2*g0-g1_minus` | 2 / 1 / 1 |
| `sam` | `theta + rho*g0/(norm(g0)+eps)` | `g1_plus` | 2 / 1 / 1 |
| `sgd` | None | `g0` | 1 / 0 / 1 |

Arguments are `--probe_mode`, `--probe_rho` (default 0.05), `--probe_eps` (default 1e-12), and `--probe_quantization_diagnostics_interval` (default 0). The radius is independent of the learning rate. Core construction uses the un-clipped global-mean clean gradient. The Trainer clips the final selected direction once through the core adapter and restores BASE before one ordinary HF AdamW commit.

## Source entry points

- `src/run_uie_lora.py`: new arguments, serial shared-dataset-cache preparation, FP32 LoRA promotion, and final audit invocation. The frozen backbone and BF16 forward policy are unchanged.
- `src/uie_trainer_lora.py`: separate optimizer family, CE-only clean pass, effective-window cache, same per-microbatch RNG replay, explicit global mean reduction once per replay, short-window mean rescaling, and scheduler/zero-grad once per actual commit.
- `src/optimer_reverse_probe.py`: exact two-point formulas and transaction/commit logic, implemented separately from `CurrentG2`.

Pure mode requires `train_method=lora`, `lora_strategy=inclora`, and an actual `decoder_token_ids_v2` collator. EWC/OLoRA strategies, RWP, incompatible optimizer epsilon/weight decay, past-state caches, DeepSpeed, FP16/Apex scaling, and unsupported modes fail explicitly. No G2/G3, projection, Fisher, history, or auxiliary regularizer is invoked.

## Output contract

`probe_step_trace.jsonl` records one `TwoPointProbeTrainerStep_v1` row per actual Trainer update. It includes the core `probe_*` receipts, mode/radius, actual microbatch count, configured gradient accumulation, short-window rescaling, replay reduction ownership, clipping policy, actual HF backend, and probe loss. Step one also records `initial_trainable_sha256`, `clean_gradient_sha256`, and `first_window_batch_sha256` for matched-control verification. Tensor fingerprints preserve RNG and are computed only on the first realized window.

`two_point_probe_training_audit.json` uses `TwoPointProbeTrainerAudit_v1`. It verifies step ordering/budget, exact BASE restoration, mode-specific phase/direction counts, one HF AdamW commit, trainer-owned globally averaged gradients, absence of Fisher/projection/history/G2/G3, FP32 trainables, and fixed backend epsilon/weight decay. It records total phase/replay/commit counts, world size, pipeline version, optimizer groups, tail-window count, trace SHA256, and first-window fingerprints.

BF16 cast diagnostics remain an observational parameter proxy (`proves_forward_visibility=false`). They do not change the forward precision, submitted direction, or fallback policy. CPU tests establish implementation behavior, not the empirical efficacy of a trained 1B/3B model.

## Validation

The test module `tests/test_reverse_probe_trainer.py` covers literal closed-form quadratic directions and one HF AdamW commit for all three modes; full and short effective windows; unchanged post-clean RNG and replayed dropout; corrected collator and native tiny Llama BF16 training; FP32-only trainables and frozen backbone; first-window matching; `max_steps` and epoch-tail accounting; gradient checkpointing; and a two-rank CPU Gloo global-mean test. No GPU training was started by this integration task.

The Gloo test needs local loopback sockets. The sandbox's network restriction initially prevented process-group initialization; rerunning only that test with loopback permission passed. Logs are `TRAINER_TESTS.log` and `TRAINER_GLOO_VERIFIED.log` in this directory.

All nine distinct new tests passed (eight offline tests and one isolated Gloo test). Four existing CurrentG2 regression tests also passed, covering family dispatch, BF16 native short-window training, both AdamW backends with clipping, and dropout RNG replay. The latter receipts are in `G2_TRAINER_REGRESSION.log`.
