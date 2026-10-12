# Future susceptibility probe

`future_probe.py` forks exactly one native Task-0 checkpoint into one continuous
Task-1 trajectory. No inherited task-end hook is called and checkpoint files
are checked unchanged. GPU launch is handled by the campaign queue.

Formal invocation:

```sh
python -B trainer/future_probe.py \
  --source FROZEN_SOURCE --config FORMATION_CONFIG.json \
  --checkpoint-dir FORMATION_OUTPUT/checkpoints/epoch_050 \
  --output-dir FRESH_FUTURE_OUTPUT --selection-split selection_split.json \
  --formation-arm s --formation-epoch 50 --seed 2 \
  --future-rule s --steps 100 --milestones 1,10,50,100
```

For future `g_match`, add `--reference-dir COMPLETED_S50_FUTURE_S_OUTPUT`.
That reference must be independently complete, hash-bound to its receipt, use
the same seed, data order, input augmentations, source, helper, split, head
initialization, and nominal LR. Only the SGD scalar changes. The target is the
actual norm of concatenated unscaled LoRA BA and full FC changes, measured
from actual factor values using FP64 Gram products. Actual FP32 step relative
matching error must be at most 1e-3; impossible/nonfinite targets fail without
fallback. Native GAM add/subtract roundtrip drift is measured separately.

Smoke adds `--smoke --steps 1 --milestones 1 --formation-epoch 1`; its g_match
reference is explicitly same-seed smoke S1→S. A smoke reference cannot satisfy
a formal run. Formation arm names accepted are `s`, `s_lr15`, `s_lr2`, `g_raw`,
`g_r025`, `g_match`, `sum`, and `mean_lr2`.

Future S means GAM g1 with rho=.05, r=1, gamma=0. It is not the original-point
CE gradient. Every step retains the inherited clean CE gradient plus four
native GAM gradient probes. The native lambda-flat projector's actual
`clean + (selected-clean)` arithmetic is retained and checked against its own
operation sequence. It need not be bitwise identical to `selected` in FP32.
There is no Fisher, EWC, weight merging, replay, or Task0 training data.

The native Task1 dataset determines epoch length. Current data has 997 samples,
batch size128, no drop-last, hence8 batches per epoch. One100-step trajectory
is12.5 epochs, with milestones1/10/50/100 on that same trajectory. Cosine LR
starts.02, floor.002, T_max50 complete native epochs. The scheduler advances
only at a completed epoch. This deliberately does not relabel100steps as10epochs.

Sampler and augmentation state is independent of model/formation RNG. Head
initialization uses a separate per-seed RNG context; old rows and the backbone
must remain exact. Pre-expansion old20-class and post-expansion K0 baselines
separate new-head competition from later training changes.

At K0 and every step, Task0 and Task1 diagnostic losses use their same fixed
subset and batch packing. Old local20 and all-seen40 path losses each yield
delta, I=sum(delta), Iplus=sum(max(0,delta)), and a telescoping assertion.
At requested K nodes, full and reporting metrics are additionally computed
from full official heldout logits. No gradients are computed in any eval;
RNG, modes, buffers, and gradients are checked/restored. A singleton class is
reporting-only, and macro accuracy averages only classes actually present.
The frozen split's sample IDs, labels, counts, coverage and metadata are checked.
Seed2 remains exploratory because its full heldout has already been inspected.

Outputs:

- `identity.json`: checkpoint/source/helper/split/data/head/protocol pins.
- `step_trace.jsonl`: 1-based `step_index`, 1-based epoch, 0-based batch, seed,
  task, `identity` input/target/index hashes, `lr_nominal`, `step_scale`,
  `actual_effective_delta`, native own-spec checks, diagnostic losses/path metrics.
- `evaluations.json`: `pre_expand.task0` and `milestones.K.task{0,1}` with full,
  diagnostic, reporting, each with local and all_seen CE/sample/macro accuracy.
- `summary.json` and identical `final_summary.json`: status, completed_steps,
  requested_steps, seed/formation origin/future rule, smoke, milestones_completed,
  counters, path_integrals, final_evaluation, checkpoint/source unchanged checks,
  trace/identity/evaluation hashes, and maximum actual matching error.

CPU validation, with no GPU:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B \
  trainer/test_future_probe_cpu.py --source FROZEN_SOURCE \
  --output trainer/FUTURE_PROBE_CPU_AUDIT.json
```

The suite covers isolated data RNG, new-head RNG and preservation, eval
restoration, split/singleton/canonical-ID positive and negative cases, path
telescoping, reference hash/data/LR/seed rejection, and the frozen native
learner's batch function with GAM S and matched-G toy steps. The latter asserts
four probes/five forwards, exact own-spec decomposition/projector/SGD, unchanged
old head rows, cosine LR, and rejects a nonfinite injected final gradient.
